import time
from typing import Any, Dict, List

import numpy as np
import torch
import transformers

from src.model_wrapper.base_model import BaseModelWrapper
from src.model_wrapper.utils.travel_util import (
    load_traj_model,
    transform_to_world,
)


def build_trajcorr_target(
    current_pose,
    current_rotation,
    coarse_goal_world,
    coarse_local,
    observation_pose=None,
    epsilon: float = 1e-6,
):
    """Build a training-scale target along the stale world's current bearing."""
    current = np.asarray(current_pose, dtype=np.float64).reshape(3)
    rotation = np.asarray(current_rotation, dtype=np.float64).reshape(3, 3)
    old_goal = np.asarray(coarse_goal_world, dtype=np.float64).reshape(3)
    old_coarse = np.asarray(coarse_local, dtype=np.float64).reshape(3)
    original_norm_m = float(np.linalg.norm(old_coarse))
    if original_norm_m <= epsilon:
        raise ValueError("cannot build TrajCorr target from a zero-length coarse vector")

    direction_world = old_goal - current
    direction_norm = float(np.linalg.norm(direction_world))
    if observation_pose is not None:
        original_world_direction = old_goal - np.asarray(
            observation_pose,
            dtype=np.float64,
        ).reshape(3)
        original_world_norm = float(np.linalg.norm(original_world_direction))
        if original_world_norm > epsilon:
            original_world_unit = original_world_direction / original_world_norm
            remaining_forward_m = float(np.dot(direction_world, original_world_unit))
            if remaining_forward_m <= epsilon:
                direction_world = original_world_unit
                direction_norm = 1.0
    if direction_norm <= epsilon:
        raise ValueError("cannot build TrajCorr target from a zero-length world direction")

    direction_unit = direction_world / direction_norm
    virtual_goal_world = current + direction_unit * original_norm_m
    local_target = rotation.T @ (virtual_goal_world - current)
    return {
        "virtual_goal_world": virtual_goal_world,
        "local_target": local_target,
        "original_coarse_norm_m": original_norm_m,
        "trajcorr_coarse_norm_m": float(np.linalg.norm(local_target)),
    }


class EdgeDNNModelWrapper(BaseModelWrapper):
    def __init__(self, model_args, data_args):
        if hasattr(transformers, "CLIPImageProcessor"):
            self.image_processor = transformers.CLIPImageProcessor.from_pretrained(model_args.image_processor)
        else:
            self.image_processor = transformers.AutoImageProcessor.from_pretrained(model_args.image_processor)
        self.traj_model = load_traj_model(model_args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.traj_model.to(dtype=torch.bfloat16, device=self.device)
        self.traj_model.eval()
        self.model_args = model_args
        self.data_args = data_args

    def eval(self):
        self.traj_model.eval()

    def _sync_cuda(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def reproject_goal_to_local(coarse_goal_world, episode):
        latest = episode[-1]
        pos_now = np.asarray(latest["sensors"]["state"]["position"][0:3], dtype=np.float64)
        rot_now = np.asarray(latest["sensors"]["imu"]["rotation"], dtype=np.float64)
        return rot_now.T @ (np.asarray(coarse_goal_world, dtype=np.float64) - pos_now)

    def run_traj_from_world_goal(self, episodes: List[List[Dict[str, Any]]], coarse_goal_world):
        local_targets = []
        for i, episode in enumerate(episodes):
            local_targets.append(self.reproject_goal_to_local(coarse_goal_world[i], episode))
        return self.run_traj_from_local_targets(episodes, np.asarray(local_targets, dtype=np.float64))

    def run_trajcorr_from_world_goal(
        self,
        episodes: List[List[Dict[str, Any]]],
        coarse_goals_world,
        coarse_locals,
        observation_poses,
    ):
        if not (
            len(episodes)
            == len(coarse_goals_world)
            == len(coarse_locals)
            == len(observation_poses)
        ):
            raise ValueError("TrajCorr batch inputs must have equal lengths")

        targets = []
        local_targets = []
        for episode, coarse_goal_world, coarse_local, observation_pose in zip(
            episodes, coarse_goals_world, coarse_locals, observation_poses
        ):
            latest = episode[-1]
            target = build_trajcorr_target(
                latest["sensors"]["state"]["position"][0:3],
                latest["sensors"]["imu"]["rotation"],
                coarse_goal_world,
                coarse_local,
                observation_pose,
            )
            targets.append(target)
            local_targets.append(target["local_target"])

        local_waypoints, profile = self._infer_local_waypoints(
            episodes, np.asarray(local_targets, dtype=np.float64)
        )
        refined_waypoints = transform_to_world(local_waypoints, episodes)
        p5_errors = []
        for index, target in enumerate(targets):
            if len(refined_waypoints[index]) >= 5:
                p5_errors.append(
                    float(np.linalg.norm(refined_waypoints[index][4] - target["virtual_goal_world"]))
                )
            else:
                p5_errors.append(None)
        profile.update(
            {
                "virtual_goal_world": [target["virtual_goal_world"].tolist() for target in targets],
                "original_coarse_norm_m": [target["original_coarse_norm_m"] for target in targets],
                "trajcorr_coarse_norm_m": [target["trajcorr_coarse_norm_m"] for target in targets],
                "trajectory_scale": [1.0 for _ in targets],
                "p5_to_virtual_goal_m": p5_errors,
            }
        )
        return refined_waypoints, profile

    def run_traj_from_local_targets(self, episodes, local_targets):
        local_waypoints, profile = self._infer_local_waypoints(episodes, local_targets)
        refined_waypoints = transform_to_world(local_waypoints, episodes)
        return refined_waypoints, profile

    def _infer_local_waypoints(self, episodes, local_targets):
        image_list = []
        for episode in episodes:
            image_list.append(episode[-1]["rgb"][0])
        images = np.stack(image_list, axis=0)
        model_dtype = next(self.traj_model.parameters()).dtype
        image = self.image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
        image = image.to(device=self.device, dtype=model_dtype)
        target = torch.tensor(np.asarray(local_targets, dtype=np.float32), device=self.device, dtype=model_dtype)
        with torch.inference_mode():
            self._sync_cuda()
            start = time.perf_counter()
            inputs = {"img": image, "target": target}
            waypoints_traj = self.traj_model(inputs, None)
            self._sync_cuda()
            traj_latency_ms = (time.perf_counter() - start) * 1000.0
        local_waypoints = waypoints_traj.detach().cpu().to(dtype=torch.float32).numpy()
        return local_waypoints, {
            "traj_latency_ms": float(traj_latency_ms),
            "reprojected_coarse": np.asarray(local_targets).tolist(),
        }
