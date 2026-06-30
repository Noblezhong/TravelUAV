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

    def run_traj_from_local_targets(self, episodes, local_targets):
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
        refined_waypoints = waypoints_traj.detach().cpu().to(dtype=torch.float32).numpy()
        refined_waypoints = transform_to_world(refined_waypoints, episodes)
        return refined_waypoints, {
            "traj_latency_ms": float(traj_latency_ms),
            "reprojected_coarse": np.asarray(local_targets).tolist(),
        }
