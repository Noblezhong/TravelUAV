"""Shared TCM (Trajectory Correction Module) runtime for the MATCH paradigms.

NEW FILE — imported only by ``match_eval.py`` (PPO+TCM) and
``continue_tcm_eval.py`` (Continuous+TCM).  No existing file is modified.

Implements the correction branch of paper Algorithm 1:

    guidance arrival -> measure Delta_m -> if Delta_m >= delta_cor:
        refresh observation -> derive coarse goal from llm_output ->
        regenerate state-aligned waypoints (build_trajcorr_target + traj DNN)
        -> begin target lock -> filter waypoints -> execute locked buffer
    unlock on goal_reached / goal_passed / buffer_exhausted -> fresh request

Fidelity contract: when ``enabled=False`` every entry point is a no-op that
returns the original result, so TC-OFF runs behave exactly like the audited
paradigms (PPO 47.9 / Continuous 25.7).
"""

import copy
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.model_wrapper.edge_traj_dnn import build_trajcorr_target
from src.model_wrapper.utils.travel_util import transform_to_world
from src.vlnce_src.trajcorr_runtime import (
    TRAJECTORY_CORRECTED,
    TRAJECTORY_ORIGINAL,
    TargetLockLifecycle,
    select_trajectory_mode,
)


class TrajcorrMixin:
    """Adds corrected-trajectory regeneration to ProfileTravelModelWrapper.

    Mirrors EdgeDNNModelWrapper._infer_local_waypoints input convention
    ({"img": ..., "target": ...} at training scale, bfloat16, model device).
    """

    def regenerate_trajectory(
        self,
        episodes,
        coarse_goal_world,
        coarse_local,
        observation_pose,
    ):
        latest = episodes[0][-1]
        target = build_trajcorr_target(
            latest["sensors"]["state"]["position"][0:3],
            latest["sensors"]["imu"]["rotation"],
            coarse_goal_world,
            coarse_local,
            observation_pose,
        )
        images = np.stack([episodes[0][-1]["rgb"][0]], axis=0)
        model_dtype = next(self.traj_model.parameters()).dtype
        image = self.image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
        image = image.to(device=self.model.device, dtype=model_dtype)
        target_tensor = torch.tensor(
            np.asarray([target["local_target"]], dtype=np.float32),
            device=self.model.device,
            dtype=model_dtype,
        )
        with torch.inference_mode():
            self._sync_cuda()
            start = time.perf_counter()
            waypoints_traj = self.traj_model({"img": image, "target": target_tensor}, None)
            self._sync_cuda()
            regen_ms = float((time.perf_counter() - start) * 1000.0)
        local_waypoints = waypoints_traj.detach().cpu().to(dtype=torch.float32).numpy()
        world_waypoints = transform_to_world(local_waypoints, episodes)
        profile = {
            "trajcorr_latency_ms": regen_ms,
            "virtual_goal_world": np.asarray(target["virtual_goal_world"]).tolist(),
            "original_coarse_norm_m": float(target["original_coarse_norm_m"]),
            "trajcorr_coarse_norm_m": float(target["trajcorr_coarse_norm_m"]),
            "p5_to_virtual_goal_m": (
                float(np.linalg.norm(world_waypoints[0][4] - target["virtual_goal_world"]))
                if len(world_waypoints[0]) >= 5
                else None
            ),
        }
        return world_waypoints, profile


class TcmEpisodeStats:
    """Per-episode TCM counters, merged into the episode_end record."""

    def __init__(self):
        self.corrected_applies = 0
        self.stale_dropped = 0
        self.lock_completions: Counter = Counter()
        self.correction_fallbacks: Counter = Counter()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tcm_corrected_applies": int(self.corrected_applies),
            "tcm_stale_dropped": int(self.stale_dropped),
            "tcm_lock_completions": dict(self.lock_completions),
            "tcm_correction_fallbacks": dict(self.correction_fallbacks),
        }


def derive_trajcorr_inputs(result) -> Tuple[Optional[Tuple[List[float], List[float]]], Optional[str]]:
    """Return the planner's request-frame coarse vector and world goal.

    The VLM output is target-aligned local data, not a world-coordinate goal.
    The planner restores these fields with the original TC-ON rotation chain.
    """
    coarse_goal_world = getattr(result, "coarse_goal_world", None)
    coarse_local = getattr(result, "coarse_local", None)
    if coarse_goal_world is None or coarse_local is None:
        return None, "missing_coarse_frame"
    coarse_goal_world = np.asarray(coarse_goal_world, dtype=np.float64).reshape(3)
    coarse_local = np.asarray(coarse_local, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(coarse_goal_world)) or not np.all(np.isfinite(coarse_local)):
        return None, "nonfinite_coarse_frame"
    if float(np.linalg.norm(coarse_local)) <= 1e-6:
        return None, "zero_length_coarse"
    return (
        coarse_goal_world.tolist(),
        coarse_local.tolist(),
    ), None


class TcmRuntime:
    """Per-episode TCM state shared by both MATCH-paradigm evaluators."""

    def __init__(self, enabled: bool, delta_cor_m: float):
        self.enabled = bool(enabled)
        self.delta_cor_m = float(delta_cor_m)
        self.target_lock = TargetLockLifecycle(goal_radius_m=0.5)
        self.stats = TcmEpisodeStats()

    @property
    def lock_active(self) -> bool:
        return self.target_lock.active

    def refresh_observation(self, state, eval_env, clock) -> float:
        """Append one fresh observation frame in the main thread (timestamp-deduped)."""
        obs_start = time.perf_counter()
        outputs = eval_env.get_obs()
        obs_ms = float((time.perf_counter() - obs_start) * 1000.0)
        clock.advance_blocking(obs_ms)
        state.process_env_output(outputs)
        return obs_ms

    def apply_result(self, state, eval_env, model_wrapper, clock, result, record_request_ne: bool = False):
        """TCM-aware application of a returned edge result.

        Returns ``(effective_result, tcm_extra)``.  ``effective_result`` is
        ``None`` when a stale result arrives during target lock and must be
        dropped (never applied, never re-locked).
        """
        tcm_extra: Dict[str, Any] = {}
        if self.lock_active:
            self.stats.stale_dropped += 1
            tcm_extra["target_lock_discarded_stale_result"] = int(result.request_id)
            tcm_extra["target_lock_active"] = True
            return None, tcm_extra
        if not self.enabled:
            return result, tcm_extra

        current_pose = state.current_sim_pose()
        decision = select_trajectory_mode(
            True,
            result.observation_pose,
            current_pose,
            self.delta_cor_m,
        )
        tcm_extra["trajectory_mode"] = decision.mode
        tcm_extra["state_shift_at_apply_m"] = float(decision.state_shift_m)
        if decision.mode == TRAJECTORY_ORIGINAL:
            return result, tcm_extra

        coarse, fallback_reason = derive_trajcorr_inputs(result)
        if coarse is None:
            self.stats.correction_fallbacks[fallback_reason] += 1
            tcm_extra["trajcorr_fallback_reason"] = fallback_reason
            return result, tcm_extra
        coarse_goal_world, coarse_local = coarse

        obs_ms = self.refresh_observation(state, eval_env, clock)
        tcm_extra["trajcorr_refresh_obs_ms"] = float(obs_ms)
        if state.dones[0]:
            # collision / DINO termination surfaced in the fresh frame — do not
            # regenerate a trajectory for a finished episode
            self.stats.correction_fallbacks["dones_after_refresh"] += 1
            tcm_extra["trajcorr_fallback_reason"] = "dones_after_refresh"
            return result, tcm_extra

        regen_start = time.perf_counter()
        try:
            world_waypoints, profile = model_wrapper.regenerate_trajectory(
                [state.episode],
                coarse_goal_world,
                coarse_local,
                result.observation_pose,
            )
        except ValueError:
            self.stats.correction_fallbacks["regeneration_value_error"] += 1
            tcm_extra["trajcorr_fallback_reason"] = "regeneration_value_error"
            return result, tcm_extra
        regen_ms = float((time.perf_counter() - regen_start) * 1000.0)
        clock.advance_blocking(regen_ms)

        waypoints = np.asarray(world_waypoints[0]).tolist()
        completion = self.target_lock.begin(
            result.observation_pose,
            current_pose,
            coarse_goal_world,
        )
        filtered: List[List[float]] = []
        if completion is None:
            filtered = self.target_lock.filter_waypoints(waypoints, current_pose)
            if not filtered:
                completion = self.target_lock.mark_buffer_exhausted()
        self.stats.corrected_applies += 1
        if completion is not None:
            self.stats.lock_completions[completion] += 1

        if record_request_ne and not state.dones[0]:
            state.record_request_ne()

        effective = copy.copy(result)
        effective.refined_waypoints = [list(p) for p in filtered]
        effective.traj_latency_ms = float(result.traj_latency_ms) + regen_ms
        if effective.ready_logical_ms is not None:
            effective.ready_logical_ms = float(effective.ready_logical_ms) + regen_ms
        tcm_extra.update(
            {
                "coarse_local": copy.deepcopy(coarse_local),
                "coarse_goal_world": copy.deepcopy(coarse_goal_world),
                "trajcorr_regen_ms": float(regen_ms),
                "target_lock_active": bool(self.target_lock.active),
                "target_lock_completion_reason": completion,
                "target_lock_goal_world": copy.deepcopy(self.target_lock.goal_world),
                "target_lock_start_distance_m": self.target_lock.start_distance_m,
                "target_lock_raw_waypoint_count": int(len(waypoints)),
                "target_lock_filtered_waypoint_count": int(len(filtered)),
                "p5_to_virtual_goal_m": profile.get("p5_to_virtual_goal_m"),
            }
        )
        return effective, tcm_extra
