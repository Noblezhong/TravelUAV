import time
from typing import Any, Dict, Tuple

import numpy as np
import torch

from src.model_wrapper.travel_llm import TravelModelWrapper


class ProfileTravelModelWrapper(TravelModelWrapper):
    def _sync_cuda(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def run_profiled(
        self,
        inputs: Dict[str, Any],
        episodes,
        rot_to_targets,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        with torch.inference_mode():
            self._sync_cuda()
            llm_start = time.perf_counter()
            waypoints_llm_new = self.run_llm_model(inputs)
            self._sync_cuda()
            llm_latency_ms = (time.perf_counter() - llm_start) * 1000.0

            self._sync_cuda()
            traj_start = time.perf_counter()
            refined_waypoints = self.run_traj_model(episodes, waypoints_llm_new, rot_to_targets)
            self._sync_cuda()
            traj_latency_ms = (time.perf_counter() - traj_start) * 1000.0

        profile_info = {
            "llm_latency_ms": llm_latency_ms,
            "traj_latency_ms": traj_latency_ms,
            "llm_output": np.asarray(waypoints_llm_new).tolist(),
        }
        return refined_waypoints, profile_info

    def run(self, inputs, episodes, rot_to_targets):
        refined_waypoints, _ = self.run_profiled(inputs, episodes, rot_to_targets)
        return refined_waypoints
