"""Local AeroDPO Q8 fallback used by the NCN ablations.

The runtime deliberately owns no edge-planning, network, or termination policy.
It only turns an on-demand pair of AirSim RGB frames into one native AeroDPO
action.  This keeps the existing edge-assisted benchmark accounting intact.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.model_wrapper.aerodpo_llamacpp_local_wrapper import (
    AeroDPOLlamaCppLocalEvalModelWrapper,
)


@dataclass
class EdgeLatencyEstimate:
    """Per-episode estimate of edge model + trajectory compute time."""

    compute_ms: float
    alpha: float = 0.5

    def update(self, llm_latency_ms: float, traj_latency_ms: float) -> None:
        observed = max(0.0, float(llm_latency_ms) + float(traj_latency_ms))
        self.compute_ms = self.alpha * observed + (1.0 - self.alpha) * self.compute_ms

    def remaining_ms(self, submitted_logical_ms: Optional[float], now_logical_ms: float, uplink_latency_ms: float) -> float:
        if submitted_logical_ms is None:
            return float("inf")
        ready_ms = float(submitted_logical_ms) + max(0.0, float(uplink_latency_ms)) + self.compute_ms
        return max(0.0, ready_ms - float(now_logical_ms))


class NCNRuntime:
    """A process-local Q8 AeroDPO model loaded only when NCN is enabled."""

    def __init__(self, model_args: Any):
        required = {
            "ncn_text_model_path": getattr(model_args, "ncn_text_model_path", None),
            "ncn_lora_model_path": getattr(model_args, "ncn_lora_model_path", None),
            "ncn_mmproj_path": getattr(model_args, "ncn_mmproj_path", None),
            "ncn_local_bridge_path": getattr(model_args, "ncn_local_bridge_path", None),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError("NCN requires: " + ", ".join(missing))
        bridge_args = SimpleNamespace(
            llama_text_model_path=required["ncn_text_model_path"],
            model_path=required["ncn_lora_model_path"],
            llama_mmproj_path=required["ncn_mmproj_path"],
            llama_local_bridge_path=required["ncn_local_bridge_path"],
            llama_local_context=int(getattr(model_args, "ncn_local_context", 2048)),
        )
        self.model = AeroDPOLlamaCppLocalEvalModelWrapper(bridge_args)

    def act(
        self,
        front_bgr: np.ndarray,
        down_bgr: np.ndarray,
        state: Dict[str, Any],
        target_position: Any,
        instruction: str,
    ) -> Tuple[Dict[str, float], bool, Dict[str, Any], str]:
        # This intentionally follows the already-validated AeroDPO prompt and
        # decoder path.  No LLaMA-UAV assist, trajectory model, or DINO prompt
        # enters this call.
        episode = [{"rgb": [front_bgr, down_bgr], "sensors": {"state": state}}]
        inputs, prompts = self.model.prepare_inputs(
            [episode], [target_position], [instruction]
        )
        actions, stops, profile = self.model.run_profiled(inputs)
        return actions[0], bool(stops[0]), profile, prompts[0]

    def close(self) -> None:
        bridge = getattr(self.model, "bridge", None)
        if bridge is not None:
            bridge.close()
