import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def speed_label(speedup: float) -> str:
    value = float(speedup)
    return str(int(value)) if value.is_integer() else str(value).replace(".", "p")


class FastEvalClock:
    """Original-speed experiment clock used while AirSim runs faster."""

    def __init__(self, enabled: bool = False, speedup: float = 5.0):
        self.enabled = bool(enabled)
        self.speedup = float(speedup)
        if self.speedup <= 0:
            raise ValueError("fast_eval_speedup must be positive")
        self._logical_ms = 0.0
        self._wall_start = time.perf_counter()

    def reset(self) -> None:
        self._logical_ms = 0.0
        self._wall_start = time.perf_counter()

    @property
    def now_ms(self) -> float:
        if self.enabled:
            return float(self._logical_ms)
        return float((time.perf_counter() - self._wall_start) * 1000.0)

    @property
    def wall_elapsed_ms(self) -> float:
        return float((time.perf_counter() - self._wall_start) * 1000.0)

    def advance_blocking(self, elapsed_ms: float) -> float:
        if self.enabled:
            self._logical_ms += max(0.0, float(elapsed_ms))
        return self.now_ms

    def advance_action(self, sim_elapsed_ms: float, wall_elapsed_ms: float) -> float:
        elapsed = float(sim_elapsed_ms) if self.enabled else float(wall_elapsed_ms)
        return self.advance_blocking(elapsed)

    def advance_to(self, target_ms: Optional[float]) -> float:
        if self.enabled and target_ms is not None:
            self._logical_ms = max(self._logical_ms, float(target_ms))
        return self.now_ms

    def age_ms(self, submitted_logical_ms: Optional[float], submitted_perf_time: float) -> float:
        if self.enabled and submitted_logical_ms is not None:
            return max(0.0, self.now_ms - float(submitted_logical_ms))
        return max(0.0, (time.perf_counter() - float(submitted_perf_time)) * 1000.0)

    def metadata(self) -> Dict[str, Any]:
        return {
            "fast_eval": bool(self.enabled),
            "fast_eval_speedup": float(self.speedup if self.enabled else 1.0),
            "wall_elapsed_ms": self.wall_elapsed_ms,
            "logical_elapsed_ms": self.now_ms,
        }


@dataclass
class FastResultTiming:
    submitted_logical_ms: Optional[float]
    uplink_latency_ms: float
    llm_latency_ms: float
    traj_latency_ms: float

    @property
    def edge_arrival_logical_ms(self) -> Optional[float]:
        if self.submitted_logical_ms is None:
            return None
        return float(self.submitted_logical_ms + self.uplink_latency_ms)

    @property
    def ready_logical_ms(self) -> Optional[float]:
        if self.submitted_logical_ms is None:
            return None
        return float(
            self.submitted_logical_ms
            + self.uplink_latency_ms
            + self.llm_latency_ms
            + self.traj_latency_ms
        )


def fast_result_is_ready(
    clock: FastEvalClock,
    fast_eval: bool,
    ready_logical_ms: Optional[float],
) -> bool:
    """Return whether a completed worker result may enter the simulation."""
    return bool(
        not fast_eval
        or ready_logical_ms is None
        or clock.now_ms >= float(ready_logical_ms)
    )


def wait_for_fast_edge_worker_if_due(
    condition,
    clock: FastEvalClock,
    fast_eval: bool,
    has_result: Callable[[], bool],
    is_inflight: Callable[[], bool],
    edge_arrival_logical_ms: Callable[[], Optional[float]],
    get_error: Callable[[], Optional[BaseException]],
) -> float:
    """Freeze logical progress once the request reaches the real edge worker.

    Fast Eval cannot know the measured compute duration until the real worker
    finishes.  Once logical upload is complete, callers therefore wait in wall
    time without advancing the logical clock.  The completed result remains
    gated separately until its modeled ``ready_logical_ms``.

    The caller must hold ``condition`` while invoking this function.
    """
    wait_start = None
    while True:
        edge_arrival = edge_arrival_logical_ms()
        if not (
            fast_eval
            and not has_result()
            and is_inflight()
            and edge_arrival is not None
            and clock.now_ms >= float(edge_arrival)
        ):
            if wait_start is None:
                return 0.0
            return float((time.perf_counter() - wait_start) * 1000.0)
        if wait_start is None:
            wait_start = time.perf_counter()
        condition.wait(timeout=0.05)
        error = get_error()
        if error is not None:
            raise error


MIN_FAST_EVAL_SIM_MS = 10.0  # one AirSim physics tick (0.01 s)


def action_timing(eval_env, measured_wall_ms: float, fast_eval: bool) -> Dict[str, float]:
    wall_ms = float(measured_wall_ms)
    timings = getattr(eval_env, "last_action_timings", None) or []
    sim_values = [float(item.get("sim_time_ms", 0.0)) for item in timings if item]
    sim_ms = max(sim_values) if sim_values else wall_ms
    clamped = False
    if fast_eval:
        if sim_ms <= 0:
            # Zero-duration AirSim action: the UAV stalls close to the target and the
            # move completes within a single physics tick, so the sim timestamp does
            # not advance. The fast-eval logical clock must still move forward, so
            # clamp to one physics tick (the action is degenerate, not free).
            sim_ms = max(MIN_FAST_EVAL_SIM_MS, wall_ms)
            clamped = True
    result = {
        "action_wall_time_ms": wall_ms,
        "action_sim_time_ms": sim_ms,
        "airsim_action_latency_ms": sim_ms if fast_eval else wall_ms,
    }
    if clamped:
        result["sim_time_clamped_zero"] = True
    return result


def configure_fast_eval_output(args, paradigm: str) -> Optional[str]:
    enabled = as_bool(getattr(args, "fast_eval", False))
    if not enabled:
        return None
    speedup = float(getattr(args, "fast_eval_speedup", 5.0))
    suffix = f"_fast_x{speed_label(speedup)}"
    if not str(args.eval_save_path).endswith(suffix):
        args.eval_save_path = str(args.eval_save_path) + suffix
    os.makedirs(args.eval_save_path, exist_ok=True)
    manifest_path = os.path.join(args.eval_save_path, "fast_eval_manifest.json")
    manifest = {
        "fast_eval": True,
        "fast_eval_speedup": speedup,
        "paradigm": str(paradigm),
        "timing_contract": "original-speed logical clock",
    }
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != manifest:
            raise RuntimeError(f"Fast Eval manifest mismatch: {manifest_path}")
    else:
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest_path
