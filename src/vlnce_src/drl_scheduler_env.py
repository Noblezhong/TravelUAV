import copy
import json
import math
import os
import queue
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

sys.path.append(str(Path(str(os.getcwd())).resolve()))

from assist import Assist
from env_uav import AirVLNENV
from src.common.param import args
from src.model_wrapper.profile_travel_llm import ProfileTravelModelWrapper
from src.vlnce_src.comm_delay import (
    BandwidthTrace,
    calculate_latency_ms,
    estimate_uplink_payload_bits_from_episodes,
)
from src.vlnce_src.hybrid_eval import (
    ContinuousEpisodeState,
    PlannerResult,
    Snapshot,
    _active_action_delay_ms,
    _active_state_drift_m,
    _as_bool,
    _metric_summary,
    _point_delta,
    _write_jsonl_line,
)
from src.vlnce_src.fast_eval_time import FastEvalClock, FastResultTiming, action_timing
from utils.logger import logger


ACTION_NAMES = {
    0: "STOP_REQUEST",
    1: "STOP_NO_REQUEST",
    2: "CONTINUE_REQUEST",
    3: "CONTINUE_NO_REQUEST",
}

BUFFER_REMAINING_NORM = 7.0
NEXT_WAYPOINT_DISTANCE_NORM_M = 1.0
BANDWIDTH_NORM_BPS = 100_000_000.0
STATE_DRIFT_NORM_M = 2.5
TIME_DRIFT_NORM_MS = 5000.0


@dataclass
class DRLPlannerJob:
    snapshot: Snapshot
    bandwidth_bps: float


class FixedBandwidthEdgePlanner:
    def __init__(
        self,
        model_wrapper: ProfileTravelModelWrapper,
        enable_comm_delay: bool,
        clock: Optional[FastEvalClock] = None,
    ):
        self.model_wrapper = model_wrapper
        self.enable_comm_delay = bool(enable_comm_delay)
        self.clock = clock or FastEvalClock(False)
        self.fast_eval = bool(self.clock.enabled)
        self._pending_job: Optional[DRLPlannerJob] = None
        self._running = False
        self._closed = False
        self._cond = threading.Condition()
        self._results: "queue.Queue[PlannerResult]" = queue.Queue()
        self._held_result: Optional[PlannerResult] = None
        self._edge_arrival_logical_ms: Optional[float] = None
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def submit(self, snapshot: Snapshot, bandwidth_bps: float) -> bool:
        with self._cond:
            if self._closed:
                return False
            snapshot.planner_started_logical_ms = self.clock.now_ms if self.fast_eval else None
            self._pending_job = DRLPlannerJob(snapshot=copy.deepcopy(snapshot), bandwidth_bps=float(bandwidth_bps))
            self._cond.notify()
            return True

    def has_inflight(self) -> bool:
        with self._cond:
            return bool(
                self._running
                or self._pending_job is not None
                or self._held_result is not None
                or not self._results.empty()
            )

    def poll_result(self) -> Optional[PlannerResult]:
        with self._cond:
            if self._held_result is None:
                try:
                    self._held_result = self._results.get_nowait()
                except queue.Empty:
                    pass
            while (
                self.fast_eval
                and self._held_result is None
                and self._running
                and self._edge_arrival_logical_ms is not None
                and self.clock.now_ms >= self._edge_arrival_logical_ms
            ):
                self._cond.wait(timeout=0.05)
                try:
                    self._held_result = self._results.get_nowait()
                except queue.Empty:
                    pass
            if self._held_result is None:
                return None
            if self.fast_eval and self._held_result.ready_logical_ms is not None and self.clock.now_ms < self._held_result.ready_logical_ms:
                return None
            result = self._held_result
            self._held_result = None
            self._edge_arrival_logical_ms = None
            return result

    def wait_result(self) -> PlannerResult:
        with self._cond:
            held = self._held_result
            self._held_result = None
        result = held if held is not None else self._results.get()
        if self.fast_eval:
            self.clock.advance_to(result.ready_logical_ms)
        with self._cond:
            self._edge_arrival_logical_ms = None
        return result

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        self._worker.join(timeout=1.0)

    def _worker_loop(self) -> None:
        while True:
            with self._cond:
                while self._pending_job is None and not self._closed:
                    self._cond.wait()
                if self._closed:
                    return
                job = self._pending_job
                self._pending_job = None
                self._running = True
            try:
                assert job is not None
                self._results.put(self._run_job(job))
            except Exception as exc:
                logger.error(f"DRL scheduler planner failed: {exc}")
            finally:
                with self._cond:
                    self._running = False
                    self._cond.notify_all()

    def _run_job(self, job: DRLPlannerJob) -> PlannerResult:
        snapshot = job.snapshot
        episodes = copy.deepcopy([snapshot.episode])
        payload_bytes, payload_bits, payload_mb = estimate_uplink_payload_bits_from_episodes(episodes)
        uplink_latency_ms = calculate_latency_ms(payload_bits, job.bandwidth_bps) if self.enable_comm_delay else 0.0
        timing_start = snapshot.planner_started_logical_ms
        if timing_start is None:
            timing_start = snapshot.submitted_logical_ms
        if self.fast_eval and timing_start is not None:
            with self._cond:
                self._edge_arrival_logical_ms = float(timing_start + uplink_latency_ms)
                self._cond.notify_all()
        if uplink_latency_ms > 0:
            divisor = self.clock.speedup if self.fast_eval else 1.0
            time.sleep(uplink_latency_ms / (1000.0 * divisor))

        inputs, rot_to_targets = self.model_wrapper.prepare_inputs(
            episodes,
            [snapshot.target_position],
            [snapshot.assist_notice],
        )
        with torch.inference_mode():
            refined_waypoints, model_profile = self.model_wrapper.run_profiled(
                inputs=inputs,
                episodes=episodes,
                rot_to_targets=rot_to_targets,
            )
        fast_timing = FastResultTiming(
            timing_start,
            uplink_latency_ms,
            float(model_profile["llm_latency_ms"]),
            float(model_profile["traj_latency_ms"]),
        )
        return PlannerResult(
            request_id=int(snapshot.request_id),
            submitted_step=int(snapshot.submitted_step),
            observation_timestamp=int(snapshot.observation_timestamp),
            observation_pose=copy.deepcopy(snapshot.observation_pose),
            submitted_perf_time=float(snapshot.submitted_perf_time),
            submitted_wall_time=float(snapshot.submitted_wall_time),
            ready_wall_time=time.time(),
            obs_latency_ms=float(snapshot.obs_latency_ms),
            groundingdino_latency_ms=float(snapshot.groundingdino_latency_ms),
            llm_latency_ms=float(model_profile["llm_latency_ms"]),
            traj_latency_ms=float(model_profile["traj_latency_ms"]),
            uplink_payload_bits=int(payload_bits),
            uplink_payload_mb=float(payload_mb),
            uplink_bandwidth_mbps=float(job.bandwidth_bps / 1_000_000.0),
            uplink_latency_ms=float(uplink_latency_ms),
            llm_output=copy.deepcopy(model_profile["llm_output"]),
            refined_waypoints=copy.deepcopy(refined_waypoints[0]),
            submitted_logical_ms=snapshot.submitted_logical_ms,
            edge_arrival_logical_ms=fast_timing.edge_arrival_logical_ms,
            ready_logical_ms=fast_timing.ready_logical_ms,
        )


class DRLSchedulerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        model_wrapper: ProfileTravelModelWrapper,
        assist: Assist,
        eval_env: AirVLNENV,
        bandwidth_trace: BandwidthTrace,
        profile_log_path: str,
        summary_path: str,
        enable_comm_delay: bool = True,
        max_waypoints: Optional[int] = None,
        deterministic_eval: bool = False,
    ):
        super().__init__()
        assert int(eval_env.batch_size) == 1, "DRL scheduler currently supports batchSize=1 only"
        self.model_wrapper = model_wrapper
        self.assist = assist
        self.eval_env = eval_env
        self.bandwidth_trace = bandwidth_trace
        self.profile_log_path = profile_log_path
        self.summary_path = summary_path
        self.enable_comm_delay = bool(enable_comm_delay)
        self.max_waypoints = int(max_waypoints or args.max_control_steps)
        self.deterministic_eval = bool(deterministic_eval)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(8,), dtype=np.float32)

        os.makedirs(os.path.dirname(self.profile_log_path), exist_ok=True)
        self.profile_fp = open(self.profile_log_path, "w", encoding="utf-8")
        self.summary_records: List[Dict[str, Any]] = []
        self.episode_records: List[Dict[str, Any]] = []

        self.env_batch = None
        self.state: Optional[ContinuousEpisodeState] = None
        self.planner: Optional[FixedBandwidthEdgePlanner] = None
        self.active_traj: List[List[float]] = []
        self.active_index = 0
        self.active_result: Optional[PlannerResult] = None
        self.request_counter = 0
        self.control_step = 0
        self.scheduler_step_count = 0
        self.episode_idx = -1
        self.episode_start_perf = 0.0
        self.clock = FastEvalClock(bool(args.fast_eval), args.fast_eval_speedup)
        self.last_observed_bandwidth_bps = float(self.bandwidth_trace.next_bandwidth_bps())
        self.closed = False
        self._safety_last_threshold: Optional[float] = None
        self._last_request_step: int = 0  # step counter of last VLM request submission

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if self.planner is not None:
            self.planner.close()
        self._finalize_previous_episode(force=True)

        self.env_batch = self.eval_env.next_minibatch()
        if self.env_batch is None:
            self.env_batch = self.eval_env.next_minibatch()
        if self.env_batch is None:
            raise RuntimeError("No more evaluation episodes for DRL scheduler env")
        self.episode_idx += 1
        self.episode_records = []
        self.episode_start_perf = time.perf_counter()
        self.state = ContinuousEpisodeState(self.env_batch[0], self.eval_env, self.assist, ignore_tiny_diff=True)
        self.clock.reset()
        self.planner = FixedBandwidthEdgePlanner(self.model_wrapper, self.enable_comm_delay, clock=self.clock)
        self.active_traj = []
        self.active_index = 0
        self.active_result = None
        self.request_counter = 0
        self.control_step = 0
        self.scheduler_step_count = 0
        self._safety_last_threshold = None

        observed_bw = self._sample_bandwidth()
        snapshot = self.state.build_snapshot(self.request_counter, self.control_step)
        snapshot.submitted_logical_ms = self.clock.now_ms if self.clock.enabled else None
        self.request_counter += 1
        assert self.planner.submit(snapshot, observed_bw)
        wait_start = time.perf_counter()
        wait_logical_start = self.clock.now_ms
        result = self.planner.wait_result()
        wait_ms = (
            self.clock.now_ms - wait_logical_start
            if self.clock.enabled
            else (time.perf_counter() - wait_start) * 1000.0
        )
        self._apply_result(result)
        self._log_record(
            record_type="decision",
            action_id=0,
            observed_bandwidth_bps=observed_bw,
            request_bandwidth_bps=observed_bw,
            elapsed_ms=wait_ms,
            reward=0.0,
            reward_parts={"cold_start": 0.0},
            terminal_reason=None,
            extra={"cold_start": True, "hover_wait_ms": wait_ms},
        )
        return self._build_observation(observed_bw), self._build_info()

    def step(self, action):
        assert self.state is not None and self.planner is not None
        action_id = int(action)
        self.scheduler_step_count += 1
        step_start = time.perf_counter()
        step_logical_start = self.clock.now_ms
        observed_bw = self._sample_bandwidth()
        request_bw = None
        terminal_reason = None
        extra: Dict[str, Any] = {}

        applied = self._poll_and_apply_result()
        if applied is not None:
            extra["trajectory_switch_applied_before_action"] = True
        prev_ne_m = self._current_ne_m()
        prev_state_drift_m = self._current_drift_m()
        prev_time_drift_ms = self._current_time_drift_ms()

        motion_stop, request_edge = self._decode_action(action_id)

        # ── hard legality guard ──────────────────────────────────────
        legal, illegal_reason = self._check_action_legal(action_id)
        action_illegal = not legal
        extra["action_illegal"] = action_illegal
        extra["action_illegal_reason"] = illegal_reason
        if action_illegal:
            # override to the safe fallback: STOP_REQUEST
            motion_stop, request_edge = True, True
            extra["action_override"] = "STOP_REQUEST"
        # ─────────────────────────────────────────────────────────────

        if motion_stop:
            if request_edge:
                request_bw = observed_bw
                self._stop_and_request(observed_bw, extra)
            else:
                # STOP_NO_REQUEST: _check_action_legal guarantees inflight exists.
                # (If inflight were absent the action would be illegal and
                # overridden to STOP_REQUEST above.)
                wait_start = time.perf_counter()
                wait_logical_start = self.clock.now_ms
                result = self.planner.wait_result()
                extra["hover_wait_ms"] = float(
                    self.clock.now_ms - wait_logical_start
                    if self.clock.enabled
                    else (time.perf_counter() - wait_start) * 1000.0
                )
                self._apply_result(result)
        else:
            if self._buffer_remaining() <= 0:
                request_bw = observed_bw
                extra["forced_fallback"] = "empty_buffer_stop_request"
                self._stop_and_request(observed_bw, extra)
            else:
                self._execute_one_waypoint(extra)
                if request_edge and not self.state.dones[0]:
                    request_bw = observed_bw
                    self._request_async(observed_bw, extra)

        self._poll_and_apply_result()
        terminated, terminal_reason = self._check_terminal()

        # ── Safety-net DINO ──────────────────────────────────────────
        # DINO is decoupled from REQUEST.  When PPO chose NO_REQUEST and
        # the drone crosses a distance threshold (25 / 20 / 15 / 10 m)
        # without running DINO, the env runs one automatically.
        # This guarantees at least 2–4 detection shots near the target
        # regardless of PPO's request policy.  LLM is NOT invoked.
        # ──────────────────────────────────────────────────────────────
        if not terminated and extra.get("request_dino_latency_ms") is None:
            self._maybe_safety_dino(extra)
            terminated, terminal_reason = self._check_terminal()
        # ──────────────────────────────────────────────────────────────

        next_ne_m = self._current_ne_m()
        next_state_drift_m = self._current_drift_m()
        next_time_drift_ms = self._current_time_drift_ms()
        ne_progress_m = 0.0 if prev_ne_m is None or next_ne_m is None else float(prev_ne_m - next_ne_m)
        state_drift_delta_m = max(0.0, next_state_drift_m - prev_state_drift_m)
        time_drift_delta_ms = max(0.0, next_time_drift_ms - prev_time_drift_ms)
        extra["ne_before_m"] = float(prev_ne_m) if prev_ne_m is not None else None
        extra["ne_after_m"] = float(next_ne_m) if next_ne_m is not None else None
        extra["ne_progress_m"] = float(ne_progress_m)
        extra["state_drift_before_m"] = float(prev_state_drift_m)
        extra["state_drift_delta_m"] = float(state_drift_delta_m)
        extra["time_drift_before_ms"] = float(prev_time_drift_ms)
        extra["time_drift_delta_ms"] = float(time_drift_delta_ms)
        elapsed_ms = (
            float(self.clock.now_ms - step_logical_start)
            if self.clock.enabled
            else float((time.perf_counter() - step_start) * 1000.0)
        )
        # CONTINUE_REQUEST runs DINO while flying; the DINO time should not
        # inflate the time penalty because the drone isn't idling.
        dino_overhead_ms = 0.0
        if not motion_stop:
            dino_overhead_ms += float(extra.get("request_obs_latency_ms", 0) or 0)
            dino_overhead_ms += float(extra.get("request_dino_latency_ms", 0) or 0)
        effective_time_drift_ms = max(0.0, time_drift_delta_ms - dino_overhead_ms)
        reward, reward_parts = self._compute_reward(
            elapsed_ms=elapsed_ms,
            dino_overhead_ms=dino_overhead_ms,
            ne_progress_m=ne_progress_m,
            state_drift_delta_m=state_drift_delta_m,
            time_drift_delta_ms=effective_time_drift_ms,
            request_edge=request_bw is not None,
            terminated=terminated,
            terminal_reason=terminal_reason,
            action_illegal=action_illegal,
            oracle_success=bool(self.state.oracle_success),
        )
        obs = self._build_observation(observed_bw)
        self._log_record(
            record_type="scheduler_step",
            action_id=action_id,
            observed_bandwidth_bps=observed_bw,
            request_bandwidth_bps=request_bw,
            elapsed_ms=elapsed_ms,
            reward=reward,
            reward_parts=reward_parts,
            terminal_reason=terminal_reason,
            extra=extra,
        )
        if terminated:
            self._finalize_episode(terminal_reason)
        return obs, float(reward), bool(terminated), False, self._build_info(terminal_reason=terminal_reason)

    def close(self):
        if self.closed:
            return
        self._finalize_previous_episode(force=True)
        if self.planner is not None:
            self.planner.close()
            self.planner = None
        if not self.profile_fp.closed:
            self.profile_fp.close()
        self.closed = True

    def write_summary(self) -> None:
        action_counts = Counter(item.get("action_name") for item in self.summary_records if item.get("record_type") == "scheduler_step")
        summary = {
            "reward": _metric_summary([item.get("reward") for item in self.summary_records if item.get("record_type") == "scheduler_step"]),
            "elapsed_ms": _metric_summary([item.get("elapsed_ms") for item in self.summary_records if item.get("record_type") == "scheduler_step"]),
            "airsim_action_latency_ms": _metric_summary([item.get("airsim_action_latency_ms") for item in self.summary_records]),
            "time_drift_ms": _metric_summary([item.get("time_drift_ms") for item in self.summary_records]),
            "state_drift_m": _metric_summary([item.get("state_drift_m") for item in self.summary_records]),
            "ne_progress_m": _metric_summary([item.get("ne_progress_m") for item in self.summary_records]),
            "observed_bandwidth_mbps": _metric_summary([item.get("observed_bandwidth_mbps") for item in self.summary_records]),
            "request_bandwidth_mbps": _metric_summary([item.get("request_bandwidth_mbps") for item in self.summary_records]),
            "uplink_latency_ms": _metric_summary([item.get("uplink_latency_ms") for item in self.summary_records]),
            "episode_latency_ms": _metric_summary([item.get("episode_latency_ms") for item in self.summary_records if item.get("record_type") == "episode_end"]),
            "action_counts": dict(action_counts),
            "forced_fallback_count": int(sum(1 for item in self.summary_records if item.get("forced_fallback"))),
            "num_records": len(self.summary_records),
        }
        with open(self.summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

    def _decode_action(self, action_id: int) -> Tuple[bool, bool]:
        if action_id == 0:
            return True, True
        if action_id == 1:
            return True, False
        if action_id == 2:
            return False, True
        if action_id == 3:
            return False, False
        return True, True

    def _check_action_legal(self, action_id: int) -> Tuple[bool, Optional[str]]:
        """Check whether *action_id* is legal in the current state.

        Returns ``(legal, error_reason)``.  When *legal* is ``False`` the
        action is overridden to a safe fallback (STOP_REQUEST) and the
        step reward is lowered by ``scheduler_illegal_action_penalty``.
        """
        buffer_empty = self._buffer_remaining() <= 0
        has_inflight = self.planner is not None and self.planner.has_inflight()
        motion_stop, request_edge = self._decode_action(action_id)

        if not motion_stop and buffer_empty:
            # CONTINUE needs at least one buffered waypoint.
            return False, "continue_on_empty_buffer"
        if motion_stop and not request_edge and not has_inflight:
            # STOP_NO_REQUEST makes no sense when nothing is in flight.
            return False, "stop_no_request_without_inflight"
        return True, None

    def _sample_bandwidth(self) -> float:
        self.last_observed_bandwidth_bps = float(self.bandwidth_trace.next_bandwidth_bps())
        return self.last_observed_bandwidth_bps

    def _apply_result(self, result: PlannerResult) -> None:
        result.applied_logical_ms = self.clock.now_ms if self.clock.enabled else None
        result.applied_exec_step = int(self.control_step)
        self.active_result = result
        self.active_traj = copy.deepcopy(result.refined_waypoints)
        self.active_index = 0
        # Every result application = one completed REQUEST cycle.
        # Record NE for per-REQUEST distance regression.
        if self.state is not None and not self.state.dones[0]:
            self.state.record_request_ne()

    def _poll_and_apply_result(self) -> Optional[PlannerResult]:
        assert self.planner is not None
        result = self.planner.poll_result()
        if result is not None:
            self._apply_result(result)
        return result

    # ── Safety-net thresholds ───────────────────────────────────────
    _SAFETY_DINO_THRESHOLDS = (25.0, 20.0, 15.0, 10.0)

    def _maybe_safety_dino(self, extra: Dict[str, Any]) -> None:
        """Run DINO once when crossing each distance threshold, decoupled from REQUEST."""
        ne = self._current_ne_m()
        if ne is None or self.state.dones[0]:
            return
        current = self._safety_last_threshold
        for t in sorted(self._SAFETY_DINO_THRESHOLDS, reverse=True):
            if ne < t and (current is None or t < current):
                self._safety_last_threshold = t
                obs_start = time.perf_counter()
                outputs = self.eval_env.get_obs()
                obs_ms = float((time.perf_counter() - obs_start) * 1000.0)
                self.clock.advance_blocking(obs_ms)
                self.state.process_env_output(outputs)
                dino_ms = 0.0
                if not self.state.dones[0]:
                    dino_ms = float(self.state.run_dino_and_update_metric(self.model_wrapper))
                    self.clock.advance_blocking(dino_ms)
                extra["safety_dino_triggered"] = True
                extra["safety_dino_ne_m"] = float(ne)
                extra["safety_dino_obs_ms"] = obs_ms
                extra["safety_dino_latency_ms"] = dino_ms
                return

    def _capture_snapshot(self) -> Tuple[Optional[Snapshot], float, float]:
        assert self.state is not None
        obs_start = time.perf_counter()
        outputs = self.eval_env.get_obs()
        obs_latency_ms = float((time.perf_counter() - obs_start) * 1000.0)
        self.clock.advance_blocking(obs_latency_ms)
        self.state.process_env_output(outputs)
        dino_latency_ms = 0.0
        if not self.state.dones[0]:
            dino_latency_ms = float(self.state.run_dino_and_update_metric(self.model_wrapper))
            self.clock.advance_blocking(dino_latency_ms)
        if self.state.dones[0]:
            return None, obs_latency_ms, dino_latency_ms
        snapshot = self.state.build_snapshot(
            self.request_counter,
            self.control_step,
            obs_latency_ms=obs_latency_ms,
            groundingdino_latency_ms=dino_latency_ms,
        )
        snapshot.submitted_logical_ms = self.clock.now_ms if self.clock.enabled else None
        self.request_counter += 1
        return snapshot, obs_latency_ms, dino_latency_ms

    def _request_async(self, bandwidth_bps: float, extra: Dict[str, Any]) -> None:
        assert self.planner is not None
        snapshot, obs_ms, dino_ms = self._capture_snapshot()
        extra["request_obs_latency_ms"] = obs_ms
        extra["request_dino_latency_ms"] = dino_ms
        if snapshot is not None:
            self.planner.submit(snapshot, bandwidth_bps)
            self._last_request_step = self.scheduler_step_count
            extra["submitted_request_id"] = int(snapshot.request_id)

    def _stop_and_request(self, bandwidth_bps: float, extra: Dict[str, Any]) -> None:
        assert self.planner is not None and self.state is not None
        wait_start = time.perf_counter()
        wait_logical_start = self.clock.now_ms
        had_inflight = self.planner.has_inflight()
        snapshot, obs_ms, dino_ms = self._capture_snapshot()
        extra["request_obs_latency_ms"] = obs_ms
        extra["request_dino_latency_ms"] = dino_ms
        extra["stale_result_discarded"] = False
        if snapshot is None or self.state.dones[0]:
            extra["hover_wait_ms"] = float(
                self.clock.now_ms - wait_logical_start
                if self.clock.enabled
                else (time.perf_counter() - wait_start) * 1000.0
            )
            return
        self.planner.submit(snapshot, bandwidth_bps)
        self._last_request_step = self.scheduler_step_count
        result = self._wait_for_request_result(snapshot.request_id, extra)
        self._apply_result(result)
        extra["submitted_request_id"] = int(snapshot.request_id)
        extra["had_inflight_before_stop_request"] = bool(had_inflight)
        extra["hover_wait_ms"] = float(
            self.clock.now_ms - wait_logical_start
            if self.clock.enabled
            else (time.perf_counter() - wait_start) * 1000.0
        )

    def _wait_for_request_result(self, request_id: int, extra: Dict[str, Any]) -> PlannerResult:
        assert self.planner is not None
        discarded = 0
        while True:
            result = self.planner.wait_result()
            if int(result.request_id) == int(request_id):
                extra["discarded_stale_results"] = int(discarded)
                extra["stale_result_discarded"] = bool(discarded > 0)
                return result
            discarded += 1

    def _execute_one_waypoint(self, extra: Dict[str, Any]) -> None:
        assert self.state is not None
        current_chunk = self.active_traj[self.active_index : self.active_index + 1]
        exec_pose = self.state.current_sim_pose()
        extra["exec_target_distance_m"] = float(_point_delta(current_chunk[0], exec_pose))
        action_start = time.perf_counter()
        self.eval_env.makeActionsChunk([current_chunk], target_idx=1)
        measured_wall_ms = float((time.perf_counter() - action_start) * 1000.0)
        action_times = action_timing(self.eval_env, measured_wall_ms, self.clock.enabled)
        self.clock.advance_action(
            action_times["action_sim_time_ms"],
            action_times["action_wall_time_ms"],
        )
        self.active_index += 1
        self.control_step += 1
        self.state.sync_runtime_status_from_sim()
        extra.update(action_times)
        extra["executed_waypoints"] = 1

    # ── TERMINATION ──────────────────────────────────────────────────
    # Matches the unified contract in hybrid_eval.py ContinuousEpisodeState.
    # Three sources: collision / DINO (vision) / NE-trend (geometry).
    # oracle_success is a PASSIVE post-hoc flag — NEVER terminates.
    # DO NOT modify without updating ALL paradigms.
    # ──────────────────────────────────────────────────────────────────
    def _check_terminal(self) -> Tuple[bool, Optional[str]]:
        assert self.state is not None
        if self.state.collisions[0]:
            return True, "collision"
        if self.state.success:
            return True, "success"
        # oracle_success is intentionally NOT a termination condition.
        # It is a passive flag set by the sim when NE ≤ 20 m, used
        # post-hoc in _finalize_episode / _compute_reward — matching the
        # original stop-and-go and continuous paradigms.
        if self.state.dones[0]:
            return True, "done"
        if self.control_step >= self.max_waypoints:
            return True, "max_waypoints"
        if self.scheduler_step_count >= int(args.scheduler_max_steps):
            return True, "max_scheduler_steps"
        return False, None

    def _compute_reward(
        self,
        elapsed_ms: float,
        ne_progress_m: float,
        state_drift_delta_m: float,
        time_drift_delta_ms: float,
        request_edge: bool,
        terminated: bool,
        terminal_reason: Optional[str],
        action_illegal: bool = False,
        oracle_success: bool = False,
        dino_overhead_ms: float = 0.0,
    ) -> Tuple[float, Dict[str, float]]:
        ne_progress_reward = float(args.scheduler_ne_progress_weight) * float(
            np.clip(ne_progress_m / float(args.scheduler_ne_norm_m), -1.0, 1.0)
        )
        # DINO/obs time incurred by a REQUEST while the drone is flying
        # (CONTINUE) should not count against the time penalty — the drone
        # isn't hovering, so it's not "wasting" time.
        effective_ms = max(0.0, elapsed_ms - dino_overhead_ms)
        time_penalty = -float(args.scheduler_time_weight) * effective_ms / float(args.scheduler_time_norm_ms)
        drift_penalty = -float(args.scheduler_drift_weight) * state_drift_delta_m / float(args.scheduler_drift_norm_m)
        time_drift_penalty = -float(args.scheduler_time_drift_weight) * time_drift_delta_ms / float(args.scheduler_time_drift_norm_ms)
        request_penalty = -float(args.scheduler_request_weight) if request_edge else 0.0
        illegal_penalty = -float(args.scheduler_illegal_action_penalty) if action_illegal else 0.0
        terminal_reward = 0.0
        if terminated:
            if terminal_reason == "success":
                terminal_reward = float(args.scheduler_success_reward)
            elif terminal_reason == "collision":
                terminal_reward = -float(args.scheduler_collision_penalty)
            elif oracle_success:
                # Episode ended without SR but the drone passed within
                # 20 m of the target at some point → OSR (post-hoc flag).
                terminal_reward = float(args.scheduler_oracle_success_reward)
            elif terminal_reason in ("max_waypoints", "max_scheduler_steps", "done"):
                terminal_reward = -float(args.scheduler_failure_penalty)
        reward = ne_progress_reward + time_penalty + drift_penalty + time_drift_penalty + request_penalty + illegal_penalty + terminal_reward
        return reward, {
            "ne_progress": ne_progress_reward,
            "time": time_penalty,
            "state_drift_delta": drift_penalty,
            "time_drift_delta": time_drift_penalty,
            "request": request_penalty,
            "illegal_action": illegal_penalty,
            "terminal": terminal_reward,
        }

    def _build_observation(self, observed_bandwidth_bps: Optional[float] = None) -> np.ndarray:
        assert self.state is not None
        if observed_bandwidth_bps is None:
            observed_bandwidth_bps = self.last_observed_bandwidth_bps
        next_distance = self._next_waypoint_distance_m()
        cur_ne = self._current_ne_m()
        ne_norm = float(getattr(args, "scheduler_ne_state_norm_m", None) or 100.0)
        obs = np.asarray(
            [
                float(self._buffer_remaining()) / BUFFER_REMAINING_NORM,
                next_distance / NEXT_WAYPOINT_DISTANCE_NORM_M,
                float(observed_bandwidth_bps) / BANDWIDTH_NORM_BPS,
                1.0 if self.planner is not None and self.planner.has_inflight() else 0.0,
                self._current_drift_m() / float(args.scheduler_drift_norm_m or STATE_DRIFT_NORM_M),
                self._current_time_drift_ms() / float(args.scheduler_time_drift_norm_ms or TIME_DRIFT_NORM_MS),
                (cur_ne / ne_norm) if cur_ne is not None else 10.0,  # Critic-only: NE
                min(float(self.scheduler_step_count - self._last_request_step), 10.0) / 10.0,  # request age
            ],
            dtype=np.float32,
        )
        return np.clip(obs, self.observation_space.low, self.observation_space.high).astype(np.float32)

    def _build_info(self, terminal_reason: Optional[str] = None) -> Dict[str, Any]:
        return {
            "episode_idx": int(self.episode_idx),
            "control_step": int(self.control_step),
            "scheduler_step_count": int(self.scheduler_step_count),
            "buffer_remaining": int(self._buffer_remaining()),
            "state_drift_m": float(self._current_drift_m()),
            "time_drift_ms": float(self._current_time_drift_ms()),
            "terminal_reason": terminal_reason,
        }

    def _buffer_remaining(self) -> int:
        return max(0, len(self.active_traj) - self.active_index)

    def _next_waypoint_distance_m(self) -> float:
        if self.state is None or self.active_index >= len(self.active_traj):
            return 0.0
        return float(_point_delta(self.active_traj[self.active_index], self.state.current_sim_pose()))

    def _current_drift_m(self) -> float:
        if self.state is None:
            return 0.0
        return float(_active_state_drift_m(self.active_result, self.state.current_sim_pose()))

    def _current_time_drift_ms(self) -> float:
        value = _active_action_delay_ms(self.active_result, self.clock)
        return 0.0 if value is None else float(value)

    def _current_ne_m(self) -> Optional[float]:
        if self.state is None or not self.state.distance_to_ends:
            return None
        return float(self.state.distance_to_ends[-1])

    def _log_record(
        self,
        record_type: str,
        action_id: int,
        observed_bandwidth_bps: float,
        request_bandwidth_bps: Optional[float],
        elapsed_ms: float,
        reward: float,
        reward_parts: Dict[str, float],
        terminal_reason: Optional[str],
        extra: Dict[str, Any],
    ) -> None:
        assert self.state is not None
        record = {
            "record_type": record_type,
            "episode_idx": int(self.episode_idx),
            "seq_names": [self.env_batch[0]["seq_name"]] if self.env_batch else [],
            "map_names": [self.env_batch[0]["map_name"]] if self.env_batch else [],
            "action_id": int(action_id),
            "action_name": ACTION_NAMES.get(int(action_id), "UNKNOWN"),
            "motion_decision": "STOP" if int(action_id) in (0, 1) else "CONTINUE",
            "request_decision": "REQUEST" if int(action_id) in (0, 2) else "NO_REQUEST",
            "control_step": int(self.control_step),
            "scheduler_step_count": int(self.scheduler_step_count),
            "buffer_remaining": int(self._buffer_remaining()),
            "planner_has_inflight": bool(self.planner.has_inflight() if self.planner is not None else False),
            "active_request_id": int(self.active_result.request_id) if self.active_result is not None else None,
            "observed_bandwidth_mbps": float(observed_bandwidth_bps / 1_000_000.0),
            "request_bandwidth_mbps": float(request_bandwidth_bps / 1_000_000.0) if request_bandwidth_bps is not None else None,
            "uplink_latency_ms": float(self.active_result.uplink_latency_ms) if self.active_result is not None else None,
            "llm_latency_ms": float(self.active_result.llm_latency_ms) if self.active_result is not None else None,
            "traj_latency_ms": float(self.active_result.traj_latency_ms) if self.active_result is not None else None,
            "decision_total_latency_ms": (
                float(
                    self.active_result.obs_latency_ms
                    + self.active_result.groundingdino_latency_ms
                    + self.active_result.uplink_latency_ms
                    + self.active_result.llm_latency_ms
                    + self.active_result.traj_latency_ms
                )
                if self.active_result is not None
                else None
            ),
            "state_drift_m": float(self._current_drift_m()),
            "time_drift_ms": float(self._current_time_drift_ms()),
            "current_pose": copy.deepcopy(self.state.current_sim_pose()),
            "next_waypoint_distance_m": float(self._next_waypoint_distance_m()),
            "airsim_action_latency_ms": float(extra.get("airsim_action_latency_ms", 0.0)),
            "elapsed_ms": float(elapsed_ms),
            "reward": float(reward),
            "reward_parts": copy.deepcopy(reward_parts),
            "terminal_reason": terminal_reason,
            "predict_dones": [bool(x) for x in self.state.predict_dones],
            "collisions": [bool(x) for x in self.state.collisions],
            "dones": [bool(x) for x in self.state.dones],
        }
        record.update(copy.deepcopy(extra))
        record.update(self.clock.metadata())
        if self.active_result is not None:
            record["result_ready_logical_ms"] = self.active_result.ready_logical_ms
            record["result_applied_logical_ms"] = getattr(self.active_result, "applied_logical_ms", None)
            record["result_applied_exec_step"] = getattr(self.active_result, "applied_exec_step", None)
        _write_jsonl_line(self.profile_fp, record)
        self.summary_records.append(record)
        self.episode_records.append(record)

    def _finalize_episode(self, terminal_reason: Optional[str]) -> None:
        assert self.state is not None
        self.state.maybe_finalize(force=True)
        episode_latency_ms = (
            float(self.clock.now_ms)
            if self.clock.enabled
            else float((time.perf_counter() - self.episode_start_perf) * 1000.0)
        )
        record = {
            "record_type": "episode_end",
            "episode_idx": int(self.episode_idx),
            "seq_names": [self.env_batch[0]["seq_name"]] if self.env_batch else [],
            "map_names": [self.env_batch[0]["map_name"]] if self.env_batch else [],
            "terminal_reason": terminal_reason,
            "success": bool(self.state.success),
            "oracle_success": bool(self.state.oracle_success),
            "collision": bool(self.state.collisions[0]),
            "control_steps": int(self.control_step),
            "scheduler_steps": int(self.scheduler_step_count),
            "final_ne_m": float(self.state.distance_to_ends[-1]) if self.state.distance_to_ends else None,
            "episode_latency_ms": episode_latency_ms,
            "num_episode_records": len(self.episode_records),
        }
        record.update(self.clock.metadata())
        _write_jsonl_line(self.profile_fp, record)
        self.summary_records.append(record)

    def _finalize_previous_episode(self, force: bool = False) -> None:
        if self.state is not None and force and not self.state.skip_saved:
            self.state.maybe_finalize(force=True)
