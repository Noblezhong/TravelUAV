"""MATCH paradigm evaluator: AHS (PPO hybrid scheduler) + TCM (trajectory correction).

NEW FILE — the MATCH paradigm of paper Algorithm 1.  It subclasses
``DRLSchedulerEnv`` and adds the TCM branch at result application plus the
target-lock state machine.  No existing evaluator file is modified.

Fidelity contract:
  - ``--trajcorr_mode off``: behavior identical to ``drl_scheduler_eval.py``
    (the audited PPO row; must reproduce SR 47.9 on the 0722 weights).
  - ``--trajcorr_mode on --trajcorr_state_shift_threshold_m 2.5``: MATCH.

record_type vocabulary is unchanged: {decision, scheduler_step, episode_end}.
TCM fields are extra keys only (compute_metrics.py auto-detection stays ppo).

Keep in sync with ``drl_scheduler_env.py``: if the parent's step() changes,
the overridden step() below must be re-checked.
"""

import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import tqdm

sys.path.append(str(Path(str(os.getcwd())).resolve()))

from assist import Assist
from src.common.param import args, data_args, model_args
from src.model_wrapper.profile_travel_llm import ProfileTravelModelWrapper
from src.vlnce_src.closeloop_util import (
    BatchIterator,
    CheckPort,
    initialize_env_eval,
    is_dist_avail_and_initialized,
    setup,
)
from src.vlnce_src.comm_delay import BandwidthTrace, default_trace_path
from src.vlnce_src.continue_eval import (
    _fmt_optional_ms,
    _fmt_optional_mbps,
    _print_episode_header,
    _print_trajectory_bundle,
)
from src.vlnce_src.dino_monitor_online import DinoMonitor
from src.vlnce_src.drl_ac_policy import SplitACPolicy  # required for PPO.load deserialization
from src.vlnce_src.drl_scheduler_env import (
    ACTION_NAMES,
    DRLSchedulerEnv,
    _as_bool,
    _metric_summary,
)
from src.vlnce_src.aerodpo_eval_contract import resolve_aerodpo_stop
from src.vlnce_src.fast_eval_time import action_timing, configure_fast_eval_output
from src.vlnce_src.ncn_runtime import EdgeLatencyEstimate, NCNRuntime
from src.vlnce_src.trajcorr_apply import TcmRuntime, TrajcorrMixin
from src.vlnce_src.rule_based_eval import _write_jsonl_line
from utils.logger import logger


class MatchProfileTravelModelWrapper(TrajcorrMixin, ProfileTravelModelWrapper):
    pass


class MatchEnv(DRLSchedulerEnv):
    def __init__(self, *pargs, ncn_runtime: Optional[NCNRuntime] = None, **kwargs):
        super().__init__(*pargs, **kwargs)
        self._tcm_enabled = str(getattr(args, "trajcorr_mode", "on")).strip().lower() == "on"
        self._delta_cor_m = float(getattr(args, "trajcorr_state_shift_threshold_m", 2.5))
        self.tcm_runtime = TcmRuntime(self._tcm_enabled, self._delta_cor_m)
        self.target_lock = self.tcm_runtime.target_lock
        self._needs_target_lock_refresh = False
        self._tcm_step_extras: Dict[str, Any] = {}
        self.ncn_runtime = ncn_runtime
        self._ncn_enabled = bool(ncn_runtime is not None and _as_bool(args.enable_ncn))
        self._ncn_active = False
        self._ncn_stale_request_ids: Set[int] = set()
        self._ncn_recovery_request_id: Optional[int] = None
        self._ncn_actions_in_interval = 0
        self._ncn_takeovers = 0
        self._ncn_stale_results = 0
        self._ncn_recoveries = 0
        self._ncn_recovery_wait_start_ms: Optional[float] = None
        self._ncn_recovery_wait_ms: List[float] = []
        self._edge_latency: Optional[EdgeLatencyEstimate] = None
        self._outage_start_ms: Optional[float] = None
        self._outage_held_results: List[Any] = []

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        self.tcm_runtime = TcmRuntime(self._tcm_enabled, self._delta_cor_m)
        self.target_lock = self.tcm_runtime.target_lock
        self._needs_target_lock_refresh = False
        self._tcm_step_extras = {}
        self._ncn_active = False
        self._ncn_stale_request_ids = set()
        self._ncn_recovery_request_id = None
        self._ncn_actions_in_interval = 0
        self._ncn_takeovers = 0
        self._ncn_stale_results = 0
        self._ncn_recoveries = 0
        self._ncn_recovery_wait_start_ms = None
        self._ncn_recovery_wait_ms = []
        self._outage_held_results = []
        obs, info = super().reset(seed=seed, options=options)
        warmup_compute_ms = 0.0
        if self.active_result is not None:
            warmup_compute_ms = float(self.active_result.llm_latency_ms) + float(self.active_result.traj_latency_ms)
        self._edge_latency = EdgeLatencyEstimate(
            compute_ms=warmup_compute_ms,
            alpha=float(args.ncn_response_ema_alpha),
        )
        self._outage_start_ms = float(self.clock.now_ms) if float(args.ncn_outage_duration_ms) > 0 else None
        return obs, info

    # ── TCM: result application (Algorithm 1 IF branch) ─────────────
    def _apply_result(self, result) -> None:
        if self._edge_latency is not None:
            self._edge_latency.update(result.llm_latency_ms, result.traj_latency_ms)
        effective, tcm_extra = self.tcm_runtime.apply_result(
            state=self.state,
            eval_env=self.eval_env,
            model_wrapper=self.model_wrapper,
            clock=self.clock,
            result=result,
            record_request_ne=False,  # super()._apply_result records it
        )
        if effective is None:
            # stale result during target lock — dropped, keep executing the lock
            return
        super()._apply_result(effective)
        self._tcm_step_extras.update(tcm_extra)

    # ── Legacy AHS+TCM path, retained bit-for-bit when NCN is disabled ─
    def step(self, action):
        if not self._ncn_enabled:
            return self._step_edge_only(action)
        return self._step_with_ncn(action)

    def _step_edge_only(self, action):
        assert self.state is not None and self.planner is not None
        action_id = int(action)
        self.scheduler_step_count += 1
        step_start = time.perf_counter()
        step_logical_start = self.clock.now_ms
        observed_bw = self._sample_bandwidth()
        request_bw = None
        terminal_reason = None
        extra: Dict[str, Any] = {}
        self._tcm_step_extras = {}

        if self._needs_target_lock_refresh and not self.state.dones[0]:
            return self._step_refresh_after_lock(observed_bw, step_start, step_logical_start)

        applied = self._poll_and_apply_result()
        if applied is not None:
            extra["trajectory_switch_applied_before_action"] = True
        if self.target_lock.active:
            return self._step_target_lock(observed_bw, step_start, step_logical_start)

        prev_ne_m = self._current_ne_m()
        prev_state_drift_m = self._current_drift_m()
        prev_time_drift_ms = self._current_time_drift_ms()
        # pre-action decision state for the pilot7 shaping rules (what the
        # agent decided on, NOT the post-execution state).
        pre_buffer_remaining = self._buffer_remaining()
        pre_has_inflight = self.planner.has_inflight()

        motion_stop, request_edge = self._decode_action(action_id)
        # original decoded intent, captured BEFORE the legality override so
        # the shaping can judge the action the agent actually chose.
        orig_motion_stop, orig_request_edge = motion_stop, request_edge

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
        reward, reward_parts = self._compute_reward(
            elapsed_ms=elapsed_ms,
            terminated=terminated,
            terminal_reason=terminal_reason,
            action_illegal=action_illegal,
            oracle_success=bool(self.state.oracle_success),
            # pre-action decision state + original intent for the shaping rules
            time_drift_ms=prev_time_drift_ms,
            state_drift_m=prev_state_drift_m,
            buffer_remaining=pre_buffer_remaining,
            observed_bw=observed_bw,
            has_inflight=pre_has_inflight,
            orig_request_edge=orig_request_edge,
            orig_motion_stop=orig_motion_stop,
        )
        obs = self._build_observation(observed_bw)
        extra.update(self._tcm_step_extras)
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

    # ── Final MATCH: NCN recovery coordinator ───────────────────────
    def _outage_active(self) -> bool:
        if self._outage_start_ms is None:
            return False
        return float(self.clock.now_ms) < self._outage_start_ms + float(args.ncn_outage_duration_ms)

    def _remaining_buffer_horizon_ms(self) -> float:
        """Original-speed duration estimate of the currently executable buffer."""
        if self.state is None or self._buffer_remaining() <= 0:
            return 0.0
        points = [np.asarray(self.state.current_sim_pose(), dtype=np.float64)]
        points.extend(np.asarray(point, dtype=np.float64) for point in self.active_traj[self.active_index :])
        distance_m = sum(float(np.linalg.norm(points[index + 1] - points[index])) for index in range(len(points) - 1))
        # AirVLNSimulatorClientTool.move_path_by_velocity_waypoints uses 1 m/s.
        return max(0.0, distance_m * 1000.0)

    def _poll_edge_result_for_match(self):
        if not self._outage_active() and self._outage_held_results:
            return self._outage_held_results.pop(0)
        result = self.planner.poll_result()
        if result is not None and self._outage_active():
            self._outage_held_results.append(result)
            return None
        return result

    def _ncn_takeover_reason(self) -> Optional[str]:
        if self.target_lock.active:
            return None
        if self._outage_active():
            return "configured_outage"
        timeout_ms = float(args.ncn_edge_timeout_ms)
        age_ms = self.planner.request_age_ms()
        if timeout_ms > 0.0 and age_ms is not None and age_ms >= timeout_ms:
            return "edge_timeout"
        if not self.planner.has_inflight() or self._edge_latency is None:
            return None
        predicted_ms = self.planner.predicted_remaining_ms(self._edge_latency.compute_ms)
        if predicted_ms is None:
            return None
        if predicted_ms > self._remaining_buffer_horizon_ms() + float(args.ncn_response_safety_margin_ms):
            return "buffer_insufficient_for_predicted_edge_response"
        return None

    def _activate_ncn(self, reason: str, *, recovery_request_id: Optional[int] = None) -> None:
        if not self._ncn_active:
            self._ncn_active = True
            self._ncn_actions_in_interval = 0
            self._ncn_takeovers += 1
        if recovery_request_id is None:
            self._ncn_stale_request_ids.update(self.planner.pending_request_ids())
            self._ncn_recovery_request_id = None
        else:
            self._ncn_recovery_request_id = int(recovery_request_id)
        if self._ncn_recovery_wait_start_ms is None:
            self._ncn_recovery_wait_start_ms = float(self.clock.now_ms)

    def _submit_ncn_recovery_request(self, observed_bw: float, extra: Dict[str, Any]) -> bool:
        """Capture a current edge observation after a stale NCN-era result."""
        snapshot, obs_ms, dino_ms = self._capture_snapshot()
        extra["recovery_request_obs_latency_ms"] = float(obs_ms)
        extra["recovery_request_dino_latency_ms"] = float(dino_ms)
        if snapshot is None or self.state.dones[0]:
            extra["recovery_submit_skipped_done"] = True
            return False
        if not self.planner.submit(snapshot, observed_bw):
            raise RuntimeError("MATCH NCN recovery request was rejected by edge planner")
        self._last_request_step = self.scheduler_step_count
        self._ncn_recovery_request_id = int(snapshot.request_id)
        extra["ncn_recovery_request_id"] = int(snapshot.request_id)
        return True

    def _complete_ncn_recovery(self) -> None:
        self._ncn_active = False
        self._ncn_stale_request_ids.clear()
        self._ncn_recovery_request_id = None
        self._ncn_actions_in_interval = 0
        self._ncn_recoveries += 1
        if self._ncn_recovery_wait_start_ms is not None:
            self._ncn_recovery_wait_ms.append(max(0.0, float(self.clock.now_ms) - self._ncn_recovery_wait_start_ms))
        self._ncn_recovery_wait_start_ms = None

    def _log_ncn_event(self, observed_bw: float, extra: Dict[str, Any], terminal_reason: Optional[str] = None):
        terminated, detected_reason = self._check_terminal()
        if terminal_reason is None:
            terminal_reason = detected_reason
        terminated = bool(terminated or terminal_reason is not None)
        extra.setdefault("action_name", "NCN")
        extra.setdefault("motion_decision", "NCN")
        extra.setdefault("request_decision", "NO_REQUEST")
        extra.setdefault("control_source", "ncn")
        extra.setdefault("ncn_active", bool(self._ncn_active))
        self._log_record(
            record_type="scheduler_step",
            action_id=3,
            observed_bandwidth_bps=observed_bw,
            request_bandwidth_bps=None,
            elapsed_ms=float(extra.get("elapsed_ms", 0.0)),
            reward=0.0,
            reward_parts={},
            terminal_reason=terminal_reason,
            extra=extra,
        )
        if terminated:
            self._finalize_episode(terminal_reason)
        return self._build_observation(observed_bw), 0.0, terminated, False, self._build_info(terminal_reason=terminal_reason)

    def _run_ncn_action(self, observed_bw: float, trigger_reason: str, seed_extra: Optional[Dict[str, Any]] = None):
        assert self.state is not None and self.ncn_runtime is not None
        self.scheduler_step_count += 1
        extra: Dict[str, Any] = dict(seed_extra or {})
        if self._ncn_actions_in_interval >= int(args.ncn_max_consecutive_actions):
            extra.update({"control_source": "ncn", "ncn_trigger_reason": trigger_reason, "ncn_action_limit_reached": True})
            return self._log_ncn_event(observed_bw, extra, terminal_reason="ncn_action_limit")

        step_start = time.perf_counter()
        logical_start_ms = float(self.clock.now_ms)
        image_start = time.perf_counter()
        observation = self.eval_env.get_aerodpo_rgb_observations()[0]
        image_fetch_ms = (time.perf_counter() - image_start) * 1000.0
        decision_start = time.perf_counter()
        action, model_stop, profile, prompt = self.ncn_runtime.act(
            observation["front_bgr"], observation["down_bgr"], observation["state"],
            self.state.target_position, self.env_batch[0]["instruction"],
        )
        image_text_to_action_ms = (time.perf_counter() - decision_start) * 1000.0
        self.clock.advance_blocking(image_text_to_action_ms)

        action_start = time.perf_counter()
        self.eval_env.makeAeroDPOActions([action])
        action_times = action_timing(self.eval_env, (time.perf_counter() - action_start) * 1000.0, self.clock.enabled)
        self.clock.advance_action(action_times["action_sim_time_ms"], action_times["action_wall_time_ms"])
        self.control_step += 1
        self._ncn_actions_in_interval += 1
        self.state.sync_runtime_status_from_sim()

        post_obs_start = time.perf_counter()
        self.state.process_env_output(self.eval_env.get_obs())
        post_obs_ms = (time.perf_counter() - post_obs_start) * 1000.0
        self.clock.advance_blocking(post_obs_ms)
        land_outcome = resolve_aerodpo_stop(
            bool(model_stop), self.state._calculate_distance_from_position(self.state.current_sim_pose())
        )
        terminal_reason = None
        if land_outcome == "success":
            self.state.success = True
            self.state.dones[0] = True
            terminal_reason = "success"
        elif land_outcome == "early_end":
            self.state.early_end = True
            self.state.dones[0] = True
            terminal_reason = "early_end"

        predicted_ms = self.planner.predicted_remaining_ms(self._edge_latency.compute_ms) if self._edge_latency else None
        extra.update({
            "control_source": "ncn",
            "ncn_trigger_reason": trigger_reason,
            "ncn_recovery_request_id": self._ncn_recovery_request_id,
            "ncn_stale_request_ids": sorted(self._ncn_stale_request_ids),
            "predicted_edge_remaining_ms": predicted_ms,
            "remaining_buffer_horizon_ms": self._remaining_buffer_horizon_ms(),
            "aerodpo_action": [action],
            "model_stop": [bool(model_stop)],
            "termination_backend": "aerodpo_land",
            "model_output_text": profile.get("model_output_text"),
            "ncn_prompt": prompt,
            "ncn_image_fetch_ms": float(image_fetch_ms),
            "image_text_to_action_latency_ms": float(image_text_to_action_ms),
            "model_inference_latency_ms": float(profile["model_inference_latency_ms"]),
            "native_bridge_latency_ms": profile.get("native_bridge_latency_ms"),
            "native_vision_encode_latency_ms": profile.get("native_vision_encode_latency_ms"),
            "native_prefill_latency_ms": profile.get("native_prefill_latency_ms"),
            "native_decode_latency_ms": profile.get("native_decode_latency_ms"),
            "airsim_action_latency_ms": float(action_times["airsim_action_latency_ms"]),
            "action_wall_time_ms": float(action_times["action_wall_time_ms"]),
            "action_sim_time_ms": float(action_times["action_sim_time_ms"]),
            "execution_post_obs_latency_ms": float(post_obs_ms),
            "elapsed_ms": (
                float(self.clock.now_ms) - logical_start_ms
                if self.clock.enabled
                else (time.perf_counter() - step_start) * 1000.0
            ),
        })
        return self._log_ncn_event(observed_bw, extra, terminal_reason=terminal_reason)

    def _step_with_ncn(self, action):
        assert self.state is not None and self.planner is not None
        observed_bw = self._sample_bandwidth()
        self._tcm_step_extras = {}

        # Target-lock is the highest-priority mode and never delegates to NCN.
        if self._needs_target_lock_refresh and not self.state.dones[0]:
            return self._step_refresh_after_lock_ncn(observed_bw)
        if self.target_lock.active:
            self.scheduler_step_count += 1
            return self._step_target_lock(observed_bw, time.perf_counter(), self.clock.now_ms)

        result = self._poll_edge_result_for_match()
        if result is not None and self._ncn_active:
            extra: Dict[str, Any] = {"control_source": "ncn", "ncn_active": True}
            if self._ncn_recovery_request_id is not None and int(result.request_id) == self._ncn_recovery_request_id:
                self._apply_result(result)
                if self.target_lock.active or self._buffer_remaining() > 0:
                    self._complete_ncn_recovery()
                    extra.update({"control_source": "edge_recovery", "ncn_recovery_reason": "valid_edge_buffer_applied"})
                    return self._log_ncn_event(observed_bw, extra)
                extra["ncn_recovery_invalid_buffer"] = True
                self._submit_ncn_recovery_request(observed_bw, extra)
                return self._run_ncn_action(observed_bw, "invalid_recovery_buffer", extra)

            # A request captured before NCN first moved is semantically stale.
            self._ncn_stale_results += 1
            self._ncn_stale_request_ids.discard(int(result.request_id))
            extra.update({"ncn_discarded_stale_result_id": int(result.request_id), "ncn_recovery_reason": "stale_edge_result"})
            self._submit_ncn_recovery_request(observed_bw, extra)
            return self._run_ncn_action(observed_bw, "stale_result_recovery", extra)

        if self._ncn_active:
            return self._run_ncn_action(observed_bw, "ncn_interval")

        reason = self._ncn_takeover_reason()
        if reason is not None:
            # An outage can start with a valid cold-start buffer but no pending
            # request.  Submit one now so its delayed return signals recovery.
            if not self.planner.has_inflight():
                bootstrap_extra: Dict[str, Any] = {}
                self._submit_ncn_recovery_request(observed_bw, bootstrap_extra)
                self._ncn_recovery_request_id = None
            self._activate_ncn(reason)
            return self._run_ncn_action(observed_bw, reason)

        # No priority branch fired: preserve the audited AHS+TCM execution.
        return self._step_edge_only(action)

    def _step_refresh_after_lock_ncn(self, observed_bw: float):
        """Fresh post-lock request is current-state recovery input, not stale."""
        self.scheduler_step_count += 1
        extra: Dict[str, Any] = {"forced_refresh_after_target_lock": True}
        extra["target_lock_completion_reason"] = self.target_lock.completion_reason
        self._needs_target_lock_refresh = False
        self.target_lock.resume_normal()
        if not self._submit_ncn_recovery_request(observed_bw, extra):
            return self._log_ncn_event(observed_bw, extra)
        self._activate_ncn("post_target_lock_refresh", recovery_request_id=self._ncn_recovery_request_id)
        return self._run_ncn_action(observed_bw, "post_target_lock_refresh", extra)

    def _stop_and_request(self, bandwidth_bps: float, extra: Dict[str, Any]) -> None:
        """Make an AHS STOP_REQUEST preemptible only for final MATCH."""
        if not self._ncn_enabled or self._ncn_active:
            return super()._stop_and_request(bandwidth_bps, extra)
        snapshot, obs_ms, dino_ms = self._capture_snapshot()
        extra["request_obs_latency_ms"] = obs_ms
        extra["request_dino_latency_ms"] = dino_ms
        if snapshot is None or self.state.dones[0]:
            return
        self.planner.submit(snapshot, bandwidth_bps)
        self._last_request_step = self.scheduler_step_count
        extra["submitted_request_id"] = int(snapshot.request_id)
        reason = self._ncn_takeover_reason()
        if reason is not None:
            self._activate_ncn(reason)
            extra["ncn_preempted_stop_request"] = True
            extra["ncn_takeover_reason"] = reason
            return
        wait_start = time.perf_counter()
        wait_logical_start = self.clock.now_ms
        result = self._wait_for_request_result(snapshot.request_id, extra)
        self._apply_result(result)
        extra["hover_wait_ms"] = float(
            self.clock.now_ms - wait_logical_start if self.clock.enabled else (time.perf_counter() - wait_start) * 1000.0
        )

    # ── TCM: target-lock step (AHS suspended, buffer-only execution) ─
    def _step_target_lock(self, observed_bw, step_start, step_logical_start):
        extra: Dict[str, Any] = {}
        # On the lock-entry step, expose the corrected apply as a shift sample
        # (same protocol as trajectory_switch_applied_before_action): the drift
        # before the locked execution is exactly the state shift at apply.
        entry_corrected = self._tcm_step_extras.get("trajectory_mode") == "corrected"
        if entry_corrected:
            extra["trajectory_switch_applied_before_action"] = True
            extra["state_drift_before_m"] = float(self._current_drift_m())
            extra["time_drift_before_ms"] = float(self._current_time_drift_ms())
        dropped = 0
        while self.planner.poll_result() is not None:
            dropped += 1
        if dropped:
            extra["target_lock_dropped_stale_results"] = int(dropped)
            self.tcm_runtime.stats.stale_dropped += dropped
        if self._buffer_remaining() > 0 and not self.state.dones[0]:
            self._execute_one_waypoint(extra)
        completion = self.target_lock.evaluate(self.state.current_sim_pose())
        if completion is None and self._buffer_remaining() <= 0 and not self.state.dones[0]:
            completion = self.target_lock.mark_buffer_exhausted()
        if completion is not None:
            extra["target_lock_completion_reason"] = completion
            self.tcm_runtime.stats.lock_completions[completion] += 1
            self._needs_target_lock_refresh = True
            self.active_traj = []
            self.active_index = 0
        extra["target_lock_active"] = True
        terminated, terminal_reason = self._check_terminal()
        if not terminated and extra.get("request_dino_latency_ms") is None:
            self._maybe_safety_dino(extra)
            terminated, terminal_reason = self._check_terminal()
        elapsed_ms = (
            float(self.clock.now_ms - step_logical_start)
            if self.clock.enabled
            else float((time.perf_counter() - step_start) * 1000.0)
        )
        obs = self._build_observation(observed_bw)
        extra.update(self._tcm_step_extras)
        self._log_record(
            record_type="scheduler_step",
            action_id=3,
            observed_bandwidth_bps=observed_bw,
            request_bandwidth_bps=None,
            elapsed_ms=elapsed_ms,
            reward=0.0,
            reward_parts={},
            terminal_reason=terminal_reason,
            extra=extra,
        )
        if terminated:
            self._finalize_episode(terminal_reason)
        return obs, 0.0, bool(terminated), False, self._build_info(terminal_reason=terminal_reason)

    # ── TCM: forced fresh request after lock completion ─────────────
    def _step_refresh_after_lock(self, observed_bw, step_start, step_logical_start):
        extra: Dict[str, Any] = {}
        reason = self.target_lock.completion_reason
        self._needs_target_lock_refresh = False
        self.target_lock.resume_normal()
        extra["forced_refresh_after_target_lock"] = True
        extra["target_lock_completion_reason"] = reason
        wait_start = time.perf_counter()
        wait_logical_start = self.clock.now_ms
        snapshot, obs_ms, dino_ms = self._capture_snapshot()
        extra["request_obs_latency_ms"] = obs_ms
        extra["request_dino_latency_ms"] = dino_ms
        extra["stale_result_discarded"] = False
        if snapshot is not None and not self.state.dones[0]:
            self.planner.submit(snapshot, observed_bw)
            self._last_request_step = self.scheduler_step_count
            result = self._wait_for_request_result(snapshot.request_id, extra)
            self._apply_result(result)
            extra["submitted_request_id"] = int(snapshot.request_id)
        else:
            extra["refresh_submit_skipped_done"] = True
        extra["hover_wait_ms"] = float(
            self.clock.now_ms - wait_logical_start
            if self.clock.enabled
            else (time.perf_counter() - wait_start) * 1000.0
        )
        terminated, terminal_reason = self._check_terminal()
        if not terminated and extra.get("request_dino_latency_ms") is None:
            self._maybe_safety_dino(extra)
            terminated, terminal_reason = self._check_terminal()
        elapsed_ms = (
            float(self.clock.now_ms - step_logical_start)
            if self.clock.enabled
            else float((time.perf_counter() - step_start) * 1000.0)
        )
        obs = self._build_observation(observed_bw)
        extra.update(self._tcm_step_extras)
        self._log_record(
            record_type="scheduler_step",
            action_id=0,
            observed_bandwidth_bps=observed_bw,
            request_bandwidth_bps=observed_bw,
            elapsed_ms=elapsed_ms,
            reward=0.0,
            reward_parts={},
            terminal_reason=terminal_reason,
            extra=extra,
        )
        if terminated:
            self._finalize_episode(terminal_reason)
        return obs, 0.0, bool(terminated), False, self._build_info(terminal_reason=terminal_reason)

    # ── TCM: merge per-episode stats into episode_end ────────────────
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
        record.update(self.tcm_runtime.stats.as_dict())
        ncn_actions = sum(
            1 for item in self.episode_records if item.get("control_source") == "ncn"
        )
        record.update({
            "ncn_enabled": bool(self._ncn_enabled),
            "ncn_actions": int(ncn_actions),
            "ncn_takeovers": int(self._ncn_takeovers),
            "ncn_stale_results": int(self._ncn_stale_results),
            "ncn_recoveries": int(self._ncn_recoveries),
            "ncn_recovery_wait_ms": list(self._ncn_recovery_wait_ms),
        })
        record.update(self.clock.metadata())
        _write_jsonl_line(self.profile_fp, record)
        self.summary_records.append(record)


def main():
    from stable_baselines3 import PPO

    if not args.scheduler_model_path:
        raise ValueError("--scheduler_model_path is required for MATCH eval")
    configure_fast_eval_output(args, "drl_based_hybrid")
    os.makedirs(args.eval_save_path, exist_ok=True)
    profile_dir = os.path.join(args.eval_save_path, "profile_logs")
    os.makedirs(profile_dir, exist_ok=True)

    enable_comm_delay = _as_bool(args.enable_comm_delay)
    trace_path = Path(args.comm_trace_csv_path) if args.comm_trace_csv_path else Path(default_trace_path())
    bandwidth_trace = BandwidthTrace(trace_path, cycle=True)
    logger.info(
        f"MATCH eval trace={trace_path}, samples={bandwidth_trace.sample_count}, "
        f"trajcorr_mode={args.trajcorr_mode}, enable_ncn={args.enable_ncn}, "
        f"trajcorr_state_shift_threshold_m={args.trajcorr_state_shift_threshold_m}"
    )

    setup()
    assert CheckPort(), "error port"
    eval_env = initialize_env_eval(
        dataset_path=args.dataset_path,
        save_path=args.eval_save_path,
        eval_json_path=args.eval_json_path,
    )
    total_episodes = len(BatchIterator(eval_env))
    if is_dist_avail_and_initialized():
        torch.distributed.destroy_process_group()
    args.DistributedDataParallel = False

    model_wrapper = MatchProfileTravelModelWrapper(model_args=model_args, data_args=data_args)
    model_wrapper.dino_moinitor = DinoMonitor.get_instance()
    model_wrapper.eval()
    assist = Assist(always_help=args.always_help, use_gt=args.use_gt)
    ncn_runtime = NCNRuntime(model_args) if _as_bool(args.enable_ncn) else None

    gym_env = MatchEnv(
        model_wrapper=model_wrapper,
        assist=assist,
        eval_env=eval_env,
        bandwidth_trace=bandwidth_trace,
        profile_log_path=os.path.join(profile_dir, f"drl_eval_{args.make_dir_time}.jsonl"),
        summary_path=os.path.join(profile_dir, f"drl_eval_{args.make_dir_time}_summary.json"),
        enable_comm_delay=enable_comm_delay,
        max_waypoints=args.max_control_steps,
        deterministic_eval=True,
        ncn_runtime=ncn_runtime,
    )
    scheduler = PPO.load(args.scheduler_model_path, env=None, device="cuda" if torch.cuda.is_available() else "cpu")

    pbar = tqdm.tqdm(total=total_episodes)
    completed = 0
    while completed < total_episodes:
        obs, _ = gym_env.reset()
        pbar.update(1)

        env_batch = gym_env.env_batch[0] if isinstance(gym_env.env_batch, list) else gym_env.env_batch
        _print_episode_header(completed, env_batch, chunk_waypoints=1)

        terminated = False
        truncated = False
        prev_control_step = gym_env.control_step
        step_count = 0
        episode_ne_start: Optional[float] = None
        episode_start_perf = time.perf_counter()
        printed_request_id: Optional[int] = None  # track to avoid re-printing same plan

        while not (terminated or truncated):
            if gym_env._ncn_enabled and gym_env._ncn_active:
                # NCN is a coordinator priority branch; AHS is suspended.
                action_id = 3
                action_name = "NCN"
            else:
                action, _ = scheduler.predict(obs, deterministic=True)
                action_id = int(action)
                action_name = ACTION_NAMES.get(action_id, "UNKNOWN")

            obs, _, terminated, truncated, _ = gym_env.step(action_id)
            step_count += 1

            # ── exec-step line ───────────────────────────────────────
            cur_control = gym_env.control_step
            waypoint_executed = cur_control > prev_control_step
            prev_control_step = cur_control

            cur_ne = gym_env._current_ne_m()
            cur_drift = gym_env._current_drift_m()
            cur_td = gym_env._current_time_drift_ms()
            if episode_ne_start is None and cur_ne is not None:
                episode_ne_start = cur_ne

            # peek last log record for tags + action latency
            tags: list = []
            action_latency = 0.0
            if gym_env.episode_records:
                last = gym_env.episode_records[-1]
                action_latency = last.get("airsim_action_latency_ms", 0.0)
                if last.get("request_dino_latency_ms") is not None:
                    tags.append("DINO")
                if last.get("safety_dino_triggered"):
                    tags.append("SAFETY")
                if last.get("action_illegal"):
                    override_to = last.get("action_override", "STOP_REQUEST")
                    tags.append(f"OVERRIDE->{override_to}")
                if last.get("predict_dones") and bool(last["predict_dones"][0]):
                    tags.append("DETECTED")
                if last.get("target_lock_active"):
                    tags.append("LOCK")
                if last.get("target_lock_completion_reason"):
                    tags.append(f"LOCK-{last['target_lock_completion_reason']}")
                if last.get("forced_refresh_after_target_lock"):
                    tags.append("REFRESH")

            tag_str = " | ".join(tags)
            tag_suffix = f"  [{tag_str}]" if tag_str else ""

            ne_str = f"{cur_ne:.1f}m" if cur_ne is not None else "---"
            drift_str = f"{cur_drift:.2f}m"
            td_str = f"{cur_td:.0f}ms"

            if waypoint_executed:
                logger.info(
                    f"[ep {completed:04d} exec_step={cur_control:03d}] "
                    f"act={action_latency:.0f}ms "
                    f"| NE={ne_str} | drift={drift_str} | td={td_str} "
                    f"| ppo={action_name}{tag_suffix}"
                )
            elif step_count == 1:
                # cold-start step — no waypoint executed
                logger.info(
                    f"[ep {completed:04d} init] "
                    f"NE={ne_str} | ppo=cold_start"
                )
            else:
                # STOP step — no waypoint, just hover/request
                logger.info(
                    f"[ep {completed:04d} step={step_count:03d}] "
                    f"| NE={ne_str} | drift={drift_str} | td={td_str} "
                    f"| ppo={action_name}{tag_suffix}"
                )

            # ── decision-step line (new trajectory applied) ──
            if (gym_env.active_result is not None
                    and int(getattr(gym_env.active_result, "request_id", -1)) != printed_request_id):
                result = gym_env.active_result
                printed_request_id = int(getattr(result, "request_id", -1))
                current_pose = gym_env.state.current_sim_pose() if gym_env.state else [0., 0., 0.]
                decision_step = printed_request_id
                total_latency = float(
                    getattr(result, "obs_latency_ms", 0)
                    + getattr(result, "groundingdino_latency_ms", 0)
                    + getattr(result, "uplink_latency_ms", 0)
                    + getattr(result, "llm_latency_ms", 0)
                    + getattr(result, "traj_latency_ms", 0)
                )
                logger.info(
                    f"[ep {completed:04d} decision_step={decision_step} timing] "
                    f"obs={getattr(result, 'obs_latency_ms', 0):.1f}ms "
                    f"dino={getattr(result, 'groundingdino_latency_ms', 0):.1f}ms "
                    f"bw={_fmt_optional_mbps(getattr(result, 'uplink_bandwidth_mbps', None))} "
                    f"uplink={_fmt_optional_ms(getattr(result, 'uplink_latency_ms', 0))} "
                    f"llm={_fmt_optional_ms(getattr(result, 'llm_latency_ms', 0))} "
                    f"traj={_fmt_optional_ms(getattr(result, 'traj_latency_ms', 0))} "
                    f"total={_fmt_optional_ms(total_latency)}"
                )
                llm_output = getattr(result, "llm_output", [])
                refined = getattr(result, "refined_waypoints", [])
                if len(llm_output) > 0 and len(refined) > 0:
                    _print_trajectory_bundle(completed, decision_step, llm_output, refined, current_pose=current_pose)

        # ── episode end summary ──
        episode_ms = (time.perf_counter() - episode_start_perf) * 1000.0
        outcome = "?"
        if gym_env.state is not None:
            if gym_env.state.success:
                outcome = "SR"
            elif gym_env.state.oracle_success:
                outcome = "OSR"
            elif gym_env.state.collisions[0]:
                outcome = "COL"
            elif gym_env.state.dones[0]:
                outcome = "DONE"
            else:
                outcome = "MAXWP"
        ne_start_str = f"init_NE={episode_ne_start:.1f}m " if episode_ne_start is not None else ""
        logger.info(
            f"[ep {completed:04d} END] {outcome} "
            f"steps={cur_control} "
            f"sched_steps={step_count} "
            f"{ne_start_str}"
            f"final_NE={ne_str} "
            f"drift={drift_str} "
            f"td={td_str} "
            f"duration={episode_ms/1000:.1f}s"
        )

        completed += 1
    pbar.close()

    gym_env.write_summary()
    gym_env.close()
    if ncn_runtime is not None:
        ncn_runtime.close()
    eval_env.delete_VectorEnvUtil()

    episode_records = [item for item in gym_env.summary_records if item.get("record_type") == "episode_end"]
    step_records = [item for item in gym_env.summary_records if item.get("record_type") == "scheduler_step"]
    success_count = sum(1 for item in episode_records if item.get("success"))
    oracle_success_count = sum(1 for item in episode_records if item.get("oracle_success"))
    collision_count = sum(1 for item in episode_records if item.get("collision"))
    action_counts = {name: sum(1 for item in step_records if item.get("action_name") == name) for name in ACTION_NAMES.values()}
    ncn_step_records = [item for item in step_records if item.get("control_source") == "ncn"]
    trigger_episodes = sum(1 for item in episode_records if item.get("tcm_corrected_applies", 0) > 0)
    final_summary = {
        "episodes": len(episode_records),
        "SR": success_count / len(episode_records) if episode_records else 0.0,
        "OSR": oracle_success_count / len(episode_records) if episode_records else 0.0,
        "CR": collision_count / len(episode_records) if episode_records else 0.0,
        "avg_waypoints": float(np.mean([item.get("control_steps", 0) for item in episode_records])) if episode_records else 0.0,
        "avg_NE_m": float(np.mean([item.get("final_ne_m", 0.0) for item in episode_records if item.get("final_ne_m") is not None])) if episode_records else 0.0,
        "avg_episode_latency_ms": _metric_summary([item.get("episode_latency_ms") for item in episode_records]),
        "avg_time_drift_ms": _metric_summary([item.get("time_drift_ms") for item in step_records]),
        "avg_state_drift_m": _metric_summary([item.get("state_drift_m") for item in step_records]),
        "avg_T_action_ms": _metric_summary([item.get("airsim_action_latency_ms") for item in step_records]),
        "avg_T_dec_ms": _metric_summary([item.get("decision_total_latency_ms") for item in step_records if item.get("decision_total_latency_ms") is not None]),
        "action_counts": action_counts,
        "forced_fallback_count": int(sum(1 for item in step_records if item.get("forced_fallback"))),
        "tcm_trigger_episodes": int(trigger_episodes),
        "tcm_trigger_rate": float(trigger_episodes / len(episode_records)) if episode_records else 0.0,
        "tcm_corrected_applies": int(sum(item.get("tcm_corrected_applies", 0) for item in episode_records)),
        "tcm_lock_completions": {
            reason: int(sum(item.get("tcm_lock_completions", {}).get(reason, 0) for item in episode_records))
            for reason in ("goal_reached", "goal_passed", "buffer_exhausted")
        },
        "ncn": {
            "enabled": bool(_as_bool(args.enable_ncn)),
            "actions": int(len(ncn_step_records)),
            "action_fraction": float(len(ncn_step_records) / len(step_records)) if step_records else 0.0,
            "takeovers": int(sum(item.get("ncn_takeovers", 0) for item in episode_records)),
            "stale_results": int(sum(item.get("ncn_stale_results", 0) for item in episode_records)),
            "recoveries": int(sum(item.get("ncn_recoveries", 0) for item in episode_records)),
            "model_inference_latency_ms": _metric_summary([item.get("model_inference_latency_ms") for item in ncn_step_records]),
            "recovery_wait_ms": _metric_summary([
                value for item in episode_records for value in item.get("ncn_recovery_wait_ms", [])
            ]),
        },
    }
    final_summary_path = os.path.join(profile_dir, f"drl_eval_metrics_{args.make_dir_time}.json")
    with open(final_summary_path, "w", encoding="utf-8") as handle:
        json.dump(final_summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    print(f"Saved MATCH eval metrics: {final_summary_path}")


if __name__ == "__main__":
    main()
