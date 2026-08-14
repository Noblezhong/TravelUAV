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
from typing import Any, Dict, Optional, Tuple

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
from src.vlnce_src.fast_eval_time import configure_fast_eval_output
from src.vlnce_src.match_tcm import TcmRuntime, TrajcorrMixin
from src.vlnce_src.rule_based_eval import _write_jsonl_line
from utils.logger import logger


class MatchProfileTravelModelWrapper(TrajcorrMixin, ProfileTravelModelWrapper):
    pass


class MatchEnv(DRLSchedulerEnv):
    def __init__(self, *pargs, **kwargs):
        super().__init__(*pargs, **kwargs)
        self._tcm_enabled = str(getattr(args, "trajcorr_mode", "on")).strip().lower() == "on"
        self._delta_cor_m = float(getattr(args, "trajcorr_state_shift_threshold_m", 2.5))
        self.tcm_runtime = TcmRuntime(self._tcm_enabled, self._delta_cor_m)
        self.target_lock = self.tcm_runtime.target_lock
        self._needs_target_lock_refresh = False
        self._tcm_step_extras: Dict[str, Any] = {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        self.tcm_runtime = TcmRuntime(self._tcm_enabled, self._delta_cor_m)
        self.target_lock = self.tcm_runtime.target_lock
        self._needs_target_lock_refresh = False
        self._tcm_step_extras = {}
        return super().reset(seed=seed, options=options)

    # ── TCM: result application (Algorithm 1 IF branch) ─────────────
    def _apply_result(self, result) -> None:
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

    # ── TCM: step() with lock / forced-refresh branches ─────────────
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
        f"trajcorr_mode={args.trajcorr_mode}, "
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
    eval_env.delete_VectorEnvUtil()

    episode_records = [item for item in gym_env.summary_records if item.get("record_type") == "episode_end"]
    step_records = [item for item in gym_env.summary_records if item.get("record_type") == "scheduler_step"]
    success_count = sum(1 for item in episode_records if item.get("success"))
    oracle_success_count = sum(1 for item in episode_records if item.get("oracle_success"))
    collision_count = sum(1 for item in episode_records if item.get("collision"))
    action_counts = {name: sum(1 for item in step_records if item.get("action_name") == name) for name in ACTION_NAMES.values()}
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
    }
    final_summary_path = os.path.join(profile_dir, f"drl_eval_metrics_{args.make_dir_time}.json")
    with open(final_summary_path, "w", encoding="utf-8") as handle:
        json.dump(final_summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    print(f"Saved MATCH eval metrics: {final_summary_path}")


if __name__ == "__main__":
    main()
