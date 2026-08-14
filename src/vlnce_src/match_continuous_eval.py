"""Continuous + TCM paradigm evaluator (2x2 ablation cell: Continuous w=5 + TCM-ON).

NEW FILE — the Continuous counterpart of the MATCH paradigm.  It reuses the
shared building blocks of ``continue_eval.py`` (state, planner, record
builders) and adds the TCM branch at the three result-application points plus
the target-lock execution.  ``continue_eval.py`` itself is NOT modified.

Fidelity contract:
  - ``--trajcorr_mode off``: behavior identical to ``continue_eval.py``
    (the audited Continuous row; must reproduce SR 25.7).
  - ``--trajcorr_mode on``: Continuous + TCM (C2 validation in the main
    environment).

record_type vocabulary is unchanged: {decision, execution, decision_stop,
episode_end}.  TCM fields are extra keys only (compute_metrics auto-detection
stays continuous).

Keep in sync with ``continue_eval.py``: if its loop changes, the loop below
must be re-checked.
"""

import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    ContinuousEpisodeState,
    LatestOnlyEdgePlanner,
    Snapshot,
    _as_bool,
    _build_planner_decision_record,
    _metric_summary,
    _point_delta,
    _print_decision_profile_line,
    _print_episode_header,
    _print_exec_profile_line,
    _print_trajectory_bundle,
    _run_decision_cycle,
    _set_console_log_message_only,
    _write_jsonl_line,
)
from src.vlnce_src.dino_monitor_online import DinoMonitor
from src.vlnce_src.fast_eval_time import (
    FastEvalClock,
    action_timing,
    configure_fast_eval_output,
)
from src.vlnce_src.match_tcm import TcmRuntime, TrajcorrMixin
from utils.logger import logger


class MatchContinuousTravelModelWrapper(TrajcorrMixin, ProfileTravelModelWrapper):
    pass


def _tcm_apply_result(tcm: TcmRuntime, state, eval_env, model_wrapper, clock, applied_result):
    """TCM-aware apply; returns (effective_result, tcm_extra) or (None, tcm_extra) on stale-drop."""
    return tcm.apply_result(
        state=state,
        eval_env=eval_env,
        model_wrapper=model_wrapper,
        clock=clock,
        result=applied_result,
        record_request_ne=True,
    )


def eval(
    model_wrapper,
    assist: Assist,
    eval_env,
    eval_save_dir,
    profile_log_path,
    summary_path,
    bandwidth_trace: BandwidthTrace,
    enable_comm_delay: bool,
    chunk_waypoints: int,
    trajcorr_enabled: bool,
    delta_cor_m: float,
):
    assert int(eval_env.batch_size) == 1, "continuous eval currently supports batchSize=1 only"
    model_wrapper.eval()
    summary_records = []

    with torch.no_grad():
        dataset = BatchIterator(eval_env)
        end_iter = len(dataset)
        pbar = tqdm.tqdm(total=end_iter)
        with open(profile_log_path, "w", encoding="utf-8") as profile_fp:
            episode_idx = 0
            while True:
                env_batchs = eval_env.next_minibatch()
                if env_batchs is None:
                    break
                pbar.update(n=1)
                _print_episode_header(episode_idx, env_batchs[0], chunk_waypoints)

                episode_ok = False
                for retry_i in range(3):
                    planner = None
                    try:
                        bandwidth_trace.reset_for_episode(env_batchs[0]["seq_name"])
                        state = ContinuousEpisodeState(
                            env_batchs[0],
                            eval_env,
                            assist,
                            ignore_tiny_diff=True,
                        )
                        episode_clock = FastEvalClock(bool(args.fast_eval), args.fast_eval_speedup)
                        tcm = TcmRuntime(trajcorr_enabled, delta_cor_m)
                        planner = LatestOnlyEdgePlanner(
                            model_wrapper,
                            bandwidth_trace,
                            enable_comm_delay,
                            clock=episode_clock,
                        )
                        request_counter = 0
                        control_step = 0
                        executed_since_decision = 0
                        active_traj: List[List[float]] = []
                        active_index = 0
                        pending_snapshot: Optional[Snapshot] = None

                        warmup_snapshot = state.build_snapshot(request_counter, control_step)
                        warmup_snapshot.submitted_logical_ms = episode_clock.now_ms if episode_clock.enabled else None
                        submitted = planner.submit(warmup_snapshot)
                        assert submitted, "failed to submit warmup snapshot"
                        warmup_hover_start = time.perf_counter()
                        warmup_logical_start = episode_clock.now_ms
                        warmup_result = planner.wait_result()
                        warmup_hover_ms = (
                            episode_clock.now_ms - warmup_logical_start
                            if episode_clock.enabled
                            else (time.perf_counter() - warmup_hover_start) * 1000.0
                        )
                        warmup_result, warmup_tcm_extra = _tcm_apply_result(
                            tcm, state, eval_env, model_wrapper, episode_clock, warmup_result
                        )
                        active_traj = copy.deepcopy(warmup_result.refined_waypoints)
                        active_index = 0
                        state.record_request_ne()
                        current_pose = state.current_sim_pose()
                        warmup_record = _build_planner_decision_record(
                            env_batchs[0],
                            warmup_result,
                            current_pose,
                            enable_comm_delay,
                            chunk_waypoints,
                            decision_step=0,
                            clock=episode_clock,
                            applied_exec_step=control_step,
                        )
                        warmup_record["hover_wait_ms"] = float(warmup_hover_ms)
                        warmup_record["loop_latency_ms"] = float(warmup_hover_ms)
                        warmup_record["predict_dones"] = [bool(x) for x in state.predict_dones]
                        warmup_record["collisions"] = [bool(x) for x in state.collisions]
                        warmup_record["dones"] = [bool(x) for x in state.dones]
                        warmup_record.update(warmup_tcm_extra)
                        _write_jsonl_line(profile_fp, warmup_record)
                        summary_records.append(warmup_record)
                        _print_decision_profile_line(episode_idx, warmup_result.request_id, warmup_record)
                        _print_trajectory_bundle(
                            episode_idx,
                            warmup_result.request_id,
                            warmup_result.llm_output,
                            warmup_result.refined_waypoints,
                            current_pose=current_pose,
                        )

                        while not state.dones[0] and control_step < int(args.max_control_steps):
                            applied_result = planner.poll_result()
                            dino_latency_ms = 0.0
                            trajectory_switch_applied = False
                            hover_wait_ms = 0.0
                            dino_predicted_this_step = False

                            if applied_result is not None:
                                applied_result, tcm_extra = _tcm_apply_result(
                                    tcm, state, eval_env, model_wrapper, episode_clock, applied_result
                                )
                                if applied_result is None:
                                    # stale result dropped during target lock
                                    dropped_record = {
                                        "record_type": "execution",
                                        "exec_step": int(control_step),
                                        "batch_size": 1,
                                        "seq_names": [env_batchs[0]["seq_name"]],
                                        "map_names": [env_batchs[0]["map_name"]],
                                        "chunk_waypoints": int(chunk_waypoints),
                                        "executed_waypoints": 0,
                                        "comm_delay_enabled": bool(enable_comm_delay),
                                        "request_id": None,
                                        "request_submitted_step": None,
                                        "request_observation_timestamp": None,
                                        "request_ready_wall_time": None,
                                        "llm_latency_ms": None,
                                        "traj_latency_ms": None,
                                        "airsim_action_latency_ms": 0.0,
                                        "action_wall_time_ms": 0.0,
                                        "action_sim_time_ms": 0.0,
                                        "obs_latency_ms": 0.0,
                                        "groundingdino_latency_ms": 0.0,
                                        "execution_post_obs_latency_ms": 0.0,
                                        "execution_post_dino_latency_ms": 0.0,
                                        "loop_latency_ms": 0.0,
                                        "decision_latency_ms": None,
                                        "uplink_payload_bits": None,
                                        "uplink_payload_mb": None,
                                        "uplink_bandwidth_mbps": None,
                                        "uplink_latency_ms": None,
                                        "action_age_ms": None,
                                        "state_drift_m": None,
                                        "trajectory_switch_applied": False,
                                        "hover_wait_ms": 0.0,
                                        "llm_output": None,
                                        "refined_waypoints": copy.deepcopy(active_traj),
                                        "current_pose": state.current_sim_pose(),
                                        "first_refined_gap_m": float(_point_delta(active_traj[0], state.current_sim_pose())) if len(active_traj) > 0 else None,
                                        "exec_target_distance_m": None,
                                        "predict_dones": [bool(x) for x in state.predict_dones],
                                        "collisions": [bool(x) for x in state.collisions],
                                        "dones": [bool(x) for x in state.dones],
                                    }
                                    dropped_record.update(tcm_extra)
                                    dropped_record.update(episode_clock.metadata())
                                    _write_jsonl_line(profile_fp, dropped_record)
                                    summary_records.append(dropped_record)
                                else:
                                    active_traj = copy.deepcopy(applied_result.refined_waypoints)
                                    active_index = 0
                                    trajectory_switch_applied = True
                                    current_pose = state.current_sim_pose()
                                    if pending_snapshot is not None and not planner.has_inflight() and not tcm.lock_active:
                                        planner.submit(pending_snapshot)
                                        pending_snapshot = None
                                    decision_record = _build_planner_decision_record(
                                        env_batchs[0],
                                        applied_result,
                                        current_pose,
                                        enable_comm_delay,
                                        chunk_waypoints,
                                        decision_step=int(applied_result.request_id),
                                        clock=episode_clock,
                                        applied_exec_step=control_step,
                                    )
                                    decision_record["predict_dones"] = [bool(x) for x in state.predict_dones]
                                    decision_record["collisions"] = [bool(x) for x in state.collisions]
                                    decision_record["dones"] = [bool(x) for x in state.dones]
                                    decision_record.update(tcm_extra)
                                    if tcm_extra.get("trajectory_mode") == "corrected":
                                        # correction time is non-moving local compute at apply —
                                        # fold it into the approved wait so the w/ TC cell's
                                        # wait metric is protocol-comparable with w/o TC
                                        decision_record["hover_wait_ms"] = (
                                            float(decision_record.get("hover_wait_ms", 0.0))
                                            + float(tcm_extra.get("trajcorr_refresh_obs_ms", 0) or 0)
                                            + float(tcm_extra.get("trajcorr_regen_ms", 0) or 0)
                                        )
                                    _write_jsonl_line(profile_fp, decision_record)
                                    summary_records.append(decision_record)
                                    _print_decision_profile_line(episode_idx, applied_result.request_id, decision_record)
                                    _print_trajectory_bundle(
                                        episode_idx,
                                        applied_result.request_id,
                                        applied_result.llm_output,
                                        applied_result.refined_waypoints,
                                        current_pose=current_pose,
                                    )

                            if active_index < len(active_traj):
                                current_chunk = active_traj[active_index : active_index + 1]
                                chunk_size = len(current_chunk)
                                exec_pose = state.current_sim_pose()
                                exec_target_distance_m = (
                                    float(_point_delta(current_chunk[0], exec_pose)) if len(current_chunk) > 0 else None
                                )
                                action_start = time.perf_counter()
                                eval_env.makeActionsChunk([current_chunk], target_idx=chunk_size)
                                action_end = time.perf_counter()
                                action_times = action_timing(
                                    eval_env,
                                    (action_end - action_start) * 1000.0,
                                    episode_clock.enabled,
                                )
                                episode_clock.advance_action(
                                    action_times["action_sim_time_ms"],
                                    action_times["action_wall_time_ms"],
                                )

                                active_index += chunk_size
                                control_step += 1
                                executed_since_decision += chunk_size
                                state.sync_runtime_status_from_sim()

                                lock_completion = None
                                if tcm.lock_active:
                                    lock_completion = tcm.target_lock.evaluate(state.current_sim_pose())
                                    if lock_completion is None and active_index >= len(active_traj):
                                        lock_completion = tcm.target_lock.mark_buffer_exhausted()

                                decision_obs_latency_ms = 0.0
                                decision_observation_pose = None
                                should_request_decision = (
                                    not state.dones[0]
                                    and not tcm.lock_active
                                    and (
                                        active_index >= len(active_traj)
                                        or executed_since_decision >= chunk_waypoints
                                    )
                                )
                                if should_request_decision:
                                    request_counter, pending_snapshot, decision_obs_latency_ms, dino_latency_ms, dino_predicted_this_step = _run_decision_cycle(
                                        state,
                                        eval_env,
                                        model_wrapper,
                                        planner,
                                        request_counter,
                                        control_step,
                                        episode_clock,
                                    )
                                    decision_observation_pose = state.current_sim_pose()
                                    executed_since_decision = 0

                                action_latency_ms = float(action_times["airsim_action_latency_ms"])
                                record = {
                                    "record_type": "execution",
                                    "exec_step": int(control_step),
                                    "batch_size": 1,
                                    "seq_names": [env_batchs[0]["seq_name"]],
                                    "map_names": [env_batchs[0]["map_name"]],
                                    "chunk_waypoints": int(chunk_waypoints),
                                    "executed_waypoints": int(chunk_size),
                                    "comm_delay_enabled": bool(enable_comm_delay),
                                    "request_id": None,
                                    "request_submitted_step": None,
                                    "request_observation_timestamp": None,
                                    "request_ready_wall_time": None,
                                    "llm_latency_ms": None,
                                    "traj_latency_ms": None,
                                    "airsim_action_latency_ms": action_latency_ms,
                                    "action_wall_time_ms": float(action_times["action_wall_time_ms"]),
                                    "action_sim_time_ms": float(action_times["action_sim_time_ms"]),
                                    "obs_latency_ms": 0.0,
                                    "groundingdino_latency_ms": 0.0,
                                    "execution_post_obs_latency_ms": decision_obs_latency_ms,
                                    "execution_post_dino_latency_ms": float(dino_latency_ms),
                                    "loop_latency_ms": action_latency_ms,
                                    "decision_latency_ms": None,
                                    "uplink_payload_bits": None,
                                    "uplink_payload_mb": None,
                                    "uplink_bandwidth_mbps": None,
                                    "uplink_latency_ms": None,
                                    "action_age_ms": None,
                                    "state_drift_m": None,
                                    "trajectory_switch_applied": bool(trajectory_switch_applied),
                                    "hover_wait_ms": float(hover_wait_ms),
                                    "llm_output": None,
                                    "refined_waypoints": copy.deepcopy(active_traj),
                                    "current_pose": state.current_sim_pose(),
                                    "first_refined_gap_m": float(_point_delta(active_traj[0], state.current_sim_pose())) if len(active_traj) > 0 else None,
                                    "exec_target_distance_m": exec_target_distance_m,
                                    "predict_dones": [bool(x) for x in state.predict_dones],
                                    "collisions": [bool(x) for x in state.collisions],
                                    "dones": [bool(x) for x in state.dones],
                                }
                                if tcm.lock_active:
                                    record["target_lock_active"] = True
                                if lock_completion is not None:
                                    record["target_lock_completion_reason"] = lock_completion
                                record.update(episode_clock.metadata())
                                _write_jsonl_line(profile_fp, record)
                                summary_records.append(record)
                                _print_exec_profile_line(
                                    episode_idx,
                                    control_step,
                                    record,
                                    current_chunk=current_chunk,
                                    current_pose=exec_pose,
                                )
                                if lock_completion is not None:
                                    tcm.stats.lock_completions[lock_completion] += 1
                                    active_traj = []
                                    active_index = 0
                                    executed_since_decision = 0
                                if state.dones[0] and dino_predicted_this_step and not state.collisions[0]:
                                    stop_pose = state.current_sim_pose()
                                    stop_record = {
                                        "record_type": "decision_stop",
                                        "decision_step": int(control_step),
                                        "batch_size": 1,
                                        "seq_names": [env_batchs[0]["seq_name"]],
                                        "map_names": [env_batchs[0]["map_name"]],
                                        "chunk_waypoints": int(chunk_waypoints),
                                        "executed_waypoints": int(chunk_size),
                                        "comm_delay_enabled": bool(enable_comm_delay),
                                        "request_id": None,
                                        "submitted_after_exec_step": int(control_step),
                                        "obs_latency_ms": decision_obs_latency_ms,
                                        "groundingdino_latency_ms": float(dino_latency_ms),
                                        "uplink_latency_ms": None,
                                        "llm_latency_ms": None,
                                        "traj_latency_ms": None,
                                        "decision_latency_ms": float(decision_obs_latency_ms + dino_latency_ms),
                                        "airsim_action_latency_ms": 0.0,
                                        "loop_latency_ms": float(decision_obs_latency_ms + dino_latency_ms),
                                        "action_age_ms": None,
                                        "state_drift_m": None,
                                        "trajectory_switch_applied": False,
                                        "hover_wait_ms": 0.0,
                                        "current_pose": copy.deepcopy(stop_pose),
                                        "predict_done": True,
                                        "stop_delay_ms": float(dino_latency_ms),
                                        "stop_drift_m": float(_point_delta(decision_observation_pose, stop_pose)),
                                        "predict_dones": [bool(x) for x in state.predict_dones],
                                        "collisions": [bool(x) for x in state.collisions],
                                        "dones": [bool(x) for x in state.dones],
                                    }
                                    _write_jsonl_line(profile_fp, stop_record)
                                    summary_records.append(stop_record)
                                    _print_decision_profile_line(episode_idx, control_step, stop_record)
                            else:
                                if tcm.lock_active:
                                    # defensive: lock with an empty buffer ends the lock
                                    completion = tcm.target_lock.mark_buffer_exhausted()
                                    tcm.stats.lock_completions[completion] += 1
                                    active_traj = []
                                    active_index = 0
                                    executed_since_decision = 0
                                    continue
                                if not state.dones[0]:
                                    request_counter, pending_snapshot, _, _, _ = _run_decision_cycle(
                                        state,
                                        eval_env,
                                        model_wrapper,
                                        planner,
                                        request_counter,
                                        control_step,
                                        episode_clock,
                                    )
                                    executed_since_decision = 0

                                if state.dones[0]:
                                    state.maybe_finalize()
                                    break

                                if pending_snapshot is not None and not planner.has_inflight():
                                    planner.submit(pending_snapshot)
                                    pending_snapshot = None

                                hover_wait_start = time.perf_counter()
                                hover_logical_start = episode_clock.now_ms
                                applied_result = planner.wait_result()
                                hover_wait_ms = (
                                    episode_clock.now_ms - hover_logical_start
                                    if episode_clock.enabled
                                    else (time.perf_counter() - hover_wait_start) * 1000.0
                                )
                                applied_result, tcm_extra = _tcm_apply_result(
                                    tcm, state, eval_env, model_wrapper, episode_clock, applied_result
                                )
                                if applied_result is None:
                                    # stale-drop in the hover branch: loop again (next poll applies fresh)
                                    continue
                                active_traj = copy.deepcopy(applied_result.refined_waypoints)
                                active_index = 0
                                current_pose = state.current_sim_pose()
                                record = _build_planner_decision_record(
                                    env_batchs[0],
                                    applied_result,
                                    current_pose,
                                    enable_comm_delay,
                                    chunk_waypoints,
                                    decision_step=int(applied_result.request_id),
                                    clock=episode_clock,
                                    applied_exec_step=control_step,
                                )
                                record["hover_wait_ms"] = float(hover_wait_ms)
                                record["loop_latency_ms"] = float(hover_wait_ms)
                                record["predict_dones"] = [bool(x) for x in state.predict_dones]
                                record["collisions"] = [bool(x) for x in state.collisions]
                                record["dones"] = [bool(x) for x in state.dones]
                                record.update(tcm_extra)
                                if tcm_extra.get("trajectory_mode") == "corrected":
                                    record["hover_wait_ms"] = (
                                        float(record.get("hover_wait_ms", 0.0))
                                        + float(tcm_extra.get("trajcorr_refresh_obs_ms", 0) or 0)
                                        + float(tcm_extra.get("trajcorr_regen_ms", 0) or 0)
                                    )
                                _write_jsonl_line(profile_fp, record)
                                summary_records.append(record)
                                _print_decision_profile_line(episode_idx, applied_result.request_id, record)
                                _print_trajectory_bundle(
                                    episode_idx,
                                    applied_result.request_id,
                                    applied_result.llm_output,
                                    applied_result.refined_waypoints,
                                    current_pose=current_pose,
                                )

                            if state.dones[0]:
                                state.maybe_finalize()
                                break

                        if not state.skip_saved:
                            state.maybe_finalize(force=True)
                        episode_record = {
                            "record_type": "episode_end",
                            "seq_names": [env_batchs[0]["seq_name"]],
                            "map_names": [env_batchs[0]["map_name"]],
                            "control_steps": int(control_step),
                            "episode_latency_ms": float(episode_clock.now_ms),
                            "success": bool(state.success),
                            "oracle_success": bool(state.oracle_success),
                            "collision": bool(state.collisions[0]),
                            "final_ne_m": float(state.distance_to_ends[-1]) if state.distance_to_ends else None,
                        }
                        episode_record.update(tcm.stats.as_dict())
                        episode_record.update(episode_clock.metadata())
                        _write_jsonl_line(profile_fp, episode_record)
                        summary_records.append(episode_record)
                        episode_ok = True
                        break
                    except Exception as e:
                        logger.error(f"Episode failed: {e}, retry {retry_i + 1}/3")
                        if retry_i < 2:
                            logger.error("Restarting scene...")
                            eval_env._changeEnv(need_change=True)
                    finally:
                        if planner is not None:
                            planner.close()

                if not episode_ok:
                    raise RuntimeError("episode failed after 3 retries")
                episode_idx += 1

        try:
            pbar.close()
        except Exception:
            pass
    execution_records = [item for item in summary_records if item.get("record_type") == "execution"]
    decision_records = [item for item in summary_records if item.get("record_type") in ("decision", "decision_stop")]
    episode_records = [item for item in summary_records if item.get("record_type") == "episode_end"]
    trigger_episodes = sum(1 for item in episode_records if item.get("tcm_corrected_applies", 0) > 0)
    summary = {
        "execution_airsim_action_latency_ms": _metric_summary([item.get("airsim_action_latency_ms") for item in execution_records]),
        "execution_loop_latency_ms": _metric_summary([item.get("loop_latency_ms") for item in execution_records]),
        "execution_target_distance_m": _metric_summary([item.get("exec_target_distance_m") for item in execution_records]),
        "decision_obs_latency_ms": _metric_summary([item.get("obs_latency_ms") for item in decision_records]),
        "decision_groundingdino_latency_ms": _metric_summary([item.get("groundingdino_latency_ms") for item in decision_records]),
        "decision_uplink_latency_ms": _metric_summary([item.get("uplink_latency_ms") for item in decision_records]),
        "decision_llm_latency_ms": _metric_summary([item.get("llm_latency_ms") for item in decision_records]),
        "decision_traj_latency_ms": _metric_summary([item.get("traj_latency_ms") for item in decision_records]),
        "decision_total_latency_ms": _metric_summary([item.get("decision_latency_ms") for item in decision_records]),
        "decision_uplink_payload_bits": _metric_summary([item.get("uplink_payload_bits") for item in decision_records]),
        "decision_uplink_payload_mb": _metric_summary([item.get("uplink_payload_mb") for item in decision_records]),
        "decision_uplink_bandwidth_mbps": _metric_summary([item.get("uplink_bandwidth_mbps") for item in decision_records]),
        "action_age_ms": _metric_summary([item.get("action_age_ms") for item in decision_records]),
        "state_drift_m": _metric_summary([item.get("state_drift_m") for item in decision_records]),
        "stop_delay_ms": _metric_summary([item.get("stop_delay_ms") for item in decision_records]),
        "stop_drift_m": _metric_summary([item.get("stop_drift_m") for item in decision_records]),
        "hover_wait_ms": _metric_summary([item.get("hover_wait_ms") for item in decision_records]),
        "episode_latency_ms": _metric_summary([item.get("episode_latency_ms") for item in episode_records]),
        "num_execution_records": len(execution_records),
        "num_decision_records": len(decision_records),
        "num_records": len(summary_records),
        "tcm_trigger_episodes": int(trigger_episodes),
        "tcm_trigger_rate": float(trigger_episodes / len(episode_records)) if episode_records else 0.0,
        "tcm_corrected_applies": int(sum(item.get("tcm_corrected_applies", 0) for item in episode_records)),
        "tcm_lock_completions": {
            reason: int(sum(item.get("tcm_lock_completions", {}).get(reason, 0) for item in episode_records))
            for reason in ("goal_reached", "goal_passed", "buffer_exhausted")
        },
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    _set_console_log_message_only()
    configure_fast_eval_output(args, "continuous")
    eval_save_path = args.eval_save_path
    eval_json_path = args.eval_json_path
    dataset_path = args.dataset_path

    if not os.path.exists(eval_save_path):
        os.makedirs(eval_save_path)
    profile_log_dir = os.path.join(eval_save_path, "profile_logs")
    os.makedirs(profile_log_dir, exist_ok=True)

    enable_comm_delay = _as_bool(args.enable_comm_delay)
    trace_path = Path(args.comm_trace_csv_path) if args.comm_trace_csv_path else Path(default_trace_path())
    if not trace_path.exists():
        raise FileNotFoundError(f"bandwidth trace not found: {trace_path}")
    chunk_waypoints = max(1, int(args.chunk_waypoints))
    trajcorr_enabled = str(getattr(args, "trajcorr_mode", "on")).strip().lower() == "on"
    delta_cor_m = float(getattr(args, "trajcorr_state_shift_threshold_m", 2.5))
    bandwidth_trace = BandwidthTrace(trace_path, cycle=True)
    logger.info(
        f"Loaded bandwidth trace: {trace_path} ({bandwidth_trace.sample_count} samples), "
        f"enable_comm_delay={enable_comm_delay}, chunk_waypoints={chunk_waypoints}, "
        f"trajcorr_mode={args.trajcorr_mode}, trajcorr_state_shift_threshold_m={delta_cor_m}"
    )

    setup()

    assert CheckPort(), "error port"

    eval_env = initialize_env_eval(dataset_path=dataset_path, save_path=eval_save_path, eval_json_path=eval_json_path)

    if is_dist_avail_and_initialized():
        torch.distributed.destroy_process_group()

    args.DistributedDataParallel = False

    model_wrapper = MatchContinuousTravelModelWrapper(model_args=model_args, data_args=data_args)
    model_wrapper.dino_moinitor = DinoMonitor.get_instance()

    assist = Assist(always_help=args.always_help, use_gt=args.use_gt)

    print("Assist setting: always_help --", args.always_help, "    use_gt --", args.use_gt)

    profile_log_path = os.path.join(profile_log_dir, f"pro_con_w{chunk_waypoints}_{args.make_dir_time}.jsonl")
    summary_path = os.path.join(profile_log_dir, f"pro_con_w{chunk_waypoints}_{args.make_dir_time}_summary.json")

    eval(
        model_wrapper=model_wrapper,
        assist=assist,
        eval_env=eval_env,
        eval_save_dir=eval_save_path,
        profile_log_path=profile_log_path,
        summary_path=summary_path,
        bandwidth_trace=bandwidth_trace,
        enable_comm_delay=enable_comm_delay,
        chunk_waypoints=chunk_waypoints,
        trajcorr_enabled=trajcorr_enabled,
        delta_cor_m=delta_cor_m,
    )
    eval_env.delete_VectorEnvUtil()
