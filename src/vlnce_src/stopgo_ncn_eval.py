import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import tqdm

sys.path.append(str(Path(str(os.getcwd())).resolve()))

from assist import Assist
from env_uav import AirVLNENV
from src.common.param import args, model_args, data_args
from src.model_wrapper.profile_travel_llm import ProfileTravelModelWrapper
from src.vlnce_src.closeloop_util import (
    BatchIterator,
    CheckPort,
    EvalBatchState,
    initialize_env_eval,
    is_dist_avail_and_initialized,
    setup,
)
from src.vlnce_src.comm_delay import (
    BandwidthTrace,
    calculate_latency_ms,
    default_trace_path,
    estimate_uplink_payload_bits_from_outputs,
)
from src.vlnce_src.dino_monitor_online import DinoMonitor
from src.vlnce_src.fast_eval_time import (
    FastEvalClock,
    action_timing,
    configure_fast_eval_output,
)
from src.vlnce_src.eval_contract import execute_stop_waypoint_chunks
from src.vlnce_src.continue_ncn_eval import (
    ContinuousEpisodeState,
    LatestOnlyEdgePlanner,
    _as_bool,
    _build_planner_decision_record,
    _run_ncn_action,
)
from src.vlnce_src.ncn_runtime import EdgeLatencyEstimate, NCNRuntime
from utils.logger import logger
from utils.utils import *


def _metric_summary(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def _write_jsonl_line(handle, payload):
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def _print_profile_line(step_idx, profile_info):
    message = (
        f"[eval] step={step_idx} "
        f"llm={profile_info['llm_latency_ms']:.3f}ms "
        f"traj={profile_info['traj_latency_ms']:.3f}ms "
        f"action={profile_info['airsim_action_latency_ms']:.3f}ms "
        f"obs={profile_info['obs_latency_ms']:.3f}ms "
        f"dino={profile_info['groundingdino_latency_ms']:.3f}ms "
        f"uplink={profile_info['uplink_latency_ms']:.3f}ms "
        f"bw={profile_info['uplink_bandwidth_mbps']:.3f}Mbps "
        f"payload={profile_info['uplink_payload_mb']:.3f}MB "
        f"llm_output={profile_info['llm_output']}"
    )
    logger.info(message)


def eval(
    model_wrapper: ProfileTravelModelWrapper,
    assist: Assist,
    eval_env: AirVLNENV,
    eval_save_dir,
    profile_log_path,
    summary_path,
    bandwidth_trace: BandwidthTrace,
    enable_comm_delay: bool,
):
    model_wrapper.eval()

    summary_records = []
    episode_records = []
    episode_latencies_ms = []
    fast_eval = bool(args.fast_eval)
    with torch.no_grad():
        dataset = BatchIterator(eval_env)
        end_iter = len(dataset)
        pbar = tqdm.tqdm(total=end_iter)
        with open(profile_log_path, "w", encoding="utf-8") as profile_fp:
            while True:
                env_batchs = eval_env.next_minibatch()
                if env_batchs is None:
                    break

                episode_ok = False
                for retry_i in range(3):
                    try:
                        if len(env_batchs) != 1:
                            raise RuntimeError(
                                f"time-anchored bandwidth requires single-episode batches, got {len(env_batchs)}"
                            )
                        for env_batch in env_batchs:
                            bandwidth_trace.reset_for_episode(env_batch["seq_name"])
                        batch_state = EvalBatchState(
                            batch_size=eval_env.batch_size,
                            env_batchs=env_batchs,
                            env=eval_env,
                            assist=assist,
                            ignore_tiny_diff=True,
                        )
                        episode_clock = FastEvalClock(fast_eval, args.fast_eval_speedup)
                        bw_t0_ms = float(episode_clock.now_ms)
                        completed_decisions = 0
                        executed_control_steps = 0
                        pbar.update(n=eval_env.batch_size)
                        assist_notices = None

                        for t in range(int(args.maxWaypoints) + 1):
                            logger.info("Step: {} \t Completed: {} / {}".format(t, int(eval_env.index_data) - int(eval_env.batch_size), end_iter))

                            is_terminate = batch_state.check_batch_termination(t)
                            if is_terminate:
                                break

                            seq_names = [item["seq_name"] for item in env_batchs]
                            map_names = [item["map_name"] for item in env_batchs]

                            step_start = time.perf_counter()
                            step_logical_start_ms = episode_clock.now_ms
                            inputs, rot_to_targets = model_wrapper.prepare_inputs(
                                batch_state.episodes,
                                batch_state.target_positions,
                                assist_notices,
                            )
                            refined_waypoints, model_profile = model_wrapper.run_profiled(
                                inputs=inputs,
                                episodes=batch_state.episodes,
                                rot_to_targets=rot_to_targets,
                            )
                            episode_clock.advance_blocking(
                                float(model_profile["llm_latency_ms"]) + float(model_profile["traj_latency_ms"])
                            )

                            action_start = time.perf_counter()
                            _, executed_this_decision = execute_stop_waypoint_chunks(
                                eval_env,
                                refined_waypoints,
                            )
                            action_end = time.perf_counter()
                            action_times = action_timing(
                                eval_env,
                                (action_end - action_start) * 1000.0,
                                fast_eval,
                            )
                            episode_clock.advance_action(
                                action_times["action_sim_time_ms"],
                                action_times["action_wall_time_ms"],
                            )
                            completed_decisions += 1
                            executed_control_steps += int(executed_this_decision)

                            obs_start = time.perf_counter()
                            outputs = eval_env.get_obs()
                            obs_end = time.perf_counter()
                            obs_latency_ms = float((obs_end - obs_start) * 1000.0)
                            episode_clock.advance_blocking(obs_latency_ms)

                            payload_bytes, payload_bits, payload_mb = estimate_uplink_payload_bits_from_outputs(outputs)
                            bandwidth_bps = bandwidth_trace.bandwidth_at_ms(float(episode_clock.now_ms) - bw_t0_ms)
                            uplink_latency_ms = calculate_latency_ms(payload_bits, bandwidth_bps) if enable_comm_delay else 0.0
                            loop_without_comm_ms = float((time.perf_counter() - step_start) * 1000.0)
                            if uplink_latency_ms > 0 and not fast_eval:
                                time.sleep(uplink_latency_ms / 1000.0)
                            if fast_eval:
                                episode_clock.advance_blocking(uplink_latency_ms)

                            batch_state.update_from_env_output(outputs)
                            dino_start = time.perf_counter()
                            batch_state.predict_dones = model_wrapper.predict_done(batch_state.episodes, batch_state.object_infos)
                            dino_end = time.perf_counter()
                            dino_latency_ms = float((dino_end - dino_start) * 1000.0)
                            episode_clock.advance_blocking(dino_latency_ms)
                            batch_state.update_metric()
                            assist_notices = batch_state.get_assist_notices()

                            loop_end = time.perf_counter()
                            loop_with_comm_ms = (
                                float(episode_clock.now_ms - step_logical_start_ms)
                                if fast_eval
                                else float((loop_end - step_start) * 1000.0)
                            )

                            profile_info = {
                                "episode_step": int(t),
                                "decision_steps": int(completed_decisions),
                                "executed_waypoints": int(executed_control_steps),
                                "batch_size": int(eval_env.batch_size),
                                "seq_names": seq_names,
                                "map_names": map_names,
                                "llm_latency_ms": float(model_profile["llm_latency_ms"]),
                                "traj_latency_ms": float(model_profile["traj_latency_ms"]),
                                "airsim_action_latency_ms": float(action_times["airsim_action_latency_ms"]),
                                "action_wall_time_ms": float(action_times["action_wall_time_ms"]),
                                "action_sim_time_ms": float(action_times["action_sim_time_ms"]),
                                "obs_latency_ms": obs_latency_ms,
                                "groundingdino_latency_ms": dino_latency_ms,
                                "loop_latency_ms": loop_without_comm_ms,
                                "uplink_payload_bits": int(payload_bits),
                                "uplink_payload_mb": float(payload_mb),
                                "uplink_bandwidth_mbps": float(bandwidth_bps / 1_000_000.0),
                                "uplink_latency_ms": float(uplink_latency_ms),
                                "loop_latency_with_comm_ms": loop_with_comm_ms,
                                "decision_latency_ms": float(
                                    obs_latency_ms
                                    + dino_latency_ms
                                    + uplink_latency_ms
                                    + model_profile["llm_latency_ms"]
                                    + model_profile["traj_latency_ms"]
                                ),
                                "llm_output": model_profile["llm_output"],
                                "predict_dones": [bool(x) for x in batch_state.predict_dones],
                                "collisions": [bool(x) for x in batch_state.collisions],
                                "dones": [bool(x) for x in batch_state.dones],
                            }
                            profile_info.update(episode_clock.metadata())
                            _write_jsonl_line(profile_fp, profile_info)
                            summary_records.append(profile_info)
                            _print_profile_line(t, profile_info)
                        for batch_idx, env_batch in enumerate(env_batchs):
                            episode_record = {
                                "record_type": "episode_end",
                                "seq_names": [env_batch["seq_name"]],
                                "map_names": [env_batch["map_name"]],
                                "decision_steps": int(completed_decisions),
                                "control_steps": int(executed_control_steps),
                                "executed_waypoints": int(executed_control_steps),
                                "episode_latency_ms": float(episode_clock.now_ms),
                                "success": bool(batch_state.success[batch_idx]),
                                "oracle_success": bool(batch_state.oracle_success[batch_idx]),
                                "collision": bool(batch_state.collisions[batch_idx]),
                                "final_ne_m": float(batch_state.distance_to_ends[batch_idx][-1]),
                            }
                            episode_record.update(episode_clock.metadata())
                            _write_jsonl_line(profile_fp, episode_record)
                            episode_records.append(episode_record)
                        episode_ok = True
                        episode_latencies_ms.append(float(episode_clock.now_ms))
                        break
                    except Exception as e:
                        logger.error(f"Episode failed: {e}, retry {retry_i + 1}/3")
                        if retry_i < 2:
                            logger.error("Restarting scene...")
                            eval_env._changeEnv(need_change=True)

                if not episode_ok:
                    raise RuntimeError("episode failed after 3 retries")

        try:
            pbar.close()
        except Exception:
            pass

    summary = {
        "llm_latency_ms": _metric_summary([item["llm_latency_ms"] for item in summary_records]),
        "traj_latency_ms": _metric_summary([item["traj_latency_ms"] for item in summary_records]),
        "airsim_action_latency_ms": _metric_summary([item["airsim_action_latency_ms"] for item in summary_records]),
        "obs_latency_ms": _metric_summary([item["obs_latency_ms"] for item in summary_records]),
        "groundingdino_latency_ms": _metric_summary([item["groundingdino_latency_ms"] for item in summary_records]),
        "loop_latency_ms": _metric_summary([item["loop_latency_ms"] for item in summary_records]),
        "uplink_payload_bits": _metric_summary([item["uplink_payload_bits"] for item in summary_records]),
        "uplink_payload_mb": _metric_summary([item["uplink_payload_mb"] for item in summary_records]),
        "uplink_bandwidth_mbps": _metric_summary([item["uplink_bandwidth_mbps"] for item in summary_records]),
        "uplink_latency_ms": _metric_summary([item["uplink_latency_ms"] for item in summary_records]),
        "loop_latency_with_comm_ms": _metric_summary([item["loop_latency_with_comm_ms"] for item in summary_records]),
        "decision_latency_ms": _metric_summary([item["decision_latency_ms"] for item in summary_records]),
        "episode_latency_ms": _metric_summary(episode_latencies_ms),
        "decision_steps": _metric_summary(
            [item["decision_steps"] for item in episode_records]
        ),
        "executed_waypoints": _metric_summary(
            [item["executed_waypoints"] for item in episode_records]
        ),
        "num_records": len(summary_records),
        "comm_delay_enabled": bool(enable_comm_delay),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def eval_stopgo_ncn(model_wrapper, assist, eval_env, profile_log_path, summary_path, bandwidth_trace, enable_comm_delay):
    """Stop-go edge navigation with immediate local NCN fallback.

    A returned edge trajectory is deliberately applied without TCM: this file
    is the NCN-only ablation, not the complete MATCH evaluator.
    """
    model_wrapper.eval()
    runtime = NCNRuntime(model_args)
    records = []
    with torch.no_grad(), open(profile_log_path, "w", encoding="utf-8") as fp:
        for episode_idx in range(len(BatchIterator(eval_env))):
            env_batchs = eval_env.next_minibatch()
            if env_batchs is None:
                break
            env_batch = env_batchs[0]
            bandwidth_trace.reset_for_episode(env_batch["seq_name"])
            state = ContinuousEpisodeState(env_batch, eval_env, assist, ignore_tiny_diff=True)
            clock = FastEvalClock(bool(args.fast_eval), args.fast_eval_speedup)
            planner = LatestOnlyEdgePlanner(model_wrapper, bandwidth_trace, enable_comm_delay, clock)
            control_step = 0
            request_id = 0
            ready_result = None
            edge_latency = EdgeLatencyEstimate(0.0, float(args.ncn_response_ema_alpha))
            try:
                snapshot = state.build_snapshot(request_id, control_step)
                snapshot.submitted_logical_ms = clock.now_ms if clock.enabled else None
                if not planner.submit(snapshot):
                    raise RuntimeError("failed to submit initial Stop-go+NCN request")
                while not state.dones[0] and control_step < int(args.max_control_steps):
                    result = ready_result or planner.poll_result()
                    ready_result = None
                    if result is not None:
                        edge_latency.update(result.llm_latency_ms, result.traj_latency_ms)
                        action_start = time.perf_counter()
                        _, executed = execute_stop_waypoint_chunks(eval_env, result.refined_waypoints)
                        action_times = action_timing(eval_env, (time.perf_counter() - action_start) * 1000.0, clock.enabled)
                        clock.advance_action(action_times["action_sim_time_ms"], action_times["action_wall_time_ms"])
                        control_step += int(executed)
                        state.sync_runtime_status_from_sim()
                        outputs = eval_env.get_obs()
                        state.process_env_output(outputs)
                        if not state.dones[0]:
                            state.record_request_ne()
                            dino_ms = state.run_dino_and_update_metric(model_wrapper)
                            clock.advance_blocking(dino_ms)
                        else:
                            dino_ms = 0.0
                        record = _build_planner_decision_record(env_batch, result, state.current_sim_pose(), enable_comm_delay, 1, result.request_id, clock, control_step)
                        record.update({
                            "record_type": "ncn_stopgo_edge",
                            "control_source": "edge_waypoint",
                            "ncn_recovery_reason": "valid_edge_result_applied",
                            "executed_waypoints": int(executed),
                            "airsim_action_latency_ms": float(action_times["airsim_action_latency_ms"]),
                            "action_sim_time_ms": float(action_times["action_sim_time_ms"]),
                            "action_wall_time_ms": float(action_times["action_wall_time_ms"]),
                            "groundingdino_latency_ms": float(dino_ms),
                            "collisions": [bool(state.collisions[0])], "dones": [bool(state.dones[0])],
                        })
                        fp.write(json.dumps(record, ensure_ascii=False) + "\\n"); fp.flush(); records.append(record)
                        if not state.dones[0]:
                            request_id += 1
                            snapshot = state.build_snapshot(request_id, control_step)
                            snapshot.submitted_logical_ms = clock.now_ms if clock.enabled else None
                            if not planner.submit(snapshot):
                                raise RuntimeError("failed to submit next Stop-go+NCN request")
                        continue

                    predicted = planner.predicted_remaining_ms(edge_latency.compute_ms)
                    if predicted is None:
                        raise RuntimeError("Stop-go+NCN lost its pending edge request")
                    if predicted <= float(args.ncn_response_safety_margin_ms):
                        ready_result = planner.wait_result()
                        continue
                    record = _run_ncn_action(
                        state, eval_env, runtime, env_batch["instruction"], clock,
                        control_step + 1, predicted, 0.0,
                        "stopgo_zero_buffer_edge_pending",
                    )
                    control_step += 1
                    record["record_type"] = "ncn_stopgo_action"
                    fp.write(json.dumps(record, ensure_ascii=False) + "\\n"); fp.flush(); records.append(record)
                state.maybe_finalize(force=True)
                terminal = {"record_type": "episode_end", "seq_names": [env_batch["seq_name"]], "map_names": [env_batch["map_name"]], "control_steps": int(control_step), "episode_latency_ms": float(clock.now_ms), "success": bool(state.success), "oracle_success": bool(state.oracle_success), "collision": bool(state.collisions[0]), "final_ne_m": float(state.distance_to_ends[-1]) if state.distance_to_ends else None}
                terminal.update(clock.metadata())
                fp.write(json.dumps(terminal, ensure_ascii=False) + "\\n"); fp.flush(); records.append(terminal)
            finally:
                planner.close()
    ncn_records = [x for x in records if x.get("control_source") == "ncn"]
    terminals = [x for x in records if x.get("record_type") == "episode_end"]
    ncn_episode_ids = {x["seq_names"][0] for x in ncn_records if x.get("seq_names")}
    summary = {"ncn_enabled": True, "ncn_action_count": len(ncn_records), "ncn_activation_rate": float(len(ncn_episode_ids) / len(terminals)) if terminals else 0.0, "ncn_image_text_to_action_latency_ms": _metric_summary([x["image_text_to_action_latency_ms"] for x in ncn_records]), "num_records": len(records)}
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    runtime.close()


if __name__ == "__main__":
    if not _as_bool(args.enable_ncn):
        raise ValueError("stopgo_ncn_eval.py requires --enable_ncn True")
    configure_fast_eval_output(args, "stopgo_ncn")
    eval_save_path = args.eval_save_path
    eval_json_path = args.eval_json_path
    dataset_path = args.dataset_path

    if not os.path.exists(eval_save_path):
        os.makedirs(eval_save_path)
    profile_log_dir = os.path.join(eval_save_path, "profile_logs")
    os.makedirs(profile_log_dir, exist_ok=True)

    enable_comm_delay = bool(args.enable_comm_delay)
    trace_path = Path(args.comm_trace_csv_path) if args.comm_trace_csv_path else Path(default_trace_path())
    if not trace_path.exists():
        raise FileNotFoundError(f"bandwidth trace not found: {trace_path}")
    bandwidth_trace = BandwidthTrace(trace_path, cycle=True)
    logger.info(f"Loaded bandwidth trace: {trace_path} ({bandwidth_trace.sample_count} samples), enable_comm_delay={enable_comm_delay}")

    setup()

    assert CheckPort(), "error port"

    eval_env = initialize_env_eval(dataset_path=dataset_path, save_path=eval_save_path, eval_json_path=eval_json_path)

    if is_dist_avail_and_initialized():
        torch.distributed.destroy_process_group()

    args.DistributedDataParallel = False

    model_wrapper = ProfileTravelModelWrapper(model_args=model_args, data_args=data_args)
    model_wrapper.dino_moinitor = DinoMonitor.get_instance()

    assist = Assist(always_help=args.always_help, use_gt=args.use_gt)

    print("Assist setting: always_help --", args.always_help, "    use_gt --", args.use_gt)

    profile_log_path = os.path.join(profile_log_dir, f"stopgo_ncn_{args.make_dir_time}.jsonl")
    summary_path = os.path.join(profile_log_dir, f"stopgo_ncn_{args.make_dir_time}_summary.json")

    eval_stopgo_ncn(
        model_wrapper=model_wrapper,
        assist=assist,
        eval_env=eval_env,
        profile_log_path=profile_log_path,
        summary_path=summary_path,
        bandwidth_trace=bandwidth_trace,
        enable_comm_delay=enable_comm_delay,
    )

    eval_env.delete_VectorEnvUtil()
