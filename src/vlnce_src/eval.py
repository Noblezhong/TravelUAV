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
                        batch_state = EvalBatchState(batch_size=eval_env.batch_size, env_batchs=env_batchs, env=eval_env, assist=assist)
                        episode_clock = FastEvalClock(fast_eval, args.fast_eval_speedup)
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
                            eval_env.makeActions(refined_waypoints)
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

                            obs_start = time.perf_counter()
                            outputs = eval_env.get_obs()
                            obs_end = time.perf_counter()
                            obs_latency_ms = float((obs_end - obs_start) * 1000.0)
                            episode_clock.advance_blocking(obs_latency_ms)

                            payload_bytes, payload_bits, payload_mb = estimate_uplink_payload_bits_from_outputs(outputs)
                            bandwidth_bps = bandwidth_trace.next_bandwidth_bps()
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
                                "control_steps": int(t),
                                "episode_latency_ms": float(episode_clock.now_ms),
                                "success": bool(batch_state.success[batch_idx]),
                                "oracle_success": bool(batch_state.oracle_success[batch_idx]),
                                "collision": bool(batch_state.collisions[batch_idx]),
                                "final_ne_m": float(batch_state.distance_to_ends[batch_idx][-1]),
                            }
                            episode_record.update(episode_clock.metadata())
                            _write_jsonl_line(profile_fp, episode_record)
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
        "num_records": len(summary_records),
        "comm_delay_enabled": bool(enable_comm_delay),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    configure_fast_eval_output(args, "stop_and_go")
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

    profile_log_path = os.path.join(profile_log_dir, f"eval_{args.make_dir_time}.jsonl")
    summary_path = os.path.join(profile_log_dir, f"eval_{args.make_dir_time}_summary.json")

    eval(
        model_wrapper=model_wrapper,
        assist=assist,
        eval_env=eval_env,
        eval_save_dir=eval_save_path,
        profile_log_path=profile_log_path,
        summary_path=summary_path,
        bandwidth_trace=bandwidth_trace,
        enable_comm_delay=enable_comm_delay,
    )

    eval_env.delete_VectorEnvUtil()
