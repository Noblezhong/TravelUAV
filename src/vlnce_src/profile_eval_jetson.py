import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import tqdm

sys.path.append(str(Path(str(os.getcwd())).resolve()))

from assist import Assist
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
from src.vlnce_src.dino_monitor_online import DinoMonitor
from env_uav import AirVLNENV
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


def _print_profile_line(step_idx, profile_info, refined_waypoints):
    coarse = profile_info["coarse_waypoints"]
    refined_shape = list(np.asarray(refined_waypoints).shape)
    message = (
        f"[profile] step={step_idx} "
        f"llm={profile_info['llm_latency_ms']:.3f}ms "
        f"traj={profile_info['traj_latency_ms']:.3f}ms "
        f"action={profile_info['airsim_action_latency_ms']:.3f}ms "
        f"obs={profile_info['obs_latency_ms']:.3f}ms "
        f"dino={profile_info['groundingdino_latency_ms']:.3f}ms "
        f"coarse={coarse} "
        f"refined_shape={refined_shape}"
    )
    logger.info(message)


def _configure_jetson_air_sim_server():
    workstation_ip = "192.168.105.17"
    args.machines_info[0]["MACHINE_IP"] = workstation_ip


def eval(model_wrapper: ProfileTravelModelWrapper, assist: Assist, eval_env: AirVLNENV, eval_save_dir, profile_log_path, summary_path):
    model_wrapper.eval()

    summary_records = []
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
                            inputs, rot_to_targets = model_wrapper.prepare_inputs(
                                batch_state.episodes,
                                batch_state.target_positions,
                                assist_notices,
                            )
                            refined_waypoints, profile_info = model_wrapper.run_profiled(inputs=inputs, episodes=batch_state.episodes, rot_to_targets=rot_to_targets)

                            action_start = time.perf_counter()
                            eval_env.makeActions(refined_waypoints)
                            action_end = time.perf_counter()

                            obs_start = time.perf_counter()
                            outputs = eval_env.get_obs()
                            obs_end = time.perf_counter()

                            dino_start = time.perf_counter()
                            batch_state.predict_dones = model_wrapper.predict_done(batch_state.episodes, batch_state.object_infos)
                            dino_end = time.perf_counter()

                            batch_state.update_from_env_output(outputs)
                            batch_state.update_metric()
                            assist_notices = batch_state.get_assist_notices()

                            loop_end = time.perf_counter()

                            profile_info = {
                                "episode_step": int(t),
                                "batch_size": int(eval_env.batch_size),
                                "seq_names": seq_names,
                                "map_names": map_names,
                                "llm_latency_ms": float(profile_info["llm_latency_ms"]),
                                "traj_latency_ms": float(profile_info["traj_latency_ms"]),
                                "airsim_action_latency_ms": float((action_end - action_start) * 1000.0),
                                "obs_latency_ms": float((obs_end - obs_start) * 1000.0),
                                "groundingdino_latency_ms": float((dino_end - dino_start) * 1000.0),
                                "loop_latency_ms": float((loop_end - step_start) * 1000.0),
                                "coarse_waypoints": profile_info["coarse_waypoints"],
                                "refined_waypoints": profile_info["refined_waypoints"],
                                "predict_dones": [bool(x) for x in batch_state.predict_dones],
                                "collisions": [bool(x) for x in batch_state.collisions],
                                "dones": [bool(x) for x in batch_state.dones],
                            }
                            _write_jsonl_line(profile_fp, profile_info)
                            summary_records.append(profile_info)
                            _print_profile_line(t, profile_info, refined_waypoints)
                        episode_ok = True
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
        "num_records": len(summary_records),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    _configure_jetson_air_sim_server()

    eval_save_path = args.eval_save_path
    eval_json_path = args.eval_json_path
    dataset_path = args.dataset_path

    if not os.path.exists(eval_save_path):
        os.makedirs(eval_save_path)
    profile_log_dir = os.path.join(eval_save_path, "profile_logs")
    os.makedirs(profile_log_dir, exist_ok=True)

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

    profile_log_path = os.path.join(profile_log_dir, f"profile_{args.make_dir_time}.jsonl")
    summary_path = os.path.join(profile_log_dir, f"profile_{args.make_dir_time}_summary.json")

    eval(
        model_wrapper=model_wrapper,
        assist=assist,
        eval_env=eval_env,
        eval_save_dir=eval_save_path,
        profile_log_path=profile_log_path,
        summary_path=summary_path,
    )

    eval_env.delete_VectorEnvUtil()
