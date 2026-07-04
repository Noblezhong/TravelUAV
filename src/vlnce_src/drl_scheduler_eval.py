import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(str(os.getcwd())).resolve()))

from assist import Assist
from src.common.param import args, data_args, model_args
from src.model_wrapper.profile_travel_llm import ProfileTravelModelWrapper
from src.vlnce_src.closeloop_util import BatchIterator, CheckPort, initialize_env_eval, is_dist_avail_and_initialized, setup
from src.vlnce_src.comm_delay import BandwidthTrace, default_trace_path
from src.vlnce_src.dino_monitor_online import DinoMonitor
from src.vlnce_src.drl_scheduler_env import ACTION_NAMES, DRLSchedulerEnv, _as_bool, _metric_summary
from utils.logger import logger


def main():
    from stable_baselines3 import PPO

    if not args.scheduler_model_path:
        raise ValueError("--scheduler_model_path is required for DRL scheduler eval")
    os.makedirs(args.eval_save_path, exist_ok=True)
    profile_dir = os.path.join(args.eval_save_path, "profile_logs")
    os.makedirs(profile_dir, exist_ok=True)

    enable_comm_delay = _as_bool(args.enable_comm_delay)
    trace_path = Path(args.comm_trace_csv_path) if args.comm_trace_csv_path else Path(default_trace_path())
    bandwidth_trace = BandwidthTrace(trace_path, cycle=True)
    logger.info(f"DRL scheduler eval trace={trace_path}, samples={bandwidth_trace.sample_count}")

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

    model_wrapper = ProfileTravelModelWrapper(model_args=model_args, data_args=data_args)
    model_wrapper.dino_moinitor = DinoMonitor.get_instance()
    model_wrapper.eval()
    assist = Assist(always_help=args.always_help, use_gt=args.use_gt)

    gym_env = DRLSchedulerEnv(
        model_wrapper=model_wrapper,
        assist=assist,
        eval_env=eval_env,
        bandwidth_trace=bandwidth_trace,
        profile_log_path=os.path.join(profile_dir, f"drl_eval_{args.make_dir_time}.jsonl"),
        summary_path=os.path.join(profile_dir, f"drl_eval_{args.make_dir_time}_summary.json"),
        enable_comm_delay=enable_comm_delay,
        max_waypoints=args.maxWaypoints,
        deterministic_eval=True,
    )
    scheduler = PPO.load(args.scheduler_model_path, env=None, device="cuda" if torch.cuda.is_available() else "cpu")

    completed = 0
    while completed < total_episodes:
        obs, _ = gym_env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = scheduler.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = gym_env.step(int(action))
        completed += 1

    gym_env.write_summary()
    gym_env.close()
    eval_env.delete_VectorEnvUtil()

    episode_records = [item for item in gym_env.summary_records if item.get("record_type") == "episode_end"]
    step_records = [item for item in gym_env.summary_records if item.get("record_type") == "scheduler_step"]
    success_count = sum(1 for item in episode_records if item.get("success"))
    oracle_success_count = sum(1 for item in episode_records if item.get("oracle_success"))
    collision_count = sum(1 for item in episode_records if item.get("collision"))
    action_counts = {name: sum(1 for item in step_records if item.get("action_name") == name) for name in ACTION_NAMES.values()}
    final_summary = {
        "episodes": len(episode_records),
        "SR": success_count / len(episode_records) if episode_records else 0.0,
        "OSR": oracle_success_count / len(episode_records) if episode_records else 0.0,
        "CR": collision_count / len(episode_records) if episode_records else 0.0,
        "avg_waypoints": float(np.mean([item.get("control_steps", 0) for item in episode_records])) if episode_records else 0.0,
        "avg_NE_m": float(np.mean([item.get("final_ne_m", 0.0) for item in episode_records if item.get("final_ne_m") is not None])) if episode_records else 0.0,
        "avg_episode_latency_ms": _metric_summary([item.get("episode_latency_ms") for item in episode_records]),
        "avg_action_delay_ms": _metric_summary([item.get("action_age_ms") for item in step_records]),
        "avg_state_drift_m": _metric_summary([item.get("state_drift_m") for item in step_records]),
        "avg_T_action_ms": _metric_summary([item.get("airsim_action_latency_ms") for item in step_records]),
        "avg_T_dec_ms": _metric_summary([item.get("decision_total_latency_ms") for item in step_records if item.get("decision_total_latency_ms") is not None]),
        "action_counts": action_counts,
        "illegal_action_count": int(sum(1 for item in step_records if item.get("illegal_action"))),
    }
    final_summary_path = os.path.join(profile_dir, f"drl_eval_metrics_{args.make_dir_time}.json")
    with open(final_summary_path, "w", encoding="utf-8") as handle:
        json.dump(final_summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    print(f"Saved DRL eval metrics: {final_summary_path}")


if __name__ == "__main__":
    main()
