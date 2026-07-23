import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import tqdm

sys.path.append(str(Path(str(os.getcwd())).resolve()))

from assist import Assist
from src.common.param import args, data_args, model_args
from src.model_wrapper.profile_travel_llm import ProfileTravelModelWrapper
from src.vlnce_src.closeloop_util import BatchIterator, CheckPort, initialize_env_eval, is_dist_avail_and_initialized, setup
from src.vlnce_src.comm_delay import BandwidthTrace, default_trace_path
from src.vlnce_src.continue_eval import (
    _fmt_point, _fmt_optional_m, _fmt_optional_ms, _fmt_optional_mbps,
    _print_episode_header, _print_exec_profile_line, _print_trajectory_bundle,
    _waypoint_segment_stats,
)
from src.vlnce_src.dino_monitor_online import DinoMonitor
from src.vlnce_src.drl_ac_policy import SplitACPolicy  # required for PPO.load deserialization
from src.vlnce_src.drl_scheduler_env import ACTION_NAMES, DRLSchedulerEnv, _metric_summary, _as_bool
from src.vlnce_src.fast_eval_time import configure_fast_eval_output
from utils.logger import logger


def main():
    from stable_baselines3 import PPO

    if not args.scheduler_model_path:
        raise ValueError("--scheduler_model_path is required for DRL scheduler eval")
    configure_fast_eval_output(args, "drl_based_hybrid")
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
            tags: List[str] = []
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
    }
    final_summary_path = os.path.join(profile_dir, f"drl_eval_metrics_{args.make_dir_time}.json")
    with open(final_summary_path, "w", encoding="utf-8") as handle:
        json.dump(final_summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    print(f"Saved DRL eval metrics: {final_summary_path}")


if __name__ == "__main__":
    main()
