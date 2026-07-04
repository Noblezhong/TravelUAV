import os
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(str(os.getcwd())).resolve()))

from assist import Assist
from src.common.param import args, data_args, model_args
from src.model_wrapper.profile_travel_llm import ProfileTravelModelWrapper
from src.vlnce_src.closeloop_util import CheckPort, initialize_env_eval, is_dist_avail_and_initialized, setup
from src.vlnce_src.comm_delay import BandwidthTrace, default_trace_path
from src.vlnce_src.dino_monitor_online import DinoMonitor
from src.vlnce_src.drl_scheduler_env import DRLSchedulerEnv, _as_bool
from utils.logger import logger


def main():
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    os.makedirs(args.eval_save_path, exist_ok=True)
    profile_dir = os.path.join(args.eval_save_path, "profile_logs")
    model_dir = os.path.join(args.eval_save_path, "scheduler_models")
    tb_dir = os.path.join(args.eval_save_path, "tensorboard")
    os.makedirs(profile_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    enable_comm_delay = _as_bool(args.enable_comm_delay)
    trace_path = Path(args.comm_trace_csv_path) if args.comm_trace_csv_path else Path(default_trace_path())
    bandwidth_trace = BandwidthTrace(trace_path, cycle=True)
    logger.info(f"DRL scheduler train trace={trace_path}, samples={bandwidth_trace.sample_count}")

    setup()
    assert CheckPort(), "error port"
    eval_env = initialize_env_eval(
        dataset_path=args.dataset_path,
        save_path=args.eval_save_path,
        eval_json_path=args.eval_json_path,
    )
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
        profile_log_path=os.path.join(profile_dir, f"drl_train_{args.make_dir_time}.jsonl"),
        summary_path=os.path.join(profile_dir, f"drl_train_{args.make_dir_time}_summary.json"),
        enable_comm_delay=enable_comm_delay,
        max_waypoints=args.maxWaypoints,
    )
    monitored_env = Monitor(gym_env, filename=os.path.join(profile_dir, f"drl_train_monitor_{args.make_dir_time}.csv"))
    scheduler = PPO(
        "MlpPolicy",
        monitored_env,
        learning_rate=float(args.scheduler_learning_rate),
        n_steps=int(args.scheduler_n_steps),
        batch_size=int(args.scheduler_batch_size),
        gamma=float(args.scheduler_gamma),
        verbose=1,
        tensorboard_log=tb_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    scheduler.learn(total_timesteps=int(args.scheduler_total_timesteps), tb_log_name="ppo_scheduler")
    final_path = os.path.join(model_dir, f"ppo_scheduler_{args.make_dir_time}")
    scheduler.save(final_path)
    gym_env.write_summary()
    gym_env.close()
    eval_env.delete_VectorEnvUtil()
    print(f"Saved PPO scheduler: {final_path}.zip")


if __name__ == "__main__":
    main()
