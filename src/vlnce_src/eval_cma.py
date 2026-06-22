"""
CMA model evaluation on TravelUAV dataset.
Simplified version of eval.py — uses CMAModelWrapper instead of LLaMA-UAV.
"""
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
from env_uav import AirVLNENV
from src.common.param import args
from src.model_wrapper.cma_wrapper import CMAModelWrapper
from src.vlnce_src.closeloop_util import (
    BatchIterator,
    CheckPort,
    EvalBatchState,
    initialize_env_eval,
    setup,
)
from utils.logger import logger


def eval(model_wrapper, assist, eval_env, eval_save_dir, results_path):
    model_wrapper.eval()

    all_results = []
    with torch.no_grad():
        dataset = BatchIterator(eval_env)
        end_iter = len(dataset)
        pbar = tqdm.tqdm(total=end_iter)

        while True:
            env_batchs = eval_env.next_minibatch()
            if env_batchs is None:
                break

            episode_ok = False
            for retry_i in range(3):
                try:
                    batch_state = EvalBatchState(
                        batch_size=eval_env.batch_size,
                        env_batchs=env_batchs,
                        env=eval_env,
                        assist=assist,
                    )
                    pbar.update(n=eval_env.batch_size)
                    assist_notices = None

                    for t in range(int(args.maxWaypoints) + 1):
                        logger.info(
                            "Step: {} \t Completed: {} / {}".format(
                                t, int(eval_env.index_data) - int(eval_env.batch_size), end_iter
                            )
                        )

                        is_terminate = batch_state.check_batch_termination(t)
                        if is_terminate:
                            break

                        inputs, rot_to_targets = model_wrapper.prepare_inputs(
                            batch_state.episodes,
                            batch_state.target_positions,
                            assist_notices,
                        )
                        refined_waypoints, _ = model_wrapper.run_profiled(
                            inputs=inputs,
                            episodes=batch_state.episodes,
                            rot_to_targets=rot_to_targets,
                        )

                        eval_env.makeActions(refined_waypoints)
                        outputs = eval_env.get_obs()
                        batch_state.update_from_env_output(outputs)
                        batch_state.predict_dones = model_wrapper.predict_done(
                            batch_state.episodes, batch_state.object_infos
                        )
                        batch_state.update_metric()
                        assist_notices = batch_state.get_assist_notices()

                    episode_ok = True
                    break
                except Exception as e:
                    logger.error(f"Episode failed: {e}, retry {retry_i + 1}/3")
                    if retry_i < 2:
                        eval_env._changeEnv(need_change=True)

            if not episode_ok:
                raise RuntimeError("Episode failed after 3 retries")

            # Collect metrics for each completed batch element
            for i in range(eval_env.batch_size):
                result = {
                    "episode_id": env_batchs[i].get("episode_id", f"batch_{i}"),
                    "success": batch_state.success[i],
                    "oracle_success": batch_state.oracle_success[i],
                    "collision": batch_state.collisions[i],
                    "steps": len(batch_state.episodes[i]),
                    "distance_to_end": batch_state.distance_to_ends[i][-1],
                }
                all_results.append(result)

        pbar.close()

    # Compute summary metrics
    success = [r["success"] for r in all_results]
    oracle_success = [r["oracle_success"] for r in all_results]
    collisions = [r["collision"] for r in all_results]
    steps = [r["steps"] for r in all_results]
    distances = [r["distance_to_end"] for r in all_results]

    summary = {
        "num_episodes": len(all_results),
        "success_rate": float(np.mean(success)),
        "oracle_success_rate": float(np.mean(oracle_success)),
        "collision_rate": float(np.mean(collisions)),
        "avg_steps": float(np.mean(steps)),
        "avg_distance_to_end": float(np.mean(distances)),
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print("CMA Evaluation Results")
    print("=" * 50)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("=" * 50)


if __name__ == "__main__":
    # Add CMA-specific args (must be done before arg parse)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cma_ckpt_path", type=str, required=True)
    parser.add_argument("--cma_vocab_path", type=str, required=True)
    cma_args, _ = parser.parse_known_args()

    eval_save_path = args.eval_save_path
    eval_json_path = args.eval_json_path
    dataset_path = args.dataset_path

    os.makedirs(eval_save_path, exist_ok=True)

    setup()

    assert CheckPort(), "error port"

    eval_env = initialize_env_eval(
        dataset_path=dataset_path,
        save_path=eval_save_path,
        eval_json_path=eval_json_path,
    )

    args.DistributedDataParallel = False

    model_wrapper = CMAModelWrapper(
        ckpt_path=cma_args.cma_ckpt_path,
        vocab_path=cma_args.cma_vocab_path,
        device=f"cuda:{args.gpu_id}",
    )

    assist = Assist(always_help=args.always_help, use_gt=args.use_gt)
    print(f"Assist: always_help={args.always_help}, use_gt={args.use_gt}")

    results_path = os.path.join(eval_save_path, f"cma_results_{args.make_dir_time}.json")

    eval(
        model_wrapper=model_wrapper,
        assist=assist,
        eval_env=eval_env,
        eval_save_dir=eval_save_path,
        results_path=results_path,
    )

    eval_env.delete_VectorEnvUtil()
