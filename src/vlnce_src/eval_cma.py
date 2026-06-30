"""
CMA model evaluation on TravelUAV dataset.
Simplified version of eval.py — uses CMAModelWrapper instead of LLaMA-UAV.
"""
import json
import os
import sys
import time
from pathlib import Path

# ---- Parse CMA-specific args first, BEFORE importing param ----
import argparse
_cma_parser = argparse.ArgumentParser()
_cma_parser.add_argument("--cma_ckpt_path", type=str, required=True)
_cma_parser.add_argument("--cma_vocab_path", type=str, required=True)
_cma_parser.add_argument("--simulator_ip", type=str, default="127.0.0.1")
_cma_args, _remaining = _cma_parser.parse_known_args()
# Remove CMA args from sys.argv so HF parser in param.py doesn't see them
sys.argv = [sys.argv[0]] + _remaining

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

                    # Episode start log
                    ep_idx = int(eval_env.index_data) - int(eval_env.batch_size)
                    for bi in range(eval_env.batch_size):
                        eb = env_batchs[bi]
                        start_p = eb['trajectory'][0]['position']
                        tgt_p = eb['object_position']
                        dist = np.linalg.norm(np.array(start_p[0:3]) - np.array(tgt_p[0:3]))
                        instr = eb.get('instruction', '')
                        if isinstance(instr, str):
                            instr = instr[:80]
                        logger.info(f"[Ep {ep_idx + bi}] scene={eb['map_name']} | start=({start_p[0]:.1f},{start_p[1]:.1f},{start_p[2]:.1f}) → target=({tgt_p[0]:.1f},{tgt_p[1]:.1f},{tgt_p[2]:.1f}) dist={dist:.1f}m | {instr}")

                    for t in range(int(args.maxWaypoints) + 1):

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
                    import traceback
                    logger.error(f"Episode failed: {e}, retry {retry_i + 1}/3")
                    traceback.print_exc()
                    if retry_i < 2:
                        eval_env._changeEnv(need_change=True)

            if not episode_ok:
                raise RuntimeError("Episode failed after 3 retries")

            # Collect metrics for each completed batch element
            ep_idx = int(eval_env.index_data) - int(eval_env.batch_size)
            for i in range(eval_env.batch_size):
                s, o, c, st = batch_state.success[i], batch_state.oracle_success[i], batch_state.collisions[i], len(batch_state.episodes[i])
                d = batch_state.distance_to_ends[i][-1]
                status = 'SUCCESS' if s else ('ORACLE' if o else ('COLLISION' if c else 'TIMEOUT/STOP'))
                logger.info(f"[Ep {ep_idx + i} result] {status} | steps={st} final_dist={d:.1f}m | success={s} oracle={o} collision={c}")
                result = {
                    "episode_id": env_batchs[i].get("episode_id", f"batch_{i}"),
                    "success": s, "oracle_success": o, "collision": c,
                    "steps": st, "distance_to_end": d,
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
    # Point to AirSim server
    args.machines_info[0]["MACHINE_IP"] = _cma_args.simulator_ip
    print(f"AirSim server: {_cma_args.simulator_ip}:{args.simulator_tool_port}")

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
        ckpt_path=_cma_args.cma_ckpt_path,
        vocab_path=_cma_args.cma_vocab_path,
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
