import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import tqdm

from src.common.param import args, model_args
from src.model_wrapper.aerodpo_eval_wrapper import AeroDPOEvalModelWrapper
from src.vlnce_src.aerodpo_eval_contract import (
    resolve_aerodpo_stop,
    summarize_latencies,
)
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
from src.vlnce_src.fast_eval_time import (
    FastEvalClock,
    action_timing,
    configure_fast_eval_output,
)
from utils.logger import logger


class AeroDPOEvalBatchState(EvalBatchState):
    """TravelUAV failure contract with AeroDPO's native LAND termination."""

    def __init__(self, *args_, **kwargs):
        super().__init__(*args_, **kwargs)
        self.termination_reasons = [None] * self.batch_size

    def update_metric(self):
        for index in range(self.batch_size):
            if self.dones[index]:
                continue
            outcome = resolve_aerodpo_stop(
                self.predict_dones[index], self.distance_to_ends[index][-1]
            )
            if outcome == "success":
                self.success[index] = True
                self.dones[index] = True
                self.termination_reasons[index] = "aerodpo_land_success"
            elif outcome == "early_end":
                self.early_end[index] = True
                self.dones[index] = True
                self.termination_reasons[index] = "aerodpo_land_early_end"

    def reason_for(self, index, step_index):
        if self.termination_reasons[index]:
            return self.termination_reasons[index]
        if self.collisions[index]:
            return "collision"
        if step_index >= int(args.maxWaypoints):
            return "timeout"
        return "ne_regression"


def _write_jsonl_line(handle, payload):
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def _metric_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def _validate_seen_split(path, allow_nonstandard=False):
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    sequence_paths = {str(row["json"]) for row in rows}
    if not allow_nonstandard and len(sequence_paths) != 1418:
        raise ValueError(
            f"AeroDPO seen evaluation requires 1,418 unique episodes, found {len(sequence_paths)}"
        )


def _terminal_record(batch_state, env_batch, index, decisions, actions, clock, step_index):
    record = {
        "record_type": "episode_end",
        "seq_names": [env_batch["seq_name"]],
        "map_names": [env_batch["map_name"]],
        "decision_steps": int(decisions),
        "control_steps": int(actions),
        "executed_waypoints": int(actions),
        "success": bool(batch_state.success[index]),
        "oracle_success": bool(batch_state.oracle_success[index]),
        "collision": bool(batch_state.collisions[index]),
        "final_ne_m": float(batch_state.distance_to_ends[index][-1]),
        "termination_reason": batch_state.reason_for(index, step_index),
        "termination_backend": "aerodpo_land",
        "time_shift_s": None,
        "state_shift_m": None,
    }
    record.update(clock.metadata())
    return record


def evaluate(
    model_wrapper,
    eval_env,
    profile_log_path,
    summary_path,
    bandwidth_trace,
    enable_comm_delay,
):
    model_wrapper.eval()
    fast_eval = bool(args.fast_eval)
    summary_records = []
    episode_records = []
    model_latency_episodes = []
    input_prepare_episodes = []
    full_decision_episodes = []

    with torch.no_grad(), open(profile_log_path, "w", encoding="utf-8") as profile_fp:
        end_iter = len(BatchIterator(eval_env))
        progress = tqdm.tqdm(total=end_iter)
        while True:
            env_batchs = eval_env.next_minibatch()
            if env_batchs is None:
                break
            if len(env_batchs) != 1:
                raise RuntimeError(
                    f"AeroDPO benchmark requires batch size 1, got {len(env_batchs)}"
                )

            episode_ok = False
            for retry_index in range(3):
                try:
                    env_batch = env_batchs[0]
                    bandwidth_trace.reset_for_episode(env_batch["seq_name"])
                    batch_state = AeroDPOEvalBatchState(
                        batch_size=eval_env.batch_size,
                        env_batchs=env_batchs,
                        env=eval_env,
                        assist=None,
                        ignore_tiny_diff=True,
                    )
                    clock = FastEvalClock(fast_eval, args.fast_eval_speedup)
                    bandwidth_t0_ms = float(clock.now_ms)
                    completed_decisions = 0
                    executed_actions = 0
                    episode_step_records = []
                    episode_model_latencies = []
                    episode_input_prepare_latencies = []
                    episode_full_decision_latencies = []
                    terminal_step = int(args.maxWaypoints)

                    for step_index in range(int(args.maxWaypoints) + 1):
                        if batch_state.check_batch_termination(step_index):
                            terminal_step = step_index
                            break

                        sequence_names = [item["seq_name"] for item in env_batchs]
                        map_names = [item["map_name"] for item in env_batchs]
                        instructions = [item["instruction"] for item in env_batchs]
                        logical_start_ms = float(clock.now_ms)
                        decision_start = time.perf_counter()

                        input_start = time.perf_counter()
                        inputs, prompts = model_wrapper.prepare_inputs(
                            batch_state.episodes,
                            batch_state.target_positions,
                            instructions,
                        )
                        input_prepare_latency_ms = (
                            time.perf_counter() - input_start
                        ) * 1000.0
                        actions, model_stops, model_profile = model_wrapper.run_profiled(inputs)
                        full_decision_latency_ms = (
                            time.perf_counter() - decision_start
                        ) * 1000.0
                        model_inference_latency_ms = float(
                            model_profile["model_inference_latency_ms"]
                        )
                        clock.advance_blocking(model_inference_latency_ms)

                        action_start = time.perf_counter()
                        eval_env.makeAeroDPOActions(actions)
                        action_wall_ms = (time.perf_counter() - action_start) * 1000.0
                        action_times = action_timing(eval_env, action_wall_ms, fast_eval)
                        clock.advance_action(
                            action_times["action_sim_time_ms"],
                            action_times["action_wall_time_ms"],
                        )
                        completed_decisions += 1
                        executed_actions += len(actions)

                        observation_start = time.perf_counter()
                        outputs = eval_env.get_obs()
                        observation_latency_ms = (
                            time.perf_counter() - observation_start
                        ) * 1000.0
                        clock.advance_blocking(observation_latency_ms)

                        _, payload_bits, payload_mb = (
                            estimate_uplink_payload_bits_from_outputs(outputs)
                        )
                        bandwidth_bps = bandwidth_trace.bandwidth_at_ms(
                            float(clock.now_ms) - bandwidth_t0_ms
                        )
                        uplink_latency_ms = (
                            calculate_latency_ms(payload_bits, bandwidth_bps)
                            if enable_comm_delay
                            else 0.0
                        )
                        if uplink_latency_ms > 0.0 and not fast_eval:
                            time.sleep(uplink_latency_ms / 1000.0)
                        if fast_eval:
                            clock.advance_blocking(uplink_latency_ms)

                        batch_state.update_from_env_output(outputs)
                        batch_state.predict_dones = [bool(value) for value in model_stops]
                        batch_state.update_metric()

                        record = {
                            "record_type": "aerodpo_action",
                            "episode_step": int(step_index),
                            "decision_steps": int(completed_decisions),
                            "executed_waypoints": int(executed_actions),
                            "batch_size": int(eval_env.batch_size),
                            "seq_names": sequence_names,
                            "map_names": map_names,
                            "model_inference_latency_ms": model_inference_latency_ms,
                            "input_prepare_latency_ms": input_prepare_latency_ms,
                            "full_model_decision_latency_ms": full_decision_latency_ms,
                            "model_output_text": model_profile["model_output_text"],
                            "aerodpo_action": actions,
                            "model_stop": [bool(value) for value in model_stops],
                            "termination_backend": "aerodpo_land",
                            "llm_latency_ms": model_inference_latency_ms,
                            "traj_latency_ms": 0.0,
                            "groundingdino_latency_ms": 0.0,
                            "airsim_action_latency_ms": float(
                                action_times["airsim_action_latency_ms"]
                            ),
                            "action_wall_time_ms": float(
                                action_times["action_wall_time_ms"]
                            ),
                            "action_sim_time_ms": float(
                                action_times["action_sim_time_ms"]
                            ),
                            "obs_latency_ms": observation_latency_ms,
                            "uplink_payload_bits": int(payload_bits),
                            "uplink_payload_mb": float(payload_mb),
                            "uplink_bandwidth_mbps": float(bandwidth_bps / 1_000_000.0),
                            "uplink_latency_ms": float(uplink_latency_ms),
                            "decision_latency_ms": (
                                model_inference_latency_ms
                                + observation_latency_ms
                                + uplink_latency_ms
                            ),
                            "prompt": prompts,
                            "collisions": [bool(value) for value in batch_state.collisions],
                            "dones": [bool(value) for value in batch_state.dones],
                        }
                        record.update(clock.metadata())
                        episode_step_records.append(record)
                        episode_model_latencies.append(model_inference_latency_ms)
                        episode_input_prepare_latencies.append(input_prepare_latency_ms)
                        episode_full_decision_latencies.append(full_decision_latency_ms)

                    for record in episode_step_records:
                        _write_jsonl_line(profile_fp, record)
                    for index, env_batch in enumerate(env_batchs):
                        terminal = _terminal_record(
                            batch_state,
                            env_batch,
                            index,
                            completed_decisions,
                            executed_actions,
                            clock,
                            terminal_step,
                        )
                        _write_jsonl_line(profile_fp, terminal)
                        episode_records.append(terminal)
                    summary_records.extend(episode_step_records)
                    model_latency_episodes.append(episode_model_latencies)
                    input_prepare_episodes.append(episode_input_prepare_latencies)
                    full_decision_episodes.append(episode_full_decision_latencies)
                    progress.update(n=eval_env.batch_size)
                    episode_ok = True
                    break
                except Exception as error:
                    logger.exception(
                        "AeroDPO episode %s failed on attempt %d/3: %s",
                        env_batchs[0]["seq_name"],
                        retry_index + 1,
                        error,
                    )
                    if retry_index < 2:
                        eval_env._changeEnv(need_change=True)

            if not episode_ok:
                raise RuntimeError(
                    f"AeroDPO episode failed after 3 retries: {env_batchs[0]['seq_name']}"
                )

        progress.close()

    summary = {
        "model_generate_latency_ms": summarize_latencies(model_latency_episodes),
        "input_prepare_latency_ms": summarize_latencies(input_prepare_episodes),
        "full_model_decision_latency_ms": summarize_latencies(full_decision_episodes),
        "airsim_action_latency_ms": _metric_summary(
            [record["airsim_action_latency_ms"] for record in summary_records]
        ),
        "obs_latency_ms": _metric_summary(
            [record["obs_latency_ms"] for record in summary_records]
        ),
        "uplink_latency_ms": _metric_summary(
            [record["uplink_latency_ms"] for record in summary_records]
        ),
        "decision_steps": _metric_summary(
            [record["decision_steps"] for record in episode_records]
        ),
        "actual_aerodpo_action_steps": _metric_summary(
            [record["control_steps"] for record in episode_records]
        ),
        "episodes": len(episode_records),
        "decisions": len(summary_records),
        "comm_delay_enabled": bool(enable_comm_delay),
        "fast_eval": fast_eval,
        "fast_eval_speedup": float(args.fast_eval_speedup if fast_eval else 1.0),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def main():
    configure_fast_eval_output(args, "aerodpo_stop_and_go")
    _validate_seen_split(
        args.eval_json_path,
        allow_nonstandard=bool(args.allow_nonstandard_eval_count),
    )
    if not os.path.exists(args.eval_save_path):
        os.makedirs(args.eval_save_path)
    profile_dir = os.path.join(args.eval_save_path, "profile_logs")
    os.makedirs(profile_dir, exist_ok=True)

    trace_path = (
        Path(args.comm_trace_csv_path)
        if args.comm_trace_csv_path
        else Path(default_trace_path())
    )
    if not trace_path.exists():
        raise FileNotFoundError(f"bandwidth trace not found: {trace_path}")
    bandwidth_trace = BandwidthTrace(trace_path, cycle=True)

    setup()
    if not CheckPort():
        raise RuntimeError("DDP master port is already in use")
    eval_env = initialize_env_eval(
        dataset_path=args.dataset_path,
        save_path=args.eval_save_path,
        eval_json_path=args.eval_json_path,
    )
    # AeroDPO neither uses GroundingDINO record frames nor its unavailable
    # Record-camera depth stream.  Keep the native policy RGB-D, and expose
    # front/down policy frames as records for trajectory serialization.
    eval_env.aerodpo_eval_mode = True
    if is_dist_avail_and_initialized():
        torch.distributed.destroy_process_group()
    args.DistributedDataParallel = False

    profile_path = os.path.join(
        profile_dir, f"aerodpo_stopgo_{args.make_dir_time}.jsonl"
    )
    summary_path = os.path.join(
        profile_dir, f"aerodpo_stopgo_{args.make_dir_time}_summary.json"
    )
    model_wrapper = AeroDPOEvalModelWrapper(model_args)
    try:
        evaluate(
            model_wrapper,
            eval_env,
            profile_path,
            summary_path,
            bandwidth_trace,
            bool(args.enable_comm_delay),
        )
    finally:
        eval_env.delete_VectorEnvUtil()


if __name__ == "__main__":
    main()
