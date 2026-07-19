import copy
import json
import math
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
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
    save_to_dataset_eval,
    setup,
    target_distance_increasing_for_10frames,
)
from src.vlnce_src.comm_delay import (
    BandwidthTrace,
    calculate_latency_ms,
    default_trace_path,
    estimate_uplink_payload_bits_from_episodes,
)
from src.vlnce_src.dino_monitor_online import DinoMonitor
from env_uav import AirVLNENV
from utils.logger import logger
from utils.utils import *


@dataclass
class Snapshot:
    request_id: int
    submitted_step: int
    episode: List[Dict[str, Any]]
    target_position: List[float]
    object_info: str
    assist_notice: Optional[str]
    observation_timestamp: int
    observation_pose: List[float]
    submitted_perf_time: float
    submitted_wall_time: float
    obs_latency_ms: float
    groundingdino_latency_ms: float


@dataclass
class PlannerResult:
    request_id: int
    submitted_step: int
    observation_timestamp: int
    observation_pose: List[float]
    submitted_perf_time: float
    submitted_wall_time: float
    ready_wall_time: float
    obs_latency_ms: float
    groundingdino_latency_ms: float
    llm_latency_ms: float
    traj_latency_ms: float
    uplink_payload_bits: int
    uplink_payload_mb: float
    uplink_bandwidth_mbps: float
    uplink_latency_ms: float
    llm_output: List[List[float]]
    refined_waypoints: List[List[float]]


def _metric_summary(values):
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
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


def _set_console_log_message_only():
    plain_formatter = logging.Formatter("%(message)s")
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(plain_formatter)


def check_collision_without_tiny_diff(episodes, current_observations, collisions, dones):
    for i, prev_episode in enumerate(episodes):
        if collisions[i]:
            if not dones[i]:
                dones[i] = True
            continue
        if len(prev_episode) == 0:
            continue

        close_collision = False
        current_episode = current_observations[i]
        for cid in range(len(current_episode[-1]["depth"])):
            zero_cnt = (current_episode[-1]["depth"][cid] <= 1).sum()
            if zero_cnt > 0.1 * current_episode[-1]["depth"][cid].size:
                close_collision = True
                break

        collisions[i] = close_collision
        if collisions[i] and not dones[i]:
            dones[i] = True
    return collisions, dones


def _print_episode_header(episode_idx, env_batch, chunk_waypoints):
    logger.info(
        f"\n[episode {episode_idx:04d}] seq={env_batch['seq_name']} "
        f"map={env_batch['map_name']} chunk={int(chunk_waypoints)}"
    )


def _fmt_point(point):
    arr = np.asarray(point, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{v:.2f}" for v in arr[:3]) + "]"


def _point_delta(a, b):
    a_arr = np.asarray(a, dtype=np.float64).reshape(-1)[:3]
    b_arr = np.asarray(b, dtype=np.float64).reshape(-1)[:3]
    return float(np.linalg.norm(a_arr - b_arr))


def _fmt_optional(value, fmt):
    if value is None:
        return "None"
    return format(value, fmt)


def _fmt_optional_ms(value, fmt=".1f"):
    if value is None:
        return "None"
    return f"{format(value, fmt)}ms"


def _fmt_optional_m(value, fmt=".2f"):
    if value is None:
        return "None"
    return f"{format(value, fmt)}m"


def _fmt_optional_mbps(value, fmt=".2f"):
    if value is None:
        return "None"
    return f"{format(value, fmt)}Mbps"


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def _waypoint_segment_stats(points):
    if points is None:
        return {"count": 0, "span_m": None, "step_mean_m": None, "step_min_m": None, "step_max_m": None, "step_list": []}
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "span_m": None, "step_mean_m": None, "step_min_m": None, "step_max_m": None, "step_list": []}
    arr = arr.reshape(-1, 3)
    if len(arr) == 1:
        return {"count": 1, "span_m": 0.0, "step_mean_m": 0.0, "step_min_m": 0.0, "step_max_m": 0.0, "step_list": []}
    segments = np.linalg.norm(np.diff(arr[:, :3], axis=0), axis=1)
    return {
        "count": int(len(arr)),
        "span_m": float(np.linalg.norm(arr[-1, :3] - arr[0, :3])),
        "step_mean_m": float(segments.mean()),
        "step_min_m": float(segments.min()),
        "step_max_m": float(segments.max()),
        "step_list": segments.tolist(),
    }


def _trim_waypoints_to_current_pose(refined_waypoints, current_pose):
    waypoints = np.asarray(refined_waypoints, dtype=np.float64)
    if waypoints.size == 0:
        return [], None, 0
    waypoints = waypoints.reshape(-1, 3)
    pose = np.asarray(current_pose, dtype=np.float64).reshape(-1)[:3]
    distances = np.linalg.norm(waypoints[:, :3] - pose, axis=1)
    trim_index = int(np.argmin(distances))
    trimmed = waypoints[trim_index:].tolist()
    return trimmed, trim_index, int(len(trimmed))


def _print_trim_line(episode_idx, decision_step, trim_index, after_count):
    logger.info(
        f"[ep {episode_idx:04d} decision_step={decision_step} trim] "
        f"trim_idx={trim_index} remain={after_count}"
    )


def _print_trajectory_bundle(episode_idx, decision_step, llm_output, refined_waypoints, current_pose=None):
    coarse_summary = _fmt_point(llm_output[0]) if len(llm_output) > 0 else "[]"
    refined_summary = _waypoint_segment_stats(refined_waypoints)
    final_waypoint = _fmt_point(refined_waypoints[-1]) if len(refined_waypoints) > 0 else "[]"
    line = (
        f"[ep {episode_idx:04d} decision_step={decision_step} plan] "
        f"coarse_local={coarse_summary} "
    )
    if current_pose is not None:
        line += f" | uav_world={_fmt_point(current_pose)}"
    line += (
        f" | final_world={final_waypoint}"
        f" | wp_n={refined_summary['count']}"
        f" | span={_fmt_optional_m(refined_summary['span_m'])}"
    )
    logger.info(line)


def _print_exec_profile_line(episode_idx, step_idx, profile_info, current_chunk=None, current_pose=None):
    line = (
        f"[ep {episode_idx:04d} exec_step={step_idx}] "
        f"act={profile_info['airsim_action_latency_ms']:.1f}ms "
    )
    if current_pose is not None:
        line += f" | uav_world={_fmt_point(current_pose)}"
    if current_chunk is not None and len(current_chunk) > 0:
        line += (
            f" | exec_world={_fmt_point(current_chunk[0])}"
            f" | exec_dist={_fmt_optional_m(profile_info.get('exec_target_distance_m'))}"
        )
    logger.info(line)


def _print_decision_profile_line(episode_idx, decision_step, profile_info):
    logger.info(
        f"[ep {episode_idx:04d} decision_step={decision_step} timing] "
        f"obs={profile_info['obs_latency_ms']:.1f}ms "
        f"dino={profile_info['groundingdino_latency_ms']:.1f}ms "
        f"bw={_fmt_optional_mbps(profile_info.get('uplink_bandwidth_mbps'))} "
        f"uplink={_fmt_optional_ms(profile_info.get('uplink_latency_ms'))} "
        f"llm={_fmt_optional_ms(profile_info.get('llm_latency_ms'))} "
        f"traj={_fmt_optional_ms(profile_info.get('traj_latency_ms'))} "
        f"total={_fmt_optional_ms(profile_info.get('decision_latency_ms'))} "
        f"stop={int(profile_info.get('predict_done', False))} "
        f"stop_delay={_fmt_optional_ms(profile_info.get('stop_delay_ms'))} "
        f"stop_drift={_fmt_optional_m(profile_info.get('stop_drift_m'))}"
    )


def _build_planner_decision_record(
    env_batch,
    applied_result: PlannerResult,
    current_pose,
    enable_comm_delay: bool,
    chunk_waypoints: int,
    decision_step: int,
):
    decision_latency_ms = (
        float(applied_result.obs_latency_ms)
        + float(applied_result.groundingdino_latency_ms)
        + float(applied_result.uplink_latency_ms)
        + float(applied_result.llm_latency_ms)
        + float(applied_result.traj_latency_ms)
    )
    return {
        "record_type": "decision",
        "decision_step": int(decision_step),
        "batch_size": 1,
        "seq_names": [env_batch["seq_name"]],
        "map_names": [env_batch["map_name"]],
        "chunk_waypoints": int(chunk_waypoints),
        "comm_delay_enabled": bool(enable_comm_delay),
        "request_id": int(applied_result.request_id),
        "request_submitted_step": int(applied_result.submitted_step),
        "submitted_after_exec_step": int(applied_result.submitted_step),
        "request_observation_timestamp": int(applied_result.observation_timestamp),
        "request_ready_wall_time": float(applied_result.ready_wall_time),
        "obs_latency_ms": float(applied_result.obs_latency_ms),
        "groundingdino_latency_ms": float(applied_result.groundingdino_latency_ms),
        "uplink_payload_bits": int(applied_result.uplink_payload_bits),
        "uplink_payload_mb": float(applied_result.uplink_payload_mb),
        "uplink_bandwidth_mbps": float(applied_result.uplink_bandwidth_mbps),
        "uplink_latency_ms": float(applied_result.uplink_latency_ms),
        "llm_latency_ms": float(applied_result.llm_latency_ms),
        "traj_latency_ms": float(applied_result.traj_latency_ms),
        "decision_latency_ms": float(decision_latency_ms),
        "airsim_action_latency_ms": 0.0,
        "loop_latency_ms": float(decision_latency_ms),
        "action_age_ms": float((time.perf_counter() - applied_result.submitted_perf_time) * 1000.0),
        "state_drift_m": float(math.dist(applied_result.observation_pose, current_pose)),
        "trajectory_switch_applied": True,
        "hover_wait_ms": 0.0,
        "llm_output": copy.deepcopy(applied_result.llm_output),
        "refined_waypoints": copy.deepcopy(applied_result.refined_waypoints),
        "current_pose": copy.deepcopy(current_pose),
        "first_refined_gap_m": float(_point_delta(applied_result.refined_waypoints[0], current_pose)) if len(applied_result.refined_waypoints) > 0 else None,
        "predict_done": False,
        "stop_delay_ms": None,
        "stop_drift_m": None,
        "predict_dones": [False],
        "collisions": [False],
        "dones": [False],
    }


def _load_object_description():
    with open(args.object_name_json_path, "r") as handle:
        return {item["object_name"]: item["object_desc"] for item in json.load(handle)}


class ContinuousEpisodeState:
    def __init__(self, env_batch, eval_env: AirVLNENV, assist: Assist, ignore_tiny_diff: bool = False):
        self.eval_env = eval_env
        self.assist = assist
        self.ignore_tiny_diff = ignore_tiny_diff
        self.env_batch = env_batch
        self.target_position = env_batch["object_position"]
        self.traj = env_batch["trajectory"]
        self.ori_data_dir = env_batch["trajectory_dir"]
        self.object_info = _load_object_description().get(env_batch["object"]["asset_name"].replace("AA", ""))

        self.episode: List[Dict[str, Any]] = []
        self.dones = [False]
        self.collisions = [False]
        self.predict_dones = [False]
        self.success = False
        self.oracle_success = False
        self.early_end = False
        self.skip_saved = False
        self.distance_to_ends: List[float] = []
        self.last_observation_timestamp: Optional[int] = None

        outputs = self.eval_env.reset()
        self.process_env_output(outputs)

    def _check_collision_for_continuous(self, observations, dones, collisions):
        if not self.ignore_tiny_diff:
            return self.assist.check_collision_by_depth([self.episode], observations, collisions, dones)
        return check_collision_without_tiny_diff([self.episode], observations, collisions, dones)

    def _append_unique_observations(self, observations: List[Dict[str, Any]]) -> None:
        for observation in observations:
            ts = observation["sensors"]["state"]["timestamp"]
            if self.last_observation_timestamp is not None and ts <= self.last_observation_timestamp:
                continue
            self.episode.append(observation)
            self.last_observation_timestamp = ts
            self.distance_to_ends.append(self._calculate_distance(observation))

    def _calculate_distance(self, observation: Dict[str, Any]) -> float:
        return float(
            np.linalg.norm(
                np.asarray(observation["sensors"]["state"]["position"], dtype=np.float64)
                - np.asarray(self.target_position, dtype=np.float64)
            )
        )

    def current_sim_pose(self) -> List[float]:
        return copy.deepcopy(self.eval_env.sim_states[0].pose[0:3])

    def process_env_output(self, outputs) -> None:
        observations, dones, collisions, oracle_success = [list(x) for x in zip(*outputs)]
        collision_flags, done_flags = self._check_collision_for_continuous(observations, dones, collisions)
        self.dones = done_flags
        self.collisions = collision_flags
        self.oracle_success = bool(oracle_success[0])
        self._append_unique_observations(observations[0])
        if target_distance_increasing_for_10frames(self.distance_to_ends):
            self.collisions[0] = True
            self.dones[0] = True

    def build_snapshot(
        self,
        request_id: int,
        submitted_step: int,
        obs_latency_ms: float = 0.0,
        groundingdino_latency_ms: float = 0.0,
    ) -> Snapshot:
        latest = self.episode[-1]
        return Snapshot(
            request_id=request_id,
            submitted_step=submitted_step,
            episode=copy.deepcopy(self.episode),
            target_position=copy.deepcopy(self.target_position),
            object_info=self.object_info,
            assist_notice=self.assist.get_assist_notice(
                [self.episode],
                [self.traj],
                [self.object_info],
                [self.target_position],
            )[0],
            observation_timestamp=int(latest["sensors"]["state"]["timestamp"]),
            observation_pose=copy.deepcopy(latest["sensors"]["state"]["position"][0:3]),
            submitted_perf_time=time.perf_counter(),
            submitted_wall_time=time.time(),
            obs_latency_ms=float(obs_latency_ms),
            groundingdino_latency_ms=float(groundingdino_latency_ms),
        )

    def run_dino_and_update_metric(self, model_wrapper: ProfileTravelModelWrapper) -> float:
        dino_start = time.perf_counter()
        self.predict_dones = model_wrapper.predict_done([self.episode], [self.object_info])
        dino_latency_ms = (time.perf_counter() - dino_start) * 1000.0
        if not self.dones[0] and self.predict_dones[0]:
            current_distance = self.distance_to_ends[-1]
            if current_distance <= 20:
                self.success = True
                self.dones[0] = True
            elif current_distance > 20:
                self.early_end = True
        return dino_latency_ms

    def maybe_finalize(self, force=False) -> bool:
        if self.skip_saved:
            return True
        if force or self.dones[0]:
            prefix = ""
            if self.success:
                prefix = "success_"
            elif self.oracle_success:
                prefix = "oracle_"
            new_traj_name = prefix + self.ori_data_dir.split("/")[-1]
            new_traj_dir = os.path.join(args.eval_save_path, new_traj_name)
            save_to_dataset_eval(self.episode, new_traj_dir, self.ori_data_dir)
            self.skip_saved = True
        return self.skip_saved


def _run_decision_cycle(
    state: ContinuousEpisodeState,
    eval_env: AirVLNENV,
    model_wrapper: ProfileTravelModelWrapper,
    planner: "LatestOnlyEdgePlanner",
    request_counter: int,
    control_step: int,
):
    decision_obs_latency_ms = 0.0
    dino_latency_ms = 0.0
    dino_predicted_this_step = False
    pending_snapshot = None

    obs_start = time.perf_counter()
    outputs = eval_env.get_obs()
    obs_end = time.perf_counter()
    state.process_env_output(outputs)
    decision_obs_latency_ms = float((obs_end - obs_start) * 1000.0)

    if not state.dones[0]:
        dino_latency_ms = state.run_dino_and_update_metric(model_wrapper)
        dino_predicted_this_step = bool(state.predict_dones[0])

    if not state.dones[0]:
        request_counter += 1
        next_snapshot = state.build_snapshot(
            request_counter,
            control_step,
            obs_latency_ms=decision_obs_latency_ms,
            groundingdino_latency_ms=dino_latency_ms,
        )
        if planner.has_inflight():
            pending_snapshot = next_snapshot
        else:
            submitted = planner.submit(next_snapshot)
            if not submitted:
                pending_snapshot = next_snapshot

    return request_counter, pending_snapshot, decision_obs_latency_ms, dino_latency_ms, dino_predicted_this_step


class LatestOnlyEdgePlanner:
    def __init__(
        self,
        model_wrapper: ProfileTravelModelWrapper,
        bandwidth_trace: BandwidthTrace,
        enable_comm_delay: bool,
    ):
        self.model_wrapper = model_wrapper
        self.bandwidth_trace = bandwidth_trace
        self.enable_comm_delay = enable_comm_delay
        self._condition = threading.Condition()
        self._pending_snapshot: Optional[Snapshot] = None
        self._result: Optional[PlannerResult] = None
        self._has_result = False
        self._error: Optional[BaseException] = None
        self._inflight = False
        self._closed = False
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def submit(self, snapshot: Snapshot) -> bool:
        with self._condition:
            if self._closed or self._inflight or self._pending_snapshot is not None:
                return False
            self._pending_snapshot = snapshot
            self._condition.notify_all()
            return True

    def has_inflight(self) -> bool:
        with self._condition:
            return self._inflight or self._pending_snapshot is not None or self._has_result

    def poll_result(self) -> Optional[PlannerResult]:
        with self._condition:
            if self._error is not None:
                raise self._error
            if not self._has_result:
                return None
            result = self._result
            self._result = None
            self._has_result = False
            return result

    def wait_result(self) -> PlannerResult:
        with self._condition:
            if self._error is not None:
                raise self._error
            while not self._has_result:
                if self._error is not None:
                    raise self._error
                self._condition.wait(timeout=0.05)
            result = self._result
            self._result = None
            self._has_result = False
            return result

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join()

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while not self._closed and self._pending_snapshot is None:
                    self._condition.wait(timeout=0.05)
                if self._closed:
                    return
                snapshot = self._pending_snapshot
                self._pending_snapshot = None
                self._inflight = True

            try:
                result = self._run_snapshot(snapshot)
            except BaseException as exc:
                with self._condition:
                    self._error = exc
                    self._inflight = False
                    self._condition.notify_all()
                return

            with self._condition:
                self._result = result
                self._has_result = True
                self._inflight = False
                self._condition.notify_all()

    def _run_snapshot(self, snapshot: Snapshot) -> PlannerResult:
        episodes = [snapshot.episode]
        payload_bytes, payload_bits, payload_mb = estimate_uplink_payload_bits_from_episodes(episodes)
        bandwidth_bps = self.bandwidth_trace.next_bandwidth_bps()
        uplink_latency_ms = calculate_latency_ms(payload_bits, bandwidth_bps) if self.enable_comm_delay else 0.0
        if uplink_latency_ms > 0:
            time.sleep(uplink_latency_ms / 1000.0)

        with torch.inference_mode():
            inputs, rot_to_targets = self.model_wrapper.prepare_inputs(
                episodes,
                [snapshot.target_position],
                [snapshot.assist_notice],
            )
            refined_waypoints, profile_info = self.model_wrapper.run_profiled(
                inputs=inputs,
                episodes=episodes,
                rot_to_targets=rot_to_targets,
            )
        return PlannerResult(
            request_id=snapshot.request_id,
            submitted_step=snapshot.submitted_step,
            observation_timestamp=snapshot.observation_timestamp,
            observation_pose=copy.deepcopy(snapshot.observation_pose),
            submitted_perf_time=snapshot.submitted_perf_time,
            submitted_wall_time=snapshot.submitted_wall_time,
            ready_wall_time=time.time(),
            obs_latency_ms=float(snapshot.obs_latency_ms),
            groundingdino_latency_ms=float(snapshot.groundingdino_latency_ms),
            llm_latency_ms=float(profile_info["llm_latency_ms"]),
            traj_latency_ms=float(profile_info["traj_latency_ms"]),
            uplink_payload_bits=int(payload_bits),
            uplink_payload_mb=float(payload_mb),
            uplink_bandwidth_mbps=float(bandwidth_bps / 1_000_000.0),
            uplink_latency_ms=float(uplink_latency_ms),
            llm_output=copy.deepcopy(profile_info["llm_output"]),
            refined_waypoints=np.asarray(refined_waypoints[0]).tolist(),
        )


def eval(
    model_wrapper: ProfileTravelModelWrapper,
    assist: Assist,
    eval_env: AirVLNENV,
    eval_save_dir,
    profile_log_path,
    summary_path,
    bandwidth_trace: BandwidthTrace,
    enable_comm_delay: bool,
    chunk_waypoints: int,
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
                    planner = LatestOnlyEdgePlanner(model_wrapper, bandwidth_trace, enable_comm_delay)
                    try:
                        state = ContinuousEpisodeState(
                            env_batchs[0],
                            eval_env,
                            assist,
                            ignore_tiny_diff=True,
                        )
                        request_counter = 0
                        control_step = 0
                        executed_since_decision = 0
                        active_traj: List[List[float]] = []
                        active_index = 0
                        pending_snapshot: Optional[Snapshot] = None

                        warmup_snapshot = state.build_snapshot(request_counter, control_step)
                        submitted = planner.submit(warmup_snapshot)
                        assert submitted, "failed to submit warmup snapshot"
                        warmup_hover_start = time.perf_counter()
                        warmup_result = planner.wait_result()
                        warmup_hover_ms = (time.perf_counter() - warmup_hover_start) * 1000.0
                        current_pose = state.current_sim_pose()
                        active_traj, trim_index, trim_after = _trim_waypoints_to_current_pose(
                            warmup_result.refined_waypoints,
                            current_pose,
                        )
                        active_index = 0
                        warmup_record = _build_planner_decision_record(
                            env_batchs[0],
                            warmup_result,
                            current_pose,
                            enable_comm_delay,
                            chunk_waypoints,
                            decision_step=0,
                        )
                        warmup_record["hover_wait_ms"] = float(warmup_hover_ms)
                        warmup_record["loop_latency_ms"] = float(warmup_hover_ms)
                        warmup_record["predict_dones"] = [bool(x) for x in state.predict_dones]
                        warmup_record["collisions"] = [bool(x) for x in state.collisions]
                        warmup_record["dones"] = [bool(x) for x in state.dones]
                        warmup_record["buffer_trim_index"] = trim_index
                        warmup_record["buffer_waypoints_after"] = trim_after
                        _write_jsonl_line(profile_fp, warmup_record)
                        summary_records.append(warmup_record)
                        _print_decision_profile_line(episode_idx, warmup_result.request_id, warmup_record)
                        _print_trim_line(
                            episode_idx,
                            warmup_result.request_id,
                            trim_index,
                            trim_after,
                        )
                        _print_trajectory_bundle(
                            episode_idx,
                            warmup_result.request_id,
                            warmup_result.llm_output,
                            warmup_result.refined_waypoints,
                            current_pose=current_pose,
                        )

                        while not state.dones[0] and control_step < int(args.maxWaypoints):
                            applied_result = planner.poll_result()
                            dino_latency_ms = 0.0
                            trajectory_switch_applied = False
                            trim_index = None
                            trim_after = None
                            hover_wait_ms = 0.0
                            dino_predicted_this_step = False

                            if applied_result is not None:
                                current_pose = state.current_sim_pose()
                                active_traj, trim_index, trim_after = _trim_waypoints_to_current_pose(
                                    applied_result.refined_waypoints,
                                    current_pose,
                                )
                                active_index = 0
                                trajectory_switch_applied = True
                                if pending_snapshot is not None and not planner.has_inflight():
                                    planner.submit(pending_snapshot)
                                    pending_snapshot = None
                                decision_record = _build_planner_decision_record(
                                    env_batchs[0],
                                    applied_result,
                                    current_pose,
                                    enable_comm_delay,
                                    chunk_waypoints,
                                    decision_step=int(applied_result.request_id),
                                )
                                decision_record["predict_dones"] = [bool(x) for x in state.predict_dones]
                                decision_record["collisions"] = [bool(x) for x in state.collisions]
                                decision_record["dones"] = [bool(x) for x in state.dones]
                                decision_record["buffer_trim_index"] = trim_index
                                decision_record["buffer_waypoints_after"] = trim_after
                                _write_jsonl_line(profile_fp, decision_record)
                                summary_records.append(decision_record)
                                _print_decision_profile_line(episode_idx, applied_result.request_id, decision_record)
                                _print_trim_line(
                                    episode_idx,
                                    applied_result.request_id,
                                    trim_index,
                                    trim_after,
                                )
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

                                active_index += chunk_size
                                control_step += 1
                                executed_since_decision += chunk_size

                                decision_obs_latency_ms = 0.0
                                decision_observation_pose = None
                                should_request_decision = (
                                    active_index >= len(active_traj)
                                    or executed_since_decision >= chunk_waypoints
                                )
                                if should_request_decision:
                                    request_counter, pending_snapshot, decision_obs_latency_ms, dino_latency_ms, dino_predicted_this_step = _run_decision_cycle(
                                        state,
                                        eval_env,
                                        model_wrapper,
                                        planner,
                                        request_counter,
                                        control_step,
                                    )
                                    decision_observation_pose = state.current_sim_pose()
                                    executed_since_decision = 0

                                action_latency_ms = float((action_end - action_start) * 1000.0)
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
                                    "buffer_trim_index": trim_index if trajectory_switch_applied else None,
                                    "buffer_waypoints_after": trim_after if trajectory_switch_applied else None,
                                    "current_pose": state.current_sim_pose(),
                                    "first_refined_gap_m": float(_point_delta(active_traj[0], state.current_sim_pose())) if len(active_traj) > 0 else None,
                                    "exec_target_distance_m": exec_target_distance_m,
                                    "predict_dones": [bool(x) for x in state.predict_dones],
                                    "collisions": [bool(x) for x in state.collisions],
                                    "dones": [bool(x) for x in state.dones],
                                }
                                _write_jsonl_line(profile_fp, record)
                                summary_records.append(record)
                                _print_exec_profile_line(
                                    episode_idx,
                                    control_step,
                                    record,
                                    current_chunk=current_chunk,
                                    current_pose=exec_pose,
                                )
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
                                if not state.dones[0]:
                                    request_counter, pending_snapshot, _, _, _ = _run_decision_cycle(
                                        state,
                                        eval_env,
                                        model_wrapper,
                                        planner,
                                        request_counter,
                                        control_step,
                                    )
                                    executed_since_decision = 0

                                if state.dones[0]:
                                    state.maybe_finalize()
                                    break

                                if pending_snapshot is not None and not planner.has_inflight():
                                    planner.submit(pending_snapshot)
                                    pending_snapshot = None

                                hover_wait_start = time.perf_counter()
                                applied_result = planner.wait_result()
                                hover_wait_ms = (time.perf_counter() - hover_wait_start) * 1000.0
                                current_pose = state.current_sim_pose()
                                active_traj, trim_index, trim_after = _trim_waypoints_to_current_pose(
                                    applied_result.refined_waypoints,
                                    current_pose,
                                )
                                active_index = 0
                                record = _build_planner_decision_record(
                                    env_batchs[0],
                                    applied_result,
                                    current_pose,
                                    enable_comm_delay,
                                    chunk_waypoints,
                                    decision_step=int(applied_result.request_id),
                                )
                                record["hover_wait_ms"] = float(hover_wait_ms)
                                record["loop_latency_ms"] = float(hover_wait_ms)
                                record["predict_dones"] = [bool(x) for x in state.predict_dones]
                                record["collisions"] = [bool(x) for x in state.collisions]
                                record["dones"] = [bool(x) for x in state.dones]
                                record["buffer_trim_index"] = trim_index
                                record["buffer_waypoints_after"] = trim_after
                                _write_jsonl_line(profile_fp, record)
                                summary_records.append(record)
                                _print_decision_profile_line(episode_idx, applied_result.request_id, record)
                                _print_trim_line(
                                    episode_idx,
                                    applied_result.request_id,
                                    trim_index,
                                    trim_after,
                                )
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
                        episode_ok = True
                        break
                    except Exception as e:
                        logger.error(f"Episode failed: {e}, retry {retry_i + 1}/3")
                        if retry_i < 2:
                            logger.error("Restarting scene...")
                            eval_env._changeEnv(need_change=True)
                    finally:
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
        "num_execution_records": len(execution_records),
        "num_decision_records": len(decision_records),
        "num_records": len(summary_records),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    _set_console_log_message_only()
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
    bandwidth_trace = BandwidthTrace(trace_path, cycle=True)
    logger.info(
        f"Loaded bandwidth trace: {trace_path} ({bandwidth_trace.sample_count} samples), "
        f"enable_comm_delay={enable_comm_delay}, chunk_waypoints={chunk_waypoints}"
    )

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

    profile_log_path = os.path.join(profile_log_dir, f"pro_con_trim_w{chunk_waypoints}_{args.make_dir_time}.jsonl")
    summary_path = os.path.join(profile_log_dir, f"pro_con_trim_w{chunk_waypoints}_{args.make_dir_time}_summary.json")

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
    )

    eval_env.delete_VectorEnvUtil()
