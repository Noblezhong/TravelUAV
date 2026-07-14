import copy
import json
import math
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
from env_uav import AirVLNENV
from src.common.param import args, data_args, model_args
from src.model_wrapper.edge_traj_dnn import EdgeDNNModelWrapper
from src.vlnce_src.closeloop_util import (
    BatchIterator,
    CheckPort,
    initialize_env_eval,
    is_dist_avail_and_initialized,
    save_to_dataset_eval,
    setup,
    target_distance_increasing_for_10frames,
)
from src.vlnce_src.continue_eval import _as_bool, _fmt_point, _point_delta, _set_console_log_message_only
from src.vlnce_src.continue_eval import check_collision_without_tiny_diff
from src.vlnce_src.comm_delay import (
    BandwidthTrace,
    calculate_latency_ms,
    default_trace_path,
    estimate_uplink_payload_bits_from_episodes,
)
from src.vlnce_src.edge_vlm_rpc import request as rpc_request
from src.vlnce_src.fast_eval_time import (
    FastEvalClock,
    FastResultTiming,
    action_timing,
    configure_fast_eval_output,
)
from utils.logger import logger
from utils.utils import *


@dataclass
class EdgeSnapshot:
    request_id: int
    submitted_step: int
    episode: List[Dict[str, Any]]
    target_position: List[float]
    object_info: str
    assist_notice: Optional[str]
    observation_timestamp: int
    observation_pose: List[float]
    observation_rotation: List[List[float]]
    submitted_perf_time: float
    submitted_wall_time: float
    submitted_logical_ms: Optional[float] = None
    payload_bits: int = 0
    payload_mb: float = 0.0


@dataclass
class EdgeCoarseResult:
    request_id: int
    submitted_step: int
    observation_timestamp: int
    observation_pose: List[float]
    observation_rotation: List[List[float]]
    submitted_perf_time: float
    submitted_wall_time: float
    ready_wall_time: float
    submitted_logical_ms: Optional[float]
    edge_arrival_logical_ms: Optional[float]
    ready_logical_ms: Optional[float]
    applied_logical_ms: Optional[float]
    coarse_local: Optional[List[float]]
    coarse_goal_world: Optional[List[float]]
    coarse_goal_world_source: str
    legacy_body_goal_world: Optional[List[float]]
    edge_llm_latency_ms: float
    edge_compute_latency_ms: float
    uplink_payload_bits: int
    uplink_payload_mb: float
    uplink_bandwidth_mbps: float
    uplink_latency_ms: float
    predict_done: bool
    dino_distance_to_target_m: float


def _write_jsonl_line(handle, payload):
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


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


def _coarse_goal_world(observation_pose, observation_rotation, coarse_local):
    pose = np.asarray(observation_pose, dtype=np.float64).reshape(3)
    rot = np.asarray(observation_rotation, dtype=np.float64).reshape(3, 3)
    coarse = np.asarray(coarse_local, dtype=np.float64).reshape(3)
    return (pose + rot @ coarse).tolist()


def _has_executable_waypoints(refined_waypoints) -> bool:
    arr = np.asarray(refined_waypoints, dtype=np.float64)
    if arr.size == 0 or arr.size % 3 != 0:
        return False
    arr = arr.reshape(-1, 3)
    if len(arr) < 1 or not np.all(np.isfinite(arr)):
        return False
    return True


def _compact_episode_for_edge(episode: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the history features but send images only for the latest frame."""
    compact = []
    last_index = len(episode) - 1
    for index, observation in enumerate(episode):
        item = {
            "instruction": observation.get("instruction", ""),
            "sensors": copy.deepcopy(observation.get("sensors", {})),
            # Presence of this key preserves the original history_waypoint
            # construction; old frames intentionally carry no image data.
            "rgb": copy.deepcopy(observation.get("rgb", [])) if index == last_index else [],
            "rgb_record": (
                copy.deepcopy(observation.get("rgb_record", [])) if index == last_index else []
            ),
            "depth_record": (
                copy.deepcopy(observation.get("depth_record", [])) if index == last_index else []
            ),
        }
        compact.append(item)
    return compact


class EdgeDNNContinuousState:
    def __init__(self, env_batch, eval_env: AirVLNENV, assist: Assist, ignore_tiny_diff: bool = True):
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
        if self.ignore_tiny_diff:
            collision_flags, done_flags = check_collision_without_tiny_diff([self.episode], observations, collisions, dones)
        else:
            collision_flags, done_flags = self.assist.check_collision_by_depth([self.episode], observations, collisions, dones)
        self.collisions = collision_flags
        self.dones = done_flags
        self.oracle_success = bool(oracle_success[0])
        self._append_unique_observations(observations[0])
        if target_distance_increasing_for_10frames(self.distance_to_ends):
            self.collisions[0] = True
            self.dones[0] = True

    def refresh_observation(self) -> float:
        start = time.perf_counter()
        outputs = self.eval_env.get_obs()
        latency_ms = (time.perf_counter() - start) * 1000.0
        self.process_env_output(outputs)
        return float(latency_ms)

    def apply_edge_stop_result(self, predict_done: bool, distance_to_target_m: float) -> None:
        self.predict_dones = [bool(predict_done)]
        if not self.dones[0] and predict_done:
            if distance_to_target_m <= 20 and not self.early_end:
                self.success = True
                self.dones[0] = True
            elif distance_to_target_m > 20:
                self.early_end = True


    def build_snapshot(self, request_id: int, submitted_step: int) -> EdgeSnapshot:
        latest = self.episode[-1]
        # Preserve the existing communication-delay accounting while keeping
        # large historical image arrays out of the cross-process snapshot.
        _, payload_bits, payload_mb = estimate_uplink_payload_bits_from_episodes([self.episode])
        return EdgeSnapshot(
            request_id=request_id,
            submitted_step=submitted_step,
            episode=_compact_episode_for_edge(self.episode),
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
            observation_rotation=copy.deepcopy(latest["sensors"]["imu"]["rotation"]),
            submitted_perf_time=time.perf_counter(),
            submitted_wall_time=time.time(),
            payload_bits=int(payload_bits),
            payload_mb=float(payload_mb),
        )

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


def _action_delay_ms(result: EdgeCoarseResult, clock: FastEvalClock) -> float:
    return clock.age_ms(result.submitted_logical_ms, result.submitted_perf_time)


def _mark_result_applied(result: EdgeCoarseResult, clock: FastEvalClock) -> EdgeCoarseResult:
    result.applied_logical_ms = clock.now_ms if clock.enabled else None
    return result


def _snapshot_with_clock(state: EdgeDNNContinuousState, request_id: int, exec_step: int, clock: FastEvalClock):
    snapshot = state.build_snapshot(request_id, exec_step)
    snapshot.submitted_logical_ms = clock.now_ms if clock.enabled else None
    return snapshot


class LatestOnlyEdgeVLMClient:
    def __init__(
        self,
        host: str,
        port: int,
        bandwidth_trace: BandwidthTrace,
        enable_comm_delay: bool,
        clock: Optional[FastEvalClock] = None,
    ):
        self.host = host
        self.port = int(port)
        self.bandwidth_trace = bandwidth_trace
        self.enable_comm_delay = bool(enable_comm_delay)
        self.clock = clock or FastEvalClock(False)
        self.fast_eval = bool(self.clock.enabled)
        self._condition = threading.Condition()
        self._pending_snapshot: Optional[EdgeSnapshot] = None
        self._result: Optional[EdgeCoarseResult] = None
        self._has_result = False
        self._error: Optional[BaseException] = None
        self._inflight = False
        self._closed = False
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def submit(self, snapshot: EdgeSnapshot) -> bool:
        with self._condition:
            if self._closed or self._inflight or self._pending_snapshot is not None:
                return False
            self._pending_snapshot = snapshot
            self._condition.notify_all()
            return True

    def has_inflight(self) -> bool:
        with self._condition:
            return self._inflight or self._pending_snapshot is not None or self._has_result

    def poll_result(self) -> Optional[EdgeCoarseResult]:
        with self._condition:
            if self._error is not None:
                raise self._error
            if not self._has_result:
                return None
            if (
                self.fast_eval
                and self._result is not None
                and self._result.ready_logical_ms is not None
                and self.clock.now_ms < self._result.ready_logical_ms
            ):
                return None
            result = self._result
            self._result = None
            self._has_result = False
            return result

    def wait_result(self, timeout_s: float = 180.0) -> EdgeCoarseResult:
        deadline = time.monotonic() + float(timeout_s)
        with self._condition:
            if self._error is not None:
                raise self._error
            while not self._has_result:
                if self._error is not None:
                    raise self._error
                if not self._inflight and self._pending_snapshot is None:
                    raise RuntimeError("cannot wait for edge result: no request is in flight")
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0:
                    raise TimeoutError(f"edge VLM request timed out after {float(timeout_s):.1f}s")
                self._condition.wait(timeout=min(0.05, remaining_s))
            result = self._result
            self._result = None
            self._has_result = False
            if self.fast_eval:
                self.clock.advance_to(result.ready_logical_ms)
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

    def _run_snapshot(self, snapshot: EdgeSnapshot) -> EdgeCoarseResult:
        payload_bits = int(snapshot.payload_bits)
        payload_mb = float(snapshot.payload_mb)
        if payload_bits <= 0:
            _, payload_bits, payload_mb = estimate_uplink_payload_bits_from_episodes([snapshot.episode])
        bandwidth_bps = self.bandwidth_trace.next_bandwidth_bps()
        uplink_latency_ms = calculate_latency_ms(payload_bits, bandwidth_bps) if self.enable_comm_delay else 0.0
        if uplink_latency_ms > 0:
            divisor = self.clock.speedup if self.fast_eval else 1.0
            time.sleep(uplink_latency_ms / (1000.0 * divisor))

        payload = {
            "request_id": snapshot.request_id,
            "episode": snapshot.episode,
            "target_position": snapshot.target_position,
            "object_info": snapshot.object_info,
            "assist_notice": snapshot.assist_notice,
            "observation_pose": snapshot.observation_pose,
            "observation_rotation": snapshot.observation_rotation,
        }
        edge_start = time.perf_counter()
        response = rpc_request(self.host, self.port, payload)
        edge_compute_latency_ms = (time.perf_counter() - edge_start) * 1000.0
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "edge VLM request failed"))
        coarse_local = response.get("coarse_local")
        coarse_goal = response.get("coarse_goal_world")
        coarse_goal_source = "edge_travel_frame" if coarse_goal is not None else "none"
        legacy_body_goal = None
        if coarse_local is not None:
            legacy_body_goal = _coarse_goal_world(snapshot.observation_pose, snapshot.observation_rotation, coarse_local)
            if coarse_goal is None:
                raise RuntimeError(
                    "edge VLM response has coarse_local but no coarse_goal_world; "
                    "restart edge_vlm_server with the updated TravelUAV-frame code"
                )
            else:
                coarse_goal = np.asarray(coarse_goal, dtype=np.float64).reshape(3).tolist()
        fast_timing = FastResultTiming(
            snapshot.submitted_logical_ms,
            float(uplink_latency_ms),
            float(edge_compute_latency_ms),
            0.0,
        )
        return EdgeCoarseResult(
            request_id=snapshot.request_id,
            submitted_step=snapshot.submitted_step,
            observation_timestamp=snapshot.observation_timestamp,
            observation_pose=copy.deepcopy(snapshot.observation_pose),
            observation_rotation=copy.deepcopy(snapshot.observation_rotation),
            submitted_perf_time=snapshot.submitted_perf_time,
            submitted_wall_time=snapshot.submitted_wall_time,
            ready_wall_time=time.time(),
            submitted_logical_ms=snapshot.submitted_logical_ms,
            edge_arrival_logical_ms=fast_timing.edge_arrival_logical_ms,
            ready_logical_ms=fast_timing.ready_logical_ms,
            applied_logical_ms=None,
            coarse_local=copy.deepcopy(coarse_local),
            coarse_goal_world=coarse_goal,
            coarse_goal_world_source=coarse_goal_source,
            legacy_body_goal_world=legacy_body_goal,
            edge_llm_latency_ms=float(response["llm_latency_ms"]),
            edge_compute_latency_ms=float(edge_compute_latency_ms),
            uplink_payload_bits=int(payload_bits),
            uplink_payload_mb=float(payload_mb),
            uplink_bandwidth_mbps=float(bandwidth_bps / 1_000_000.0),
            uplink_latency_ms=float(uplink_latency_ms),
            predict_done=bool(response.get("predict_done", False)),
            dino_distance_to_target_m=float(response.get("dino_distance_to_target_m", 1e9)),
        )


def _load_object_description():
    with open(args.object_name_json_path, "r") as handle:
        return {item["object_name"]: item["object_desc"] for item in json.load(handle)}


def _configure_local_air_sim_server():
    args.machines_info[0]["MACHINE_IP"] = "127.0.0.1"


def _print_episode_header(episode_idx, env_batch, chunk_waypoints):
    logger.info(
        f"\n[edge-dnn episode {episode_idx:04d}] seq={env_batch['seq_name']} "
        f"map={env_batch['map_name']} chunk={int(chunk_waypoints)}"
    )


def _print_edge_result(episode_idx, result: EdgeCoarseResult, action_delay_ms, state_drift_m):
    coarse_text = "None" if result.coarse_local is None else _fmt_point(result.coarse_local)
    goal_text = "None" if result.coarse_goal_world is None else _fmt_point(result.coarse_goal_world)
    legacy_goal_text = "None" if result.legacy_body_goal_world is None else _fmt_point(result.legacy_body_goal_world)
    logger.info(
        f"[ep {episode_idx:04d} edge request={result.request_id}] "
        f"uplink={result.uplink_latency_ms:.1f}ms bw={result.uplink_bandwidth_mbps:.2f}Mbps "
        f"llm={result.edge_llm_latency_ms:.1f}ms "
        f"delay={action_delay_ms:.1f}ms drift={state_drift_m:.2f}m "
        f"predict_done={result.predict_done} target_dist={result.dino_distance_to_target_m:.2f}m "
        f"coarse={coarse_text} goal={goal_text} legacy_goal={legacy_goal_text}"
    )


def _print_exec(episode_idx, exec_step, record):
    dnn_latency = record.get("jetson_dnn_latency_ms")
    dnn_text = "buffer" if dnn_latency is None else f"{dnn_latency:.1f}ms"
    logger.info(
        f"[ep {episode_idx:04d} exec_step={exec_step}] "
        f"dnn={dnn_text} "
        f"act={record['airsim_action_latency_ms']:.1f}ms "
        f"reproj={_fmt_point(record['reprojected_coarse'])}"
    )


def eval(
    model_wrapper: EdgeDNNModelWrapper,
    assist: Assist,
    eval_env: AirVLNENV,
    profile_log_path,
    summary_path,
    bandwidth_trace: BandwidthTrace,
    enable_comm_delay: bool,
    chunk_waypoints: int,
):
    assert int(eval_env.batch_size) == 1, "edge DNN eval currently supports batchSize=1 only"
    model_wrapper.eval()
    summary_records = []

    with torch.no_grad():
        dataset = BatchIterator(eval_env)
        pbar = tqdm.tqdm(total=len(dataset))
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
                    episode_clock = FastEvalClock(bool(args.fast_eval), args.fast_eval_speedup)
                    edge_client = LatestOnlyEdgeVLMClient(
                        args.edge_vlm_host,
                        args.edge_vlm_port,
                        bandwidth_trace,
                        enable_comm_delay,
                        clock=episode_clock,
                    )
                    try:
                        state = EdgeDNNContinuousState(env_batchs[0], eval_env, assist)
                        request_counter = 0
                        exec_step = 0
                        executed_since_request = 0
                        pending_snapshot: Optional[EdgeSnapshot] = None
                        active_coarse_result: Optional[EdgeCoarseResult] = None
                        active_traj: List[List[float]] = []
                        active_index = 0
                        active_dnn_profile: Optional[Dict[str, Any]] = None

                        warmup_snapshot = _snapshot_with_clock(state, request_counter, exec_step, episode_clock)
                        assert edge_client.submit(warmup_snapshot), "failed to submit warmup VLM request"
                        active_coarse_result = _mark_result_applied(edge_client.wait_result(), episode_clock)
                        action_delay_ms = _action_delay_ms(active_coarse_result, episode_clock)
                        state_drift_m = math.dist(active_coarse_result.observation_pose, state.current_sim_pose())
                        state.apply_edge_stop_result(
                            active_coarse_result.predict_done,
                            active_coarse_result.dino_distance_to_target_m,
                        )
                        _print_edge_result(episode_idx, active_coarse_result, action_delay_ms, state_drift_m)

                        while not state.dones[0] and exec_step < int(args.maxWaypoints):
                            new_result = edge_client.poll_result()
                            if new_result is not None:
                                active_coarse_result = _mark_result_applied(new_result, episode_clock)
                                active_traj = []
                                active_index = 0
                                active_dnn_profile = None
                                action_delay_ms = _action_delay_ms(active_coarse_result, episode_clock)
                                state_drift_m = math.dist(active_coarse_result.observation_pose, state.current_sim_pose())
                                state.apply_edge_stop_result(
                                    active_coarse_result.predict_done,
                                    active_coarse_result.dino_distance_to_target_m,
                                )
                                _print_edge_result(episode_idx, active_coarse_result, action_delay_ms, state_drift_m)
                                if state.dones[0]:
                                    break
                                if pending_snapshot is not None and not edge_client.has_inflight():
                                    edge_client.submit(pending_snapshot)
                                    pending_snapshot = None

                            if active_coarse_result is None:
                                active_coarse_result = _mark_result_applied(edge_client.wait_result(), episode_clock)
                                active_traj = []
                                active_index = 0
                                active_dnn_profile = None
                                action_delay_ms = _action_delay_ms(active_coarse_result, episode_clock)
                                state_drift_m = math.dist(active_coarse_result.observation_pose, state.current_sim_pose())
                                state.apply_edge_stop_result(
                                    active_coarse_result.predict_done,
                                    active_coarse_result.dino_distance_to_target_m,
                                )
                                _print_edge_result(episode_idx, active_coarse_result, action_delay_ms, state_drift_m)
                                if state.dones[0]:
                                    break
                                continue

                            current_pose = state.current_sim_pose()
                            if active_coarse_result.coarse_goal_world is None:
                                raise RuntimeError(
                                    "edge VLM returned no coarse goal without terminating the episode: "
                                    f"request_id={active_coarse_result.request_id}, "
                                    f"predict_done={active_coarse_result.predict_done}, "
                                    f"distance_to_target_m={active_coarse_result.dino_distance_to_target_m:.2f}"
                                )
                            coarse_goal_distance = _point_delta(active_coarse_result.coarse_goal_world, current_pose)

                            dnn_ran_this_step = False
                            if active_index >= len(active_traj):
                                if active_dnn_profile is not None:
                                    if not state.dones[0] and not edge_client.has_inflight():
                                        if pending_snapshot is not None:
                                            edge_client.submit(pending_snapshot)
                                            pending_snapshot = None
                                        else:
                                            request_counter += 1
                                            edge_client.submit(
                                                _snapshot_with_clock(state, request_counter, exec_step, episode_clock)
                                            )
                                    active_coarse_result = None
                                    active_traj = []
                                    active_index = 0
                                    active_dnn_profile = None
                                    continue
                                refined_waypoints, dnn_profile = model_wrapper.run_traj_from_world_goal(
                                    [state.episode],
                                    [active_coarse_result.coarse_goal_world],
                                )
                                if episode_clock.enabled:
                                    episode_clock.advance_blocking(float(dnn_profile.get("traj_latency_ms", 0.0)))
                                refined_current = np.asarray(refined_waypoints[0]).tolist()
                                if not _has_executable_waypoints(refined_current):
                                    state.dones[0] = True
                                    record = {
                                        "record_type": "dnn_invalid",
                                        "exec_step": int(exec_step),
                                        "request_id": int(active_coarse_result.request_id),
                                        "seq_names": [env_batchs[0]["seq_name"]],
                                        "map_names": [env_batchs[0]["map_name"]],
                                        "coarse_local": copy.deepcopy(active_coarse_result.coarse_local),
                                        "coarse_goal_world": copy.deepcopy(active_coarse_result.coarse_goal_world),
                                        "coarse_goal_world_source": active_coarse_result.coarse_goal_world_source,
                                        "legacy_body_goal_world": copy.deepcopy(
                                            active_coarse_result.legacy_body_goal_world
                                        ),
                                        "edge_compute_latency_ms": float(active_coarse_result.edge_compute_latency_ms),
                                        "fast_eval": bool(episode_clock.enabled),
                                        "fast_eval_speedup": float(
                                            episode_clock.speedup if episode_clock.enabled else 1.0
                                        ),
                                        "wall_elapsed_ms": float(episode_clock.wall_elapsed_ms),
                                        "logical_elapsed_ms": float(episode_clock.now_ms),
                                        "result_ready_logical_ms": active_coarse_result.ready_logical_ms,
                                        "result_applied_logical_ms": active_coarse_result.applied_logical_ms,
                                        "reprojected_coarse": dnn_profile["reprojected_coarse"][0],
                                        "refined_waypoints": copy.deepcopy(refined_current),
                                        "dones": [bool(x) for x in state.dones],
                                        "collisions": [bool(x) for x in state.collisions],
                                        "success": bool(state.success),
                                    }
                                    _write_jsonl_line(profile_fp, record)
                                    summary_records.append(record)
                                    continue
                                active_traj = refined_current
                                active_index = 0
                                active_dnn_profile = dnn_profile
                                dnn_ran_this_step = True

                            current_chunk = active_traj[active_index : active_index + 1]
                            if len(current_chunk) == 0:
                                active_coarse_result = None
                                active_dnn_profile = None
                                continue
                            action_start = time.perf_counter()
                            eval_env.makeActionsChunk([current_chunk], target_idx=1)
                            measured_action_wall_ms = (time.perf_counter() - action_start) * 1000.0
                            action_times = action_timing(
                                eval_env,
                                measured_action_wall_ms,
                                bool(episode_clock.enabled),
                            )
                            episode_clock.advance_action(
                                action_times["action_sim_time_ms"],
                                action_times["action_wall_time_ms"],
                            )
                            active_index += 1
                            exec_step += 1
                            executed_since_request += 1

                            state.refresh_observation()
                            if executed_since_request >= chunk_waypoints:
                                if not state.dones[0]:
                                    request_counter += 1
                                    snapshot = _snapshot_with_clock(state, request_counter, exec_step, episode_clock)
                                    if edge_client.has_inflight():
                                        pending_snapshot = snapshot
                                    else:
                                        edge_client.submit(snapshot)
                                executed_since_request = 0

                            action_delay_ms = _action_delay_ms(active_coarse_result, episode_clock)
                            state_drift_m = math.dist(active_coarse_result.observation_pose, state.current_sim_pose())
                            record = {
                                "record_type": "execution",
                                "exec_step": int(exec_step),
                                "chunk_waypoints": int(chunk_waypoints),
                                "request_id": int(active_coarse_result.request_id),
                                "seq_names": [env_batchs[0]["seq_name"]],
                                "map_names": [env_batchs[0]["map_name"]],
                                "comm_delay_enabled": bool(enable_comm_delay),
                                "uplink_payload_bits": int(active_coarse_result.uplink_payload_bits),
                                "uplink_payload_mb": float(active_coarse_result.uplink_payload_mb),
                                "uplink_bandwidth_mbps": float(active_coarse_result.uplink_bandwidth_mbps),
                                "uplink_latency_ms": float(active_coarse_result.uplink_latency_ms),
                                "edge_llm_latency_ms": float(active_coarse_result.edge_llm_latency_ms),
                                "edge_compute_latency_ms": float(active_coarse_result.edge_compute_latency_ms),
                                "jetson_dnn_latency_ms": (
                                    float(active_dnn_profile["traj_latency_ms"]) if dnn_ran_this_step else None
                                ),
                                "airsim_action_latency_ms": float(action_times["airsim_action_latency_ms"]),
                                "action_wall_time_ms": float(action_times["action_wall_time_ms"]),
                                "action_sim_time_ms": float(action_times["action_sim_time_ms"]),
                                "action_delay_ms": float(action_delay_ms),
                                "state_drift_m": float(state_drift_m),
                                "coarse_goal_distance_m": float(coarse_goal_distance),
                                "fast_eval": bool(episode_clock.enabled),
                                "fast_eval_speedup": float(episode_clock.speedup if episode_clock.enabled else 1.0),
                                "wall_elapsed_ms": float(episode_clock.wall_elapsed_ms),
                                "logical_elapsed_ms": float(episode_clock.now_ms),
                                "result_ready_logical_ms": active_coarse_result.ready_logical_ms,
                                "result_applied_logical_ms": active_coarse_result.applied_logical_ms,
                                "coarse_local": copy.deepcopy(active_coarse_result.coarse_local),
                                "coarse_goal_world": copy.deepcopy(active_coarse_result.coarse_goal_world),
                                "coarse_goal_world_source": active_coarse_result.coarse_goal_world_source,
                                "legacy_body_goal_world": copy.deepcopy(active_coarse_result.legacy_body_goal_world),
                                "reprojected_coarse": copy.deepcopy(active_dnn_profile["reprojected_coarse"][0]),
                                "refined_waypoints": copy.deepcopy(active_traj),
                                "executed_waypoint": copy.deepcopy(current_chunk[0]),
                                "success": bool(state.success),
                                "collision": bool(state.collisions[0]),
                                "done": bool(state.dones[0]),
                            }
                            _write_jsonl_line(profile_fp, record)
                            summary_records.append(record)
                            _print_exec(episode_idx, exec_step, record)

                            if state.dones[0]:
                                break

                        if not state.skip_saved:
                            state.maybe_finalize(force=True)
                        episode_ok = True
                        break
                    except Exception as exc:
                        logger.error(f"Episode failed: {exc}, retry {retry_i + 1}/3")
                        if retry_i < 2:
                            logger.error("Restarting scene...")
                            eval_env._changeEnv(need_change=True)
                    finally:
                        edge_client.close()

                if not episode_ok:
                    raise RuntimeError("episode failed after 3 retries")
                episode_idx += 1

        pbar.close()

    execution_records = [item for item in summary_records if item.get("record_type") == "execution"]
    summary = {
        "uplink_latency_ms": _metric_summary([item.get("uplink_latency_ms") for item in execution_records]),
        "edge_llm_latency_ms": _metric_summary([item.get("edge_llm_latency_ms") for item in execution_records]),
        "edge_compute_latency_ms": _metric_summary([item.get("edge_compute_latency_ms") for item in execution_records]),
        "jetson_dnn_latency_ms": _metric_summary([item.get("jetson_dnn_latency_ms") for item in execution_records]),
        "airsim_action_latency_ms": _metric_summary([item.get("airsim_action_latency_ms") for item in execution_records]),
        "action_wall_time_ms": _metric_summary([item.get("action_wall_time_ms") for item in execution_records]),
        "action_sim_time_ms": _metric_summary([item.get("action_sim_time_ms") for item in execution_records]),
        "action_delay_ms": _metric_summary([item.get("action_delay_ms") for item in execution_records]),
        "state_drift_m": _metric_summary([item.get("state_drift_m") for item in execution_records]),
        "num_records": len(summary_records),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    _configure_local_air_sim_server()
    _set_console_log_message_only()
    configure_fast_eval_output(args, "continuous_dnn")
    eval_save_path = args.eval_save_path
    os.makedirs(eval_save_path, exist_ok=True)
    profile_log_dir = os.path.join(eval_save_path, "profile_logs")
    os.makedirs(profile_log_dir, exist_ok=True)

    setup()
    assert CheckPort(), "error port"
    eval_env = initialize_env_eval(
        dataset_path=args.dataset_path,
        save_path=eval_save_path,
        eval_json_path=args.eval_json_path,
    )
    if is_dist_avail_and_initialized():
        torch.distributed.destroy_process_group()
    args.DistributedDataParallel = False

    model_wrapper = EdgeDNNModelWrapper(model_args=model_args, data_args=data_args)
    assist = Assist(always_help=args.always_help, use_gt=args.use_gt)
    print("Assist setting: always_help --", args.always_help, "    use_gt --", args.use_gt)
    print(
        f"Edge DNN setting: host={args.edge_vlm_host}:{args.edge_vlm_port}, "
        f"chunk_waypoints={args.chunk_waypoints}, enable_comm_delay={args.enable_comm_delay}, "
        f"fast_eval={args.fast_eval}, fast_eval_speedup={args.fast_eval_speedup}"
    )

    chunk_waypoints = max(1, int(args.chunk_waypoints))
    trace_path = args.comm_trace_csv_path or default_trace_path()
    bandwidth_trace = BandwidthTrace(trace_path)
    enable_comm_delay = _as_bool(args.enable_comm_delay)
    profile_log_path = os.path.join(profile_log_dir, f"edge_dnn_w{chunk_waypoints}_{args.make_dir_time}.jsonl")
    summary_path = os.path.join(profile_log_dir, f"edge_dnn_w{chunk_waypoints}_{args.make_dir_time}_summary.json")
    eval(
        model_wrapper=model_wrapper,
        assist=assist,
        eval_env=eval_env,
        profile_log_path=profile_log_path,
        summary_path=summary_path,
        bandwidth_trace=bandwidth_trace,
        enable_comm_delay=enable_comm_delay,
        chunk_waypoints=chunk_waypoints,
    )
    eval_env.delete_VectorEnvUtil()
