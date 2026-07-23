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
)
from src.vlnce_src.continue_eval import _as_bool, _fmt_point, _point_delta, _set_console_log_message_only
from src.vlnce_src.comm_delay import (
    BandwidthTrace,
    calculate_latency_ms,
    default_trace_path,
    estimate_uplink_payload_bits_from_episodes,
)
from src.vlnce_src.eval_contract import (
    check_collision_without_tiny_diff,
    control_budget_reached,
)
from src.vlnce_src.edge_vlm_rpc import request as rpc_request
from src.vlnce_src.fast_eval_time import (
    FastEvalClock,
    FastResultTiming,
    action_timing,
    configure_fast_eval_output,
)
from src.vlnce_src.trajcorr_runtime import (
    COMPLETION_BUFFER_EXHAUSTED,
    COMPLETION_GOAL_PASSED,
    COMPLETION_GOAL_REACHED,
    ContinuousRequestCounter,
    PHASE_NORMAL,
    PHASE_TARGET_LOCK,
    PHASE_WAIT_REFRESH,
    TRAJECTORY_CORRECTED,
    TRAJECTORY_ORIGINAL,
    TargetLockLifecycle,
    select_trajectory_mode,
)
from utils.logger import logger
from utils.utils import *

TARGET_LOCK_GOAL_RADIUS_M = 0.5
TARGET_LOCK_REQUEST_REASONS = {
    COMPLETION_GOAL_REACHED: "correction_complete",
    COMPLETION_GOAL_PASSED: "correction_passed_goal",
    COMPLETION_BUFFER_EXHAUSTED: "correction_buffer_exhausted",
}


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


@dataclass
class RequestContext:
    dnn_episode: List[Dict[str, Any]]
    request_reason: str


@dataclass
class PendingRequest:
    snapshot: EdgeSnapshot
    context: RequestContext


@dataclass
class PreparedTrajectory:
    mode: str
    waypoints: List[List[float]]
    profile: Dict[str, Any]
    coarse_state_shift_m: float
    coarse_time_shift_ms: float
    traj_state_shift_m: float
    traj_time_shift_ms: float
    observation_latency_ms: float
    discard_reason: Optional[str] = None


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


def _has_executable_waypoints(refined_waypoints, min_waypoints: int = 1) -> bool:
    arr = np.asarray(refined_waypoints, dtype=np.float64)
    if arr.size == 0 or arr.size % 3 != 0:
        return False
    arr = arr.reshape(-1, 3)
    if len(arr) < int(min_waypoints) or not np.all(np.isfinite(arr)):
        return False
    return True


def _has_request_ne_regression(request_ne_history: List[float], count: int = 10) -> bool:
    if len(request_ne_history) < count:
        return False
    recent = request_ne_history[-count:]
    return all(recent[index] <= recent[index + 1] for index in range(len(recent) - 1))


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
        self.request_ne_history: List[float] = []
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

    def refresh_observation(self) -> float:
        start = time.perf_counter()
        outputs = self.eval_env.get_obs()
        latency_ms = (time.perf_counter() - start) * 1000.0
        self.process_env_output(outputs)
        return float(latency_ms)

    def record_request_ne(self) -> None:
        self.request_ne_history.append(self._calculate_distance_from_position(self.current_sim_pose()))
        if _has_request_ne_regression(self.request_ne_history):
            self.dones[0] = True

    def _calculate_distance_from_position(self, position) -> float:
        return float(
            np.linalg.norm(
                np.asarray(position, dtype=np.float64)
                - np.asarray(self.target_position, dtype=np.float64)
            )
        )

    def apply_edge_stop_result(self, predict_done: bool, distance_to_target_m: float) -> None:
        self.predict_dones = [bool(predict_done)]
        if not self.dones[0] and predict_done:
            if distance_to_target_m <= 20:
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


def _build_request(
    state: EdgeDNNContinuousState,
    request_id: int,
    exec_step: int,
    clock: FastEvalClock,
    reason: str,
) -> PendingRequest:
    snapshot = _snapshot_with_clock(state, request_id, exec_step, clock)
    return PendingRequest(
        snapshot=snapshot,
        context=RequestContext(
            dnn_episode=[copy.deepcopy(state.episode[-1])],
            request_reason=str(reason),
        ),
    )


def _submit_prepared_request(
    edge_client,
    request_contexts: Dict[int, RequestContext],
    request: PendingRequest,
) -> None:
    if not edge_client.submit(request.snapshot):
        raise RuntimeError(
            f"failed to submit edge request {request.snapshot.request_id}: planner is busy"
        )
    request_contexts[request.snapshot.request_id] = request.context


def _prepare_trajectory(
    model_wrapper: EdgeDNNModelWrapper,
    state: EdgeDNNContinuousState,
    result: EdgeCoarseResult,
    request_context: RequestContext,
    clock: FastEvalClock,
    correction_threshold_m: float,
    correction_enabled: bool,
) -> PreparedTrajectory:
    current_pose = state.current_sim_pose()
    coarse_time_shift_ms = _action_delay_ms(result, clock)
    decision = select_trajectory_mode(
        correction_enabled,
        result.observation_pose,
        current_pose,
        correction_threshold_m,
    )

    observation_latency_ms = 0.0
    if decision.mode == TRAJECTORY_ORIGINAL:
        model_episode = request_context.dnn_episode
        traj_reference_pose = copy.deepcopy(result.observation_pose)
        traj_reference_logical_ms = result.submitted_logical_ms
        traj_reference_perf_time = result.submitted_perf_time
        refined_waypoints, profile = model_wrapper.run_traj_from_world_goal(
            [model_episode],
            [result.coarse_goal_world],
        )
        refined_current = np.asarray(refined_waypoints[0], dtype=np.float64).tolist()
        coarse_norm = float(np.linalg.norm(np.asarray(result.coarse_local, dtype=np.float64)))
        p5_error = (
            float(_point_delta(refined_current[4], result.coarse_goal_world))
            if len(refined_current) >= 5
            else None
        )
        profile.update(
            {
                "virtual_goal_world": [copy.deepcopy(result.coarse_goal_world)],
                "original_coarse_norm_m": [coarse_norm],
                "trajcorr_coarse_norm_m": [coarse_norm],
                "trajectory_scale": [1.0],
                "p5_to_virtual_goal_m": [p5_error],
            }
        )
    else:
        observation_latency_ms = state.refresh_observation()
        clock.advance_blocking(observation_latency_ms)
        traj_reference_pose = state.current_sim_pose()
        traj_reference_logical_ms = clock.now_ms if clock.enabled else None
        traj_reference_perf_time = time.perf_counter()
        if state.dones[0]:
            return PreparedTrajectory(
                mode=decision.mode,
                waypoints=[],
                profile={},
                coarse_state_shift_m=decision.state_shift_m,
                coarse_time_shift_ms=coarse_time_shift_ms,
                traj_state_shift_m=0.0,
                traj_time_shift_ms=0.0,
                observation_latency_ms=observation_latency_ms,
                discard_reason="terminated_during_correction_observation",
            )
        current_episode = [copy.deepcopy(state.episode[-1])]
        refined_waypoints, profile = model_wrapper.run_trajcorr_from_world_goal(
            [current_episode],
            [result.coarse_goal_world],
            [result.coarse_local],
            [result.observation_pose],
        )
        refined_current = np.asarray(refined_waypoints[0], dtype=np.float64).tolist()

    clock.advance_blocking(float(profile.get("traj_latency_ms", 0.0)))
    traj_state_shift_m = math.dist(traj_reference_pose, state.current_sim_pose())
    if clock.enabled and traj_reference_logical_ms is not None:
        traj_time_shift_ms = max(0.0, clock.now_ms - float(traj_reference_logical_ms))
    else:
        traj_time_shift_ms = max(
            0.0,
            (time.perf_counter() - float(traj_reference_perf_time)) * 1000.0,
        )
    return PreparedTrajectory(
        mode=decision.mode,
        waypoints=refined_current,
        profile=profile,
        coarse_state_shift_m=decision.state_shift_m,
        coarse_time_shift_ms=coarse_time_shift_ms,
        traj_state_shift_m=traj_state_shift_m,
        traj_time_shift_ms=traj_time_shift_ms,
        observation_latency_ms=observation_latency_ms,
    )


def _profile_first(profile: Dict[str, Any], key: str, default=None):
    value = profile.get(key, default)
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


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


def _configure_air_sim_server():
    args.machines_info[0]["MACHINE_IP"] = str(args.simulator_tool_host)


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
    trajcorr_mode: str,
):
    assert int(eval_env.batch_size) == 1, "edge DNN eval currently supports batchSize=1 only"
    assert int(chunk_waypoints) == 5, "TrajCorr comparison requires Continuous w=5"
    trajcorr_mode = str(trajcorr_mode).strip().lower()
    if trajcorr_mode not in {"off", "on"}:
        raise ValueError("trajcorr_mode must be 'off' or 'on'")
    correction_enabled = trajcorr_mode == "on"
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
                    bandwidth_trace.reset_for_episode(env_batchs[0]["seq_name"])
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
                        request_contexts: Dict[int, RequestContext] = {}
                        pending_request: Optional[PendingRequest] = None
                        active_coarse_result: Optional[EdgeCoarseResult] = None
                        active_traj: List[List[float]] = []
                        active_index = 0
                        active_dnn_profile: Dict[str, Any] = {}
                        active_mode: Optional[str] = None
                        active_coarse_state_shift_m = 0.0
                        active_coarse_time_shift_ms = 0.0
                        active_traj_state_shift_m = 0.0
                        active_traj_time_shift_ms = 0.0
                        active_refine_obs_latency_ms = 0.0
                        active_target_lock_dropped_pending = False
                        continuous_counter = ContinuousRequestCounter(request_interval=5)
                        target_lock = TargetLockLifecycle(
                            goal_radius_m=TARGET_LOCK_GOAL_RADIUS_M
                        )
                        correction_threshold_m = float(args.trajcorr_state_shift_threshold_m)

                        def result_enters_target_lock(
                            result: EdgeCoarseResult,
                        ) -> bool:
                            decision = select_trajectory_mode(
                                correction_enabled,
                                result.observation_pose,
                                state.current_sim_pose(),
                                correction_threshold_m,
                            )
                            return decision.mode == TRAJECTORY_CORRECTED

                        def drop_pending_request() -> bool:
                            nonlocal pending_request
                            dropped = pending_request is not None
                            pending_request = None
                            return dropped

                        def submit_pending_if_idle() -> bool:
                            nonlocal pending_request
                            if pending_request is None or edge_client.has_inflight():
                                return False
                            _submit_prepared_request(
                                edge_client,
                                request_contexts,
                                pending_request,
                            )
                            pending_request = None
                            return True

                        def schedule_request(reason: str, refresh: bool) -> tuple[float, Optional[int]]:
                            nonlocal request_counter, pending_request
                            observation_latency_ms = 0.0
                            if refresh:
                                observation_latency_ms = state.refresh_observation()
                                episode_clock.advance_blocking(observation_latency_ms)
                            if state.dones[0]:
                                return observation_latency_ms, None
                            state.record_request_ne()
                            if state.dones[0]:
                                return observation_latency_ms, None
                            request_counter += 1
                            request = _build_request(
                                state,
                                request_counter,
                                exec_step,
                                episode_clock,
                                reason,
                            )
                            if edge_client.has_inflight():
                                pending_request = request
                            else:
                                _submit_prepared_request(
                                    edge_client,
                                    request_contexts,
                                    request,
                                )
                            return observation_latency_ms, request_counter

                        def consume_result(result: EdgeCoarseResult) -> Optional[PreparedTrajectory]:
                            request_context = request_contexts.pop(result.request_id, None)
                            if request_context is None:
                                raise RuntimeError(
                                    f"missing request-time DNN observation for request {result.request_id}"
                                )
                            action_delay_ms = _action_delay_ms(result, episode_clock)
                            state_drift_m = math.dist(result.observation_pose, state.current_sim_pose())
                            state.apply_edge_stop_result(
                                result.predict_done,
                                result.dino_distance_to_target_m,
                            )
                            _print_edge_result(episode_idx, result, action_delay_ms, state_drift_m)

                            prepared = None
                            if not state.dones[0]:
                                if result.coarse_goal_world is None:
                                    raise RuntimeError(
                                        "edge VLM returned no coarse goal without terminating the episode: "
                                        f"request_id={result.request_id}, "
                                        f"predict_done={result.predict_done}, "
                                        f"distance_to_target_m={result.dino_distance_to_target_m:.2f}"
                                    )
                                if result.coarse_local is None:
                                    raise RuntimeError(
                                        "edge VLM returned no coarse vector without terminating the episode: "
                                        f"request_id={result.request_id}"
                                    )
                                prepared = _prepare_trajectory(
                                    model_wrapper,
                                    state,
                                    result,
                                    request_context,
                                    episode_clock,
                                    correction_threshold_m,
                                    correction_enabled,
                                )
                            _mark_result_applied(result, episode_clock)

                            profile = prepared.profile if prepared is not None else {}
                            coarse_state_shift_m = (
                                prepared.coarse_state_shift_m
                                if prepared is not None
                                else state_drift_m
                            )
                            coarse_time_shift_ms = (
                                prepared.coarse_time_shift_ms
                                if prepared is not None
                                else action_delay_ms
                            )
                            enters_target_lock = bool(
                                prepared is not None
                                and prepared.mode == TRAJECTORY_CORRECTED
                            )
                            target_lock_distance_m = (
                                _point_delta(
                                    result.coarse_goal_world,
                                    state.current_sim_pose(),
                                )
                                if enters_target_lock
                                and result.coarse_goal_world is not None
                                else None
                            )
                            result_record = {
                                "record_type": "trajectory_result",
                                "exec_step": int(exec_step),
                                "request_id": int(result.request_id),
                                "request_reason": request_context.request_reason,
                                "seq_names": [env_batchs[0]["seq_name"]],
                                "map_names": [env_batchs[0]["map_name"]],
                                "comm_delay_enabled": bool(enable_comm_delay),
                                "uplink_payload_bits": int(result.uplink_payload_bits),
                                "uplink_payload_mb": float(result.uplink_payload_mb),
                                "uplink_bandwidth_mbps": float(result.uplink_bandwidth_mbps),
                                "uplink_latency_ms": float(result.uplink_latency_ms),
                                "edge_llm_latency_ms": float(result.edge_llm_latency_ms),
                                "edge_compute_latency_ms": float(result.edge_compute_latency_ms),
                                "jetson_dnn_latency_ms": (
                                    float(profile.get("traj_latency_ms"))
                                    if profile.get("traj_latency_ms") is not None
                                    else None
                                ),
                                "dnn_refine_observation_latency_ms": (
                                    float(prepared.observation_latency_ms) if prepared is not None else None
                                ),
                                "action_delay_ms": float(coarse_time_shift_ms),
                                "state_drift_m": float(coarse_state_shift_m),
                                "coarse_time_shift_ms": float(coarse_time_shift_ms),
                                "coarse_state_shift_m": float(coarse_state_shift_m),
                                "traj_time_shift_ms": (
                                    float(prepared.traj_time_shift_ms)
                                    if prepared is not None
                                    else None
                                ),
                                "traj_state_shift_m": (
                                    float(prepared.traj_state_shift_m)
                                    if prepared is not None
                                    else None
                                ),
                                "trajectory_mode": prepared.mode if prepared is not None else None,
                                "execution_phase": (
                                    PHASE_TARGET_LOCK
                                    if enters_target_lock
                                    else PHASE_NORMAL
                                ),
                                "target_lock_active": enters_target_lock,
                                "target_lock_goal_world": (
                                    copy.deepcopy(result.coarse_goal_world)
                                    if enters_target_lock
                                    else None
                                ),
                                "target_lock_distance_m": target_lock_distance_m,
                                "continuous_counter_frozen": enters_target_lock,
                                "target_lock_completion_reason": None,
                                "dropped_pending_request": bool(
                                    enters_target_lock and pending_request is not None
                                ),
                                "state_shift_at_apply_m": (
                                    float(coarse_state_shift_m)
                                ),
                                "time_shift_at_apply_ms": (
                                    float(coarse_time_shift_ms)
                                ),
                                "trajcorr_mode": trajcorr_mode,
                                "correction_threshold_m": float(correction_threshold_m),
                                "ne_at_apply_m": float(
                                    state._calculate_distance_from_position(
                                        state.current_sim_pose()
                                    )
                                ),
                                "virtual_goal_world": copy.deepcopy(
                                    _profile_first(profile, "virtual_goal_world")
                                ),
                                "trajectory_scale": _profile_first(profile, "trajectory_scale"),
                                "request_trigger_reason": request_context.request_reason,
                                "request_trigger_waypoint_index": (
                                    5
                                    if request_context.request_reason == "continuous_w5"
                                    else None
                                ),
                                "p5_to_virtual_goal_m": _profile_first(
                                    profile, "p5_to_virtual_goal_m"
                                ),
                                "result_discard_reason": (
                                    prepared.discard_reason if prepared is not None else None
                                ),
                                "coarse_local": copy.deepcopy(result.coarse_local),
                                "coarse_goal_world": copy.deepcopy(result.coarse_goal_world),
                                "coarse_goal_world_source": result.coarse_goal_world_source,
                                "legacy_body_goal_world": copy.deepcopy(result.legacy_body_goal_world),
                                "reprojected_coarse": copy.deepcopy(
                                    _profile_first(profile, "reprojected_coarse")
                                ),
                                "original_coarse_norm_m": _profile_first(
                                    profile, "original_coarse_norm_m"
                                ),
                                "trajcorr_coarse_norm_m": _profile_first(
                                    profile, "trajcorr_coarse_norm_m"
                                ),
                                "refined_waypoints": (
                                    copy.deepcopy(prepared.waypoints) if prepared is not None else []
                                ),
                                "fast_eval": bool(episode_clock.enabled),
                                "fast_eval_speedup": float(
                                    episode_clock.speedup if episode_clock.enabled else 1.0
                                ),
                                "wall_elapsed_ms": float(episode_clock.wall_elapsed_ms),
                                "logical_elapsed_ms": float(episode_clock.now_ms),
                                "result_ready_logical_ms": result.ready_logical_ms,
                                "result_applied_logical_ms": result.applied_logical_ms,
                                "success": bool(state.success),
                                "collision": bool(state.collisions[0]),
                                "done": bool(state.dones[0]),
                            }
                            _write_jsonl_line(profile_fp, result_record)
                            summary_records.append(result_record)
                            return prepared

                        def activate_trajectory(
                            result: EdgeCoarseResult,
                            prepared: Optional[PreparedTrajectory],
                        ) -> tuple[bool, Optional[str], bool]:
                            nonlocal active_coarse_result
                            nonlocal active_traj, active_index, active_dnn_profile, active_mode
                            nonlocal active_coarse_state_shift_m, active_coarse_time_shift_ms
                            nonlocal active_traj_state_shift_m, active_traj_time_shift_ms
                            nonlocal active_refine_obs_latency_ms
                            nonlocal active_target_lock_dropped_pending
                            if prepared is None:
                                return False, None, False

                            completion_reason = None
                            dropped_pending = False
                            min_waypoints = 7
                            if prepared.mode == TRAJECTORY_CORRECTED:
                                dropped_pending = drop_pending_request()
                                active_target_lock_dropped_pending = dropped_pending
                                completion_reason = target_lock.begin(
                                    result.observation_pose,
                                    state.current_sim_pose(),
                                    result.coarse_goal_world,
                                )
                                raw_waypoint_count = len(prepared.waypoints)
                                if completion_reason is None:
                                    prepared.waypoints = target_lock.filter_waypoints(
                                        prepared.waypoints,
                                        state.current_sim_pose(),
                                    )
                                    if not prepared.waypoints:
                                        completion_reason = (
                                            target_lock.mark_buffer_exhausted()
                                        )
                                else:
                                    prepared.waypoints = []
                                prepared.profile.update(
                                    {
                                        "target_lock_raw_waypoint_count": raw_waypoint_count,
                                        "target_lock_filtered_waypoint_count": len(
                                            prepared.waypoints
                                        ),
                                    }
                                )
                                min_waypoints = 1
                            else:
                                target_lock.resume_normal()
                                active_target_lock_dropped_pending = False

                            if (
                                completion_reason is None
                                and not _has_executable_waypoints(
                                    prepared.waypoints,
                                    min_waypoints=min_waypoints,
                                )
                            ):
                                state.dones[0] = True
                                invalid_record = {
                                    "record_type": "dnn_invalid",
                                    "exec_step": int(exec_step),
                                    "request_id": int(result.request_id),
                                    "seq_names": [env_batchs[0]["seq_name"]],
                                    "map_names": [env_batchs[0]["map_name"]],
                                    "trajectory_mode": prepared.mode,
                                    "refined_waypoints": copy.deepcopy(prepared.waypoints),
                                    "dones": [bool(x) for x in state.dones],
                                    "collisions": [bool(x) for x in state.collisions],
                                    "success": bool(state.success),
                                }
                                _write_jsonl_line(profile_fp, invalid_record)
                                summary_records.append(invalid_record)
                                return False, None, dropped_pending
                            active_coarse_result = result
                            active_traj = prepared.waypoints
                            active_index = 0
                            active_dnn_profile = prepared.profile
                            active_mode = prepared.mode
                            active_coarse_state_shift_m = prepared.coarse_state_shift_m
                            active_coarse_time_shift_ms = prepared.coarse_time_shift_ms
                            active_traj_state_shift_m = prepared.traj_state_shift_m
                            active_traj_time_shift_ms = prepared.traj_time_shift_ms
                            active_refine_obs_latency_ms = prepared.observation_latency_ms
                            return True, completion_reason, dropped_pending

                        def complete_target_lock(
                            completion_reason: str,
                            dropped_pending: bool = False,
                        ) -> tuple[float, Optional[int], bool]:
                            nonlocal active_traj, active_index
                            nonlocal active_target_lock_dropped_pending
                            request_reason = TARGET_LOCK_REQUEST_REASONS.get(
                                completion_reason
                            )
                            if request_reason is None:
                                raise ValueError(
                                    f"unknown target-lock completion reason: {completion_reason}"
                                )

                            dropped_pending = (
                                drop_pending_request()
                                or dropped_pending
                                or active_target_lock_dropped_pending
                            )
                            active_target_lock_dropped_pending = False
                            active_traj = []
                            active_index = 0
                            observation_latency_ms, triggered_request_id = schedule_request(
                                request_reason,
                                refresh=True,
                            )
                            if triggered_request_id is not None:
                                continuous_counter.mark_request_submitted()

                            transition_record = {
                                "record_type": "target_lock_transition",
                                "exec_step": int(exec_step),
                                "seq_names": [env_batchs[0]["seq_name"]],
                                "map_names": [env_batchs[0]["map_name"]],
                                "execution_phase": PHASE_WAIT_REFRESH,
                                "target_lock_active": False,
                                "target_lock_goal_world": copy.deepcopy(
                                    target_lock.goal_world
                                ),
                                "target_lock_distance_m": (
                                    target_lock.distance_to_goal(
                                        state.current_sim_pose()
                                    )
                                ),
                                "continuous_counter_frozen": False,
                                "target_lock_completion_reason": completion_reason,
                                "dropped_pending_request": bool(dropped_pending),
                                "request_trigger_reason": request_reason,
                                "triggered_request_id": triggered_request_id,
                                "request_observation_latency_ms": float(
                                    observation_latency_ms
                                ),
                                "logical_elapsed_ms": float(episode_clock.now_ms),
                                "success": bool(state.success),
                                "collision": bool(state.collisions[0]),
                                "done": bool(state.dones[0]),
                            }
                            _write_jsonl_line(profile_fp, transition_record)
                            summary_records.append(transition_record)
                            return (
                                observation_latency_ms,
                                triggered_request_id,
                                dropped_pending,
                            )

                        # Cold start is the only blocking request before an active trajectory exists.
                        state.record_request_ne()
                        cold_start_request = _build_request(
                            state,
                            request_counter,
                            exec_step,
                            episode_clock,
                            "cold_start",
                        )
                        _submit_prepared_request(
                            edge_client,
                            request_contexts,
                            cold_start_request,
                        )

                        while not state.dones[0] and not control_budget_reached(
                            exec_step,
                            args.max_control_steps,
                        ):
                            new_result = edge_client.poll_result()
                            if new_result is not None:
                                if not result_enters_target_lock(new_result):
                                    submit_pending_if_idle()
                                prepared = consume_result(new_result)
                                if state.dones[0]:
                                    break
                                activated, completion_reason, dropped_pending = (
                                    activate_trajectory(new_result, prepared)
                                )
                                if not activated:
                                    continue
                                if active_mode == TRAJECTORY_ORIGINAL:
                                    submit_pending_if_idle()
                                elif completion_reason is not None:
                                    complete_target_lock(
                                        completion_reason,
                                        dropped_pending=dropped_pending,
                                    )
                                    if state.dones[0]:
                                        break

                            if active_coarse_result is None or active_index >= len(active_traj):
                                if target_lock.active:
                                    complete_target_lock(
                                        target_lock.mark_buffer_exhausted()
                                    )
                                    if state.dones[0]:
                                        break
                                submit_pending_if_idle()
                                if not edge_client.has_inflight():
                                    raise RuntimeError(
                                        "trajectory buffer exhausted without an edge request in flight"
                                    )
                                wait_reason = (
                                    "cold_start" if active_coarse_result is None else "buffer_exhausted"
                                )
                                wait_record = {
                                    "record_type": "buffer_wait",
                                    "exec_step": int(exec_step),
                                    "seq_names": [env_batchs[0]["seq_name"]],
                                    "map_names": [env_batchs[0]["map_name"]],
                                    "buffer_exhausted": wait_reason == "buffer_exhausted",
                                    "wait_reason": wait_reason,
                                    "logical_elapsed_ms": float(episode_clock.now_ms),
                                }
                                _write_jsonl_line(profile_fp, wait_record)
                                summary_records.append(wait_record)
                                waited_result = edge_client.wait_result()
                                if not result_enters_target_lock(waited_result):
                                    submit_pending_if_idle()
                                prepared = consume_result(waited_result)
                                if state.dones[0]:
                                    break
                                activated, completion_reason, dropped_pending = (
                                    activate_trajectory(waited_result, prepared)
                                )
                                if not activated:
                                    continue
                                if active_mode == TRAJECTORY_ORIGINAL:
                                    submit_pending_if_idle()
                                elif completion_reason is not None:
                                    complete_target_lock(
                                        completion_reason,
                                        dropped_pending=dropped_pending,
                                    )
                                    if state.dones[0]:
                                        break
                                    continue

                            current_index = active_index
                            current_chunk = active_traj[current_index : current_index + 1]
                            if not current_chunk:
                                continue
                            execution_phase = (
                                PHASE_TARGET_LOCK
                                if target_lock.active
                                else PHASE_NORMAL
                            )
                            counter_frozen = target_lock.counter_frozen
                            target_lock_goal_world = copy.deepcopy(
                                target_lock.goal_world
                            )
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
                            target_lock_distance_m = target_lock.distance_to_goal(
                                state.current_sim_pose()
                            )
                            target_lock_completion_reason = None
                            request_reason = None
                            request_obs_latency_ms = 0.0
                            triggered_request_id = None
                            dropped_pending_for_action = False

                            if counter_frozen:
                                target_lock_completion_reason = target_lock.evaluate(
                                    state.current_sim_pose()
                                )
                                if (
                                    target_lock_completion_reason is None
                                    and active_index >= len(active_traj)
                                ):
                                    target_lock_completion_reason = (
                                        target_lock.mark_buffer_exhausted()
                                    )
                                if target_lock_completion_reason is not None:
                                    request_reason = TARGET_LOCK_REQUEST_REASONS[
                                        target_lock_completion_reason
                                    ]
                                    (
                                        request_obs_latency_ms,
                                        triggered_request_id,
                                        dropped_pending_for_action,
                                    ) = complete_target_lock(
                                        target_lock_completion_reason
                                    )
                            elif continuous_counter.mark_execution(frozen=False):
                                request_reason = "continuous_w5"
                                request_obs_latency_ms, triggered_request_id = schedule_request(
                                    request_reason, refresh=True
                                )
                                if triggered_request_id is not None:
                                    continuous_counter.mark_request_submitted()

                            action_delay_ms = _action_delay_ms(active_coarse_result, episode_clock)
                            state_drift_m = math.dist(active_coarse_result.observation_pose, state.current_sim_pose())
                            coarse_goal_distance = _point_delta(
                                active_coarse_result.coarse_goal_world,
                                state.current_sim_pose(),
                            )
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
                                    float(active_dnn_profile["traj_latency_ms"])
                                    if current_index == 0
                                    else None
                                ),
                                "dnn_refine_observation_latency_ms": (
                                    float(active_refine_obs_latency_ms) if current_index == 0 else None
                                ),
                                "request_observation_latency_ms": float(request_obs_latency_ms),
                                "triggered_request_id": triggered_request_id,
                                "execution_phase": execution_phase,
                                "continuous_waypoints_since_request": int(continuous_counter.count),
                                "target_lock_active": bool(counter_frozen),
                                "target_lock_goal_world": target_lock_goal_world,
                                "target_lock_distance_m": target_lock_distance_m,
                                "continuous_counter_frozen": bool(counter_frozen),
                                "target_lock_completion_reason": (
                                    target_lock_completion_reason
                                ),
                                "dropped_pending_request": bool(
                                    dropped_pending_for_action
                                ),
                                "request_trigger_reason": request_reason,
                                "airsim_action_latency_ms": float(action_times["airsim_action_latency_ms"]),
                                "action_wall_time_ms": float(action_times["action_wall_time_ms"]),
                                "action_sim_time_ms": float(action_times["action_sim_time_ms"]),
                                "action_delay_ms": float(action_delay_ms),
                                "state_drift_m": float(state_drift_m),
                                "coarse_goal_distance_m": float(coarse_goal_distance),
                                "navigation_error_m": float(
                                    state._calculate_distance_from_position(
                                        state.current_sim_pose()
                                    )
                                ),
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
                                "reprojected_coarse": copy.deepcopy(
                                    _profile_first(active_dnn_profile, "reprojected_coarse")
                                ),
                                "trajectory_mode": active_mode,
                                "trajcorr_mode": trajcorr_mode,
                                "coarse_state_shift_m": float(active_coarse_state_shift_m),
                                "coarse_time_shift_ms": float(active_coarse_time_shift_ms),
                                "traj_state_shift_m": float(active_traj_state_shift_m),
                                "traj_time_shift_ms": float(active_traj_time_shift_ms),
                                "state_shift_at_apply_m": float(active_coarse_state_shift_m),
                                "time_shift_at_apply_ms": float(active_coarse_time_shift_ms),
                                "correction_threshold_m": float(correction_threshold_m),
                                "virtual_goal_world": copy.deepcopy(
                                    _profile_first(active_dnn_profile, "virtual_goal_world")
                                ),
                                "trajectory_scale": _profile_first(
                                    active_dnn_profile, "trajectory_scale"
                                ),
                                "request_trigger_waypoint_index": (
                                    5 if request_reason == "continuous_w5" else None
                                ),
                                "p5_to_virtual_goal_m": _profile_first(
                                    active_dnn_profile, "p5_to_virtual_goal_m"
                                ),
                                "result_discard_reason": None,
                                "active_waypoint_index": int(active_index),
                                "original_coarse_norm_m": float(
                                    _profile_first(active_dnn_profile, "original_coarse_norm_m")
                                ),
                                "trajcorr_coarse_norm_m": float(
                                    _profile_first(active_dnn_profile, "trajcorr_coarse_norm_m")
                                ),
                                "refined_waypoints": copy.deepcopy(active_traj),
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
                        episode_record = {
                            "record_type": "episode_end",
                            "seq_names": [env_batchs[0]["seq_name"]],
                            "map_names": [env_batchs[0]["map_name"]],
                            "control_steps": int(exec_step),
                            "max_control_steps": int(args.max_control_steps),
                            "control_budget_reached": control_budget_reached(
                                exec_step,
                                args.max_control_steps,
                            ),
                            "episode_latency_ms": float(episode_clock.now_ms),
                            "success": bool(state.success),
                            "oracle_success": bool(state.oracle_success),
                            "collision": bool(state.collisions[0]),
                            "final_ne_m": float(
                                state._calculate_distance_from_position(state.current_sim_pose())
                            ),
                        }
                        episode_record.update(episode_clock.metadata())
                        _write_jsonl_line(profile_fp, episode_record)
                        summary_records.append(episode_record)
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
    episode_records = [item for item in summary_records if item.get("record_type") == "episode_end"]
    decision_records = [
        item for item in summary_records if item.get("record_type") == "trajectory_result"
    ]
    target_lock_records = [
        item
        for item in summary_records
        if item.get("record_type") == "target_lock_transition"
    ]
    mode_counts = {
        mode: sum(item.get("trajectory_mode") == mode for item in decision_records)
        for mode in (TRAJECTORY_ORIGINAL, TRAJECTORY_CORRECTED)
    }
    mode_total = sum(mode_counts.values())
    applied_count = mode_counts[TRAJECTORY_ORIGINAL] + mode_counts[TRAJECTORY_CORRECTED]
    decision_by_key = {
        ((item.get("seq_names") or [None])[0], item.get("request_id")): item
        for item in decision_records
    }
    execution_by_key = {}
    for item in execution_records:
        key = ((item.get("seq_names") or [None])[0], item.get("request_id"))
        execution_by_key.setdefault(key, []).append(item)
    execution_counts = {
        key: len(records)
        for key, records in execution_by_key.items()
    }

    def ne_progress_after_five(mode):
        values = []
        for key, decision in decision_by_key.items():
            if decision.get("trajectory_mode") != mode:
                continue
            start_ne = decision.get("ne_at_apply_m")
            records = sorted(
                execution_by_key.get(key, []),
                key=lambda item: int(item.get("exec_step", 0)),
            )
            if start_ne is None or not records:
                continue
            endpoint = records[min(4, len(records) - 1)].get("navigation_error_m")
            if endpoint is not None:
                values.append(float(start_ne) - float(endpoint))
        return _metric_summary(values)

    summary = {
        "decision_total_latency_ms": _metric_summary(
            [
                float(item.get("uplink_latency_ms", 0.0))
                + float(item.get("edge_llm_latency_ms", 0.0))
                for item in decision_records
            ]
        ),
        "uplink_latency_ms": _metric_summary([item.get("uplink_latency_ms") for item in decision_records]),
        "edge_llm_latency_ms": _metric_summary([item.get("edge_llm_latency_ms") for item in decision_records]),
        "edge_compute_latency_ms": _metric_summary([item.get("edge_compute_latency_ms") for item in decision_records]),
        "jetson_dnn_latency_ms": _metric_summary(
            [item.get("jetson_dnn_latency_ms") for item in decision_records]
        ),
        "airsim_action_latency_ms": _metric_summary([item.get("airsim_action_latency_ms") for item in execution_records]),
        "action_wall_time_ms": _metric_summary([item.get("action_wall_time_ms") for item in execution_records]),
        "action_sim_time_ms": _metric_summary([item.get("action_sim_time_ms") for item in execution_records]),
        "action_delay_ms": _metric_summary(
            [item.get("coarse_time_shift_ms") for item in decision_records]
        ),
        "state_drift_m": _metric_summary(
            [item.get("coarse_state_shift_m") for item in decision_records]
        ),
        "coarse_time_shift_ms": _metric_summary(
            [item.get("coarse_time_shift_ms") for item in decision_records]
        ),
        "coarse_state_shift_m": _metric_summary(
            [item.get("coarse_state_shift_m") for item in decision_records]
        ),
        "traj_time_shift_ms": _metric_summary(
            [item.get("traj_time_shift_ms") for item in decision_records]
        ),
        "traj_state_shift_m": _metric_summary(
            [item.get("traj_state_shift_m") for item in decision_records]
        ),
        "episode_latency_ms": _metric_summary([item.get("episode_latency_ms") for item in episode_records]),
        "final_ne_m": _metric_summary([item.get("final_ne_m") for item in episode_records]),
        "trajectory_mode_counts": mode_counts,
        "trajectory_mode_ratio": {
            key: (float(value) / mode_total if mode_total else 0.0)
            for key, value in mode_counts.items()
        },
        "correction_trigger_rate": (
            float(mode_counts[TRAJECTORY_CORRECTED]) / applied_count if applied_count else 0.0
        ),
        "p5_endpoint_error_m": _metric_summary(
            [item.get("p5_to_virtual_goal_m") for item in decision_records]
        ),
        "executed_waypoints_per_trajectory": _metric_summary(execution_counts.values()),
        "corrected_execution_waypoints": int(
            sum(
                execution_counts.get(key, 0)
                for key, decision in decision_by_key.items()
                if decision.get("trajectory_mode") == TRAJECTORY_CORRECTED
            )
        ),
        "original_ne_progress_5_steps_m": ne_progress_after_five(
            TRAJECTORY_ORIGINAL
        ),
        "corrected_ne_progress_5_steps_m": ne_progress_after_five(
            TRAJECTORY_CORRECTED
        ),
        "p5_request_count": sum(
            item.get("request_trigger_waypoint_index") == 5 for item in execution_records
        ),
        "buffer_exhausted_wait_count": sum(
            item.get("record_type") == "buffer_wait"
            and item.get("wait_reason") == "buffer_exhausted"
            for item in summary_records
        ),
        "request_reason_counts": {
            reason: sum(
                item.get("request_reason") == reason
                for item in decision_records
            )
            for reason in (
                "cold_start",
                "continuous_w5",
                "correction_complete",
                "correction_passed_goal",
                "correction_buffer_exhausted",
            )
        },
        "target_lock_completion_counts": {
            reason: sum(
                item.get("target_lock_completion_reason") == reason
                for item in target_lock_records
            )
            for reason in (
                COMPLETION_GOAL_REACHED,
                COMPLETION_GOAL_PASSED,
                COMPLETION_BUFFER_EXHAUSTED,
            )
        },
        "target_lock_distance_m": _metric_summary(
            [
                item.get("target_lock_distance_m")
                for item in execution_records
                if item.get("target_lock_active")
            ]
        ),
        "target_lock_execution_steps": sum(
            item.get("execution_phase") == PHASE_TARGET_LOCK
            for item in execution_records
        ),
        "dropped_pending_request_count": sum(
            bool(item.get("dropped_pending_request"))
            for item in target_lock_records
        ),
        "trajcorr_mode": trajcorr_mode,
        "max_control_steps": int(args.max_control_steps),
        "correction_threshold_m": float(args.trajcorr_state_shift_threshold_m),
        "target_lock_goal_radius_m": TARGET_LOCK_GOAL_RADIUS_M,
        "num_decision_records": len(decision_records),
        "num_episode_records": len(episode_records),
        "num_records": len(summary_records),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    _configure_air_sim_server()
    _set_console_log_message_only()
    trajcorr_mode = str(args.trajcorr_mode).strip().lower()
    if trajcorr_mode not in {"off", "on"}:
        raise ValueError("trajcorr_mode must be 'off' or 'on'")
    configure_fast_eval_output(args, f"trajcorr_{trajcorr_mode}")
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
        f"trajcorr_mode={trajcorr_mode}, "
        f"chunk_waypoints={args.chunk_waypoints}, enable_comm_delay={args.enable_comm_delay}, "
        f"fast_eval={args.fast_eval}, fast_eval_speedup={args.fast_eval_speedup}"
    )

    chunk_waypoints = max(1, int(args.chunk_waypoints))
    trace_path = args.comm_trace_csv_path or default_trace_path()
    bandwidth_trace = BandwidthTrace(trace_path)
    enable_comm_delay = _as_bool(args.enable_comm_delay)
    profile_log_path = os.path.join(
        profile_log_dir,
        f"trajcorr_{trajcorr_mode}_{args.make_dir_time}.jsonl",
    )
    summary_path = os.path.join(
        profile_log_dir,
        f"trajcorr_{trajcorr_mode}_{args.make_dir_time}_summary.json",
    )
    eval(
        model_wrapper=model_wrapper,
        assist=assist,
        eval_env=eval_env,
        profile_log_path=profile_log_path,
        summary_path=summary_path,
        bandwidth_trace=bandwidth_trace,
        enable_comm_delay=enable_comm_delay,
        chunk_waypoints=chunk_waypoints,
        trajcorr_mode=trajcorr_mode,
    )
    eval_env.delete_VectorEnvUtil()
