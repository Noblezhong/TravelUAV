from dataclasses import dataclass
from typing import List, Optional

import numpy as np


TRAJECTORY_ORIGINAL = "original"
TRAJECTORY_CORRECTED = "corrected"

PHASE_NORMAL = "normal"
PHASE_TARGET_LOCK = "target_lock"
PHASE_WAIT_REFRESH = "wait_refresh"

COMPLETION_GOAL_REACHED = "goal_reached"
COMPLETION_GOAL_PASSED = "goal_passed"
COMPLETION_BUFFER_EXHAUSTED = "buffer_exhausted"


@dataclass(frozen=True)
class TrajectoryDecision:
    mode: str
    state_shift_m: float


def select_trajectory_mode(
    enabled: bool,
    observation_pose,
    current_pose,
    correction_threshold_m: float,
) -> TrajectoryDecision:
    observation = np.asarray(observation_pose, dtype=np.float64).reshape(3)
    current = np.asarray(current_pose, dtype=np.float64).reshape(3)
    state_shift_m = float(np.linalg.norm(current - observation))
    mode = (
        TRAJECTORY_CORRECTED
        if bool(enabled) and state_shift_m >= float(correction_threshold_m)
        else TRAJECTORY_ORIGINAL
    )
    return TrajectoryDecision(mode=mode, state_shift_m=state_shift_m)


@dataclass
class ContinuousRequestCounter:
    """Global Naive Continuous waypoint counter, independent of buffer replacement."""

    request_interval: int = 5
    count: int = 0

    def mark_execution(self, frozen: bool = False) -> bool:
        if frozen:
            return False
        self.count += 1
        return self.count >= self.request_interval

    def mark_normal_execution(self) -> bool:
        return self.mark_execution(frozen=False)

    def mark_request_submitted(self) -> None:
        self.count = 0


def filter_target_lock_waypoints(
    waypoints,
    current_pose,
    goal_world,
    direction_world,
    goal_radius_m: float,
    epsilon: float = 1e-6,
) -> List[List[float]]:
    """Keep forward waypoints that monotonically approach the locked goal."""
    current = np.asarray(current_pose, dtype=np.float64).reshape(3)
    goal = np.asarray(goal_world, dtype=np.float64).reshape(3)
    direction = np.asarray(direction_world, dtype=np.float64).reshape(3)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= epsilon:
        return []
    direction = direction / direction_norm

    accepted: List[List[float]] = []
    previous = current
    previous_goal_distance = float(np.linalg.norm(goal - previous))
    for waypoint in waypoints:
        point = np.asarray(waypoint, dtype=np.float64).reshape(3)
        goal_distance = float(np.linalg.norm(goal - point))
        forward_progress = float(np.dot(point - previous, direction))
        signed_goal_remaining = float(np.dot(goal - point, direction))

        if goal_distance <= float(goal_radius_m):
            if forward_progress > epsilon:
                accepted.append(point.tolist())
            break
        if signed_goal_remaining <= 0.0:
            break
        if forward_progress <= epsilon:
            continue
        if goal_distance >= previous_goal_distance - epsilon:
            continue

        accepted.append(point.tolist())
        previous = point
        previous_goal_distance = goal_distance
    return accepted


@dataclass
class TargetLockLifecycle:
    goal_radius_m: float = 0.5
    phase: str = PHASE_NORMAL
    goal_world: Optional[List[float]] = None
    lock_origin_world: Optional[List[float]] = None
    direction_world: Optional[List[float]] = None
    start_distance_m: Optional[float] = None
    completion_reason: Optional[str] = None

    @property
    def active(self) -> bool:
        return self.phase == PHASE_TARGET_LOCK

    @property
    def waiting(self) -> bool:
        return self.phase == PHASE_WAIT_REFRESH

    @property
    def counter_frozen(self) -> bool:
        return self.phase == PHASE_TARGET_LOCK

    def _complete(self, reason: str) -> str:
        self.phase = PHASE_WAIT_REFRESH
        self.completion_reason = str(reason)
        return self.completion_reason

    def begin(
        self,
        observation_pose,
        current_pose,
        goal_world,
        epsilon: float = 1e-6,
    ) -> Optional[str]:
        observation = np.asarray(observation_pose, dtype=np.float64).reshape(3)
        current = np.asarray(current_pose, dtype=np.float64).reshape(3)
        goal = np.asarray(goal_world, dtype=np.float64).reshape(3)

        self.phase = PHASE_TARGET_LOCK
        self.goal_world = goal.tolist()
        self.lock_origin_world = current.tolist()
        self.start_distance_m = float(np.linalg.norm(goal - current))
        self.completion_reason = None

        if self.start_distance_m <= float(self.goal_radius_m):
            self.direction_world = None
            return self._complete(COMPLETION_GOAL_REACHED)

        original_direction = goal - observation
        original_norm = float(np.linalg.norm(original_direction))
        if original_norm <= epsilon:
            original_direction = goal - current
            original_norm = float(np.linalg.norm(original_direction))
        if original_norm <= epsilon:
            self.direction_world = None
            return self._complete(COMPLETION_GOAL_REACHED)
        original_direction /= original_norm

        if float(np.dot(goal - current, original_direction)) <= 0.0:
            self.direction_world = original_direction.tolist()
            return self._complete(COMPLETION_GOAL_PASSED)

        lock_direction = goal - current
        lock_norm = float(np.linalg.norm(lock_direction))
        if lock_norm <= epsilon:
            self.direction_world = None
            return self._complete(COMPLETION_GOAL_REACHED)
        self.direction_world = (lock_direction / lock_norm).tolist()
        return None

    def distance_to_goal(self, current_pose) -> Optional[float]:
        if self.goal_world is None:
            return None
        current = np.asarray(current_pose, dtype=np.float64).reshape(3)
        goal = np.asarray(self.goal_world, dtype=np.float64).reshape(3)
        return float(np.linalg.norm(goal - current))

    def evaluate(self, current_pose) -> Optional[str]:
        if not self.active or self.goal_world is None:
            return None

        distance = self.distance_to_goal(current_pose)
        if distance is not None and distance <= float(self.goal_radius_m):
            return self._complete(COMPLETION_GOAL_REACHED)

        if self.direction_world is None:
            return self._complete(COMPLETION_GOAL_REACHED)
        current = np.asarray(current_pose, dtype=np.float64).reshape(3)
        goal = np.asarray(self.goal_world, dtype=np.float64).reshape(3)
        direction = np.asarray(self.direction_world, dtype=np.float64).reshape(3)
        if float(np.dot(goal - current, direction)) <= 0.0:
            return self._complete(COMPLETION_GOAL_PASSED)
        return None

    def filter_waypoints(self, waypoints, current_pose) -> List[List[float]]:
        if not self.active or self.goal_world is None or self.direction_world is None:
            return []
        return filter_target_lock_waypoints(
            waypoints,
            current_pose,
            self.goal_world,
            self.direction_world,
            self.goal_radius_m,
        )

    def mark_buffer_exhausted(self) -> str:
        return self._complete(COMPLETION_BUFFER_EXHAUSTED)

    def resume_normal(self) -> None:
        self.phase = PHASE_NORMAL
        self.goal_world = None
        self.lock_origin_world = None
        self.direction_world = None
        self.start_distance_m = None
        self.completion_reason = None
