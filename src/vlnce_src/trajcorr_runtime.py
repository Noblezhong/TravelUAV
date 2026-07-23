from dataclasses import dataclass

import numpy as np


TRAJECTORY_ORIGINAL = "original"
TRAJECTORY_CORRECTED = "corrected"


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

    def mark_normal_execution(self) -> bool:
        self.count += 1
        return self.count >= self.request_interval

    def mark_request_submitted(self) -> None:
        self.count = 0
