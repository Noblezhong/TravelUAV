from typing import Sequence


STOP_WAYPOINTS_PER_DECISION = 5


def stop_executed_waypoint_count(decision_steps: int) -> int:
    return max(0, int(decision_steps)) * STOP_WAYPOINTS_PER_DECISION


def stop_executed_waypoints(waypoints: Sequence):
    return list(waypoints[:STOP_WAYPOINTS_PER_DECISION])


def control_budget_reached(control_steps: int, max_control_steps: int) -> bool:
    return int(control_steps) >= int(max_control_steps)


def mark_ne_regression_failure(collisions, dones, index: int):
    dones[index] = True
    return collisions, dones


def check_collision_without_tiny_diff(
    episodes,
    current_observations,
    collisions,
    dones,
):
    for index, previous_episode in enumerate(episodes):
        if collisions[index]:
            if not dones[index]:
                dones[index] = True
            continue
        if len(previous_episode) == 0:
            continue

        close_collision = False
        current_episode = current_observations[index]
        for depth in current_episode[-1]["depth"]:
            zero_count = (depth <= 1).sum()
            if zero_count > 0.1 * depth.size:
                close_collision = True
                break

        collisions[index] = close_collision
        if close_collision and not dones[index]:
            dones[index] = True
    return collisions, dones
