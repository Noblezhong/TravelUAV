from typing import Sequence


STOP_WAYPOINTS_PER_DECISION = 5


def stop_executed_waypoint_count(decision_steps: int) -> int:
    return max(0, int(decision_steps)) * STOP_WAYPOINTS_PER_DECISION


def stop_executed_waypoints(waypoints: Sequence):
    return list(waypoints[:STOP_WAYPOINTS_PER_DECISION])


def stop_single_waypoint_chunks(waypoints_batch: Sequence[Sequence]):
    """Split one Stop decision into shared single-waypoint controller calls.

    Stop-and-go still consumes at most five predicted waypoints before the next
    blocking request.  The split only aligns its action-call granularity with
    the asynchronous evaluators.
    """
    if not waypoints_batch:
        return []
    waypoint_count = min(
        STOP_WAYPOINTS_PER_DECISION,
        min(len(waypoints) for waypoints in waypoints_batch),
    )
    return [
        [[waypoints[waypoint_index]] for waypoints in waypoints_batch]
        for waypoint_index in range(waypoint_count)
    ]


def execute_stop_waypoint_chunks(eval_env, waypoints_batch: Sequence[Sequence]):
    """Execute a Stop decision through the same one-waypoint API as Continuous.

    ``makeActionsChunk`` overwrites ``last_action_timings`` on every call, so
    this helper restores per-decision aggregate timings for the existing Fast
    Eval clock and profiler.
    """
    batch_size = len(waypoints_batch)
    aggregate_results = [[] for _ in range(batch_size)]
    aggregate_timings = [
        {"wall_time_ms": 0.0, "sim_time_ms": 0.0}
        for _ in range(batch_size)
    ]
    executed_waypoints = 0

    for waypoint_chunk in stop_single_waypoint_chunks(waypoints_batch):
        chunk_results = eval_env.makeActionsChunk(waypoint_chunk, target_idx=1)
        executed_waypoints += 1
        for batch_index in range(batch_size):
            aggregate_results[batch_index].extend(chunk_results[batch_index])
            timing = eval_env.last_action_timings[batch_index]
            aggregate_timings[batch_index]["wall_time_ms"] += float(
                timing.get("wall_time_ms", 0.0)
            )
            aggregate_timings[batch_index]["sim_time_ms"] += float(
                timing.get("sim_time_ms", 0.0)
            )

        sim_states = getattr(eval_env, "sim_states", None)
        if sim_states and all(
            bool(getattr(state, "is_end", False))
            or bool(getattr(state, "is_collisioned", False))
            for state in sim_states
        ):
            break

    eval_env.last_action_timings = aggregate_timings
    return aggregate_results, executed_waypoints


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
