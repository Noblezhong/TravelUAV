import unittest

import numpy as np

from src.vlnce_src.eval_contract import (
    check_collision_without_tiny_diff,
    control_budget_reached,
    execute_stop_waypoint_chunks,
    mark_ne_regression_failure,
    stop_executed_waypoint_count,
    stop_executed_waypoints,
    stop_single_waypoint_chunks,
)


class EvaluationBudgetTest(unittest.TestCase):
    def test_stop_200_decisions_equal_1000_waypoints(self):
        self.assertEqual(stop_executed_waypoint_count(200), 1000)

    def test_tc_stops_at_1000_control_steps(self):
        self.assertFalse(control_budget_reached(999, 1000))
        self.assertTrue(control_budget_reached(1000, 1000))


class StopOracleWaypointTest(unittest.TestCase):
    def test_unexecuted_p6_p7_are_excluded(self):
        waypoints = [[float(index), 0.0, 0.0] for index in range(1, 8)]
        self.assertEqual(
            stop_executed_waypoints(waypoints),
            waypoints[:5],
        )

    def test_stop_decision_is_split_into_five_single_waypoint_calls(self):
        waypoints = [[[float(index), 0.0, 0.0] for index in range(1, 8)]]
        chunks = stop_single_waypoint_chunks(waypoints)
        self.assertEqual(len(chunks), 5)
        self.assertTrue(all(len(chunk[0]) == 1 for chunk in chunks))
        self.assertEqual(
            [chunk[0][0] for chunk in chunks],
            waypoints[0][:5],
        )

    def test_stop_uses_shared_chunk_controller_and_aggregates_timing(self):
        class FakeEnv:
            def __init__(self):
                self.calls = []
                self.last_action_timings = []
                self.sim_states = []

            def makeActionsChunk(self, waypoint_chunk, target_idx):
                self.calls.append((waypoint_chunk, target_idx))
                self.last_action_timings = [
                    {"wall_time_ms": 2.0, "sim_time_ms": 3.0}
                ]
                return [[{"waypoint": waypoint_chunk[0][0]}]]

        env = FakeEnv()
        waypoints = [[[float(index), 0.0, 0.0] for index in range(1, 7)]]
        results, executed = execute_stop_waypoint_chunks(env, waypoints)
        self.assertEqual(executed, 5)
        self.assertEqual(len(env.calls), 5)
        self.assertTrue(all(target_idx == 1 for _, target_idx in env.calls))
        self.assertEqual(len(results[0]), 5)
        self.assertEqual(env.last_action_timings[0]["wall_time_ms"], 10.0)
        self.assertEqual(env.last_action_timings[0]["sim_time_ms"], 15.0)


class CollisionContractTest(unittest.TestCase):
    def test_stationary_rgb_depth_difference_is_not_a_collision(self):
        previous = [[{"depth": [np.full((2, 2), 10.0)]}]]
        current = [[{"depth": [np.full((2, 2), 10.0)]}]]
        collisions, dones = check_collision_without_tiny_diff(
            previous,
            current,
            [False],
            [False],
        )
        self.assertEqual(collisions, [False])
        self.assertEqual(dones, [False])

    def test_close_depth_is_a_collision(self):
        previous = [[{"depth": [np.full((2, 2), 10.0)]}]]
        current = [[{"depth": [np.zeros((2, 2))]}]]
        collisions, dones = check_collision_without_tiny_diff(
            previous,
            current,
            [False],
            [False],
        )
        self.assertEqual(collisions, [True])
        self.assertEqual(dones, [True])


class NavigationRegressionContractTest(unittest.TestCase):
    def test_ne_regression_is_failure_not_collision(self):
        collisions, dones = mark_ne_regression_failure(
            collisions=[False],
            dones=[False],
            index=0,
        )
        self.assertEqual(collisions, [False])
        self.assertEqual(dones, [True])


if __name__ == "__main__":
    unittest.main()
