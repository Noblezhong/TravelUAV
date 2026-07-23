import unittest

import numpy as np

from src.vlnce_src.eval_contract import (
    check_collision_without_tiny_diff,
    control_budget_reached,
    mark_ne_regression_failure,
    stop_executed_waypoint_count,
    stop_executed_waypoints,
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
