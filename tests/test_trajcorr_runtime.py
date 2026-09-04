import unittest
from types import SimpleNamespace

import numpy as np
from src.vlnce_src.trajcorr_apply import derive_trajcorr_inputs
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
    coarse_target_to_world_goal,
    filter_target_lock_waypoints,
    select_trajectory_mode,
)


class TrajectoryDecisionTest(unittest.TestCase):
    def test_small_shift_uses_original_trajectory(self):
        decision = select_trajectory_mode(
            enabled=True,
            observation_pose=[0.0, 0.0, 0.0],
            current_pose=[2.4, 0.0, 0.0],
            correction_threshold_m=2.5,
        )
        self.assertEqual(decision.mode, TRAJECTORY_ORIGINAL)

    def test_threshold_shift_uses_correction(self):
        decision = select_trajectory_mode(
            enabled=True,
            observation_pose=[0.0, 0.0, 0.0],
            current_pose=[2.5, 0.0, 0.0],
            correction_threshold_m=2.5,
        )
        self.assertEqual(decision.mode, TRAJECTORY_CORRECTED)

    def test_off_always_uses_original_trajectory(self):
        decision = select_trajectory_mode(
            enabled=False,
            observation_pose=[0.0, 0.0, 0.0],
            current_pose=[20.0, 0.0, 0.0],
            correction_threshold_m=2.5,
        )
        self.assertEqual(decision.mode, TRAJECTORY_ORIGINAL)


class CoarseFrameTest(unittest.TestCase):
    def test_converts_target_aligned_vector_to_world_goal(self):
        rotation_90_z = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        episode = [
            {"sensors": {"imu": {"rotation": np.eye(3).tolist()}, "state": {"position": [0.0, 0.0, 0.0]}}},
            {"sensors": {"imu": {"rotation": np.eye(3).tolist()}, "state": {"position": [10.0, 20.0, 30.0]}}},
        ]
        goal = coarse_target_to_world_goal(episode, [2.0, 0.0, 0.0], rotation_90_z)
        self.assertEqual(goal, [10.0, 22.0, 30.0])

    def test_uses_explicit_request_frame_not_llm_coordinates(self):
        result = SimpleNamespace(
            coarse_local=[0.0, 0.0, -5.0],
            coarse_goal_world=[100.0, 200.0, -15.0],
            llm_output=[[999.0, 999.0, 999.0]],
        )
        inputs, reason = derive_trajcorr_inputs(result)
        self.assertIsNone(reason)
        self.assertEqual(inputs, ([100.0, 200.0, -15.0], [0.0, 0.0, -5.0]))

    def test_missing_request_frame_falls_back_safely(self):
        inputs, reason = derive_trajcorr_inputs(SimpleNamespace(llm_output=[[1.0, 2.0, 3.0]]))
        self.assertIsNone(inputs)
        self.assertEqual(reason, "missing_coarse_frame")


class ContinuousRequestCounterTest(unittest.TestCase):
    def test_normal_counter_requests_after_five_executions(self):
        counter = ContinuousRequestCounter(request_interval=5)
        triggers = [counter.mark_normal_execution() for _ in range(5)]
        self.assertEqual(triggers, [False, False, False, False, True])

    def test_buffer_replacement_does_not_reset_counter(self):
        counter = ContinuousRequestCounter(request_interval=5)
        counter.mark_request_submitted()
        counter.mark_normal_execution()
        counter.mark_normal_execution()
        self.assertEqual(counter.count, 2)
        self.assertFalse(counter.mark_normal_execution())
        self.assertFalse(counter.mark_normal_execution())
        self.assertTrue(counter.mark_normal_execution())

    def test_submitted_request_resets_counter(self):
        counter = ContinuousRequestCounter(request_interval=5)
        triggers = [counter.mark_normal_execution() for _ in range(5)]
        self.assertEqual(triggers, [False, False, False, False, True])
        counter.mark_request_submitted()
        self.assertEqual(counter.count, 0)

    def test_frozen_execution_does_not_advance_counter(self):
        counter = ContinuousRequestCounter(request_interval=5, count=3)
        self.assertFalse(counter.mark_execution(frozen=True))
        self.assertEqual(counter.count, 3)


class TargetLockLifecycleTest(unittest.TestCase):
    def test_starts_target_lock_and_freezes_counter(self):
        lifecycle = TargetLockLifecycle(goal_radius_m=0.5)
        reason = lifecycle.begin(
            observation_pose=[0.0, 0.0, 0.0],
            current_pose=[2.0, 0.0, 0.0],
            goal_world=[5.0, 0.0, 0.0],
        )
        self.assertIsNone(reason)
        self.assertEqual(lifecycle.phase, PHASE_TARGET_LOCK)
        self.assertTrue(lifecycle.counter_frozen)
        self.assertEqual(lifecycle.goal_world, [5.0, 0.0, 0.0])

    def test_goal_radius_completes_target_lock(self):
        lifecycle = TargetLockLifecycle(goal_radius_m=0.5)
        lifecycle.begin(
            observation_pose=[0.0, 0.0, 0.0],
            current_pose=[2.0, 0.0, 0.0],
            goal_world=[5.0, 0.0, 0.0],
        )
        reason = lifecycle.evaluate([4.6, 0.0, 0.0])
        self.assertEqual(reason, COMPLETION_GOAL_REACHED)
        self.assertEqual(lifecycle.phase, PHASE_WAIT_REFRESH)

    def test_already_passed_goal_never_enters_reverse_lock(self):
        lifecycle = TargetLockLifecycle(goal_radius_m=0.5)
        reason = lifecycle.begin(
            observation_pose=[0.0, 0.0, 0.0],
            current_pose=[6.0, 0.0, 0.0],
            goal_world=[5.0, 0.0, 0.0],
        )
        self.assertEqual(reason, COMPLETION_GOAL_PASSED)
        self.assertEqual(lifecycle.phase, PHASE_WAIT_REFRESH)

    def test_crossing_goal_plane_completes_without_flying_back(self):
        lifecycle = TargetLockLifecycle(goal_radius_m=0.5)
        lifecycle.begin(
            observation_pose=[0.0, 0.0, 0.0],
            current_pose=[2.0, 0.0, 0.0],
            goal_world=[5.0, 0.0, 0.0],
        )
        reason = lifecycle.evaluate([5.8, 0.0, 0.0])
        self.assertEqual(reason, COMPLETION_GOAL_PASSED)
        self.assertEqual(lifecycle.phase, PHASE_WAIT_REFRESH)

    def test_buffer_exhaustion_waits_for_fresh_edge_result(self):
        lifecycle = TargetLockLifecycle(goal_radius_m=0.5)
        lifecycle.begin(
            observation_pose=[0.0, 0.0, 0.0],
            current_pose=[2.0, 0.0, 0.0],
            goal_world=[5.0, 0.0, 0.0],
        )
        reason = lifecycle.mark_buffer_exhausted()
        self.assertEqual(reason, COMPLETION_BUFFER_EXHAUSTED)
        self.assertEqual(lifecycle.phase, PHASE_WAIT_REFRESH)

    def test_fresh_result_restores_normal_mode(self):
        lifecycle = TargetLockLifecycle(goal_radius_m=0.5)
        lifecycle.begin(
            observation_pose=[0.0, 0.0, 0.0],
            current_pose=[2.0, 0.0, 0.0],
            goal_world=[5.0, 0.0, 0.0],
        )
        lifecycle.mark_buffer_exhausted()
        lifecycle.resume_normal()
        self.assertEqual(lifecycle.phase, PHASE_NORMAL)
        self.assertFalse(lifecycle.counter_frozen)
        self.assertIsNone(lifecycle.goal_world)

    def test_filters_backward_and_overshooting_waypoints(self):
        filtered = filter_target_lock_waypoints(
            waypoints=[
                [1.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ],
            current_pose=[0.0, 0.0, 0.0],
            goal_world=[5.0, 0.0, 0.0],
            direction_world=[1.0, 0.0, 0.0],
            goal_radius_m=0.5,
        )
        self.assertEqual(filtered, [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
