import unittest

from src.vlnce_src.trajcorr_runtime import (
    ContinuousRequestCounter,
    TRAJECTORY_CORRECTED,
    TRAJECTORY_ORIGINAL,
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

    def test_corrected_execution_uses_same_global_counter(self):
        counter = ContinuousRequestCounter(request_interval=5)
        triggers = [counter.mark_normal_execution() for _ in range(5)]
        self.assertEqual(triggers, [False, False, False, False, True])
        counter.mark_request_submitted()
        self.assertEqual(counter.count, 0)


if __name__ == "__main__":
    unittest.main()
