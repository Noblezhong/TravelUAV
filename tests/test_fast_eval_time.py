import tempfile
import time
import unittest
from types import SimpleNamespace

from src.vlnce_src.fast_eval_time import (
    FastEvalClock,
    FastResultTiming,
    action_timing,
    configure_fast_eval_output,
)


class FastEvalClockTest(unittest.TestCase):
    def test_fast_clock_advances_original_speed_time(self):
        clock = FastEvalClock(enabled=True, speedup=5)
        clock.advance_action(sim_elapsed_ms=1000, wall_elapsed_ms=200)
        self.assertEqual(clock.now_ms, 1000)
        clock.advance_to(5000)
        self.assertEqual(clock.now_ms, 5000)

    def test_normal_clock_uses_wall_time(self):
        clock = FastEvalClock(enabled=False, speedup=5)
        time.sleep(0.002)
        self.assertGreater(clock.now_ms, 0)

    def test_result_ready_time_includes_real_model_latencies(self):
        timing = FastResultTiming(1000, 5000, 200, 100)
        self.assertEqual(timing.edge_arrival_logical_ms, 6000)
        self.assertEqual(timing.ready_logical_ms, 6300)

    def test_action_timing_uses_sim_time_only_in_fast_mode(self):
        env = SimpleNamespace(last_action_timings=[{"sim_time_ms": 980.0}])
        self.assertEqual(action_timing(env, 205.0, True)["airsim_action_latency_ms"], 980.0)
        self.assertEqual(action_timing(env, 205.0, False)["airsim_action_latency_ms"], 205.0)

    def test_fast_output_is_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(fast_eval=True, fast_eval_speedup=5.0, eval_save_path=f"{tmp}/eval")
            configure_fast_eval_output(args, "continuous")
            self.assertTrue(args.eval_save_path.endswith("_fast_x5"))


if __name__ == "__main__":
    unittest.main()
