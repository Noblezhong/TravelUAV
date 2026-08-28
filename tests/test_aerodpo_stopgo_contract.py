import unittest

from src.vlnce_src.aerodpo_eval_contract import (
    AERODPO_SUCCESS_DISTANCE_M,
    parse_aerodpo_action_text,
    resolve_aerodpo_stop,
    summarize_latencies,
)


class AeroDPOStopContractTest(unittest.TestCase):
    def test_land_inside_success_radius_is_success(self):
        self.assertEqual(
            resolve_aerodpo_stop(model_stop=True, distance_m=AERODPO_SUCCESS_DISTANCE_M),
            "success",
        )

    def test_land_outside_success_radius_is_immediate_early_end(self):
        self.assertEqual(
            resolve_aerodpo_stop(model_stop=True, distance_m=20.01),
            "early_end",
        )

    def test_non_stop_never_terminates_from_land_contract(self):
        self.assertEqual(resolve_aerodpo_stop(model_stop=False, distance_m=1.0), "continue")


class AeroDPOActionParserTest(unittest.TestCase):
    def test_parses_and_dequantizes_native_bins(self):
        action, should_stop = parse_aerodpo_action_text("Action: 98 49 49")
        self.assertAlmostEqual(action["fwd"], 5.0)
        self.assertAlmostEqual(action["down"], 0.0)
        self.assertAlmostEqual(action["yaw"], 0.0)
        self.assertFalse(should_stop)

    def test_land_token_stops_even_when_action_text_contains_numbers(self):
        _, should_stop = parse_aerodpo_action_text("<LAND> 98 98 98")
        self.assertTrue(should_stop)


class AeroDPOLatencySummaryTest(unittest.TestCase):
    def test_reports_decision_and_episode_balanced_means(self):
        summary = summarize_latencies([[10.0, 30.0], [100.0]])
        self.assertEqual(summary["count"], 3)
        self.assertAlmostEqual(summary["mean"], 140.0 / 3.0)
        self.assertAlmostEqual(summary["episode_balanced_mean"], 60.0)
        self.assertEqual(summary["first_call_ms"], 10.0)
