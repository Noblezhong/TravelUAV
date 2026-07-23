import unittest

from scripts.compare_trajcorr_ablation import mcnemar_exact, paired_bootstrap


class CompareTrajCorrAblationTest(unittest.TestCase):
    def test_mcnemar_counts_paired_success_changes(self):
        off = {
            "a": {"success": True},
            "b": {"success": False},
            "c": {"success": False},
        }
        on = {
            "a": {"success": False},
            "b": {"success": True},
            "c": {"success": True},
        }
        result = mcnemar_exact(off, on, ["a", "b", "c"])
        self.assertEqual(result["off_success_on_failure"], 1)
        self.assertEqual(result["off_failure_on_success"], 2)
        self.assertEqual(result["discordant_pairs"], 3)

    def test_bootstrap_reports_on_minus_off(self):
        result = paired_bootstrap(
            {"a": 10.0, "b": 20.0},
            {"a": 8.0, "b": 17.0},
            samples=100,
            seed=0,
        )
        self.assertEqual(result["count"], 2)
        self.assertAlmostEqual(result["mean_difference_on_minus_off"], -2.5)


if __name__ == "__main__":
    unittest.main()
