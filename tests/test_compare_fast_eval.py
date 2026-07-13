import unittest

from scripts.compare_fast_eval import apply_steps, episode_outcomes, relative_error


class CompareFastEvalTest(unittest.TestCase):
    def test_aligns_request_application_steps(self):
        records = [
            {
                "seq_names": ["episode-a"],
                "request_id": 3,
                "result_applied_exec_step": 9,
            }
        ]
        self.assertEqual(apply_steps(records), {("episode-a", 3): 9})

    def test_extracts_episode_outcome(self):
        records = [
            {
                "record_type": "episode_end",
                "seq_names": ["episode-a"],
                "success": True,
                "oracle_success": True,
                "collision": False,
            }
        ]
        self.assertEqual(episode_outcomes(records)["episode-a"], (True, True, False))

    def test_relative_error(self):
        self.assertAlmostEqual(relative_error(100.0, 104.0), 0.04)


if __name__ == "__main__":
    unittest.main()
