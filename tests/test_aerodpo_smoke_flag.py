import sys
import unittest

sys.argv = [sys.argv[0]]

from src.common.param import CommonArguments


class AeroDPOSmokeFlagTest(unittest.TestCase):
    def test_nonstandard_episode_count_is_disabled_by_default(self):
        self.assertFalse(CommonArguments().allow_nonstandard_eval_count)
