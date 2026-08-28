import sys
import unittest

sys.argv = [sys.argv[0]]

from src.common.param import ModelArguments


class ModelArgumentsTest(unittest.TestCase):
    def test_aerodpo_base_model_path_is_available(self):
        self.assertIsNone(ModelArguments().base_model_path)
