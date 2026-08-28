import sys
import unittest
from types import SimpleNamespace

import torch

sys.argv = [sys.argv[0]]

from src.model_wrapper.aerodpo_eval_wrapper import AeroDPOEvalModelWrapper


class _Model:
    def generate(self, **_kwargs):
        return torch.tensor([[10, 11, 98, 49, 49]])


class _Processor:
    tokenizer = SimpleNamespace(eos_token_id=0)

    def decode(self, _token_ids, skip_special_tokens=True):
        return "Action: 98 49 49"


class AeroDPOProfileWrapperTest(unittest.TestCase):
    def test_profiles_only_generate_and_returns_native_action(self):
        wrapper = AeroDPOEvalModelWrapper.__new__(AeroDPOEvalModelWrapper)
        wrapper.model = _Model()
        wrapper.processor = _Processor()
        wrapper._sync_cuda = lambda: None

        actions, stops, profile = wrapper.run_profiled(
            {"input_ids": torch.tensor([[10, 11]])}
        )

        self.assertEqual(stops, [False])
        self.assertAlmostEqual(actions[0]["fwd"], 5.0)
        self.assertEqual(profile["model_output_text"], ["Action: 98 49 49"])
        self.assertGreater(profile["model_inference_latency_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
