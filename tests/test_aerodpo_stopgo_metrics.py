import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.argv = [sys.argv[0]]

_METRICS_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compute_metrics.py"
_SPEC = importlib.util.spec_from_file_location("compute_metrics", _METRICS_PATH)
compute_metrics = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compute_metrics)


class StopGoMetricsTest(unittest.TestCase):
    def test_stopgo_shift_na_does_not_emit_missing_shift_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root / "profile.jsonl"
            profile.write_text(
                json.dumps({
                    "record_type": "aerodpo_action",
                    "seq_names": ["episode-1"],
                    "action_sim_time_ms": 100.0,
                }) + "\n" + json.dumps({
                    "record_type": "episode_end",
                    "seq_names": ["episode-1"],
                    "success": False,
                    "oracle_success": False,
                    "collision": False,
                    "final_ne_m": 30.0,
                    "control_steps": 1,
                    "logical_elapsed_ms": 200.0,
                }) + "\n",
                encoding="utf-8",
            )
            invalid = []
            terminals, events, records = compute_metrics.extract_action_line(
                profile, invalid, "stopgo", "e2e-action"
            )
            result = compute_metrics.build_result(
                profile, "stopgo", terminals, events, records, invalid, root
            )

        self.assertFalse(
            any("without shift samples" in warning for warning in result["warnings"])
        )
        self.assertIsNone(result["summary"]["time_shift_s"])
        self.assertIsNone(result["summary"]["state_shift_m"])
