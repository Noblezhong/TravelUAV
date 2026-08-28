import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.argv = [sys.argv[0]]

from src.vlnce_src.aerodpo_stopgo_eval import _validate_seen_split


class AeroDPOSplitValidationTest(unittest.TestCase):
    def test_smoke_split_can_opt_out_of_1418_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            split_path = Path(tmp) / "smoke.json"
            split_path.write_text(json.dumps([{"json": "one/merged_data.json"}]))
            _validate_seen_split(split_path, allow_nonstandard=True)
