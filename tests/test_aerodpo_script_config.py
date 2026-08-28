import unittest
from pathlib import Path


class AeroDPOScriptConfigTest(unittest.TestCase):
    @staticmethod
    def _script_text():
        root = Path(__file__).resolve().parents[1]
        return (root / "scripts" / "aerodpo_eval.sh").read_text(encoding="utf-8")

    def test_model_paths_are_overridable_without_committing_weights(self):
        script = self._script_text()
        self.assertIn("AERODPO_LORA_PATH", script)
        self.assertIn("AERODPO_BASE_MODEL_PATH", script)
        self.assertIn('--model_path "${aerodpo_lora_path}"', script)
        self.assertIn('--base_model_path "${aerodpo_base_model_path}"', script)

    def test_formal_eval_outputs_are_ignored(self):
        root = Path(__file__).resolve().parents[1]
        gitignore = (root / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("eval_aerodpo_stopgo*/", gitignore)
