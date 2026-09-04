"""Keep unused bitsandbytes auto-discovery out of the NCN evaluator."""

import importlib.util
import os


if os.environ.get("NCN_DISABLE_BITSANDBYTES", "0") == "1":
    _find_spec = importlib.util.find_spec

    def _find_spec_without_bitsandbytes(name, package=None):
        if name == "bitsandbytes" or name.startswith("bitsandbytes."):
            return None
        return _find_spec(name, package)

    importlib.util.find_spec = _find_spec_without_bitsandbytes
