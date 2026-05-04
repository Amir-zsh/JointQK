"""Tier 1.1 — Run upstream's validate_v3 unchanged (TurboQuant only).

Same import-alias trick as run_upstream_synthetic.py — upstream's script does
`from turboquant.compressors_v3 import TurboQuantV3` and we have the package
as `turboquant_pytorch`. Alias the module name and exec.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import turboquant_pytorch  # noqa: F401
import turboquant_pytorch.compressors  # noqa: F401
import turboquant_pytorch.compressors_v3  # noqa: F401

sys.modules["turboquant"] = sys.modules["turboquant_pytorch"]
sys.modules["turboquant.compressors"] = sys.modules["turboquant_pytorch.compressors"]
sys.modules["turboquant.compressors_v3"] = sys.modules["turboquant_pytorch.compressors_v3"]

UPSTREAM_VALIDATE = REPO / "turboquant_pytorch" / "validate_v3.py"
runpy.run_path(str(UPSTREAM_VALIDATE), run_name="__main__")
