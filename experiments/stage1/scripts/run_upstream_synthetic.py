"""Tier 0.1 — Run upstream's synthetic test_turboquant.

Upstream's test imports `from turboquant import ...` because their package is
named `turboquant`. Our vendored copy is symlinked as `turboquant_pytorch`
(directory has a hyphen). Alias the module name and exec the test file in a
fresh namespace so it can be run unchanged.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import turboquant_pytorch  # noqa: F401

sys.modules["turboquant"] = sys.modules["turboquant_pytorch"]

UPSTREAM_TEST = REPO / "turboquant_pytorch" / "test_turboquant.py"
runpy.run_path(str(UPSTREAM_TEST), run_name="__main__")
