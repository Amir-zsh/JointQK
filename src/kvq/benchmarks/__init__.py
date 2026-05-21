"""Thin shim over vendor/kvpress/evaluation/benchmarks/ (Apache-2.0, NVIDIA).

The actual per-benchmark scorers live in
`vendor/kvpress/evaluation/benchmarks/<name>/calculate_metrics.py`. This
package adds the upstream's `evaluation/` dir to `sys.path` so that
`from benchmarks.<name>.calculate_metrics import ...` (the flat-import
form upstream itself uses) resolves to the vendored copy.

See VENDORED.md for provenance and the trimmed-vs-upstream diff.
"""

import sys
from pathlib import Path

_KVPRESS_EVALUATION = Path(__file__).resolve().parents[3] / "vendor" / "kvpress" / "evaluation"
if _KVPRESS_EVALUATION.is_dir() and str(_KVPRESS_EVALUATION) not in sys.path:
    sys.path.insert(0, str(_KVPRESS_EVALUATION))
