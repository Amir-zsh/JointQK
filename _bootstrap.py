"""sys.path bootstrap — makes vendored libraries importable without `pip install`.

Each entry-point script under pipelines/ and tests/ starts with:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[N]))  # N: depth to repo root
    import _bootstrap  # noqa: F401  (adds vendor/kvpress to sys.path)

The script's own header puts the **project root** on sys.path so `import kvq.*`
and `import pipelines.*` resolve. This module appends **vendor/kvpress/** so
`import kvpress` resolves through the upstream-vendored copy.

No pip install, no editable mode, no PYTHONPATH — just two `sys.path` entries
that every entry-point script declares for itself.
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent
for _path in (_REPO / "vendor" / "kvpress",):
    _p = str(_path)
    if _p not in sys.path:
        sys.path.insert(0, _p)
