# `kvq.benchmarks` — thin shim over kvpress evaluation/benchmarks

This package no longer ships its own copy of the per-benchmark scorers.
The actual scorer code lives in
[`vendor/kvpress/evaluation/benchmarks/`](../../../vendor/kvpress/evaluation/benchmarks/)
(Apache-2.0, NVIDIA).

## What this directory contains

- **`__init__.py`** — at import time, inserts
  `vendor/kvpress/evaluation/` into `sys.path` so the flat
  `from benchmarks.<name>.calculate_metrics import ...` form (which
  upstream's own `evaluate.py` uses) resolves to the vendored copy.

- **`evaluate_registry.py`** — trimmed copy of
  `vendor/kvpress/evaluation/evaluate_registry.py` with the
  press-registry block removed (press wiring lives in `kvq`,
  not here). Re-exports `DATASET_REGISTRY` and `SCORER_REGISTRY` for
  in-project callers (`analysis/*`, `kvq.data.kvpress_adapter`,
  `tests/test_evaluation.py`).

- **`LICENSE`, `NOTICE`** — kept for attribution; the SPDX headers in
  the upstream scorers under `vendor/kvpress/` already carry the same
  licence text.

## Why a shim and not the upstream registry directly?

Two reasons:
1. The upstream registry imports `PRESS_REGISTRY` from `kvpress`, which
   pulls in the entire press hierarchy and KVzipPress's stdout banner.
   Our pipelines only want scorers, not presses.
2. The upstream registry expects `benchmarks/` to be on `sys.path`
   (`evaluate.py` does this implicitly via cwd). The shim normalises
   that into a clean `from kvq.benchmarks.evaluate_registry import ...`
   for in-project consumers.

## Usage

```python
from kvq.benchmarks.evaluate_registry import SCORER_REGISTRY, DATASET_REGISTRY

scorer = SCORER_REGISTRY["longbench-e"]
scores = scorer(df)  # df has per-benchmark expected columns
```

## Updating the upstream snapshot

There is nothing local to update — the scorers ARE the vendored copy.
To take a newer upstream commit, just bump `vendor/kvpress/`:

```bash
git -C vendor/kvpress fetch origin
git -C vendor/kvpress checkout <commit-or-tag>
```

Re-run `tests/test_evaluation.py::test_longbench_e_scorer_returns_bucketed_scores`
and the bench smoke to confirm no behavioural regressions.
