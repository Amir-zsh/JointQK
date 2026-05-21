# Vendored from kvpress

This tree is a vendored copy of [`kvpress/evaluation/benchmarks/`](https://github.com/NVIDIA/kvpress) — scorers and dataset-prep scripts for 9 long-context benchmarks (LongBench, LongBench-v2, RULER, Loogle, InfiniteBench, NIAH, AIME25, Math500, ZeroScrolls).

- **Source**: <https://github.com/NVIDIA/kvpress>
- **Upstream path**: `kvpress/evaluation/benchmarks/`
- **Snapshot**: local checkout at commit `d8349a952e31b16642eaa962c56eecb10c3873ca` (2026-04-13)
- **License**: Apache-2.0 — see `LICENSE` and `NOTICE`. Per-file SPDX headers preserved verbatim.

## Modifications

Everything under the per-benchmark subdirectories (`aime25/`, `infinite_bench/`, `longbench/`, `longbenchv2/`, `loogle/`, `math500/`, `needle_in_haystack/`, `ruler/`, `zero_scrolls/`) is byte-identical to upstream — `calculate_metrics.py`, `create_huggingface_dataset.py`, `utils.py`, READMEs, and `__init__.py` files are all unchanged.

Two files are new or modified at this tree's root:

- **`evaluate_registry.py`** — trimmed copy of `kvpress/evaluation/evaluate_registry.py`. Changes:
  - `PRESS_REGISTRY` and the `from kvpress import ...` block were removed (press-side compression is orthogonal to our pipeline).
  - Scorer import paths rewritten from `benchmarks.<name>.calculate_metrics` (flat `sys.path`-based imports used by upstream's `evaluate.py`) to `kvq.benchmarks.<name>.calculate_metrics` (absolute imports now that this tree is a Python package).
  - `DATASET_REGISTRY` and `SCORER_REGISTRY` contents are unchanged.

- **`__init__.py`** (top-level) — new file; upstream ships only per-subdirectory `__init__.py`s.

## Usage

Scorers are looked up by dataset name:

```python
from kvq.benchmarks.evaluate_registry import SCORER_REGISTRY
scorer = SCORER_REGISTRY["longbench-e"]
scores = scorer(df)  # df has per-benchmark expected columns; see kvpress/evaluation/evaluate.py for the shape
```

`pipelines/evaluate_generations.py` uses this registry directly.

A byte-equal-to-upstream regression test (`tests/test_evaluation.py::test_vendored_longbench_matches_upstream_kvpress`) guards against silent drift if this directory is re-vendored.

## Re-vendoring from upstream

```bash
# From the repo root, assuming ../kvpress is the upstream checkout.
rm -rf src/kvq/benchmarks/{aime25,infinite_bench,longbench,longbenchv2,loogle,math500,needle_in_haystack,ruler,zero_scrolls}
cp -r kvpress/evaluation/benchmarks/* src/kvq/benchmarks/
# Then manually re-apply the evaluate_registry.py modifications above
# and update the commit SHA in NOTICE + this file.
```

## Citation

From `kvpress/CITATION.cff`:

> Simon Jegou, Maximilian Jeblick, Alessio Devoto. *Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution*. 2025. arXiv:[2510.00636](https://arxiv.org/abs/2510.00636). <https://github.com/NVIDIA/kvpress>
