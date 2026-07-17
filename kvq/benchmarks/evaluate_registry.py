# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Trimmed shim over vendor/kvpress/evaluation/evaluate_registry.py.
# - Scorers are imported directly from the vendored kvpress copy (no second-source
#   duplication of calculate_metrics.py files).
# - PRESS_REGISTRY and `from kvpress import ...` block are intentionally omitted —
#   press wiring lives in kvq, not here.

import kvq.benchmarks  # noqa: F401  — side-effect: adds vendor/kvpress/evaluation/ to sys.path

from kvq.benchmarks.gpqa_adapter import calculate_metrics as gpqa_scorer  # noqa: E402
from kvq.benchmarks.math_verify_scorer import calculate_metrics as math_verify_scorer  # noqa: E402
from benchmarks.infinite_bench.calculate_metrics import calculate_metrics as infinite_bench_scorer  # noqa: E402
from benchmarks.longbench.calculate_metrics import calculate_metrics as longbench_scorer  # noqa: E402
from benchmarks.longbench.calculate_metrics import calculate_metrics_e as longbench_scorer_e  # noqa: E402
from benchmarks.longbenchv2.calculate_metrics import calculate_metrics as longbenchv2_scorer  # noqa: E402
from benchmarks.loogle.calculate_metrics import calculate_metrics as loogle_scorer  # noqa: E402
from benchmarks.needle_in_haystack.calculate_metrics import calculate_metrics as needle_in_haystack_scorer  # noqa: E402
from benchmarks.ruler.calculate_metrics import calculate_metrics as ruler_scorer  # noqa: E402
from benchmarks.zero_scrolls.calculate_metrics import calculate_metrics as zero_scrolls_scorer  # noqa: E402


import pathlib as _pathlib

_REPO_ROOT = _pathlib.Path(__file__).resolve().parents[2]

DATASET_REGISTRY = {
    "loogle": "simonjegou/loogle",
    "ruler": "simonjegou/ruler",
    # RULER-NIAH regenerated at 8k-64k (simonjegou/ruler caps at 16k): Samuel's
    # gen_niah rows (see third_party/samuel_vq/PROVENANCE.md), converted to a
    # local parquet folder; data_dir = context length, same schema as ruler.
    "niah": str(_REPO_ROOT / "artifacts" / "niah_bench"),
    "zero_scrolls": "simonjegou/zero_scrolls",
    "infinitebench": "MaxJeblick/InfiniteBench",
    "longbench": "Xnhyacinth/LongBench",
    "longbench-e": "Xnhyacinth/LongBench",
    "longbench-v2": "simonjegou/LongBench-v2",
    "needle_in_haystack": "alessiodevoto/paul_graham_essays",
    "aime25": "alessiodevoto/aime25",
    "math500": "alessiodevoto/math500",
    # simple-evals CSV via kvq.benchmarks.gpqa_adapter.load_gpqa_df (not a HF
    # dataset id) — the exporter special-cases this key.
    "gpqa": "gpqa_adapter:diamond",
}

SCORER_REGISTRY = {
    "loogle": loogle_scorer,
    "ruler": ruler_scorer,
    "niah": ruler_scorer,
    "zero_scrolls": zero_scrolls_scorer,
    "infinitebench": infinite_bench_scorer,
    "longbench": longbench_scorer,
    "longbench-e": longbench_scorer_e,
    "longbench-v2": longbenchv2_scorer,
    "needle_in_haystack": needle_in_haystack_scorer,
    # math tasks score with HF math-verify (symbolic equivalence); the vendored
    # regex accuracy is reported alongside as accuracy_boxed_exact.
    "aime25": math_verify_scorer,
    "math500": math_verify_scorer,
    "gpqa": gpqa_scorer,
}
