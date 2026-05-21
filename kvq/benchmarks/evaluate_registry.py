# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Trimmed shim over vendor/kvpress/evaluation/evaluate_registry.py.
# - Scorers are imported directly from the vendored kvpress copy (no second-source
#   duplication of calculate_metrics.py files).
# - PRESS_REGISTRY and `from kvpress import ...` block are intentionally omitted —
#   press wiring lives in kvq, not here.

import kvq.benchmarks  # noqa: F401  — side-effect: adds vendor/kvpress/evaluation/ to sys.path

from benchmarks.aime25.calculate_metrics import calculate_metrics as aime25_scorer  # noqa: E402
from benchmarks.infinite_bench.calculate_metrics import calculate_metrics as infinite_bench_scorer  # noqa: E402
from benchmarks.longbench.calculate_metrics import calculate_metrics as longbench_scorer  # noqa: E402
from benchmarks.longbench.calculate_metrics import calculate_metrics_e as longbench_scorer_e  # noqa: E402
from benchmarks.longbenchv2.calculate_metrics import calculate_metrics as longbenchv2_scorer  # noqa: E402
from benchmarks.loogle.calculate_metrics import calculate_metrics as loogle_scorer  # noqa: E402
from benchmarks.math500.calculate_metrics import calculate_metrics as math500_scorer  # noqa: E402
from benchmarks.needle_in_haystack.calculate_metrics import calculate_metrics as needle_in_haystack_scorer  # noqa: E402
from benchmarks.ruler.calculate_metrics import calculate_metrics as ruler_scorer  # noqa: E402
from benchmarks.zero_scrolls.calculate_metrics import calculate_metrics as zero_scrolls_scorer  # noqa: E402


DATASET_REGISTRY = {
    "loogle": "simonjegou/loogle",
    "ruler": "simonjegou/ruler",
    "zero_scrolls": "simonjegou/zero_scrolls",
    "infinitebench": "MaxJeblick/InfiniteBench",
    "longbench": "Xnhyacinth/LongBench",
    "longbench-e": "Xnhyacinth/LongBench",
    "longbench-v2": "simonjegou/LongBench-v2",
    "needle_in_haystack": "alessiodevoto/paul_graham_essays",
    "aime25": "alessiodevoto/aime25",
    "math500": "alessiodevoto/math500",
}

SCORER_REGISTRY = {
    "loogle": loogle_scorer,
    "ruler": ruler_scorer,
    "zero_scrolls": zero_scrolls_scorer,
    "infinitebench": infinite_bench_scorer,
    "longbench": longbench_scorer,
    "longbench-e": longbench_scorer_e,
    "longbench-v2": longbenchv2_scorer,
    "needle_in_haystack": needle_in_haystack_scorer,
    "aime25": aime25_scorer,
    "math500": math500_scorer,
}
