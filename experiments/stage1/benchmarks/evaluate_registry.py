# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Trimmed copy of kvpress/evaluation/evaluate_registry.py — only DATASET_REGISTRY
# and SCORER_REGISTRY are kept. PRESS_REGISTRY is omitted (kvpress-side compression,
# orthogonal to our pipeline). Scorer imports are rewritten to point into the
# vendored experiments.stage1.benchmarks tree.

from experiments.stage1.benchmarks.aime25.calculate_metrics import calculate_metrics as aime25_scorer
from experiments.stage1.benchmarks.infinite_bench.calculate_metrics import calculate_metrics as infinite_bench_scorer
from experiments.stage1.benchmarks.longbench.calculate_metrics import calculate_metrics as longbench_scorer
from experiments.stage1.benchmarks.longbench.calculate_metrics import calculate_metrics_e as longbench_scorer_e
from experiments.stage1.benchmarks.longbenchv2.calculate_metrics import calculate_metrics as longbenchv2_scorer
from experiments.stage1.benchmarks.loogle.calculate_metrics import calculate_metrics as loogle_scorer
from experiments.stage1.benchmarks.math500.calculate_metrics import calculate_metrics as math500_scorer
from experiments.stage1.benchmarks.needle_in_haystack.calculate_metrics import calculate_metrics as needle_in_haystack_scorer
from experiments.stage1.benchmarks.ruler.calculate_metrics import calculate_metrics as ruler_scorer
from experiments.stage1.benchmarks.zero_scrolls.calculate_metrics import calculate_metrics as zero_scrolls_scorer


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
