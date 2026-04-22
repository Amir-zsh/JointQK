from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from experiments.stage1.benchmarks.evaluate_registry import DATASET_REGISTRY, SCORER_REGISTRY
from experiments.stage1.benchmarks.longbench.calculate_metrics import calculate_metrics_e
from experiments.stage1.data import build_kvpress_dataset_spec, get_dataset_spec


EXPECTED_REGISTRY_KEYS = {
    "aime25",
    "infinitebench",
    "longbench",
    "longbench-e",
    "longbench-v2",
    "loogle",
    "math500",
    "needle_in_haystack",
    "ruler",
    "zero_scrolls",
}


def test_scorer_registry_has_expected_keys():
    assert set(SCORER_REGISTRY) == EXPECTED_REGISTRY_KEYS


def test_dataset_registry_keys_match_scorer_registry():
    assert set(DATASET_REGISTRY) == set(SCORER_REGISTRY)


def test_build_kvpress_dataset_spec_sets_hf_path_from_registry():
    spec = build_kvpress_dataset_spec(
        name="longbench-e",
        config_names=("qasper_e",),
        metadata_fields=("task", "answers", "length"),
    )
    assert spec.hf_path == "Xnhyacinth/LongBench"
    assert "max_new_tokens" in spec.metadata_fields
    assert "task" in spec.metadata_fields


def test_build_kvpress_dataset_spec_rejects_unknown_name():
    with pytest.raises(KeyError):
        build_kvpress_dataset_spec(
            name="not-a-real-benchmark",
            config_names=(),
            metadata_fields=(),
        )


def test_registered_longbench_e_spec_has_kvpress_fields():
    spec = get_dataset_spec("longbench-e")
    assert spec.hf_path == "Xnhyacinth/LongBench"
    for required in ("task", "answers", "length", "all_classes", "max_new_tokens"):
        assert required in spec.metadata_fields, f"missing {required}"


def _tiny_longbench_e_df() -> pd.DataFrame:
    # Three rows, lengths 2000 / 6000 / 10000 hit all three buckets (0-4k, 4-8k, 8k+).
    # task="qasper" => QA F1 scorer. One perfect match, one mismatch, one verbose.
    return pd.DataFrame(
        [
            {
                "predicted_answer": "Paris",
                "answers": ["Paris"],
                "task": "qasper",
                "all_classes": None,
                "length": 2000,
            },
            {
                "predicted_answer": "blue",
                "answers": ["red"],
                "task": "qasper",
                "all_classes": None,
                "length": 6000,
            },
            {
                "predicted_answer": "the answer is Tokyo",
                "answers": ["Tokyo"],
                "task": "qasper",
                "all_classes": None,
                "length": 10000,
            },
        ]
    )


def test_longbench_e_scorer_returns_bucketed_scores():
    result = SCORER_REGISTRY["longbench-e"](_tiny_longbench_e_df())
    assert set(result) == {"0-4k", "4-8k", "8k+"}
    for value in result.values():
        assert isinstance(value, float)
        assert 0.0 <= value <= 100.0


def test_vendored_longbench_matches_upstream_kvpress():
    """Lock the vendored copy against kvpress to catch silent drift if we re-vendor."""
    upstream_path = Path("/vault/amir/efficient-llm/teamily-project/kvpress/evaluation")
    sys.path.insert(0, str(upstream_path))
    try:
        from benchmarks.longbench.calculate_metrics import (
            calculate_metrics_e as upstream_calculate_metrics_e,
        )
    finally:
        sys.path.remove(str(upstream_path))

    df = _tiny_longbench_e_df()
    ours = calculate_metrics_e(df.copy())
    upstream = upstream_calculate_metrics_e(df.copy())
    assert ours == upstream


def test_evaluator_end_to_end_smoke(tmp_path: Path):
    stats_dir = tmp_path / "stats"
    (stats_dir / "examples").mkdir(parents=True)

    manifest = {
        "dataset": "longbench-e",
        "configs": ["qasper", "hotpotqa", "passage_retrieval_en"],
        "num_examples": 3,
        "examples": [
            {
                "file": "examples/ex_000.pt",
                "dataset": "longbench-e",
                "config": "qasper_e",
                "row_index": 7,
                "prompt_length": 4000,
                "total_length": 4032,
                "captured_length": 4031,
                "n_generated": 32,
                "max_new_tokens_used": 148,
                "generated_text": "Paris",
                "metadata": {
                    "answers": ["Paris"],
                    "task": "qasper",
                    "length": 4000,
                    "all_classes": None,
                    "max_new_tokens": 148,
                },
            },
            {
                "file": "examples/ex_001.pt",
                "dataset": "longbench-e",
                "config": "hotpotqa_e",
                "row_index": 11,
                "prompt_length": 6500,
                "total_length": 6532,
                "captured_length": 6531,
                "n_generated": 32,
                "max_new_tokens_used": 52,
                "generated_text": "Barack Obama",
                "metadata": {
                    "answers": ["Barack Obama"],
                    "task": "hotpotqa",
                    "length": 6500,
                    "all_classes": None,
                    "max_new_tokens": 52,
                },
            },
            {
                "file": "examples/ex_002.pt",
                "dataset": "longbench-e",
                "config": "passage_retrieval_en_e",
                "row_index": 3,
                "prompt_length": 9000,
                "total_length": 9032,
                "captured_length": 9031,
                "n_generated": 32,
                "max_new_tokens_used": 52,
                "generated_text": "The answer is Paragraph 7",
                "metadata": {
                    "answers": ["Paragraph 7"],
                    "task": "passage_retrieval_en",
                    "length": 9000,
                    "all_classes": None,
                    "max_new_tokens": 52,
                },
            },
        ],
        "config": {},
    }
    (stats_dir / "manifest.json").write_text(json.dumps(manifest))

    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.stage1.evaluate_generations",
            "--stats_dir",
            str(stats_dir),
        ],
        cwd="/vault/amir/efficient-llm/teamily-project",
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr

    evaluation = json.loads((stats_dir / "evaluation" / "evaluation.json").read_text())
    assert evaluation["dataset"] == "longbench-e"
    # Three configs → three entries. Each is a bucket dict from calculate_metrics_e.
    assert set(evaluation["per_config"]) == {"qasper_e", "hotpotqa_e", "passage_retrieval_en_e"}
    for config_result in evaluation["per_config"].values():
        assert set(config_result) == {"0-4k", "4-8k", "8k+"}

    summary = (stats_dir / "evaluation" / "summary.md").read_text()
    assert "# Evaluation: longbench-e" in summary
    assert "## qasper_e" in summary
