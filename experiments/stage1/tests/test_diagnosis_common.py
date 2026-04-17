from __future__ import annotations

from pathlib import Path

import torch

from experiments.stage1.diagnosis_common import (
    aggregate_metric_rows,
    build_correlation_table,
    effective_rank_from_states,
    instrument_compressor_path,
    participation_ratio_from_matrix,
    run_baseline_path,
    run_current_oracle_path,
    transform_matrix_stats,
    variance_spread,
)
from experiments.stage1.common import Stage1MSECompressor


def test_variance_spread_and_participation_ratio_helpers():
    variances = torch.tensor([1.0, 2.0, 4.0, 8.0])
    assert abs(variance_spread(variances) - 8.0) < 1e-6

    matrix = torch.diag(torch.tensor([4.0, 1.0, 0.0]))
    ratio = participation_ratio_from_matrix(matrix)
    assert 1.0 < ratio < 2.0


def test_transform_matrix_stats_report_condition_numbers():
    factors = torch.stack([torch.diag(torch.tensor([4.0, 2.0, 1.0]))], dim=0)
    metrics = torch.stack([torch.diag(torch.tensor([16.0, 4.0, 1.0]))], dim=0)
    stats = transform_matrix_stats(factors, metrics)
    assert abs(stats["transform_condition_number_mean"] - 4.0) < 1e-5
    assert abs(stats["metric_condition_number_mean"] - 16.0) < 1e-4


def test_instrument_compressor_path_emits_expected_diagnostics():
    states = torch.randn(1, 2, 5, 8)
    compressor = Stage1MSECompressor(head_dim=8, bits=3, seed=17, device="cpu")
    result = instrument_compressor_path(states, compressor)
    assert result["reconstructed"].shape == states.shape
    assert "pre_norm_cv" in result["diagnostics"]
    assert "normalized_effective_rank" in result["diagnostics"]
    assert "rotated_variance_spread" in result["diagnostics"]
    assert result["diagnostics"]["rotated_variance_spread"] >= 1.0


def test_baseline_and_oracle_diagnostics_paths_run():
    prefix_keys = torch.randn(1, 2, 7, 8)
    future_queries = torch.randn(1, 4, 3, 8)
    metrics = torch.stack([torch.eye(8), 2.0 * torch.eye(8)], dim=0)

    baseline = run_baseline_path(prefix_keys, future_queries, metrics, bits=2, seed=7)
    oracle = run_current_oracle_path(prefix_keys, future_queries, metrics, bits=2, seed=7, eps=1e-6)

    assert baseline["reconstructed"].shape == prefix_keys.shape
    assert oracle["reconstructed"].shape == prefix_keys.shape
    assert "geometry_distortion" in baseline["metrics"]
    assert "transform_condition_number_mean" in oracle["diagnostics"]
    assert oracle["diagnostics"]["transform_condition_number_mean"] >= 1.0


def test_aggregate_rows_and_correlations():
    rows = [
        {
            "bits": 2,
            "layer_idx": 0,
            "baseline_metrics": {"logit_mse": 1.0, "geometry_distortion": 1.0, "top1_match": 0.8, "top5_containment": 0.9, "key_mse": 0.1, "logit_cosine": 0.9},
            "oracle_metrics": {"logit_mse": 2.0, "geometry_distortion": 2.0, "top1_match": 0.7, "top5_containment": 0.85, "key_mse": 0.2, "logit_cosine": 0.85},
            "oracle_diagnostics": {
                "transform_condition_number_mean": 2.0,
                "transformed_norm_cv": 0.2,
                "rotated_variance_spread": 1.2,
                "transformed_effective_rank": 3.0,
            },
        },
        {
            "bits": 2,
            "layer_idx": 1,
            "baseline_metrics": {"logit_mse": 1.5, "geometry_distortion": 1.5, "top1_match": 0.82, "top5_containment": 0.92, "key_mse": 0.15, "logit_cosine": 0.91},
            "oracle_metrics": {"logit_mse": 3.0, "geometry_distortion": 3.0, "top1_match": 0.6, "top5_containment": 0.8, "key_mse": 0.3, "logit_cosine": 0.82},
            "oracle_diagnostics": {
                "transform_condition_number_mean": 4.0,
                "transformed_norm_cv": 0.4,
                "rotated_variance_spread": 1.8,
                "transformed_effective_rank": 2.0,
            },
        },
    ]
    summary = aggregate_metric_rows(rows)
    assert abs(summary["baseline"]["logit_mse"] - 1.25) < 1e-6
    correlations = build_correlation_table(rows)
    assert "transform_condition_number_mean__vs__delta_logit_mse" in correlations


def test_existing_oracle_artifact_summary_matches_helpers():
    artifact_path = Path("artifacts/stage1/oracle_v3_study_fixed_clean/oracle_study.pt")
    assert artifact_path.exists()
    payload = torch.load(artifact_path, map_location="cpu")

    rows = []
    for example in payload["per_example"]:
        for layer in example["layers"]:
            for bits in [2, 3, 4]:
                rows.append(
                    {
                        "bits": bits,
                        "layer_idx": layer["layer_idx"],
                        "baseline_metrics": layer[f"baseline_{bits}bit"],
                        "oracle_metrics": layer[f"oracle_{bits}bit"],
                        "oracle_diagnostics": {
                            "transform_condition_number_mean": 1.0,
                            "transformed_norm_cv": 0.0,
                            "rotated_variance_spread": 1.0,
                            "transformed_effective_rank": 1.0,
                        },
                    }
                )

    for bits in [2, 3, 4]:
        helper_summary = aggregate_metric_rows(rows, lambda row, bits=bits: int(row["bits"]) == bits)
        expected = payload["summary"][f"baseline_{bits}bit"]["logit_mse"]
        assert abs(helper_summary["baseline"]["logit_mse"] - expected) < 1e-6

    layer0_rows = [row for row in rows if int(row["layer_idx"]) != 0]
    layer0_summary = aggregate_metric_rows(layer0_rows, lambda row: int(row["bits"]) == 4)
    assert abs(layer0_summary["baseline"]["logit_mse"] - 0.1250) < 5e-4
    assert abs(layer0_summary["oracle"]["logit_mse"] - 0.1881) < 5e-4


def test_effective_rank_from_states_is_bounded():
    states = torch.randn(1, 2, 9, 8)
    rank = effective_rank_from_states(states)
    assert 1.0 <= rank <= 8.0
