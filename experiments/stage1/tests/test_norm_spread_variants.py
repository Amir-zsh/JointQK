from __future__ import annotations

import torch

from experiments.stage1.toolkit import (
    TRANSFORM_FAMILIES,
    build_metric_transform,
    prepare_variant_states,
)
from experiments.stage1.diagnosis import (
    aggregate_variant_diagnostics,
    aggregate_variant_metric_rows,
    run_transform_variant_path,
)
from experiments.stage1.run_oracle_norm_spread_study import (
    compute_baseline_deltas,
    compute_correlations,
    summarize_sweep,
)


def test_basis_only_transform_is_orthogonal_and_norm_preserving():
    metrics = torch.stack([torch.diag(torch.tensor([9.0, 4.0, 1.0]))], dim=0)
    payload = build_metric_transform(metrics, variant="basis_only", eps=1e-6)
    transform = payload["transform"]
    inverse = payload["inverse"]

    identity = torch.eye(transform.shape[-1]).unsqueeze(0)
    assert torch.allclose(transform @ inverse, identity, atol=1e-5)

    states = torch.randn(1, 1, 7, 3)
    prepared = prepare_variant_states(states, metrics, variant="basis_only", eps=1e-6)
    input_norms = torch.linalg.vector_norm(states.float(), dim=-1)
    output_norms = torch.linalg.vector_norm(prepared["compressor_input"].float(), dim=-1)
    assert torch.allclose(input_norms, output_norms, atol=1e-5)


def test_trace_matched_full_metric_preserves_average_squared_norm_for_isotropic_inputs():
    torch.manual_seed(0)
    metrics = torch.stack([torch.diag(torch.tensor([9.0, 4.0, 1.0]))], dim=0)
    states = torch.randn(8, 1, 2048, 3)
    prepared = prepare_variant_states(states, metrics, variant="trace_matched_full_metric", eps=1e-6)

    input_sq = states.float().pow(2).sum(dim=-1).mean().item()
    output_sq = prepared["compressor_input"].float().pow(2).sum(dim=-1).mean().item()
    assert abs(input_sq - output_sq) / max(input_sq, 1e-8) < 0.05


def test_per_token_norm_matched_variant_preserves_per_token_norms_before_compression():
    torch.manual_seed(0)
    metrics = torch.stack([torch.diag(torch.tensor([16.0, 4.0, 1.0]))], dim=0)
    states = torch.randn(1, 1, 13, 3)
    prepared = prepare_variant_states(states, metrics, variant="per_token_norm_matched_full_metric", eps=1e-6)

    input_norms = torch.linalg.vector_norm(states.float(), dim=-1)
    matched_norms = torch.linalg.vector_norm(prepared["compressor_input"].float(), dim=-1)
    assert torch.allclose(input_norms, matched_norms, atol=1e-5)


def test_run_transform_variant_path_runs_for_all_transform_families():
    torch.manual_seed(0)
    prefix_keys = torch.randn(1, 2, 7, 8)
    future_queries = torch.randn(1, 4, 3, 8)
    metrics = torch.stack([torch.eye(8), 2.0 * torch.eye(8)], dim=0)

    for variant in TRANSFORM_FAMILIES:
        result = run_transform_variant_path(
            prefix_keys,
            future_queries,
            metrics,
            bits=3,
            seed=11,
            eps=1e-6,
            variant=variant,
        )
        assert result["reconstructed"].shape == prefix_keys.shape
        assert "geometry_distortion" in result["metrics"]
        assert "transformed_norm_cv" in result["diagnostics"]


def test_gamma_sweep_matches_basis_and_trace_matched_endpoints():
    metrics = torch.stack([torch.diag(torch.tensor([9.0, 4.0, 1.0]))], dim=0)

    basis = build_metric_transform(metrics, variant="basis_only", eps=1e-6)
    gamma0 = build_metric_transform(metrics, variant="gamma_sweep", eps=1e-6, gamma=0.0)
    assert torch.allclose(basis["transform"], gamma0["transform"], atol=1e-5)
    assert torch.allclose(basis["inverse"], gamma0["inverse"], atol=1e-5)

    trace_matched = build_metric_transform(metrics, variant="trace_matched_full_metric", eps=1e-6)
    gamma1 = build_metric_transform(metrics, variant="gamma_sweep", eps=1e-6, gamma=1.0)
    assert torch.allclose(trace_matched["transform"], gamma1["transform"], atol=1e-5)
    assert torch.allclose(trace_matched["inverse"], gamma1["inverse"], atol=1e-5)


def test_variant_aggregation_helpers_average_multi_variant_rows():
    rows = [
        {
            "variant": "basis_only",
            "metrics": {"logit_mse": 1.0, "top1_match": 0.8},
            "diagnostics": {"transformed_norm_cv": 0.2, "trace_scale_alpha_mean": 1.0},
        },
        {
            "variant": "basis_only",
            "metrics": {"logit_mse": 3.0, "top1_match": 0.6},
            "diagnostics": {"transformed_norm_cv": 0.4, "trace_scale_alpha_mean": 1.0},
        },
        {
            "variant": "full_metric",
            "metrics": {"logit_mse": 5.0, "top1_match": 0.5},
            "diagnostics": {"transformed_norm_cv": 0.9, "trace_scale_alpha_mean": 1.2},
        },
    ]

    metric_summary = aggregate_variant_metric_rows(rows, "basis_only")
    diag_summary = aggregate_variant_diagnostics(rows, "basis_only", ["transformed_norm_cv", "trace_scale_alpha_mean"])

    assert abs(metric_summary["logit_mse"] - 2.0) < 1e-6
    assert abs(metric_summary["top1_match"] - 0.7) < 1e-6
    assert abs(diag_summary["transformed_norm_cv"] - 0.3) < 1e-6
    assert abs(diag_summary["trace_scale_alpha_mean"] - 1.0) < 1e-6


def test_stage1d_correlations_ignore_degenerate_control_variants():
    rows = [
        {
            "config": "t",
            "row_index": 0,
            "layer_idx": 1,
            "bits": 3,
            "variant": "baseline_raw",
            "metrics": {"geometry_distortion": 1.0, "top1_match": 0.5},
            "diagnostics": {"transformed_norm_cv": 0.1},
        },
        {
            "config": "t",
            "row_index": 0,
            "layer_idx": 1,
            "bits": 3,
            "variant": "basis_only",
            "metrics": {"geometry_distortion": 0.9, "top1_match": 0.55},
            "diagnostics": {"transformed_norm_cv": 0.1},
        },
        {
            "config": "t",
            "row_index": 0,
            "layer_idx": 1,
            "bits": 3,
            "variant": "full_metric",
            "metrics": {"geometry_distortion": 1.3, "top1_match": 0.45},
            "diagnostics": {"transformed_norm_cv": 0.4},
        },
        {
            "config": "t",
            "row_index": 0,
            "layer_idx": 1,
            "bits": 3,
            "variant": "trace_matched_full_metric",
            "metrics": {"geometry_distortion": 5.0, "top1_match": 0.1},
            "diagnostics": {"transformed_norm_cv": 99.0},
        },
    ]

    delta_rows = compute_baseline_deltas(rows)
    correlations = compute_correlations(delta_rows)

    assert "transformed_norm_cv__vs__delta_geometry_distortion" in correlations
    assert correlations["transformed_norm_cv__vs__delta_geometry_distortion"] == 1.0


def test_stage1d_sweep_summary_uses_geometry_distortion_delta():
    rows = [
        {
            "config": "t",
            "row_index": 0,
            "example_offset": 0,
            "layer_idx": 1,
            "split_at": 8,
            "bits": 3,
            "variant": "baseline_raw",
            "gamma": None,
            "metrics": {"geometry_distortion": 1.0, "top1_match": 0.5},
            "diagnostics": {"transformed_norm_cv": 0.1},
        },
        {
            "config": "t",
            "row_index": 0,
            "example_offset": 0,
            "layer_idx": 1,
            "split_at": 8,
            "bits": 3,
            "variant": "gamma_sweep",
            "gamma": 0.5,
            "metrics": {"geometry_distortion": 0.7, "top1_match": 0.6},
            "diagnostics": {"transformed_norm_cv": 0.2},
        },
    ]

    summary = summarize_sweep(rows)

    assert abs(summary["0.5"]["mean_delta_geometry_distortion"] + 0.3) < 1e-6
    assert abs(summary["0.5"]["mean_delta_top1_match"] - 0.1) < 1e-6
