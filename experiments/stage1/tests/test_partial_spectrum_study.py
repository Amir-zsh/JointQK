from __future__ import annotations

import torch

from experiments.stage1.run_oracle_partial_spectrum_study import (
    assess_gamma_025_success,
    compute_variant_stats,
    variant_key,
)
from experiments.stage1.toolkit import apply_headwise_linear, build_metric_transform


def _row(
    *,
    row_index: int,
    layer_idx: int,
    bits: int,
    variant: str,
    gamma: float | None,
    geometry: float,
    top1: float,
) -> dict:
    return {
        "config": "qasper",
        "row_index": row_index,
        "layer_idx": layer_idx,
        "bits": bits,
        "variant": variant,
        "gamma": gamma,
        "metrics": {
            "geometry_distortion": geometry,
            "top1_match": top1,
        },
        "diagnostics": {},
    }


def test_gamma_quarter_transform_roundtrips_toy_states():
    metrics = torch.stack([torch.diag(torch.tensor([16.0, 4.0, 1.0]))], dim=0)
    payload = build_metric_transform(metrics, variant="gamma_sweep", eps=1e-6, gamma=0.25)
    transform = payload["transform"]
    inverse = payload["inverse"]
    states = torch.randn(1, 1, 5, 3)

    reconstructed = apply_headwise_linear(apply_headwise_linear(states, transform), inverse)

    assert variant_key({"variant": "gamma_sweep", "gamma": 0.25}) == "gamma_0.25"
    assert torch.allclose(reconstructed, states, atol=1e-5)


def test_partial_spectrum_stats_include_medians_win_rates_and_layer0_filtering():
    rows = [
        _row(row_index=0, layer_idx=0, bits=3, variant="baseline_raw", gamma=None, geometry=10.0, top1=0.1),
        _row(row_index=0, layer_idx=0, bits=3, variant="gamma_sweep", gamma=0.25, geometry=1.0, top1=0.9),
        _row(row_index=1, layer_idx=1, bits=3, variant="baseline_raw", gamma=None, geometry=4.0, top1=0.5),
        _row(row_index=1, layer_idx=1, bits=3, variant="gamma_sweep", gamma=0.25, geometry=3.0, top1=0.6),
        _row(row_index=2, layer_idx=2, bits=3, variant="baseline_raw", gamma=None, geometry=5.0, top1=0.7),
        _row(row_index=2, layer_idx=2, bits=3, variant="gamma_sweep", gamma=0.25, geometry=6.0, top1=0.8),
    ]

    full = compute_variant_stats(rows)
    layer0 = compute_variant_stats(rows, exclude_layer0=True)

    assert full["3"]["gamma_0.25"]["n"] == 3
    assert layer0["3"]["gamma_0.25"]["n"] == 2
    assert abs(layer0["3"]["gamma_0.25"]["median_delta_geometry_distortion"] - 0.0) < 1e-6
    assert abs(layer0["3"]["gamma_0.25"]["median_delta_top1_match"] - 0.1) < 1e-6
    assert abs(layer0["3"]["gamma_0.25"]["geometry_win_rate"] - 0.5) < 1e-6
    assert abs(layer0["3"]["gamma_0.25"]["top1_win_rate"] - 1.0) < 1e-6
    assert abs(layer0["3"]["gamma_0.25"]["both_win_rate"] - 0.5) < 1e-6


def test_partial_spectrum_success_assessment_counts_bitwidth_wins():
    summary = {
        "2": {
            "baseline_raw": {"mean_geometry_distortion": 2.0, "mean_top1_match": 0.5},
            "gamma_0.25": {"mean_geometry_distortion": 1.5, "mean_top1_match": 0.55},
        },
        "3": {
            "baseline_raw": {"mean_geometry_distortion": 1.0, "mean_top1_match": 0.6},
            "gamma_0.25": {"mean_geometry_distortion": 0.5, "mean_top1_match": 0.65},
        },
        "4": {
            "baseline_raw": {"mean_geometry_distortion": 0.3, "mean_top1_match": 0.8},
            "gamma_0.25": {"mean_geometry_distortion": 0.4, "mean_top1_match": 0.79},
        },
    }

    assert assess_gamma_025_success(summary) == "strong_success"
