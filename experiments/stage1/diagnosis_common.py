from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import torch

from experiments.stage1.common import (
    Stage1MSECompressor,
    apply_headwise_linear,
    compute_attention_metrics,
    compute_geometry_distortion,
    factorize_metric_batch,
)


def flatten_states(states: torch.Tensor) -> torch.Tensor:
    return states.reshape(-1, states.shape[-1]).float()


def sample_values(tensor: torch.Tensor, max_points: int = 4096, seed: int = 0) -> list[float]:
    flat = tensor.reshape(-1).float().cpu()
    if flat.numel() <= max_points:
        return flat.tolist()
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(flat.numel()), max_points))
    return flat[indices].tolist()


def safe_mean(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.mean().item())


def tensor_norm_stats(states: torch.Tensor) -> dict[str, float]:
    flat = flatten_states(states)
    norms = torch.linalg.vector_norm(flat, dim=-1)
    mean = norms.mean()
    std = norms.std(unbiased=False)
    cv = std / mean.clamp_min(1e-8)
    return {
        "norm_mean": float(mean.item()),
        "norm_std": float(std.item()),
        "norm_cv": float(cv.item()),
    }


def compute_second_moment(flat: torch.Tensor) -> torch.Tensor:
    return (flat.transpose(0, 1) @ flat) / max(flat.shape[0], 1)


def participation_ratio_from_matrix(matrix: torch.Tensor) -> float:
    sym = 0.5 * (matrix + matrix.transpose(-1, -2))
    eigenvalues = torch.linalg.eigvalsh(sym).clamp_min(0.0)
    numerator = eigenvalues.sum().pow(2)
    denominator = eigenvalues.pow(2).sum().clamp_min(1e-12)
    return float((numerator / denominator).item())


def effective_rank_from_states(states: torch.Tensor) -> float:
    return participation_ratio_from_matrix(compute_second_moment(flatten_states(states)))


def variance_spread(variances: torch.Tensor, eps: float = 1e-8) -> float:
    clipped = variances.clamp_min(eps)
    return float((clipped.max() / clipped.min()).item())


def rotated_coordinate_stats(rotated_flat: torch.Tensor) -> dict[str, float]:
    centered = rotated_flat - rotated_flat.mean(dim=0, keepdim=True)
    variances = centered.pow(2).mean(dim=0)
    std = variances.sqrt().clamp_min(1e-8)
    standardized = centered / std.unsqueeze(0)
    skew = standardized.pow(3).mean(dim=0).abs().mean()
    excess_kurtosis = (standardized.pow(4).mean(dim=0) - 3.0).abs().mean()
    return {
        "rotated_variance_mean": float(variances.mean().item()),
        "rotated_variance_std": float(variances.std(unbiased=False).item()),
        "rotated_variance_spread": variance_spread(variances),
        "rotated_mean_abs_skew": float(skew.item()),
        "rotated_mean_abs_excess_kurtosis": float(excess_kurtosis.item()),
    }


def transform_matrix_stats(matrices: torch.Tensor, metrics: torch.Tensor) -> dict[str, float]:
    singular_values = torch.linalg.svdvals(matrices.float())
    cond = singular_values[:, 0] / singular_values[:, -1].clamp_min(1e-8)
    metric_singular = torch.linalg.svdvals(metrics.float())
    metric_cond = metric_singular[:, 0] / metric_singular[:, -1].clamp_min(1e-8)
    return {
        "transform_singular_max_mean": float(singular_values[:, 0].mean().item()),
        "transform_singular_min_mean": float(singular_values[:, -1].mean().item()),
        "transform_condition_number_mean": float(cond.mean().item()),
        "transform_condition_number_max": float(cond.max().item()),
        "metric_condition_number_mean": float(metric_cond.mean().item()),
        "metric_condition_number_max": float(metric_cond.max().item()),
        "metric_frobenius_norm_mean": float(torch.linalg.matrix_norm(metrics.float(), ord="fro", dim=(-2, -1)).mean().item()),
    }


def instrument_compressor_path(states: torch.Tensor, compressor: Stage1MSECompressor) -> dict[str, Any]:
    flat = flatten_states(states)
    norms = torch.linalg.vector_norm(flat, dim=-1)
    normalized = flat / norms.unsqueeze(-1).clamp_min(1e-8)
    rotated = normalized @ compressor.Pi.T
    diffs = rotated.unsqueeze(-1) - compressor.centroids
    indices = diffs.abs().argmin(dim=-1)
    quantized_rotated = compressor.centroids[indices]
    reconstructed_flat = (quantized_rotated @ compressor.Pi) * norms.unsqueeze(-1)
    reconstructed = reconstructed_flat.reshape_as(states)

    diagnostics = {}
    diagnostics.update({f"pre_{k}": v for k, v in tensor_norm_stats(states).items()})
    diagnostics["pre_effective_rank"] = effective_rank_from_states(states)
    diagnostics.update({f"normalized_{k}": v for k, v in tensor_norm_stats(normalized).items()})
    diagnostics["normalized_effective_rank"] = participation_ratio_from_matrix(compute_second_moment(normalized))
    diagnostics.update(rotated_coordinate_stats(rotated))
    diagnostics["transformed_quant_mse"] = float((reconstructed_flat - flat).pow(2).mean().item())
    return {
        "reconstructed": reconstructed,
        "rotated_flat": rotated,
        "normalized_flat": normalized,
        "diagnostics": diagnostics,
    }


def run_baseline_path(
    prefix_keys: torch.Tensor,
    future_queries: torch.Tensor,
    metrics: torch.Tensor,
    bits: int,
    seed: int,
) -> dict[str, Any]:
    compressor = Stage1MSECompressor(prefix_keys.shape[-1], bits, seed=seed, device=prefix_keys.device)
    instrumented = instrument_compressor_path(prefix_keys, compressor)
    reconstructed = instrumented["reconstructed"]
    metric_values = compute_attention_metrics(future_queries, prefix_keys, reconstructed)
    metric_values["key_mse"] = float((reconstructed.float() - prefix_keys.float()).pow(2).mean().item())
    metric_values["geometry_distortion"] = compute_geometry_distortion(reconstructed, prefix_keys, metrics)
    return {
        "reconstructed": reconstructed,
        "metrics": metric_values,
        "diagnostics": instrumented["diagnostics"],
    }


def run_current_oracle_path(
    prefix_keys: torch.Tensor,
    future_queries: torch.Tensor,
    metrics: torch.Tensor,
    bits: int,
    seed: int,
    eps: float,
) -> dict[str, Any]:
    factors, inverses = factorize_metric_batch(metrics.to(prefix_keys.device), eps)
    transformed = apply_headwise_linear(prefix_keys.float(), factors)
    compressor = Stage1MSECompressor(prefix_keys.shape[-1], bits, seed=seed, device=prefix_keys.device)
    instrumented = instrument_compressor_path(transformed, compressor)
    mapped_back = apply_headwise_linear(instrumented["reconstructed"], inverses)

    metric_values = compute_attention_metrics(future_queries, prefix_keys, mapped_back)
    metric_values["key_mse"] = float((mapped_back.float() - prefix_keys.float()).pow(2).mean().item())
    metric_values["geometry_distortion"] = compute_geometry_distortion(mapped_back, prefix_keys, metrics)

    transform_stats = transform_matrix_stats(factors, metrics)
    transform_stats.update({f"transformed_{k}": v for k, v in tensor_norm_stats(transformed).items()})
    transform_stats["transformed_effective_rank"] = effective_rank_from_states(transformed)

    return {
        "reconstructed": mapped_back,
        "transformed": transformed,
        "transformed_reconstructed": instrumented["reconstructed"],
        "rotated_flat": instrumented["rotated_flat"],
        "metrics": metric_values,
        "diagnostics": {**instrumented["diagnostics"], **transform_stats},
        "factors": factors,
        "inverses": inverses,
    }


def aggregate_metric_rows(rows: list[dict[str, Any]], predicate: Any | None = None) -> dict[str, dict[str, float]]:
    filtered = [row for row in rows if predicate is None or predicate(row)]
    if not filtered:
        return {}

    metric_names = sorted(filtered[0]["baseline_metrics"].keys())
    summary: dict[str, dict[str, float]] = {}
    for label in ["baseline", "oracle"]:
        block = {}
        for metric_name in metric_names:
            values = [row[f"{label}_metrics"][metric_name] for row in filtered]
            block[metric_name] = float(sum(values) / len(values))
        summary[label] = block
    return summary


def aggregate_per_layer(rows: list[dict[str, Any]], label: str, metric_name: str) -> list[float]:
    by_layer: dict[int, list[float]] = {}
    for row in rows:
        by_layer.setdefault(int(row["layer_idx"]), []).append(float(row[f"{label}_metrics"][metric_name]))
    return [sum(by_layer[layer]) / len(by_layer[layer]) for layer in sorted(by_layer)]


def aggregate_per_layer_delta(rows: list[dict[str, Any]], metric_name: str) -> list[float]:
    by_layer: dict[int, list[float]] = {}
    for row in rows:
        delta = float(row["oracle_metrics"][metric_name] - row["baseline_metrics"][metric_name])
        by_layer.setdefault(int(row["layer_idx"]), []).append(delta)
    return [sum(by_layer[layer]) / len(by_layer[layer]) for layer in sorted(by_layer)]


def pearson_correlation(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return 0.0
    x = torch.tensor(x_values, dtype=torch.float32)
    y = torch.tensor(y_values, dtype=torch.float32)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = torch.linalg.vector_norm(x_centered) * torch.linalg.vector_norm(y_centered)
    if float(denom.item()) < 1e-12:
        return 0.0
    return float((x_centered @ y_centered / denom).item())


def build_correlation_table(rows: list[dict[str, Any]]) -> dict[str, float]:
    x_sources = {
        "transform_condition_number_mean": [float(row["oracle_diagnostics"]["transform_condition_number_mean"]) for row in rows],
        "transformed_norm_cv": [float(row["oracle_diagnostics"]["transformed_norm_cv"]) for row in rows],
        "rotated_variance_spread": [float(row["oracle_diagnostics"]["rotated_variance_spread"]) for row in rows],
        "transformed_effective_rank": [float(row["oracle_diagnostics"]["transformed_effective_rank"]) for row in rows],
    }
    y_sources = {
        "delta_logit_mse": [
            float(row["oracle_metrics"]["logit_mse"] - row["baseline_metrics"]["logit_mse"])
            for row in rows
        ],
        "delta_top1_match": [
            float(row["oracle_metrics"]["top1_match"] - row["baseline_metrics"]["top1_match"])
            for row in rows
        ],
    }
    correlations = {}
    for x_name, x_values in x_sources.items():
        for y_name, y_values in y_sources.items():
            correlations[f"{x_name}__vs__{y_name}"] = pearson_correlation(x_values, y_values)
    return correlations


def choose_representative_case(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return max(rows, key=lambda row: float(row["oracle_diagnostics"]["transform_condition_number_mean"]))


def decide_diagnosis(rows: list[dict[str, Any]]) -> str:
    layer0_excluded = [row for row in rows if int(row["layer_idx"]) != 0]
    if not layer0_excluded:
        return "mixed"

    worse_on_logit = all(
        aggregate_metric_rows(layer0_excluded, lambda row, bits=bits: int(row["bits"]) == bits)["oracle"]["logit_mse"]
        > aggregate_metric_rows(layer0_excluded, lambda row, bits=bits: int(row["bits"]) == bits)["baseline"]["logit_mse"]
        for bits in sorted({int(row["bits"]) for row in layer0_excluded})
    )
    correlations = build_correlation_table(layer0_excluded)
    strongest = max((abs(value) for value in correlations.values()), default=0.0)
    if worse_on_logit and strongest >= 0.25:
        return "supported"
    if worse_on_logit and strongest >= 0.10:
        return "mixed"
    return "unsupported"


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    def _convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, torch.Tensor):
            return value.tolist()
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_convert(v) for v in value]
        return value

    Path(path).write_text(__import__("json").dumps(_convert(payload), indent=2, sort_keys=True))
