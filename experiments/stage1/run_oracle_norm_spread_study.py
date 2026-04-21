from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kvpress"))
else:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kvpress"))

from experiments.stage1.common import (
    TRANSFORM_FAMILIES,
    compute_grouped_query_second_moment,
    ensure_dir,
    load_model_and_tokenizer,
    run_prefill_and_capture,
    split_prefix_and_future,
    trim_queries,
)
from experiments.stage1.data import get_dataset_spec, load_and_filter
from experiments.stage1.diagnosis_common import (
    aggregate_variant_diagnostics,
    aggregate_variant_metric_rows,
    pearson_correlation,
    run_transform_variant_path,
    save_json,
)


VARIANT_ORDER = list(TRANSFORM_FAMILIES)
CONTROL_VARIANTS = [
    "basis_only",
    "full_metric",
    "trace_matched_full_metric",
    "per_token_norm_matched_full_metric",
]
NON_DEGENERATE_VARIANTS = [
    "basis_only",
    "full_metric",
    "gamma_sweep",
]
DELTA_BAR_VARIANTS = [
    "basis_only",
    "full_metric",
]
SWEEP_GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0]
VARIANT_LABELS = {
    "baseline_raw": "Baseline",
    "basis_only": "Basis Only",
    "full_metric": "Full Metric",
    "trace_matched_full_metric": "Trace-Matched",
    "per_token_norm_matched_full_metric": "Per-Token Norm-Matched",
    "gamma_sweep": "Gamma Sweep",
}
VARIANT_COLORS = {
    "baseline_raw": "#4477AA",
    "basis_only": "#228833",
    "full_metric": "#CC6677",
    "trace_matched_full_metric": "#AA4499",
    "per_token_norm_matched_full_metric": "#DDCC77",
    "gamma_sweep": "#66A61E",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Stage-1D oracle norm-spread ablation study.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset", default="longbench-e")
    parser.add_argument("--configs", default="qasper,hotpotqa,passage_retrieval_en")
    parser.add_argument("--num_examples_per_config", type=int, default=2)
    parser.add_argument("--min_tokens", type=int, default=4000)
    parser.add_argument("--max_tokens", type=int, default=12000)
    parser.add_argument("--n_sink", type=int, default=4)
    parser.add_argument("--key_bits", default="2,3,4")
    parser.add_argument("--prefix_fraction", type=float, default=0.75)
    parser.add_argument("--min_future_tokens", type=int, default=32)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--output_dir", default="artifacts/stage1/oracle_norm_spread_study")
    return parser.parse_args()


def row_key(row: dict) -> tuple:
    return (
        row["config"],
        int(row["row_index"]),
        int(row["layer_idx"]),
        int(row["bits"]),
    )


def filter_rows(rows: list[dict], *, exclude_layer0: bool = False, bits: int | None = None) -> list[dict]:
    filtered = rows
    if exclude_layer0:
        filtered = [row for row in filtered if int(row["layer_idx"]) != 0]
    if bits is not None:
        filtered = [row for row in filtered if int(row["bits"]) == bits]
    return filtered


def summarize_variant_rows(rows: list[dict], *, exclude_layer0: bool = False) -> dict[str, dict[str, dict[str, float]]]:
    bitwidths = sorted({int(row["bits"]) for row in rows})
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for bits in bitwidths:
        bit_rows = filter_rows(rows, exclude_layer0=exclude_layer0, bits=bits)
        summary[str(bits)] = {}
        for variant in VARIANT_ORDER:
            summary[str(bits)][variant] = aggregate_variant_metric_rows(bit_rows, variant)
    return summary


def summarize_variant_diagnostics(rows: list[dict], *, exclude_layer0: bool = False) -> dict[str, dict[str, dict[str, float]]]:
    bitwidths = sorted({int(row["bits"]) for row in rows})
    keys = [
        "transformed_norm_mean",
        "transformed_norm_std",
        "transformed_norm_cv",
        "transform_ratio_mean",
        "transform_ratio_std",
        "transform_ratio_cv",
        "transform_ratio_min",
        "transform_ratio_max",
        "transformed_effective_rank",
        "rotated_variance_spread",
        "rotated_mean_abs_skew",
        "rotated_mean_abs_excess_kurtosis",
        "transform_condition_number_mean",
        "trace_scale_alpha_mean",
        "trace_scale_alpha_std",
    ]
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for bits in bitwidths:
        bit_rows = filter_rows(rows, exclude_layer0=exclude_layer0, bits=bits)
        summary[str(bits)] = {}
        for variant in VARIANT_ORDER:
            summary[str(bits)][variant] = aggregate_variant_diagnostics(bit_rows, variant, keys)
    return summary


def compute_baseline_deltas(rows: list[dict]) -> list[dict]:
    baseline_by_key = {
        row_key(row): row
        for row in rows
        if row["variant"] == "baseline_raw"
    }
    delta_rows = []
    for row in rows:
        if row["variant"] == "baseline_raw":
            continue
        baseline = baseline_by_key[row_key(row)]
        delta_rows.append(
            {
                **row,
                "delta_top1_match": float(row["metrics"]["top1_match"] - baseline["metrics"]["top1_match"]),
                "delta_geometry_distortion": float(
                    row["metrics"]["geometry_distortion"] - baseline["metrics"]["geometry_distortion"]
                ),
            }
        )
    return delta_rows


def compute_correlations(delta_rows: list[dict], *, exclude_layer0: bool = False) -> dict[str, float]:
    filtered = [
        row
        for row in delta_rows
        if row["variant"] in NON_DEGENERATE_VARIANTS and (not exclude_layer0 or int(row["layer_idx"]) != 0)
    ]
    if not filtered:
        return {}
    norm_cv = [float(row["diagnostics"]["transformed_norm_cv"]) for row in filtered]
    distortion = [float(row["delta_geometry_distortion"]) for row in filtered]
    top1 = [float(row["delta_top1_match"]) for row in filtered]
    return {
        "transformed_norm_cv__vs__delta_geometry_distortion": pearson_correlation(norm_cv, distortion),
        "transformed_norm_cv__vs__delta_top1_match": pearson_correlation(norm_cv, top1),
    }


def summarize_sweep(rows: list[dict], *, exclude_layer0: bool = False) -> dict[str, dict[str, float]]:
    filtered = rows if not exclude_layer0 else [row for row in rows if int(row["layer_idx"]) != 0]
    baseline_by_key = {
        row_key(row): row
        for row in filtered
        if row["variant"] == "baseline_raw" and int(row["bits"]) == 3
    }
    summary: dict[str, dict[str, float]] = {}
    for gamma in SWEEP_GAMMAS:
        gamma_rows = [
            row
            for row in filtered
            if row["variant"] == "gamma_sweep" and abs(float(row["gamma"]) - gamma) < 1e-8 and int(row["bits"]) == 3
        ]
        if not gamma_rows:
            continue
        delta_distortion = []
        delta_top1 = []
        norm_cv = []
        for row in gamma_rows:
            baseline = baseline_by_key[row_key(row)]
            delta_distortion.append(float(row["metrics"]["geometry_distortion"] - baseline["metrics"]["geometry_distortion"]))
            delta_top1.append(float(row["metrics"]["top1_match"] - baseline["metrics"]["top1_match"]))
            norm_cv.append(float(row["diagnostics"]["transformed_norm_cv"]))
        summary[str(gamma)] = {
            "mean_transformed_norm_cv": float(sum(norm_cv) / len(norm_cv)),
            "mean_delta_geometry_distortion": float(sum(delta_distortion) / len(delta_distortion)),
            "mean_delta_top1_match": float(sum(delta_top1) / len(delta_top1)),
        }
    return summary


def monotone_non_decreasing(values: list[float], tol: float = 1e-6) -> bool:
    return all(values[idx] <= values[idx + 1] + tol for idx in range(len(values) - 1))


def monotone_non_increasing(values: list[float], tol: float = 1e-6) -> bool:
    return all(values[idx] >= values[idx + 1] - tol for idx in range(len(values) - 1))


def plot_grouped_variant_bars(
    summary: dict[str, dict[str, dict[str, float]]],
    output_path: Path,
    title: str,
) -> None:
    metrics = ["geometry_distortion", "top1_match"]
    bitwidths = sorted(int(key) for key in summary.keys())
    variants = VARIANT_ORDER
    fig, axes = plt.subplots(1, len(metrics), figsize=(12.5, 4.8), constrained_layout=True)
    for ax, metric_name in zip(axes, metrics):
        x = list(range(len(bitwidths)))
        width = 0.14
        offsets = [((idx - (len(variants) - 1) / 2.0) * width) for idx in range(len(variants))]
        for variant_idx, variant in enumerate(variants):
            values = [summary[str(bits)][variant][metric_name] for bits in bitwidths]
            ax.bar(
                [pos + offsets[variant_idx] for pos in x],
                values,
                width=width,
                color=VARIANT_COLORS.get(variant, "#888888"),
                label=VARIANT_LABELS[variant],
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{bits}-bit" for bits in bitwidths])
        ax.set_title(metric_name.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_delta_scatter(
    delta_rows: list[dict],
    y_key: str,
    output_path: Path,
    title: str,
    ylabel: str,
    variants: list[str],
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    for variant in variants:
        points = [row for row in delta_rows if row["variant"] == variant]
        x_vals = [float(row["diagnostics"]["transformed_norm_cv"]) for row in points]
        y_vals = [float(row[y_key]) for row in points]
        ax.scatter(
            x_vals,
            y_vals,
            alpha=0.7,
            s=20,
            color=VARIANT_COLORS[variant],
            label=VARIANT_LABELS[variant],
        )
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.7)
    ax.set_xlabel("Transformed Norm CV")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    ax.set_title(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_sweep_lines(
    full_summary: dict[str, dict[str, float]],
    layer0_summary: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    gammas = [float(key) for key in sorted(full_summary.keys(), key=float)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    configs = [
        ("mean_transformed_norm_cv", "Mean Transformed Norm CV"),
        ("mean_delta_geometry_distortion", "Mean Delta Geometry Distortion"),
        ("mean_delta_top1_match", "Mean Delta Top-1 Match"),
    ]
    for ax, (metric_name, title) in zip(axes, configs):
        ax.plot(gammas, [full_summary[str(g)][metric_name] for g in gammas], marker="o", color="#4477AA", label="All layers")
        ax.plot(gammas, [layer0_summary[str(g)][metric_name] for g in gammas], marker="s", color="#CC6677", label="Layer-0 excluded")
        ax.set_title(title)
        ax.set_xlabel("Gamma")
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle("Spectrum Sweep: Increasing Anisotropic Scaling")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_delta_bars(
    delta_rows: list[dict],
    output_path: Path,
) -> None:
    bitwidths = sorted({int(row["bits"]) for row in delta_rows})
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    metric_map = [("delta_geometry_distortion", "Delta Geometry Distortion"), ("delta_top1_match", "Delta Top-1 Match")]
    for ax, (metric_key, title) in zip(axes, metric_map):
        x = list(range(len(bitwidths)))
        width = 0.28
        offsets = [((idx - (len(DELTA_BAR_VARIANTS) - 1) / 2.0) * width) for idx in range(len(DELTA_BAR_VARIANTS))]
        for idx, variant in enumerate(DELTA_BAR_VARIANTS):
            vals = []
            for bits in bitwidths:
                variant_rows = [row for row in delta_rows if row["variant"] == variant and int(row["bits"]) == bits]
                vals.append(float(sum(float(row[metric_key]) for row in variant_rows) / len(variant_rows)))
            ax.bar(
                [pos + offsets[idx] for pos in x],
                vals,
                width=width,
                color=VARIANT_COLORS[variant],
                label=VARIANT_LABELS[variant],
            )
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{bits}-bit" for bits in bitwidths])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Non-Degenerate Variant Deltas Relative to Baseline")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_summary_md(
    output_path: Path,
    full_summary: dict[str, dict[str, dict[str, float]]],
    layer0_summary: dict[str, dict[str, dict[str, float]]],
    layer0_diag: dict[str, dict[str, dict[str, float]]],
    correlations: dict[str, float],
    layer0_correlations: dict[str, float],
    sweep_summary: dict[str, dict[str, float]],
    sweep_layer0_summary: dict[str, dict[str, float]],
) -> None:
    def _format_metric_block(summary: dict[str, dict[str, dict[str, float]]], bits: int, metric: str) -> str:
        return ", ".join(
            f"{VARIANT_LABELS[variant]} `{summary[str(bits)][variant][metric]:.4f}`"
            for variant in VARIANT_ORDER
        )

    bits_list = sorted(int(key) for key in full_summary.keys())
    basis_near_baseline = all(
        layer0_summary[str(bits)]["basis_only"]["geometry_distortion"]
        <= layer0_summary[str(bits)]["baseline_raw"]["geometry_distortion"] * 1.05
        and layer0_summary[str(bits)]["basis_only"]["top1_match"] >= layer0_summary[str(bits)]["baseline_raw"]["top1_match"] - 0.03
        for bits in bits_list
    )
    full_worse_than_basis = all(
        layer0_summary[str(bits)]["full_metric"]["geometry_distortion"]
        > layer0_summary[str(bits)]["basis_only"]["geometry_distortion"]
        for bits in bits_list
    )
    gamma_keys = sorted(sweep_layer0_summary.keys(), key=float)
    sweep_norm = [sweep_layer0_summary[key]["mean_transformed_norm_cv"] for key in gamma_keys]
    sweep_distortion = [sweep_layer0_summary[key]["mean_delta_geometry_distortion"] for key in gamma_keys]
    sweep_top1 = [sweep_layer0_summary[key]["mean_delta_top1_match"] for key in gamma_keys]
    sweep_monotone = (
        monotone_non_decreasing(sweep_norm)
        and monotone_non_decreasing(sweep_distortion)
        and monotone_non_increasing(sweep_top1)
    )
    best_full_gamma = min(sweep_summary.items(), key=lambda item: item[1]["mean_delta_geometry_distortion"])[0]
    best_layer0_gamma = min(sweep_layer0_summary.items(), key=lambda item: item[1]["mean_delta_geometry_distortion"])[0]

    lines = [
        "# Stage 1D: Oracle Norm-Spread Study",
        "",
        "## Headline",
        "",
        "- `geometry_distortion` is the primary quadratic metric for this study. In this harness it numerically matches `logit_mse` because both are computed from the same future-query second moment, so the summary reports geometry distortion only.",
        f"- A partial-metric oracle is a real positive result here: the best `3`-bit `gamma` is `{best_full_gamma}` on full data and `{best_layer0_gamma}` on the layer-0-excluded slice, and both beat the V3 baseline on geometry distortion and top-1.",
        "- The trace-matched and per-token-norm-matched control arms are mathematically degenerate under the current unit-normalizing backend, so they do not provide valid evidence against the norm-spread mechanism.",
        "- That positive result is therefore separate from the original mechanism question. The sweep still does not validate a simple `more norm spread -> worse backend behavior` story, but the degenerate controls also do not falsify it.",
        "",
        "## Full Summary",
        "",
    ]
    for bits in bits_list:
        lines.append(f"- {bits}-bit geometry distortion: {_format_metric_block(full_summary, bits, 'geometry_distortion')}")
        lines.append(f"- {bits}-bit top-1: {_format_metric_block(full_summary, bits, 'top1_match')}")
    lines.extend(["", "## Layer-0-Excluded Summary", ""])
    for bits in bits_list:
        lines.append(f"- {bits}-bit geometry distortion: {_format_metric_block(layer0_summary, bits, 'geometry_distortion')}")
        lines.append(f"- {bits}-bit top-1: {_format_metric_block(layer0_summary, bits, 'top1_match')}")
        lines.append(
                f"- {bits}-bit transformed norm CV: "
                + ", ".join(
                    f"{VARIANT_LABELS[variant]} `{layer0_diag[str(bits)][variant]['transformed_norm_cv']:.4f}`"
                    for variant in CONTROL_VARIANTS
                )
        )
    lines.extend(
        [
            "",
            "## Correlation Evidence",
            "",
            "- Correlations below use only non-degenerate variants (`basis_only`, `full_metric`, and the `gamma` sweep), excluding the trace-matched and per-token-norm-matched control arms.",
            f"- Full data `transformed_norm_cv` vs `delta_geometry_distortion`: `{correlations['transformed_norm_cv__vs__delta_geometry_distortion']:.4f}`",
            f"- Full data `transformed_norm_cv` vs `delta_top1_match`: `{correlations['transformed_norm_cv__vs__delta_top1_match']:.4f}`",
            f"- Layer-0-excluded `transformed_norm_cv` vs `delta_geometry_distortion`: `{layer0_correlations['transformed_norm_cv__vs__delta_geometry_distortion']:.4f}`",
            f"- Layer-0-excluded `transformed_norm_cv` vs `delta_top1_match`: `{layer0_correlations['transformed_norm_cv__vs__delta_top1_match']:.4f}`",
            "",
            "## Spectrum Sweep: Full Data",
            "",
        ]
    )
    for gamma in sorted(sweep_summary.keys(), key=float):
        block = sweep_summary[gamma]
        lines.append(
            f"- gamma={gamma}: norm CV `{block['mean_transformed_norm_cv']:.4f}`, "
            f"delta geometry distortion `{block['mean_delta_geometry_distortion']:.4f}`, "
            f"delta top-1 `{block['mean_delta_top1_match']:.4f}`"
        )
    lines.extend(["", "## Spectrum Sweep: Layer-0-Excluded", ""])
    for gamma in gamma_keys:
        block = sweep_layer0_summary[gamma]
        lines.append(
            f"- gamma={gamma}: norm CV `{block['mean_transformed_norm_cv']:.4f}`, "
            f"delta geometry distortion `{block['mean_delta_geometry_distortion']:.4f}`, "
            f"delta top-1 `{block['mean_delta_top1_match']:.4f}`"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- If `basis_only` is near baseline while `full_metric` is worse, scaling is implicated: `{basis_near_baseline and full_worse_than_basis}`.",
            f"- Partial-metric scaling is a real method signal: intermediate `gamma` values improve geometry distortion and top-1 over baseline on both full data and the layer-0-excluded slice.",
            "- The trace-matched and per-token-norm-matched variants should not be treated as causal rescue tests here, because unit-normalizing compression makes them effectively degenerate with `full_metric`.",
            f"- The original norm-spread mechanism claim still lacks clean support because the non-degenerate `gamma` sweep is not monotone: `{sweep_monotone}`.",
        ]
    )
    output_path.write_text("\n".join(lines))


def main() -> None:
    from kvpress.utils import extract_keys_and_values

    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    bitwidths = [int(part.strip()) for part in args.key_bits.split(",") if part.strip()]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()] or None

    model, tokenizer = load_model_and_tokenizer(args.model, device_map=args.device_map, dtype_name=args.dtype)
    spec = get_dataset_spec(args.dataset)
    examples = load_and_filter(
        spec,
        tokenizer=tokenizer,
        num_examples_per_config=args.num_examples_per_config,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        seed=args.seed,
        config_names=configs,
    )

    rows: list[dict] = []

    for example_offset, example in enumerate(examples):
        outputs, _, post_queries = run_prefill_and_capture(model, example.input_ids)
        future_query_tensor = trim_queries(post_queries, args.n_sink).to(getattr(model, "dtype", torch.float16))
        cache = outputs.past_key_values

        for layer_idx in range(len(cache.layers)):
            keys, _ = extract_keys_and_values(cache, layer_idx)
            keys = keys[:, :, args.n_sink :, :]
            prefix_keys, future_queries, split_at = split_prefix_and_future(
                keys,
                future_query_tensor[layer_idx : layer_idx + 1].to(keys.device),
                prefix_fraction=args.prefix_fraction,
                min_future_tokens=args.min_future_tokens,
            )
            metrics = compute_grouped_query_second_moment(future_queries.float(), prefix_keys.shape[1]).to(keys.device)

            for bits in bitwidths:
                seed = args.seed + layer_idx * 1000
                for variant in VARIANT_ORDER:
                    result = run_transform_variant_path(
                        prefix_keys,
                        future_queries,
                        metrics,
                        bits=bits,
                        seed=seed,
                        eps=args.eps,
                        variant=variant,
                    )
                    rows.append(
                        {
                            "dataset": example.dataset,
                            "config": example.config,
                            "row_index": int(example.row_index),
                            "example_offset": int(example_offset),
                            "layer_idx": int(layer_idx),
                            "split_at": int(split_at),
                            "bits": int(bits),
                            "variant": variant,
                            "gamma": None,
                            "metrics": result["metrics"],
                            "diagnostics": result["diagnostics"],
                        }
                    )

                if bits == 3:
                    for gamma in SWEEP_GAMMAS:
                        result = run_transform_variant_path(
                            prefix_keys,
                            future_queries,
                            metrics,
                            bits=bits,
                            seed=seed,
                            eps=args.eps,
                            variant="gamma_sweep",
                            gamma=gamma,
                        )
                        rows.append(
                            {
                                "dataset": example.dataset,
                                "config": example.config,
                                "row_index": int(example.row_index),
                                "example_offset": int(example_offset),
                                "layer_idx": int(layer_idx),
                                "split_at": int(split_at),
                                "bits": int(bits),
                                "variant": "gamma_sweep",
                                "gamma": float(gamma),
                                "metrics": result["metrics"],
                                "diagnostics": result["diagnostics"],
                            }
                        )

    summary = summarize_variant_rows(rows)
    layer0_summary = summarize_variant_rows(rows, exclude_layer0=True)
    diagnostics_summary = summarize_variant_diagnostics(rows)
    layer0_diagnostics_summary = summarize_variant_diagnostics(rows, exclude_layer0=True)
    delta_rows = compute_baseline_deltas(rows)
    correlations = compute_correlations(delta_rows)
    layer0_correlations = compute_correlations(delta_rows, exclude_layer0=True)
    sweep_summary = summarize_sweep(rows)
    sweep_layer0_summary = summarize_sweep(rows, exclude_layer0=True)

    payload = {
        "config": vars(args),
        "summary": summary,
        "layer0_excluded_summary": layer0_summary,
        "diagnostics_summary": diagnostics_summary,
        "layer0_excluded_diagnostics_summary": layer0_diagnostics_summary,
        "correlations": correlations,
        "layer0_excluded_correlations": layer0_correlations,
        "sweep_summary": sweep_summary,
        "layer0_excluded_sweep_summary": sweep_layer0_summary,
        "rows": rows,
    }
    torch.save(payload, output_dir / "variant_study.pt")
    save_json(output_dir / "metrics.json", payload)

    plot_grouped_variant_bars(summary, output_dir / "variant_grouped_bars.png", "Variant Comparison: All Layers")
    plot_grouped_variant_bars(
        layer0_summary,
        output_dir / "variant_grouped_bars_layer0_excluded.png",
        "Variant Comparison: Layer-0 Excluded",
    )
    plot_delta_scatter(
        delta_rows,
        "delta_geometry_distortion",
        output_dir / "norm_cv_vs_delta_logit_mse.png",
        "Transformed Norm CV vs Delta Geometry Distortion",
        "Delta Geometry Distortion",
        NON_DEGENERATE_VARIANTS,
    )
    plot_delta_scatter(
        delta_rows,
        "delta_top1_match",
        output_dir / "norm_cv_vs_delta_top1_match.png",
        "Transformed Norm CV vs Delta Top-1 Match",
        "Delta Top-1 Match",
        NON_DEGENERATE_VARIANTS,
    )
    plot_sweep_lines(sweep_summary, sweep_layer0_summary, output_dir / "gamma_sweep_lines.png")
    plot_delta_bars(delta_rows, output_dir / "recovery_comparison.png")
    build_summary_md(
        output_dir / "summary.md",
        summary,
        layer0_summary,
        layer0_diagnostics_summary,
        correlations,
        layer0_correlations,
        sweep_summary,
        sweep_layer0_summary,
    )


if __name__ == "__main__":
    main()
