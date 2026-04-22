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

from experiments.stage1.toolkit import (
    compute_grouped_query_second_moment,
    ensure_dir,
    load_model_and_tokenizer,
    run_prefill_and_capture,
    save_json,
    split_prefix_and_future,
    trim_queries,
)
from experiments.stage1.data import get_dataset_spec, load_and_filter, register_all_benchmark_specs
from experiments.stage1.diagnosis import (
    aggregate_metric_rows,
    aggregate_per_layer,
    aggregate_per_layer_delta,
    build_correlation_table,
    choose_representative_case,
    decide_diagnosis,
    run_baseline_path,
    run_current_oracle_path,
    sample_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose the current Cholesky-based oracle backend mismatch.")
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
    parser.add_argument("--output_dir", default="artifacts/stage1/current_oracle_diagnosis")
    return parser.parse_args()


def plot_summary_bars(summary: dict[str, dict[str, dict[str, float]]], output_path: Path, title: str) -> None:
    metrics = ["geometry_distortion", "logit_mse", "top1_match", "top5_containment"]
    bitwidths = sorted(int(key) for key in summary.keys())
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4.5), constrained_layout=True)
    for ax, metric_name in zip(axes, metrics):
        x = range(len(bitwidths))
        width = 0.36
        baseline = [summary[str(bits)]["baseline"][metric_name] for bits in bitwidths]
        oracle = [summary[str(bits)]["oracle"][metric_name] for bits in bitwidths]
        ax.bar([i - width / 2 for i in x], baseline, width=width, color="#4477AA", label="Baseline")
        ax.bar([i + width / 2 for i in x], oracle, width=width, color="#CC6677", label="Current Oracle")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{bits}-bit" for bits in bitwidths])
        ax.set_title(metric_name.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend()
    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_per_layer_metric(rows: list[dict], metric_name: str, output_path: Path, title: str, ylabel: str) -> None:
    bitwidths = sorted({int(row["bits"]) for row in rows})
    fig, axes = plt.subplots(len(bitwidths), 1, figsize=(10, 10), constrained_layout=True)
    if len(bitwidths) == 1:
        axes = [axes]
    for ax, bits in zip(axes, bitwidths):
        bit_rows = [row for row in rows if int(row["bits"]) == bits]
        baseline = aggregate_per_layer(bit_rows, "baseline", metric_name)
        oracle = aggregate_per_layer(bit_rows, "oracle", metric_name)
        layers = range(len(baseline))
        ax.plot(layers, baseline, linewidth=2, color="#4477AA", label="Baseline")
        ax.plot(layers, oracle, linewidth=2, color="#CC6677", label="Current Oracle")
        ax.set_title(f"{bits}-bit")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Layer")
    axes[0].legend()
    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_per_layer_delta(rows: list[dict], metric_name: str, output_path: Path, title: str, ylabel: str) -> None:
    bitwidths = sorted({int(row["bits"]) for row in rows})
    fig, axes = plt.subplots(len(bitwidths), 1, figsize=(10, 10), constrained_layout=True)
    if len(bitwidths) == 1:
        axes = [axes]
    colors = {2: "#228833", 3: "#CCBB44", 4: "#EE6677"}
    for ax, bits in zip(axes, bitwidths):
        bit_rows = [row for row in rows if int(row["bits"]) == bits]
        deltas = aggregate_per_layer_delta(bit_rows, metric_name)
        layers = range(len(deltas))
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.7)
        ax.bar(layers, deltas, color=colors.get(bits, "#4477AA"), alpha=0.85)
        ax.set_title(f"{bits}-bit")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("Layer")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_scatter(rows: list[dict], x_key: str, y_delta_metric: str, output_path: Path, title: str, xlabel: str, ylabel: str) -> None:
    bitwidths = sorted({int(row["bits"]) for row in rows})
    fig, axes = plt.subplots(1, len(bitwidths), figsize=(15, 4.8), constrained_layout=True)
    if len(bitwidths) == 1:
        axes = [axes]
    colors = {2: "#228833", 3: "#CCBB44", 4: "#EE6677"}
    for ax, bits in zip(axes, bitwidths):
        bit_rows = [row for row in rows if int(row["bits"]) == bits]
        x_vals = [float(row["oracle_diagnostics"][x_key]) for row in bit_rows]
        y_vals = [float(row["oracle_metrics"][y_delta_metric] - row["baseline_metrics"][y_delta_metric]) for row in bit_rows]
        ax.scatter(x_vals, y_vals, alpha=0.65, s=18, color=colors.get(bits, "#4477AA"))
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.7)
        ax.set_title(f"{bits}-bit")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_histograms(representative: dict, output_path: Path) -> None:
    if not representative:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    axes[0].hist(representative["raw_values"], bins=40, density=True, alpha=0.8, color="#4477AA")
    axes[0].set_title("Original Keys")
    axes[1].hist(representative["transformed_values"], bins=40, density=True, alpha=0.8, color="#CC6677")
    axes[1].set_title("Current Oracle Transform")
    axes[2].hist(representative["rotated_values"], bins=40, density=True, alpha=0.8, color="#228833")
    axes[2].set_title("After V3 Rotation")
    title = (
        f"Representative Case: layer {representative['layer_idx']}, bits {representative['bits']}, "
        f"cond={representative['transform_condition_number_mean']:.2f}"
    )
    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_summary_md(
    output_path: Path,
    full_summary: dict[str, dict[str, dict[str, float]]],
    layer0_summary: dict[str, dict[str, dict[str, float]]],
    correlations: dict[str, float],
    decision: str,
) -> None:
    lines = [
        "# Current Oracle Diagnosis",
        "",
        "## Full Summary",
        "",
    ]
    for bits in sorted(full_summary.keys(), key=int):
        baseline = full_summary[bits]["baseline"]
        oracle = full_summary[bits]["oracle"]
        lines.append(f"- {bits}-bit geometry distortion: `{baseline['geometry_distortion']:.4f} -> {oracle['geometry_distortion']:.4f}`")
        lines.append(f"- {bits}-bit logit MSE: `{baseline['logit_mse']:.4f} -> {oracle['logit_mse']:.4f}`")
        lines.append(f"- {bits}-bit top-1: `{baseline['top1_match']:.4f} -> {oracle['top1_match']:.4f}`")
    lines.extend(["", "## Layer-0-Excluded Summary", ""])
    for bits in sorted(layer0_summary.keys(), key=int):
        baseline = layer0_summary[bits]["baseline"]
        oracle = layer0_summary[bits]["oracle"]
        lines.append(f"- {bits}-bit geometry distortion: `{baseline['geometry_distortion']:.4f} -> {oracle['geometry_distortion']:.4f}`")
        lines.append(f"- {bits}-bit logit MSE: `{baseline['logit_mse']:.4f} -> {oracle['logit_mse']:.4f}`")
        lines.append(f"- {bits}-bit top-1: `{baseline['top1_match']:.4f} -> {oracle['top1_match']:.4f}`")
    lines.extend(["", "## Correlations", ""])
    for name, value in sorted(correlations.items()):
        lines.append(f"- `{name}`: `{value:.4f}`")
    lines.extend(["", "## Decision", "", f"- `{decision}`"])
    output_path.write_text("\n".join(lines))


def main() -> None:
    from kvpress.utils import extract_keys_and_values

    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    bitwidths = [int(part.strip()) for part in args.key_bits.split(",") if part.strip()]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()] or None

    register_all_benchmark_specs()
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
    representative_case: dict = {}

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
                baseline = run_baseline_path(prefix_keys, future_queries, metrics, bits=bits, seed=seed)
                oracle = run_current_oracle_path(prefix_keys, future_queries, metrics, bits=bits, seed=seed, eps=args.eps)
                row = {
                    "dataset": example.dataset,
                    "config": example.config,
                    "row_index": int(example.row_index),
                    "example_offset": int(example_offset),
                    "layer_idx": int(layer_idx),
                    "split_at": int(split_at),
                    "bits": int(bits),
                    "baseline_metrics": baseline["metrics"],
                    "baseline_diagnostics": baseline["diagnostics"],
                    "oracle_metrics": oracle["metrics"],
                    "oracle_diagnostics": oracle["diagnostics"],
                }
                rows.append(row)

                current_cond = float(oracle["diagnostics"]["transform_condition_number_mean"])
                best_cond = float(representative_case.get("transform_condition_number_mean", float("-inf")))
                if current_cond > best_cond:
                    representative_case = {
                        "dataset": example.dataset,
                        "config": example.config,
                        "row_index": int(example.row_index),
                        "layer_idx": int(layer_idx),
                        "bits": int(bits),
                        "transform_condition_number_mean": current_cond,
                        "raw_values": sample_values(prefix_keys, seed=seed),
                        "transformed_values": sample_values(oracle["transformed"], seed=seed),
                        "rotated_values": sample_values(oracle["rotated_flat"], seed=seed),
                    }

    full_summary = {
        str(bits): aggregate_metric_rows(rows, lambda row, bits=bits: int(row["bits"]) == bits)
        for bits in bitwidths
    }
    layer0_rows = [row for row in rows if int(row["layer_idx"]) != 0]
    layer0_summary = {
        str(bits): aggregate_metric_rows(layer0_rows, lambda row, bits=bits: int(row["bits"]) == bits)
        for bits in bitwidths
    }
    correlations = build_correlation_table(layer0_rows)
    decision = decide_diagnosis(rows)
    per_layer_summary = {
        str(bits): {
            "baseline_geometry_distortion": aggregate_per_layer(
                [row for row in rows if int(row["bits"]) == bits],
                "baseline",
                "geometry_distortion",
            ),
            "oracle_geometry_distortion": aggregate_per_layer(
                [row for row in rows if int(row["bits"]) == bits],
                "oracle",
                "geometry_distortion",
            ),
            "top1_delta": aggregate_per_layer_delta(
                [row for row in rows if int(row["bits"]) == bits],
                "top1_match",
            ),
        }
        for bits in bitwidths
    }

    payload = {
        "config": vars(args),
        "full_summary": full_summary,
        "layer0_excluded_summary": layer0_summary,
        "correlations": correlations,
        "decision": decision,
        "per_layer_summary": per_layer_summary,
        "representative_case": representative_case,
        "rows": rows,
    }
    torch.save(payload, output_dir / "diagnosis.pt")
    save_json(output_dir / "metrics.json", payload)
    build_summary_md(output_dir / "summary.md", full_summary, layer0_summary, correlations, decision)

    plot_summary_bars(full_summary, output_dir / "global_current_vs_baseline.png", "Current Oracle vs Baseline")
    plot_summary_bars(
        layer0_summary,
        output_dir / "layer0_excluded_current_vs_baseline.png",
        "Current Oracle vs Baseline (Layer 0 Excluded)",
    )
    plot_per_layer_metric(
        rows,
        "geometry_distortion",
        output_dir / "per_layer_geometry_distortion.png",
        "Per-Layer Geometry Distortion",
        "Geometry Distortion",
    )
    plot_per_layer_delta(
        rows,
        "top1_match",
        output_dir / "per_layer_top1_delta.png",
        "Per-Layer Top-1 Delta (Oracle - Baseline)",
        "Top-1 Delta",
    )
    plot_scatter(
        layer0_rows,
        "transform_condition_number_mean",
        "logit_mse",
        output_dir / "transform_condition_number_vs_logit_delta.png",
        "Transform Condition Number vs Logit MSE Delta",
        "Transform Condition Number",
        "Oracle - Baseline Logit MSE",
    )
    plot_scatter(
        layer0_rows,
        "transform_condition_number_mean",
        "top1_match",
        output_dir / "transform_condition_number_vs_top1_delta.png",
        "Transform Condition Number vs Top-1 Delta",
        "Transform Condition Number",
        "Oracle - Baseline Top-1",
    )
    plot_scatter(
        layer0_rows,
        "transformed_norm_cv",
        "logit_mse",
        output_dir / "norm_cv_vs_logit_delta.png",
        "Transformed Norm CV vs Logit MSE Delta",
        "Transformed Norm CV",
        "Oracle - Baseline Logit MSE",
    )
    plot_scatter(
        layer0_rows,
        "rotated_variance_spread",
        "logit_mse",
        output_dir / "rotation_variance_spread_vs_logit_delta.png",
        "Rotation Variance Spread vs Logit MSE Delta",
        "Rotated Variance Spread",
        "Oracle - Baseline Logit MSE",
    )
    plot_histograms(representative_case, output_dir / "representative_pre_post_transform_histograms.png")


if __name__ == "__main__":
    main()
