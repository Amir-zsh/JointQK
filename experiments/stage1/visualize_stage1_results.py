from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate charts for stage-1 results.")
    parser.add_argument("--query_stats_dir", default="artifacts/stage1/query_stats")
    parser.add_argument("--oracle_dir", default="artifacts/stage1/oracle_v3_study")
    parser.add_argument("--output_dir", default="artifacts/stage1/visuals")
    return parser.parse_args()


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_query_analysis(query_stats_dir: Path) -> dict:
    return json.loads((query_stats_dir / "analysis" / "analysis.json").read_text())


def load_query_stats(query_stats_dir: Path) -> dict:
    return torch.load(query_stats_dir / "query_stats.pt", map_location="cpu")


def load_oracle_study(oracle_dir: Path) -> dict:
    return torch.load(oracle_dir / "oracle_study.pt", map_location="cpu")


def plot_query_summary_bars(analysis: dict, output_path: Path) -> None:
    metrics = [
        ("mean_abs_skew", "Mean |Skew|"),
        ("mean_abs_excess_kurtosis", "Mean |Excess Kurtosis|"),
        ("frac_abs_skew_lt_0p5", "Frac |Skew| < 0.5"),
        ("frac_abs_excess_kurtosis_lt_1", "Frac |Kurtosis| < 1"),
        ("mean_prompt_stability", "Mean Prompt Stability"),
    ]
    pre = [analysis["pre"][key] for key, _ in metrics]
    post = [analysis["post"][key] for key, _ in metrics]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(metrics))
    width = 0.38
    ax.bar([i - width / 2 for i in x], pre, width=width, label="Pre-RoPE", color="#4477AA")
    ax.bar([i + width / 2 for i in x], post, width=width, label="Post-RoPE", color="#CC6677")
    ax.set_xticks(list(x))
    ax.set_xticklabels([label for _, label in metrics], rotation=20, ha="right")
    ax.set_title("Stage 1A Query Distribution Summary")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_second_moment_stability(query_stats: dict, output_path: Path) -> None:
    pre_global = query_stats["pre_second_moment"].float()
    post_global = query_stats["post_second_moment"].float()
    pre_prompt = query_stats["prompt_pre_second_moment"].float()
    post_prompt = query_stats["prompt_post_second_moment"].float()

    def compute_stability(prompt_tensor: torch.Tensor, global_tensor: torch.Tensor) -> torch.Tensor:
        centered = prompt_tensor - global_tensor.unsqueeze(0)
        numerator = torch.linalg.matrix_norm(centered, ord="fro", dim=(-2, -1))
        denominator = torch.linalg.matrix_norm(global_tensor, ord="fro", dim=(-2, -1)).clamp_min(1e-6)
        return numerator / denominator.unsqueeze(0)

    pre_stability = compute_stability(pre_prompt, pre_global).mean(dim=0).mean(dim=-1)
    post_stability = compute_stability(post_prompt, post_global).mean(dim=0).mean(dim=-1)

    layers = range(pre_stability.shape[0])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(layers, pre_stability.numpy(), label="Pre-RoPE", color="#4477AA", linewidth=2)
    ax.plot(layers, post_stability.numpy(), label="Post-RoPE", color="#CC6677", linewidth=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Relative Frobenius Drift")
    ax.set_title("Prompt-to-Prompt Second-Moment Stability by Layer")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def aggregate_per_layer(orcl: dict, bits: int, metric: str, mode: str) -> list[float]:
    layer_values: dict[int, list[float]] = defaultdict(list)
    for example in orcl["per_example"]:
        for layer in example["layers"]:
            layer_values[layer["layer_idx"]].append(layer[f"{mode}_{bits}bit"][metric])
    return [sum(layer_values[layer]) / len(layer_values[layer]) for layer in sorted(layer_values)]


def aggregate_deltas(orcl: dict, bits: int, metric: str) -> list[float]:
    deltas: dict[int, list[float]] = defaultdict(list)
    for example in orcl["per_example"]:
        for layer in example["layers"]:
            oracle_metric = layer[f"oracle_{bits}bit"][metric]
            baseline_metric = layer[f"baseline_{bits}bit"][metric]
            deltas[layer["layer_idx"]].append(oracle_metric - baseline_metric)
    return [sum(deltas[layer]) / len(deltas[layer]) for layer in sorted(deltas)]


def plot_global_oracle_vs_baseline(summary: dict, output_path: Path) -> None:
    metrics = [
        ("logit_mse", "Logit MSE"),
        ("logit_cosine", "Logit Cosine"),
        ("top1_match", "Top-1 Match"),
        ("top5_containment", "Top-5 Containment"),
        ("geometry_distortion", "Geometry Distortion"),
    ]
    bits_list = [2, 3, 4]
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4.8), constrained_layout=True)
    colors = {"baseline": "#4477AA", "oracle": "#CC6677"}

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        x = range(len(bits_list))
        baseline_vals = [summary[f"baseline_{bits}bit"][metric_key] for bits in bits_list]
        oracle_vals = [summary[f"oracle_{bits}bit"][metric_key] for bits in bits_list]
        width = 0.36
        ax.bar([i - width / 2 for i in x], baseline_vals, width=width, color=colors["baseline"], label="Baseline")
        ax.bar([i + width / 2 for i in x], oracle_vals, width=width, color=colors["oracle"], label="Oracle")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{bits}-bit" for bits in bits_list])
        ax.set_title(metric_label)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend()
    fig.suptitle("Stage 1B Global Oracle vs Baseline Comparison")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_per_layer_metric(orcl: dict, metric: str, output_path: Path, title: str, ylabel: str) -> None:
    bits_list = [2, 3, 4]
    fig, axes = plt.subplots(len(bits_list), 1, figsize=(10, 10), constrained_layout=True)
    colors = {"baseline": "#4477AA", "oracle": "#CC6677"}
    for ax, bits in zip(axes, bits_list):
        baseline = aggregate_per_layer(orcl, bits, metric, "baseline")
        oracle = aggregate_per_layer(orcl, bits, metric, "oracle")
        layers = range(len(baseline))
        ax.plot(layers, baseline, color=colors["baseline"], linewidth=2, label="Baseline")
        ax.plot(layers, oracle, color=colors["oracle"], linewidth=2, label="Oracle")
        ax.set_title(f"{bits}-bit")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Layer")
    axes[0].legend()
    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_per_layer_delta(orcl: dict, metric: str, output_path: Path, title: str, ylabel: str) -> None:
    bits_list = [2, 3, 4]
    fig, axes = plt.subplots(len(bits_list), 1, figsize=(10, 10), constrained_layout=True)
    colors = {2: "#228833", 3: "#CCBB44", 4: "#EE6677"}
    for ax, bits in zip(axes, bits_list):
        delta = aggregate_deltas(orcl, bits, metric)
        layers = range(len(delta))
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.7)
        ax.bar(layers, delta, color=colors[bits], alpha=0.85)
        ax.set_title(f"{bits}-bit")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("Layer")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_delta_scatter(orcl: dict, output_path: Path) -> None:
    bits_list = [2, 3, 4]
    colors = {2: "#228833", 3: "#CCBB44", 4: "#EE6677"}
    fig, axes = plt.subplots(1, len(bits_list), figsize=(15, 4.8), constrained_layout=True)
    for ax, bits in zip(axes, bits_list):
        x_vals = []
        y_vals = []
        for example in orcl["per_example"]:
            for layer in example["layers"]:
                baseline = layer[f"baseline_{bits}bit"]
                oracle = layer[f"oracle_{bits}bit"]
                x_vals.append(oracle["logit_mse"] - baseline["logit_mse"])
                y_vals.append(oracle["top1_match"] - baseline["top1_match"])
        ax.scatter(x_vals, y_vals, alpha=0.65, s=18, color=colors[bits])
        ax.axvline(0.0, color="black", linewidth=1, alpha=0.7)
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.7)
        ax.set_title(f"{bits}-bit")
        ax.set_xlabel("Oracle - Baseline Logit MSE")
        ax.set_ylabel("Oracle - Baseline Top-1")
        ax.grid(alpha=0.2)
    fig.suptitle("Per Layer-Example Tradeoff: Better MSE vs Better Ranking")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_layer0_dominance(orcl: dict, output_path: Path) -> None:
    bits_list = [2, 3, 4]
    fig, axes = plt.subplots(1, len(bits_list), figsize=(15, 4.5), constrained_layout=True)
    for ax, bits in zip(axes, bits_list):
        baseline = aggregate_per_layer(orcl, bits, "logit_mse", "baseline")
        layers = list(range(len(baseline)))
        colors = ["#CC6677" if layer == 0 else "#4477AA" for layer in layers]
        ax.bar(layers, baseline, color=colors)
        ax.set_title(f"{bits}-bit Baseline Logit MSE")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Avg Logit MSE")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Layer Contribution Imbalance in the Oracle Study")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary_md(analysis: dict, oracle: dict, output_path: Path) -> None:
    lines = [
        "# Stage 1 Visualization Summary",
        "",
        "## Query Distribution",
        "",
        f"- Pre-RoPE mean absolute skew: `{analysis['pre']['mean_abs_skew']:.4f}`",
        f"- Post-RoPE mean absolute skew: `{analysis['post']['mean_abs_skew']:.4f}`",
        f"- Pre-RoPE mean prompt stability: `{analysis['pre']['mean_prompt_stability']:.4f}`",
        f"- Post-RoPE mean prompt stability: `{analysis['post']['mean_prompt_stability']:.4f}`",
        "",
        "## Oracle Study",
        "",
    ]
    for bits in [2, 3, 4]:
        baseline = oracle["summary"][f"baseline_{bits}bit"]
        oracle_metrics = oracle["summary"][f"oracle_{bits}bit"]
        lines.append(f"- {bits}-bit logit MSE: `{baseline['logit_mse']:.4f} -> {oracle_metrics['logit_mse']:.4f}`")
        lines.append(f"- {bits}-bit top-1: `{baseline['top1_match']:.4f} -> {oracle_metrics['top1_match']:.4f}`")
    lines.extend(
        [
            "",
            "## Charts",
            "",
            "- `query_summary_bars.png`",
            "- `second_moment_stability.png`",
            "- `oracle_global_comparison.png`",
            "- `oracle_per_layer_logit_mse.png`",
            "- `oracle_per_layer_top1_delta.png`",
            "- `oracle_delta_scatter.png`",
            "- `layer0_dominance.png`",
        ]
    )
    output_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    query_stats_dir = Path(args.query_stats_dir)
    oracle_dir = Path(args.oracle_dir)
    output_dir = ensure_dir(args.output_dir)

    analysis = load_query_analysis(query_stats_dir)
    query_stats = load_query_stats(query_stats_dir)
    oracle = load_oracle_study(oracle_dir)

    plot_query_summary_bars(analysis, output_dir / "query_summary_bars.png")
    plot_second_moment_stability(query_stats, output_dir / "second_moment_stability.png")
    plot_global_oracle_vs_baseline(oracle["summary"], output_dir / "oracle_global_comparison.png")
    plot_per_layer_metric(
        oracle,
        metric="logit_mse",
        output_path=output_dir / "oracle_per_layer_logit_mse.png",
        title="Per-Layer Average Logit MSE: Baseline vs Oracle",
        ylabel="Avg Logit MSE",
    )
    plot_per_layer_delta(
        oracle,
        metric="top1_match",
        output_path=output_dir / "oracle_per_layer_top1_delta.png",
        title="Per-Layer Oracle - Baseline Top-1 Delta",
        ylabel="Top-1 Delta",
    )
    plot_delta_scatter(oracle, output_dir / "oracle_delta_scatter.png")
    plot_layer0_dominance(oracle, output_dir / "layer0_dominance.png")
    write_summary_md(analysis, oracle, output_dir / "summary.md")


if __name__ == "__main__":
    main()
