from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from scipy import stats

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.stage1.plots.utils import (
    apply_figure_title,
    choose_representative_heads,
    flatten_samples,
    standardize_samples,
)
from experiments.stage1.toolkit import ensure_dir

DIST_NOTE = "Skew: asymmetry (0 = symmetric). Excess kurtosis: tail-heaviness vs Gaussian (0 = Gaussian-like tails)."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make focused query-distribution charts from stage-1 artifacts.")
    parser.add_argument("--stats_dir", default="artifacts/stage1/query_stats")
    parser.add_argument("--output_dir", default="artifacts/stage1/query_distribution_charts")
    parser.add_argument("--sample_heads", type=int, default=6)
    return parser.parse_args()


def compute_skew_kurtosis(samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = standardize_samples(samples)
    skew = z.pow(3).mean(dim=2).mean(dim=-1)
    kurt = (z.pow(4).mean(dim=2) - 3.0).mean(dim=-1)
    return skew, kurt


def plot_histograms(samples: torch.Tensor, head_pairs: list[tuple[int, int]], output_path: Path, title: str) -> None:
    rows = len(head_pairs)
    fig, axes = plt.subplots(rows, 1, figsize=(8, max(3, rows * 2.4)), constrained_layout=True)
    if rows == 1:
        axes = [axes]

    reference_x = torch.linspace(-4, 4, steps=400)
    reference_y = torch.exp(-0.5 * reference_x**2) / torch.sqrt(torch.tensor(2 * torch.pi))

    for axis, (layer, head) in zip(axes, head_pairs):
        values = samples[layer, head].reshape(-1)
        z = (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)
        axis.hist(z.numpy(), bins=60, density=True, alpha=0.7, color="#4477AA")
        axis.plot(reference_x.numpy(), reference_y.numpy(), color="#CC3311", linewidth=1.6)
        axis.set_title(f"Layer {layer}, Head {head}")
        axis.set_xlim(-4, 4)
    apply_figure_title(fig, title, note=DIST_NOTE)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_qq(samples: torch.Tensor, head_pairs: list[tuple[int, int]], output_path: Path, title: str) -> None:
    rows = len(head_pairs)
    fig, axes = plt.subplots(rows, 1, figsize=(8, max(3, rows * 2.5)), constrained_layout=True)
    if rows == 1:
        axes = [axes]

    for axis, (layer, head) in zip(axes, head_pairs):
        values = samples[layer, head].reshape(-1)
        z = ((values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)).numpy()
        max_points = min(4000, z.shape[0])
        osm, osr = stats.probplot(z[:max_points], dist="norm", fit=False)
        axis.scatter(osm, osr, s=6, alpha=0.5, color="#4477AA")
        lo = min(min(osm), min(osr))
        hi = max(max(osm), max(osr))
        axis.plot([lo, hi], [lo, hi], color="#CC3311", linewidth=1.4)
        axis.set_title(f"Layer {layer}, Head {head}")
        axis.set_xlabel("Theoretical Quantiles")
        axis.set_ylabel("Sample Quantiles")
    apply_figure_title(fig, title, note=DIST_NOTE)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_heatmap(matrix: torch.Tensor, output_path: Path, title: str, cmap: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    image = ax.imshow(matrix.numpy(), aspect="auto", cmap=cmap)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    apply_figure_title(fig, title, note=DIST_NOTE)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_layer_boxplots(metric: torch.Tensor, output_path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    ax.boxplot([metric[layer].tolist() for layer in range(metric.shape[0])], showfliers=False)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    apply_figure_title(fig, title, note=DIST_NOTE)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_summary(pre_skew: torch.Tensor, pre_kurt: torch.Tensor, post_skew: torch.Tensor, post_kurt: torch.Tensor, output_path: Path, head_pairs: list[tuple[int, int]]) -> None:
    payload = {
        "representative_heads": [{"layer": layer, "head": head} for layer, head in head_pairs],
        "pre": {
            "mean_abs_skew": float(pre_skew.abs().mean().item()),
            "mean_abs_excess_kurtosis": float(pre_kurt.abs().mean().item()),
        },
        "post": {
            "mean_abs_skew": float(post_skew.abs().mean().item()),
            "mean_abs_excess_kurtosis": float(post_kurt.abs().mean().item()),
        },
    }
    output_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    stats_dir = Path(args.stats_dir)
    output_dir = ensure_dir(args.output_dir)

    payload = torch.load(stats_dir / "query_stats.pt", map_location="cpu")
    pre_samples = flatten_samples(payload["sampled_pre_queries"])
    post_samples = flatten_samples(payload["sampled_post_queries"])
    pre_second = payload["pre_second_moment"]

    score_matrix = torch.linalg.matrix_norm(pre_second.float(), ord="fro", dim=(-2, -1))
    head_pairs = choose_representative_heads(score_matrix, args.sample_heads)
    pre_skew, pre_kurt = compute_skew_kurtosis(pre_samples)
    post_skew, post_kurt = compute_skew_kurtosis(post_samples)

    plot_histograms(pre_samples, head_pairs, output_dir / "pre_query_histograms.png", "Pre-RoPE Query Distributions vs Gaussian")
    plot_histograms(post_samples, head_pairs, output_dir / "post_query_histograms.png", "Post-RoPE Query Distributions vs Gaussian")
    plot_qq(pre_samples, head_pairs, output_dir / "pre_query_qq.png", "Pre-RoPE Query QQ Plots")
    plot_qq(post_samples, head_pairs, output_dir / "post_query_qq.png", "Post-RoPE Query QQ Plots")
    plot_heatmap(pre_skew.abs(), output_dir / "pre_skew_heatmap.png", "Pre-RoPE Mean |Skew| by Layer/Head", cmap="YlGnBu")
    plot_heatmap(post_skew.abs(), output_dir / "post_skew_heatmap.png", "Post-RoPE Mean |Skew| by Layer/Head", cmap="YlGnBu")
    plot_heatmap(pre_kurt.abs(), output_dir / "pre_kurtosis_heatmap.png", "Pre-RoPE Mean |Excess Kurtosis| by Layer/Head", cmap="YlOrRd")
    plot_heatmap(post_kurt.abs(), output_dir / "post_kurtosis_heatmap.png", "Post-RoPE Mean |Excess Kurtosis| by Layer/Head", cmap="YlOrRd")
    plot_layer_boxplots(pre_skew.abs(), output_dir / "pre_skew_by_layer_boxplot.png", "Pre-RoPE |Skew| Distribution Across Heads", "|Skew|")
    plot_layer_boxplots(post_skew.abs(), output_dir / "post_skew_by_layer_boxplot.png", "Post-RoPE |Skew| Distribution Across Heads", "|Skew|")
    plot_layer_boxplots(pre_kurt.abs(), output_dir / "pre_kurtosis_by_layer_boxplot.png", "Pre-RoPE |Excess Kurtosis| Distribution Across Heads", "|Excess Kurtosis|")
    plot_layer_boxplots(post_kurt.abs(), output_dir / "post_kurtosis_by_layer_boxplot.png", "Post-RoPE |Excess Kurtosis| Distribution Across Heads", "|Excess Kurtosis|")
    save_summary(pre_skew, pre_kurt, post_skew, post_kurt, output_dir / "summary.json", head_pairs)

    summary_md = [
        "# Query Distribution Charts",
        "",
        "Representative heads:",
    ]
    for layer, head in head_pairs:
        summary_md.append(f"- Layer {layer}, Head {head}")
    summary_md.extend(
        [
            "",
            "- Skew measures asymmetry. `0` means symmetric.",
            "- Excess kurtosis measures tail-heaviness relative to a Gaussian. `0` means Gaussian-like tails.",
            "",
            f"- Pre-RoPE mean |skew|: `{pre_skew.abs().mean().item():.4f}`",
            f"- Post-RoPE mean |skew|: `{post_skew.abs().mean().item():.4f}`",
            f"- Pre-RoPE mean |excess kurtosis|: `{pre_kurt.abs().mean().item():.4f}`",
            f"- Post-RoPE mean |excess kurtosis|: `{post_kurt.abs().mean().item():.4f}`",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(summary_md))


if __name__ == "__main__":
    main()
