from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from scipy import stats

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.stage1.plots.utils import choose_representative_heads, flatten_samples
from experiments.stage1.toolkit import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze stage-1 query distributions.")
    parser.add_argument("--stats_dir", required=True)
    parser.add_argument("--sample_heads", type=int, default=6)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def compute_head_summary(samples: torch.Tensor) -> dict[str, torch.Tensor]:
    mean = samples.mean(dim=2)
    std = samples.std(dim=2, unbiased=False).clamp_min(1e-6)
    z = (samples - mean.unsqueeze(2)) / std.unsqueeze(2)
    skew = z.pow(3).mean(dim=2)
    kurt = z.pow(4).mean(dim=2) - 3.0
    return {
        "mean_abs_skew": skew.abs().mean(dim=-1),
        "mean_abs_excess_kurtosis": kurt.abs().mean(dim=-1),
        "frac_abs_skew_lt_0p5": (skew.abs() < 0.5).float().mean(dim=-1),
        "frac_abs_excess_kurtosis_lt_1": (kurt.abs() < 1.0).float().mean(dim=-1),
    }


def compute_prompt_stability(prompt_second_moment: torch.Tensor, global_second_moment: torch.Tensor) -> torch.Tensor:
    centered = prompt_second_moment.float() - global_second_moment.float().unsqueeze(0)
    numerator = torch.linalg.matrix_norm(centered, ord="fro", dim=(-2, -1))
    denominator = torch.linalg.matrix_norm(global_second_moment.float(), ord="fro", dim=(-2, -1)).clamp_min(1e-6)
    return numerator / denominator.unsqueeze(0)


def plot_histograms(samples: torch.Tensor, head_pairs: list[tuple[int, int]], output_path: Path, title: str) -> None:
    rows = len(head_pairs)
    fig, axes = plt.subplots(rows, 1, figsize=(8, max(3, rows * 2.5)), constrained_layout=True)
    if rows == 1:
        axes = [axes]
    reference_x = torch.linspace(-4, 4, steps=400)
    reference_y = torch.exp(-0.5 * reference_x**2) / torch.sqrt(torch.tensor(2 * torch.pi))
    for axis, (layer, head) in zip(axes, head_pairs):
        values = samples[layer, head].reshape(-1)
        z = (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)
        axis.hist(z.numpy(), bins=50, density=True, alpha=0.7, color="#4477aa")
        axis.plot(reference_x.numpy(), reference_y.numpy(), color="#cc3311", linewidth=1.5)
        axis.set_title(f"Layer {layer}, Head {head}")
    fig.suptitle(title)
    fig.savefig(output_path)
    plt.close(fig)


def plot_eigenspectra(second_moment: torch.Tensor, head_pairs: list[tuple[int, int]], output_path: Path, title: str) -> None:
    rows = len(head_pairs)
    fig, axes = plt.subplots(rows, 1, figsize=(8, max(3, rows * 2.5)), constrained_layout=True)
    if rows == 1:
        axes = [axes]
    for axis, (layer, head) in zip(axes, head_pairs):
        matrix = 0.5 * (second_moment[layer, head] + second_moment[layer, head].transpose(-1, -2))
        eigenvalues = torch.linalg.eigvalsh(matrix.float()).flip(0).clamp_min(1e-8)
        axis.plot(eigenvalues.numpy(), marker="o", linewidth=1)
        axis.set_yscale("log")
        axis.set_title(f"Layer {layer}, Head {head}")
        axis.set_ylabel("eigenvalue")
    fig.suptitle(title)
    fig.savefig(output_path)
    plt.close(fig)


def normality_pvalues(samples: torch.Tensor, head_pairs: list[tuple[int, int]]) -> dict[str, float]:
    pvalues = {}
    for layer, head in head_pairs:
        values = samples[layer, head].reshape(-1)
        z = (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)
        max_points = min(8192, z.numel())
        subset = z[:max_points].numpy()
        pvalues[f"layer{layer}_head{head}"] = float(stats.normaltest(subset).pvalue)
    return pvalues


def build_mode_summary(samples: torch.Tensor, prompt_second_moment: torch.Tensor, global_second_moment: torch.Tensor) -> dict[str, object]:
    head_summary = compute_head_summary(samples)
    stability = compute_prompt_stability(prompt_second_moment, global_second_moment)
    summary = {
        "mean_abs_skew": float(head_summary["mean_abs_skew"].mean().item()),
        "mean_abs_excess_kurtosis": float(head_summary["mean_abs_excess_kurtosis"].mean().item()),
        "frac_abs_skew_lt_0p5": float(head_summary["frac_abs_skew_lt_0p5"].mean().item()),
        "frac_abs_excess_kurtosis_lt_1": float(head_summary["frac_abs_excess_kurtosis_lt_1"].mean().item()),
        "mean_prompt_stability": float(stability.mean().item()),
        "median_prompt_stability": float(stability.median().item()),
    }
    return summary


def main() -> None:
    args = parse_args()
    stats_dir = Path(args.stats_dir)
    output_dir = ensure_dir(args.output_dir or stats_dir / "analysis")
    payload = torch.load(stats_dir / "query_stats.pt", map_location="cpu")

    pre_samples = flatten_samples(payload["sampled_pre_queries"])
    post_samples = flatten_samples(payload["sampled_post_queries"])
    pre_second = payload["pre_second_moment"]
    post_second = payload["post_second_moment"]
    pre_prompt_second = payload["prompt_pre_second_moment"]
    post_prompt_second = payload["prompt_post_second_moment"]

    score_matrix = torch.linalg.matrix_norm(pre_second.float(), ord="fro", dim=(-2, -1))
    head_pairs = choose_representative_heads(score_matrix, args.sample_heads)

    pre_summary = build_mode_summary(pre_samples, pre_prompt_second, pre_second)
    post_summary = build_mode_summary(post_samples, post_prompt_second, post_second)
    pre_summary["normality_pvalues"] = normality_pvalues(pre_samples, head_pairs)
    post_summary["normality_pvalues"] = normality_pvalues(post_samples, head_pairs)

    plot_histograms(pre_samples, head_pairs, output_dir / "pre_histograms.png", "Pre-RoPE Standardized Query Samples")
    plot_histograms(post_samples, head_pairs, output_dir / "post_histograms.png", "Post-RoPE Standardized Query Samples")
    plot_eigenspectra(pre_second, head_pairs, output_dir / "pre_eigenspectra.png", "Pre-RoPE Second-Moment Eigenspectra")
    plot_eigenspectra(post_second, head_pairs, output_dir / "post_eigenspectra.png", "Post-RoPE Second-Moment Eigenspectra")

    analysis = {
        "representative_heads": [{"layer": layer, "head": head} for layer, head in head_pairs],
        "pre": pre_summary,
        "post": post_summary,
    }
    save_json(output_dir / "analysis.json", analysis)

    lines = [
        "# Query Distribution Analysis",
        "",
        "## Representative Heads",
        "",
    ]
    for layer, head in head_pairs:
        lines.append(f"- Layer {layer}, Head {head}")
    lines.extend(
        [
            "",
            "## Pre-RoPE Summary",
            "",
            f"- Mean absolute skew: `{pre_summary['mean_abs_skew']:.4f}`",
            f"- Mean absolute excess kurtosis: `{pre_summary['mean_abs_excess_kurtosis']:.4f}`",
            f"- Fraction |skew| < 0.5: `{pre_summary['frac_abs_skew_lt_0p5']:.4f}`",
            f"- Fraction |excess kurtosis| < 1: `{pre_summary['frac_abs_excess_kurtosis_lt_1']:.4f}`",
            f"- Mean prompt stability: `{pre_summary['mean_prompt_stability']:.4f}`",
            "",
            "## Post-RoPE Summary",
            "",
            f"- Mean absolute skew: `{post_summary['mean_abs_skew']:.4f}`",
            f"- Mean absolute excess kurtosis: `{post_summary['mean_abs_excess_kurtosis']:.4f}`",
            f"- Fraction |skew| < 0.5: `{post_summary['frac_abs_skew_lt_0p5']:.4f}`",
            f"- Fraction |excess kurtosis| < 1: `{post_summary['frac_abs_excess_kurtosis_lt_1']:.4f}`",
            f"- Mean prompt stability: `{post_summary['mean_prompt_stability']:.4f}`",
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
