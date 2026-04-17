from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot distribution diagnostics for the current oracle diagnosis run.")
    parser.add_argument(
        "--diagnosis_dir",
        default="artifacts/stage1/current_oracle_diagnosis_gpu0",
        help="Directory containing diagnosis.pt from diagnose_current_oracle_backend.py",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional output directory. Defaults to <diagnosis_dir>/distribution_charts",
    )
    return parser.parse_args()


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def dedupe_rows(rows: list[dict], exclude_layer0: bool = True) -> list[dict]:
    seen = set()
    deduped = []
    for row in rows:
        if exclude_layer0 and int(row["layer_idx"]) == 0:
            continue
        key = (row["task"], int(row["example_index"]), int(row["layer_idx"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def compute_standardized_moments(values: list[float]) -> tuple[float, float]:
    tensor = torch.tensor(values, dtype=torch.float32)
    centered = tensor - tensor.mean()
    std = centered.std(unbiased=False).clamp_min(1e-8)
    z = centered / std
    skew = float(z.pow(3).mean().item())
    excess_kurtosis = float((z.pow(4).mean() - 3.0).item())
    return skew, excess_kurtosis


def plot_boxpair(
    rows: list[dict],
    baseline_key: str,
    oracle_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    baseline = [float(row["baseline_diagnostics"][baseline_key]) for row in rows]
    oracle = [float(row["oracle_diagnostics"][oracle_key]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    box = ax.boxplot(
        [baseline, oracle],
        labels=["Baseline k", "Oracle Lk"],
        patch_artist=True,
        widths=0.55,
        showfliers=False,
    )
    colors = ["#4477AA", "#CC6677"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_skew_kurtosis_scatter(rows: list[dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    baseline_skew = [float(row["baseline_diagnostics"]["rotated_mean_abs_skew"]) for row in rows]
    baseline_kurt = [float(row["baseline_diagnostics"]["rotated_mean_abs_excess_kurtosis"]) for row in rows]
    oracle_skew = [float(row["oracle_diagnostics"]["rotated_mean_abs_skew"]) for row in rows]
    oracle_kurt = [float(row["oracle_diagnostics"]["rotated_mean_abs_excess_kurtosis"]) for row in rows]

    ax.scatter(baseline_skew, baseline_kurt, alpha=0.65, s=22, color="#4477AA", label="Baseline k after rotation")
    ax.scatter(oracle_skew, oracle_kurt, alpha=0.65, s=22, color="#CC6677", label="Oracle Lk after rotation")

    ax.set_xlabel("Mean |Skew| after rotation")
    ax.set_ylabel("Mean |Excess Kurtosis| after rotation")
    ax.set_title("Post-Rotation Distribution Shape: Baseline vs Oracle")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_representative_histograms(representative: dict, output_path: Path) -> None:
    raw_skew, raw_kurt = compute_standardized_moments(representative["raw_values"])
    transformed_skew, transformed_kurt = compute_standardized_moments(representative["transformed_values"])
    rotated_skew, rotated_kurt = compute_standardized_moments(representative["rotated_values"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    series = [
        ("Original k", representative["raw_values"], "#4477AA", raw_skew, raw_kurt),
        ("Transformed Lk", representative["transformed_values"], "#CC6677", transformed_skew, transformed_kurt),
        ("Rotated Lk", representative["rotated_values"], "#228833", rotated_skew, rotated_kurt),
    ]
    for ax, (label, values, color, skew, kurt) in zip(axes, series):
        ax.hist(values, bins=50, density=True, color=color, alpha=0.8)
        ax.set_title(label)
        ax.set_xlabel(f"skew={skew:.2f}\nexcess kurtosis={kurt:.2f}")
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        "Representative Distribution Shift Under Current Oracle Transform\n"
        f"task={representative['task']} layer={representative['layer_idx']} bits={representative['bits']} "
        f"cond={representative['transform_condition_number_mean']:.1f}"
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary_md(rows: list[dict], representative: dict, output_path: Path) -> None:
    baseline_norm_cv = [float(row["baseline_diagnostics"]["pre_norm_cv"]) for row in rows]
    oracle_norm_cv = [float(row["oracle_diagnostics"]["pre_norm_cv"]) for row in rows]
    baseline_rank = [float(row["baseline_diagnostics"]["pre_effective_rank"]) for row in rows]
    oracle_rank = [float(row["oracle_diagnostics"]["pre_effective_rank"]) for row in rows]
    baseline_varspread = [float(row["baseline_diagnostics"]["rotated_variance_spread"]) for row in rows]
    oracle_varspread = [float(row["oracle_diagnostics"]["rotated_variance_spread"]) for row in rows]
    baseline_skew = [float(row["baseline_diagnostics"]["rotated_mean_abs_skew"]) for row in rows]
    oracle_skew = [float(row["oracle_diagnostics"]["rotated_mean_abs_skew"]) for row in rows]
    baseline_kurt = [float(row["baseline_diagnostics"]["rotated_mean_abs_excess_kurtosis"]) for row in rows]
    oracle_kurt = [float(row["oracle_diagnostics"]["rotated_mean_abs_excess_kurtosis"]) for row in rows]

    lines = [
        "# Distribution Diagnostics Summary",
        "",
        "These charts use layer-0-excluded, bit-deduplicated rows from `diagnosis.pt`.",
        "",
        f"- Mean pre-transform norm CV: baseline `{sum(baseline_norm_cv)/len(baseline_norm_cv):.4f}`, oracle `{sum(oracle_norm_cv)/len(oracle_norm_cv):.4f}`",
        f"- Mean pre-transform effective rank: baseline `{sum(baseline_rank)/len(baseline_rank):.2f}`, oracle `{sum(oracle_rank)/len(oracle_rank):.2f}`",
        f"- Mean post-rotation variance spread: baseline `{sum(baseline_varspread)/len(baseline_varspread):.4f}`, oracle `{sum(oracle_varspread)/len(oracle_varspread):.4f}`",
        f"- Mean post-rotation |skew|: baseline `{sum(baseline_skew)/len(baseline_skew):.4f}`, oracle `{sum(oracle_skew)/len(oracle_skew):.4f}`",
        f"- Mean post-rotation |excess kurtosis|: baseline `{sum(baseline_kurt)/len(baseline_kurt):.4f}`, oracle `{sum(oracle_kurt)/len(oracle_kurt):.4f}`",
        "",
        "## Representative Case",
        "",
        f"- task `{representative['task']}`, layer `{representative['layer_idx']}`, bits `{representative['bits']}`",
        f"- transform condition number `{representative['transform_condition_number_mean']:.2f}`",
        "",
        "## Charts",
        "",
        "- `pre_transform_norm_cv_comparison.png`",
        "- `pre_transform_effective_rank_comparison.png`",
        "- `post_rotation_variance_spread_comparison.png`",
        "- `post_rotation_skew_comparison.png`",
        "- `post_rotation_excess_kurtosis_comparison.png`",
        "- `post_rotation_skew_vs_kurtosis.png`",
        "- `representative_distribution_moments.png`",
        "",
        "Interpretation:",
        "",
        "- Higher norm CV and lower effective rank indicate stronger anisotropy before rotation.",
        "- Higher variance spread, |skew|, and |excess kurtosis| indicate a less isotropic / less Gaussian-like distribution after rotation.",
    ]
    output_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    diagnosis_dir = Path(args.diagnosis_dir)
    output_dir = ensure_dir(args.output_dir or (diagnosis_dir / "distribution_charts"))
    payload = torch.load(diagnosis_dir / "diagnosis.pt", map_location="cpu")
    rows = dedupe_rows(payload["rows"], exclude_layer0=True)
    representative = payload["representative_case"]

    plot_boxpair(
        rows,
        baseline_key="pre_norm_cv",
        oracle_key="pre_norm_cv",
        ylabel="Norm Coefficient of Variation",
        title="Before Rotation: k vs Lk Norm Dispersion",
        output_path=output_dir / "pre_transform_norm_cv_comparison.png",
    )
    plot_boxpair(
        rows,
        baseline_key="pre_effective_rank",
        oracle_key="pre_effective_rank",
        ylabel="Effective Rank / Participation Ratio",
        title="Before Rotation: k vs Lk Effective Rank",
        output_path=output_dir / "pre_transform_effective_rank_comparison.png",
    )
    plot_boxpair(
        rows,
        baseline_key="rotated_variance_spread",
        oracle_key="rotated_variance_spread",
        ylabel="Max Var / Min Var",
        title="After Same Random Rotation: Coordinate Variance Spread",
        output_path=output_dir / "post_rotation_variance_spread_comparison.png",
    )
    plot_boxpair(
        rows,
        baseline_key="rotated_mean_abs_skew",
        oracle_key="rotated_mean_abs_skew",
        ylabel="Mean |Skew|",
        title="After Same Random Rotation: |Skew| Comparison",
        output_path=output_dir / "post_rotation_skew_comparison.png",
    )
    plot_boxpair(
        rows,
        baseline_key="rotated_mean_abs_excess_kurtosis",
        oracle_key="rotated_mean_abs_excess_kurtosis",
        ylabel="Mean |Excess Kurtosis|",
        title="After Same Random Rotation: |Excess Kurtosis| Comparison",
        output_path=output_dir / "post_rotation_excess_kurtosis_comparison.png",
    )
    plot_skew_kurtosis_scatter(rows, output_dir / "post_rotation_skew_vs_kurtosis.png")
    plot_representative_histograms(representative, output_dir / "representative_distribution_moments.png")
    write_summary_md(rows, representative, output_dir / "summary.md")


if __name__ == "__main__":
    main()
