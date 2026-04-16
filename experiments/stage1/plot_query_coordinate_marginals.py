from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from scipy import stats

DIST_NOTE = "Each panel is a single coordinate marginal. Skew: asymmetry (0 = symmetric). Excess kurtosis: tail-heaviness vs Gaussian (0 = Gaussian-like tails)."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot individual query-coordinate marginals from stage-1 artifacts.")
    parser.add_argument("--stats_dir", default="artifacts/stage1/query_stats")
    parser.add_argument("--output_dir", default="artifacts/stage1/query_coordinate_marginals")
    parser.add_argument("--sample_heads", type=int, default=4)
    return parser.parse_args()


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def apply_figure_title(fig: plt.Figure, title: str) -> None:
    fig.suptitle(f"{title}\n{DIST_NOTE}", fontsize=12)


def flatten_samples(sample_list: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([sample.float() for sample in sample_list], dim=2)


def standardize(samples: torch.Tensor) -> torch.Tensor:
    mean = samples.mean(dim=2, keepdim=True)
    std = samples.std(dim=2, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (samples - mean) / std


def compute_coordinate_stats(samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = standardize(samples)
    skew = z.pow(3).mean(dim=2)
    kurt = z.pow(4).mean(dim=2) - 3.0
    return skew, kurt


def choose_representative_heads(second_moment: torch.Tensor, sample_heads: int) -> list[tuple[int, int]]:
    score_matrix = torch.linalg.matrix_norm(second_moment.float(), ord="fro", dim=(-2, -1))
    layer_count, head_count = score_matrix.shape
    pairs = []
    for layer in torch.linspace(0, layer_count - 1, steps=min(sample_heads, layer_count)).round().long().tolist():
        pairs.append((int(layer), 0))
    remaining = sample_heads - len(pairs)
    if remaining > 0:
        flat_indices = score_matrix.flatten().argsort(descending=True).tolist()
        for index in flat_indices:
            layer = index // head_count
            head = index % head_count
            pair = (int(layer), int(head))
            if pair not in pairs:
                pairs.append(pair)
            if len(pairs) >= sample_heads:
                break
    return pairs[:sample_heads]


def choose_coordinates(
    skew: torch.Tensor,
    kurt: torch.Tensor,
    head_pairs: list[tuple[int, int]],
) -> dict[tuple[int, int], list[dict[str, int | float | str]]]:
    selected: dict[tuple[int, int], list[dict[str, int | float | str]]] = {}
    for layer, head in head_pairs:
        head_skew = skew[layer, head].abs()
        head_kurt = kurt[layer, head].abs()
        gaussian_score = head_skew + head_kurt

        gaussian_like = int(torch.argmin(gaussian_score).item())
        asymmetric = int(torch.argmax(head_skew).item())
        heavy_tailed = int(torch.argmax(head_kurt).item())

        coords = []
        seen: set[int] = set()
        for label, coord in [
            ("Gaussian-like", gaussian_like),
            ("Most skewed", asymmetric),
            ("Heaviest tail", heavy_tailed),
        ]:
            if coord in seen:
                continue
            seen.add(coord)
            coords.append(
                {
                    "label": label,
                    "coord": coord,
                    "abs_skew": float(head_skew[coord].item()),
                    "abs_excess_kurtosis": float(head_kurt[coord].item()),
                }
            )
        selected[(layer, head)] = coords
    return selected


def plot_coordinate_histograms(
    samples: torch.Tensor,
    selection: dict[tuple[int, int], list[dict[str, int | float | str]]],
    output_path: Path,
    title: str,
) -> None:
    head_pairs = list(selection.keys())
    n_rows = len(head_pairs)
    n_cols = max(len(coords) for coords in selection.values())
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, max(3.2, 2.8 * n_rows)), constrained_layout=True)
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    reference_x = torch.linspace(-4, 4, steps=400)
    reference_y = torch.exp(-0.5 * reference_x**2) / torch.sqrt(torch.tensor(2 * torch.pi))

    for row_idx, (layer, head) in enumerate(head_pairs):
        coords = selection[(layer, head)]
        for col_idx in range(n_cols):
            axis = axes[row_idx][col_idx]
            if col_idx >= len(coords):
                axis.axis("off")
                continue
            coord_info = coords[col_idx]
            coord = int(coord_info["coord"])
            values = samples[layer, head, :, coord]
            z = (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)
            axis.hist(z.numpy(), bins=50, density=True, alpha=0.72, color="#4477AA")
            axis.plot(reference_x.numpy(), reference_y.numpy(), color="#CC3311", linewidth=1.6)
            axis.set_xlim(-4, 4)
            axis.set_title(
                f"L{layer} H{head} C{coord}\n{coord_info['label']} | |skew|={coord_info['abs_skew']:.2f}, |kurt|={coord_info['abs_excess_kurtosis']:.2f}",
                fontsize=10,
            )

    apply_figure_title(fig, title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_coordinate_qq(
    samples: torch.Tensor,
    selection: dict[tuple[int, int], list[dict[str, int | float | str]]],
    output_path: Path,
    title: str,
) -> None:
    head_pairs = list(selection.keys())
    n_rows = len(head_pairs)
    n_cols = max(len(coords) for coords in selection.values())
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, max(3.4, 3.0 * n_rows)), constrained_layout=True)
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    for row_idx, (layer, head) in enumerate(head_pairs):
        coords = selection[(layer, head)]
        for col_idx in range(n_cols):
            axis = axes[row_idx][col_idx]
            if col_idx >= len(coords):
                axis.axis("off")
                continue
            coord_info = coords[col_idx]
            coord = int(coord_info["coord"])
            values = samples[layer, head, :, coord]
            z = ((values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)).numpy()
            max_points = min(4000, z.shape[0])
            osm, osr = stats.probplot(z[:max_points], dist="norm", fit=False)
            axis.scatter(osm, osr, s=6, alpha=0.5, color="#4477AA")
            lo = min(min(osm), min(osr))
            hi = max(max(osm), max(osr))
            axis.plot([lo, hi], [lo, hi], color="#CC3311", linewidth=1.4)
            axis.set_title(
                f"L{layer} H{head} C{coord}\n{coord_info['label']}",
                fontsize=10,
            )
            axis.set_xlabel("Theoretical Quantiles")
            axis.set_ylabel("Sample Quantiles")

    apply_figure_title(fig, title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_head_coordinate_heatmap(metric: torch.Tensor, output_path: Path, title: str, cmap: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    image = ax.imshow(metric.numpy().T, aspect="auto", cmap=cmap, origin="lower")
    ax.set_xlabel("Layer/Head selection index")
    ax.set_ylabel("Coordinate")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    apply_figure_title(fig, title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    stats_dir = Path(args.stats_dir)
    output_dir = ensure_dir(args.output_dir)

    payload = torch.load(stats_dir / "query_stats.pt", map_location="cpu")
    pre_samples = flatten_samples(payload["sampled_pre_queries"])
    post_samples = flatten_samples(payload["sampled_post_queries"])
    pre_second = payload["pre_second_moment"]

    head_pairs = choose_representative_heads(pre_second, args.sample_heads)
    pre_skew, pre_kurt = compute_coordinate_stats(pre_samples)
    post_skew, post_kurt = compute_coordinate_stats(post_samples)
    selection = choose_coordinates(post_skew, post_kurt, head_pairs)

    plot_coordinate_histograms(
        pre_samples,
        selection,
        output_dir / "pre_coordinate_histograms.png",
        "Pre-RoPE Individual Coordinate Marginals vs Gaussian",
    )
    plot_coordinate_histograms(
        post_samples,
        selection,
        output_dir / "post_coordinate_histograms.png",
        "Post-RoPE Individual Coordinate Marginals vs Gaussian",
    )
    plot_coordinate_qq(
        pre_samples,
        selection,
        output_dir / "pre_coordinate_qq.png",
        "Pre-RoPE Individual Coordinate QQ Plots",
    )
    plot_coordinate_qq(
        post_samples,
        selection,
        output_dir / "post_coordinate_qq.png",
        "Post-RoPE Individual Coordinate QQ Plots",
    )

    selection_indices = torch.tensor(head_pairs, dtype=torch.long)
    pre_skew_selected = pre_skew[selection_indices[:, 0], selection_indices[:, 1]]
    post_skew_selected = post_skew[selection_indices[:, 0], selection_indices[:, 1]]
    pre_kurt_selected = pre_kurt[selection_indices[:, 0], selection_indices[:, 1]]
    post_kurt_selected = post_kurt[selection_indices[:, 0], selection_indices[:, 1]]
    plot_head_coordinate_heatmap(
        pre_skew_selected.abs(),
        output_dir / "pre_coordinate_skew_heatmap.png",
        "Pre-RoPE |Skew| by Coordinate for Selected Heads",
        cmap="YlGnBu",
    )
    plot_head_coordinate_heatmap(
        post_skew_selected.abs(),
        output_dir / "post_coordinate_skew_heatmap.png",
        "Post-RoPE |Skew| by Coordinate for Selected Heads",
        cmap="YlGnBu",
    )
    plot_head_coordinate_heatmap(
        pre_kurt_selected.abs(),
        output_dir / "pre_coordinate_kurtosis_heatmap.png",
        "Pre-RoPE |Excess Kurtosis| by Coordinate for Selected Heads",
        cmap="YlOrRd",
    )
    plot_head_coordinate_heatmap(
        post_kurt_selected.abs(),
        output_dir / "post_coordinate_kurtosis_heatmap.png",
        "Post-RoPE |Excess Kurtosis| by Coordinate for Selected Heads",
        cmap="YlOrRd",
    )

    summary = {
        "representative_heads": [],
    }
    summary_md = [
        "# Individual Coordinate Marginals",
        "",
        "These plots show true single-coordinate marginals, not pooled head-level values.",
        "",
        "- Coordinates were chosen per selected head as:",
        "  - one most Gaussian-like coordinate",
        "  - one most skewed coordinate",
        "  - one heaviest-tail coordinate",
        "",
        "- Skew measures asymmetry. `0` means symmetric.",
        "- Excess kurtosis measures tail-heaviness relative to a Gaussian. `0` means Gaussian-like tails.",
        "",
        "Selected heads and coordinates:",
    ]
    for layer, head in head_pairs:
        coords = selection[(layer, head)]
        coord_payload = []
        coord_text = ", ".join(
            f"{item['label']}: C{item['coord']} (|skew|={item['abs_skew']:.2f}, |kurt|={item['abs_excess_kurtosis']:.2f})"
            for item in coords
        )
        summary_md.append(f"- Layer {layer}, Head {head}: {coord_text}")
        for item in coords:
            coord_payload.append(
                {
                    "label": item["label"],
                    "coord": int(item["coord"]),
                    "abs_skew": float(item["abs_skew"]),
                    "abs_excess_kurtosis": float(item["abs_excess_kurtosis"]),
                }
            )
        summary["representative_heads"].append({"layer": layer, "head": head, "coordinates": coord_payload})

    (output_dir / "summary.md").write_text("\n".join(summary_md))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
