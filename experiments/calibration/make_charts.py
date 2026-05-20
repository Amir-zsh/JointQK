#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.calibration.common import DEFAULT_ARTIFACT_ROOT, DEFAULT_RUN_ID, RunPaths, read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create final calibration analysis charts.")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    return parser.parse_args()


def split_key(key: str) -> dict[str, str]:
    parts = key.split("|")
    out = {"method": parts[0]}
    for part in parts[1:]:
        k, v = part.split("=", 1)
        out[k] = v
    return out


def collect(summary: dict[str, Any], *, family: str, eval_name: str, metric: str, layer0: bool) -> dict[str, list[tuple[int, float]]]:
    out: dict[str, list[tuple[int, float]]] = {}
    for key, group in summary["groups"].items():
        info = split_key(key)
        if info.get("family") != family or info.get("eval") != eval_name or info.get("layer0") != str(layer0):
            continue
        if metric not in group:
            continue
        out.setdefault(info["method"], []).append((int(info["n"]), float(group[metric]["mean"])))
    for vals in out.values():
        vals.sort()
    return out


def line_chart(data: dict[str, list[tuple[int, float]]], path: Path, *, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for label, vals in sorted(data.items()):
        xs = [x for x, _ in vals]
        ys = [y for _, y in vals]
        ax.plot(xs, ys, marker="o", linewidth=2.0, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Calibration examples per task/source")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def bar_latest(data: dict[str, list[tuple[int, float]]], path: Path, *, title: str, ylabel: str) -> None:
    labels = []
    values = []
    for label, vals in sorted(data.items()):
        if vals:
            labels.append(label)
            values.append(vals[-1][1])
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(labels, values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def prefixed(prefix: str, data: dict[str, list[tuple[int, float]]]) -> dict[str, list[tuple[int, float]]]:
    return {f"{label} {prefix}": vals for label, vals in data.items()}


def main() -> None:
    args = parse_args()
    paths = RunPaths.from_args(args.artifact_root, args.run_id)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    summary = read_json(paths.analysis_dir / "analysis_summary.json")
    outputs: list[Path] = []

    data = collect(summary, family="k", eval_name="pooled", metric="logit_error_waterfill_k3", layer0=False)
    path = paths.reports_dir / "k_pooled_logit_error_k3.png"
    line_chart(data, path, title="Pooled held-out K logit error with water-fill at K=3", ylabel="Analytic logit error")
    outputs.append(path)

    path = paths.reports_dir / "k_method_comparison_k3.png"
    bar_latest(data, path, title="K water-fill basis comparison at largest calibration size", ylabel="Analytic logit error")
    outputs.append(path)

    data_k_mse = collect(summary, family="k", eval_name="pooled", metric="k_mse_waterfill_k3", layer0=False)
    path = paths.reports_dir / "k_pooled_mse_k3.png"
    line_chart(data_k_mse, path, title="Pooled held-out K reconstruction error with water-fill at K=3", ylabel="Analytic K MSE")
    outputs.append(path)

    path = paths.reports_dir / "k_mse_method_comparison_k3.png"
    bar_latest(data_k_mse, path, title="K water-fill MSE basis comparison at largest calibration size", ylabel="Analytic K MSE")
    outputs.append(path)

    data_uniform_logit = collect(summary, family="k", eval_name="pooled", metric="logit_error_uniform_k3", layer0=False)
    data_waterfill_logit = collect(summary, family="k", eval_name="pooled", metric="logit_error_waterfill_k3", layer0=False)
    path = paths.reports_dir / "k_waterfill_vs_uniform_logit_k3.png"
    line_chart(
        {**prefixed("uniform", data_uniform_logit), **prefixed("water-fill", data_waterfill_logit)},
        path,
        title="K logit error: uniform vs water-fill at K=3",
        ylabel="Analytic logit error",
    )
    outputs.append(path)

    data_uniform_mse = collect(summary, family="k", eval_name="pooled", metric="k_mse_uniform_k3", layer0=False)
    data_waterfill_mse = collect(summary, family="k", eval_name="pooled", metric="k_mse_waterfill_k3", layer0=False)
    path = paths.reports_dir / "k_waterfill_vs_uniform_mse_k3.png"
    line_chart(
        {**prefixed("uniform", data_uniform_mse), **prefixed("water-fill", data_waterfill_mse)},
        path,
        title="K reconstruction error: uniform vs water-fill at K=3",
        ylabel="Analytic K MSE",
    )
    outputs.append(path)

    data_active = collect(summary, family="k", eval_name="pooled", metric="waterfill_active_coords_k3", layer0=False)
    path = paths.reports_dir / "k_waterfill_active_coords_k3.png"
    line_chart(data_active, path, title="K water-fill active coordinates at K=3", ylabel="Average active coordinates")
    outputs.append(path)

    data_v = collect(summary, family="v", eval_name="pooled", metric="v_mse_v3", layer0=False)
    path = paths.reports_dir / "v_pooled_mse_v3.png"
    line_chart(data_v, path, title="Pooled held-out V reconstruction error at V=3", ylabel="Analytic V MSE")
    outputs.append(path)

    data_l0 = {
        "K layer0 excluded": collect(summary, family="k", eval_name="pooled", metric="logit_error_waterfill_k3", layer0=False).get("jointqk", []),
        "K layer0 included": collect(summary, family="k", eval_name="pooled", metric="logit_error_waterfill_k3", layer0=True).get("jointqk", []),
    }
    path = paths.reports_dir / "layer0_sensitivity.png"
    line_chart(data_l0, path, title="JointQK layer-0 sensitivity", ylabel="Analytic logit error")
    outputs.append(path)

    lines = [
        "# Final Calibration Charts",
        "",
        f"- Analysis rows: `{summary['metadata']['num_rows']}`",
        f"- Expected trials: `{summary['metadata']['expected_trials']}`",
        "",
        "Generated files:",
    ]
    for output in outputs:
        lines.append(f"- `{output.name}`")
    (paths.reports_dir / "README.md").write_text("\n".join(lines) + "\n")
    outputs.append(paths.reports_dir / "README.md")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
