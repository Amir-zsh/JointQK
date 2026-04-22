from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "kvpress"))
else:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "kvpress"))

from experiments.stage1.toolkit import (
    Stage1MSECompressor,
    apply_headwise_linear,
    compute_grouped_query_second_moment,
    ensure_dir,
    factorize_metric_batch,
    load_model_and_tokenizer,
    run_prefill_and_capture,
    split_prefix_and_future,
    trim_queries,
)
from experiments.stage1.data import fetch_example, get_dataset_spec
from experiments.stage1.diagnosis import compute_second_moment, instrument_compressor_path, transform_matrix_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 2x2 distribution panel charts for selected layers.")
    parser.add_argument("--diagnosis_dir", default="artifacts/stage1/current_oracle_diagnosis_gpu0")
    parser.add_argument(
        "--model",
        default="/vault/amir/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218",
    )
    parser.add_argument("--dataset", default="longbench-e")
    parser.add_argument("--n_sink", type=int, default=4)
    parser.add_argument("--prefix_fraction", type=float, default=0.75)
    parser.add_argument("--min_future_tokens", type=int, default=32)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--layers",
        default="auto",
        help="Comma-separated layer indices to render, or 'auto' for layer 0 + early/mid/late representatives.",
    )
    return parser.parse_args()


def standardized_moments_from_matrix(matrix: torch.Tensor) -> tuple[float, float]:
    flat = matrix.reshape(-1).float()
    centered = flat - flat.mean()
    std = centered.std(unbiased=False).clamp_min(1e-8)
    z = centered / std
    skew = float(z.pow(3).mean().item())
    excess_kurtosis = float((z.pow(4).mean() - 3.0).item())
    return skew, excess_kurtosis


def sample_condition_number(matrix_2d: torch.Tensor) -> float:
    second = compute_second_moment(matrix_2d.float())
    singular = torch.linalg.svdvals(second)
    return float((singular[0] / singular[-1].clamp_min(1e-8)).item())


def choose_representative_layers(rows: list[dict]) -> list[int]:
    by_layer = {}
    for row in rows:
        delta = row["oracle_metrics"]["logit_mse"] - row["baseline_metrics"]["logit_mse"]
        by_layer.setdefault(int(row["layer_idx"]), []).append(float(delta))

    averages = {layer: sum(vals) / len(vals) for layer, vals in by_layer.items()}

    def best_in_bucket(lo: int, hi: int) -> int:
        candidates = {layer: avg for layer, avg in averages.items() if lo <= layer <= hi}
        return max(candidates, key=candidates.get)

    selected = [0, best_in_bucket(1, 11), best_in_bucket(12, 23), best_in_bucket(24, 35)]
    seen = []
    for layer in selected:
        if layer not in seen:
            seen.append(layer)
    return seen


def choose_row_for_layer(rows: list[dict], layer_idx: int) -> dict:
    layer_rows = [row for row in rows if int(row["layer_idx"]) == layer_idx]
    return max(layer_rows, key=lambda row: float(row["oracle_metrics"]["logit_mse"] - row["baseline_metrics"]["logit_mse"]))


def gather_panel_data(
    args: argparse.Namespace,
    row: dict,
    model,
    tokenizer,
    spec,
    prompt_cache: dict[tuple[str, int], dict],
) -> dict:
    from kvpress.utils import extract_keys_and_values

    config = row.get("config") or row["task"]
    row_index = int(row.get("row_index", row.get("example_index")))
    key = (config, row_index)
    if key not in prompt_cache:
        example = fetch_example(spec, config=config, row_index=row_index, tokenizer=tokenizer)
        outputs, _, post_queries = run_prefill_and_capture(model, example.input_ids)
        prompt_cache[key] = {
            "example": example,
            "future_query_tensor": trim_queries(post_queries, args.n_sink).to(getattr(model, "dtype", torch.float16)),
            "cache": outputs.past_key_values,
        }

    example = prompt_cache[key]["example"]
    future_query_tensor = prompt_cache[key]["future_query_tensor"]
    cache = prompt_cache[key]["cache"]
    layer_idx = int(row["layer_idx"])
    keys, _ = extract_keys_and_values(cache, layer_idx)
    keys = keys[:, :, args.n_sink :, :]
    prefix_keys, future_queries, _ = split_prefix_and_future(
        keys,
        future_query_tensor[layer_idx : layer_idx + 1].to(keys.device),
        prefix_fraction=args.prefix_fraction,
        min_future_tokens=args.min_future_tokens,
    )
    metrics = compute_grouped_query_second_moment(future_queries.float(), prefix_keys.shape[1]).to(keys.device)
    bits = int(row["bits"])
    seed = args.seed + layer_idx * 1000
    compressor = Stage1MSECompressor(prefix_keys.shape[-1], bits, seed=seed, device=prefix_keys.device)
    baseline = instrument_compressor_path(prefix_keys, compressor)

    factors, _ = factorize_metric_batch(metrics.to(prefix_keys.device), args.eps)
    transformed = apply_headwise_linear(prefix_keys.float(), factors)
    oracle = instrument_compressor_path(transformed, compressor)
    transform_stats = transform_matrix_stats(factors, metrics)

    return {
        "config": example.config,
        "row_index": example.row_index,
        "layer_idx": layer_idx,
        "bits": bits,
        "k": prefix_keys.detach().cpu().reshape(-1, prefix_keys.shape[-1]).float(),
        "rotated_k": baseline["rotated_flat"].detach().cpu().float(),
        "Lk": transformed.detach().cpu().reshape(-1, transformed.shape[-1]).float(),
        "rotated_Lk": oracle["rotated_flat"].detach().cpu().float(),
        "transform_condition_number": transform_stats["transform_condition_number_mean"],
    }


def render_panel(panel_data: dict, output_path: Path) -> None:
    series = [
        ("k", panel_data["k"], "#4477AA"),
        ("rotated k", panel_data["rotated_k"], "#6699CC"),
        ("Lk", panel_data["Lk"], "#CC6677"),
        ("rotated Lk", panel_data["rotated_Lk"], "#228833"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), constrained_layout=True)
    for ax, (label, matrix, color) in zip(axes.reshape(-1), series):
        skew, excess_kurtosis = standardized_moments_from_matrix(matrix)
        cond = sample_condition_number(matrix)
        flat = matrix.reshape(-1).numpy()
        ax.hist(flat, bins=60, density=True, color=color, alpha=0.82)
        ax.set_title(label)
        ax.set_xlabel(
            f"skew={skew:.2f}\nexcess kurtosis={excess_kurtosis:.2f}\ncond={cond:.2f}"
        )
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        f"Layer {panel_data['layer_idx']} Distribution Comparison\n"
        f"config={panel_data['config']} row={panel_data['row_index']} bits={panel_data['bits']} "
        f"transform cond={panel_data['transform_condition_number']:.2f}"
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(path: Path, selected_rows: list[dict], selected_layers: list[int]) -> None:
    lines = [
        "# 2x2 Panel Chart Summary",
        "",
        "Selected layers:",
        "",
        f"- {', '.join(str(layer) for layer in selected_layers)}",
        "",
        "Selection rule:",
        "",
        "- layer 0 is included explicitly",
        "- one early representative layer with the highest average oracle-baseline logit-MSE delta",
        "- one middle representative layer with the highest average oracle-baseline logit-MSE delta",
        "- one late representative layer with the highest average oracle-baseline logit-MSE delta",
        "",
        "Chosen example per layer:",
        "",
    ]
    for row in selected_rows:
        config = row.get("config") or row.get("task")
        row_index = row.get("row_index", row.get("example_index"))
        lines.append(
            f"- layer {row['layer_idx']}: config `{config}`, row `{row_index}`, bits `{row['bits']}`"
        )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    diagnosis_dir = Path(args.diagnosis_dir)
    output_dir = ensure_dir(args.output_dir or (diagnosis_dir / "panel_charts"))
    payload = torch.load(diagnosis_dir / "diagnosis.pt", map_location="cpu")
    rows = payload["rows"]
    model, tokenizer = load_model_and_tokenizer(args.model, device_map=args.device_map, dtype_name=args.dtype)
    prompt_cache: dict[tuple[str, int], dict] = {}

    if args.layers == "auto":
        selected_layers = choose_representative_layers(rows)
    else:
        selected_layers = [int(part.strip()) for part in args.layers.split(",") if part.strip()]

    spec = get_dataset_spec(args.dataset)
    selected_rows = [choose_row_for_layer(rows, layer_idx) for layer_idx in selected_layers]
    for row in selected_rows:
        panel_data = gather_panel_data(args, row, model, tokenizer, spec, prompt_cache)
        render_panel(panel_data, output_dir / f"layer_{int(row['layer_idx']):02d}_panel.png")

    write_summary(output_dir / "summary.md", selected_rows, selected_layers)


if __name__ == "__main__":
    main()
