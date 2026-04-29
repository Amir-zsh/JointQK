from __future__ import annotations

import argparse
from pathlib import Path
from statistics import median
from typing import Any

import torch

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kvpress"))
else:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kvpress"))

from experiments.stage1.data import get_dataset_spec, load_and_filter, register_all_benchmark_specs
from experiments.stage1.diagnosis import run_transform_variant_path
from experiments.stage1.toolkit import (
    compute_grouped_query_second_moment,
    ensure_dir,
    load_model_and_tokenizer,
    run_prefill_and_capture,
    save_json,
    split_prefix_and_future,
    trim_queries,
)


CONTROL_VARIANTS = [
    "baseline_raw",
    "basis_only",
    "full_metric",
    "trace_matched_full_metric",
]
DEFAULT_GAMMAS = [0.125, 0.25, 0.5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Stage-1E narrow partial-spectrum oracle study.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset", default="longbench-e")
    parser.add_argument("--configs", default="qasper,hotpotqa,passage_retrieval_en")
    parser.add_argument("--num_examples_per_config", type=int, default=2)
    parser.add_argument("--min_tokens", type=int, default=4000)
    parser.add_argument("--max_tokens", type=int, default=12000)
    parser.add_argument("--n_sink", type=int, default=4)
    parser.add_argument("--key_bits", default="2,3,4")
    parser.add_argument("--gammas", default="0.125,0.25,0.5")
    parser.add_argument("--prefix_fraction", type=float, default=0.75)
    parser.add_argument("--min_future_tokens", type=int, default=32)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--output_dir", default="artifacts/stage1/oracle_partial_spectrum_study")
    return parser.parse_args()


def parse_number_list(raw: str, cast: type = float) -> list:
    return [cast(part.strip()) for part in raw.split(",") if part.strip()]


def row_key(row: dict[str, Any]) -> tuple:
    return (
        row["config"],
        int(row["row_index"]),
        int(row["layer_idx"]),
        int(row["bits"]),
    )


def variant_key(row: dict[str, Any]) -> str:
    if row["variant"] == "gamma_sweep":
        return f"gamma_{float(row['gamma']):g}"
    return str(row["variant"])


def variant_label(key: str) -> str:
    labels = {
        "baseline_raw": "Baseline",
        "basis_only": "Basis Only",
        "full_metric": "Full Metric",
        "trace_matched_full_metric": "Trace-Matched Full Metric",
    }
    if key.startswith("gamma_"):
        return f"Gamma {key.removeprefix('gamma_')}"
    return labels.get(key, key)


def filter_rows(rows: list[dict[str, Any]], *, exclude_layer0: bool = False) -> list[dict[str, Any]]:
    if not exclude_layer0:
        return rows
    return [row for row in rows if int(row["layer_idx"]) != 0]


def compute_variant_stats(rows: list[dict[str, Any]], *, exclude_layer0: bool = False) -> dict[str, dict[str, dict[str, float | int]]]:
    filtered = filter_rows(rows, exclude_layer0=exclude_layer0)
    baseline_by_key = {row_key(row): row for row in filtered if row["variant"] == "baseline_raw"}
    bitwidths = sorted({int(row["bits"]) for row in filtered})
    variant_keys = sorted({variant_key(row) for row in filtered}, key=_variant_sort_key)
    summary: dict[str, dict[str, dict[str, float | int]]] = {}

    for bits in bitwidths:
        bit_rows = [row for row in filtered if int(row["bits"]) == bits]
        summary[str(bits)] = {}
        for key in variant_keys:
            variant_rows = [row for row in bit_rows if variant_key(row) == key]
            if not variant_rows:
                continue
            geometry = [float(row["metrics"]["geometry_distortion"]) for row in variant_rows]
            top1 = [float(row["metrics"]["top1_match"]) for row in variant_rows]
            if key == "baseline_raw":
                delta_geometry = [0.0 for _ in variant_rows]
                delta_top1 = [0.0 for _ in variant_rows]
                geometry_wins = [0.0 for _ in variant_rows]
                top1_wins = [0.0 for _ in variant_rows]
            else:
                delta_geometry = []
                delta_top1 = []
                geometry_wins = []
                top1_wins = []
                for row in variant_rows:
                    baseline = baseline_by_key[row_key(row)]
                    geometry_delta = float(row["metrics"]["geometry_distortion"] - baseline["metrics"]["geometry_distortion"])
                    top1_delta = float(row["metrics"]["top1_match"] - baseline["metrics"]["top1_match"])
                    delta_geometry.append(geometry_delta)
                    delta_top1.append(top1_delta)
                    geometry_wins.append(1.0 if geometry_delta < 0 else 0.0)
                    top1_wins.append(1.0 if top1_delta > 0 else 0.0)
            both_wins = [1.0 if geometry_wins[idx] and top1_wins[idx] else 0.0 for idx in range(len(variant_rows))]
            summary[str(bits)][key] = {
                "n": len(variant_rows),
                "mean_geometry_distortion": _mean(geometry),
                "mean_top1_match": _mean(top1),
                "median_delta_geometry_distortion": float(median(delta_geometry)),
                "median_delta_top1_match": float(median(delta_top1)),
                "geometry_win_rate": _mean(geometry_wins),
                "top1_win_rate": _mean(top1_wins),
                "both_win_rate": _mean(both_wins),
            }
    return summary


def assess_gamma_025_success(layer0_summary: dict[str, dict[str, dict[str, float | int]]]) -> str:
    wins = 0
    neutral_or_better = 0
    for bits, bit_summary in layer0_summary.items():
        baseline = bit_summary.get("baseline_raw")
        gamma = bit_summary.get("gamma_0.25")
        if baseline is None or gamma is None:
            continue
        geom_better = gamma["mean_geometry_distortion"] < baseline["mean_geometry_distortion"]
        top1_better = gamma["mean_top1_match"] > baseline["mean_top1_match"]
        top1_neutral = gamma["mean_top1_match"] >= baseline["mean_top1_match"] - 1e-6
        if geom_better and top1_better:
            wins += 1
        if geom_better and top1_neutral:
            neutral_or_better += 1

    gamma3 = layer0_summary.get("3", {}).get("gamma_0.25")
    baseline3 = layer0_summary.get("3", {}).get("baseline_raw")
    repeats_3bit = bool(
        gamma3
        and baseline3
        and gamma3["mean_geometry_distortion"] < baseline3["mean_geometry_distortion"]
        and gamma3["mean_top1_match"] > baseline3["mean_top1_match"]
    )
    if wins >= 2:
        return "strong_success"
    if repeats_3bit and neutral_or_better >= 2:
        return "moderate_success"
    return "failure"


def build_summary_md(
    output_path: Path,
    full_summary: dict[str, dict[str, dict[str, float | int]]],
    layer0_summary: dict[str, dict[str, dict[str, float | int]]],
    decision: str,
) -> None:
    lines = [
        "# Stage 1E: Oracle Partial-Spectrum Study",
        "",
        "## Headline",
        "",
        "- Primary question: does the robust `gamma=0.25` 3-bit Stage 1D result generalize to 2-bit and 4-bit?",
        "- `geometry_distortion` is the primary quadratic metric; top-1 match is the ranking-sensitive companion metric.",
        f"- Decision: `{decision}`.",
        "",
    ]
    lines.extend(_summary_table("Full Summary", full_summary))
    lines.extend(_summary_table("Layer-0-Excluded Summary", layer0_summary))
    lines.extend(["## Gamma 0.25 Success Check", ""])
    for bits in sorted(layer0_summary.keys(), key=int):
        bit_summary = layer0_summary[bits]
        baseline = bit_summary.get("baseline_raw")
        gamma = bit_summary.get("gamma_0.25")
        if not baseline or not gamma:
            continue
        geom_delta = gamma["mean_geometry_distortion"] - baseline["mean_geometry_distortion"]
        top1_delta = gamma["mean_top1_match"] - baseline["mean_top1_match"]
        lines.append(
            f"- {bits}-bit: mean geometry delta `{geom_delta:.4f}`, "
            f"mean top-1 delta `{top1_delta:.4f}`, both-win rate `{gamma['both_win_rate']:.3f}`"
        )
    output_path.write_text("\n".join(lines))


def _summary_table(title: str, summary: dict[str, dict[str, dict[str, float | int]]]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Bits | Variant | Mean Geometry | Mean Top-1 | Median Delta Geometry | Median Delta Top-1 | Geometry Win | Top-1 Win | Both Win |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for bits in sorted(summary.keys(), key=int):
        for key in sorted(summary[bits].keys(), key=_variant_sort_key):
            block = summary[bits][key]
            lines.append(
                f"| {bits} | {variant_label(key)} | "
                f"{block['mean_geometry_distortion']:.4f} | "
                f"{block['mean_top1_match']:.4f} | "
                f"{block['median_delta_geometry_distortion']:.4f} | "
                f"{block['median_delta_top1_match']:.4f} | "
                f"{block['geometry_win_rate']:.3f} | "
                f"{block['top1_win_rate']:.3f} | "
                f"{block['both_win_rate']:.3f} |"
            )
    lines.extend([""])
    return lines


def _variant_sort_key(key: str) -> tuple[int, float | str]:
    order = {
        "baseline_raw": 0,
        "basis_only": 1,
        "full_metric": 2,
        "trace_matched_full_metric": 3,
    }
    if key.startswith("gamma_"):
        return (4, float(key.removeprefix("gamma_")))
    return (order.get(key, 99), key)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def main() -> None:
    from kvpress.utils import extract_keys_and_values

    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    bitwidths = parse_number_list(args.key_bits, int)
    gammas = parse_number_list(args.gammas, float) or DEFAULT_GAMMAS
    configs = [config.strip() for config in args.configs.split(",") if config.strip()] or None

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

    rows: list[dict[str, Any]] = []

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
                for variant in CONTROL_VARIANTS:
                    result = run_transform_variant_path(
                        prefix_keys,
                        future_queries,
                        metrics,
                        bits=bits,
                        seed=seed,
                        eps=args.eps,
                        variant=variant,
                    )
                    rows.append(_build_row(example, example_offset, layer_idx, split_at, bits, variant, None, result))

                for gamma in gammas:
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
                    rows.append(_build_row(example, example_offset, layer_idx, split_at, bits, "gamma_sweep", gamma, result))

    full_summary = compute_variant_stats(rows)
    layer0_summary = compute_variant_stats(rows, exclude_layer0=True)
    decision = assess_gamma_025_success(layer0_summary)

    payload = {
        "config": {**vars(args), "parsed_key_bits": bitwidths, "parsed_gammas": gammas},
        "summary": full_summary,
        "layer0_excluded_summary": layer0_summary,
        "decision": decision,
        "rows": rows,
    }
    torch.save(payload, output_dir / "variant_study.pt")
    save_json(output_dir / "metrics.json", payload)
    build_summary_md(output_dir / "summary.md", full_summary, layer0_summary, decision)


def _build_row(
    example: Any,
    example_offset: int,
    layer_idx: int,
    split_at: int,
    bits: int,
    variant: str,
    gamma: float | None,
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset": example.dataset,
        "config": example.config,
        "row_index": int(example.row_index),
        "example_offset": int(example_offset),
        "layer_idx": int(layer_idx),
        "split_at": int(split_at),
        "bits": int(bits),
        "variant": variant,
        "gamma": None if gamma is None else float(gamma),
        "metrics": result["metrics"],
        "diagnostics": result["diagnostics"],
    }


if __name__ == "__main__":
    main()
