from __future__ import annotations

import argparse
from pathlib import Path

import torch

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kvpress"))
else:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kvpress"))

from experiments.stage1.common import (
    Stage1MSECompressor,
    compute_attention_metrics,
    compute_geometry_distortion,
    compute_grouped_query_second_moment,
    ensure_dir,
    geometry_aware_roundtrip,
    load_model_and_tokenizer,
    run_prefill_and_capture,
    save_json,
    split_prefix_and_future,
    summarize_metrics,
    trim_queries,
    write_markdown_table,
)
from experiments.stage1.data import get_dataset_spec, load_and_filter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the stage-1 oracle V3 study.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset", default="longbench-e")
    parser.add_argument("--configs", default="qasper,hotpotqa,passage_retrieval_en")
    parser.add_argument("--num_examples_per_config", type=int, default=2)
    parser.add_argument("--min_tokens", type=int, default=4000)
    parser.add_argument("--max_tokens", type=int, default=12000)
    parser.add_argument("--n_sink", type=int, default=4)
    parser.add_argument("--key_bits", default="2,3,4")
    parser.add_argument("--geometry_mode", choices=["none", "oracle", "both"], default="both")
    parser.add_argument("--prefix_fraction", type=float, default=0.75)
    parser.add_argument("--min_future_tokens", type=int, default=32)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--output_dir", default="artifacts/stage1/oracle_v3_study")
    return parser.parse_args()


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

    per_bit_baseline: dict[int, list[dict[str, float]]] = {bits: [] for bits in bitwidths}
    per_bit_oracle: dict[int, list[dict[str, float]]] = {bits: [] for bits in bitwidths}
    per_example = []

    for example_offset, example in enumerate(examples):
        outputs, _, post_queries = run_prefill_and_capture(model, example.input_ids)
        future_query_tensor = trim_queries(post_queries, args.n_sink).to(getattr(model, "dtype", torch.float16))
        cache = outputs.past_key_values

        example_row = {
            "dataset": example.dataset,
            "config": example.config,
            "row_index": example.row_index,
            "layers": [],
        }

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

            layer_metrics = {
                "layer_idx": layer_idx,
                "split_at": split_at,
            }

            for bits in bitwidths:
                seed = args.seed + layer_idx * 1000
                baseline_compressor = Stage1MSECompressor(prefix_keys.shape[-1], bits, seed=seed, device=prefix_keys.device)
                baseline_recon = baseline_compressor.roundtrip(prefix_keys)
                baseline_metrics = compute_attention_metrics(future_queries, prefix_keys, baseline_recon)
                baseline_metrics["key_mse"] = (baseline_recon.float() - prefix_keys.float()).pow(2).mean().item()
                baseline_metrics["geometry_distortion"] = compute_geometry_distortion(baseline_recon, prefix_keys, metrics)
                per_bit_baseline[bits].append(baseline_metrics)
                layer_metrics[f"baseline_{bits}bit"] = baseline_metrics

                if args.geometry_mode in {"oracle", "both"}:
                    oracle_recon = geometry_aware_roundtrip(prefix_keys, metrics, bits=bits, seed=seed, eps=args.eps)
                    oracle_metrics = compute_attention_metrics(future_queries, prefix_keys, oracle_recon)
                    oracle_metrics["key_mse"] = (oracle_recon.float() - prefix_keys.float()).pow(2).mean().item()
                    oracle_metrics["geometry_distortion"] = compute_geometry_distortion(oracle_recon, prefix_keys, metrics)
                    per_bit_oracle[bits].append(oracle_metrics)
                    layer_metrics[f"oracle_{bits}bit"] = oracle_metrics

            example_row["layers"].append(layer_metrics)

        per_example.append(example_row)

    summary = {}
    for bits in bitwidths:
        summary[f"baseline_{bits}bit"] = summarize_metrics(per_bit_baseline[bits])
        if args.geometry_mode in {"oracle", "both"}:
            summary[f"oracle_{bits}bit"] = summarize_metrics(per_bit_oracle[bits])

    torch.save(
        {
            "config": vars(args),
            "summary": summary,
            "per_example": per_example,
        },
        output_dir / "oracle_study.pt",
    )
    save_json(output_dir / "metrics.json", summary)
    write_markdown_table(output_dir / "summary.md", "Oracle V3 Study", summary)


if __name__ == "__main__":
    main()
