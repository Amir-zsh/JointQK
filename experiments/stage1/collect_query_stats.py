from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.stage1.toolkit import (
    QueryMomentsAccumulator,
    ensure_dir,
    load_model_and_tokenizer,
    run_generation_and_capture,
    save_json,
)
from experiments.stage1.data import get_dataset_spec, load_and_filter, register_all_benchmark_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Q/K/V statistics during generation for stage 1.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset", default="longbench-e")
    parser.add_argument(
        "--configs",
        default="qasper,hotpotqa,passage_retrieval_en",
        help="Comma-separated dataset configs (dataset spec's alias is applied).",
    )
    parser.add_argument("--num_examples_per_config", type=int, default=8)
    parser.add_argument("--min_tokens", type=int, default=4000)
    parser.add_argument("--max_tokens", type=int, default=12000)
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help=(
            "Override the per-task generation budget. If None (default), use the "
            "dataset row's `max_new_tokens` column (kvpress convention); fall back to 256 "
            "if the row doesn't carry the column."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--output_dir", default="artifacts/stage1/query_stats_v2")
    return parser.parse_args()


def _update_prefill_moments(
    accumulator: QueryMomentsAccumulator,
    tensor: torch.Tensor,
    prompt_length: int,
) -> None:
    accumulator.update(tensor[:, :, :prompt_length, :])


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    examples_dir = ensure_dir(Path(args.output_dir) / "examples")

    register_all_benchmark_specs()
    model, tokenizer = load_model_and_tokenizer(
        args.model, device_map=args.device_map, dtype_name=args.dtype
    )
    spec = get_dataset_spec(args.dataset)
    configs = [c.strip() for c in args.configs.split(",") if c.strip()] or None

    selected = load_and_filter(
        spec,
        tokenizer=tokenizer,
        num_examples_per_config=args.num_examples_per_config,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        seed=args.seed,
        config_names=configs,
    )

    if not selected:
        raise RuntimeError(
            f"No examples passed the length filter [{args.min_tokens}, {args.max_tokens}] tokens "
            f"for dataset '{args.dataset}' with configs {configs}."
        )

    pooled_q_pre = QueryMomentsAccumulator()
    pooled_q_post = QueryMomentsAccumulator()
    pooled_k_post = QueryMomentsAccumulator()
    pooled_v = QueryMomentsAccumulator()

    manifest_examples: list[dict] = []
    length_stats: list[int] = []

    for example_idx, example in enumerate(selected):
        row_budget = example.metadata.get("max_new_tokens")
        max_new_tokens = int(args.max_new_tokens if args.max_new_tokens is not None else (row_budget or 256))
        captured = run_generation_and_capture(
            model=model,
            tokenizer=tokenizer,
            input_ids=example.input_ids,
            max_new_tokens=max_new_tokens,
        )

        prompt_length = int(captured["prompt_length"])
        _update_prefill_moments(pooled_q_pre, captured["q_pre"], prompt_length)
        _update_prefill_moments(pooled_q_post, captured["q_post"], prompt_length)
        _update_prefill_moments(pooled_k_post, captured["k_post"], prompt_length)
        _update_prefill_moments(pooled_v, captured["v"], prompt_length)

        generated_text = tokenizer.decode(
            captured["generated_token_ids"].tolist(), skip_special_tokens=True
        )

        artifact = {
            "q_pre": captured["q_pre"],
            "q_post": captured["q_post"],
            "k_post": captured["k_post"],
            "v": captured["v"],
            "prompt_length": prompt_length,
            "total_length": int(captured["total_length"]),
            "captured_length": int(captured["captured_length"]),
            "generated_token_ids": captured["generated_token_ids"],
            "generated_text": generated_text,
            "dataset": example.dataset,
            "config": example.config,
            "row_index": example.row_index,
            "metadata": example.metadata,
            "messages": example.messages,
            "prompt_text": example.prompt_text,
        }

        filename = f"ex_{example_idx:03d}.pt"
        torch.save(artifact, examples_dir / filename)

        manifest_examples.append(
            {
                "file": f"examples/{filename}",
                "dataset": example.dataset,
                "config": example.config,
                "row_index": example.row_index,
                "prompt_length": prompt_length,
                "total_length": int(captured["total_length"]),
                "captured_length": int(captured["captured_length"]),
                "n_generated": int(captured["total_length"]) - prompt_length,
                "max_new_tokens_used": max_new_tokens,
                "generated_text": generated_text,
                "metadata": example.metadata,
            }
        )
        length_stats.append(prompt_length)

        del captured, artifact
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pooled_artifact = {
        "q_pre": pooled_q_pre.finalize(),
        "q_post": pooled_q_post.finalize(),
        "k_post": pooled_k_post.finalize(),
        "v": pooled_v.finalize(),
        "pooled_prefill_tokens": pooled_q_post.total_count,
    }
    torch.save(pooled_artifact, output_dir / "pooled_stats.pt")

    manifest = {
        "config": vars(args),
        "dataset": args.dataset,
        "configs": configs,
        "num_examples": len(manifest_examples),
        "examples": manifest_examples,
    }
    save_json(output_dir / "manifest.json", manifest)

    q_post_mean, _, _ = pooled_artifact["q_post"]
    n_layers = int(q_post_mean.shape[0])
    n_q_heads = int(q_post_mean.shape[1])
    head_dim = int(q_post_mean.shape[2])
    k_post_mean, _, _ = pooled_artifact["k_post"]
    n_kv_heads = int(k_post_mean.shape[1])

    min_len = min(length_stats) if length_stats else 0
    max_len = max(length_stats) if length_stats else 0
    mean_len = sum(length_stats) / len(length_stats) if length_stats else 0.0

    (output_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Query/Key/Value Stats Collection (v2)",
                "",
                f"- Model: `{args.model}`",
                f"- Dataset: `{args.dataset}`",
                f"- Configs: `{', '.join(configs or spec.config_names)}`",
                f"- Examples collected: `{len(manifest_examples)}`",
                f"- Length filter: `[{args.min_tokens}, {args.max_tokens}]` tokens",
                f"- Max new tokens: `{args.max_new_tokens}`",
                f"- Prompt length stats: min=`{min_len}`, max=`{max_len}`, mean=`{mean_len:.1f}`",
                f"- Layers: `{n_layers}`",
                f"- Query heads: `{n_q_heads}`",
                f"- KV heads: `{n_kv_heads}`",
                f"- Head dim: `{head_dim}`",
                f"- Prefill tokens pooled: `{pooled_q_post.total_count}`",
            ]
        )
    )


if __name__ == "__main__":
    main()
