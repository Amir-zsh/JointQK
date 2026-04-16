from __future__ import annotations

import argparse
from pathlib import Path

import torch

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.stage1.common import (
    QueryMomentsAccumulator,
    build_prompt,
    compute_query_moments,
    ensure_dir,
    load_longbench_slice,
    load_model_and_tokenizer,
    parse_task_names,
    run_prefill_and_capture,
    sample_queries,
    save_json,
    trim_queries,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect query statistics for stage 1.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset", default="longbench-e")
    parser.add_argument("--task_names", default="qasper,hotpotqa,passage_retrieval_en")
    parser.add_argument("--num_examples", type=int, default=8)
    parser.add_argument("--max_context_length", type=int, default=4096)
    parser.add_argument("--n_sink", type=int, default=4)
    parser.add_argument("--query_sample_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--output_dir", default="artifacts/stage1/query_stats")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    model, tokenizer = load_model_and_tokenizer(args.model, device_map=args.device_map, dtype_name=args.dtype)
    task_names = parse_task_names(args.task_names)
    records = load_longbench_slice(task_names, args.num_examples, seed=args.seed, dataset_name=args.dataset)

    pre_accumulator = QueryMomentsAccumulator()
    post_accumulator = QueryMomentsAccumulator()
    prompt_pre_m2 = []
    prompt_post_m2 = []
    sampled_pre = []
    sampled_post = []
    example_metadata = []

    for example_offset, record in enumerate(records):
        prompt = build_prompt(record)
        outputs, pre_queries, post_queries, inputs = run_prefill_and_capture(
            model,
            tokenizer,
            prompt,
            max_context_length=args.max_context_length,
        )
        del outputs
        pre_trimmed = trim_queries(pre_queries, args.n_sink)
        post_trimmed = trim_queries(post_queries, args.n_sink)

        pre_accumulator.update(pre_trimmed)
        post_accumulator.update(post_trimmed)

        _, _, pre_second = compute_query_moments(pre_trimmed)
        _, _, post_second = compute_query_moments(post_trimmed)
        prompt_pre_m2.append(pre_second)
        prompt_post_m2.append(post_second)

        sampled_pre.append(sample_queries(pre_trimmed, args.query_sample_size, args.seed + example_offset))
        sampled_post.append(sample_queries(post_trimmed, args.query_sample_size, args.seed + 1000 + example_offset))

        example_metadata.append(
            {
                "task": record.task,
                "example_index": record.example_index,
                "token_count": int(inputs["input_ids"].shape[-1]),
                "captured_layers": int(pre_queries.shape[0]),
                "captured_heads": int(pre_queries.shape[1]),
                "captured_seq_len": int(pre_queries.shape[2]),
            }
        )

    pre_mean, pre_cov, pre_second = pre_accumulator.finalize()
    post_mean, post_cov, post_second = post_accumulator.finalize()

    artifact = {
        "config": vars(args),
        "examples": example_metadata,
        "pre_mean": pre_mean,
        "pre_cov": pre_cov,
        "pre_second_moment": pre_second,
        "post_mean": post_mean,
        "post_cov": post_cov,
        "post_second_moment": post_second,
        "prompt_pre_second_moment": torch.stack(prompt_pre_m2, dim=0),
        "prompt_post_second_moment": torch.stack(prompt_post_m2, dim=0),
        "sampled_pre_queries": sampled_pre,
        "sampled_post_queries": sampled_post,
    }
    torch.save(artifact, output_dir / "query_stats.pt")

    summary = {
        "num_examples": len(records),
        "num_layers": int(pre_mean.shape[0]),
        "num_heads": int(pre_mean.shape[1]),
        "head_dim": int(pre_mean.shape[2]),
        "pre_total_tokens": int(pre_accumulator.total_count),
        "post_total_tokens": int(post_accumulator.total_count),
        "tasks": task_names,
    }
    save_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Query Stats Collection",
                "",
                f"- Model: `{args.model}`",
                f"- Dataset: `{args.dataset}`",
                f"- Tasks: `{', '.join(task_names)}`",
                f"- Examples: `{len(records)}`",
                f"- Layers: `{pre_mean.shape[0]}`",
                f"- Heads: `{pre_mean.shape[1]}`",
                f"- Head dim: `{pre_mean.shape[2]}`",
                f"- Tokens used for pre-RoPE stats: `{pre_accumulator.total_count}`",
                f"- Tokens used for post-RoPE stats: `{post_accumulator.total_count}`",
            ]
        )
    )


if __name__ == "__main__":
    main()
