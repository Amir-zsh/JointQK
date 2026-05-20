#!/usr/bin/env python3
"""Collect prefill-only post-RoPE Q/K tensors for K-basis calibration.

This writes the same manifest/examples layout consumed by
run_phase1d_k_basis_stability.py, but stores only:
    q_post, k_post, prompt_length, metadata, prompt text

It intentionally does not run decode generation and does not store q_pre or V.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.data import get_dataset_spec, load_and_filter, register_all_benchmark_specs
from experiments.toolkit import (
    ensure_dir,
    load_model_and_tokenizer,
    run_prefill_qk_post_capture,
    save_json,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{sec:02d}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect prefill-only Q/K calibration tensors.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset", default="longbench-e")
    parser.add_argument("--configs", default="qasper,hotpotqa,passage_retrieval_en")
    parser.add_argument("--num-examples-per-config", type=int, default=25)
    parser.add_argument("--min-tokens", type=int, default=4000)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--output-dir", default="artifacts/query_stats_longbench_under4k_25per_qk")
    parser.add_argument("--dry-run", action="store_true", help="Only select/tokenize examples; do not load model.")
    parser.add_argument("--resume", action="store_true", help="Reuse completed examples listed in manifest.partial.json.")
    return parser.parse_args()


def write_summary(output_dir: Path, args: argparse.Namespace, manifest_examples: list[dict], length_stats: list[int]) -> None:
    by_config: dict[str, int] = defaultdict(int)
    for ex in manifest_examples:
        by_config[str(ex["config"])] += 1
    min_len = min(length_stats) if length_stats else 0
    max_len = max(length_stats) if length_stats else 0
    mean_len = sum(length_stats) / len(length_stats) if length_stats else 0.0
    lines = [
        "# Prefill Q/K Stats Collection",
        "",
        f"- Model: `{args.model}`",
        f"- Dataset: `{args.dataset}`",
        f"- Configs requested: `{args.configs}`",
        f"- Examples collected: `{len(manifest_examples)}`",
        f"- Examples by config: `{dict(sorted(by_config.items()))}`",
        f"- Length filter: `[{args.min_tokens}, {args.max_tokens}]` tokens",
        f"- Prompt length stats: min=`{min_len}`, max=`{max_len}`, mean=`{mean_len:.1f}`",
        "- Stored tensors: `q_post`, `k_post` only, prefill positions only",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    examples_dir = ensure_dir(output_dir / "examples")
    partial_manifest_path = output_dir / "manifest.partial.json"

    register_all_benchmark_specs()
    from transformers import AutoTokenizer

    log(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
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
            f"No examples passed length filter [{args.min_tokens}, {args.max_tokens}] "
            f"for dataset {args.dataset} with configs {configs}."
        )
    selected_counts: dict[str, int] = defaultdict(int)
    selected_lengths: dict[str, list[int]] = defaultdict(list)
    for ex in selected:
        selected_counts[ex.config] += 1
        selected_lengths[ex.config].append(ex.prompt_length)
    for cfg in sorted(selected_counts):
        lengths = selected_lengths[cfg]
        log(
            f"selected {cfg}: n={selected_counts[cfg]} "
            f"tokens min={min(lengths)} max={max(lengths)} mean={sum(lengths) / len(lengths):.1f}"
        )
    if args.dry_run:
        return

    manifest_examples: list[dict] = []
    if args.resume and partial_manifest_path.exists():
        partial = json.loads(partial_manifest_path.read_text())
        manifest_examples = list(partial.get("examples", []))
        log(f"Resume enabled: loaded {len(manifest_examples)} partial manifest entries")
    completed_files = {entry["file"] for entry in manifest_examples}

    log(f"Loading model: {args.model} device_map={args.device_map} dtype={args.dtype}")
    model, _ = load_model_and_tokenizer(args.model, device_map=args.device_map, dtype_name=args.dtype)

    t_all = time.time()
    for example_idx, example in enumerate(selected):
        filename = f"ex_{example_idx:03d}.pt"
        rel_file = f"examples/{filename}"
        out_path = examples_dir / filename
        if args.resume and rel_file in completed_files and out_path.exists():
            log(f"skip {example_idx + 1:03d}/{len(selected)} {rel_file} already collected")
            continue

        t0 = time.time()
        log(
            f"capture {example_idx + 1:03d}/{len(selected)} cfg={example.config} "
            f"row={example.row_index} prompt_tokens={example.prompt_length}"
        )
        captured = run_prefill_qk_post_capture(model=model, input_ids=example.input_ids)
        artifact = {
            "q_post": captured["q_post"],
            "k_post": captured["k_post"],
            "prompt_length": int(captured["prompt_length"]),
            "total_length": int(captured["total_length"]),
            "captured_length": int(captured["captured_length"]),
            "generated_token_ids": captured["generated_token_ids"],
            "generated_text": "",
            "dataset": example.dataset,
            "config": example.config,
            "row_index": example.row_index,
            "metadata": example.metadata,
            "messages": example.messages,
            "prompt_text": example.prompt_text,
            "collection": "prefill_qk_post_only",
        }
        torch.save(artifact, out_path)
        manifest_examples.append(
            {
                "file": rel_file,
                "dataset": example.dataset,
                "config": example.config,
                "row_index": example.row_index,
                "prompt_length": int(captured["prompt_length"]),
                "total_length": int(captured["total_length"]),
                "captured_length": int(captured["captured_length"]),
                "n_generated": 0,
                "max_new_tokens_used": 0,
                "generated_text": "",
                "metadata": example.metadata,
            }
        )
        manifest = {
            "config": vars(args),
            "dataset": args.dataset,
            "configs": configs,
            "num_examples": len(manifest_examples),
            "examples": manifest_examples,
        }
        save_json(partial_manifest_path, manifest)
        log(f"wrote {rel_file} in {fmt_duration(time.time() - t0)}")

        del captured, artifact
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    length_stats = [int(ex["prompt_length"]) for ex in manifest_examples]
    manifest = {
        "config": vars(args),
        "dataset": args.dataset,
        "configs": configs,
        "num_examples": len(manifest_examples),
        "examples": manifest_examples,
    }
    save_json(output_dir / "manifest.json", manifest)
    write_summary(output_dir, args, manifest_examples, length_stats)
    log(f"Complete: collected {len(manifest_examples)} examples in {fmt_duration(time.time() - t_all)}")
    log(f"Wrote manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
