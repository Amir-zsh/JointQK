#!/usr/bin/env python3
"""One-off helper: capture prefill Q for 5 test prompts of 2wikimqa on Llama-3.1-8B.

Output filename pattern matches `longbench__2wikimqa__row<N>__test.pt` so the
analyze_q_distribution_shift.py script picks them up.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402, F401

import argparse
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kvq.capture.hooks import capture_rope_qk
from kvq.capture.model import get_model_device, load_model_and_tokenizer
from kvq.data.kvpress_adapter import build_kvpress_dataset_spec
from kvq.data.base import fetch_example


def ts() -> str: return time.strftime("%H:%M:%S")


def _assemble_per_layer(chunks, n_layers):
    if len(chunks) % n_layers != 0:
        raise RuntimeError(f"chunk count {len(chunks)} not multiple of n_layers {n_layers}")
    per_layer = [torch.cat(chunks[L::n_layers], dim=2) for L in range(n_layers)]
    return torch.stack(per_layer, dim=0).squeeze(1)


@torch.inference_mode()
def capture_one(model, tokenizer, prompt_text: str, max_tokens: int = 16000):
    device = get_model_device(model)
    n_layers = int(model.config.num_hidden_layers)
    enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_tokens)
    input_ids = enc["input_ids"].to(device)
    T = int(input_ids.shape[-1])
    with capture_rope_qk(model) as (_q_pre, q_post_chunks, _k_pre, _k_post):
        outputs = model(input_ids=input_ids, use_cache=True)
    q_post = _assemble_per_layer(q_post_chunks, n_layers)  # (n_layers, n_q, T, d)
    k_post = []
    for L in range(n_layers):
        k_post.append(outputs.past_key_values.layers[L].keys.detach().to("cpu", dtype=torch.float16).squeeze(0))
    k_post = torch.stack(k_post, dim=0)
    return q_post, k_post, T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--task", default="2wikimqa")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--out-dir", default=str(REPO / "artifacts/calibration/longbench_extras_qkv_llama31_8b/01_raw/shard_000"))
    p.add_argument("--max-tokens", type=int, default=16000)
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{ts()}] loading {args.model}...", flush=True)
    model, tokenizer = load_model_and_tokenizer(args.model, dtype_name="float16")
    print(f"[{ts()}] model on {get_model_device(model)}", flush=True)

    from datasets import load_dataset
    from kvq.benchmarks.evaluate_registry import DATASET_REGISTRY
    ds = load_dataset(DATASET_REGISTRY["longbench"], args.task, split="test")
    print(f"[{ts()}] {args.task}: {len(ds)} rows", flush=True)

    for i in range(args.n):
        row_idx = i  # first n test rows
        spec = build_kvpress_dataset_spec(name="longbench", config_names=(args.task,),
                                           metadata_fields=("task",))
        try:
            ex = fetch_example(spec, args.task, row_idx, tokenizer)
        except Exception as e:
            print(f"[{ts()}]   {i}: fetch failed: {e}", flush=True)
            continue
        t0 = time.time()
        q, k, T = capture_one(model, tokenizer, ex.prompt_text, args.max_tokens)
        out_path = out_dir / f"longbench__{args.task}__row{row_idx:05d}__test.pt"
        torch.save({
            "q_post": q.to(torch.float16),
            "k_post": k,
            "v": torch.empty(0),  # not needed for our Q analysis
            "prompt_length": T, "total_length": T, "captured_length": T,
            "config": args.task, "task": args.task, "split": "test",
            "row_index": row_idx,
            "dataset": "longbench",
        }, out_path)
        print(f"[{ts()}]   {i}/{args.n} row{row_idx} T={T} | {time.time()-t0:.1f}s | {out_path.name}", flush=True)


if __name__ == "__main__":
    main()
