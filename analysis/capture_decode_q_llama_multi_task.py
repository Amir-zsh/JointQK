#!/usr/bin/env python3
"""Capture decode-time Q distributions across 8 tasks on Llama-3.1-8B.

For each task: run model.generate (greedy, full-precision K/V — no compression press)
on N test prompts, capturing per-step Q via the apply_rotary_pos_emb hook from
`kvq.capture.hooks.capture_rope_qk`.

Output: per-task per-prompt .pt file with key 'decode_q' shape (n_layers, n_q_heads, n_steps, d)
saved under artifacts/decode_q_captures_llama/<task>/row<idx>.pt
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402, F401

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kvq.capture.hooks import capture_rope_qk
from kvq.capture.model import get_model_device, load_model_and_tokenizer
from kvq.data.kvpress_adapter import build_kvpress_dataset_spec
from kvq.data.base import fetch_example


def ts() -> str:
    return time.strftime("%H:%M:%S")


@torch.inference_mode()
def capture_decode_q(model, tokenizer, prompt_text: str,
                      max_new_tokens: int, max_prompt_tokens: int = 12000) -> dict[str, Any]:
    device = get_model_device(model)
    n_layers = int(model.config.num_hidden_layers)
    enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_tokens)
    input_ids = enc["input_ids"].to(device)
    T = int(input_ids.shape[-1])

    with capture_rope_qk(model) as (_q_pre, q_post_chunks, _k_pre, _k_post):
        outputs = model(input_ids=input_ids, use_cache=True)
        # First decode token from prefill last logit
        past_kv = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        decoded = [int(next_token.item())]
        eos_ids = getattr(model.generation_config, "eos_token_id", None)
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        elif eos_ids is None:
            eos_ids = []
        for _ in range(max_new_tokens - 1):
            out = model(input_ids=next_token, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tok = int(next_token.item())
            decoded.append(tok)
            if tok in eos_ids:
                break
        all_q_chunks = list(q_post_chunks)

    # First n_layers chunks = prefill q (shape (1, n_q, T, d)); subsequent = decode (shape (1, n_q, 1, d))
    n_decode_calls = len(all_q_chunks) // n_layers - 1
    if n_decode_calls < 1:
        return {"decode_q": torch.empty(n_layers, 0, 0, 0), "n_steps": 0,
                "prompt_length": T, "decoded_token_ids": decoded}
    # Extract per-layer decode chunks
    decode_per_layer: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    idx = n_layers
    for step in range(n_decode_calls):
        for L in range(n_layers):
            decode_per_layer[L].append(all_q_chunks[idx])
            idx += 1
    decode_q = []
    for L in range(n_layers):
        if not decode_per_layer[L]:
            continue
        # Each chunk is (1, n_q, 1, d) → cat along step dim
        cat = torch.cat(decode_per_layer[L], dim=2).squeeze(0).cpu().to(torch.float16)
        decode_q.append(cat)
    decode_q = torch.stack(decode_q, dim=0)  # (n_layers, n_q, n_steps, d)
    return {"decode_q": decode_q, "n_steps": int(decode_q.shape[2]),
            "prompt_length": T, "decoded_token_ids": decoded}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--tasks", default="hotpotqa,qasper,qmsum,repobench-p,musique,multi_news,lcc,2wikimqa")
    parser.add_argument("--n-per-task", type=int, default=20)
    parser.add_argument("--exclude",
        default=str(REPO / "artifacts/calibration_splits/longbench_compact8_60_seed20260504_2k32k/exclude_train_indices_for_eval.json"))
    parser.add_argument("--out-dir", default=str(REPO / "artifacts/decode_q_captures_llama"))
    parser.add_argument("--max-context-length", type=int, default=12000)
    parser.add_argument("--max-new-tokens-override", type=int, default=0,
        help="If > 0, cap max_new_tokens at this value (saves time on qmsum which defaults to 532).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    exclude_map = json.loads(Path(args.exclude).read_text()) if Path(args.exclude).exists() else {}

    print(f"[{ts()}] loading {args.model}...", flush=True)
    model, tokenizer = load_model_and_tokenizer(args.model, dtype_name="bfloat16")
    print(f"[{ts()}] model on {get_model_device(model)}", flush=True)

    from datasets import load_dataset
    from kvq.benchmarks.evaluate_registry import DATASET_REGISTRY

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for task in tasks:
        task_dir = out_dir / task; task_dir.mkdir(exist_ok=True)
        ds = load_dataset(DATASET_REGISTRY["longbench"], task, split="test").to_pandas()
        to_drop = set(int(i) for i in exclude_map.get(task, []))
        keep_idx = [i for i in range(len(ds)) if i not in to_drop]
        rows = ds.iloc[keep_idx[: args.n_per_task]]
        print(f"\n[{ts()}] === {task}: {len(rows)} prompts ===", flush=True)

        for ri, (_, row) in enumerate(rows.iterrows()):
            row_idx = int(row.name) if hasattr(row, "name") else ri
            out_path = task_dir / f"row{row_idx:05d}.pt"
            if out_path.exists():
                continue
            spec = build_kvpress_dataset_spec(name="longbench", config_names=(task,),
                                              metadata_fields=("task",))
            try:
                ex = fetch_example(spec, task, row_idx, tokenizer)
            except Exception as e:
                print(f"[{ts()}]   {ri}: fetch failed: {e}", flush=True)
                continue
            mnt = int(row["max_new_tokens"])
            if args.max_new_tokens_override > 0:
                mnt = min(mnt, args.max_new_tokens_override)
            t0 = time.time()
            try:
                payload = capture_decode_q(model, tokenizer, ex.prompt_text, mnt, args.max_context_length)
            except Exception as e:
                print(f"[{ts()}]   {ri}: ERROR {type(e).__name__}: {e}", flush=True)
                continue
            n_steps = payload["n_steps"]
            torch.save(payload, out_path)
            print(f"[{ts()}]   {ri:>3}/{len(rows)} row{row_idx} T={payload['prompt_length']:>5} "
                  f"steps={n_steps:>3} | {time.time()-t0:.1f}s | {out_path.name}", flush=True)

    print(f"\n[{ts()}] done.", flush=True)


if __name__ == "__main__":
    main()
