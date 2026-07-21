#!/usr/bin/env python3
"""Capture post-RoPE Q/K/V from a model on GPQA-Diamond and emit them in the
per-layer chunk layout that OSCAR's compute_kv_rotation.py consumes:

  <out>/layer_<id>/q/<chunk>.pt   [T, n_heads,  head_dim]  (post-RoPE)
  <out>/layer_<id>/k/<chunk>.pt   [T, kv_heads, head_dim]  (post-RoPE)
  <out>/layer_<id>/v/<chunk>.pt   [T, kv_heads, head_dim]  (raw V)

This replaces the sglang dump-hook server (Amir's pipeline) with our own HF
capture so we can build a new model's OSCAR rotations without that stack. Same
198-prompt GPQA calibration, chat-templated per the served model. compute's
"all" mode skips chunk 0, so prompts are written as chunks 1..N (dummy 0).
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # our depth: third_party/samuel_vq/
import _bootstrap  # noqa: F401
from pipelines.calibration.capture_raw import run_prefill_qkv_capture  # noqa
from kvq.capture.model import get_model_device, load_model_and_tokenizer  # noqa

TMPL = ("Answer the following multiple choice question. The last line of your response "
        "should be of the following format: 'Answer: $LETTER' (without quotes) where "
        "LETTER is one of ABCD. Think step by step before answering.\n\n{Question}\n\n"
        "A) {A}\nB) {B}\nC) {C}\nD) {D}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    ap.add_argument("--csv", default="/vault/samuel/efficient-llm/JointQK/artifacts/gpqa_diamond/gpqa_diamond.csv")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num-prompts", type=int, default=198)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-chat-template", action="store_true")
    ap.add_argument("--device-map", default="single",
                    help="'single' pins to --gpu; 'auto' spreads across CUDA_VISIBLE_DEVICES "
                         "(needed for 32B which does not fit one 40GB card).")
    args = ap.parse_args()
    if args.device_map == "single":
        torch.cuda.set_device(args.gpu)
        dmap = {"": args.gpu}
    else:
        dmap = "auto"

    model, tok = load_model_and_tokenizer(args.model, device_map=dmap, dtype_name="float16")
    get_model_device(model)

    df = pd.read_csv(args.csv)
    prompts = [TMPL.format(Question=r["Question"], A=r["Correct Answer"],
                           B=r["Incorrect Answer 1"], C=r["Incorrect Answer 2"],
                           D=r["Incorrect Answer 3"]) for _, r in df.iterrows()][: args.num_prompts]
    out = Path(args.out)
    n_layers = None
    # dummy chunk 0 (6 tokens) so compute's "all" mode (which skips chunk 0) keeps every real prompt
    for pi, p in enumerate(prompts):
        if args.no_chat_template:
            ids = tok(p, return_tensors="pt", add_special_tokens=True).input_ids
        else:
            text = tok.apply_chat_template([{"role": "user", "content": p}],
                                           add_generation_prompt=True, tokenize=False)
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
        cap = run_prefill_qkv_capture(model, ids)
        q, k, v = cap["q_post"], cap["k_post"], cap["v"]   # (L,H,T,d)
        n_layers = q.shape[0]
        chunk = pi + 1
        for l in range(n_layers):
            for name, t in (("q", q[l]), ("k", k[l]), ("v", v[l])):
                d = out / f"layer_{l}" / name
                d.mkdir(parents=True, exist_ok=True)
                torch.save(t.permute(1, 0, 2).to(torch.float16).contiguous(), d / f"{chunk}.pt")  # [T,H,d]
        if pi == 0:  # write the dummy chunk 0 per layer/name from a 6-token slice
            for l in range(n_layers):
                for name, t in (("q", q[l]), ("k", k[l]), ("v", v[l])):
                    d = out / f"layer_{l}" / name
                    torch.save(t.permute(1, 0, 2)[:6].to(torch.float16).contiguous(), d / "0.pt")
        del cap, q, k, v
        if (pi + 1) % 25 == 0:
            print(f"  captured {pi+1}/{len(prompts)} (T={ids.shape[1]})", flush=True)
    print(f"DONE dump -> {out}  ({len(prompts)} prompts, {n_layers} layers)", flush=True)


if __name__ == "__main__":
    main()
