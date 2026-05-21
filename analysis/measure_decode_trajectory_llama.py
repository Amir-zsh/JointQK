#!/usr/bin/env python3
"""Multi-step decode trajectory: teacher-forced per-step logit KL on Llama-3.1-8B.

Setup: prefill once per method (FP, JointQK, TurboQuant). Then greedy-decode N
tokens, where the trajectory follows FP's argmax. At each step t, all three
caches advance by feeding the FP-chosen token; we capture FP/JQ/TQ logits at
that position and compute:

  - top1_match_JQ[t]: argmax(p_JQ[t]) == argmax(p_FP[t])
  - top1_match_TQ[t]: same for TQ
  - KL(p_FP[t] || p_JQ[t]) per step
  - p_method[FP_argmax[t]] (probability mass each method places on FP's chosen token)

This is teacher-forcing under FP's trajectory: all three methods see identical
conversation history at every step. The per-step JQ-vs-TQ KL curve reveals
whether JQ's small first-step distortion compounds (catches up to or surpasses
TQ's distortion) over the answer-generation window.

Hypothesis: on lcc/hotpotqa, JQ's per-step KL curve crosses above TQ's at some
t > 0, explaining the F1 inversion despite first-token superiority.
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
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from transformers import DynamicCache
from kvq.capture.model import get_model_device, load_model_and_tokenizer
from kvq.presses.jointqk_press import JointQKPress
from kvq.presses.turboquant_press import TurboQuantPress


def ts() -> str:
    return time.strftime("%H:%M:%S")


def chat_templated_context_and_question(tokenizer, context: str, question: str, answer_prefix: str,
                                        enable_thinking: bool = False):
    """Replicate kvpress pipeline's context/question split exactly."""
    separator = "#" * (len(context) + 10)
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": context + separator}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=enable_thinking,
    )
    context_text, question_suffix = full.split(separator)
    question_text = question + question_suffix + (answer_prefix or "")
    context_ids = tokenizer.encode(context_text, return_tensors="pt", add_special_tokens=False)
    question_ids = tokenizer.encode(question_text, return_tensors="pt", add_special_tokens=False)
    return context_ids, question_ids


@torch.inference_mode()
def prefill_with_press(model, context_ids: torch.Tensor, press, max_context_length: int) -> DynamicCache:
    device = get_model_device(model)
    if context_ids.shape[1] > max_context_length:
        context_ids = context_ids[:, :max_context_length]
    cache = DynamicCache()
    if press is None:
        model.model(input_ids=context_ids.to(device), past_key_values=cache)
    else:
        with press(model):
            model.model(input_ids=context_ids.to(device), past_key_values=cache)
    return cache


@torch.inference_mode()
def forward_token(model, input_ids: torch.Tensor, cache: DynamicCache, position_id: int):
    """Single-token forward; returns (vocab,) logits and mutates cache in place."""
    device = get_model_device(model)
    pos = torch.tensor([[position_id]], device=device)
    out = model(
        input_ids=input_ids.to(device),
        past_key_values=cache,
        position_ids=pos,
        num_logits_to_keep=1,
    )
    return out.logits[0, -1, :].detach().float().cpu()


@torch.inference_mode()
def consume_question_under_method(model, cache: DynamicCache, question_ids: torch.Tensor,
                                   context_length: int) -> torch.Tensor:
    """Feed the question_ids through the prefilled cache (Mode A: no decode-time
    compression). Returns the logits at the last question position (which would
    produce the first answer token)."""
    device = get_model_device(model)
    qids = question_ids.to(device)
    position_ids = torch.arange(context_length, context_length + qids.shape[1],
                                 device=device).unsqueeze(0)
    out = model(
        input_ids=qids,
        past_key_values=cache,
        position_ids=position_ids,
        num_logits_to_keep=1,
    )
    return out.logits[0, -1, :].detach().float().cpu()


def kl_div(log_p: torch.Tensor, log_q: torch.Tensor) -> float:
    p = log_p.exp()
    return float((p * (log_p - log_q)).sum().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--cca", default=str(REPO / "artifacts/bases/cca_stats_llama31_8b_longbench_compact8_n400.pt"))
    parser.add_argument("--vst", default=str(REPO / "artifacts/v_bases/v_stats_llama31_8b_longbench_compact8_n400.pt"))
    parser.add_argument("--exclude", default=str(REPO / "artifacts/calibration_splits/longbench_compact8_60_seed20260504_2k32k/exclude_train_indices_for_eval.json"))
    parser.add_argument("--k-bits", type=int, default=2)
    parser.add_argument("--v-bits", type=int, default=3)
    parser.add_argument("--tasks", nargs="+", default=["lcc", "repobench-p", "hotpotqa"])
    parser.add_argument("--n-per-task", type=int, default=15)
    parser.add_argument("--n-decode", type=int, default=50)
    parser.add_argument("--max-context-length", type=int, default=12000)
    parser.add_argument("--out", default=str(REPO / "artifacts/calibration/decode_trajectory_llama_k2.json"))
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    exclude_map = json.loads(Path(args.exclude).read_text())

    print(f"[{ts()}] loading {args.model}...", flush=True)
    model, tokenizer = load_model_and_tokenizer(args.model, dtype_name="bfloat16")
    device = get_model_device(model)
    print(f"[{ts()}] model on {device}", flush=True)

    jq_press = JointQKPress(
        cca_stats_path=args.cca,
        v_stats_path=args.vst,
        v_method="v_turboquant",
        k_method="r_sym_waterfill",
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        compress_decode=False,
        layer0_full_precision=True,
        quantize_k=True,
        quantize_v=True,
    )
    tq_press = TurboQuantPress(
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        compress_decode=False,
        layer0_full_precision=True,
    )
    jq_press.post_init_from_model(model)
    tq_press.post_init_from_model(model)
    print(f"[{ts()}] presses initialized", flush=True)

    from datasets import load_dataset
    from kvq.benchmarks.evaluate_registry import DATASET_REGISTRY

    results = {
        "model": args.model,
        "k_bits": args.k_bits, "v_bits": args.v_bits,
        "n_per_task": args.n_per_task, "n_decode": args.n_decode,
        "max_context_length": args.max_context_length,
        "tasks": {},
    }

    for task in args.tasks:
        print(f"\n[{ts()}] === {task} ===", flush=True)
        ds = load_dataset(DATASET_REGISTRY["longbench"], task, split="test").to_pandas()
        to_drop = set(int(i) for i in exclude_map.get(task, []))
        keep_idx = [i for i in range(len(ds)) if i not in to_drop]
        rows = ds.iloc[keep_idx[: args.n_per_task]]

        # Per-step accumulators: step t → list of metrics across prompts
        N = args.n_decode
        kl_jq_per_step = [[] for _ in range(N)]
        kl_tq_per_step = [[] for _ in range(N)]
        match_jq_per_step = [[] for _ in range(N)]
        match_tq_per_step = [[] for _ in range(N)]
        per_prompt = []

        for ri, (_, row) in enumerate(rows.iterrows()):
            context_ids, question_ids = chat_templated_context_and_question(
                tokenizer, row["context"], row["question"], row.get("answer_prefix", "")
            )
            T = min(int(context_ids.shape[1]), args.max_context_length)
            t0 = time.time()
            try:
                cache_fp = prefill_with_press(model, context_ids, press=None,
                                              max_context_length=args.max_context_length)
                cache_jq = prefill_with_press(model, context_ids, press=jq_press,
                                              max_context_length=args.max_context_length)
                cache_tq = prefill_with_press(model, context_ids, press=tq_press,
                                              max_context_length=args.max_context_length)

                # Feed question_ids through all three caches to advance to first-decode position.
                logits_fp = consume_question_under_method(model, cache_fp, question_ids, T)
                logits_jq = consume_question_under_method(model, cache_jq, question_ids, T)
                logits_tq = consume_question_under_method(model, cache_tq, question_ids, T)

                # Step 0: log first-decode logits and the FP-greedy token.
                prompt_record = {"row_index": int(row.name) if hasattr(row,"name") else ri,
                                 "context_len": T, "question_len": int(question_ids.shape[1]),
                                 "per_step": []}
                next_pos = T + int(question_ids.shape[1])  # position id for next forward
                for t in range(N):
                    log_fp = F.log_softmax(logits_fp, dim=-1)
                    log_jq = F.log_softmax(logits_jq, dim=-1)
                    log_tq = F.log_softmax(logits_tq, dim=-1)
                    kl_jq = kl_div(log_fp, log_jq)
                    kl_tq = kl_div(log_fp, log_tq)
                    argmax_fp = int(log_fp.argmax().item())
                    argmax_jq = int(log_jq.argmax().item())
                    argmax_tq = int(log_tq.argmax().item())
                    match_jq = (argmax_jq == argmax_fp)
                    match_tq = (argmax_tq == argmax_fp)
                    p_jq_at_fpmax = float(log_jq[argmax_fp].exp().item())
                    p_tq_at_fpmax = float(log_tq[argmax_fp].exp().item())

                    kl_jq_per_step[t].append(kl_jq)
                    kl_tq_per_step[t].append(kl_tq)
                    match_jq_per_step[t].append(int(match_jq))
                    match_tq_per_step[t].append(int(match_tq))
                    prompt_record["per_step"].append({
                        "kl_jq": kl_jq, "kl_tq": kl_tq,
                        "match_jq": match_jq, "match_tq": match_tq,
                        "p_jq_at_fpmax": p_jq_at_fpmax, "p_tq_at_fpmax": p_tq_at_fpmax,
                        "argmax_fp_id": argmax_fp,
                    })

                    # Stop if FP hits EOS
                    if argmax_fp in getattr(model.generation_config, "eos_token_id", []) or argmax_fp == 128009:
                        break

                    # Advance all three caches by the FP-chosen token
                    next_token = torch.tensor([[argmax_fp]])
                    logits_fp = forward_token(model, next_token, cache_fp, next_pos)
                    logits_jq = forward_token(model, next_token, cache_jq, next_pos)
                    logits_tq = forward_token(model, next_token, cache_tq, next_pos)
                    next_pos += 1

                per_prompt.append(prompt_record)
                wall = time.time() - t0
                # Quick summary line for this prompt
                first_kl_jq = prompt_record["per_step"][0]["kl_jq"]
                first_kl_tq = prompt_record["per_step"][0]["kl_tq"]
                last_t = len(prompt_record["per_step"]) - 1
                last_kl_jq = prompt_record["per_step"][last_t]["kl_jq"]
                last_kl_tq = prompt_record["per_step"][last_t]["kl_tq"]
                m_jq = sum(s["match_jq"] for s in prompt_record["per_step"]) / max(1, len(prompt_record["per_step"]))
                m_tq = sum(s["match_tq"] for s in prompt_record["per_step"]) / max(1, len(prompt_record["per_step"]))
                print(f"[{ts()}]   {ri:>3}/{len(rows)} T={T:>5} steps={len(prompt_record['per_step']):>2} | "
                      f"step0 KL_JQ/TQ={first_kl_jq:.3f}/{first_kl_tq:.3f} → "
                      f"step{last_t} KL={last_kl_jq:.3f}/{last_kl_tq:.3f} | "
                      f"match_JQ={m_jq:.2f} TQ={m_tq:.2f} | {wall:.1f}s", flush=True)
            except Exception as e:
                print(f"[{ts()}]   prompt {ri}: ERROR {type(e).__name__}: {e}", flush=True)
                continue

        # Per-step aggregates across this task's prompts
        per_step_agg = []
        for t in range(N):
            if not kl_jq_per_step[t]:
                continue
            per_step_agg.append({
                "t": t,
                "n": len(kl_jq_per_step[t]),
                "mean_kl_jq": sum(kl_jq_per_step[t]) / len(kl_jq_per_step[t]),
                "mean_kl_tq": sum(kl_tq_per_step[t]) / len(kl_tq_per_step[t]),
                "frac_match_jq": sum(match_jq_per_step[t]) / len(match_jq_per_step[t]),
                "frac_match_tq": sum(match_tq_per_step[t]) / len(match_tq_per_step[t]),
            })
        results["tasks"][task] = {"per_prompt": per_prompt, "per_step_agg": per_step_agg}

        # Quick step-vs-step summary print
        print(f"\n[{ts()}] {task} per-step summary (averaged across prompts):")
        print(f"{'t':>3} {'n':>3} {'KL_JQ':>8} {'KL_TQ':>8} {'ΔKL':>8} | {'match_JQ':>8} {'match_TQ':>8}")
        for s in per_step_agg[::5]:  # every 5 steps
            print(f"{s['t']:>3} {s['n']:>3} {s['mean_kl_jq']:>8.4f} {s['mean_kl_tq']:>8.4f} "
                  f"{s['mean_kl_jq']-s['mean_kl_tq']:>+8.4f} | "
                  f"{s['frac_match_jq']:>8.3f} {s['frac_match_tq']:>8.3f}")
        # Show the last few steps too
        if len(per_step_agg) > 5:
            print(" --- last 3 steps ---")
            for s in per_step_agg[-3:]:
                print(f"{s['t']:>3} {s['n']:>3} {s['mean_kl_jq']:>8.4f} {s['mean_kl_tq']:>8.4f} "
                      f"{s['mean_kl_jq']-s['mean_kl_tq']:>+8.4f} | "
                      f"{s['frac_match_jq']:>8.3f} {s['frac_match_tq']:>8.3f}")
        out_path.write_text(json.dumps(results, indent=2, default=str) + "\n")
        print(f"[{ts()}] partial save → {out_path}", flush=True)

    print(f"\n=== Final per-task crossover summary ===")
    print(f"{'task':<14}  {'crossover_t':>11}  {'@t=0_ΔKL':>10}  {'@last_ΔKL':>10}  {'@t=0_matchΔ':>12}  {'@last_matchΔ':>13}")
    for task, td in results["tasks"].items():
        agg = td["per_step_agg"]
        if not agg: continue
        crossover = None
        for s in agg:
            if s["mean_kl_jq"] > s["mean_kl_tq"]:
                crossover = s["t"]; break
        first = agg[0]; last = agg[-1]
        co_str = str(crossover) if crossover is not None else "never"
        print(f"{task:<14}  {co_str:>11}  "
              f"{first['mean_kl_jq']-first['mean_kl_tq']:>+10.4f}  "
              f"{last['mean_kl_jq']-last['mean_kl_tq']:>+10.4f}  "
              f"{first['frac_match_jq']-first['frac_match_tq']:>+12.3f}  "
              f"{last['frac_match_jq']-last['frac_match_tq']:>+13.3f}")


if __name__ == "__main__":
    main()
