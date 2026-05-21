#!/usr/bin/env python3
"""Logit-KL divergence at first decode position on Llama-3.1-8B.

Goal: pinpoint where the K-fidelity → F1 disconnect on Llama lives. Top-1, top-5,
K-MSE, attention KL, and attention output L2 all say JointQK ≫ TurboQuant. F1 says
JQ < TQ on lcc/repobench-p/hotpotqa at K=2 V=3. The disconnect must be downstream
of the attention output.

For each prompt + method M ∈ {FP, JointQK, TurboQuant}:
  1. Apply chat template (matches kvpress pipeline at eval time).
  2. Prefill context_ids with M's press hook → cache has (compressed) K.
  3. Decode pass on question_ids → outputs.logits[0, -1, :].
  4. log_p_M = log_softmax(logits).

Per prompt:
  - KL(p_FP || p_JQ) and KL(p_FP || p_TQ)
  - argmax_FP vs argmax_M (top-1 token agreement)
  - p_M[argmax_FP] (how much probability mass at FP's top token)

Per task: mean / median / max of these metrics. The clean signal is:
  if mean(KL_FP→JQ) > mean(KL_FP→TQ) on lcc despite JQ winning K-fidelity,
  the F1 inversion is a downstream-propagation phenomenon (W_O / MLP / later layers).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from transformers import DynamicCache
from kvq.toolkit.model import get_model_device, load_model_and_tokenizer
from kvq.toolkit.jointqk_press import JointQKPress
from kvq.toolkit.turboquant_press import TurboQuantPress


def ts() -> str:
    return time.strftime("%H:%M:%S")


def chat_templated_context_and_question(tokenizer, context: str, question: str, answer_prefix: str,
                                        enable_thinking: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
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
def first_decode_logits(model, context_ids: torch.Tensor, question_ids: torch.Tensor,
                        press, max_context_length: int = 16000) -> torch.Tensor:
    """Run prefill + first decode pass; return logits over vocab at the position
    that would generate the first answer token. Shape: (vocab,)."""
    device = get_model_device(model)
    context_ids = context_ids.to(device)
    question_ids = question_ids.to(device)
    if context_ids.shape[1] > max_context_length:
        context_ids = context_ids[:, :max_context_length]
    context_length = context_ids.shape[1]

    cache = DynamicCache()
    if press is None:
        model.model(input_ids=context_ids, past_key_values=cache)
    else:
        with press(model):
            model.model(input_ids=context_ids, past_key_values=cache)

    # Decode pass on question_ids (no press hooks active for decode — Mode A)
    position_ids = torch.arange(context_length, context_length + question_ids.shape[1],
                                device=device).unsqueeze(0)
    outputs = model(
        input_ids=question_ids,
        past_key_values=cache,
        position_ids=position_ids,
        num_logits_to_keep=1,
    )
    return outputs.logits[0, -1, :].detach().float().cpu()


def kl_div(log_p: torch.Tensor, log_q: torch.Tensor) -> float:
    """KL(P || Q) = sum p * (log p - log q). Inputs are log-probs over vocab."""
    p = log_p.exp()
    return float((p * (log_p - log_q)).sum().item())


def js_div(log_p: torch.Tensor, log_q: torch.Tensor) -> float:
    p = log_p.exp()
    q = log_q.exp()
    m = ((p + q) / 2).clamp_min(1e-30)
    log_m = m.log()
    return 0.5 * (float((p * (log_p - log_m)).sum().item()) +
                  float((q * (log_q - log_m)).sum().item()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--cca", default=str(REPO / "artifacts/bases/cca_stats_llama31_8b_longbench_compact8_n400.pt"))
    parser.add_argument("--vst", default=str(REPO / "artifacts/v_bases/v_stats_llama31_8b_longbench_compact8_n400.pt"))
    parser.add_argument("--exclude", default=str(REPO / "artifacts/calibration_splits/longbench_compact8_60_seed20260504_2k32k/exclude_train_indices_for_eval.json"))
    parser.add_argument("--k-bits", type=int, default=2)
    parser.add_argument("--v-bits", type=int, default=3)
    parser.add_argument("--tasks", nargs="+", default=["lcc", "repobench-p", "hotpotqa", "qasper"])
    parser.add_argument("--n-per-task", type=int, default=20)
    parser.add_argument("--max-context-length", type=int, default=12000)
    parser.add_argument("--out", default=str(REPO / "artifacts/calibration/logit_kl_llama_k2.json"))
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    exclude_map = json.loads(Path(args.exclude).read_text())

    print(f"[{ts()}] loading {args.model}...", flush=True)
    model, tokenizer = load_model_and_tokenizer(args.model, dtype_name="bfloat16")
    device = get_model_device(model)
    print(f"[{ts()}] model on {device}", flush=True)

    # Build the JointQK and TurboQuant presses once (cached after first post_init_from_model)
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
    # Pre-init so each prompt doesn't pay the build cost
    jq_press.post_init_from_model(model)
    tq_press.post_init_from_model(model)
    print(f"[{ts()}] presses initialized", flush=True)

    from datasets import load_dataset
    from kvq.benchmarks.evaluate_registry import DATASET_REGISTRY

    results: dict[str, Any] = {
        "model": args.model,
        "k_bits": args.k_bits,
        "v_bits": args.v_bits,
        "n_per_task": args.n_per_task,
        "max_context_length": args.max_context_length,
        "tasks": {},
    }

    for task in args.tasks:
        print(f"\n[{ts()}] === {task} ===", flush=True)
        ds = load_dataset(DATASET_REGISTRY["longbench"], task, split="test").to_pandas()
        # Exclude calibration train rows
        to_drop = set(int(i) for i in exclude_map.get(task, []))
        keep_idx = [i for i in range(len(ds)) if i not in to_drop]
        rows = ds.iloc[keep_idx[: args.n_per_task]]
        print(f"[{ts()}] using {len(rows)} test prompts (after exclude)", flush=True)

        per_prompt = []
        for ri, (_, row) in enumerate(rows.iterrows()):
            context_ids, question_ids = chat_templated_context_and_question(
                tokenizer, row["context"], row["question"], row.get("answer_prefix", "")
            )
            T = min(int(context_ids.shape[1]), args.max_context_length)
            t0 = time.time()
            try:
                logits_fp = first_decode_logits(model, context_ids, question_ids, press=None,
                                                 max_context_length=args.max_context_length)
                logits_jq = first_decode_logits(model, context_ids, question_ids, press=jq_press,
                                                 max_context_length=args.max_context_length)
                logits_tq = first_decode_logits(model, context_ids, question_ids, press=tq_press,
                                                 max_context_length=args.max_context_length)
            except Exception as e:
                print(f"[{ts()}]   prompt {ri}: ERROR {type(e).__name__}: {e}", flush=True)
                continue

            log_fp = F.log_softmax(logits_fp, dim=-1)
            log_jq = F.log_softmax(logits_jq, dim=-1)
            log_tq = F.log_softmax(logits_tq, dim=-1)

            kl_fp_jq = kl_div(log_fp, log_jq)
            kl_fp_tq = kl_div(log_fp, log_tq)
            js_fp_jq = js_div(log_fp, log_jq)
            js_fp_tq = js_div(log_fp, log_tq)

            argmax_fp = int(log_fp.argmax().item())
            argmax_jq = int(log_jq.argmax().item())
            argmax_tq = int(log_tq.argmax().item())
            p_jq_at_fpmax = float(log_jq[argmax_fp].exp().item())
            p_tq_at_fpmax = float(log_tq[argmax_fp].exp().item())
            p_fp_at_fpmax = float(log_fp[argmax_fp].exp().item())

            row_result = {
                "row_index": int(row.name) if hasattr(row, "name") else ri,
                "context_len": T,
                "question_len": int(question_ids.shape[1]),
                "kl_fp_jq": kl_fp_jq,
                "kl_fp_tq": kl_fp_tq,
                "js_fp_jq": js_fp_jq,
                "js_fp_tq": js_fp_tq,
                "top1_match_jq": argmax_jq == argmax_fp,
                "top1_match_tq": argmax_tq == argmax_fp,
                "p_jq_at_fpmax": p_jq_at_fpmax,
                "p_tq_at_fpmax": p_tq_at_fpmax,
                "p_fp_max": p_fp_at_fpmax,
            }
            per_prompt.append(row_result)
            wall = time.time() - t0
            print(f"[{ts()}]   {ri:>3}/{len(rows)} T={T:>5} | "
                  f"KL_JQ={kl_fp_jq:.4f}  KL_TQ={kl_fp_tq:.4f}  "
                  f"top1_JQ={int(row_result['top1_match_jq'])} top1_TQ={int(row_result['top1_match_tq'])} | "
                  f"{wall:.1f}s", flush=True)

        # Task aggregate
        if per_prompt:
            def mean(key): return sum(d[key] for d in per_prompt) / len(per_prompt)
            def med(key):
                vs = sorted(d[key] for d in per_prompt); return vs[len(vs)//2]
            agg = {
                "n": len(per_prompt),
                "mean_kl_fp_jq": mean("kl_fp_jq"),
                "mean_kl_fp_tq": mean("kl_fp_tq"),
                "med_kl_fp_jq": med("kl_fp_jq"),
                "med_kl_fp_tq": med("kl_fp_tq"),
                "mean_js_fp_jq": mean("js_fp_jq"),
                "mean_js_fp_tq": mean("js_fp_tq"),
                "top1_match_jq_frac": sum(d["top1_match_jq"] for d in per_prompt) / len(per_prompt),
                "top1_match_tq_frac": sum(d["top1_match_tq"] for d in per_prompt) / len(per_prompt),
                "mean_p_jq_at_fpmax": mean("p_jq_at_fpmax"),
                "mean_p_tq_at_fpmax": mean("p_tq_at_fpmax"),
                "mean_p_fp_max": mean("p_fp_max"),
            }
            print(f"[{ts()}] {task} aggregate: "
                  f"mean KL_JQ={agg['mean_kl_fp_jq']:.4f}  KL_TQ={agg['mean_kl_fp_tq']:.4f}  "
                  f"ΔKL(JQ-TQ)={agg['mean_kl_fp_jq']-agg['mean_kl_fp_tq']:+.4f}  "
                  f"top1_JQ={agg['top1_match_jq_frac']:.3f} top1_TQ={agg['top1_match_tq_frac']:.3f}",
                  flush=True)
            results["tasks"][task] = {"per_prompt": per_prompt, "aggregate": agg}
        else:
            results["tasks"][task] = {"per_prompt": [], "aggregate": None}

        out_path.write_text(json.dumps(results, indent=2, default=str) + "\n")
        print(f"[{ts()}] partial save → {out_path}", flush=True)

    print(f"\n=== Final summary (Llama-3.1-8B, K={args.k_bits} V={args.v_bits}) ===")
    print(f"{'task':<14} | {'n':>3} {'KL_JQ':>8} {'KL_TQ':>8} {'ΔKL':>8} | {'top1_JQ':>8} {'top1_TQ':>8}")
    for task, td in results["tasks"].items():
        a = td["aggregate"]
        if a is None: continue
        d = a["mean_kl_fp_jq"] - a["mean_kl_fp_tq"]
        print(f"{task:<14} | {a['n']:>3} {a['mean_kl_fp_jq']:>8.4f} {a['mean_kl_fp_tq']:>8.4f} {d:>+8.4f} | "
              f"{a['top1_match_jq_frac']:>8.3f} {a['top1_match_tq_frac']:>8.3f}")


if __name__ == "__main__":
    main()
