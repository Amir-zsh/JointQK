#!/usr/bin/env python3
"""Diagnose K/Q distribution drift between fp16 (calibration) and bf16 (eval) for Llama and Qwen3.

Hypothesis: Llama-3.1-8B was trained in bf16, but our calibration captures K in
fp16. Eval (kvpress pipeline) loads the model in bf16. If the K distribution
differs significantly between dtypes, R_sym fitted on fp16 K is mis-fit for
bf16 K used at eval time.

For each model:
  1. Load in fp16, run prefill on one prompt, capture K_post.
  2. Load in bf16, run prefill on same prompt, capture K_post.
  3. Report per-(layer, head) cosine sim, relative L2 error, eigenvalue spectrum
     drift in Σ_K, basis stability of R_sym.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _bootstrap  # noqa: E402, F401

import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pipelines.calibration.analyze_bases import jointqk_basis
from kvq.toolkit.capture import capture_rope_qk
from kvq.toolkit.model import get_model_device, load_model_and_tokenizer
from kvq.data.kvpress_adapter import build_kvpress_dataset_spec
from kvq.data.base import fetch_example


def ts() -> str:
    return time.strftime("%H:%M:%S")


@torch.inference_mode()
def capture_prefill_qk(model, tokenizer, prompt_text: str, max_tokens: int = 4000):
    device = get_model_device(model)
    n_layers = int(model.config.num_hidden_layers)
    enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_tokens)
    input_ids = enc["input_ids"].to(device)
    T = int(input_ids.shape[-1])
    with capture_rope_qk(model) as (_q_pre, q_post_chunks, _k_pre, _k_post_chunks):
        outputs = model(input_ids=input_ids, use_cache=True)
    # k_post from cache (shape: B, H_kv, T, d)
    k_post = []
    for L in range(n_layers):
        k_post.append(outputs.past_key_values.layers[L].keys.squeeze(0).cpu())
    k_post = torch.stack(k_post, dim=0)  # (L, H_kv, T, d)
    # q_post: concat per layer (chunks are emitted in layer order on single prefill forward)
    q_post = []
    for L in range(n_layers):
        q_post.append(q_post_chunks[L].squeeze(0).cpu())
    q_post = torch.stack(q_post, dim=0)
    return q_post, k_post, T


def analyze_model(model_name: str, prompt_text: str, max_tokens: int = 4000):
    print(f"\n{'='*80}\n=== {model_name} ===\n{'='*80}", flush=True)

    print(f"[{ts()}] loading {model_name} in fp16...", flush=True)
    model_fp16, tokenizer = load_model_and_tokenizer(model_name, dtype_name="float16")
    q_fp16, k_fp16, T = capture_prefill_qk(model_fp16, tokenizer, prompt_text, max_tokens)
    del model_fp16
    torch.cuda.empty_cache()

    print(f"[{ts()}] loading {model_name} in bf16...", flush=True)
    model_bf16, _ = load_model_and_tokenizer(model_name, dtype_name="bfloat16")
    q_bf16, k_bf16, _ = capture_prefill_qk(model_bf16, tokenizer, prompt_text, max_tokens)
    del model_bf16
    torch.cuda.empty_cache()

    # Convert to fp32 for comparison
    q_fp16_f = q_fp16.float()
    q_bf16_f = q_bf16.float()
    k_fp16_f = k_fp16.float()
    k_bf16_f = k_bf16.float()

    n_layers, n_q_heads, T_q, d = q_fp16_f.shape
    _, n_kv_heads, T_k, _ = k_fp16_f.shape

    print(f"[{ts()}] shapes: q ({n_layers}, {n_q_heads}, {T_q}, {d}), k ({n_layers}, {n_kv_heads}, {T_k}, {d})", flush=True)

    # Per-(layer, head) drift: relative L2 error in K, cosine similarity, Σ_K drift, R_sym alignment
    print(f"\n{'L':>3} {'h':>2} | {'cosK':>8} {'relK':>8} | {'cosQ':>8} {'relQ':>8} | "
          f"{'eig0_fp':>10} {'eig0_bf':>10} | {'R_sym_cos':>10}", flush=True)
    print("-" * 90)

    band_rows = {"L1-7": [], "L8-15": [], "L16-23": [], "L24+": []}
    for L in range(n_layers):
        if L == 0:
            continue
        cosK_l, relK_l = [], []
        cosQ_l, relQ_l = [], []
        eig0_diff = []
        R_sym_cos_l = []

        for h in range(n_kv_heads):
            kf = k_fp16_f[L, h]  # (T, d)
            kb = k_bf16_f[L, h]
            # Token-wise cosine + relative L2
            dot = (kf * kb).sum(dim=-1)
            nf = kf.norm(dim=-1).clamp_min(1e-30)
            nb = kb.norm(dim=-1).clamp_min(1e-30)
            cosK = (dot / (nf * nb)).mean().item()
            relK = ((kf - kb).pow(2).sum() / kf.pow(2).sum()).sqrt().item()
            cosK_l.append(cosK)
            relK_l.append(relK)

            # Σ_K eigenvalue spectrum
            Sk_fp = (kf.T @ kf) / kf.shape[0]
            Sk_bf = (kb.T @ kb) / kb.shape[0]
            eig_fp = torch.linalg.eigvalsh(Sk_fp).flip(0)
            eig_bf = torch.linalg.eigvalsh(Sk_bf).flip(0)
            eig0_diff.append((float(eig_fp[0].item()), float(eig_bf[0].item())))

            # R_sym (joint Q-K basis) stability
            group = n_q_heads // n_kv_heads
            qf = q_fp16_f[L, h * group:(h + 1) * group].reshape(-1, d)
            qb = q_bf16_f[L, h * group:(h + 1) * group].reshape(-1, d)
            Sq_fp = (qf.T @ qf) / qf.shape[0]
            Sq_bf = (qb.T @ qb) / qb.shape[0]
            dot_q = (qf * qb).sum(dim=-1)
            cosQ_l.append((dot_q / (qf.norm(dim=-1).clamp_min(1e-30) * qb.norm(dim=-1).clamp_min(1e-30))).mean().item())
            relQ_l.append(((qf - qb).pow(2).sum() / qf.pow(2).sum()).sqrt().item())

            R_fp = jointqk_basis(Sq_fp.unsqueeze(0).unsqueeze(0), Sk_fp.unsqueeze(0).unsqueeze(0), eps=1e-4).squeeze()
            R_bf = jointqk_basis(Sq_bf.unsqueeze(0).unsqueeze(0), Sk_bf.unsqueeze(0).unsqueeze(0), eps=1e-4).squeeze()
            # Top-3 eigenvector alignment: mean abs cosine of first 3 columns
            top_k = 3
            cos_per_col = [abs((R_fp[:, j] * R_bf[:, j]).sum().item()) for j in range(top_k)]
            R_sym_cos_l.append(sum(cos_per_col) / top_k)

        cosK_mean = sum(cosK_l) / len(cosK_l)
        relK_mean = sum(relK_l) / len(relK_l)
        cosQ_mean = sum(cosQ_l) / len(cosQ_l)
        relQ_mean = sum(relQ_l) / len(relQ_l)
        eig0_fp_mean = sum(e[0] for e in eig0_diff) / len(eig0_diff)
        eig0_bf_mean = sum(e[1] for e in eig0_diff) / len(eig0_diff)
        R_cos_mean = sum(R_sym_cos_l) / len(R_sym_cos_l)
        print(f"{L:>3}  - | {cosK_mean:>8.5f} {relK_mean:>8.5f} | {cosQ_mean:>8.5f} {relQ_mean:>8.5f} | "
              f"{eig0_fp_mean:>10.4f} {eig0_bf_mean:>10.4f} | {R_cos_mean:>10.5f}", flush=True)

        # Band aggregation
        if L < 8:
            band = "L1-7"
        elif L < 16:
            band = "L8-15"
        elif L < 24:
            band = "L16-23"
        else:
            band = "L24+"
        band_rows[band].append({"cosK": cosK_mean, "relK": relK_mean, "cosQ": cosQ_mean, "relQ": relQ_mean,
                                "R_cos": R_cos_mean,
                                "eig0_fp": eig0_fp_mean, "eig0_bf": eig0_bf_mean})

    print(f"\n=== Layer-band summary ({model_name}) ===")
    print(f"{'band':<10} {'cosK_mean':>10} {'relK_mean':>10} {'cosQ_mean':>10} {'relQ_mean':>10} {'R_sym_cos':>10}")
    print("-" * 70)
    for band in ("L1-7", "L8-15", "L16-23", "L24+"):
        rows = band_rows[band]
        if not rows:
            continue
        cosK = sum(r["cosK"] for r in rows) / len(rows)
        relK = sum(r["relK"] for r in rows) / len(rows)
        cosQ = sum(r["cosQ"] for r in rows) / len(rows)
        relQ = sum(r["relQ"] for r in rows) / len(rows)
        R = sum(r["R_cos"] for r in rows) / len(rows)
        print(f"{band:<10} {cosK:>10.5f} {relK:>10.5f} {cosQ:>10.5f} {relQ:>10.5f} {R:>10.5f}")


def main():
    # Use one Llama test prompt
    tokenizer_dummy = None
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", trust_remote_code=True)
    spec = build_kvpress_dataset_spec(name="longbench", config_names=("hotpotqa",), metadata_fields=("task",))
    ex = fetch_example(spec, "hotpotqa", 166, tokenizer)
    print(f"prompt length: {ex.prompt_length} tokens")

    for model_name in ("meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen3-8B"):
        analyze_model(model_name, ex.prompt_text, max_tokens=4000)


if __name__ == "__main__":
    main()
