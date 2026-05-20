#!/usr/bin/env python3
"""Measure softmax-KL between full and compressed attention distributions.

Tests hypothesis A1: top-1 measures argmax preservation, but real attention is a
softmax. JointQK may preserve top-1 well while distorting the softmax tail (its
zero-bit minor coords erase contributions, biasing the score landscape). TurboQuant
preserves top-1 less but adds unbiased noise.

Method:
  For each test prompt × layer × head:
    p_true = softmax(q_prefill ⊤ k_full / √d)
    p_comp = softmax(q_prefill ⊤ k_compressed / √d)
    Compute KL(p_true || p_comp) per query, average over (queries, heads, layers).

Scope: 8 test prompts × all 36 layers × 8 KV heads × b=4.
Methods: v3, jointqk (NEW pooled basis).

Output: artifacts/calibration/longbench_compact8_qkv/05_reports/softmax_kl.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.calibration.analyze_bases import combine_stats, jointqk_basis
from experiments.calibration.common import RunPaths
from experiments.toolkit.per_coord_quantization import PerCoordCompressor, round_bits_to_integer
from experiments.toolkit.metric_transform import water_fill
from experiments.toolkit.quantization import Stage1MSECompressor


def ts() -> str:
    return time.strftime("%H:%M:%S")


def allocate_bits_waterfill(scores, b_avg: int, max_coord_bits: int = 8):
    d = int(scores.shape[-1])
    total_bits = b_avg * d
    flat = scores.detach().float().cpu().reshape(-1, d).clamp_min(1e-30)
    cont = water_fill(flat, total_bits=float(total_bits))
    bits_int = round_bits_to_integer(cont, total_bits=total_bits)
    out = bits_int.clone()
    for i in range(out.shape[0]):
        excess = int((out[i] - max_coord_bits).clamp_min(0).sum().item())
        out[i].clamp_(max=max_coord_bits)
        while excess > 0:
            below = (out[i] < max_coord_bits).nonzero(as_tuple=True)[0]
            if below.numel() == 0: break
            min_val = out[i, below].min()
            cands = below[out[i, below] == min_val]
            take = min(int(cands.numel()), excess)
            out[i, cands[:take]] += 1
            excess -= take
    return out.reshape(scores.shape)


def main() -> None:
    bits = 4
    n_prompts = 8

    artifact_root = REPO / "artifacts/calibration"
    paths = RunPaths.from_args(artifact_root, "longbench_compact8_qkv")
    agg = torch.load(paths.stats_dir / "aggregate.pt", map_location="cpu", weights_only=False)
    per_example = agg["per_example"]
    test_examples = [p for p in per_example if p["split"] == "test"][:n_prompts]
    train_indices = [int(p["index"]) for p in per_example if p["split"] == "train"]
    print(f"[{ts()}] pooling train stats over {len(train_indices)} examples", flush=True)
    train_stats = combine_stats(paths, per_example, train_indices, torch.device("cpu"))
    sigma_q = train_stats["sigma_q"].float()
    sigma_k = train_stats["sigma_k"].float()
    n_layers, n_kv_heads, head_dim, _ = sigma_q.shape

    print(f"[{ts()}] computing R_sym...", flush=True)
    R_sym_all = jointqk_basis(sigma_q, sigma_k, eps=1e-4).float()
    train_q_diag = (R_sym_all.transpose(-1, -2) @ sigma_q @ R_sym_all).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    train_k_diag = (R_sym_all.transpose(-1, -2) @ sigma_k @ R_sym_all).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    alloc = allocate_bits_waterfill(train_q_diag * train_k_diag, bits, max_coord_bits=8).cpu()
    train_k_std = torch.sqrt(train_k_diag).cpu()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sqrt_d = float(head_dim) ** 0.5

    accums = {
        "v3": {"kl_sum": 0.0, "n": 0, "top1_num": 0, "top1_den": 0},
        "jointqk": {"kl_sum": 0.0, "n": 0, "top1_num": 0, "top1_den": 0},
    }
    # Per-layer breakdown for late-layer hypothesis
    per_layer_kl = {m: {l: {"sum": 0.0, "n": 0} for l in range(n_layers)} for m in accums}

    for pi, meta in enumerate(test_examples):
        if meta.get("raw_file") is None:
            continue
        rp = paths.root / meta["raw_file"]
        if not rp.exists():
            continue
        print(f"[{ts()}] [{pi+1}/{len(test_examples)}] {meta['config']} loading {rp.name} ...", flush=True)
        raw = torch.load(rp, map_location="cpu", weights_only=False)
        T = int(raw["prompt_length"])
        q_all = raw["q_post"]   # (n_layers, n_q_heads, T, d)
        k_all = raw["k_post"]   # (n_layers, n_kv_heads, T, d)
        n_q_heads = q_all.shape[1]
        group = n_q_heads // n_kv_heads

        for L in range(1, n_layers):  # skip layer 0
            for h in range(n_kv_heads):
                k = k_all[L, h, :T, :].to(device).float()
                q = q_all[L, h*group:(h+1)*group, :T, :].to(device).float().reshape(-1, head_dim)

                # v3 reconstruction
                comp_v3 = Stage1MSECompressor(head_dim=head_dim, bits=bits, seed=20260505, device=device)
                k_recon_v3 = comp_v3.roundtrip(k.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

                # jointqk reconstruction
                R_h = R_sym_all[L, h].to(device)
                comp_jq = PerCoordCompressor(
                    bits_per_coord=alloc[L, h], std_per_coord=train_k_std[L, h],
                    forward_map=R_sym_all[L, h], inverse_map=R_sym_all[L, h].transpose(-1, -2),
                ).to(device)
                k_recon_jq = comp_jq.roundtrip(k)

                # Chunk queries to bound peak memory at ~1 GiB per logit matrix.
                CHUNK_BYTES = 1 << 30
                chunk_q = max(1, CHUNK_BYTES // (max(T, 1) * 4))
                k_t = k.T
                k_v3_t = k_recon_v3.T
                k_jq_t = k_recon_jq.T
                kl_v3_sum = 0.0; kl_jq_sum = 0.0
                top1_v3 = 0; top1_jq = 0; n_q = 0
                for qstart in range(0, q.shape[0], chunk_q):
                    qchunk = q[qstart:qstart + chunk_q]
                    full_logits = (qchunk @ k_t) / sqrt_d
                    v3_logits = (qchunk @ k_v3_t) / sqrt_d
                    jq_logits = (qchunk @ k_jq_t) / sqrt_d
                    p_true = F.log_softmax(full_logits, dim=-1)
                    logp_v3 = F.log_softmax(v3_logits, dim=-1)
                    logp_jq = F.log_softmax(jq_logits, dim=-1)
                    p = p_true.exp()
                    kl_v3_sum += float((p * (p_true - logp_v3)).sum(dim=-1).sum().item())
                    kl_jq_sum += float((p * (p_true - logp_jq)).sum(dim=-1).sum().item())
                    t_argmax = full_logits.argmax(dim=-1)
                    top1_v3 += int((v3_logits.argmax(dim=-1) == t_argmax).sum().item())
                    top1_jq += int((jq_logits.argmax(dim=-1) == t_argmax).sum().item())
                    n_q += int(qchunk.shape[0])
                    del full_logits, v3_logits, jq_logits, p_true, logp_v3, logp_jq, p, t_argmax
                kl_v3 = kl_v3_sum / max(1, n_q)
                kl_jq = kl_jq_sum / max(1, n_q)
                accums["v3"]["kl_sum"] += kl_v3 * n_q
                accums["v3"]["n"] += n_q
                accums["jointqk"]["kl_sum"] += kl_jq * n_q
                accums["jointqk"]["n"] += n_q

                per_layer_kl["v3"][L]["sum"] += kl_v3 * n_q
                per_layer_kl["v3"][L]["n"] += n_q
                per_layer_kl["jointqk"][L]["sum"] += kl_jq * n_q
                per_layer_kl["jointqk"][L]["n"] += n_q

                accums["v3"]["top1_num"] += top1_v3
                accums["jointqk"]["top1_num"] += top1_jq
                accums["v3"]["top1_den"] += n_q
                accums["jointqk"]["top1_den"] += n_q

                del comp_v3, comp_jq, k, q, k_recon_v3, k_recon_jq, k_t, k_v3_t, k_jq_t
            torch.cuda.empty_cache()
        del raw
        import gc; gc.collect()

    print(f"\n=== Softmax KL summary (b={bits}, layer-0 excluded) ===")
    for m in ("v3", "jointqk"):
        kl = accums[m]["kl_sum"] / max(1, accums[m]["n"])
        top1 = accums[m]["top1_num"] / max(1, accums[m]["top1_den"])
        print(f"  {m}: mean KL = {kl:.4f}   top1 = {top1:.4f}")

    print(f"\n=== Late-layer (24-35) KL ===")
    for m in ("v3", "jointqk"):
        s = sum(per_layer_kl[m][L]["sum"] for L in range(24, n_layers))
        n = sum(per_layer_kl[m][L]["n"] for L in range(24, n_layers))
        print(f"  {m}: mean KL on layers 24-35 = {s/max(1,n):.4f}")

    out = {
        "bits": bits,
        "n_prompts": n_prompts,
        "summary": {m: {
            "mean_kl": accums[m]["kl_sum"] / max(1, accums[m]["n"]),
            "top1": accums[m]["top1_num"] / max(1, accums[m]["top1_den"]),
        } for m in accums},
        "per_layer_kl": {m: {L: per_layer_kl[m][L]["sum"] / max(1, per_layer_kl[m][L]["n"])
                              for L in range(n_layers)} for m in accums},
    }
    out_path = REPO / "artifacts/calibration/longbench_compact8_qkv/05_reports/softmax_kl.json"
    out_path.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
