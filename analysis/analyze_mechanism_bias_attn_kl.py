#!/usr/bin/env python3
"""Systematic mechanism analysis for the JQ K-fidelity → F1 disconnect on Llama.

We have shown: JointQK (A) dominates TurboQuant (C) on K-MSE, top-1, top-5 (both
prefill-q and decode-q) on both Qwen3 and Llama, but JQ loses F1 by 1.7 pp on
Llama at K=2 V=3. This script tests four mechanistic hypotheses, comparing
Qwen3 vs Llama at b=2 on the same 4 test prompts (hotpotqa/musique/qasper/qmsum,
intersection across the two raw stores):

  1) **Bit allocation profile**: histogram of bits per coord. If Llama's K
     spectrum makes water-fill zero more coords than Qwen3, the systematic bias
     would be larger.
  2) **Error decomposition (bias vs noise)**: per (L, h) split K reconstruction
     error into bias = mean_t(err) (systematic per-coord shift across tokens)
     and noise = err - bias. Report bias fraction (||bias||² · T / ||err||²).
     Softmax over keys + V averaging tolerates noise, amplifies bias.
  3) **Attention probability KL divergence**: per query g, compute
     KL(softmax(q · K_full^T / √d) || softmax(q · K_recon^T / √d)).
     Top-1 sees only argmax; KL captures full softmax distortion.
  4) **Attention output L2 error**: per query, compute relative L2 error of
     softmax(qK_full)·V vs softmax(qK_recon)·V — what the next layer receives.

Outputs:
  artifacts/calibration/mechanism_analysis_bias_attn_kl.json
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402, F401

import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kvq.compression.metric_transform import water_fill
from kvq.compression.per_coord import round_bits_to_integer, PerCoordCompressor
from kvq.compression.lloyd_max import Stage1MSECompressor


def ts() -> str:
    return time.strftime("%H:%M:%S")


def allocate_bits(scores, b_avg, max_coord_bits=8):
    leading_shape = scores.shape[:-1]
    d = int(scores.shape[-1])
    total_bits = int(b_avg) * d
    flat = scores.detach().float().cpu().reshape(-1, d).clamp_min(1e-30)
    cont = water_fill(flat, total_bits=float(total_bits))
    bits = round_bits_to_integer(cont, total_bits=total_bits)
    out = bits.clone()
    for i in range(out.shape[0]):
        excess = int((out[i] - max_coord_bits).clamp_min(0).sum().item())
        out[i].clamp_(max=max_coord_bits)
        while excess > 0:
            below = (out[i] < max_coord_bits).nonzero(as_tuple=True)[0]
            if below.numel() == 0:
                break
            min_val = out[i, below].min()
            cands = below[out[i, below] == min_val]
            take = min(int(cands.numel()), excess)
            out[i, cands[:take]] += 1
            excess -= take
    return out.reshape(leading_shape + (d,))


def bias_noise_decomp(err: torch.Tensor) -> dict[str, float]:
    """err: (T, d). Decompose into bias (per-coord mean) + noise (residual)."""
    T = err.shape[0]
    bias = err.mean(dim=0)                     # (d,)
    bias_energy = float((bias.pow(2).sum() * T).item())
    total_energy = float(err.pow(2).sum().item())
    return {
        "bias_energy": bias_energy,
        "noise_energy": total_energy - bias_energy,
        "total_energy": total_energy,
        "bias_frac": bias_energy / max(1e-30, total_energy),
    }


@torch.inference_mode()
def attn_metrics(
    q: torch.Tensor,            # (n_q, d) — pre-subsampled queries for this (L, h)
    k_full: torch.Tensor,       # (T, d)
    k_recon: torch.Tensor,      # (T, d)
    v: torch.Tensor,            # (T, d)
    device: torch.device,
) -> dict[str, float]:
    """Mean attention-prob KL and attention-output relative L2 error."""
    d = float(q.shape[-1])
    scale = 1.0 / (d ** 0.5)
    q = q.to(device).float()
    k_full = k_full.to(device).float()
    k_recon = k_recon.to(device).float()
    v = v.to(device).float()

    logits_full = (q @ k_full.T) * scale       # (n_q, T)
    logits_recon = (q @ k_recon.T) * scale
    p_full = F.softmax(logits_full, dim=-1)
    p_recon = F.softmax(logits_recon, dim=-1)
    log_p_full = F.log_softmax(logits_full, dim=-1)
    log_p_recon = F.log_softmax(logits_recon, dim=-1)

    # KL(p_full || p_recon) = sum p_full * (log p_full - log p_recon)
    kl = (p_full * (log_p_full - log_p_recon)).sum(dim=-1)  # (n_q,)
    # Symmetric also useful — but stick with directional for now.

    # Attention output L2
    o_full = p_full @ v                # (n_q, d)
    o_recon = p_recon @ v
    err_sq = (o_full - o_recon).pow(2).sum(dim=-1)   # (n_q,)
    full_sq = o_full.pow(2).sum(dim=-1).clamp_min(1e-30)
    rel_err = err_sq / full_sq

    return {
        "kl_mean": float(kl.mean().item()),
        "kl_p95": float(kl.quantile(0.95).item()),
        "attn_out_relerr_mean": float(rel_err.mean().item()),
        "attn_out_relerr_p95": float(rel_err.quantile(0.95).item()),
    }


_FNAME_RE = re.compile(
    r"^longbench__(?P<config>[A-Za-z0-9_\-]+)__row(?P<row>\d+)__(?P<split>train|test)\.pt$"
)


def find_prompts(raw_root: Path, target_tasks: list[str]) -> dict[str, Path]:
    """Map task → raw file path (first found per task)."""
    out: dict[str, Path] = {}
    for shard in sorted(raw_root.iterdir()):
        if not shard.is_dir():
            continue
        for f in sorted(shard.iterdir()):
            m = _FNAME_RE.match(f.name)
            if not m or m.group("split") != "test":
                continue
            task = m.group("config")
            if task in target_tasks and task not in out:
                out[task] = f
        if len(out) == len(target_tasks):
            break
    return out


def analyze_model(
    label: str,
    cca_path: Path,
    raw_root: Path,
    target_tasks: list[str],
    bits: int,
    n_query_subsample: int = 1024,
    max_T: int = 8000,
    device: torch.device = torch.device("cuda"),
) -> dict:
    print(f"\n[{ts()}] ==== {label} (b={bits}) ====", flush=True)
    cca = torch.load(cca_path, map_location="cpu", weights_only=False)
    R_sym = cca["R_sym"].float()
    Sq, Sk = cca["sigma_q"].float(), cca["sigma_k"].float()
    n_layers, n_kv_heads, head_dim, _ = R_sym.shape
    print(f"[{ts()}] R_sym {tuple(R_sym.shape)}", flush=True)

    Q_diag = (R_sym.transpose(-1, -2) @ Sq @ R_sym).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    K_diag = (R_sym.transpose(-1, -2) @ Sk @ R_sym).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    scores = Q_diag * K_diag
    train_k_std = K_diag.sqrt()
    cal_alloc = allocate_bits(scores, b_avg=bits, max_coord_bits=8)  # (L, h, d) int

    # ---- (1) Bit allocation histogram per layer-band (and overall) ----
    bits_overall = cal_alloc[1:].flatten().tolist()  # skip layer 0
    hist = {}
    for b in range(0, 9):
        hist[str(b)] = int(sum(1 for x in bits_overall if x == b))
    zero_frac = hist["0"] / max(1, sum(hist.values()))
    one_frac = hist["1"] / max(1, sum(hist.values()))
    print(f"[{ts()}] bit-alloc hist (layer-0 excl): "
          f"0={hist['0']}  1={hist['1']}  2={hist['2']}  3={hist['3']}  4={hist['4']}  "
          f">=5={sum(hist[str(b)] for b in range(5,9))} | zero_frac={zero_frac:.3f}", flush=True)

    # Build compressors per (L, h)
    print(f"[{ts()}] building compressors...", flush=True)
    comps_A = {}
    for L in range(n_layers):
        for h in range(n_kv_heads):
            comps_A[(L, h)] = PerCoordCompressor(
                bits_per_coord=cal_alloc[L, h],
                std_per_coord=train_k_std[L, h],
                forward_map=R_sym[L, h],
                inverse_map=R_sym[L, h].transpose(-1, -2),
            ).to(device)
    comp_C = Stage1MSECompressor(head_dim=head_dim, bits=bits, seed=42, device=device)

    raw_map = find_prompts(raw_root, target_tasks)
    print(f"[{ts()}] prompts: {[(t, str(raw_map[t].name)) for t in target_tasks if t in raw_map]}", flush=True)

    # Per-(method, task, layer) accumulators
    accums = {
        m: defaultdict(lambda: {
            "bias_energy": 0.0, "noise_energy": 0.0, "total_energy": 0.0,
            "kl_sum": 0.0, "kl_count": 0,
            "ao_rel_sum": 0.0, "ao_count": 0,
            "ao_rel_p95_max": 0.0,
            "kl_p95_max": 0.0,
        }) for m in ("A", "C")
    }

    rng = torch.Generator(device="cpu").manual_seed(20260513)
    for task in target_tasks:
        if task not in raw_map:
            print(f"[{ts()}] skipping {task} (no raw)", flush=True)
            continue
        raw = torch.load(raw_map[task], map_location="cpu", weights_only=False)
        q_all = raw["q_post"]
        k_all = raw["k_post"]
        v_all = raw["v"]
        T_full = int(raw["prompt_length"])
        T = min(T_full, max_T)
        n_q_heads = q_all.shape[1]
        group = n_q_heads // n_kv_heads
        print(f"[{ts()}] [{label}] {task:<14} T={T_full} (truncated to {T})", flush=True)

        for L in range(1, n_layers):
            for h in range(n_kv_heads):
                k_full = k_all[L, h, :T, :].to(device).float()
                q = q_all[L, h * group:(h + 1) * group, :T, :].to(device).float()  # (group, T, d)
                v = v_all[L, h, :T, :].to(device).float()

                # Subsample queries: pick n_query_subsample queries uniformly across (group*T)
                gT = group * T
                if n_query_subsample < gT:
                    idx = torch.randperm(gT, generator=rng)[:n_query_subsample]
                    q_flat = q.reshape(gT, head_dim)[idx]
                else:
                    q_flat = q.reshape(gT, head_dim)

                recon_A = comps_A[(L, h)].roundtrip(k_full)
                recon_C = comp_C.roundtrip(k_full.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

                for m, recon in (("A", recon_A), ("C", recon_C)):
                    # Bias / noise
                    err = recon - k_full
                    bn = bias_noise_decomp(err.cpu())
                    a = accums[m][(task, L)]
                    a["bias_energy"] += bn["bias_energy"]
                    a["noise_energy"] += bn["noise_energy"]
                    a["total_energy"] += bn["total_energy"]

                    # Attention metrics (KL + attn-out L2)
                    am = attn_metrics(q_flat, k_full, recon, v, device)
                    a["kl_sum"] += am["kl_mean"] * q_flat.shape[0]
                    a["kl_count"] += q_flat.shape[0]
                    a["ao_rel_sum"] += am["attn_out_relerr_mean"] * q_flat.shape[0]
                    a["ao_count"] += q_flat.shape[0]
                    a["ao_rel_p95_max"] = max(a["ao_rel_p95_max"], am["attn_out_relerr_p95"])
                    a["kl_p95_max"] = max(a["kl_p95_max"], am["kl_p95"])

                del k_full, q, v, recon_A, recon_C, q_flat
            if L % 8 == 0:
                torch.cuda.empty_cache()
        del raw, q_all, k_all, v_all
        torch.cuda.empty_cache()

    # Aggregate per (task, model) and overall
    def agg(method: str, filter_fn=lambda key: True) -> dict[str, float]:
        keys = [k for k in accums[method].keys() if filter_fn(k)]
        if not keys:
            return {}
        bias_e = sum(accums[method][k]["bias_energy"] for k in keys)
        total_e = sum(accums[method][k]["total_energy"] for k in keys)
        kl_sum = sum(accums[method][k]["kl_sum"] for k in keys)
        kl_count = sum(accums[method][k]["kl_count"] for k in keys)
        ao_sum = sum(accums[method][k]["ao_rel_sum"] for k in keys)
        ao_count = sum(accums[method][k]["ao_count"] for k in keys)
        return {
            "bias_frac": bias_e / max(1e-30, total_e),
            "kl_mean": kl_sum / max(1, kl_count),
            "attn_out_relerr_mean": ao_sum / max(1, ao_count),
            "kl_p95_max": max((accums[method][k]["kl_p95_max"] for k in keys), default=0.0),
            "ao_rel_p95_max": max((accums[method][k]["ao_rel_p95_max"] for k in keys), default=0.0),
            "total_energy": total_e,
            "n_cells": len(keys),
        }

    result = {
        "label": label,
        "bits": bits,
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "bit_alloc_histogram_excl_layer0": hist,
        "zero_bit_coord_fraction": zero_frac,
        "one_bit_coord_fraction": one_frac,
        "overall": {"A": agg("A"), "C": agg("C")},
        "per_task": {
            t: {"A": agg("A", lambda k, t=t: k[0] == t),
                "C": agg("C", lambda k, t=t: k[0] == t)}
            for t in target_tasks if t in raw_map
        },
        "per_layer_band": {},
    }
    # 4 layer bands (early / mid-early / mid-late / late)
    bands = [(1, n_layers // 4),
             (n_layers // 4, n_layers // 2),
             (n_layers // 2, (3 * n_layers) // 4),
             ((3 * n_layers) // 4, n_layers)]
    for lo, hi in bands:
        label_band = f"L{lo}-{hi-1}"
        result["per_layer_band"][label_band] = {
            "A": agg("A", lambda k, lo=lo, hi=hi: lo <= k[1] < hi),
            "C": agg("C", lambda k, lo=lo, hi=hi: lo <= k[1] < hi),
        }

    return result


def main():
    out_path = REPO / "artifacts/calibration/mechanism_analysis_bias_attn_kl.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_tasks = ["hotpotqa", "musique", "qasper", "qmsum"]
    bits = 2

    models = [
        {
            "label": "Qwen3-8B",
            "cca": REPO / "artifacts/bases/cca_stats_longbench_compact8_n400.pt",
            "raw": REPO / "artifacts/calibration/longbench_compact8_qkv/01_raw",
        },
        {
            "label": "Llama-3.1-8B",
            "cca": REPO / "artifacts/bases/cca_stats_llama31_8b_longbench_compact8_n400.pt",
            "raw": REPO / "artifacts/calibration/longbench_compact8_qkv_llama31_8b/01_raw",
        },
    ]

    results = {}
    for m in models:
        r = analyze_model(
            label=m["label"],
            cca_path=m["cca"],
            raw_root=m["raw"],
            target_tasks=target_tasks,
            bits=bits,
            device=device,
        )
        results[m["label"]] = r
        # Save partial after each model.
        out_path.write_text(json.dumps(results, indent=2, default=str) + "\n")
        print(f"\n[{ts()}] partial save → {out_path}", flush=True)

    # Cross-model summary print
    print("\n" + "=" * 100)
    print(f"=== CROSS-MODEL MECHANISM SUMMARY (b={bits}, layer-0 excluded, target tasks: {target_tasks}) ===")
    print("=" * 100)
    for label, r in results.items():
        print(f"\n--- {label} ---")
        print(f"Bit-alloc 0-bit fraction: {r['zero_bit_coord_fraction']:.3f}   "
              f"1-bit fraction: {r['one_bit_coord_fraction']:.3f}")
        for method, mname in (("A", "JointQK"), ("C", "TurboQuant")):
            o = r["overall"][method]
            print(f"  {mname:<10}: bias_frac={o['bias_frac']:.4f}  "
                  f"KL(p_full || p_recon)_mean={o['kl_mean']:.4f}  "
                  f"attn_out_relerr_mean={o['attn_out_relerr_mean']:.4f}  "
                  f"(KL p95 max={o['kl_p95_max']:.4f}, attn_out p95 max={o['ao_rel_p95_max']:.4f})")

    print(f"\n[{ts()}] all results → {out_path}", flush=True)


if __name__ == "__main__":
    main()
