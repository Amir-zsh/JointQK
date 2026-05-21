#!/usr/bin/env python3
"""Empirical K-MSE + top-1 + top-5 retention on Llama-3.1-8B test captures.

For every (layer, head) and every test prompt:
  A = JointQK = R_sym basis + calibrated water-fill allocation
  C = TurboQuant V3 = random Hadamard + uniform Lloyd-Max

Reports:
  - mean K-MSE per (L,h) for A vs C
  - mean top-1 retention per (L,h)
  - mean top-5 retention per (L,h)
  - per-task breakdown
  - per-layer-band breakdown
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _bootstrap  # noqa: E402, F401

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
import sys
sys.path.insert(0, '.')

from kvq.toolkit.metric_transform import water_fill
from kvq.toolkit.per_coord_quantization import round_bits_to_integer, PerCoordCompressor
from kvq.toolkit.quantization import Stage1MSECompressor


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
            if below.numel() == 0: break
            min_val = out[i, below].min()
            cands = below[out[i, below] == min_val]
            take = min(int(cands.numel()), excess)
            out[i, cands[:take]] += 1
            excess -= take
    return out.reshape(leading_shape + (d,))


def topk_pair(q_h, k_full, k_recon, device, k_top=5):
    """Per-query top-1 + top-5 retention. q_h: (group*T, d), k_*: (T, d)."""
    q_h = q_h.to(device).float()
    kf = k_full.to(device).float()
    kr = k_recon.to(device).float()
    T = kf.shape[0]
    CHUNK_BYTES = 1 << 29
    chunk_q = max(1, CHUNK_BYTES // (max(T, 1) * 4))
    top1_num, top1_den = 0, 0
    top5_num, top5_den = 0, 0
    mse_num, mse_den = 0.0, 0
    actual_top5 = min(k_top, T)
    for qstart in range(0, q_h.shape[0], chunk_q):
        qc = q_h[qstart:qstart + chunk_q]
        full_logits = qc @ kf.T
        recon_logits = qc @ kr.T
        true_argmax = full_logits.argmax(dim=-1)
        recon_argmax = recon_logits.argmax(dim=-1)
        top1_num += int((true_argmax == recon_argmax).sum().item())
        top1_den += int(true_argmax.numel())
        recon_top5 = recon_logits.topk(actual_top5, dim=-1).indices
        matches5 = (recon_top5 == true_argmax.unsqueeze(-1)).any(dim=-1)
        top5_num += int(matches5.sum().item())
        top5_den += int(matches5.numel())
        del full_logits, recon_logits, true_argmax, recon_argmax, recon_top5, matches5
    # K-MSE on full keys (independent of q)
    err = kf - kr
    mse_num = float(err.pow(2).sum().item())
    mse_den = int(err.numel())
    return {
        "top1_num": top1_num, "top1_den": top1_den,
        "top5_num": top5_num, "top5_den": top5_den,
        "mse_num": mse_num, "mse_den": mse_den,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cca-path", default="artifacts/bases/cca_stats_llama31_8b_longbench_compact8_n400.pt")
    parser.add_argument("--raw-dir", default="artifacts/calibration/longbench_compact8_qkv_llama31_8b/01_raw")
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--out", default="artifacts/calibration/longbench_compact8_qkv_llama31_8b/05_reports/llama_empirical_kmse_top1_top5.json")
    parser.add_argument("--label", default="Llama-3.1-8B")
    args = parser.parse_args()

    cca = torch.load(args.cca_path, map_location="cpu", weights_only=False)
    R_sym = cca["R_sym"].float()
    Sq, Sk = cca["sigma_q"].float(), cca["sigma_k"].float()
    n_layers, n_kv_heads, head_dim, _ = R_sym.shape
    print(f"[{time.strftime('%H:%M:%S')}] {args.label}: R_sym {tuple(R_sym.shape)}", flush=True)

    Q_diag = (R_sym.transpose(-1, -2) @ Sq @ R_sym).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    K_diag = (R_sym.transpose(-1, -2) @ Sk @ R_sym).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    scores = Q_diag * K_diag
    train_k_std = K_diag.sqrt()

    cal_alloc = allocate_bits(scores, b_avg=args.bits, max_coord_bits=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{time.strftime('%H:%M:%S')}] building compressors...", flush=True)
    comps_A = {}
    for L in range(n_layers):
        for h in range(n_kv_heads):
            comps_A[(L, h)] = PerCoordCompressor(
                bits_per_coord=cal_alloc[L, h],
                std_per_coord=train_k_std[L, h],
                forward_map=R_sym[L, h],
                inverse_map=R_sym[L, h].transpose(-1, -2),
            ).to(device)
    comp_C = Stage1MSECompressor(head_dim=head_dim, bits=args.bits, seed=42, device=device)

    raw_files = sorted(Path(args.raw_dir).rglob("*__test.pt"))
    print(f"[{time.strftime('%H:%M:%S')}] found {len(raw_files)} test raw files", flush=True)

    # Per-(method, task, layer) accumulators
    accums = {m: defaultdict(lambda: {"top1_num": 0, "top1_den": 0,
                                        "top5_num": 0, "top5_den": 0,
                                        "mse_num": 0.0, "mse_den": 0})
              for m in ("A", "C")}

    for pi, raw_path in enumerate(raw_files):
        # Extract task from filename: longbench__<task>__row<N>__<split>.pt
        parts = raw_path.name.split("__")
        task = parts[1] if len(parts) >= 4 else "unknown"
        raw = torch.load(raw_path, map_location="cpu", weights_only=False)
        q_all = raw["q_post"]
        k_all = raw["k_post"]
        T = int(raw["prompt_length"])
        n_q_heads = q_all.shape[1]
        group = n_q_heads // n_kv_heads
        print(f"[{time.strftime('%H:%M:%S')}] [{pi+1}/{len(raw_files)}] {task:<14} T={T:>6}", flush=True)
        for L in range(1, n_layers):
            for h in range(n_kv_heads):
                k = k_all[L, h, :T, :].to(device).float()
                q = q_all[L, h * group:(h + 1) * group, :T, :].to(device).float().reshape(-1, head_dim)
                recon_A = comps_A[(L, h)].roundtrip(k)
                recon_C = comp_C.roundtrip(k.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                for m, recon in (("A", recon_A), ("C", recon_C)):
                    r = topk_pair(q, k, recon, device, k_top=5)
                    key = (task, L)
                    a = accums[m][key]
                    for s in ("top1_num", "top1_den", "top5_num", "top5_den", "mse_num", "mse_den"):
                        a[s] += r[s]
                del k, q, recon_A, recon_C
        del raw, q_all, k_all
        torch.cuda.empty_cache()

    # Aggregate
    print(f"\n[{time.strftime('%H:%M:%S')}] === FINAL AGGREGATE (b={args.bits}, layer-0 excluded) ===\n")

    def mean(num, den): return num / max(1, den)

    # Overall
    for m, name in (("A", "JointQK (R_sym + cal alloc)"), ("C", "TurboQuant (random Hadamard)")):
        total = {"top1_num": 0, "top1_den": 0, "top5_num": 0, "top5_den": 0, "mse_num": 0.0, "mse_den": 0}
        for key, v in accums[m].items():
            for s in total:
                total[s] += v[s]
        print(f"{name:<35}  top1={mean(total['top1_num'], total['top1_den']):.4f}  top5={mean(total['top5_num'], total['top5_den']):.4f}  K-MSE={mean(total['mse_num'], total['mse_den']):.4e}")

    # Delta
    print()
    print("Δ JointQK − TurboQuant:")
    da = {}
    for s in ("top1", "top5", "mse"):
        a_num = sum(v[f"{s}_num"] for v in accums["A"].values())
        a_den = sum(v[f"{s}_den"] for v in accums["A"].values())
        c_num = sum(v[f"{s}_num"] for v in accums["C"].values())
        c_den = sum(v[f"{s}_den"] for v in accums["C"].values())
        da[s] = (mean(a_num, a_den), mean(c_num, c_den))
    print(f"  top1:  JQ={da['top1'][0]:.4f}  TQ={da['top1'][1]:.4f}  Δ={da['top1'][0]-da['top1'][1]:+.4f}")
    print(f"  top5:  JQ={da['top5'][0]:.4f}  TQ={da['top5'][1]:.4f}  Δ={da['top5'][0]-da['top5'][1]:+.4f}")
    print(f"  K-MSE: JQ={da['mse'][0]:.4e} TQ={da['mse'][1]:.4e} ratio={da['mse'][0]/da['mse'][1]:.4f}")

    # Per-task breakdown
    print("\n=== Per-task ===")
    print(f"{'task':<14} | {'JQ top1':>8} | {'TQ top1':>8} | {'Δ t1':>7} | {'JQ top5':>8} | {'TQ top5':>8} | {'Δ t5':>7} | {'JQ MSE':>10} | {'TQ MSE':>10} | {'JQ/TQ':>6}")
    print("-" * 120)
    tasks = sorted(set(k[0] for k in accums["A"].keys()))
    per_task_results = {}
    for task in tasks:
        rs = {}
        for m in "AC":
            a_top1n = sum(v["top1_num"] for k, v in accums[m].items() if k[0] == task)
            a_top1d = sum(v["top1_den"] for k, v in accums[m].items() if k[0] == task)
            a_top5n = sum(v["top5_num"] for k, v in accums[m].items() if k[0] == task)
            a_top5d = sum(v["top5_den"] for k, v in accums[m].items() if k[0] == task)
            a_msen = sum(v["mse_num"] for k, v in accums[m].items() if k[0] == task)
            a_msed = sum(v["mse_den"] for k, v in accums[m].items() if k[0] == task)
            rs[m] = (a_top1n/max(1, a_top1d), a_top5n/max(1, a_top5d), a_msen/max(1, a_msed))
        per_task_results[task] = rs
        jt1, jt5, jm = rs["A"]
        ct1, ct5, cm = rs["C"]
        print(f"{task:<14} | {jt1:>8.4f} | {ct1:>8.4f} | {jt1-ct1:>+7.4f} | {jt5:>8.4f} | {ct5:>8.4f} | {jt5-ct5:>+7.4f} | {jm:>10.4e} | {cm:>10.4e} | {jm/cm:>6.3f}")

    # Per-layer-band
    print("\n=== Per-layer band ===")
    bands = [("early (1-12)",  range(1, min(12, n_layers))),
             ("middle (12-23)", range(min(12, n_layers), min(23, n_layers))),
             ("late",           range(min(23, n_layers), n_layers))]
    for label, rng in bands:
        rng_set = set(rng)
        for m, name in (("A", "JointQK"), ("C", "TurboQuant")):
            top1n = sum(v["top1_num"] for k, v in accums[m].items() if k[1] in rng_set)
            top1d = sum(v["top1_den"] for k, v in accums[m].items() if k[1] in rng_set)
            top5n = sum(v["top5_num"] for k, v in accums[m].items() if k[1] in rng_set)
            top5d = sum(v["top5_den"] for k, v in accums[m].items() if k[1] in rng_set)
            msen  = sum(v["mse_num"] for k, v in accums[m].items() if k[1] in rng_set)
            msed  = sum(v["mse_den"] for k, v in accums[m].items() if k[1] in rng_set)
            print(f"  {label:<16}  {name:<12}  top1={top1n/max(1,top1d):.4f}  top5={top5n/max(1,top5d):.4f}  K-MSE={msen/max(1,msed):.4e}")

    # Save JSON
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": args.label,
        "bits": args.bits,
        "n_test_prompts": len(raw_files),
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "overall": {m: {s: sum(v[f"{s}_num"] for v in accums[m].values()) / max(1, sum(v[f"{s}_den"] for v in accums[m].values()))
                         for s in ("top1", "top5", "mse")} for m in "AC"},
        "per_task": {t: {m: {s: rs[i] for i, s in enumerate(("top1", "top5", "mse"))}
                          for m, rs in per_task_results[t].items()} for t in per_task_results},
    }
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"\n[{time.strftime('%H:%M:%S')}] wrote {out}")


if __name__ == "__main__":
    main()
