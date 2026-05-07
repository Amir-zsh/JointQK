#!/usr/bin/env python
"""Empirical verification for jointqk_motivation.md §10 with proper train/test split.

The bases (V_Q, V_h, R_sym, P_K) are extracted from 400 LongBench training
prompts via build_calibration_artifacts_from_pool.py (stored in
cca_stats_longbench_compact8_n400.pt). Evaluation uses the held-out 80 test
prompts whose Σ_Q, Σ_K, raw Q/K are completely disjoint from training.

Phases:
  A. Predicted-only (fast, all 80 test prompts): geomean ratios on test
     moments using train bases; M-indefinite check on train Σ_Q, Σ_K;
     active-set sizes for water-fill; per-task cross-task spread; Hadamard
     floor and headroom.
  B. Real quantization (slow, subset of test files): runs the actual
     PerCoordCompressor on test K, computes Q-weighted geo-distortion against
     test Σ_Q, and top-1 retention against test Q.

Outputs JSON consumed by §10's tables.

Usage:
    ./.venv/bin/python notes/stage1/stage1e_cca_vs_waterfill/jointqk_motivation_review/code/run_empirical_verification.py
        [--n-test-files-quant N]    # files for phase B (default 16)
        [--device cuda|cpu]
        [--skip-quant]              # skip phase B
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

THIS_DIR = Path(__file__).resolve().parent
PROJ_ROOT = THIS_DIR.parents[4]
sys.path.insert(0, str(PROJ_ROOT))

from experiments.stage1.toolkit.metric_transform import (  # noqa: E402
    compute_cca_basis,
    water_fill,
    whitening_factor,
)
from experiments.stage1.toolkit.per_coord_quantization import (  # noqa: E402
    build_method_compressor,
)
from experiments.stage1.toolkit.quantization import Stage1MSECompressor  # noqa: E402
from experiments.stage1.run_cca_vs_waterfill_study import _derive_vh_rsym  # noqa: E402


# ---------------------------------------------------------------------------
# CLI

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cca-stats", default=str(PROJ_ROOT / "artifacts/stage1/cca_vs_waterfill_study/cca_stats_longbench_compact8_n400.pt"))
    p.add_argument("--bundle", default=str(PROJ_ROOT / "artifacts/stage1/calibration/longbench_compact8_qkv"))
    p.add_argument("--output", default=str(THIS_DIR.parent / "results" / "empirical_results.json"))
    p.add_argument("--log", default=str(THIS_DIR.parent / "results" / "empirical_run.log"))
    p.add_argument("--n-test-files-quant", type=int, default=16, help="files for real-quantization phase (B)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--b-avgs", default="2,3,4")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--skip-quant", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers — pure-statistics analyses (Phase A)

def per_head_geomean(R: torch.Tensor, Sq: torch.Tensor, Sk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For orthogonal R, return per-(layer,kv_head) geomean of:
        Q-side: (R^T Σ_Q R)_{jj}
        K-side: (R^T Σ_K R)_{jj}
        product: above × above
    All inputs (..., d, d). Returns three (n_layers, n_kv_heads) tensors.
    """
    A = R.transpose(-1, -2) @ Sq @ R
    B = R.transpose(-1, -2) @ Sk @ R
    a = A.diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    b = B.diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    log_q = a.log().mean(-1)
    log_k = b.log().mean(-1)
    return log_q.exp(), log_k.exp(), (log_q + log_k).exp()


def hadamard_floor_per_head(Sq: torch.Tensor, Sk: torch.Tensor) -> torch.Tensor:
    """Per-head (det Σ_Q · det Σ_K)^{1/d}."""
    sym_q = 0.5 * (Sq + Sq.transpose(-1, -2))
    sym_k = 0.5 * (Sk + Sk.transpose(-1, -2))
    _, log_det_q = torch.linalg.slogdet(sym_q)
    _, log_det_k = torch.linalg.slogdet(sym_k)
    d = sym_q.shape[-1]
    return ((log_det_q + log_det_k) / d).exp()


def m_indefinite_check(Sq: torch.Tensor, Sk: torch.Tensor) -> dict[str, Any]:
    """Signature of M = (Σ_Q Σ_K + Σ_K Σ_Q)/2 across all heads."""
    sq_flat = Sq.reshape(-1, Sq.shape[-1], Sq.shape[-1]).double()
    sk_flat = Sk.reshape(-1, Sk.shape[-1], Sk.shape[-1]).double()
    M = 0.5 * (sq_flat @ sk_flat + sk_flat @ sq_flat)
    M = 0.5 * (M + M.transpose(-1, -2))
    eigs = torch.linalg.eigvalsh(M)
    n_total = eigs.numel()
    n_neg = int((eigs < 0).sum())
    min_per_head = eigs.min(dim=-1).values
    max_per_head = eigs.max(dim=-1).values
    n_heads_neg = int((min_per_head < 0).sum())
    return {
        "min_eigenvalue_overall": float(eigs.min()),
        "max_eigenvalue_overall": float(eigs.max()),
        "n_negative_eigvals": n_neg,
        "n_total_eigvals": int(n_total),
        "frac_negative": n_neg / n_total,
        "n_heads_with_negative": n_heads_neg,
        "n_heads_total": int(min_per_head.numel()),
        "min_max_ratio_min": float((min_per_head / max_per_head).min()),
        "min_max_ratio_max": float((min_per_head / max_per_head).max()),
    }


def compute_v_k_basis(sigma_k: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Eigvecs of Σ_K (descending). Same regularization as run_cca_vs_waterfill_study.

    Operates on a flattened (n_heads, d, d) view to keep gather indexing simple,
    then reshapes back to the input's leading shape.
    """
    sym = 0.5 * (sigma_k + sigma_k.transpose(-1, -2))
    leading_shape = sym.shape[:-2]
    d = sym.shape[-1]
    flat = sym.reshape(-1, d, d)
    trace = flat.diagonal(dim1=-2, dim2=-1).sum(-1)
    eye_d = torch.eye(d, dtype=flat.dtype, device=flat.device)
    reg = (eps * (trace / d).clamp_min(1e-12)).unsqueeze(-1).unsqueeze(-1) * eye_d
    eigvals, eigvecs = torch.linalg.eigh(flat + reg)
    sort_idx = torch.argsort(eigvals, dim=-1, descending=True)
    eigvecs = torch.gather(eigvecs, -1, sort_idx.unsqueeze(-2).expand(-1, d, -1))
    return eigvecs.reshape(leading_shape + (d, d))


def random_orthogonal(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    A = torch.randn(*shape, dtype=torch.float64)
    Q, _ = torch.linalg.qr(A)
    return Q


def active_set_sizes(R: torch.Tensor, Sq: torch.Tensor, Sk: torch.Tensor, b_avgs: list[int]) -> dict[int, dict[str, float]]:
    """Run reverse water-fill matching the production runner, count active coords."""
    A = R.transpose(-1, -2) @ Sq @ R
    B = R.transpose(-1, -2) @ Sk @ R
    w = A.diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    sig = B.diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    prod = w * sig
    d = prod.shape[-1]
    flat = prod.reshape(-1, d).float()
    out: dict[int, dict[str, float]] = {}
    for b_avg in b_avgs:
        total_bits = b_avg * d
        bits_continuous = water_fill(flat, total_bits=total_bits)
        floor_bits = bits_continuous.floor()
        remainder = bits_continuous - floor_bits
        bits_int = floor_bits.long()
        target = int(total_bits)
        for i in range(flat.shape[0]):
            deficit = target - int(bits_int[i].sum().item())
            if deficit > 0:
                _, top = torch.topk(remainder[i], k=deficit)
                bits_int[i, top] += 1
        active = (bits_int > 0).sum(-1)
        out[b_avg] = {
            "min": int(active.min()),
            "median": int(active.median().item()),
            "max": int(active.max()),
            "mean": float(active.float().mean()),
        }
    return out


# ---------------------------------------------------------------------------
# Real-quantization analyses (Phase B)

def evaluate_method_on_payload(
    method: str,
    q_dev: torch.Tensor,            # (n_layers, n_q_heads, L, d) on device
    k_dev: torch.Tensor,            # (n_layers, n_kv_heads, L, d) on device
    train_calib_dev: dict[str, torch.Tensor],  # all on device
    Sq_test_dev: torch.Tensor,      # (n_layers, n_kv_heads, d, d) on device
    b_avg: int,
    rank: int,
    seed: int,
    device: str,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    """Run quantization + reconstruction for one method on one test payload.

    All input tensors are pre-moved to device. Inner loop avoids any
    CPU-to-GPU transfers.

    Returns per-(layer, kv_head) geometry distortion (Q-weighted with TEST Σ_Q)
    and top-1 retention (against TEST queries).
    """
    n_layers, n_q_heads, L, d = q_dev.shape
    n_kv_heads = k_dev.shape[1]
    group = n_q_heads // n_kv_heads
    sqrt_d = math.sqrt(d)

    geo_per_layer_head = torch.zeros(n_layers, n_kv_heads, device=device)
    top1_per_layer_head = torch.zeros(n_layers, n_kv_heads, device=device)

    if method == "v3":
        compressor = Stage1MSECompressor(d, int(round(b_avg)), seed=seed, device=device)
        for layer in range(n_layers):
            k_layer = k_dev[layer]                                           # (n_kv_heads, L, d)
            q_layer = q_dev[layer]                                           # (n_q_heads, L, d)
            k_hat = compressor.roundtrip(k_layer.unsqueeze(0)).squeeze(0)    # (n_kv_heads, L, d)
            delta = k_layer - k_hat                                          # (n_kv_heads, L, d)
            for kv_head in range(n_kv_heads):
                Sq_h = Sq_test_dev[layer, kv_head]
                d_h = delta[kv_head]
                geo = (d_h @ Sq_h * d_h).sum(dim=-1).mean() / d
                geo_per_layer_head[layer, kv_head] = geo
                gq = q_layer[kv_head * group:(kv_head + 1) * group]
                k_h = k_layer[kv_head]
                k_hat_h = k_hat[kv_head]
                logits_real = (gq @ k_h.T) / sqrt_d
                logits_apx = (gq @ k_hat_h.T) / sqrt_d
                top1_real = logits_real.argmax(dim=-1)
                top1_apx = logits_apx.argmax(dim=-1)
                top1_per_layer_head[layer, kv_head] = (top1_real == top1_apx).float().mean()
        return {"geo": geo_per_layer_head.cpu(), "top1": top1_per_layer_head.cpu()}

    sigma_q_dev = train_calib_dev["sigma_q"]
    sigma_k_dev = train_calib_dev["sigma_k"]
    P_K_dev = train_calib_dev["P_K"]
    P_K_inv_dev = train_calib_dev["P_K_inv"]
    rho_dev = train_calib_dev["rho"]
    mq_eigvals_dev = train_calib_dev["mq_eigvals"]
    mq_eigvecs_dev = train_calib_dev["mq_eigvecs"]
    V_h_dev = train_calib_dev["V_h"]
    R_sym_dev = train_calib_dev["R_sym"]

    for layer in range(n_layers):
        k_layer = k_dev[layer]
        q_layer = q_dev[layer]
        for kv_head in range(n_kv_heads):
            cca = {
                "P_K": P_K_dev[layer, kv_head],
                "P_K_inv": P_K_inv_dev[layer, kv_head],
                "rho": rho_dev[layer, kv_head],
            }
            compressor = build_method_compressor(
                method=method,
                sigma_q_for_head=sigma_q_dev[layer, kv_head],
                sigma_k_for_head=sigma_k_dev[layer, kv_head],
                cca=cca,
                mq_eigvals=mq_eigvals_dev[layer, kv_head],
                mq_eigvecs=mq_eigvecs_dev[layer, kv_head],
                b_avg=float(b_avg),
                r=rank,
                seed=seed,
                head_dim=d,
                V_h=V_h_dev[layer, kv_head],
                R_sym=R_sym_dev[layer, kv_head],
            )
            assert compressor is not None
            compressor = compressor.to(device)
            k_h = k_layer[kv_head]                                # (L, d)
            k_hat = compressor.roundtrip(k_h)
            Sq_h_test = Sq_test_dev[layer, kv_head]
            delta = k_h - k_hat
            geo = (delta @ Sq_h_test * delta).sum(dim=-1).mean() / d
            geo_per_layer_head[layer, kv_head] = geo
            gq = q_layer[kv_head * group:(kv_head + 1) * group]   # (group, L, d)
            logits_real = (gq @ k_h.T) / sqrt_d
            logits_apx = (gq @ k_hat.T) / sqrt_d
            top1_real = logits_real.argmax(dim=-1)
            top1_apx = logits_apx.argmax(dim=-1)
            top1_per_layer_head[layer, kv_head] = (top1_real == top1_apx).float().mean()
            # free intermediate
            del compressor, k_hat, delta, logits_real, logits_apx
    return {"geo": geo_per_layer_head.cpu(), "top1": top1_per_layer_head.cpu()}


# ---------------------------------------------------------------------------
# Main

def main() -> None:
    args = parse_args()

    log_path = Path(args.log)
    out_path = Path(args.output)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log(f"args: {vars(args)}")
    log(f"device={args.device}, torch={torch.__version__}")

    # === Load train moments + R_sym; recompute the other bases from moments ===
    log(f"loading train moments from {args.cca_stats}")
    train_pt = torch.load(args.cca_stats, map_location="cpu", weights_only=False)
    Sq_train = train_pt["sigma_q"]
    Sk_train = train_pt["sigma_k"]
    Cqk_train = train_pt["cqk"]
    R_sym_loaded = train_pt["R_sym"]
    n_layers, n_kv_heads, head_dim, _ = Sq_train.shape
    log(f"  Sq_train shape={tuple(Sq_train.shape)}; train tokens={int(train_pt['total_prefill_tokens'])}")

    # The stored mq_eigvecs / V_h / P_K in this artifact are placeholder identity
    # matrices (verified: max abs diff from I = 0). Only the moments and R_sym
    # are real. Recompute V_Q (mq_eigvecs), V_h, P_K, P_K_inv, P_Q from the
    # moments using the same toolkit functions the runner uses.
    log("  recomputing V_Q, V_h, P_K from moments (stored bases were placeholder identity)")
    eps = 1e-4
    flat_sq = Sq_train.reshape(-1, head_dim, head_dim)
    flat_sk = Sk_train.reshape(-1, head_dim, head_dim)
    flat_cqk = Cqk_train.reshape(-1, head_dim, head_dim)
    cca = compute_cca_basis(flat_sq, flat_sk, flat_cqk, eps=eps)
    P_K = cca["P_K"].view(n_layers, n_kv_heads, head_dim, head_dim)
    P_K_inv = cca["P_K_inv"].view(n_layers, n_kv_heads, head_dim, head_dim)
    P_Q = cca["P_Q"].view(n_layers, n_kv_heads, head_dim, head_dim)
    rho = cca["rho"].view(n_layers, n_kv_heads, head_dim)

    sym_mq = 0.5 * (flat_sq + flat_sq.transpose(-1, -2))
    trace_sq = sym_mq.diagonal(dim1=-2, dim2=-1).sum(-1)
    reg = (eps * (trace_sq / head_dim).clamp_min(1e-12)).unsqueeze(-1).unsqueeze(-1) * torch.eye(head_dim).unsqueeze(0)
    sym_mq_reg = sym_mq + reg
    mq_eigvals_flat, mq_eigvecs_flat = torch.linalg.eigh(sym_mq_reg)
    sort_idx = torch.argsort(mq_eigvals_flat, dim=-1, descending=True)
    mq_eigvals_flat = torch.gather(mq_eigvals_flat, -1, sort_idx)
    mq_eigvecs_flat = torch.gather(mq_eigvecs_flat, -1, sort_idx.unsqueeze(-2).expand(-1, head_dim, -1))
    V_Q = mq_eigvecs_flat.view(n_layers, n_kv_heads, head_dim, head_dim)
    mq_eigvals = mq_eigvals_flat.view(n_layers, n_kv_heads, head_dim)

    V_h_recomputed, R_sym_recomputed = _derive_vh_rsym(Sq_train, Sk_train, P_K, eps=eps)
    V_h = V_h_recomputed
    # Sanity: stored R_sym should match recomputed R_sym (both come from same recipe)
    rsym_diff = (R_sym_loaded - R_sym_recomputed).abs().max().item()
    log(f"    R_sym stored vs recomputed: max abs diff = {rsym_diff:.4e}")
    R_sym = R_sym_recomputed
    log(f"    P_K, V_Q, V_h, R_sym all computed from moments")

    log("  computing V_K (eigvecs of Σ_K)")
    V_K = compute_v_k_basis(Sk_train.double()).float()
    log(f"    V_K shape={tuple(V_K.shape)}")
    log("  generating random orthogonal basis (TurboQuant proxy)")
    R_rand = random_orthogonal((n_layers, n_kv_heads, head_dim, head_dim), seed=args.seed).float()
    log(f"    R_rand shape={tuple(R_rand.shape)}")

    # === Load test moments ===
    bundle = Path(args.bundle)
    agg_path = bundle / "02_stats" / "aggregate.pt"
    log(f"loading aggregate stats from {agg_path}")
    agg = torch.load(agg_path, map_location="cpu", weights_only=False)
    g_test = agg["groups"]["pooled|test"]
    Sq_test = g_test["sigma_q"]
    Sk_test = g_test["sigma_k"]
    test_tokens = int(g_test["tokens"])
    log(f"  pooled test tokens={test_tokens}")

    # Per-task test moments (for cross-task analysis)
    test_tasks = sorted({k.split("|")[0] for k in agg["groups"].keys() if k.endswith("|test") and not k.startswith("pooled")})
    per_task_test = {
        t: {"sigma_q": agg["groups"][f"{t}|test"]["sigma_q"],
            "sigma_k": agg["groups"][f"{t}|test"]["sigma_k"],
            "tokens": int(agg["groups"][f"{t}|test"]["tokens"])}
        for t in test_tasks
    }
    log(f"  test tasks: {test_tasks}")

    results: dict[str, Any] = {
        "calibration_train": {
            "source": train_pt.get("calibration_source", "longbench_compact8_qkv pooled (400 train prompts)"),
            "n_train_prompts": 400,
            "total_train_tokens": int(train_pt["total_prefill_tokens"]),
            "calibration_date": train_pt.get("calibration_date", "unknown"),
        },
        "evaluation_test": {
            "source": "longbench_compact8_qkv pooled|test (80 prompts, 8 LongBench tasks, disjoint from train)",
            "total_test_tokens": test_tokens,
            "tasks": test_tasks,
        },
        "config": {
            "n_layers": int(n_layers),
            "n_kv_heads": int(n_kv_heads),
            "head_dim": int(head_dim),
            "device": args.device,
            "seed": args.seed,
            "rank": args.rank,
            "b_avgs": args.b_avgs,
        },
    }

    # === Phase A1: per-head geomean predictions on TEST moments using TRAIN bases ===
    log("[A1] geomean predictions on test moments using train bases")
    bases = {
        "V_Q": V_Q,
        "V_K": V_K,
        "V_h": V_h,
        "R_sym": R_sym,
        "random_orth_TQ_proxy": R_rand,
    }
    geomean_results: dict[str, dict[str, float]] = {}
    for label, R in bases.items():
        gq, gk, prod = per_head_geomean(R.double(), Sq_test.double(), Sk_test.double())
        geomean_results[label] = {
            "Q_side_geomean__l0excl": float(gq[1:].mean()),
            "K_side_geomean__l0excl": float(gk[1:].mean()),
            "product_geomean__l0excl": float(prod[1:].mean()),
        }
        log(f"  {label}: Q={geomean_results[label]['Q_side_geomean__l0excl']:.4f} "
            f"K={geomean_results[label]['K_side_geomean__l0excl']:.4f} "
            f"prod={geomean_results[label]['product_geomean__l0excl']:.4f}")
    floor = hadamard_floor_per_head(Sq_test.double(), Sk_test.double())
    geomean_results["hadamard_floor"] = {
        "per_head_floor_l0excl_mean": float(floor[1:].mean()),
        "per_head_floor_all_mean": float(floor.mean()),
    }
    log(f"  hadamard floor (per-head, l0excl mean): {geomean_results['hadamard_floor']['per_head_floor_l0excl_mean']:.4f}")
    results["section_10_1_geomean_predictions"] = geomean_results

    # === Headroom (§8.2) ===
    rsym_geo = geomean_results["R_sym"]["product_geomean__l0excl"]
    floor_val = geomean_results["hadamard_floor"]["per_head_floor_l0excl_mean"]
    vq_geo = geomean_results["V_Q"]["product_geomean__l0excl"]
    rrand_geo = geomean_results["random_orth_TQ_proxy"]["product_geomean__l0excl"]
    results["section_8_2_headroom"] = {
        "R_sym_geomean": rsym_geo,
        "V_Q_geomean": vq_geo,
        "hadamard_floor": floor_val,
        "R_sym_over_floor": rsym_geo / floor_val,
        "V_Q_over_floor": vq_geo / floor_val,
    }
    log(f"  R_sym/floor = {rsym_geo/floor_val:.3f}  (was 1.37 on the older bundle)")

    # === Predicted ratios (§10.2 first row) ===
    results["section_10_2_predicted_ratios"] = {
        "R_sym_over_V_Q": rsym_geo / vq_geo,
        "V_Q_over_TQ_proxy": vq_geo / rrand_geo,
        "R_sym_over_TQ_proxy": rsym_geo / rrand_geo,
    }
    log(f"  predicted R_sym/V_Q = {rsym_geo/vq_geo:.4f}")
    log(f"  predicted V_Q/TQ    = {vq_geo/rrand_geo:.4f}")

    # === Phase A2: M-indefinite check on TRAIN moments ===
    log("[A2] M-indefinite check on train Σ_Q, Σ_K")
    results["section_8_M_indefinite"] = m_indefinite_check(Sq_train, Sk_train)
    mi = results["section_8_M_indefinite"]
    log(f"  # heads with negative eigenvalue: {mi['n_heads_with_negative']}/{mi['n_heads_total']}")
    log(f"  min/max ratio range: [{mi['min_max_ratio_min']:.3f}, {mi['min_max_ratio_max']:.3f}]")

    # === Phase A3: active-set sizes for R_sym water-fill on TEST moments ===
    log("[A3] active-set sizes for R_sym water-fill on test moments")
    b_list = [int(b) for b in args.b_avgs.split(",")]
    results["section_5_active_set_R_sym"] = active_set_sizes(R_sym.double(), Sq_test.double(), Sk_test.double(), b_list)
    for b_avg, stats in results["section_5_active_set_R_sym"].items():
        log(f"  b_avg={b_avg}: active min={stats['min']}, median={stats['median']}, max={stats['max']}")

    # === Phase A4: per-task cross-task spread of R_sym geomean ===
    log("[A4] per-task cross-task spread (predicted geomean) for each method")
    cross_task: dict[str, dict[str, float]] = {}
    for label, R in bases.items():
        per_task_geo = []
        for t in test_tasks:
            _, _, prod = per_head_geomean(R.double(), per_task_test[t]["sigma_q"].double(), per_task_test[t]["sigma_k"].double())
            per_task_geo.append(float(prod[1:].mean()))
        cross_task[label] = {
            "per_task_geomeans": dict(zip(test_tasks, per_task_geo)),
            "mean": float(sum(per_task_geo) / len(per_task_geo)),
            "min": float(min(per_task_geo)),
            "max": float(max(per_task_geo)),
            "spread_max_minus_min": float(max(per_task_geo) - min(per_task_geo)),
            "spread_relative": float((max(per_task_geo) - min(per_task_geo)) / (sum(per_task_geo) / len(per_task_geo))),
        }
        log(f"  {label}: spread = {cross_task[label]['spread_relative']*100:.2f}% across {len(test_tasks)} tasks")
    results["section_10_4_cross_task_predicted_spread"] = cross_task

    # === Phase B: real quantization on test files ===
    if not args.skip_quant:
        log(f"[B] real quantization on test files (n={args.n_test_files_quant})")
        per_example_dir = bundle / "01_raw"
        # Pick test files: 2 per task (or as many as available, up to n)
        manifest_path = bundle / "00_split" / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        test_rows = [r for r in manifest["rows"] if r["split"] == "test"]
        per_task: dict[str, list[dict]] = {}
        for r in test_rows:
            per_task.setdefault(r["task"], []).append(r)
        # Round-robin select files until we hit the cap
        chosen: list[dict] = []
        for i in range(max(len(v) for v in per_task.values())):
            for t in sorted(per_task.keys()):
                if i < len(per_task[t]) and len(chosen) < args.n_test_files_quant:
                    chosen.append(per_task[t][i])
            if len(chosen) >= args.n_test_files_quant:
                break
        log(f"  selected test files (one per task in round-robin):")
        for r in chosen:
            log(f"    {r['task']} row{r['row_index']} prompt_len={r['prompt_length']}")

        # Build raw-file path for each chosen row
        # Raw files live under 01_raw/shard_*/longbench__<task>__row{NNNNN}__test.pt
        # Need to find which shard each file is in
        all_raw = list(per_example_dir.rglob("*.pt"))
        raw_index = {p.name: p for p in all_raw}

        b_avg = 3
        rank = args.rank
        methods = [
            "v3",
            "v_truncate", "v_waterfill",
            "cca_uniform", "cca_waterfill",
            "cca_orth_uniform", "cca_orth_waterfill",
            "r_sym_uniform", "r_sym_waterfill",
        ]
        per_method_geo: dict[str, list[torch.Tensor]] = {m: [] for m in methods}
        per_method_top1: dict[str, list[torch.Tensor]] = {m: [] for m in methods}

        log("  pre-moving train calib + test Σ_Q to device")
        train_calib_dev = {
            "sigma_q": Sq_train.to(args.device),
            "sigma_k": Sk_train.to(args.device),
            "P_K": P_K.to(args.device),
            "P_K_inv": P_K_inv.to(args.device),
            "rho": rho.to(args.device),
            "mq_eigvals": mq_eigvals.to(args.device),
            "mq_eigvecs": V_Q.to(args.device),
            "V_h": V_h.to(args.device),
            "R_sym": R_sym.to(args.device),
        }
        Sq_test_dev = Sq_test.to(args.device)

        for i, row in enumerate(chosen):
            fname = f"longbench__{row['task']}__row{row['row_index']:05d}__test.pt"
            if fname not in raw_index:
                log(f"  WARN: {fname} not found, skipping")
                continue
            t0 = time.time()
            log(f"  [{i+1}/{len(chosen)}] loading {fname}")
            payload = torch.load(raw_index[fname], map_location="cpu", weights_only=False)
            L = int(payload["prompt_length"])
            q_dev = payload["q_post"][:, :, :L, :].float().to(args.device)
            k_dev = payload["k_post"][:, :, :L, :].float().to(args.device)
            del payload
            log(f"    payload moved to device, q={tuple(q_dev.shape)} k={tuple(k_dev.shape)}, took {time.time()-t0:.1f}s")
            for m in methods:
                t1 = time.time()
                out = evaluate_method_on_payload(
                    method=m, q_dev=q_dev, k_dev=k_dev,
                    train_calib_dev=train_calib_dev, Sq_test_dev=Sq_test_dev,
                    b_avg=b_avg, rank=rank, seed=args.seed, device=args.device,
                    head_dim=head_dim,
                )
                per_method_geo[m].append(out["geo"])
                per_method_top1[m].append(out["top1"])
                log(f"    {m}: geo_l0excl={out['geo'][1:].mean():.4f} top1_l0excl={out['top1'][1:].mean():.4f} ({time.time()-t1:.1f}s)")
            del q_dev, k_dev
            torch.cuda.empty_cache() if args.device == "cuda" else None
            log(f"  ({time.time()-t0:.1f}s total for this file)")

        # Aggregate
        log("[B] aggregating real-quant results")
        real_quant: dict[str, dict[str, Any]] = {}
        for m in methods:
            if not per_method_geo[m]:
                continue
            geo_stack = torch.stack(per_method_geo[m])     # (n_files, n_layers, n_kv_heads)
            top1_stack = torch.stack(per_method_top1[m])
            real_quant[m] = {
                "geo_l0excl_mean": float(geo_stack[:, 1:, :].mean()),
                "geo_all_mean": float(geo_stack.mean()),
                "top1_l0excl_mean": float(top1_stack[:, 1:, :].mean()),
                "top1_all_mean": float(top1_stack.mean()),
                "per_layer_geo_l0excl_mean": geo_stack.mean(dim=(0, 2))[1:].tolist(),
                "per_layer_top1_l0excl_mean": top1_stack.mean(dim=(0, 2))[1:].tolist(),
                "n_files": geo_stack.shape[0],
            }
            log(f"  {m}: geo={real_quant[m]['geo_l0excl_mean']:.4f} top1={real_quant[m]['top1_l0excl_mean']:.4f} (n={real_quant[m]['n_files']})")

        results["phase_B_real_quantization"] = {
            "b_avg": b_avg,
            "rank": rank,
            "methods": real_quant,
            "n_test_files": len(chosen),
            "test_files_used": [f"{r['task']} row{r['row_index']}" for r in chosen if f"longbench__{r['task']}__row{r['row_index']:05d}__test.pt" in raw_index],
        }

        # === Predicted vs measured (§10.2 second row) ===
        log("[B] computing predicted-vs-measured ratios")
        if "v_waterfill" in real_quant and "r_sym_waterfill" in real_quant:
            measured_rsym_over_v = real_quant["r_sym_waterfill"]["geo_l0excl_mean"] / real_quant["v_waterfill"]["geo_l0excl_mean"]
            results["section_10_2_predicted_vs_measured"] = {
                "R_sym_over_V_Q": {
                    "predicted_geomean": rsym_geo / vq_geo,
                    "measured_geo_distortion": measured_rsym_over_v,
                    "agreement_pct": abs(rsym_geo/vq_geo - measured_rsym_over_v) / (rsym_geo/vq_geo) * 100,
                },
            }
            if "v3" in real_quant:
                v3_geo = real_quant["v3"]["geo_l0excl_mean"]
                vw_geo = real_quant["v_waterfill"]["geo_l0excl_mean"]
                rs_geo = real_quant["r_sym_waterfill"]["geo_l0excl_mean"]
                results["section_10_2_predicted_vs_measured"].update({
                    "V_Q_over_TQ_real": {
                        "predicted_geomean": vq_geo / rrand_geo,
                        "measured_geo_distortion": vw_geo / v3_geo,
                    },
                    "R_sym_over_TQ_real": {
                        "predicted_geomean": rsym_geo / rrand_geo,
                        "measured_geo_distortion": rs_geo / v3_geo,
                    },
                })

        # === Per-layer Pearson (predicted vs measured) ===
        log("[B] computing per-layer Pearson correlation")
        # Predicted per-layer geomean for each method, on test pooled moments
        per_layer_pearson: dict[str, float] = {}
        for m, R in [("v_waterfill", V_Q), ("r_sym_waterfill", R_sym), ("cca_orth_waterfill", V_h)]:
            if m not in real_quant:
                continue
            _, _, prod = per_head_geomean(R.double(), Sq_test.double(), Sk_test.double())
            pred_per_layer = prod.mean(dim=-1)[1:]   # (n_layers-1,) average across kv_heads, l0excl
            measured_per_layer = torch.tensor(real_quant[m]["per_layer_geo_l0excl_mean"])
            # Pearson
            x = pred_per_layer - pred_per_layer.mean()
            y = measured_per_layer - measured_per_layer.mean()
            r = float((x * y).sum() / (x.norm() * y.norm() + 1e-30))
            per_layer_pearson[m] = r
            log(f"  Pearson(pred_geomean, measured_geo) per layer for {m}: {r:.3f}")
        results["section_10_2_per_layer_pearson"] = per_layer_pearson

    # === Save ===
    log(f"writing {out_path}")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    log("done")


if __name__ == "__main__":
    main()
