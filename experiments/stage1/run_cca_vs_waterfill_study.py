"""Stage 1E E3/E4/E5 runner: CCA vs water-filling vs V3 with real per-coord quantization.

Reads pre-computed CCA stats from `artifacts/stage1/cca_vs_waterfill_study/cca_stats.pt`
(produced by run_cca_diagnostics.py), iterates over the 24-example bundle, applies each
method's transform + quantization, and reports geometry distortion / logit MSE / top-1
retention per layer.

Phases:
    e3:  baseline study at given b_avg, methods over (r) grid where applicable.
    e4a: cross-task — calibrate CCA from a single config, evaluate on all 24 examples.
    e4b: within-task LOO — for each (config, held-out example), calibrate from the other 7.
    e5:  decode-phase Q evaluation (in addition to prefill) on the prefill-keys cache.

Progress contract:
    - python -u (unbuffered).
    - Structured progress lines.
    - Heartbeat file under logs/<run_name>.heartbeat (updated every progress tick).
    - <run_name>.summary.json on clean exit; <run_name>.FAILED on exception (last 100 log lines).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.stage1.toolkit import (
    CrossMomentsAccumulator,
    PerCoordCompressor,
    Stage1MSECompressor,
    build_method_compressor,
    compute_attention_metrics,
    compute_cca_basis,
    compute_geometry_distortion,
    ensure_dir,
    save_json,
    split_prefill_and_decode,
)
from experiments.stage1.toolkit.metric_transform import whitening_factor


METHODS_DEFAULT = "v3,v_truncate,v_waterfill,cca_uniform,cca_waterfill"
RANK_DEFAULT = 64
PROGRESS_INTERVAL_SEC_DEFAULT = 30


def _derive_vh_rsym(
    sigma_q: torch.Tensor,
    sigma_k: torch.Tensor,
    P_K: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive V_h (orthogonal CCA basis) and R_sym (joint Q-K eigenbasis) per head.

    V_h = P_K @ Σ_K^(1/2) (orthogonal — the right-singular-vectors of the whitened cross-moment SVD).
    R_sym = eigvec_descending((Σ_Q Σ_K + Σ_K Σ_Q)/2) using regularized inputs.

    sigma_q/sigma_k/P_K: (..., d, d). Returns V_h, R_sym of same leading shape.

    Raises if V_h fails orthogonality at threshold 1e-3 worst-case across heads.
    """
    leading_shape = sigma_q.shape[:-2]
    d = sigma_q.shape[-1]
    flat_sq = sigma_q.reshape(-1, d, d)
    flat_sk = sigma_k.reshape(-1, d, d)
    flat_pk = P_K.reshape(-1, d, d)
    n = flat_sq.shape[0]

    _, W_K_inv = whitening_factor(flat_sk, eps=eps)  # W_K_inv = Σ_K^(1/2)
    V_h_raw = flat_pk @ W_K_inv

    # V_h is orthogonal by construction (right-singular-vectors of W_Q C_QK W_K^T),
    # but P_K @ Σ_K^(1/2) inherits regularization noise from whitening, so orthogonality
    # is only approximate. Polar-orthogonalize: closest orthogonal matrix to V_h_raw is
    # U @ V^T from its SVD. This preserves the basis directions while restoring V_h V_h^T = I.
    raw_err = (V_h_raw @ V_h_raw.transpose(-1, -2) - torch.eye(d, dtype=V_h_raw.dtype, device=V_h_raw.device)).abs().reshape(n, -1).max(dim=-1).values
    worst_raw = float(raw_err.max())
    U_pol, _, Vh_pol = torch.linalg.svd(V_h_raw, full_matrices=False)
    V_h = U_pol @ Vh_pol

    eye = torch.eye(d, dtype=V_h.dtype, device=V_h.device)
    err = (V_h @ V_h.transpose(-1, -2) - eye).abs().reshape(n, -1).max(dim=-1).values
    worst = float(err.max())
    if worst > 1e-3:
        n_bad = int((err > 1e-3).sum())
        raise RuntimeError(
            f"V_h orthogonality check failed after polar fix: {n_bad}/{n} heads with err > 1e-3, "
            f"worst={worst:.4e} (raw worst before polar={worst_raw:.4e})"
        )

    sym_sq = 0.5 * (flat_sq + flat_sq.transpose(-1, -2))
    sym_sk = 0.5 * (flat_sk + flat_sk.transpose(-1, -2))
    trace_sq = sym_sq.diagonal(dim1=-2, dim2=-1).sum(-1)
    trace_sk = sym_sk.diagonal(dim1=-2, dim2=-1).sum(-1)
    eye_d = torch.eye(d, dtype=sym_sq.dtype, device=sym_sq.device)
    reg_q = (eps * (trace_sq / d).clamp_min(1e-12)).unsqueeze(-1).unsqueeze(-1) * eye_d
    reg_k = (eps * (trace_sk / d).clamp_min(1e-12)).unsqueeze(-1).unsqueeze(-1) * eye_d
    sq_reg = sym_sq + reg_q
    sk_reg = sym_sk + reg_k
    qk = sq_reg @ sk_reg
    sym = 0.5 * (qk + qk.transpose(-1, -2))
    eigvals, eigvecs = torch.linalg.eigh(sym)
    sort_idx = torch.argsort(eigvals, dim=-1, descending=True)
    R_sym = torch.gather(eigvecs, -1, sort_idx.unsqueeze(-2).expand(-1, d, -1))

    err_r = (R_sym @ R_sym.transpose(-1, -2) - eye).abs().reshape(n, -1).max(dim=-1).values
    worst_r = float(err_r.max())
    if worst_r > 1e-3:
        n_bad = int((err_r > 1e-3).sum())
        raise RuntimeError(
            f"R_sym orthogonality check failed: {n_bad}/{n} heads with err > 1e-3, worst={worst_r:.4e}"
        )

    return V_h.reshape(leading_shape + (d, d)), R_sym.reshape(leading_shape + (d, d))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1E real-quantization runner (E3/E4/E5).")
    parser.add_argument("--bundle", type=str, default="artifacts/stage1/query_stats_longbench_under4k")
    parser.add_argument(
        "--cca-stats",
        type=str,
        default="artifacts/stage1/cca_vs_waterfill_study/cca_stats.pt",
        help="Pre-computed Σ_Q, Σ_K, C_QK, P_K, ρ etc. from run_cca_diagnostics.",
    )
    parser.add_argument("--output-subdir", type=str, default="e3")
    parser.add_argument("--run-name", type=str, default=None, help="Logs/heartbeat key. Defaults to phase+suffix.")
    parser.add_argument("--phase", choices=["e3", "e4a", "e4b", "e5"], default="e3")
    parser.add_argument("--b-avg", type=float, default=3.0)
    parser.add_argument("--rank", type=int, default=RANK_DEFAULT)
    parser.add_argument("--methods", type=str, default=METHODS_DEFAULT)
    parser.add_argument(
        "--query-phase",
        choices=["prefill", "both"],
        default="prefill",
        help="Which Q to evaluate against the compressed prefill K. 'both' covers decode-phase Q "
        "in addition to prefill (standalone 'decode' is unsupported because the prefill metrics "
        "would be empty/NaN).",
    )
    parser.add_argument("--calibration-config", type=str, default=None,
                        help="For e4a: restrict CCA calibration to examples with this config.")
    parser.add_argument("--loo-index", type=int, default=None,
                        help="For e4b: hold out a single example index from CCA calibration.")
    parser.add_argument("--loo-config", type=str, default=None, help="For e4b: config of the held-out example.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-examples", type=int, default=None,
                        help="If set, only process the first N examples (for smoke tests).")
    parser.add_argument("--limit-layers", type=int, default=None,
                        help="If set, only process the first N layers (for smoke tests).")
    parser.add_argument("--progress-interval-sec", type=float, default=PROGRESS_INTERVAL_SEC_DEFAULT)
    parser.add_argument("--full-precision-smoke-test", action="store_true",
                        help="If set, also run a 16-bit smoke test on first example/layer for the gate.")
    parser.add_argument("--eps", type=float, default=1e-4)
    return parser.parse_args()


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str, run_name: str, log_path: Path) -> None:
    line = f"[{_ts()}] {msg}"
    print(line, flush=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")


def _heartbeat(heartbeat_path: Path) -> None:
    heartbeat_path.touch()


def _read_manifest(bundle_dir: Path) -> list[dict]:
    with open(bundle_dir / "manifest.json") as f:
        m = json.load(f)
    return m["examples"]


def _example_path(bundle_dir: Path, ex: dict) -> Path:
    return bundle_dir / ex["file"]


def _select_examples_by_config(examples: list[dict], config: str | None) -> list[int]:
    if config is None:
        return list(range(len(examples)))
    indices = []
    for i, ex in enumerate(examples):
        if ex["config"] == config or ex["config"] == config + "_e":
            indices.append(i)
    return indices


def _accumulate_calibration_stats(
    bundle_dir: Path,
    examples: list[dict],
    indices: list[int],
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    device: str,
    log_fn,
    heartbeat_path: Path | None = None,
) -> dict[str, torch.Tensor]:
    """Recompute Σ_Q (GQA-pooled), Σ_K, C_QK from the given subset of examples (prefill positions).
    Uses GPU if device is "cuda" — much faster than CPU for the einsum and outer-product accumulation.
    Returns CPU tensors of shape (n_layers, n_kv_heads, d, d) each.
    """
    accum_device = device if device == "cuda" else "cpu"
    sq_acc = torch.zeros(n_layers, n_kv_heads, head_dim, head_dim, device=accum_device)
    sk_acc = torch.zeros(n_layers, n_kv_heads, head_dim, head_dim, device=accum_device)
    cqk_acc = torch.zeros(n_layers, n_kv_heads, head_dim, head_dim, device=accum_device)
    total_tokens = 0
    group: int | None = None
    for k, idx in enumerate(indices):
        ex = examples[idx]
        path = _example_path(bundle_dir, ex)
        t0 = time.time()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        L = int(payload["prompt_length"])
        q = payload["q_post"][:, :, :L, :].to(accum_device).float()
        k_pre = payload["k_post"][:, :, :L, :].to(accum_device).float()
        n_q_heads = q.shape[1]
        if n_q_heads % n_kv_heads != 0:
            raise ValueError(f"GQA mismatch at example {ex['file']}")
        ex_group = n_q_heads // n_kv_heads
        if group is None:
            group = ex_group
        elif group != ex_group:
            raise ValueError(f"Inconsistent GQA group size: {group} vs {ex_group} at {ex['file']}")
        # Σ_Q convention "treat each query head as a sample": matches E1's load_pooled_stats and
        # E3's global cca_stats, so cross-task / LOO calibrations are comparable to in-domain.
        # Equivalent to mean_g E[q_g q_g^T]; we accumulate the sum and divide by (group * total_tokens)
        # at finalization.
        q_per_head_outer = torch.einsum("lhsd,lhse->lhde", q, q)  # (n_layers, n_q_heads, d, d)
        sq_acc += q_per_head_outer.view(n_layers, n_kv_heads, group, head_dim, head_dim).sum(dim=2)
        # C_QK is linear in q so GQA-pooling before the outer is fine (matches CrossMomentsAccumulator).
        q_grouped = q.view(n_layers, n_kv_heads, group, L, head_dim).mean(dim=2)
        sk_acc += torch.einsum("lhsd,lhse->lhde", k_pre, k_pre)
        cqk_acc += torch.einsum("lhsd,lhse->lhde", q_grouped, k_pre)
        total_tokens += L
        del payload, q, k_pre, q_grouped, q_per_head_outer
        log_fn(f"  calib accum {k+1}/{len(indices)} ({path.name}) prefill_len={L} in {time.time()-t0:.1f}s")
        if heartbeat_path is not None:
            heartbeat_path.touch()
    if total_tokens == 0 or group is None:
        raise RuntimeError("no calibration tokens accumulated")
    sigma_q = (sq_acc / (group * total_tokens)).cpu()
    sigma_k = (sk_acc / total_tokens).cpu()
    cqk = (cqk_acc / total_tokens).cpu()
    return {"sigma_q": sigma_q, "sigma_k": sigma_k, "cqk": cqk, "total_tokens": total_tokens}


def build_per_head_calibration(
    sigma_q: torch.Tensor,
    sigma_k: torch.Tensor,
    cqk: torch.Tensor,
    eps: float,
    compute_newbases: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute CCA + V eigendecomposition for each (layer, kv_head). Returns per-head dicts.

    Inputs shape: (n_layers, n_kv_heads, d, d) each.
    Returns:
        rho: (n_layers, n_kv_heads, d)
        P_K, P_K_inv, P_Q: (n_layers, n_kv_heads, d, d)
        mq_eigvals: (n_layers, n_kv_heads, d)
        mq_eigvecs: (n_layers, n_kv_heads, d, d)
        Mq: (n_layers, n_kv_heads, d, d)  -- the per-head Q second-moment for geometry distortion.
    """
    n_layers, n_kv_heads, d, _ = sigma_q.shape
    flat_sq = sigma_q.reshape(-1, d, d)
    flat_sk = sigma_k.reshape(-1, d, d)
    flat_cqk = cqk.reshape(-1, d, d)
    cca = compute_cca_basis(flat_sq, flat_sk, flat_cqk, eps=eps)

    sym_mq = 0.5 * (flat_sq + flat_sq.transpose(-1, -2))
    trace_sq = sym_mq.diagonal(dim1=-2, dim2=-1).sum(-1)
    reg = (eps * (trace_sq / d).clamp_min(1e-12)).unsqueeze(-1).unsqueeze(-1) * torch.eye(d).unsqueeze(0)
    sym_mq_reg = sym_mq + reg
    mq_eigvals, mq_eigvecs = torch.linalg.eigh(sym_mq_reg)
    sort_idx = torch.argsort(mq_eigvals, dim=-1, descending=True)
    mq_eigvals = torch.gather(mq_eigvals, -1, sort_idx)
    mq_eigvecs = torch.gather(mq_eigvecs, -1, sort_idx.unsqueeze(-2).expand(-1, d, -1))

    P_K_view = cca["P_K"].view(n_layers, n_kv_heads, d, d)
    out = {
        "rho": cca["rho"].view(n_layers, n_kv_heads, d),
        "P_K": P_K_view,
        "P_K_inv": cca["P_K_inv"].view(n_layers, n_kv_heads, d, d),
        "P_Q": cca["P_Q"].view(n_layers, n_kv_heads, d, d),
        "mq_eigvals": mq_eigvals.view(n_layers, n_kv_heads, d),
        "mq_eigvecs": mq_eigvecs.view(n_layers, n_kv_heads, d, d),
        "Mq": sym_mq.view(n_layers, n_kv_heads, d, d),
        "sigma_k": sigma_k,
    }
    if compute_newbases:
        V_h, R_sym = _derive_vh_rsym(sigma_q, sigma_k, P_K_view, eps=eps)
        out["V_h"] = V_h
        out["R_sym"] = R_sym
    return out


def evaluate_method_on_example(
    method: str,
    layer: int,
    kv_head: int,
    keys: torch.Tensor,
    queries_prefill: torch.Tensor,
    queries_decode: torch.Tensor | None,
    calibration: dict[str, torch.Tensor],
    bits_avg: float,
    rank: int,
    seed: int,
    head_dim: int,
    device: str,
) -> dict[str, float]:
    """Compress prefill keys for a single (layer, kv_head) using `method`; evaluate metrics.

    Inputs:
        keys: (1, 1, L, d) prefill keys for this (layer, kv_head).
        queries_prefill: (1, n_q_per_kv, L, d) prefill queries belonging to this kv_head's group.
        queries_decode:  (1, n_q_per_kv, L_decode, d) or None — decode-phase queries.

    Returns metrics dict with keys: geometry_distortion, logit_mse_prefill, top1_prefill,
    top5_prefill, and (if decode) logit_mse_decode, top1_decode, top5_decode.
    """
    Mq = calibration["Mq"][layer, kv_head:kv_head + 1].to(device)
    metrics: dict[str, float] = {}

    if method == "v3":
        compressor = Stage1MSECompressor(head_dim, int(round(bits_avg)), seed=seed, device=device)
        keys_recon = compressor.roundtrip(keys.to(device))
    else:
        V_h = calibration.get("V_h")
        R_sym = calibration.get("R_sym")
        per_head_compressor = build_method_compressor(
            method=method,
            sigma_q_for_head=calibration["Mq"][layer, kv_head].to(device),
            sigma_k_for_head=calibration["sigma_k"][layer, kv_head].to(device),
            cca={
                "P_K": calibration["P_K"][layer, kv_head].to(device),
                "P_K_inv": calibration["P_K_inv"][layer, kv_head].to(device),
                "rho": calibration["rho"][layer, kv_head].to(device),
            },
            mq_eigvals=calibration["mq_eigvals"][layer, kv_head].to(device),
            mq_eigvecs=calibration["mq_eigvecs"][layer, kv_head].to(device),
            b_avg=bits_avg,
            r=rank,
            seed=seed,
            head_dim=head_dim,
            V_h=V_h[layer, kv_head].to(device) if V_h is not None else None,
            R_sym=R_sym[layer, kv_head].to(device) if R_sym is not None else None,
        )
        keys_recon = per_head_compressor.roundtrip(keys.to(device))

    metrics["geometry_distortion"] = compute_geometry_distortion(keys_recon, keys.to(device), Mq)

    if queries_prefill.shape[1] > 0 and queries_prefill.shape[2] > 0:
        prefill_metrics = compute_attention_metrics(
            queries_prefill.to(device), keys.to(device), keys_recon
        )
        metrics["logit_mse_prefill"] = prefill_metrics["logit_mse"]
        metrics["top1_prefill"] = prefill_metrics["top1_match"]
        metrics["top5_prefill"] = prefill_metrics["top5_containment"]
        metrics["logit_cosine_prefill"] = prefill_metrics["logit_cosine"]

    if queries_decode is not None and queries_decode.shape[2] > 0:
        decode_metrics = compute_attention_metrics(
            queries_decode.to(device), keys.to(device), keys_recon
        )
        metrics["logit_mse_decode"] = decode_metrics["logit_mse"]
        metrics["top1_decode"] = decode_metrics["top1_match"]
        metrics["top5_decode"] = decode_metrics["top5_containment"]
        metrics["logit_cosine_decode"] = decode_metrics["logit_cosine"]
        metrics["decode_query_count"] = float(queries_decode.shape[2])

    return metrics


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    bundle_dir = (repo_root / args.bundle).resolve()
    cca_stats_path = (repo_root / args.cca_stats).resolve()
    out_dir = (repo_root / "artifacts/stage1/cca_vs_waterfill_study" / args.output_subdir).resolve()
    ensure_dir(out_dir)

    run_name = args.run_name or f"{args.phase}_b{args.b_avg}_r{args.rank}"
    logs_dir = repo_root / "experiments/stage1/logs"
    ensure_dir(logs_dir)
    log_path = logs_dir / f"{run_name}.log"
    heartbeat_path = logs_dir / f"{run_name}.heartbeat"
    summary_path = logs_dir / f"{run_name}.summary.json"
    failed_path = logs_dir / f"{run_name}.FAILED"
    summary_path.unlink(missing_ok=True)
    failed_path.unlink(missing_ok=True)
    log_path.write_text("")

    def log_fn(msg: str) -> None:
        _log(msg, run_name, log_path)

    try:
        log_fn(
            f"Stage 1E run {run_name}: phase={args.phase} b_avg={args.b_avg} rank={args.rank} "
            f"methods={args.methods} query_phase={args.query_phase} device={args.device}"
        )
        _heartbeat(heartbeat_path)

        # --- load CCA stats baseline ---
        if not cca_stats_path.exists():
            raise FileNotFoundError(f"cca_stats not found at {cca_stats_path}; run run_cca_diagnostics first.")
        log_fn(f"Loading CCA stats from {cca_stats_path}")
        cca_stats = torch.load(cca_stats_path, map_location="cpu", weights_only=False)
        n_layers = int(cca_stats["n_layers"])
        n_kv_heads = int(cca_stats["n_kv_heads"])
        head_dim = int(cca_stats["head_dim"])
        if args.limit_layers is not None:
            n_layers_eff = min(args.limit_layers, n_layers)
        else:
            n_layers_eff = n_layers
        log_fn(f"  n_layers={n_layers} (effective {n_layers_eff}) n_kv_heads={n_kv_heads} head_dim={head_dim}")

        # --- determine calibration source for the run ---
        manifest_examples = _read_manifest(bundle_dir)

        # Determine if any requested method needs the new bases (V_h / R_sym).
        _requested_methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        _need_newbases = any(m.startswith(("cca_orth_", "r_sym_")) for m in _requested_methods)

        if args.phase == "e4a":
            if args.calibration_config is None:
                raise ValueError("e4a requires --calibration-config")
            calib_indices = _select_examples_by_config(manifest_examples, args.calibration_config)
            if not calib_indices:
                raise ValueError(f"no examples for config={args.calibration_config}")
            log_fn(f"e4a: calibrating from {len(calib_indices)} examples of config={args.calibration_config}")
            t0 = time.time()
            calib_raw = _accumulate_calibration_stats(
                bundle_dir, manifest_examples, calib_indices, n_layers, n_kv_heads, head_dim, args.device, log_fn, heartbeat_path
            )
            calibration = build_per_head_calibration(
                calib_raw["sigma_q"], calib_raw["sigma_k"], calib_raw["cqk"], eps=args.eps,
                compute_newbases=_need_newbases,
            )
            log_fn(f"  calibration built in {time.time()-t0:.1f}s (newbases={_need_newbases})")
            _heartbeat(heartbeat_path)
        elif args.phase == "e4b":
            if args.loo_index is None:
                raise ValueError("e4b requires --loo-index")
            if args.loo_config is None:
                raise ValueError("e4b requires --loo-config")
            config_indices = _select_examples_by_config(manifest_examples, args.loo_config)
            if not config_indices:
                raise ValueError(f"no examples for config={args.loo_config}")
            if args.loo_index not in config_indices:
                raise ValueError(
                    f"loo-index {args.loo_index} not in config {args.loo_config} (indices={config_indices})"
                )
            calib_indices = [i for i in config_indices if i != args.loo_index]
            log_fn(
                f"e4b: holding out example {args.loo_index} (config={args.loo_config}); "
                f"calibrating from {len(calib_indices)} examples"
            )
            t0 = time.time()
            calib_raw = _accumulate_calibration_stats(
                bundle_dir, manifest_examples, calib_indices, n_layers, n_kv_heads, head_dim, args.device, log_fn, heartbeat_path
            )
            calibration = build_per_head_calibration(
                calib_raw["sigma_q"], calib_raw["sigma_k"], calib_raw["cqk"], eps=args.eps,
                compute_newbases=_need_newbases,
            )
            log_fn(f"  calibration built in {time.time()-t0:.1f}s (newbases={_need_newbases})")
            _heartbeat(heartbeat_path)
        else:
            # e3 / e5 use the global cca_stats; derive V_h / R_sym lazily only if needed.
            calibration = {
                "rho": cca_stats["rho"],
                "P_K": cca_stats["P_K"],
                "P_K_inv": cca_stats["P_K_inv"],
                "P_Q": cca_stats["P_Q"],
                "mq_eigvals": cca_stats["mq_eigvals"],
                "mq_eigvecs": cca_stats["mq_eigvecs"],
                "Mq": cca_stats["sigma_q"],
                "sigma_k": cca_stats["sigma_k"],
            }
            if _need_newbases:
                t0 = time.time()
                V_h, R_sym = _derive_vh_rsym(
                    cca_stats["sigma_q"],
                    cca_stats["sigma_k"],
                    cca_stats["P_K"],
                    eps=args.eps,
                )
                calibration["V_h"] = V_h
                calibration["R_sym"] = R_sym
                log_fn(
                    f"e3/e5: using global CCA stats from run_cca_diagnostics; "
                    f"derived V_h, R_sym in {time.time()-t0:.1f}s"
                )
            else:
                log_fn("e3/e5: using global CCA stats from run_cca_diagnostics")

        # --- determine evaluation set ---
        if args.phase == "e4b":
            eval_indices = [args.loo_index]
        else:
            eval_indices = list(range(len(manifest_examples)))
        if args.limit_examples is not None:
            eval_indices = eval_indices[: args.limit_examples]

        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        log_fn(f"Evaluating {len(eval_indices)} examples × {n_layers_eff} layers × {n_kv_heads} kv_heads × {len(methods)} methods")

        results: list[dict] = []
        last_progress_t = time.time()
        total_units = len(eval_indices) * n_layers_eff * n_kv_heads * len(methods)
        unit_count = 0
        run_start = time.time()

        for ex_pos, ex_idx in enumerate(eval_indices):
            ex = manifest_examples[ex_idx]
            ex_path = _example_path(bundle_dir, ex)
            t0 = time.time()
            payload = torch.load(ex_path, map_location="cpu", weights_only=False)
            prompt_length = int(payload["prompt_length"])
            total_length = int(payload["total_length"])
            captured_length = int(payload["captured_length"])
            actual_decode_count = max(0, captured_length - prompt_length)
            q_post = payload["q_post"]
            k_post = payload["k_post"]
            n_q_heads = q_post.shape[1]
            group = n_q_heads // n_kv_heads
            log_fn(
                f"example {ex_pos+1}/{len(eval_indices)} ({ex['file']}) config={ex['config']} "
                f"prompt_len={prompt_length} total_len={total_length} decode_q_count={actual_decode_count} "
                f"loaded in {time.time()-t0:.1f}s"
            )
            _heartbeat(heartbeat_path)

            for layer in range(n_layers_eff):
                # Slice prefill keys, prefill queries (per kv-head), decode queries (if applicable).
                k_layer = k_post[layer].unsqueeze(0)  # (1, n_kv_heads, T, d)
                q_layer = q_post[layer].unsqueeze(0)  # (1, n_q_heads, T, d)
                prefill_q, decode_q, prefill_k = split_prefill_and_decode(
                    q_layer, k_layer, prompt_length=prompt_length
                )
                prefill_q_grouped = prefill_q.view(1, n_kv_heads, group, prompt_length, head_dim)
                decode_q_grouped = (
                    decode_q.view(1, n_kv_heads, group, decode_q.shape[2], head_dim)
                    if decode_q.shape[2] > 0
                    else None
                )

                for kv_head in range(n_kv_heads):
                    keys_h = prefill_k[:, kv_head:kv_head + 1].float()  # (1, 1, L, d)
                    qpref_h = prefill_q_grouped[0, kv_head].unsqueeze(0)  # (1, group, L, d)
                    qdec_h = (
                        decode_q_grouped[0, kv_head].unsqueeze(0)
                        if decode_q_grouped is not None and (
                            args.query_phase in {"decode", "both"}
                        )
                        else None
                    )
                    for method in methods:
                        try:
                            m = evaluate_method_on_example(
                                method=method,
                                layer=layer,
                                kv_head=kv_head,
                                keys=keys_h,
                                queries_prefill=qpref_h if args.query_phase in {"prefill", "both"} else qpref_h[:, :0],
                                queries_decode=qdec_h,
                                calibration=calibration,
                                bits_avg=args.b_avg,
                                rank=args.rank,
                                seed=args.seed,
                                head_dim=head_dim,
                                device=args.device,
                            )
                        except Exception as exc:
                            raise RuntimeError(
                                f"method {method} failed at example {ex_idx}, layer {layer}, kv_head {kv_head}: {exc}"
                            ) from exc
                        m.update(
                            {
                                "example_index": ex_idx,
                                "example_file": ex["file"],
                                "config": ex["config"],
                                "layer": layer,
                                "kv_head": kv_head,
                                "method": method,
                                "b_avg": args.b_avg,
                                "rank": args.rank,
                                "prompt_length": prompt_length,
                                "decode_query_count": actual_decode_count,
                            }
                        )
                        results.append(m)
                        unit_count += 1
                        now = time.time()
                        if now - last_progress_t >= args.progress_interval_sec or unit_count == total_units:
                            elapsed = now - run_start
                            eta = (
                                elapsed * (total_units - unit_count) / max(1, unit_count)
                                if unit_count > 0
                                else 0.0
                            )
                            log_fn(
                                f"progress: ex={ex_pos+1}/{len(eval_indices)} layer={layer+1}/{n_layers_eff} "
                                f"head={kv_head+1}/{n_kv_heads} method={method} unit={unit_count}/{total_units} "
                                f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m"
                            )
                            _heartbeat(heartbeat_path)
                            last_progress_t = now

            del payload, q_post, k_post

        # --- aggregate ---
        log_fn(f"Aggregating {len(results)} result rows...")
        aggregated = aggregate_results(results, methods, n_layers_eff)
        all_summary = {
            "run_name": run_name,
            "phase": args.phase,
            "b_avg": args.b_avg,
            "rank": args.rank,
            "methods": methods,
            "query_phase": args.query_phase,
            "calibration_config": args.calibration_config,
            "loo_index": args.loo_index,
            "loo_config": args.loo_config,
            "n_examples_evaluated": len(eval_indices),
            "n_layers_eff": n_layers_eff,
            "n_kv_heads": n_kv_heads,
            "head_dim": head_dim,
            "wallclock_sec": time.time() - run_start,
            "device": args.device,
            "aggregated": aggregated,
        }

        # save raw results for downstream gates
        torch.save({"rows": results}, out_dir / f"{run_name}_rows.pt")
        save_json(out_dir / f"{run_name}_summary.json", all_summary)
        save_json(summary_path, all_summary)
        log_fn(f"Run done in {time.time()-run_start:.1f}s. Wrote {out_dir / (run_name + '_summary.json')}")
        _heartbeat(heartbeat_path)

        # smoke test (high bits, effectively full precision) gate hint
        if args.full_precision_smoke_test:
            smoke_bits = 8.0  # 256 levels/coord → quasi-full-precision; avoids V3 OOM at b=16.
            log_fn(f"Running quasi-full-precision (b={smoke_bits}) smoke test on first example, layer 0, kv_head 0, all methods...")
            ex_idx = eval_indices[0]
            ex = manifest_examples[ex_idx]
            payload = torch.load(_example_path(bundle_dir, ex), map_location="cpu", weights_only=False)
            prompt_length = int(payload["prompt_length"])
            n_q = payload["q_post"].shape[1]
            group = n_q // n_kv_heads
            keys = payload["k_post"][0, 0:1, :prompt_length].unsqueeze(0).float()  # (1, 1, L, d)
            qpref = payload["q_post"][0, :group, :prompt_length].unsqueeze(0).float()  # (1, group, L, d)
            smoke = {}
            for method in methods:
                smoke_m = evaluate_method_on_example(
                    method=method, layer=0, kv_head=0,
                    keys=keys, queries_prefill=qpref, queries_decode=None,
                    calibration=calibration, bits_avg=smoke_bits, rank=head_dim,
                    seed=args.seed, head_dim=head_dim, device=args.device,
                )
                smoke[method] = smoke_m
            save_json(out_dir / f"{run_name}_smoke_b16.json", smoke)
            log_fn(f"Smoke test results: {smoke}")

        return 0
    except Exception:
        tb = traceback.format_exc()
        try:
            with open(failed_path, "w") as f:
                f.write(tb)
                if log_path.exists():
                    f.write("\n--- last 100 log lines ---\n")
                    lines = log_path.read_text().splitlines()
                    f.write("\n".join(lines[-100:]))
        finally:
            print(tb, file=sys.stderr, flush=True)
        return 1


def aggregate_results(rows: list[dict], methods: list[str], n_layers: int) -> dict:
    """Aggregate per-method, per-layer summaries with full-data and layer-0-excluded variants."""
    out: dict = {}
    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)

    metric_keys = [
        k
        for k in (rows[0].keys() if rows else [])
        if k not in {"example_index", "example_file", "config", "layer", "kv_head", "method", "b_avg", "rank", "prompt_length", "decode_query_count"}
        and isinstance(rows[0].get(k), (int, float))
    ]

    for method, mrows in by_method.items():
        # Per-layer aggregation
        per_layer = defaultdict(lambda: defaultdict(list))
        for r in mrows:
            for k in metric_keys:
                if k in r:
                    per_layer[k][r["layer"]].append(r[k])
        method_summary: dict = {}
        for k in metric_keys:
            layer_means = []
            for layer in range(n_layers):
                vals = per_layer[k].get(layer, [])
                if vals:
                    layer_means.append(float(sum(vals) / len(vals)))
                else:
                    layer_means.append(float("nan"))
            method_summary[k] = {
                "per_layer": layer_means,
                "all_mean": float(sum(layer_means) / max(1, len(layer_means))) if layer_means else float("nan"),
                "l0excl_mean": float(sum(layer_means[1:]) / max(1, len(layer_means[1:]))) if len(layer_means) > 1 else float("nan"),
            }
        # Bootstrap CI over examples for the headline metric (top1_prefill if available, else logit_mse_prefill).
        if metric_keys:
            headline_key = "top1_prefill" if "top1_prefill" in metric_keys else metric_keys[0]
            ex_means: list[float] = []
            per_ex = defaultdict(list)
            for r in mrows:
                if headline_key in r:
                    per_ex[r["example_index"]].append(r[headline_key])
            for ex_idx, vals in per_ex.items():
                if vals:
                    ex_means.append(float(sum(vals) / len(vals)))
            if ex_means and len(ex_means) >= 4:
                import numpy as np
                arr = np.array(ex_means)
                B = 1000
                rng = np.random.default_rng(0)
                samples = rng.choice(arr, size=(B, len(arr)), replace=True).mean(axis=1)
                ci = np.quantile(samples, [0.025, 0.975]).tolist()
                method_summary["bootstrap_ci"] = {
                    "metric": headline_key,
                    "lo95": float(ci[0]),
                    "hi95": float(ci[1]),
                    "mean": float(arr.mean()),
                }
        out[method] = method_summary
    return out


if __name__ == "__main__":
    sys.exit(main())
