"""Stage 1E E1-2: Q/K distributional diagnostics across phases and tasks.

Standalone, post-hoc. Reads existing per-example payloads + cca_stats.pt + e3/e4a summaries.
Does NOT trigger any rerun of E1-E5.

Σ_Q convention (CRITICAL — see fixes_to_apply.md F1):
    This module uses the canonical "treat-each-query-head-as-sample" formula, matching E1's
    load_pooled_stats and QueryMomentsAccumulator:

        Σ_Q[kv_head]
            = (1 / (group * N))  Σ_t  Σ_{g in this kv_head's group}  q_g(t)  q_g(t)^T
            = mean_g  E[q_g q_g^T]

    NOT the average-then-outer formula (E[(mean_g q_g)(mean_g q_g)^T]) used in E4's old
    _accumulate_calibration_stats. The two differ by within-group cross terms; using the
    wrong one would make E1-2 not directly comparable to E1's cca_stats.pt (which the gate
    explicitly verifies).

Outputs:
    artifacts/stage1/cca_vs_waterfill_study/distribution_diagnostics/
        distribution_stats.pt   — per (task, phase, layer, kv_head) Σ + eigendecomp + per-task CCA
        metrics_e1_2.json       — gate-readable summary

Run:
    python -u -m experiments.stage1.run_distribution_diagnostics
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.stage1.toolkit import (
    compute_cca_basis,
    ensure_dir,
    save_json,
    whitening_factor,
)


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1E E1-2 distributional diagnostics.")
    parser.add_argument("--bundle", type=str, default="artifacts/stage1/query_stats_longbench_under4k")
    parser.add_argument(
        "--cca-stats",
        type=str,
        default="artifacts/stage1/cca_vs_waterfill_study/cca_stats.pt",
        help="E1 output (used for the regression test).",
    )
    parser.add_argument("--output", type=str, default="artifacts/stage1/cca_vs_waterfill_study/distribution_diagnostics")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eps", type=float, default=1e-4, help="Tikhonov regularization for whitening / CCA.")
    parser.add_argument("--low-confidence-threshold", type=int, default=50,
                        help="Decode-token count below which a (task, decode) entry is flagged low-confidence.")
    parser.add_argument("--limit-examples", type=int, default=None, help="If set, only process the first N examples (smoke test).")
    return parser.parse_args()


def _select_configs(manifest_examples: list[dict]) -> list[str]:
    return sorted({ex["config"] for ex in manifest_examples})


def _accumulate_per_task_phase(
    bundle_dir: Path,
    examples: list[dict],
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    accum_device: str,
    log_fn,
) -> dict:
    """Single-pass loop over per-example payloads.

    Accumulates per (task, phase, layer, kv_head):
        sum_q_outer[task, phase] : (n_layers, n_kv_heads, d, d) — sum over (group, t) of q_g(t) q_g(t)^T
        sum_qk[task, phase]      : (n_layers, n_kv_heads, d, d) — sum over t of q_grouped(t) k(t)^T
                                                                  (q_grouped = mean over group; C_QK is linear in q
                                                                   so this matches the canonical convention).
        sum_k_outer[task]        : (n_layers, n_kv_heads, d, d) — sum over t of k(t) k(t)^T (prefill only;
                                                                  decode keys aren't used since we only compress prefill).
        n_tokens[task, phase]    : int — total positions counted (denominator for Σ_Q after dividing by group).
        n_pos_K[task]            : int — positions for K accumulator (= prefill positions).
    """
    configs = _select_configs(examples)
    phases = ("prefill", "decode")

    sum_q_outer = {(c, p): torch.zeros(n_layers, n_kv_heads, head_dim, head_dim, device=accum_device) for c in configs for p in phases}
    sum_qk      = {(c, p): torch.zeros(n_layers, n_kv_heads, head_dim, head_dim, device=accum_device) for c in configs for p in phases}
    sum_k_outer = {c:      torch.zeros(n_layers, n_kv_heads, head_dim, head_dim, device=accum_device) for c in configs}
    n_tokens    = {(c, p): 0 for c in configs for p in phases}
    n_pos_K     = {c: 0 for c in configs}

    group: int | None = None
    for ex_idx, ex in enumerate(examples):
        ex_path = bundle_dir / ex["file"]
        cfg = ex["config"]
        prompt_length = int(ex["prompt_length"])
        captured_length = int(ex["captured_length"])
        t0 = time.time()
        payload = torch.load(ex_path, map_location="cpu", weights_only=False)
        q_post = payload["q_post"]  # (n_layers, n_q_heads, T, d)
        k_post = payload["k_post"]  # (n_layers, n_kv_heads, T, d)
        T = q_post.shape[2]
        if T < prompt_length:
            raise RuntimeError(f"{ex_path}: captured_length={T} < prompt_length={prompt_length}")
        n_q_heads = q_post.shape[1]
        if n_q_heads % n_kv_heads != 0:
            raise ValueError(f"GQA mismatch at {ex_path}: {n_q_heads} q heads, {n_kv_heads} kv heads")
        ex_group = n_q_heads // n_kv_heads
        if group is None:
            group = ex_group
        elif group != ex_group:
            raise ValueError(f"Inconsistent GQA group size at {ex_path}: {group} vs {ex_group}")

        # Prefill slice
        q_pre = q_post[:, :, :prompt_length, :].to(accum_device).float()  # (n_layers, n_q_heads, L, d)
        k_pre = k_post[:, :, :prompt_length, :].to(accum_device).float()  # (n_layers, n_kv_heads, L, d)

        # Σ_Q (per-head outer products, then sum across the kv-head's group of query heads).
        # This is the F1-correct convention: equivalent to mean_g E[q_g q_g^T] after dividing
        # by (group * total tokens) at finalization.
        q_pre_per_head_outer = torch.einsum("lhsd,lhse->lhde", q_pre, q_pre)  # (n_layers, n_q_heads, d, d)
        q_pre_summed_over_group = (
            q_pre_per_head_outer
            .view(n_layers, n_kv_heads, group, head_dim, head_dim)
            .sum(dim=2)
        )
        sum_q_outer[(cfg, "prefill")] += q_pre_summed_over_group

        # Σ_K accumulator (prefill keys only; one per kv head, no GQA pooling).
        sum_k_outer[cfg] += torch.einsum("lhsd,lhse->lhde", k_pre, k_pre)
        n_pos_K[cfg] += prompt_length

        # C_QK (linear in q so GQA-pool-then-outer is equivalent; matches CrossMomentsAccumulator).
        q_pre_grouped = q_pre.view(n_layers, n_kv_heads, group, prompt_length, head_dim).mean(dim=2)
        sum_qk[(cfg, "prefill")] += torch.einsum("lhsd,lhse->lhde", q_pre_grouped, k_pre)

        n_tokens[(cfg, "prefill")] += prompt_length
        del q_pre, k_pre, q_pre_per_head_outer, q_pre_summed_over_group, q_pre_grouped

        # Decode slice (decode-phase queries only; we don't compress decode keys).
        if captured_length > prompt_length:
            decode_len = captured_length - prompt_length
            q_dec = q_post[:, :, prompt_length:captured_length, :].to(accum_device).float()
            # Decode keys for the C_QK term — these are the prefill keys that decode-Q reads.
            # But for the *cross moment* of (decode-Q, prefill-K) we'd need a per-token alignment;
            # E1-2's purpose is the marginal Σ_Q^{decode}, not a cross-moment with K. We still compute
            # a synthetic C_QK^{decode} using the prefill-K pool as the "reference K distribution",
            # to allow per-task CCA solves with decode-Q. The CCA basis from this is interpretive only.
            q_dec_per_head_outer = torch.einsum("lhsd,lhse->lhde", q_dec, q_dec)
            q_dec_summed_over_group = (
                q_dec_per_head_outer
                .view(n_layers, n_kv_heads, group, head_dim, head_dim)
                .sum(dim=2)
            )
            sum_q_outer[(cfg, "decode")] += q_dec_summed_over_group
            n_tokens[(cfg, "decode")] += decode_len
            # No C_QK for decode here; decode-Q vs prefill-K cross-moment not well-defined per token.
            del q_dec, q_dec_per_head_outer, q_dec_summed_over_group

        del payload, q_post, k_post
        log_fn(f"  ex {ex_idx+1}/{len(examples)} ({ex['file']}, cfg={cfg}, prompt_len={prompt_length}, decode_len={captured_length - prompt_length}) in {time.time()-t0:.1f}s")

    return {
        "sum_q_outer": sum_q_outer,
        "sum_qk": sum_qk,
        "sum_k_outer": sum_k_outer,
        "n_tokens": n_tokens,
        "n_pos_K": n_pos_K,
        "configs": configs,
        "group": group,
    }


def _finalize_sigmas(
    accum: dict,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    low_conf_threshold: int,
    keep_on_device: bool = True,
) -> dict:
    """Finalize accumulators into Σ matrices. Keeps tensors on the accumulation device by default
    (so downstream eigh / CCA stays on GPU); set keep_on_device=False to move to CPU."""
    configs = accum["configs"]
    phases = ("prefill", "decode")
    group = accum["group"]
    sigma_q: dict[tuple[str, str], torch.Tensor] = {}
    sigma_k: dict[str, torch.Tensor] = {}
    cqk: dict[tuple[str, str], torch.Tensor] = {}
    n_tokens = {k: int(v) for k, v in accum["n_tokens"].items()}
    n_pos_K = {k: int(v) for k, v in accum["n_pos_K"].items()}
    low_conf_tasks = []

    def _maybe_cpu(t: torch.Tensor) -> torch.Tensor:
        return t if keep_on_device else t.cpu()

    for cfg in configs:
        for phase in phases:
            n = max(1, n_tokens[(cfg, phase)])
            # Σ_Q[h] = sum_q_outer / (group * n)  — matches E1's mean_g E[q_g q_g^T] convention.
            sigma_q[(cfg, phase)] = _maybe_cpu(accum["sum_q_outer"][(cfg, phase)] / (group * n))
            # C_QK only meaningful for prefill (no per-token decode-Q vs prefill-K alignment).
            if phase == "prefill":
                cqk[(cfg, phase)] = _maybe_cpu(accum["sum_qk"][(cfg, phase)] / max(1, n))
            if phase == "decode" and n_tokens[(cfg, phase)] < low_conf_threshold:
                low_conf_tasks.append(f"{cfg}/decode")
        sigma_k[cfg] = _maybe_cpu(accum["sum_k_outer"][cfg] / max(1, n_pos_K[cfg]))
    return {
        "sigma_q": sigma_q,
        "sigma_k": sigma_k,
        "cqk": cqk,
        "n_tokens": n_tokens,
        "n_pos_K": n_pos_K,
        "low_confidence_tasks": low_conf_tasks,
        "group": group,
    }


def _eigh_batched(sigma_batched: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """sigma_batched: (..., d, d). Returns (eigvals descending, eigvecs)."""
    sym = 0.5 * (sigma_batched + sigma_batched.transpose(-1, -2))
    d = sym.shape[-1]
    trace = sym.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    scale = (trace / d).clamp_min(1e-12)
    reg_eye = torch.eye(d, dtype=sym.dtype, device=sym.device)
    while reg_eye.dim() < sym.dim():
        reg_eye = reg_eye.unsqueeze(0)
    reg = eps * scale[..., None, None] * reg_eye
    eigvals, eigvecs = torch.linalg.eigh(sym + reg)
    eigvals = eigvals.clamp_min(0.0)
    # eigh returns ascending; flip to descending.
    eigvals = torch.flip(eigvals, dims=[-1])
    eigvecs = torch.flip(eigvecs, dims=[-1])
    return eigvals, eigvecs


def _compute_r95(eigvals: torch.Tensor) -> torch.Tensor:
    """eigvals: (..., d) descending. Returns r95 per leading-dim entry."""
    cum = torch.cumsum(eigvals, dim=-1)
    total = cum[..., -1:].clamp_min(1e-30)
    frac = cum / total
    return ((frac >= 0.95).int().argmax(dim=-1) + 1).long()


def _cumulative_energy_at_ranks(eigvals: torch.Tensor, ranks: list[int]) -> dict[int, torch.Tensor]:
    cum = torch.cumsum(eigvals, dim=-1)
    total = cum[..., -1:].clamp_min(1e-30)
    frac = cum / total
    out = {}
    for r in ranks:
        idx = min(r, eigvals.shape[-1]) - 1
        out[r] = frac[..., idx]
    return out


def _bures_one_sided_from_eig(
    eigvals_a: torch.Tensor, eigvecs_a: torch.Tensor, sigma_b: torch.Tensor, eps: float
) -> torch.Tensor:
    """Compute || W_a Σ_b W_a^T - I ||_F  in batched form using a's cached eigendecomposition.

    eigvals_a: (..., d) descending, eigvecs_a: (..., d, d).  W_a = diag(λ_reg^{-1/2}) @ U^T.
    sigma_b:   (..., d, d).
    Output:    (...) Frobenius norm per batch entry.
    """
    d = eigvals_a.shape[-1]
    trace = eigvals_a.sum(dim=-1)
    scale = (trace / d).clamp_min(1e-12)
    eigvals_reg = eigvals_a.clamp_min(eps * scale.unsqueeze(-1))
    sqrt_inv = 1.0 / torch.sqrt(eigvals_reg)  # (..., d)
    middle = eigvecs_a.transpose(-1, -2) @ sigma_b @ eigvecs_a  # (..., d, d)
    M = sqrt_inv.unsqueeze(-1) * middle * sqrt_inv.unsqueeze(-2)
    eye = torch.eye(d, dtype=M.dtype, device=M.device)
    return torch.linalg.norm(M - eye, ord="fro", dim=(-2, -1))


def _bures_distance_batched(
    sigma_a: torch.Tensor, sigma_b: torch.Tensor,
    eigvals_a: torch.Tensor, eigvecs_a: torch.Tensor,
    eigvals_b: torch.Tensor, eigvecs_b: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Symmetrized Bures-style distance using cached eigendecompositions (no fresh eigh)."""
    d_ab = _bures_one_sided_from_eig(eigvals_a, eigvecs_a, sigma_b, eps)
    d_ba = _bures_one_sided_from_eig(eigvals_b, eigvecs_b, sigma_a, eps)
    return 0.5 * (d_ab + d_ba)


def _subspace_overlap(eigvecs_a: torch.Tensor, eigvecs_b: torch.Tensor, r: int) -> torch.Tensor:
    """eigvecs are (..., d, d) with columns as eigenvectors descending. Returns (..., ) overlap in [0, 1]."""
    P_a = eigvecs_a[..., :, :r]
    P_b = eigvecs_b[..., :, :r]
    M = P_a.transpose(-1, -2) @ P_b  # (..., r, r)
    return (M.pow(2).sum(dim=(-2, -1)) / r).clamp_max(1.0)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    bundle_dir = (repo_root / args.bundle).resolve()
    cca_stats_path = (repo_root / args.cca_stats).resolve()
    out_dir = (repo_root / args.output).resolve()
    ensure_dir(out_dir)

    _log(f"Bundle: {bundle_dir}")
    _log(f"Output: {out_dir}")
    _log(f"Device: {args.device}")

    with open(bundle_dir / "manifest.json") as f:
        manifest = json.load(f)
    examples = manifest["examples"]
    if args.limit_examples is not None:
        examples = examples[: args.limit_examples]

    # Discover layer/head/dim from cca_stats.pt (E1 output).
    if not cca_stats_path.exists():
        raise FileNotFoundError(f"cca_stats.pt missing at {cca_stats_path}; run E1 first.")
    cca_stats = torch.load(cca_stats_path, map_location="cpu", weights_only=False)
    n_layers = int(cca_stats["n_layers"])
    n_kv_heads = int(cca_stats["n_kv_heads"])
    head_dim = int(cca_stats["head_dim"])
    _log(f"  n_layers={n_layers} n_kv_heads={n_kv_heads} head_dim={head_dim}")

    accum_device = args.device

    # ---- Phase 1: accumulate per (task, phase) Σ_Q, Σ_K, C_QK ----
    _log("Accumulating per-(task, phase) Σ_Q / Σ_K / C_QK from per-example payloads...")
    t0 = time.time()
    accum = _accumulate_per_task_phase(
        bundle_dir, examples, n_layers, n_kv_heads, head_dim, accum_device, _log
    )
    finalized = _finalize_sigmas(accum, n_layers, n_kv_heads, head_dim, args.low_confidence_threshold)
    _log(f"Accumulation done in {time.time()-t0:.1f}s")
    _log(f"  configs: {finalized['n_tokens']}")
    _log(f"  K positions per task: {finalized['n_pos_K']}")
    if finalized["low_confidence_tasks"]:
        _log(f"  LOW-CONFIDENCE tasks (decode-token count < {args.low_confidence_threshold}): {finalized['low_confidence_tasks']}")

    sigma_q = finalized["sigma_q"]
    sigma_k = finalized["sigma_k"]
    cqk = finalized["cqk"]

    # ---- Phase 2: marginal eigendecompositions for rank analysis ----
    _log("Eigendecomposing all Σ matrices for marginal rank analysis...")
    t0 = time.time()
    metric_tensors = {}  # (metric, task) -> dict {sigma, eigvals, eigvecs, r95, cum_energy}
    ranks_for_table = [16, 32, 48, 64, 96]
    for cfg, sk in sigma_k.items():
        eigvals, eigvecs = _eigh_batched(sk, args.eps)
        r95 = _compute_r95(eigvals)
        cum_e = _cumulative_energy_at_ranks(eigvals, ranks_for_table)
        metric_tensors[("K", cfg)] = {
            "sigma": sk, "eigvals": eigvals, "eigvecs": eigvecs, "r95": r95,
            "cum_energy_at_r": {r: v for r, v in cum_e.items()},
        }
    for (cfg, phase), sq in sigma_q.items():
        eigvals, eigvecs = _eigh_batched(sq, args.eps)
        r95 = _compute_r95(eigvals)
        cum_e = _cumulative_energy_at_ranks(eigvals, ranks_for_table)
        metric_tensors[(f"Q_{phase}", cfg)] = {
            "sigma": sq, "eigvals": eigvals, "eigvecs": eigvecs, "r95": r95,
            "cum_energy_at_r": {r: v for r, v in cum_e.items()},
        }
    _log(f"Eigendecomps + rank stats done in {time.time()-t0:.1f}s")

    # ---- Phase 3: pairwise distances ----
    _log("Computing pairwise Bures distances + subspace overlaps (using cached eigvecs)...")
    t0 = time.time()
    configs = sorted({c for (c, _) in sigma_q.keys()})

    def _eig_for(metric: str, cfg: str) -> tuple[torch.Tensor, torch.Tensor]:
        v = metric_tensors[(metric, cfg)]
        return v["eigvals"], v["eigvecs"]

    # Phase distance per task (prefill vs decode for Q).
    phase_distances = {}
    for cfg in configs:
        ev_a, U_a = _eig_for("Q_prefill", cfg)
        ev_b, U_b = _eig_for("Q_decode", cfg)
        d = _bures_distance_batched(
            sigma_q[(cfg, "prefill")], sigma_q[(cfg, "decode")],
            ev_a, U_a, ev_b, U_b, args.eps,
        )
        phase_distances[cfg] = d  # keep on device

    # Cross-task distances per metric.
    cross_task_distances = {}
    metrics_for_cross = ["Q_prefill", "Q_decode", "K"]
    for m in metrics_for_cross:
        for i, ca in enumerate(configs):
            for cb in configs[i + 1:]:
                if m == "K":
                    sa, sb = sigma_k[ca], sigma_k[cb]
                else:
                    phase = m.replace("Q_", "")
                    sa, sb = sigma_q[(ca, phase)], sigma_q[(cb, phase)]
                ev_a, U_a = _eig_for(m, ca)
                ev_b, U_b = _eig_for(m, cb)
                d = _bures_distance_batched(sa, sb, ev_a, U_a, ev_b, U_b, args.eps)
                cross_task_distances[(m, ca, cb)] = d

    # Subspace overlaps at r ∈ {16, 32, 64} for each metric × pair.
    subspace_overlaps = {}
    for m in metrics_for_cross:
        for i, ca in enumerate(configs):
            for cb in configs[i + 1:]:
                _, U_a = _eig_for(m, ca)
                _, U_b = _eig_for(m, cb)
                for r in [16, 32, 64]:
                    subspace_overlaps[(m, ca, cb, r)] = _subspace_overlap(U_a, U_b, r)
    # Phase overlap per task.
    phase_subspace_overlaps = {}
    for cfg in configs:
        _, U_a = _eig_for("Q_prefill", cfg)
        _, U_b = _eig_for("Q_decode", cfg)
        for r in [16, 32, 64]:
            phase_subspace_overlaps[(cfg, r)] = _subspace_overlap(U_a, U_b, r)
    _log(f"Distances + overlaps done in {time.time()-t0:.1f}s")

    # ---- Phase 4: per-task CCA basis stability ----
    _log("Computing per-task CCA bases + comparing to global E1 P_K...")
    t0 = time.time()
    device = next(iter(sigma_q.values())).device
    global_PK = cca_stats["P_K"].to(device).float()  # (n_layers, n_kv_heads, d, d)
    eye = torch.eye(head_dim, device=device)
    per_task_cca = {}  # cfg -> {P_K, P_K_inv, P_Q, rho}
    cca_identity_max_err: dict[str, float] = {}
    cca_subspace_overlap_vs_global: dict[str, dict[int, torch.Tensor]] = {}
    for cfg in configs:
        sq_flat = sigma_q[(cfg, "prefill")].reshape(-1, head_dim, head_dim)
        sk_flat = sigma_k[cfg].reshape(-1, head_dim, head_dim)
        cqk_flat = cqk[(cfg, "prefill")].reshape(-1, head_dim, head_dim)
        out = compute_cca_basis(sq_flat, sk_flat, cqk_flat, eps=args.eps)
        P_K = out["P_K"].view(n_layers, n_kv_heads, head_dim, head_dim)
        P_K_inv = out["P_K_inv"].view(n_layers, n_kv_heads, head_dim, head_dim)
        rho = out["rho"].view(n_layers, n_kv_heads, head_dim)
        # F4: identity check.
        err = (P_K_inv @ P_K - eye).abs().reshape(-1, head_dim, head_dim).amax(dim=(-2, -1))
        cca_identity_max_err[cfg] = float(err.max().item())
        per_task_cca[cfg] = {
            "P_K": P_K, "P_K_inv": P_K_inv,
            "P_Q": out["P_Q"].view(n_layers, n_kv_heads, head_dim, head_dim),
            "rho": rho,
        }
        cca_subspace_overlap_vs_global[cfg] = {}
        for r in [16, 32, 64]:
            R_t = P_K[..., :r, :]
            R_g = global_PK[..., :r, :]
            M = R_t @ R_g.transpose(-1, -2)
            cca_subspace_overlap_vs_global[cfg][r] = (M.pow(2).sum(dim=(-2, -1)) / r).clamp_max(1.0)
    _log(f"Per-task CCA + identity checks done in {time.time()-t0:.1f}s")
    for cfg, err in cca_identity_max_err.items():
        _log(f"  P_K_inv·P_K identity max abs err for cfg={cfg}: {err:.4e}")

    # ---- Phase 5: regression test against E1's global Σ_Q ----
    _log("Regression test: combining per-task Σ_Q^prefill must match E1's global Σ_Q...")
    n_pre_total = sum(finalized["n_tokens"][(c, "prefill")] for c in configs)
    sigma_q_global_recon = sum(
        sigma_q[(c, "prefill")] * finalized["n_tokens"][(c, "prefill")] for c in configs
    ) / max(1, n_pre_total)
    sigma_q_e1 = cca_stats["sigma_q"].to(sigma_q_global_recon.device).float()
    diff = (sigma_q_global_recon - sigma_q_e1).abs()
    rel_err_per_head = diff.reshape(-1, head_dim, head_dim).amax(dim=(-2, -1)) / sigma_q_e1.reshape(-1, head_dim, head_dim).abs().amax(dim=(-2, -1)).clamp_min(1e-12)
    regression_max_rel_err = float(rel_err_per_head.max().item())
    _log(f"  Σ_Q^global recon vs cca_stats Σ_Q: max rel err = {regression_max_rel_err:.4e}")

    # ---- Save artifacts ----
    _log("Saving distribution_stats.pt and metrics_e1_2.json...")

    def _t(x):
        return x.detach().cpu() if isinstance(x, torch.Tensor) else x

    def _cpu_tree(obj):
        if isinstance(obj, torch.Tensor):
            return _t(obj)
        if isinstance(obj, dict):
            return {k: _cpu_tree(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(_cpu_tree(v) for v in obj)
        return obj

    save_payload = _cpu_tree({
        "sigma_q": {f"{cfg}/{phase}": sigma_q[(cfg, phase)] for (cfg, phase) in sigma_q},
        "sigma_k": dict(sigma_k),
        "cqk": {f"{cfg}/{phase}": cqk[(cfg, phase)] for (cfg, phase) in cqk},
        "metric_tensors": {f"{m}/{cfg}": v for (m, cfg), v in metric_tensors.items()},
        "phase_distances": {cfg: d for cfg, d in phase_distances.items()},
        "cross_task_distances": {f"{m}/{ca}_vs_{cb}": d for (m, ca, cb), d in cross_task_distances.items()},
        "subspace_overlaps": {f"{m}/{ca}_vs_{cb}/r{r}": ov for (m, ca, cb, r), ov in subspace_overlaps.items()},
        "phase_subspace_overlaps": {f"{cfg}/r{r}": ov for (cfg, r), ov in phase_subspace_overlaps.items()},
        "per_task_cca": per_task_cca,
        "cca_subspace_overlap_vs_global": cca_subspace_overlap_vs_global,
        "n_tokens": finalized["n_tokens"],
        "n_pos_K": finalized["n_pos_K"],
        "low_confidence_tasks": finalized["low_confidence_tasks"],
        "configs": configs,
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "group": finalized["group"],
        "regression_max_rel_err": regression_max_rel_err,
    })
    torch.save(save_payload, out_dir / "distribution_stats.pt")

    # JSON summary (no large tensors).
    def _r95_summary(t: torch.Tensor) -> dict:
        flat = t.reshape(-1)
        l0excl = t[1:].reshape(-1)
        return {
            "min": int(flat.min().item()),
            "median": int(flat.median().item()),
            "max": int(flat.max().item()),
            "p10": int(flat.float().quantile(0.10).item()),
            "p90": int(flat.float().quantile(0.90).item()),
            "l0excl_min": int(l0excl.min().item()),
            "l0excl_median": int(l0excl.median().item()),
            "l0excl_max": int(l0excl.max().item()),
            "l0excl_p10": int(l0excl.float().quantile(0.10).item()),
            "l0excl_p90": int(l0excl.float().quantile(0.90).item()),
        }

    rank_summary = {}
    for (m, cfg), v in metric_tensors.items():
        rank_summary[f"{m}/{cfg}"] = {
            "r95": _r95_summary(v["r95"]),
            "cum_energy_at_r": {str(r): {
                "median_l0excl": float(v["cum_energy_at_r"][r][1:].median().item()),
                "p10_l0excl": float(v["cum_energy_at_r"][r][1:].float().quantile(0.10).item()),
                "p90_l0excl": float(v["cum_energy_at_r"][r][1:].float().quantile(0.90).item()),
            } for r in ranks_for_table},
        }

    distance_summary = {}
    for cfg, d in phase_distances.items():
        distance_summary[f"phase/{cfg}"] = {
            "median_l0excl": float(d[1:].median().item()),
            "p10_l0excl": float(d[1:].float().quantile(0.10).item()),
            "p90_l0excl": float(d[1:].float().quantile(0.90).item()),
            "n_tokens_decode": finalized["n_tokens"][(cfg, "decode")],
        }
    for (m, ca, cb), d in cross_task_distances.items():
        distance_summary[f"{m}/{ca}_vs_{cb}"] = {
            "median_l0excl": float(d[1:].median().item()),
            "p10_l0excl": float(d[1:].float().quantile(0.10).item()),
            "p90_l0excl": float(d[1:].float().quantile(0.90).item()),
        }

    overlap_summary = {}
    for (m, ca, cb, r), ov in subspace_overlaps.items():
        overlap_summary[f"{m}/{ca}_vs_{cb}/r{r}"] = {
            "median_l0excl": float(ov[1:].median().item()),
            "p10_l0excl": float(ov[1:].float().quantile(0.10).item()),
            "p90_l0excl": float(ov[1:].float().quantile(0.90).item()),
        }
    for (cfg, r), ov in phase_subspace_overlaps.items():
        overlap_summary[f"phase/{cfg}/r{r}"] = {
            "median_l0excl": float(ov[1:].median().item()),
            "p10_l0excl": float(ov[1:].float().quantile(0.10).item()),
            "p90_l0excl": float(ov[1:].float().quantile(0.90).item()),
        }

    # JSON serialization can't handle tuple keys — flatten to "task/phase" strings.
    n_tokens_json = {f"{cfg}/{phase}": v for (cfg, phase), v in finalized["n_tokens"].items()}
    summary = {
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "configs": configs,
        "n_tokens": n_tokens_json,
        "n_pos_K": dict(finalized["n_pos_K"]),
        "low_confidence_tasks": finalized["low_confidence_tasks"],
        "low_confidence_threshold": args.low_confidence_threshold,
        "regression_max_rel_err": regression_max_rel_err,
        "cca_identity_max_err_per_task": cca_identity_max_err,
        "rank_analysis": rank_summary,
        "distances": distance_summary,
        "subspace_overlaps": overlap_summary,
        "ranks_for_table": ranks_for_table,
    }
    save_json(out_dir / "metrics_e1_2.json", summary)
    _log(f"E1-2 distributional diagnostics complete. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
