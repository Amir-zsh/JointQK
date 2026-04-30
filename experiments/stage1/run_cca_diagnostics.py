"""Stage 1E E1+E2: headless CCA diagnostics + closed-form rate-distortion simulation.

Outputs:
    artifacts/stage1/cca_vs_waterfill_study/
        cca_stats.pt          — Σ_Q, Σ_K, C_QK per (layer, kv_head) and ρ spectra.
        metrics_e1_e2.json    — gate-friendly summary numbers.
        figures/
            spectrum_overlay.png      — 280-curve ρ_i vs i.
            r95_heatmap.png           — min r reaching 95% of canonical-correlation energy.
            spectrum_per_layer.png    — per-layer median ρ.
            sim_log_ratio_heatmap.png — log2(D_method / D_v3) at b_avg=3.
            sim_per_layer_lines.png   — per-method, per-layer mean log-ratio.
            sim_pareto.png            — (rate, distortion) Pareto across the (b_avg, r) grid.

Run:
    python -u -m experiments.stage1.run_cca_diagnostics \
        --bundle artifacts/stage1/query_stats_longbench_under4k \
        --output artifacts/stage1/cca_vs_waterfill_study
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.stage1.toolkit import (
    CrossMomentsAccumulator,
    compute_cca_basis,
    ensure_dir,
    save_json,
    water_fill,
)


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1E CCA diagnostics + simulation (E1+E2).")
    parser.add_argument(
        "--bundle",
        type=str,
        default="artifacts/stage1/query_stats_longbench_under4k",
        help="Path to the query-stats bundle (must contain pooled_stats.pt and examples/).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/stage1/cca_vs_waterfill_study",
        help="Output directory (will be created).",
    )
    parser.add_argument("--eps", type=float, default=1e-4, help="Whitening regularization scale.")
    parser.add_argument(
        "--b-avg-grid",
        type=str,
        default="2,3,4",
        help="Comma-separated b_avg values for the simulation.",
    )
    parser.add_argument(
        "--r-grid",
        type=str,
        default="16,32,48,64,96",
        help="Comma-separated r values (rank cutoffs) for CCA/V truncation methods.",
    )
    return parser.parse_args()


def load_pooled_stats(bundle_dir: Path) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Returns (sigma_q_grouped, sigma_k, n_layers, n_kv_heads).

    sigma_q_grouped is the GQA-pooled Q second-moment per (layer, kv_head):
    Σ_Q[h] = mean over query heads in group h of E[q q^T].
    """
    pooled_path = bundle_dir / "pooled_stats.pt"
    pooled = torch.load(pooled_path, map_location="cpu", weights_only=False)
    _, _, q_post_secm = pooled["q_post"]
    _, _, k_post_secm = pooled["k_post"]
    n_layers, n_q_heads, dim, _ = q_post_secm.shape
    n_kv_heads = k_post_secm.shape[1]
    if n_q_heads % n_kv_heads != 0:
        raise ValueError(f"GQA mismatch: {n_q_heads} query heads, {n_kv_heads} kv heads")
    group = n_q_heads // n_kv_heads
    sigma_q = q_post_secm.float().view(n_layers, n_kv_heads, group, dim, dim).mean(dim=2)
    sigma_k = k_post_secm.float()
    return sigma_q, sigma_k, n_layers, n_kv_heads


def accumulate_cqk(bundle_dir: Path, n_layers: int, n_kv_heads: int) -> tuple[torch.Tensor, int]:
    """One-pass loop over per-example payloads to accumulate C_QK over PREFILL positions only.

    Returns (cqk, total_tokens). cqk shape: (n_layers, n_kv_heads, dim, dim).
    """
    examples_dir = bundle_dir / "examples"
    paths = sorted(examples_dir.glob("ex_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No examples found in {examples_dir}")
    accumulators = [CrossMomentsAccumulator() for _ in range(n_layers)]
    for i, p in enumerate(paths):
        t0 = time.time()
        payload = torch.load(p, map_location="cpu", weights_only=False)
        prompt_length = int(payload["prompt_length"])
        q_post = payload["q_post"]
        k_post = payload["k_post"]
        if q_post.shape[2] < prompt_length:
            raise RuntimeError(
                f"{p}: captured_length={q_post.shape[2]} < prompt_length={prompt_length}"
            )
        q_prefill = q_post[:, :, :prompt_length, :]
        k_prefill = k_post[:, :, :prompt_length, :]
        for layer in range(n_layers):
            q_layer = q_prefill[layer].unsqueeze(0)
            k_layer = k_prefill[layer].unsqueeze(0)
            accumulators[layer].update(q_layer, k_layer, num_kv_heads=n_kv_heads)
        del payload, q_post, k_post, q_prefill, k_prefill
        _log(
            f"  C_QK accumulated example {i+1}/{len(paths)} ({p.name}) "
            f"prefill_len={prompt_length} in {time.time()-t0:.1f}s"
        )
    cqk = torch.stack([acc.finalize() for acc in accumulators], dim=0)
    total_tokens = accumulators[0].total_count
    return cqk, total_tokens


def closed_form_distortion(
    sigma_k_diag_in_basis: torch.Tensor,
    bits_alloc: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """High-rate Bennett distortion: D = sum_j w_j * σ²_j * 2^(-2 b_j).

    All inputs shape (..., d). 'weights' is the eigenvalue or canonical-correlation² weighting
    (so that D is the Q-weighted distortion on the original axes when basis preserves M_q).

    For coords with b_j = 0, the term contributes the full w_j * σ²_j (no quantization).
    The closed-form 2^0 = 1 already handles this.
    """
    return (weights * sigma_k_diag_in_basis * torch.pow(2.0, -2.0 * bits_alloc)).sum(dim=-1)


def simulate_method(
    method: str,
    sigma_k: torch.Tensor,
    sigma_q_eigvals: torch.Tensor,
    sigma_q_eigvecs: torch.Tensor,
    cca: dict[str, torch.Tensor],
    b_avg: float,
    r: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form Q-weighted distortion under each method.

    Returns (distortion_per_head, bits_alloc_per_head_per_coord).

    Method definitions:
        v3:           random rotation; uniform b_avg bits/coord; weights = M_q diagonal in random basis.
                      Approximated by E[trace(M_q Σ_K)] * 2^(-2 b_avg) / d * d = trace(M_q Σ_K) * 2^(-2 b_avg).
                      (Random rotation makes the per-coord variance uniform on average.)
        v_truncate:   V basis (eigvecs of M_q); top r coords get b_avg*d/r bits, rest get 0.
        v_waterfill:  V basis; continuous water-fill on λ_j σ²_j(V).
        cca_uniform:  CCA basis P_K; top r coords get b_avg*d/r bits, rest get 0.
        cca_waterfill:CCA basis; continuous water-fill on ρ_j² σ²_j(CCA).
    """
    H = sigma_k.shape[0]
    d = head_dim
    total_bits = b_avg * d

    if method == "v3":
        # Closed form expected distortion under random rotation + uniform bits + V3-style normalize.
        # In V3, after unit-normalize and rotate, each coord has variance 1/d. So per-coord D = (1/d) * 2^(-2b_avg).
        # Multiplied by trace(M_q Σ_K) (the energy that V3 would put on 'attention-error' axis) gives total D.
        # Equivalently: D = trace(M_q Σ_K) * 2^(-2 b_avg).
        sigma_q_full = (sigma_q_eigvecs * sigma_q_eigvals.unsqueeze(-2)) @ sigma_q_eigvecs.transpose(-1, -2)
        trace_mq_sigmak = torch.einsum("hij,hji->h", sigma_q_full, sigma_k)
        bits = torch.full((H, d), float(b_avg))
        distortion = trace_mq_sigmak * (2.0 ** (-2.0 * b_avg))
        return distortion, bits

    if method.startswith("v_"):
        # V basis: rotate by eigvecs of M_q. Compute σ²_j(V) = (V^T Σ_K V)_jj
        # The Q-weighted distortion contribution is λ_j * σ²_j(V) * 2^(-2 b_j).
        V = sigma_q_eigvecs
        Sk_in_V = V.transpose(-1, -2) @ sigma_k @ V
        sigma_k_diag = Sk_in_V.diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
        weights = sigma_q_eigvals.clamp_min(1e-30)
    elif method.startswith("cca_"):
        P_K = cca["P_K"]
        rho = cca["rho"].clamp(min=0.0, max=1.0)
        # Project K to canonical basis: Σ_K_in_cca = P_K @ Σ_K @ P_K^T
        Sk_in_cca = P_K @ sigma_k @ P_K.transpose(-1, -2)
        sigma_k_diag = Sk_in_cca.diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
        # Weight by ρ_j² (Q-weighted distortion under whitened-CCA decomposition).
        weights = (rho.float() ** 2).clamp_min(1e-30)
    else:
        raise ValueError(f"Unknown method '{method}'")

    if method.endswith("_truncate") or method.endswith("_uniform"):
        if r is None or r <= 0 or r > d:
            raise ValueError(f"Method '{method}' requires 1 <= r <= d, got r={r}")
        per_coord_bits = total_bits / r
        bits = torch.zeros((H, d))
        bits[:, :r] = per_coord_bits
    elif method.endswith("_waterfill"):
        # Allocate by water-filling on (weights_j * σ²_j) — that's what minimizes Q-weighted distortion.
        wf_input = weights * sigma_k_diag
        bits = water_fill(wf_input, total_bits=total_bits)
    else:
        raise ValueError(f"Unknown allocation in method '{method}'")

    distortion = closed_form_distortion(sigma_k_diag, bits, weights)
    return distortion, bits


def aggregate_per_layer(values: torch.Tensor, n_layers: int) -> dict[str, list[float]]:
    """values shape: (n_layers * n_kv_heads,) or (n_layers, n_kv_heads). Returns per-layer mean/median/min/max."""
    if values.dim() == 1:
        n_kv_heads = values.shape[0] // n_layers
        values = values.view(n_layers, n_kv_heads)
    return {
        "mean": values.mean(dim=-1).tolist(),
        "median": values.median(dim=-1).values.tolist(),
        "min": values.min(dim=-1).values.tolist(),
        "max": values.max(dim=-1).values.tolist(),
    }


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    bundle_dir = (repo_root / args.bundle).resolve()
    out_dir = (repo_root / args.output).resolve()
    figures_dir = out_dir / "figures"
    ensure_dir(out_dir)
    ensure_dir(figures_dir)

    _log(f"Bundle: {bundle_dir}")
    _log(f"Output: {out_dir}")

    # ---- Phase 1: load second moments ----
    _log("Loading pooled second moments (Σ_Q GQA-pooled, Σ_K)...")
    sigma_q, sigma_k, n_layers, n_kv_heads = load_pooled_stats(bundle_dir)
    head_dim = sigma_q.shape[-1]
    _log(f"  n_layers={n_layers} n_kv_heads={n_kv_heads} head_dim={head_dim}")

    # ---- Phase 2: accumulate C_QK ----
    cqk_cache_path = out_dir / "cqk_cache.pt"
    if cqk_cache_path.exists():
        _log(f"Loading cached C_QK from {cqk_cache_path}")
        cache = torch.load(cqk_cache_path, map_location="cpu", weights_only=False)
        cqk = cache["cqk"]
        total_tokens = cache["total_tokens"]
    else:
        _log("Accumulating C_QK from per-example payloads (prefill positions only)...")
        t0 = time.time()
        cqk, total_tokens = accumulate_cqk(bundle_dir, n_layers, n_kv_heads)
        _log(f"C_QK accumulated in {time.time()-t0:.1f}s; total prefill tokens = {total_tokens}")
        torch.save({"cqk": cqk, "total_tokens": total_tokens}, cqk_cache_path)

    sigma_q_flat = sigma_q.reshape(n_layers * n_kv_heads, head_dim, head_dim)
    sigma_k_flat = sigma_k.reshape(n_layers * n_kv_heads, head_dim, head_dim)
    cqk_flat = cqk.reshape(n_layers * n_kv_heads, head_dim, head_dim)

    cca_stats_cached_path = out_dir / "cca_stats.pt"
    if cca_stats_cached_path.exists():
        _log(f"Loading cached CCA stats + V eigendecomp from {cca_stats_cached_path}")
        cs = torch.load(cca_stats_cached_path, map_location="cpu", weights_only=False)
        rho = cs["rho"]
        rho_flat = rho.reshape(-1, head_dim)
        P_K_flat = cs["P_K"].reshape(-1, head_dim, head_dim)
        cca_payload = {
            "rho": rho_flat,
            "P_K": P_K_flat,
            "P_K_inv": cs["P_K_inv"].reshape(-1, head_dim, head_dim),
            "P_Q": cs["P_Q"].reshape(-1, head_dim, head_dim),
        }
        mq_eigvals = cs["mq_eigvals"].reshape(-1, head_dim)
        mq_eigvecs = cs["mq_eigvecs"].reshape(-1, head_dim, head_dim)
    else:
        # ---- E1: CCA spectrum ----
        _log("Computing CCA spectra and projections per (layer, kv_head)...")
        cca_payload = compute_cca_basis(sigma_q_flat, sigma_k_flat, cqk_flat, eps=args.eps)
        rho_flat = cca_payload["rho"]  # (H, d)
        P_K_flat = cca_payload["P_K"]
        rho = rho_flat.view(n_layers, n_kv_heads, head_dim)

        # Eigendecomposition of M_q for V basis
        _log("Eigendecomposing M_q (V basis)...")
        sym_mq = 0.5 * (sigma_q_flat + sigma_q_flat.transpose(-1, -2))
        sym_mq = sym_mq + args.eps * (sym_mq.diagonal(dim1=-2, dim2=-1).sum(-1) / head_dim).clamp_min(1e-12)[
            :, None, None
        ] * torch.eye(head_dim).unsqueeze(0)
        mq_eigvals, mq_eigvecs = torch.linalg.eigh(sym_mq)
        # Sort eigenvalues descending so V is "top eigvecs first"
        sort_idx = torch.argsort(mq_eigvals, dim=-1, descending=True)
        mq_eigvals = torch.gather(mq_eigvals, -1, sort_idx)
        mq_eigvecs = torch.gather(
            mq_eigvecs, -1, sort_idx.unsqueeze(-2).expand(-1, head_dim, -1)
        )

    # ---- E1 plots ----
    _log("Drawing E1 figures...")
    fig, ax = plt.subplots(figsize=(10, 6))
    for h in range(rho_flat.shape[0]):
        ax.plot(rho_flat[h].cpu(), color="steelblue", alpha=0.06, linewidth=0.5)
    median_rho = rho_flat.median(dim=0).values.cpu()
    ax.plot(median_rho, color="black", linewidth=2.0, label="median")
    ax.set_xlabel("rank index i")
    ax.set_ylabel("ρ_i (canonical correlation)")
    ax.set_title(f"CCA spectrum across {rho_flat.shape[0]} (layer, kv_head) pairs")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(figures_dir / "spectrum_overlay.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 95%-energy r per (layer, head)
    rho_sq = rho.float() ** 2
    cum_energy = torch.cumsum(rho_sq, dim=-1)
    total_energy = rho_sq.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    fraction = cum_energy / total_energy
    threshold = 0.95
    r95 = (fraction >= threshold).int().argmax(dim=-1) + 1
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(r95.cpu(), aspect="auto", cmap="viridis")
    ax.set_xlabel("kv_head")
    ax.set_ylabel("layer")
    ax.set_title(f"min r so that Σρ²[:r] / Σρ² ≥ 0.95 (head_dim={head_dim})")
    plt.colorbar(im, ax=ax)
    fig.savefig(figures_dir / "r95_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Per-layer median spectrum
    per_layer_median = rho.median(dim=1).values  # (n_layers, d)
    fig, ax = plt.subplots(figsize=(10, 6))
    for layer in range(n_layers):
        color = plt.cm.viridis(layer / max(1, n_layers - 1))
        ax.plot(per_layer_median[layer].cpu(), color=color, alpha=0.7, linewidth=0.8)
    ax.set_xlabel("rank index i")
    ax.set_ylabel("median ρ_i across kv_heads")
    ax.set_title("Per-layer median CCA spectrum")
    ax.grid(alpha=0.3)
    fig.savefig(figures_dir / "spectrum_per_layer.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- E2: closed-form simulation ----
    b_avg_grid = [float(x) for x in args.b_avg_grid.split(",")]
    r_grid = [int(x) for x in args.r_grid.split(",")]
    methods = ["v3", "v_truncate", "v_waterfill", "cca_uniform", "cca_waterfill"]
    _log(f"Running E2 simulation: b_avg ∈ {b_avg_grid}, r ∈ {r_grid}, methods={methods}")

    cca_for_sim = {"P_K": P_K_flat, "rho": rho_flat, "P_K_inv": cca_payload["P_K_inv"]}

    sim_results: dict[str, dict[str, dict[str, list[float]]]] = {}
    layer_summaries: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}
    layer0_excluded_summaries: dict[str, dict[str, dict[str, float]]] = {}
    raw_per_head: dict[str, dict[str, dict[str, list[float]]]] = {}

    for b_avg in b_avg_grid:
        sim_results[f"b{b_avg}"] = {}
        layer_summaries[f"b{b_avg}"] = {}
        layer0_excluded_summaries[f"b{b_avg}"] = {}
        raw_per_head[f"b{b_avg}"] = {}
        # V3 first (no r dependency)
        D_v3, _ = simulate_method(
            "v3", sigma_k_flat, mq_eigvals, mq_eigvecs, cca_for_sim, b_avg, r=head_dim, head_dim=head_dim
        )
        sim_results[f"b{b_avg}"]["v3"] = {"D": D_v3.tolist()}
        layer_summaries[f"b{b_avg}"]["v3"] = aggregate_per_layer(D_v3, n_layers)
        d_v3_layered = D_v3.view(n_layers, n_kv_heads)
        excluded = d_v3_layered[1:].mean().item()
        layer0_excluded_summaries[f"b{b_avg}"]["v3"] = {"D_mean": excluded}
        raw_per_head[f"b{b_avg}"]["v3"] = {"D": D_v3.tolist()}

        for r in r_grid:
            for method in methods:
                if method == "v3":
                    continue
                if method.endswith("_waterfill") and r != r_grid[0]:
                    # waterfill is r-agnostic — only run once per b_avg
                    continue
                # Skip absurd uniform allocations (per-coord bits > 16 is effectively full precision).
                if (method.endswith("_uniform") or method.endswith("_truncate")) and (b_avg * head_dim) / r > 16.0:
                    continue
                key = f"{method}_r{r}" if not method.endswith("_waterfill") else f"{method}"
                if key in sim_results[f"b{b_avg}"]:
                    continue
                D, bits = simulate_method(
                    method,
                    sigma_k_flat,
                    mq_eigvals,
                    mq_eigvecs,
                    cca_for_sim,
                    b_avg,
                    r=r,
                    head_dim=head_dim,
                )
                # Bit budget conservation check
                bit_sum = bits.sum(dim=-1)
                expected = b_avg * head_dim
                if not torch.allclose(bit_sum, torch.full_like(bit_sum, expected), atol=1e-3):
                    _log(
                        f"  WARN bit-budget drift in {method} b_avg={b_avg} r={r}: "
                        f"sum range [{bit_sum.min().item():.4f}, {bit_sum.max().item():.4f}]"
                    )
                sim_results[f"b{b_avg}"][key] = {"D": D.tolist(), "bits_min": bits.min().item(), "bits_max": bits.max().item()}
                layer_summaries[f"b{b_avg}"][key] = aggregate_per_layer(D, n_layers)
                d_layered = D.view(n_layers, n_kv_heads)
                layer0_excluded_summaries[f"b{b_avg}"][key] = {"D_mean": d_layered[1:].mean().item()}
                raw_per_head[f"b{b_avg}"][key] = {"D": D.tolist()}
        # log-ratio table
        v3_d = D_v3
        for key in list(sim_results[f"b{b_avg}"].keys()):
            if key == "v3":
                continue
            d_method = torch.tensor(sim_results[f"b{b_avg}"][key]["D"])
            log_ratio = torch.log2(d_method.clamp_min(1e-30) / v3_d.clamp_min(1e-30))
            log_ratio_layered = log_ratio.view(n_layers, n_kv_heads)
            sim_results[f"b{b_avg}"][key]["log2_D_over_Dv3_mean"] = log_ratio.mean().item()
            sim_results[f"b{b_avg}"][key]["log2_D_over_Dv3_l0excl_mean"] = log_ratio_layered[1:].mean().item()
            sim_results[f"b{b_avg}"][key]["frac_better_than_v3"] = (log_ratio < 0).float().mean().item()
            sim_results[f"b{b_avg}"][key]["frac_better_than_v3_l0excl"] = (log_ratio_layered[1:] < 0).float().mean().item()

    # ---- E2 figures ----
    _log("Drawing E2 figures...")
    # Heatmap of log2(D_method / D_v3) at b_avg=3 for the headline method (cca_waterfill).
    if 3.0 in b_avg_grid:
        D_v3 = torch.tensor(sim_results["b3.0"]["v3"]["D"]).view(n_layers, n_kv_heads)
        for headline in ["v_waterfill", "cca_waterfill", f"v_truncate_r64", f"cca_uniform_r64"]:
            if headline in sim_results["b3.0"]:
                D_m = torch.tensor(sim_results["b3.0"][headline]["D"]).view(n_layers, n_kv_heads)
                log_ratio = torch.log2(D_m.clamp_min(1e-30) / D_v3.clamp_min(1e-30))
                fig, ax = plt.subplots(figsize=(8, 8))
                vmax = log_ratio.abs().quantile(0.98).item()
                im = ax.imshow(log_ratio.cpu(), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
                ax.set_xlabel("kv_head")
                ax.set_ylabel("layer")
                ax.set_title(f"log₂(D[{headline}] / D[v3]) @ b_avg=3 (negative = method beats V3)")
                plt.colorbar(im, ax=ax)
                fig.savefig(figures_dir / f"sim_log_ratio_{headline}_b3.png", dpi=120, bbox_inches="tight")
                plt.close(fig)

    # Per-layer line plot showing best method at each layer
    if 3.0 in b_avg_grid:
        fig, ax = plt.subplots(figsize=(12, 6))
        D_v3 = torch.tensor(sim_results["b3.0"]["v3"]["D"]).view(n_layers, n_kv_heads)
        for key in sim_results["b3.0"]:
            if key == "v3":
                continue
            D_m = torch.tensor(sim_results["b3.0"][key]["D"]).view(n_layers, n_kv_heads)
            log_ratio = torch.log2(D_m.clamp_min(1e-30) / D_v3.clamp_min(1e-30))
            per_layer = log_ratio.median(dim=-1).values
            ax.plot(per_layer.cpu(), label=key, linewidth=1.0)
        ax.axhline(0.0, color="black", linestyle="--", alpha=0.5)
        ax.set_xlabel("layer")
        ax.set_ylabel("median log₂(D_method / D_v3)")
        ax.set_title("Per-layer median log-ratio across methods at b_avg=3 (negative = beats V3)")
        ax.legend(fontsize=8, loc="best", ncol=2)
        ax.grid(alpha=0.3)
        fig.savefig(figures_dir / "sim_per_layer_lines.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    # Pareto frontier in (rate, distortion). One point per (b_avg, r, method).
    fig, ax = plt.subplots(figsize=(10, 7))
    for method in methods:
        xs = []
        ys = []
        for b_avg in b_avg_grid:
            for r in r_grid if not method.endswith("_waterfill") else [head_dim]:
                key = f"{method}_r{r}" if not method.endswith("_waterfill") else method
                if method == "v3":
                    key = "v3"
                if key not in sim_results[f"b{b_avg}"]:
                    continue
                D = torch.tensor(sim_results[f"b{b_avg}"][key]["D"]).view(n_layers, n_kv_heads)
                # Layer-0-excluded mean as the headline
                xs.append(b_avg)
                ys.append(D[1:].mean().item())
        if xs:
            ax.plot(xs, ys, "o-", label=method, alpha=0.7)
    ax.set_xlabel("rate (b_avg)")
    ax.set_ylabel("Q-weighted distortion (layer-0-excluded mean)")
    ax.set_title("Closed-form rate-distortion across methods")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(figures_dir / "sim_pareto.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- save artifacts ----
    _log("Saving cca_stats.pt and metrics_e1_e2.json...")
    cca_stats = {
        "sigma_q": sigma_q,
        "sigma_k": sigma_k,
        "cqk": cqk,
        "rho": rho,  # (n_layers, n_kv_heads, d)
        "P_K": P_K_flat.view(n_layers, n_kv_heads, head_dim, head_dim),
        "P_K_inv": cca_payload["P_K_inv"].view(n_layers, n_kv_heads, head_dim, head_dim),
        "P_Q": cca_payload["P_Q"].view(n_layers, n_kv_heads, head_dim, head_dim),
        "mq_eigvals": mq_eigvals.view(n_layers, n_kv_heads, head_dim),
        "mq_eigvecs": mq_eigvecs.view(n_layers, n_kv_heads, head_dim, head_dim),
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "total_prefill_tokens": total_tokens,
    }
    torch.save(cca_stats, out_dir / "cca_stats.pt")

    # gate-friendly summary
    summary = {
        "n_layers": int(n_layers),
        "n_kv_heads": int(n_kv_heads),
        "head_dim": int(head_dim),
        "total_prefill_tokens": int(total_tokens),
        "rho_min": float(rho_flat.min().item()),
        "rho_max": float(rho_flat.max().item()),
        "rho_layer0_max": float(rho.view(n_layers, n_kv_heads, head_dim)[0].max().item()),
        "rho_layer0_min": float(rho.view(n_layers, n_kv_heads, head_dim)[0].min().item()),
        "r95_min": int(r95.min().item()),
        "r95_max": int(r95.max().item()),
        "r95_median": int(r95.median().item()),
        "b_avg_grid": b_avg_grid,
        "r_grid": r_grid,
        "methods": methods,
        "simulation": sim_results,
        "layer_summaries": layer_summaries,
        "layer0_excluded_summaries": layer0_excluded_summaries,
    }
    save_json(out_dir / "metrics_e1_e2.json", summary)

    _log("E1+E2 diagnostics complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
