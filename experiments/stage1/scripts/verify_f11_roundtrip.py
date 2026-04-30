"""Deeper F11 verification: confirm the *real* CCA-waterfill roundtrip MSE matches
the closed-form trace formula that the water-fill objective is optimizing.

If any of these are inconsistent (allocation, codebook scaling, forward/inverse
maps, or trace-formula weight), the closed-form and the actual roundtrip diverge.

This is a stronger check than verify_f11_allocation.py, which only verified that
the compressor's allocation matches an independent trace-formula allocation.
Here we verify the *MSE prediction itself*.

For each of a few real (layer, kv_head) pairs:
  1. Build the cca_waterfill compressor from real CCA stats.
  2. Generate synthetic keys k ~ N(0, Sigma_K) and queries q ~ N(0, Sigma_Q).
  3. Roundtrip the keys; measure Q-weighted MSE = E[(q.T (k - k_hat))^2].
  4. Compute the closed-form trace prediction:
       D_pred = sum_j weight_j * sigma_k_diag_j * 2^(-2 b_j)
       (with weight_j = (P_K_inv.T Sigma_Q P_K_inv)_jj,
        sigma_k_diag_j = (P_K Sigma_K P_K.T)_jj)
     plus a Bennett constant for Lloyd-Max codebooks at integer b.
  5. Match within 5% (tighter than F8's 0.6%; we're testing the high-rate Bennett
     approximation against real Lloyd-Max code, so the constant matters).

Also checks the same for v_waterfill (V-basis), which is the unaffected baseline.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.stage1.toolkit.metric_transform import water_fill
from experiments.stage1.toolkit.per_coord_quantization import (
    build_method_compressor,
    round_bits_to_integer,
    unit_gaussian_centroids,
)


REPO = Path(__file__).resolve().parents[3]
CCA_STATS = REPO / "artifacts/stage1/cca_vs_waterfill_study/cca_stats.pt"


def gaussian_lloyd_max_distortion(bits: int) -> float:
    """Return the per-coord MSE of a Lloyd-Max codebook on a unit-Gaussian source.

    For b bits we use the cached centroids from `unit_gaussian_centroids`; the
    distortion is computed by Monte-Carlo over a moderate sample, in chunks to
    avoid the n × 2^bits tensor blowup at large b.
    """
    if bits <= 0:
        return 1.0
    centroids = unit_gaussian_centroids(bits).double()
    K = centroids.numel()
    g = torch.Generator().manual_seed(0)
    n = 200_000
    chunk = max(1, min(n, 4_000_000 // max(K, 1)))
    total_sq = 0.0
    total_n = 0
    remaining = n
    while remaining > 0:
        m = min(chunk, remaining)
        x = torch.randn(m, generator=g, dtype=torch.float64)
        diffs = (x.unsqueeze(-1) - centroids.unsqueeze(0)).abs()
        idx = diffs.argmin(dim=-1)
        quant = centroids[idx]
        total_sq += float(((x - quant) ** 2).sum().item())
        total_n += m
        remaining -= m
    return total_sq / max(total_n, 1)


def closed_form_q_weighted_mse(
    sigma_q: torch.Tensor,
    sigma_k: torch.Tensor,
    P_K: torch.Tensor,
    P_K_inv: torch.Tensor,
    bits_int: torch.Tensor,
    bennett_const: list[float],
) -> float:
    """D = sum_j w_j * sigma_k_diag_j * c(b_j)  where c(b) is the Lloyd-Max
    distortion of a unit-Gaussian source with b bits.
    """
    Sk_in_basis = P_K @ sigma_k @ P_K.transpose(-1, -2)
    sigma_k_diag = Sk_in_basis.diagonal(dim1=-2, dim2=-1)
    metric = P_K_inv.transpose(-1, -2) @ sigma_q @ P_K_inv
    weights = metric.diagonal(dim1=-2, dim2=-1)
    d = bits_int.numel()
    total = 0.0
    for j in range(d):
        b = int(bits_int[j])
        c = bennett_const[b] if b < len(bennett_const) else float(2.0 ** (-2.0 * b))
        total += float(weights[j]) * float(sigma_k_diag[j]) * c
    return total


def closed_form_q_weighted_mse_v(
    mq_eigvals: torch.Tensor,
    mq_eigvecs: torch.Tensor,
    sigma_k: torch.Tensor,
    bits_int: torch.Tensor,
    bennett_const: list[float],
) -> float:
    Sk_in_V = mq_eigvecs.transpose(-1, -2) @ sigma_k @ mq_eigvecs
    sigma_k_diag = Sk_in_V.diagonal(dim1=-2, dim2=-1)
    weights = mq_eigvals
    d = bits_int.numel()
    total = 0.0
    for j in range(d):
        b = int(bits_int[j])
        c = bennett_const[b] if b < len(bennett_const) else float(2.0 ** (-2.0 * b))
        total += float(weights[j]) * float(sigma_k_diag[j]) * c
    return total


def sample_centered_gaussian(cov: torch.Tensor, n: int, generator: torch.Generator) -> torch.Tensor:
    """Sample n vectors with covariance `cov` (uncentered second moment if cov is not centered)."""
    sym = 0.5 * (cov + cov.T)
    eigvals, eigvecs = torch.linalg.eigh(sym.double())
    eigvals = eigvals.clamp_min(0.0)
    L = eigvecs * eigvals.sqrt().unsqueeze(0)
    z = torch.randn(n, cov.shape[-1], generator=generator, dtype=torch.float64)
    return (z @ L.T).float()


def main() -> int:
    torch.set_num_threads(4)
    print("loading cca_stats.pt...", flush=True)
    cca = torch.load(CCA_STATS, map_location="cpu", weights_only=False)
    sigma_q_all = cca["sigma_q"].float().reshape(-1, cca["head_dim"], cca["head_dim"])
    sigma_k_all = cca["sigma_k"].float().reshape(-1, cca["head_dim"], cca["head_dim"])
    P_K_all = cca["P_K"].float().reshape(-1, cca["head_dim"], cca["head_dim"])
    P_K_inv_all = cca["P_K_inv"].float().reshape(-1, cca["head_dim"], cca["head_dim"])
    rho_all = cca["rho"].float().reshape(-1, cca["head_dim"])
    mq_eigvals_all = cca["mq_eigvals"].float().reshape(-1, cca["head_dim"])
    mq_eigvecs_all = cca["mq_eigvecs"].float().reshape(-1, cca["head_dim"], cca["head_dim"])
    n_kv_heads = int(cca["n_kv_heads"])

    head_dim = sigma_q_all.shape[-1]
    # Lazy-compute Lloyd-Max c(b) only for the b values actually used.
    bennett_cache: dict[int, float] = {}

    def get_c(b: int) -> float:
        if b not in bennett_cache:
            bennett_cache[b] = gaussian_lloyd_max_distortion(b)
        return bennett_cache[b]

    # bennett_const compatibility: callers index it; provide a small dict-backed list-like.
    class _LazyConst:
        def __getitem__(self, b: int) -> float:
            return get_c(b)

        def __len__(self) -> int:
            return 17  # callers use len() to decide fallback; effectively unbounded

    bennett_const = _LazyConst()
    # Show first few values
    for b in range(7):
        get_c(b)
    print(f"Lloyd-Max c(b) for b=0..6: {[round(bennett_cache[b], 4) for b in range(7)]}")
    print(f"Bennett 2^(-2b) for b=0..6: {[round(2.0 ** (-2.0 * b), 4) for b in range(7)]}")
    print()

    # Test layers/heads spanning the model.
    test_pairs = [(1, 0), (12, 5), (24, 7)]
    n_samples = 50_000
    failures = []
    print(f"Running {len(test_pairs)} (layer, head) pairs x 3 b_avg x 2 methods, {n_samples} MC samples each...", flush=True)

    for layer, kv_head in test_pairs:
        h = layer * n_kv_heads + kv_head
        sigma_q = sigma_q_all[h]
        sigma_k = sigma_k_all[h]
        P_K = P_K_all[h]
        P_K_inv = P_K_inv_all[h]
        rho = rho_all[h]

        for b_avg in (2.0, 3.0, 4.0):
            comp = build_method_compressor(
                method="cca_waterfill",
                sigma_q_for_head=sigma_q,
                sigma_k_for_head=sigma_k,
                cca={"P_K": P_K, "P_K_inv": P_K_inv, "rho": rho},
                mq_eigvals=mq_eigvals_all[h],
                mq_eigvecs=mq_eigvecs_all[h],
                b_avg=b_avg,
                r=head_dim,
                seed=0,
                head_dim=head_dim,
            )
            bits_int = comp.bits_int.cpu()

            # Predicted MSE from closed form.
            d_pred = closed_form_q_weighted_mse(sigma_q, sigma_k, P_K, P_K_inv, bits_int, bennett_const)

            # Monte-Carlo: generate keys ~ N(0, Sigma_K), queries ~ N(0, Sigma_Q).
            g = torch.Generator().manual_seed(int(layer * 1000 + kv_head * 10 + b_avg))
            keys = sample_centered_gaussian(sigma_k, n_samples, g)
            queries = sample_centered_gaussian(sigma_q, n_samples, g)
            keys_3d = keys.unsqueeze(0).unsqueeze(0)  # (1, 1, n, d)
            with torch.no_grad():
                keys_recon = comp.roundtrip(keys_3d).squeeze(0).squeeze(0)
            err = keys - keys_recon
            d_mc = float(((queries * err).sum(dim=-1) ** 2).mean().item())

            rel_err = abs(d_mc - d_pred) / max(d_mc, 1e-30)
            tag = "OK" if rel_err < 0.05 else "FAIL"
            print(
                f"  CCA-waterfill layer={layer:2d} head={kv_head} b={int(b_avg)}: "
                f"D_pred={d_pred:.4e}  D_mc={d_mc:.4e}  rel_err={rel_err*100:.2f}%  [{tag}]"
            )
            if rel_err >= 0.05:
                failures.append(("cca_waterfill", layer, kv_head, b_avg, d_pred, d_mc))

            # Also do v_waterfill as a sanity baseline.
            comp_v = build_method_compressor(
                method="v_waterfill",
                sigma_q_for_head=sigma_q,
                sigma_k_for_head=sigma_k,
                cca=None,
                mq_eigvals=mq_eigvals_all[h],
                mq_eigvecs=mq_eigvecs_all[h],
                b_avg=b_avg,
                r=head_dim,
                seed=0,
                head_dim=head_dim,
            )
            bits_int_v = comp_v.bits_int.cpu()
            d_pred_v = closed_form_q_weighted_mse_v(
                mq_eigvals_all[h], mq_eigvecs_all[h], sigma_k, bits_int_v, bennett_const
            )
            with torch.no_grad():
                keys_recon_v = comp_v.roundtrip(keys_3d).squeeze(0).squeeze(0)
            err_v = keys - keys_recon_v
            d_mc_v = float(((queries * err_v).sum(dim=-1) ** 2).mean().item())
            rel_err_v = abs(d_mc_v - d_pred_v) / max(d_mc_v, 1e-30)
            tag_v = "OK" if rel_err_v < 0.05 else "FAIL"
            print(
                f"  V-waterfill   layer={layer:2d} head={kv_head} b={int(b_avg)}: "
                f"D_pred={d_pred_v:.4e}  D_mc={d_mc_v:.4e}  rel_err={rel_err_v*100:.2f}%  [{tag_v}]"
            )
            if rel_err_v >= 0.05:
                failures.append(("v_waterfill", layer, kv_head, b_avg, d_pred_v, d_mc_v))

    print()
    if failures:
        print(f"FAILED: {len(failures)} mismatches > 5%")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"All {len(test_pairs) * 3 * 2} closed-form-vs-MC checks passed (rel err < 5%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
