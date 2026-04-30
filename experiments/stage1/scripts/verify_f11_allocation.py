"""Verify F11: real CCA-waterfill uses the trace-formula allocation.

F8 corrected E2's CCA Q-weighted distortion objective from rho^2 to
diag((P_K_inv)^T Sigma_Q P_K_inv). F11 is the analogous real-quantizer bug:
`build_method_compressor` still allocated CCA-waterfill bits with rho^2.

This script checks:
1. A synthetic diagonal case where the trace-formula allocation is independently
   computable from Sigma_Q and differs from the old rho^2 allocation.
2. All real Qwen3-8B Stage 1E heads, verifying `build_method_compressor`
   returns exactly the same integer allocations as an independent trace-formula
   implementation at b_avg in {2, 3, 4}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.stage1.toolkit.metric_transform import water_fill
from experiments.stage1.toolkit.per_coord_quantization import (
    build_method_compressor,
    round_bits_to_integer,
)


REPO = Path(__file__).resolve().parents[3]
CCA_STATS = REPO / "artifacts/stage1/cca_vs_waterfill_study/cca_stats.pt"


def expected_trace_bits(
    sigma_q: torch.Tensor,
    sigma_k: torch.Tensor,
    P_K: torch.Tensor,
    P_K_inv: torch.Tensor,
    b_avg: float,
) -> torch.Tensor:
    """Independent CCA trace-formula allocation."""
    d = sigma_q.shape[-1]
    Sk_in_basis = P_K @ sigma_k @ P_K.transpose(-1, -2)
    sigma_k_diag = Sk_in_basis.diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    metric_in_basis = P_K_inv.transpose(-1, -2) @ sigma_q @ P_K_inv
    weights = metric_in_basis.diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    bits_cont = water_fill(weights * sigma_k_diag, total_bits=b_avg * d)
    return round_bits_to_integer(bits_cont.unsqueeze(0), total_bits=int(b_avg * d)).squeeze(0)


def old_rho_bits(rho: torch.Tensor, sigma_k: torch.Tensor, P_K: torch.Tensor, b_avg: float) -> torch.Tensor:
    d = rho.numel()
    Sk_in_basis = P_K @ sigma_k @ P_K.transpose(-1, -2)
    sigma_k_diag = Sk_in_basis.diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    bits_cont = water_fill((rho.clamp(0, 1) ** 2).clamp_min(1e-30) * sigma_k_diag, total_bits=b_avg * d)
    return round_bits_to_integer(bits_cont.unsqueeze(0), total_bits=int(b_avg * d)).squeeze(0)


def compressor_bits(
    sigma_q: torch.Tensor,
    sigma_k: torch.Tensor,
    P_K: torch.Tensor,
    P_K_inv: torch.Tensor,
    rho: torch.Tensor,
    b_avg: float,
) -> torch.Tensor:
    d = sigma_q.shape[-1]
    comp = build_method_compressor(
        method="cca_waterfill",
        sigma_q_for_head=sigma_q,
        sigma_k_for_head=sigma_k,
        cca={"P_K": P_K, "P_K_inv": P_K_inv, "rho": rho},
        mq_eigvals=None,
        mq_eigvecs=None,
        b_avg=b_avg,
        r=d,
        seed=0,
        head_dim=d,
    )
    return comp.bits_int.cpu()


def synthetic_check() -> None:
    d = 4
    sigma_q = torch.diag(torch.tensor([64.0, 16.0, 4.0, 1.0]))
    sigma_k = torch.eye(d)
    P_K = torch.eye(d)
    P_K_inv = torch.eye(d)
    # Deliberately make rho flat so the old allocation differs from the trace objective.
    rho = torch.ones(d)
    b_avg = 1.0

    got = compressor_bits(sigma_q, sigma_k, P_K, P_K_inv, rho, b_avg)
    expected = expected_trace_bits(sigma_q, sigma_k, P_K, P_K_inv, b_avg)
    old = old_rho_bits(rho, sigma_k, P_K, b_avg)

    if not torch.equal(got, expected):
        raise AssertionError(f"synthetic trace allocation mismatch: got {got.tolist()} expected {expected.tolist()}")
    if torch.equal(got, old):
        raise AssertionError(f"synthetic check failed to distinguish old rho^2 allocation: {got.tolist()}")
    print(f"synthetic: trace bits = {got.tolist()}, old rho^2 bits = {old.tolist()}")


def real_qwen_check() -> None:
    cca = torch.load(CCA_STATS, map_location="cpu", weights_only=False)
    sigma_q = cca["sigma_q"].float().reshape(-1, cca["head_dim"], cca["head_dim"])
    sigma_k = cca["sigma_k"].float().reshape(-1, cca["head_dim"], cca["head_dim"])
    P_K = cca["P_K"].float().reshape(-1, cca["head_dim"], cca["head_dim"])
    P_K_inv = cca["P_K_inv"].float().reshape(-1, cca["head_dim"], cca["head_dim"])
    rho = cca["rho"].float().reshape(-1, cca["head_dim"])
    n_heads, d = rho.shape

    for b_avg in (2.0, 3.0, 4.0):
        total_abs_diff = 0.0
        old_total_abs_diff = 0.0
        old_identical = 0
        zero_counts = []
        for h in range(n_heads):
            got = compressor_bits(sigma_q[h], sigma_k[h], P_K[h], P_K_inv[h], rho[h], b_avg)
            expected = expected_trace_bits(sigma_q[h], sigma_k[h], P_K[h], P_K_inv[h], b_avg)
            old = old_rho_bits(rho[h], sigma_k[h], P_K[h], b_avg)
            diff = (got - expected).abs().sum().item()
            if diff != 0:
                layer = h // int(cca["n_kv_heads"])
                kv_head = h % int(cca["n_kv_heads"])
                raise AssertionError(
                    f"real allocation mismatch at b={b_avg}, layer={layer}, kv_head={kv_head}: "
                    f"L1 diff={diff}, got={got.tolist()}, expected={expected.tolist()}"
                )
            total_abs_diff += diff
            old_l1 = (got - old).abs().sum().item()
            old_total_abs_diff += old_l1
            old_identical += int(old_l1 == 0)
            zero_counts.append(int((got == 0).sum().item()))

        z = torch.tensor(zero_counts, dtype=torch.float32)
        print(
            f"real b={b_avg:.0f}: matched trace allocation for {n_heads} heads; "
            f"old rho^2 identical heads={old_identical}; "
            f"mean old-vs-new L1={old_total_abs_diff / n_heads:.2f}; "
            f"mean zero coords={z.mean().item():.2f}, max zero coords={int(z.max().item())}"
        )


def main() -> int:
    synthetic_check()
    real_qwen_check()
    print("F11 allocation verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
