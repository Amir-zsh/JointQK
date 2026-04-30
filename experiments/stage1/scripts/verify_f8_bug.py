"""Verify F8: the E2 simulation's CCA per-coord weight is wrong; the proposed fix corrects it.

Strategy:
1. Synthetic case where the answer is computable by hand (Σ_Q = Σ_K = I, C_QK = diag(ρ)).
2. Confirm trace formula gives the right answer.
3. Confirm current simulation gives the wrong answer.
4. Confirm proposed fix gives the right answer.
5. Cross-check on real Qwen3-8B data: current sim vs corrected sim vs Monte-Carlo estimate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.stage1.toolkit import compute_cca_basis  # noqa: E402


def section(name: str) -> None:
    print()
    print("=" * 76)
    print(name)
    print("=" * 76)


# ============================================================================
# Step 1: Synthetic case — Σ_Q = Σ_K = I, C_QK = diag(ρ)
# ============================================================================
section("Step 1: Synthetic case — Σ_Q = Σ_K = I, C_QK = diag([0.9, 0.5, 0.1, 0])")

d = 4
sigma_q_synth = torch.eye(d).double()
sigma_k_synth = torch.eye(d).double()
rho_true = torch.tensor([0.9, 0.5, 0.1, 0.0]).double()
cqk_synth = torch.diag(rho_true)

# CCA: M = W_Q C_QK W_K = I · diag(ρ) · I = diag(ρ). SVD trivial: U = V_h = I, ρ = ρ.
out = compute_cca_basis(sigma_q_synth, sigma_k_synth, cqk_synth, eps=1e-6)
print(f"  ρ from compute_cca_basis: {out['rho'].tolist()}")
print(f"  P_K (should be ≈ I):    {out['P_K'].numpy()}")
print(f"  P_K_inv (should be ≈ I): {out['P_K_inv'].numpy()}")

# In this case, P_K = I (modulo tiny eps regularization), so quantizing in canonical = quantizing in original.
# Quantize each coord with bits b_j and Bennett noise variance σ²_j · 2^{-2 b_j}.
# Since input is unit-variance per coord, noise variance = 2^{-2 b_j}.
# Q-weighted distortion = trace(Σ_Q · E[Δk Δk^T]) = trace(I · diag(2^{-2 b_j})) = sum_j 2^{-2 b_j}.

bits = torch.tensor([3.0, 3.0, 3.0, 3.0])  # 12 total
expected_D = (2.0 ** (-2.0 * bits)).sum().item()
print(f"\n  Direct trace formula at b=[3,3,3,3]: D = sum_j 2^{{-2·3}} = {expected_D:.6f}")

# Simulation's formula: weights = ρ² · σ²(CCA basis) = ρ² · 1
# (canonical-K is whitened, so σ²(CCA) ≈ 1)
sim_weights = rho_true ** 2
sim_sigma_k_diag = torch.ones(d).double()
sim_D = (sim_weights * sim_sigma_k_diag * 2.0 ** (-2.0 * bits)).sum().item()
print(f"  Buggy sim (weights = ρ²):         D = {sim_D:.6f}  → underestimates by {expected_D / sim_D:.2f}×")

# Proposed fix: weights = ((P_K^-1)^T Σ_Q P_K^-1)_jj
P_K_inv = out["P_K_inv"].double()
fix_weights = (P_K_inv.transpose(-1, -2) @ sigma_q_synth @ P_K_inv).diagonal()
fix_sigma_k_diag = torch.ones(d).double()
fix_D = (fix_weights * fix_sigma_k_diag * 2.0 ** (-2.0 * bits)).sum().item()
print(f"  Proposed-fix sim:                 D = {fix_D:.6f}  → matches expected to {abs(fix_D - expected_D):.2e}")

assert abs(fix_D - expected_D) < 1e-4, f"FIX FAILED in synthetic case: {fix_D} vs {expected_D}"
print("  ✓ Fix matches expected analytic answer in synthetic case.")


# ============================================================================
# Step 2: Synthetic case — non-trivial Σ_Q, Σ_K
# ============================================================================
section("Step 2: Synthetic case — non-trivial Σ_Q, Σ_K, plus Monte-Carlo cross-check")

torch.manual_seed(0)
d = 8
A = torch.randn(d, d).double()
B = torch.randn(d, d).double()
sigma_q_synth = (A @ A.T) + 0.1 * torch.eye(d).double()
sigma_k_synth = (B @ B.T) + 0.1 * torch.eye(d).double()
# Make a joint Gaussian (q, k) ~ N(0, [[Σ_Q, C_QK], [C_QK^T, Σ_K]]) by construction:
# pick a coupling matrix M with all singular values ≤ rho_max < 1, then
# C_QK = Σ_Q^{1/2} · M · Σ_K^{1/2}. The Schur complement is then Σ_K^{1/2} (I - M^T M) Σ_K^{1/2} ≻ 0.
sq_sqrt = torch.linalg.cholesky(sigma_q_synth)
sk_sqrt = torch.linalg.cholesky(sigma_k_synth)
U_M, _, Vh_M = torch.linalg.svd(torch.randn(d, d).double())
rho_target = torch.linspace(0.85, 0.05, d).double()  # canonical correlations
M_inner = U_M @ torch.diag(rho_target) @ Vh_M
cqk_synth = sq_sqrt @ M_inner @ sk_sqrt.T
# Verify joint is PSD by checking Schur complement
schur = sigma_k_synth - cqk_synth.T @ torch.linalg.inv(sigma_q_synth) @ cqk_synth
schur_eigvals = torch.linalg.eigvalsh(schur)
print(f"  Joint Gaussian Schur eigvals min: {schur_eigvals.min().item():.4f} (must be > 0 for valid joint)")
assert schur_eigvals.min().item() > 0, "Bad synthetic setup"

out = compute_cca_basis(sigma_q_synth, sigma_k_synth, cqk_synth, eps=1e-6)
P_K = out["P_K"].double()
P_K_inv = out["P_K_inv"].double()
rho = out["rho"].double()

# Verify P_K_inv · P_K ≈ I
recon = P_K_inv @ P_K
print(f"  P_K_inv · P_K identity check: max abs err = {(recon - torch.eye(d).double()).abs().max().item():.2e}")

# Now: simulate quantization with given bits, compare three methods.
bits = torch.tensor([5.0, 4.0, 4.0, 3.0, 3.0, 2.0, 2.0, 1.0])
n_samples = 500_000

# --- (a) Direct trace formula ---
# In canonical-K basis, K is whitened: σ²(CCA) ≈ 1. Bennett noise variance per coord = 2^{-2 b_j}.
# Δk = P_K_inv ε. trace(Σ_Q · E[Δk Δk^T]) = sum_j ((P_K_inv)^T Σ_Q P_K_inv)_jj · 2^{-2 b_j}.
sigma_k_diag_canonical = (P_K @ sigma_k_synth @ P_K.T).diagonal()  # should be ≈ 1
fix_weights = (P_K_inv.transpose(-1, -2) @ sigma_q_synth @ P_K_inv).diagonal()
trace_D = (fix_weights * sigma_k_diag_canonical * 2.0 ** (-2.0 * bits)).sum().item()

# --- (b) Buggy simulation ---
buggy_weights = rho ** 2
sim_D = (buggy_weights * sigma_k_diag_canonical * 2.0 ** (-2.0 * bits)).sum().item()

# --- (c) Monte-Carlo estimate of E[(q^T Δk)²] ---
# Sample (q, k) from joint Gaussian, simulate per-coord Gaussian noise in canonical-K basis with
# variance σ²(CCA) · 2^{-2 b_j}, un-rotate, compute (q^T Δk)².
joint_cov = torch.zeros(2 * d, 2 * d).double()
joint_cov[:d, :d] = sigma_q_synth
joint_cov[d:, d:] = sigma_k_synth
joint_cov[:d, d:] = cqk_synth
joint_cov[d:, :d] = cqk_synth.T
joint_chol = torch.linalg.cholesky(joint_cov)
z = torch.randn(n_samples, 2 * d).double() @ joint_chol.T
q_samples = z[:, :d]
k_samples = z[:, d:]

# Quantization noise in canonical: variance σ²_j(CCA) · 2^{-2 b_j} per coord, independent of (q, k).
noise_std = torch.sqrt(sigma_k_diag_canonical * 2.0 ** (-2.0 * bits))
eps_samples = torch.randn(n_samples, d).double() * noise_std
delta_k = eps_samples @ P_K_inv.T  # un-rotate: Δk = P_K_inv ε  (row form: Δk_row = ε_row @ P_K_inv^T)

mc_D = ((q_samples * delta_k).sum(dim=-1) ** 2).mean().item()
print(f"\n  D from direct trace formula (proposed fix): {trace_D:.6f}")
print(f"  D from Monte-Carlo (n={n_samples}):           {mc_D:.6f}  (should match trace)")
print(f"  D from buggy simulation (weights = ρ²):     {sim_D:.6f}  (should differ)")
print(f"  trace vs MC error: {abs(trace_D - mc_D):.6f}  ({100 * abs(trace_D - mc_D) / trace_D:.2f}%)")
print(f"  buggy vs MC error: {abs(sim_D - mc_D):.6f}   ({100 * abs(sim_D - mc_D) / mc_D:.2f}%)")

assert abs(trace_D - mc_D) / trace_D < 0.05, "Trace formula should match Monte-Carlo within 5%"
assert abs(sim_D - mc_D) / mc_D > 0.5, "Buggy formula should differ from Monte-Carlo by >50% on this case"
print("  ✓ Proposed-fix formula matches Monte-Carlo; buggy formula doesn't.")


# ============================================================================
# Step 3: V-basis case — verify the fix's reduction matches current V code
# ============================================================================
section("Step 3: V-basis case — fix should give same result as current V code")

# For V basis: P_K = V^T (orthogonal forward), P_K_inv = V (orthogonal inverse).
# Per-coord input variance is σ²_j(V) (not 1, since V doesn't whiten).
# Per-coord weight via trace formula: ((P_K_inv)^T Σ_Q P_K_inv)_jj = (V^T Σ_Q V)_jj = λ_j.
# So D = sum_j λ_j · σ²_j(V) · 2^{-2 b_j}, matching the current V simulation. ✓

eigvals, eigvecs = torch.linalg.eigh(sigma_q_synth)
sort_idx = torch.argsort(eigvals, descending=True)
eigvals = eigvals[sort_idx]
eigvecs = eigvecs[:, sort_idx]
V = eigvecs

# Current V simulation:
v_weights = eigvals
v_sigma_k_diag = (V.T @ sigma_k_synth @ V).diagonal()
v_D_current = (v_weights * v_sigma_k_diag * 2.0 ** (-2.0 * bits)).sum().item()

# Trace formula at V basis (P_K = V^T, P_K_inv = V):
P_K_v = V.T
P_K_inv_v = V
v_weights_fix = (P_K_inv_v.T @ sigma_q_synth @ P_K_inv_v).diagonal()
v_sigma_k_diag_fix = (P_K_v @ sigma_k_synth @ P_K_v.T).diagonal()
v_D_fix = (v_weights_fix * v_sigma_k_diag_fix * 2.0 ** (-2.0 * bits)).sum().item()

print(f"  D from current V simulation:        {v_D_current:.6f}")
print(f"  D from trace-formula fix at V:      {v_D_fix:.6f}")
print(f"  Difference:                         {abs(v_D_current - v_D_fix):.2e}")

assert abs(v_D_current - v_D_fix) < 1e-6, "V case should match the trace formula"
print("  ✓ V case is consistent with the trace formula (current V code is correct).")


# ============================================================================
# Step 4: Real Qwen3-8B data — corrected sim vs Monte-Carlo
# ============================================================================
section("Step 4: Real Qwen3-8B data — Monte-Carlo cross-check on (layer=1, kv_head=0)")

cca = torch.load(REPO / "artifacts/stage1/cca_vs_waterfill_study/cca_stats.pt", map_location="cpu", weights_only=False)
layer, h = 1, 0
sigma_q_r = cca["sigma_q"][layer, h].double()
sigma_k_r = cca["sigma_k"][layer, h].double()
cqk_r = cca["cqk"][layer, h].double()

# Recompute CCA basis from the same data (in float64 for numerical clarity)
out_r = compute_cca_basis(sigma_q_r, sigma_k_r, cqk_r, eps=1e-4)
P_K_r = out_r["P_K"].double()
P_K_inv_r = out_r["P_K_inv"].double()
rho_r = out_r["rho"].double()
d_r = sigma_q_r.shape[-1]

# Pick a non-trivial bit allocation (waterfill-like with skew, but no zero coords for cleaner MC)
bits_r = torch.linspace(8.0, 0.5, d_r).double()  # decreasing, all positive

# (a) Trace formula (proposed fix)
sigma_k_diag_canonical = (P_K_r @ sigma_k_r @ P_K_r.T).diagonal().clamp_min(1e-30)
fix_weights = (P_K_inv_r.T @ sigma_q_r @ P_K_inv_r).diagonal()
trace_D = (fix_weights * sigma_k_diag_canonical * 2.0 ** (-2.0 * bits_r)).sum().item()

# (b) Buggy simulation
buggy_weights = rho_r ** 2
buggy_D = (buggy_weights * sigma_k_diag_canonical * 2.0 ** (-2.0 * bits_r)).sum().item()

# (c) Monte-Carlo estimate
# Build joint covariance from (sigma_q, sigma_k, cqk).
# (Sigma_Q is approximate so use a bit of regularization)
joint_cov = torch.zeros(2 * d_r, 2 * d_r).double()
joint_cov[:d_r, :d_r] = sigma_q_r + 1e-6 * torch.eye(d_r).double() * sigma_q_r.diag().mean()
joint_cov[d_r:, d_r:] = sigma_k_r + 1e-6 * torch.eye(d_r).double() * sigma_k_r.diag().mean()
joint_cov[:d_r, d_r:] = cqk_r
joint_cov[d_r:, :d_r] = cqk_r.T
# Make joint PSD by checking eigvals; if not, regularize more
joint_eigs = torch.linalg.eigvalsh(joint_cov)
if joint_eigs.min() < 0:
    print(f"  joint cov min eigval: {joint_eigs.min().item():.6e}; regularizing")
    joint_cov += (-joint_eigs.min().item() + 1e-6) * torch.eye(2 * d_r).double()
joint_chol = torch.linalg.cholesky(joint_cov)
n_samples_r = 200_000
z = torch.randn(n_samples_r, 2 * d_r).double() @ joint_chol.T
q_samples = z[:, :d_r]
k_samples = z[:, d_r:]

# Quantization noise in canonical-K basis: variance σ²_j(CCA) · 2^{-2 b_j}
noise_std = torch.sqrt(sigma_k_diag_canonical * 2.0 ** (-2.0 * bits_r))
eps_samples = torch.randn(n_samples_r, d_r).double() * noise_std
delta_k = eps_samples @ P_K_inv_r.T  # un-rotate

mc_D = ((q_samples * delta_k).sum(dim=-1) ** 2).mean().item()
print(f"  D trace-formula (proposed fix):  {trace_D:.4f}")
print(f"  D Monte-Carlo (n={n_samples_r}):    {mc_D:.4f}  (relative err {100*abs(trace_D - mc_D)/mc_D:.2f}%)")
print(f"  D buggy simulation:              {buggy_D:.4f}  (relative err {100*abs(buggy_D - mc_D)/mc_D:.2f}%)")

assert abs(trace_D - mc_D) / mc_D < 0.10, f"Trace formula should match MC within 10%; got {abs(trace_D - mc_D)/mc_D:.4f}"
print("  ✓ Trace-formula fix matches Monte-Carlo on real Qwen3-8B data.")
print(f"  ✓ Buggy simulation under-predicts by {mc_D / buggy_D:.1f}× on this (layer, head) pair.")

print()
print("=" * 76)
print("ALL VERIFICATIONS PASSED")
print("F8 bug claim is correct; proposed fix gives the right answer in synthetic and real cases.")
print("=" * 76)
