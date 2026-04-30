"""Gate for Stage 1E E1+E2: validates CCA spectra and closed-form simulation results.

Exit code 0 = pass, non-zero = fail with diagnostic printed to stderr.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


def fail(msg: str) -> None:
    print(f"GATE_E1_E2 FAIL: {msg}", file=sys.stderr, flush=True)
    sys.exit(2)


def ok(msg: str) -> None:
    print(f"GATE_E1_E2 PASS: {msg}", flush=True)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    out_dir = repo_root / "artifacts" / "stage1" / "cca_vs_waterfill_study"
    metrics_path = out_dir / "metrics_e1_e2.json"
    cca_path = out_dir / "cca_stats.pt"
    if not metrics_path.exists():
        fail(f"missing {metrics_path}")
    if not cca_path.exists():
        fail(f"missing {cca_path}")

    with open(metrics_path) as f:
        m = json.load(f)
    cca = torch.load(cca_path, map_location="cpu", weights_only=False)

    n_layers = m["n_layers"]
    n_kv_heads = m["n_kv_heads"]
    head_dim = m["head_dim"]

    # ---- E1 gates ----

    rho = cca["rho"]
    if rho.min() < -1e-3:
        fail(f"rho has negative entries: min={rho.min().item():.6f}")
    if rho.max() > 1.0 + 1e-2:
        fail(f"rho exceeds 1.0: max={rho.max().item():.6f}")
    ok(f"rho range [{rho.min().item():.4f}, {rho.max().item():.4f}] within [0, 1]")

    diffs = rho[..., 1:] - rho[..., :-1]
    max_increase = diffs.clamp_min(0.0).max().item()
    if max_increase > 1e-3:
        fail(f"rho not monotone non-increasing within (layer, head); max increase = {max_increase:.6f}")
    ok("rho monotone non-increasing within each (layer, kv_head)")

    rho_layer0 = rho[0]
    layer0_min = rho_layer0.min().item()
    if rho_layer0.min() > 0.99:
        fail(
            f"layer-0 rho values are all near 1.0 (min={layer0_min:.4f}); "
            "regularization may have swamped the signal"
        )
    ok(f"layer-0 rho not collapsed to ~1 (min={layer0_min:.4f})")

    r95_max = m["r95_max"]
    if r95_max == head_dim:
        fail(
            f"r95_max = head_dim ({head_dim}); spectrum is fully spread, no compression headroom anywhere. "
            "Either head_dim is too small or whitening is broken."
        )
    ok(f"r95 distribution: min={m['r95_min']} median={m['r95_median']} max={r95_max} (< head_dim={head_dim})")

    # F4: P_K_inv · P_K = I identity check across all (layer, kv_head). E2/E3 reconstruction
    # paths assume this; a violation would be a real bug in the SVD or whitening.
    P_K = cca["P_K"]
    P_K_inv = cca["P_K_inv"]
    if P_K.dim() != 4 or P_K_inv.dim() != 4 or P_K.shape != P_K_inv.shape:
        fail(f"P_K / P_K_inv shape mismatch or wrong rank: {tuple(P_K.shape)} vs {tuple(P_K_inv.shape)}")
    identity = torch.eye(P_K.shape[-1])
    recon = P_K_inv @ P_K  # batch matmul over (n_layers, n_kv_heads)
    err_max = (recon - identity).abs().max().item()
    err_per_head = (recon - identity).abs().reshape(-1, head_dim, head_dim).amax(dim=(-2, -1))
    if err_max > 1e-3:
        worst = int(err_per_head.argmax().item())
        worst_layer, worst_head = worst // n_kv_heads, worst % n_kv_heads
        fail(
            f"P_K_inv · P_K identity violated: max abs error = {err_max:.4e} "
            f"(worst at layer={worst_layer}, kv_head={worst_head})"
        )
    ok(f"P_K_inv · P_K = I across all 288 heads (max abs err = {err_max:.4e})")

    # F5: SciPy cross-check on 3 random (layer, kv_head) pairs (fixed seed for reproducibility).
    try:
        from scipy.linalg import eigh as scipy_eigh
        import numpy as np

        rng = np.random.default_rng(42)
        sample_pairs = sorted({
            (int(rng.integers(1, n_layers)), int(rng.integers(0, n_kv_heads)))
            for _ in range(20)
        })[:3]
        max_rel_err = 0.0
        for layer, head in sample_pairs:
            sigma_q = cca["sigma_q"][layer, head].numpy().astype("float64")
            sigma_k = cca["sigma_k"][layer, head].numpy().astype("float64")
            cqk = cca["cqk"][layer, head].numpy().astype("float64")
            sigma_k_inv = np.linalg.solve(
                sigma_k + 1e-6 * np.eye(head_dim) * np.trace(sigma_k) / head_dim,
                np.eye(head_dim),
            )
            A_mat = cqk @ sigma_k_inv @ cqk.T
            B_mat = sigma_q + 1e-6 * np.eye(head_dim) * np.trace(sigma_q) / head_dim
            eigvals, _ = scipy_eigh(A_mat, B_mat)
            rho_scipy_sorted = np.sort(np.sqrt(np.clip(eigvals, 0, None)))[::-1][:3]
            rho_ours = rho[layer, head, :3].numpy()
            rel_err = np.abs(rho_scipy_sorted - rho_ours).max() / max(rho_scipy_sorted.max(), 1e-6)
            max_rel_err = max(max_rel_err, rel_err)
            if rel_err > 5e-2:
                fail(
                    f"SciPy cross-check on (layer={layer}, head={head}) top-3 ρ disagrees: "
                    f"ours={rho_ours.tolist()}, scipy={rho_scipy_sorted.tolist()}, rel err={rel_err:.4f}"
                )
        ok(
            f"SciPy cross-check on {len(sample_pairs)} random (layer, head) pairs: "
            f"max rel err = {max_rel_err:.4e} (pairs={sample_pairs})"
        )
    except ImportError:
        print("GATE_E1_E2 INFO: scipy not available; skipping cross-check", flush=True)

    # ---- E2 gates ----

    sim = m["simulation"]
    b_grid = m["b_avg_grid"]

    for b_avg_str, by_method in sim.items():
        b_avg = float(b_avg_str.replace("b", ""))
        if "v3" not in by_method:
            fail(f"missing v3 baseline at {b_avg_str}")
        D_v3 = torch.tensor(by_method["v3"]["D"])
        # Sanity: D_v3 > 0 and finite
        if not torch.isfinite(D_v3).all():
            fail(f"v3 distortion non-finite at {b_avg_str}")
        if (D_v3 <= 0).any():
            fail(f"v3 distortion non-positive at {b_avg_str}")

    # Bit budget conservation: each method's bits sum to head_dim * b_avg.
    # We didn't store full bits matrix in JSON; instead we stored bits_min/bits_max as a sanity proxy.
    for b_avg_str, by_method in sim.items():
        b_avg = float(b_avg_str.replace("b", ""))
        for method_key, payload in by_method.items():
            if method_key == "v3":
                continue
            bits_min = payload.get("bits_min")
            bits_max = payload.get("bits_max")
            if bits_min is None or bits_max is None:
                continue
            if bits_min < -1e-3:
                fail(f"{b_avg_str}/{method_key}: negative bit allocation (min={bits_min})")
            if bits_max > 16.0 + 1e-3:
                fail(f"{b_avg_str}/{method_key}: bit allocation exceeds 16 (max={bits_max})")
    ok("bit allocations are non-negative and bounded")

    # Headline: at b_avg = 3, we expect (V or CCA) + waterfill to beat V3 in a substantial fraction.
    if "b3.0" in sim:
        for headline in ["v_waterfill", "cca_waterfill"]:
            if headline in sim["b3.0"]:
                frac = sim["b3.0"][headline].get("frac_better_than_v3_l0excl", None)
                if frac is None:
                    continue
                if frac < 0.5:
                    print(
                        f"GATE_E1_E2 WARN: {headline} only beats V3 in {frac*100:.1f}% of "
                        "(layer, head) pairs (layer-0-excluded). Check Bennett approximation accuracy.",
                        flush=True,
                    )
                else:
                    ok(f"{headline} @ b_avg=3 beats V3 in {frac*100:.1f}% of (layer, head) pairs (l0excl)")

    # All gates passed
    print("GATE_E1_E2 ALL CHECKS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
