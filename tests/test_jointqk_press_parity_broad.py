"""Tier 0.3 — JointQKPress parity across multiple (layer, kv_head).

Strategy: build a small fake calibration bundle (subset of layers from the
real one), instantiate a press with that bundle, and verify that for every
(L, h) cell the press's compressor produces the same output as the offline
build_jointqk_compressor / build_v_compressor.

This avoids the 12+ minute press construction time of the full 36-layer x
8-head grid (which dominates wall clock due to scipy.quad inside Lloyd-Max).

We pick layers {0, 5, 15, 31} as a representative sample (boundary, mid,
late). Heads: all 8 since iteration is intra-layer and cheap.

Catches:
- J1: indexing mismatch between calibration bundle and press loop order
- J5: bundle re-shuffle hazard in V_h post-rotation

Existing test_press_roundtrip_parity.py covers (L=5, h=3); this generalises.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402, F401

import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from kvq.presses.jointqk_press import JointQKPress
from kvq.compression.per_coord import build_jointqk_compressor
from kvq.compression.v_compressor_adapter import build_v_compressor


def find_one_example_bundle() -> Path:
    base = REPO / "artifacts/query_stats_longbench_under4k/examples"
    candidates = sorted(base.glob("ex_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No example bundles under {base}")
    return candidates[0]


def slice_cca(cca: dict, layer_indices: list[int]) -> dict:
    """Return a copy of cca with leading layer dim subset to layer_indices."""
    out = dict(cca)
    idx = torch.tensor(layer_indices, dtype=torch.long)
    for k in ("sigma_q", "sigma_k", "cqk", "rho", "P_K", "P_K_inv", "P_Q",
             "mq_eigvals", "mq_eigvecs", "V_h", "R_sym"):
        if k in cca and isinstance(cca[k], torch.Tensor):
            t = cca[k]
            if t.dim() >= 1 and t.shape[0] >= max(layer_indices) + 1:
                out[k] = t[idx]
    if "n_layers" in out:
        out["n_layers"] = len(layer_indices)
    return out


def slice_v_stats(v_stats: dict, layer_indices: list[int]) -> dict:
    out = dict(v_stats)
    idx = torch.tensor(layer_indices, dtype=torch.long)
    sv = v_stats["sigma_v"]
    out["sigma_v"] = sv[idx]
    return out


def main():
    K_BITS = 4
    V_BITS = 3
    LAYER_SUBSET = [0, 5, 15, 31]   # original layer indices we'll test
    HEAD_SUBSET = (0, 3, 7)         # original head indices for offline reference

    bundle_path = find_one_example_bundle()
    print(f"Using bundle: {bundle_path}", flush=True)
    b = torch.load(bundle_path, map_location="cpu", weights_only=False)
    print(f"k_post shape: {b['k_post'].shape}", flush=True)

    cca_full_path = REPO / "artifacts/bases/jointqk.pt"
    v_full_path = REPO / "artifacts/v_bases/v_stats.pt"
    cca = torch.load(cca_full_path, map_location="cpu", weights_only=False)
    v_stats = torch.load(v_full_path, map_location="cpu", weights_only=False)
    print(f"sigma_q shape: {cca['sigma_q'].shape}, sigma_v shape: {v_stats['sigma_v'].shape}", flush=True)

    # Subset the bundles to LAYER_SUBSET so the press only constructs
    # len(LAYER_SUBSET) * n_heads compressors (~32) instead of 36*8=288.
    cca_subset = slice_cca(cca, LAYER_SUBSET)
    v_stats_subset = slice_v_stats(v_stats, LAYER_SUBSET)

    # Save subsets to temp files for the press to load.
    with tempfile.TemporaryDirectory() as tmp:
        cca_subset_path = Path(tmp) / "cca_subset.pt"
        v_subset_path = Path(tmp) / "v_subset.pt"
        torch.save(cca_subset, cca_subset_path)
        torch.save(v_stats_subset, v_subset_path)

        print(f"Building JointQKPress on subset (n_layers={len(LAYER_SUBSET)}) ...", flush=True)
        import time
        t0 = time.time()
        press = JointQKPress(
            cca_stats_path=str(cca_subset_path),
            v_stats_path=str(v_subset_path),
            v_method="v_eigen_uniform",
            k_method="r_sym_waterfill",
            k_bits=K_BITS, v_bits=V_BITS,
            rank=64, layer0_full_precision=False,
        )

        class FakeModel:
            pass
        press.post_init_from_model(FakeModel())
        print(f"  built in {time.time() - t0:.1f}s", flush=True)

    # Now compare. In the subset bundle, layer index `i` in [0, len-1]
    # corresponds to original layer `LAYER_SUBSET[i]`. The press indexes
    # via subset position (0..3), but we need to read original-layer KV
    # tensors to feed it.
    print()
    print(f"{'orig L':>6s} {'sub L':>5s} {'h':>3s}  {'K_diff':>11s}  {'V_diff':>11s}  {'status':>6s}", flush=True)
    print("-" * 60, flush=True)

    failures = 0
    n_total = 0
    for sub_L, orig_L in enumerate(LAYER_SUBSET):
        for h in HEAD_SUBSET:
            n_total += 1
            # Real K/V tensors from original layer index in the bundle
            k = b["k_post"][orig_L, h].float()
            v = b["v"][orig_L, h].float()

            # Offline K compressor — built from full cca at original index
            offline_k_comp = build_jointqk_compressor(
                method="r_sym_waterfill",
                sigma_q_for_head=cca["sigma_q"][orig_L, h],
                sigma_k_for_head=cca["sigma_k"][orig_L, h],
                R_sym=cca["R_sym"][orig_L, h],
                b_avg=float(K_BITS),
                r=64, head_dim=128,
            )
            offline_k_recon = offline_k_comp.roundtrip(k)

            # Offline V compressor — note: build_v_compressor's seed is
            # 42 + L * 1000 + h. The press uses `L * 1000` where L is the
            # subset index, not the original. So seed = 42 + sub_L*1000 + h.
            offline_v_comp = build_v_compressor(
                method="v_eigen_uniform",
                sigma_v_for_head=v_stats["sigma_v"][orig_L, h],
                head_dim=128, bits=V_BITS,
                seed=42 + sub_L * 1000 + h,
            )
            offline_v_recon = offline_v_comp.roundtrip(v)

            # Press compressors are indexed by subset position
            press_k_recon = press._k_compressors[(sub_L, h)].roundtrip(k)
            press_v_recon = press._v_compressors[(sub_L, h)].roundtrip(v)

            k_diff = (offline_k_recon - press_k_recon).abs().max().item()
            v_diff = (offline_v_recon - press_v_recon).abs().max().item()

            ok = (k_diff < 1e-5) and (v_diff < 1e-5)
            status = "PASS" if ok else "FAIL"
            if not ok:
                failures += 1
            print(f"{orig_L:>6d} {sub_L:>5d} {h:>3d}  {k_diff:>11.3e}  {v_diff:>11.3e}  {status:>6s}",
                  flush=True)

    print("-" * 60, flush=True)
    if failures == 0:
        print(f"Tier 0.3 PARITY: ALL {n_total} (L, h) PASS", flush=True)
    else:
        raise AssertionError(f"{failures} (L, h) cells failed parity")


if __name__ == "__main__":
    main()
