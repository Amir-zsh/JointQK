"""Parity test: press path produces same recon as offline build_jointqk_compressor.

For one (layer, kv_head) on a Qwen example, run:
- offline: build_jointqk_compressor(...).roundtrip(k)
- press:   JointQKPress(...)._k_compressors[(L,h)].roundtrip(k)
Assert max-abs-diff < 1e-5.

Same for V via build_v_compressor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from kvq.toolkit.jointqk_press import JointQKPress
from kvq.toolkit.per_coord_quantization import build_jointqk_compressor
from kvq.toolkit.v_compressor_adapter import build_v_compressor


def main():
    LAYER, HEAD = 5, 3
    K_BITS = 4
    V_BITS = 2

    # Load real K/V tensor and stats
    bundle = Path("artifacts/query_stats_longbench_under4k/examples/ex_000.pt")
    b = torch.load(bundle, map_location="cpu", weights_only=False)
    k = b["k_post"][LAYER, HEAD].float()  # (S, D)
    v = b["v"][LAYER, HEAD].float()

    cca_path = "artifacts/bases/cca_stats.pt"
    cca = torch.load(cca_path, map_location="cpu", weights_only=False)
    v_path = "artifacts/v_bases/v_stats.pt"
    v_stats = torch.load(v_path, map_location="cpu", weights_only=False)

    # Offline K
    offline_k_comp = build_jointqk_compressor(
        method="r_sym_waterfill",
        sigma_q_for_head=cca["sigma_q"][LAYER, HEAD],
        sigma_k_for_head=cca["sigma_k"][LAYER, HEAD],
        R_sym=cca["R_sym"][LAYER, HEAD],
        b_avg=float(K_BITS),
        r=64,
        head_dim=128,
    )
    offline_k_recon = offline_k_comp.roundtrip(k)

    # Offline V
    offline_v_comp = build_v_compressor(
        method="v_eigen_waterfill",
        sigma_v_for_head=v_stats["sigma_v"][LAYER, HEAD],
        head_dim=128,
        bits=V_BITS,
        seed=42 + LAYER * 1000 + HEAD,
    )
    offline_v_recon = offline_v_comp.roundtrip(v)

    # Press path: instantiate JointQKPress and load same compressors
    press = JointQKPress(
        cca_stats_path=cca_path,
        v_stats_path=v_path,
        v_method="v_eigen_waterfill",
        k_method="r_sym_waterfill",
        k_bits=K_BITS,
        v_bits=V_BITS,
        rank=64,
    )

    class FakeModel:
        pass
    press.post_init_from_model(FakeModel())

    press_k_recon = press._k_compressors[(LAYER, HEAD)].roundtrip(k)
    press_v_recon = press._v_compressors[(LAYER, HEAD)].roundtrip(v)

    k_diff = (offline_k_recon - press_k_recon).abs().max().item()
    v_diff = (offline_v_recon - press_v_recon).abs().max().item()

    print(f"K parity max-abs-diff: {k_diff:.2e}")
    print(f"V parity max-abs-diff: {v_diff:.2e}")

    assert k_diff < 1e-5, f"K parity FAILED ({k_diff:.4e})"
    assert v_diff < 1e-5, f"V parity FAILED ({v_diff:.4e})"
    print("PARITY TEST PASSED")


if __name__ == "__main__":
    main()
