"""KIVI parity test — local PyTorch recon vs official jy-yuan/KIVI pack path.

Three regimes tested:

1. V parity: both implementations group along the D (head_dim) axis and reduce
   per-token.

2. K parity for divisible sequence lengths: both implementations group along
   the T (sequence) axis and reduce per-channel inside each token block.

3. K non-divisible sequence length: local wrapper quantizes the divisible
   prefix and leaves the tail full precision, matching KIVI's residual-cache
   semantics.

4. KIVIPress parity: the kvpress wrapper returns the same recon as the direct
   local functions, including the K residual tail.

Reference: vendor/kivi/quant/new_pack.py (jy-yuan/KIVI, MIT license).
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402, F401

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
# KIVI's new_pack.py is at vendor/kivi/quant/new_pack.py
sys.path.insert(0, str(REPO / "vendor" / "kivi"))

from kvq.compression.kivi_quantizer import (
    kivi_quantize_keys,
    kivi_quantize_values,
)
from kvq.presses.kivi_press import KIVIPress
from quant.new_pack import (
    quant_and_pack_kcache,
    quant_and_pack_vcache,
    unpack_and_dequant_kcache,
    unpack_and_dequant_vcache,
)


def kivi_reference_keys(k, group_size, bits):
    """Round-trip through upstream KIVI's K path: quantize+pack → unpack+dequant."""
    code, scale, mn = quant_and_pack_kcache(k.contiguous().to(torch.float16),
                                            group_size, bits)
    return unpack_and_dequant_kcache(code, scale, mn, group_size, bits)


def kivi_reference_values(v, group_size, bits):
    """Round-trip through upstream KIVI's V path."""
    code, scale, mn = quant_and_pack_vcache(v.contiguous().to(torch.float16),
                                            group_size, bits)
    return unpack_and_dequant_vcache(code, scale, mn, group_size, bits)


def make_kv(B, H, S, D, seed=0, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    K = torch.randn(B, H, S, D, generator=g).to(device=device, dtype=torch.float16)
    V = torch.randn(B, H, S, D, generator=g).to(device=device, dtype=torch.float16)
    return K, V


def report(label, ours, ref, *, label_ours="ours", label_ref="kivi"):
    diff = (ours.float() - ref.float()).abs()
    max_d = diff.max().item()
    mse = diff.pow(2).mean().item()
    rmse = mse ** 0.5
    print(f"  {label:<55s}  max_abs_diff={max_d:.4e}  rmse={rmse:.4e}")
    return max_d, rmse


def assert_close(label, ours, ref, max_abs_tol=1e-3):
    max_d, rmse = report(label, ours, ref)
    if max_d >= max_abs_tol:
        raise AssertionError(
            f"{label}: max_abs_diff={max_d:.4e} exceeds {max_abs_tol:.4e}; "
            f"rmse={rmse:.4e}"
        )


def test_v_parity(device):
    print(f"\n=== V parity, device={device} ===")
    K_BITS = 4
    GROUP_SIZE = 128

    # KIVI requires D % group_size == 0
    cases = [
        ("V: (1,8,512,128) group=128", (1, 8, 512, 128), 128),
        ("V: (1,8,512,128) group=64",  (1, 8, 512, 128), 64),
        ("V: (1,8,512,128) group=32",  (1, 8, 512, 128), 32),
        ("V: (1,8,2048,128) group=128", (1, 8, 2048, 128), 128),
    ]
    for idx, (label, shape, group_size) in enumerate(cases):
        _, V = make_kv(*shape, seed=1000 + idx, device=device)
        ours = kivi_quantize_values(V, bits=K_BITS, group_size=group_size)
        ref = kivi_reference_values(V, group_size=group_size, bits=K_BITS)
        assert_close(label, ours, ref)


def test_k_parity_divisible(device):
    """K parity for the official KIVI grouping: chunks along sequence length."""
    print(f"\n=== K parity, divisible sequence length, device={device} ===")
    K_BITS = 4
    GROUP_SIZE = 128
    cases = [
        ("K: (1,8,128,128) group=128", (1, 8, 128, 128)),
        ("K: (1,8,512,128) group=128", (1, 8, 512, 128)),
        ("K: (1,8,2048,128) group=128", (1, 8, 2048, 128)),
    ]
    for idx, (label, shape) in enumerate(cases):
        K, _ = make_kv(*shape, seed=2000 + idx, device=device)
        ours = kivi_quantize_keys(K, bits=K_BITS, group_size=GROUP_SIZE)
        ref = kivi_reference_keys(K, group_size=GROUP_SIZE, bits=K_BITS)
        assert_close(label, ours, ref)


def test_k_residual_tail(device):
    """For T % group_size != 0, quantize prefix and keep tail full precision."""
    print(f"\n=== K residual-tail parity, device={device} ===")
    K_BITS = 4
    GROUP_SIZE = 128

    cases = [
        ("K: (1,8,129,128)  prefix=128 tail=1", (1, 8, 129, 128)),
        ("K: (1,8,513,128)  prefix=512 tail=1", (1, 8, 513, 128)),
        ("K: (1,8,2079,128) prefix=2048 tail=31", (1, 8, 2079, 128)),
    ]
    for idx, (label, shape) in enumerate(cases):
        K, _ = make_kv(*shape, seed=3000 + idx, device=device)
        prefix_len = (shape[2] // GROUP_SIZE) * GROUP_SIZE
        ref_prefix = kivi_reference_keys(K[:, :, :prefix_len, :], group_size=GROUP_SIZE, bits=K_BITS)
        ref = torch.cat([ref_prefix, K[:, :, prefix_len:, :]], dim=2)
        ours = kivi_quantize_keys(K, bits=K_BITS, group_size=GROUP_SIZE)
        assert_close(label, ours, ref)


def test_press_parity(device):
    """KIVIPress.compress should be equivalent to the direct local functions."""
    print(f"\n=== KIVIPress.compress parity, device={device} ===")
    K_BITS = 4
    V_BITS = 4
    GROUP_SIZE = 128

    cases = [
        ("Press: (1,8,512,128)", (1, 8, 512, 128)),
        ("Press: (1,8,513,128)", (1, 8, 513, 128)),
    ]
    press = KIVIPress(k_bits=K_BITS, v_bits=V_BITS, group_size=GROUP_SIZE)
    for idx, (label, shape) in enumerate(cases):
        K, V = make_kv(*shape, seed=4000 + idx, device=device)
        k_ref = kivi_quantize_keys(K, bits=K_BITS, group_size=GROUP_SIZE)
        v_ref = kivi_quantize_values(V, bits=V_BITS, group_size=GROUP_SIZE)
        k_out, v_out = press.compress(None, None, K, V, None, {})
        assert_close(f"{label} K", k_out, k_ref)
        assert_close(f"{label} V", v_out, v_ref)


def main():
    if not torch.cuda.is_available():
        print("CUDA required (KIVI's triton kernels are CUDA-only)", flush=True)
        sys.exit(1)
    device = "cuda"
    test_v_parity(device)
    test_k_parity_divisible(device)
    test_k_residual_tail(device)
    test_press_parity(device)
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
