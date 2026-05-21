"""Tier 0.2 — TurboQuantPress vs direct TurboQuantV3 byte-exact parity.

For seeded random (B,H,S,D) tensors, verify our wrapper produces byte-exact
output vs calling TurboQuantV3.compress_kv → decompress_kv directly.

Cases:
  A: short on CPU (no chunking)
  B: short on CUDA (lazy GPU move triggered)
  C: long on CUDA, exactly 2 chunks (S=4096)
  D: long on CUDA, non-divisible boundary (S=8193)
  E: short on CUDA with protected_layers=4 (layer in protected band)

Construction cost: TurboQuantV3 builds Lloyd-Max codebooks (scipy.quad) per
instance. We reuse one TurboQuantPress across cases of the same config to
avoid rebuilding 32 codebook stacks per case.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402, F401

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from kvq.toolkit.turboquant_press import TurboQuantPress
from turboquant_pytorch.compressors_v3 import TurboQuantV3


N_LAYERS = 16   # smaller than 32 to keep codebook construction tractable; we
                # still test layer indices well within this band.


@dataclass
class FakeConfig:
    head_dim: int
    num_hidden_layers: int


@dataclass
class FakeModel:
    config: FakeConfig


@dataclass
class FakeModule:
    layer_idx: int


def make_inputs(B, H, S, D, device, seed=0, dtype=torch.float16):
    g = torch.Generator(device="cpu").manual_seed(seed)
    K = torch.randn(B, H, S, D, generator=g).to(device=device, dtype=dtype)
    V = torch.randn(B, H, S, D, generator=g).to(device=device, dtype=dtype)
    return K, V


def direct_recon(K, V, *, k_bits, v_bits, layer_idx, n_layers,
                 residual_window=0, protected_layers=0, seed=42, device="cuda"):
    tq = TurboQuantV3(
        head_dim=K.shape[-1],
        key_bits=k_bits,
        value_bits=v_bits,
        residual_window=residual_window,
        layer_idx=layer_idx,
        n_layers=n_layers,
        protected_layers=protected_layers,
        seed=seed,
        device=device,
    )
    ck, cv = tq.compress_kv(K, V)
    Kr, Vr = tq.decompress_kv(ck, cv)
    return Kr, Vr


def press_recon(press, K, V, layer_idx):
    module = FakeModule(layer_idx=layer_idx)
    return press.compress(module, hidden_states=None, keys=K, values=V,
                          attentions=None, kwargs={})


def assert_byte_exact(label, K_ref, V_ref, K_press, V_press):
    k_diff = (K_ref - K_press).abs().max().item()
    v_diff = (V_ref - V_press).abs().max().item()
    print(f"  {label:<40s}  K_max_abs_diff={k_diff:.3e}  V_max_abs_diff={v_diff:.3e}",
          flush=True)
    assert k_diff == 0.0, f"{label}: K diff non-zero: {k_diff}"
    assert v_diff == 0.0, f"{label}: V diff non-zero: {v_diff}"


def main():
    print("=" * 70, flush=True)
    print("Tier 0.2 — TurboQuantPress vs TurboQuantV3 byte-exact parity", flush=True)
    print(f"n_layers={N_LAYERS}", flush=True)
    print("=" * 70, flush=True)

    K_BITS = 4
    V_BITS = 2

    # Build the shared press once. post_init_from_model is the expensive step
    # (Lloyd-Max codebook construction × n_layers × 2 compressors).
    print(f"Building TurboQuantPress (n_layers={N_LAYERS}, K={K_BITS}, V={V_BITS})...", flush=True)
    t = time.time()
    press = TurboQuantPress(
        k_bits=K_BITS, v_bits=V_BITS, protected_layers=0, residual_window=0, seed=42)
    press.post_init_from_model(FakeModel(FakeConfig(128, N_LAYERS)))
    print(f"  built in {time.time() - t:.1f}s", flush=True)

    # Case A: short on CPU
    K, V = make_inputs(1, 8, 512, 128, device="cpu", seed=1)
    K_ref, V_ref = direct_recon(K, V, k_bits=K_BITS, v_bits=V_BITS, layer_idx=5,
                                n_layers=N_LAYERS, device="cpu")
    K_p, V_p = press_recon(press, K, V, layer_idx=5)
    assert_byte_exact("A: short, CPU", K_ref, V_ref, K_p, V_p)

    if torch.cuda.is_available():
        # Case B: short on CUDA — triggers lazy CPU→GPU move in wrapper
        # NOTE: after this, press's L=5 compressor lives on GPU. Direct comparison
        # uses a separate TurboQuantV3 built directly on GPU. Both paths use
        # identical Pi/centroids (same seed) so output should match byte-exactly.
        K, V = make_inputs(1, 8, 512, 128, device="cuda", seed=2)
        K_ref, V_ref = direct_recon(K, V, k_bits=K_BITS, v_bits=V_BITS, layer_idx=5,
                                    n_layers=N_LAYERS, device="cuda")
        K_p, V_p = press_recon(press, K, V, layer_idx=5)
        assert_byte_exact("B: short, CUDA (lazy move)", K_ref, V_ref, K_p, V_p)

        # Case C: long on CUDA, exactly 2 chunks (use a different layer to
        # exercise a fresh per-layer compressor on GPU)
        K, V = make_inputs(1, 8, 4096, 128, device="cuda", seed=3)
        K_ref, V_ref = direct_recon(K, V, k_bits=K_BITS, v_bits=V_BITS, layer_idx=10,
                                    n_layers=N_LAYERS, device="cuda")
        K_p, V_p = press_recon(press, K, V, layer_idx=10)
        assert_byte_exact("C: 4096, divisible 2-chunk", K_ref, V_ref, K_p, V_p)

        # Case D: long on CUDA, non-divisible boundary (chunk both paths to
        # avoid the (N,D,K) diffs OOM on direct call)
        K, V = make_inputs(1, 8, 8193, 128, device="cuda", seed=4)
        Kr_chunks, Vr_chunks = [], []
        # Build the direct comparison TurboQuantV3 once and reuse across chunks
        tq_direct = TurboQuantV3(
            head_dim=128, key_bits=K_BITS, value_bits=V_BITS,
            residual_window=0, layer_idx=12, n_layers=N_LAYERS,
            protected_layers=0, seed=42, device="cuda")
        for start in range(0, K.shape[2], 2048):
            end = min(start + 2048, K.shape[2])
            ck, cv = tq_direct.compress_kv(K[:, :, start:end], V[:, :, start:end])
            kr, vr = tq_direct.decompress_kv(ck, cv)
            Kr_chunks.append(kr)
            Vr_chunks.append(vr)
        K_ref = torch.cat(Kr_chunks, dim=2)
        V_ref = torch.cat(Vr_chunks, dim=2)
        K_p, V_p = press_recon(press, K, V, layer_idx=12)
        assert_byte_exact("D: 8193, non-divisible chunks", K_ref, V_ref, K_p, V_p)

        # Case E: protected layer (need a press with protected_layers=4)
        print("Building TurboQuantPress (protected_layers=4) for case E...", flush=True)
        t = time.time()
        press_prot = TurboQuantPress(
            k_bits=K_BITS, v_bits=V_BITS, protected_layers=4,
            residual_window=0, seed=42)
        press_prot.post_init_from_model(FakeModel(FakeConfig(128, N_LAYERS)))
        print(f"  built in {time.time() - t:.1f}s", flush=True)

        K, V = make_inputs(1, 8, 512, 128, device="cuda", seed=5)
        K_ref, V_ref = direct_recon(K, V, k_bits=K_BITS, v_bits=V_BITS, layer_idx=2,
                                    n_layers=N_LAYERS, protected_layers=4,
                                    device="cuda")
        K_p, V_p = press_recon(press_prot, K, V, layer_idx=2)
        assert_byte_exact("E: protected_layers=4, L=2", K_ref, V_ref, K_p, V_p)

    print("=" * 70, flush=True)
    print("Tier 0.2 PARITY: ALL CASES BYTE-EXACT", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
