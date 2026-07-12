#!/usr/bin/env python3
"""T-sweep: how does attention cost scale with sequence length, single head.
fp16 flash vs Triton residual attn vs decode estimate. Synthetic K/V/Q (timing
only — flash FLOPs/bytes don't care if data is real). Finds where flash stops
being free, i.e. where the bandwidth path could win."""
import numpy as np, torch, time
import torch.nn.functional as Fnn
from torch.nn.attention import sdpa_kernel, SDPBackend
from step2b_triton import triton_resid_attn

dev = torch.device("cuda")
d = 128; gs = 4
sm = 1.0 / np.sqrt(d)
Ts = [2048, 4096, 8192, 16384, 32768, 65536, 131072]

# decode cost model: entropy kernel ~ linear in T (tokens). From your runs:
# ~550ms / 280 heads at T=3743 -> per-head-per-token. We report decode est per head.
DECODE_MS_PER_HEAD_AT_3743 = 550.0 / 280.0      # ~1.96 ms/head entropy kernel
def decode_est(T):  # linear in T
    return DECODE_MS_PER_HEAD_AT_3743 * (T / 3743.0)

def timeit(f, reps=5, warmup=2):
    for _ in range(warmup): f()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3

print(f"{'T':>7} {'fp16_flash':>12} {'triton_attn':>12} {'decode_est':>12} "
      f"{'flash_O(T^2)?':>14} {'decode/flash':>12}")
prev_tf = None
for T in Ts:
    # synthetic tensors, correct shapes
    q = torch.randn(gs, T, d, device=dev, dtype=torch.float32)
    k = torch.randn(T, d, device=dev, dtype=torch.float32)
    v = torch.randn(T, d, device=dev, dtype=torch.float32)

    def run_fp16():
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            Fnn.scaled_dot_product_attention(
                q.half().unsqueeze(0),
                k.half().unsqueeze(0).unsqueeze(0).expand(1, gs, T, d),
                v.half().unsqueeze(0).unsqueeze(0).expand(1, gs, T, d),
                is_causal=True)

    def run_triton():
        triton_resid_attn(q.contiguous(), k.contiguous(), v.contiguous(), sm)

    try:
        tf = timeit(run_fp16)
    except RuntimeError as e:
        print(f"{T:>7} fp16 OOM/err: {str(e)[:40]}"); break
    try:
        tt = timeit(run_triton)
    except RuntimeError as e:
        tt = float('nan')
    de = decode_est(T)
    # O(T^2) check: flash time ratio vs previous T (should ~4x per 2x T if compute-bound)
    ratio = f"{tf/prev_tf:.2f}x/2xT" if prev_tf else "  -"
    prev_tf = tf
    print(f"{T:>7} {tf:>10.2f}ms {tt:>10.2f}ms {de:>10.2f}ms {ratio:>14} {de/tf:>10.2f}x")

print("\nReading it:")
print("- flash O(T^2)?: if ~4x per 2x T, compute-bound; if ~2x, bandwidth/linear-bound.")
print("- decode/flash: when this drops below ~1, decode is cheaper than attention ->")
print("  the regime where fused/bandwidth path can win. >1 everywhere = flash too cheap, no win.")