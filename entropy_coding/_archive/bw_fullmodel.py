#!/usr/bin/env python3
"""Full-model (288-head) decode cost in the FUSED-KERNEL / bandwidth-bound framing
-- the correct one for the serving-relevant question, all 7 methods, real byte
counts. Supersedes decode_timing_fullmodel.py, which measured the WRONG thing:
that script materialized a full-precision reconstructed K back to HBM via a dense
fp64 inverse-rotation matmul, turning a bandwidth measurement into a compute one
(and BF16, whose "decode" is a no-op, trivially "won"). See chat 2026-07-08.

Correct fused-kernel framing (this file), matching bw_clean.py / vq_throughput_fused.py:
  - A real fused attention-decode kernel READS the compressed KV from HBM (few
    bytes), reconstructs in SRAM/registers, and does q.k^T there -- it NEVER
    writes a full-precision K back to HBM.
  - The inverse rotation is fused into the QUERY side (rotate the single q vector
    once, keys stay in coded space), so it is NOT an O(T.d^2) per-key cost. Excluded.
  - Therefore the decode cost of a fixed-width method is its HBM READ of the
    compressed representation: time = bytes / bandwidth. Reconstruction (unpack,
    affine dequant, codebook gather) is a handful of in-register ops hidden under
    the read.

Two regimes fall out, and the table labels them:
  BANDWIDTH-BOUND (BF16, INT2, TurboQuant, VQ, OSCAR): time tracks bytes read.
    The saturating wide-load kernel (from bw_clean.py) runs at ~90% peak HBM BW,
    so this is a real hardware floor, not an idealization. 2-bit methods read ~8x
    fewer bytes than BF16 -> ~7x faster decode.
  COMPUTE-BOUND (rANS, Exp-Golomb): read the same few bytes, but their SERIAL
    entropy decode is not a bandwidth op and cannot be hidden under the read.
    Real measured serial-decode rates (kernel-only, from eg_vs_rans_matched.py):
    rANS 12.394 us/page, Exp-Golomb ~1.9 us/page. These dominate and make the
    entropy coders far slower than even BF16's read despite their tiny byte count
    -- the accuracy/rate win of entropy coding is paid for in decode latency.

Byte counts are REAL: fixed-width are exact by construction; OSCAR from its own
BPE accounting (oscar_codec.bits_per_coord, incl. BF16 sink/recent + scale/zero);
rANS/Exp-Golomb from their measured achieved rate on real held-out K (2.132 /
2.131 bits/coord, from the A6/B4 runs on calib_idx=[0,5,6]).
"""
import torch, triton, triton.language as tl, time

dev = torch.device("cuda")
L, Hkv, d = 36, 8, 128
PEAK = 1555.0          # A100 HBM peak GB/s
P = 64                 # tokens/page (rANS/EG codec convention)
G = 6; NG = d // G     # VQ group size / groups per head

# Real measured serial entropy-decode rates (kernel-only, eg_vs_rans_matched.py,
# full 280-head grid, calib_idx=[0,5,6]).
RANS_US_PER_PAGE = 12.394
EG_US_PER_PAGE = 1.9          # ~m=1.65 rate-matched (B4: m=1.0->1.403, m=1.75->1.975)
# Real achieved compressed rate on held-out K (A6/B4), bits/coord.
RANS_BITS_PER_COORD = 2.132
EG_BITS_PER_COORD = 2.131
# OSCAR mixed-precision cache (paper defaults).
OSCAR_SINK, OSCAR_RECENT = 64, 256


@triton.jit
def read_bw_kernel(ptr, n_i32, BLOCK: tl.constexpr, out):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = offs < n_i32
    x = tl.load(ptr + offs, mask=m, other=0)
    tl.store(out + pid, tl.sum(x))


def read_bytes(buf_bytes):
    """Time a saturating wide-load read of a buffer of buf_bytes (the bandwidth floor)."""
    n_i32 = max(1, buf_bytes // 4)
    b = torch.randint(-2**31, 2**31 - 1, (n_i32,), device=dev, dtype=torch.int32)
    BLOCK = 4096
    grid = (triton.cdiv(n_i32, BLOCK),)
    out = torch.zeros(grid[0], device=dev, dtype=torch.int32)
    def f(): read_bw_kernel[grid](b, n_i32, BLOCK, out)
    for _ in range(10): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(30): f()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / 30
    del b, out; torch.cuda.empty_cache()
    return dt


def oscar_bytes(n_heads, T):
    """OSCAR real byte count: INT2 middle + BF16 sink/recent + BF16 scale/zero
    (single group per token, oscar default G=d)."""
    s = min(OSCAR_SINK, T)
    r = min(max(T - s, 0), OSCAR_RECENT)
    m = T - s - r
    per_head = m * d * 0.25 + (s + r) * d * 2 + m * 2 * 2   # int2 + bf16 band + (scale,zero) fp16
    return int(per_head * n_heads)


print(f"A100 peak ~{PEAK/1000:.2f} TB/s. Full model: {L}x{Hkv}={L*Hkv} heads. "
      f"Fused-kernel framing (read compressed KV, reconstruct in SRAM, rotation query-side).\n")
header = (f"{'T':>7} {'method':>11} {'B/coord':>8} {'read GB':>8} {'read ms':>8} "
         f"{'%pk':>5} {'decode ms':>10} {'vs BF16':>8}  regime")
for T in [32768, 65536, 100000]:
    n_coord = L * Hkv * T * d
    n_heads = L * Hkv
    n_pages = n_heads * ((T + P - 1) // P)
    print(header)
    # (name, bytes, decode_ms_override or None, regime)
    specs = [
        ("BF16",       n_coord * 2,                                    None, "bandwidth"),
        ("INT2",       n_coord // 4,                                   None, "bandwidth"),
        ("TurboQuant", n_coord // 4 + n_heads * T * 2,                 None, "bandwidth"),
        ("VQ",         n_coord // 4,                                   None, "bandwidth"),  # 2b/coord packed
        ("OSCAR",      oscar_bytes(n_heads, T),                        None, "bandwidth"),
        ("rANS",       int(n_coord * RANS_BITS_PER_COORD / 8),
                       RANS_US_PER_PAGE * n_pages / 1e3,                     "compute(serial)"),
        ("Exp-Golomb", int(n_coord * EG_BITS_PER_COORD / 8),
                       EG_US_PER_PAGE * n_pages / 1e3,                       "compute(serial)"),
    ]
    tb = None
    for name, nbytes, dec_override, regime in specs:
        nbytes = (nbytes + 3) // 4 * 4
        t_read = read_bytes(nbytes)
        if name == "BF16":
            tb = t_read
        gbs = nbytes / 1e9 / t_read; pct = gbs / PEAK * 100
        dec_ms = dec_override if dec_override is not None else t_read * 1e3
        bpc = nbytes / n_coord
        speed = (tb * 1e3) / dec_ms
        print(f"{T:>7} {name:>11} {bpc:>6.3f}B {nbytes/1e9:>6.2f}GB {t_read*1e3:>7.3f} "
              f"{pct:>4.0f}% {dec_ms:>9.3f} {speed:>7.2f}x  {regime}")
    print()

print("BANDWIDTH-BOUND (BF16/INT2/TurboQuant/VQ/OSCAR): decode ms = the saturating HBM read "
      "of the compressed KV (~90% peak). 2-bit methods read ~8x fewer bytes => ~7x faster than BF16.")
print("COMPUTE-BOUND (rANS/Exp-Golomb): decode ms = real serial entropy-decode kernel "
      "(rANS 12.394 us/page, EG ~1.9 us/page) x pages -- reads few bytes but the serial decode "
      "can't be hidden under the read, so far slower than even BF16 despite the tiny byte count.")
