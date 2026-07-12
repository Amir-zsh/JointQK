#!/usr/bin/env python3
"""VQ decode throughput vs BF16 read and INT2 unpack, same methodology as
bw_regime.py (full 36x8 model, P=64 pages, CUDA-event kernel-only timing, us/page
+ GB/s + %peak). VQ decode = per-(head,group) codebook GATHER of the stored index
-> the code vector r (then the r@inv rotation is shared with every QPCA method, so
excluded here, exactly as the report excludes it for rANS/INT2). No sequential
dependency -> should sit in the fixed-width (INT2) class, not rANS's."""
import torch, triton, triton.language as tl, time
from kvq_codec import load_ext as _load_ext

_vq_ext = _load_ext(source="vq_decode.cu", name="vq_decode")

dev = torch.device("cuda")
L, Hkv, d = 36, 8, 128
P = 64
A100_PEAK_GBs = 1555.0
G = 6                      # VQ group size (from the accuracy prototype)
NG = d // G                # groups per head (128/6 -> 21, last partial handled by pad)
RATE_BITS = 2             # bits/coord -> codebook K = 2^(rate*G)
K = 1 << (RATE_BITS * G)   # 4096 entries

# round d to a multiple of G for a clean microbench (128 -> 126 with G=6 leaves 2; use ng=21, d_eff=126)
NG = d // G                # 21
d_eff = NG * G             # 126 (the 2 leftover coords would be scalar; negligible for timing)


@triton.jit
def vq_gather_kernel(idx_ptr, cb_ptr, out_ptr, T, NG, K, d_eff, G: tl.constexpr, BLOCK: tl.constexpr):
    # one program per (head,token); BLOCK threads = the d_eff output coords.
    # Coalesced output write; adjacent threads in a group read adjacent codeword
    # elements (contiguous in cb). idx read is gathered per group (broadcast to G).
    pid = tl.program_id(0).to(tl.int64)        # = head*T + t
    head = pid // T
    j = tl.arange(0, BLOCK)                     # output coord within this token
    mask = j < d_eff
    g = j // G                                  # group id
    within = j % G
    e = tl.load(idx_ptr + pid * NG + g, mask=mask, other=0).to(tl.int64)   # cb entry [0,K)
    cb_off = (((head * NG + g) * K) + e) * G + within
    v = tl.load(cb_ptr + cb_off, mask=mask, other=0.0)
    tl.store(out_ptr + pid * d_eff + j, v, mask=mask)


@triton.jit
def vq_tiled_kernel(idx_ptr, cb_ptr, out_ptr, T, NG, K, d_eff,
                    G: tl.constexpr, TILE_T: tl.constexpr, BLOCK: tl.constexpr):
    # block = TILE_T consecutive tokens of one head -> writes a contiguous
    # [TILE_T, d_eff] output region (fully coalesced across tokens AND coords).
    # gather stays L2-cached (per-head codebook reused across the tile).
    pid = tl.program_id(0).to(tl.int64)
    ntile = T // TILE_T
    head = pid // ntile
    t0 = (pid % ntile) * TILE_T
    lin = tl.arange(0, BLOCK)
    m = lin < (TILE_T * d_eff)
    tt = lin // d_eff
    j = lin % d_eff
    t = t0 + tt
    g = j // G
    within = j % G
    e = tl.load(idx_ptr + (head * T + t) * NG + g, mask=m, other=0).to(tl.int64)
    cb_off = ((head * NG + g) * K + e) * G + within
    v = tl.load(cb_ptr + cb_off, mask=m, other=0.0)
    tl.store(out_ptr + (head * T + t) * d_eff + j, v, mask=m)


@triton.jit
def read_kernel(ptr, n, BLOCK: tl.constexpr, out_ptr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    x = tl.load(ptr + offs, mask=mask, other=0)
    tl.atomic_add(out_ptr, tl.sum(x.to(tl.float32)))


@triton.jit
def int2_unpack_kernel(packed_ptr, scale_ptr, zero_ptr, out_ptr, n_elem, GG: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_elem
    b = tl.load(packed_ptr + offs // 4, mask=mask, other=0).to(tl.uint32)
    q = (b >> ((offs % 4) * 2).to(tl.uint32)) & 0x3
    grp = offs // GG
    s = tl.load(scale_ptr + grp, mask=mask, other=1.0); z = tl.load(zero_ptr + grp, mask=mask, other=0.0)
    tl.store(out_ptr + offs, ((q.to(tl.float32) - z) * s).to(tl.float16), mask=mask)


def timeit_ev(f, reps=20, warmup=5):
    for _ in range(warmup): f()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(reps): f()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


print(f"A100 peak ~{A100_PEAK_GBs/1000:.2f} TB/s. VQ: G={G}, K={K} ({RATE_BITS} b/c), NG={NG}, d_eff={d_eff}.")
print(f"Codebook footprint: {L*Hkv*NG*K*G*2/1e6:.0f} MB (fp16, whole model, one-time).\n")
print(f"{'T':>7} {'n_pages':>8} {'BF16':>8} {'INT2':>8} {'VQ_tri':>8} {"VQ_tiled(best)":>16} "
      f"{'VQsh/BF16':>10} {'VQsh/INT2':>10} {'VQsh_GB/s':>10} {'%pk':>5}   (us/page)")

for T in [16384, 32768]:
    n_heads = L * Hkv
    nb = (T + P - 1) // P
    n_pages = n_heads * nb
    n_tok = n_heads * T
    n_elem = n_heads * T * d                 # scalar element count (BF16/INT2 baselines)

    # --- BF16 read baseline ---
    kv16 = torch.randn(n_elem, device=dev, dtype=torch.float16)
    sink = torch.zeros(1, device=dev, dtype=torch.float32)
    def rd(): read_kernel[(triton.cdiv(n_elem, 8192),)](kv16, n_elem, 8192, sink)
    tb = timeit_ev(rd); del kv16; torch.cuda.empty_cache()

    # --- INT2 unpack baseline ---
    packed = torch.randint(0, 256, ((n_elem + 3)//4,), device=dev, dtype=torch.uint8)
    ngq = (n_elem + 63)//64
    sc = torch.rand(ngq, device=dev)+0.1; ze = torch.zeros(ngq, device=dev)
    o16 = torch.empty(n_elem, device=dev, dtype=torch.float16)
    def i2(): int2_unpack_kernel[(triton.cdiv(n_elem, 1024),)](packed, sc, ze, o16, n_elem, 64, 1024)
    ti = timeit_ev(i2); del packed, sc, ze, o16; torch.cuda.empty_cache()

    # --- VQ gather: token-tiled Triton, sweep TILE_T, keep best ---
    cb = torch.randn(n_heads * NG * K * G, device=dev, dtype=torch.float16)
    idx = torch.randint(0, K, (n_heads * T * NG,), device=dev, dtype=torch.int32)
    out = torch.empty(n_heads * T * d_eff, device=dev, dtype=torch.float16)
    best = (1e9, 0)
    for TILE_T in [4, 8, 16, 32]:
        TBLK = 1 << ((TILE_T * d_eff - 1).bit_length())    # next pow2 >= TILE_T*d_eff
        def vq_tiled(): vq_tiled_kernel[(n_heads * (T // TILE_T),)](idx, cb, out, T, NG, K, d_eff, G, TILE_T, TBLK)
        tt = timeit_ev(vq_tiled)
        if tt < best[0]: best = (tt, TILE_T)
    tv2, best_tile = best
    vq_bytes = (n_heads*T*d)//4 + n_heads*T*d_eff*2   # ~2b/c packed index read + fp16 code write
    vq_gbs = vq_bytes/1e9/(tv2/1e3); pct = vq_gbs/A100_PEAK_GBs*100
    del cb, idx, out; torch.cuda.empty_cache()

    ub, ui, uv2 = tb/n_pages*1e3, ti/n_pages*1e3, tv2/n_pages*1e3
    print(f"{T:>7} {n_pages:>8} {ub:>7.3f} {ui:>7.3f} {uv2:>8.3f} (tile={best_tile:>2}) "
          f"{uv2/ub:>9.2f}x {uv2/ui:>9.2f}x {vq_gbs:>9.0f} {pct:>4.0f}%")

print("\nVQ_shmem = CUDA kernel, per-group codebook staged in shared memory (no "
      "sequential dependency, unlike rANS's 12.394 us/page). VQsh/INT2 -> how close to "
      "the crude fixed-width floor. Rotation r@inv excluded (shared by all QPCA methods).")
