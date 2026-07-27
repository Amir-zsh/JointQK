"""Head-to-head: scattered tl.load (what vq2 does today) vs tl.gather.

The cycle model says tl.gather's warp-local lowering (~34 instrs/element) costs
about what the L1 sector traffic it replaces costs, i.e. no win. This measures
both in isolation so the kernel rewrite is decided on data, not on my arithmetic.

Both kernels do the same thing: for each of NG groups, look up a codeword per
token from that group's 256-entry table, and accumulate.

  docker exec oscar-ab bash -lc '<repo> && /opt/venv-oscar/bin/python logs/probe_gather_ab.py'
"""
import torch
import triton
import triton.language as tl


@triton.jit
def k_load(idx_ptr, cb_ptr, out_ptr, S,
           NG: tl.constexpr, K: tl.constexpr, BLOCK_N: tl.constexpr):
    """Today's shape: scattered dependent loads into the codebook (L1-bound)."""
    pid = tl.program_id(0)
    idx_ptr += pid * NG * S
    cb_ptr += pid * NG * K
    out_ptr += pid * NG * BLOCK_N
    g = tl.arange(0, NG)[:, None]
    acc = tl.zeros([NG, BLOCK_N], dtype=tl.float32)
    for start_n in range(0, S, BLOCK_N):
        n = start_n + tl.arange(0, BLOCK_N)[None, :]
        idx = tl.load(idx_ptr + g * S + n)
        acc += tl.load(cb_ptr + g * K + idx)
    o = tl.arange(0, NG)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    tl.store(out_ptr + o, acc)


@triton.jit
def k_gather(idx_ptr, cb_ptr, out_ptr, S,
             NG: tl.constexpr, K: tl.constexpr, BLOCK_N: tl.constexpr):
    """Proposed shape: hoist the table, gather on-chip."""
    pid = tl.program_id(0)
    idx_ptr += pid * NG * S
    cb_ptr += pid * NG * K
    out_ptr += pid * NG * BLOCK_N
    g = tl.arange(0, NG)[:, None]
    kk = tl.arange(0, K)[None, :]
    cb = tl.load(cb_ptr + g * K + kk)
    acc = tl.zeros([NG, BLOCK_N], dtype=tl.float32)
    for start_n in range(0, S, BLOCK_N):
        n = start_n + tl.arange(0, BLOCK_N)[None, :]
        idx = tl.load(idx_ptr + g * S + n)
        acc += tl.gather(cb, idx, axis=1)
    o = tl.arange(0, NG)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    tl.store(out_ptr + o, acc)


def main() -> int:
    NG, K, BLOCK_N, S = 32, 256, 128, 32768
    GRID = 528          # 132 SMs x 4 CTAs: saturated, not latency-bound
    dev = "cuda"
    cb = torch.randn(GRID, NG, K, device=dev, dtype=torch.float32)
    idx = torch.randint(0, K, (GRID, NG, S), device=dev, dtype=torch.int32)
    out = torch.empty(GRID, NG, BLOCK_N, device=dev, dtype=torch.float32)

    print(f"NG={NG} K={K} BLOCK_N={BLOCK_N} S={S}  (one (layer,head) slice)")
    for name, fn in (("scattered tl.load (today)", k_load),
                     ("tl.gather (proposed)", k_gather)):
        for warps in (4, 8):
            f = lambda: fn[(GRID,)](idx, cb, out, S, NG=NG, K=K, BLOCK_N=BLOCK_N,
                                 num_warps=warps, num_stages=3)
            ms = triton.testing.do_bench(f, warmup=25, rep=100)
            kern = f()
            elems = GRID * NG * S
            print(f"  {name:28s} warps={warps}  {ms*1e3:8.1f} us   "
                  f"{elems/ms/1e6:7.2f} G lookups/s   shared={kern.metadata.shared}B")

    # Codebook size drives L1 sector spread: a K-entry table of 4 B codewords
    # is K*4/32 sectors, and 32 random lanes touch about that many distinct
    # sectors (saturating at 32). Smaller K -> fewer transactions -> faster,
    # at the cost of bits/coord. This maps the trade.
    print(f"\n=== lookup rate vs codebook size K (NG={NG}, scattered load) ===")
    print(f"  {'K':>5} {'bits/coord':>11} {'sub-table':>10} {'sectors':>8} "
          f"{'us':>9} {'G lookups/s':>13} {'vs K=256':>9}")
    base = None
    for kk in (16, 32, 64, 128, 256):
        cb2 = torch.randn(GRID, NG, kk, device=dev, dtype=torch.float32)
        idx2 = torch.randint(0, kk, (GRID, NG, S), device=dev, dtype=torch.int32)
        f = lambda: k_load[(GRID,)](idx2, cb2, out, S, NG=NG, K=kk,
                                 BLOCK_N=BLOCK_N, num_warps=8, num_stages=3)
        ms = triton.testing.do_bench(f, warmup=25, rep=100)
        rate = GRID * NG * S / ms / 1e6
        if kk == 256:
            base = rate
        sect = max(1, kk * 4 // 32)
        print(f"  {kk:>5} {kk.bit_length()-1:>8}/4 c {kk*4:>9}B {sect:>8} "
              f"{ms*1e3:>9.1f} {rate:>13.2f} "
              f"{'' if base is None else f'{rate/base:>8.2f}x'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
