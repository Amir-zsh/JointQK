"""Does the codebook GROUP SIZE fix what no kernel change could? INCONCLUSIVE.

Every kernel-side lever failed because none reduces L1 sector volume, which ncu
identifies as the binding constraint (vq2: 392 M sectors at 73.8% of L1 peak;
int2: 94 M at 30.4%). Sector volume looks like a property of the *format*, not
the code:

    loads per (token, head) = NG
    bytes fetched per load  = 32 (one sector), of which G are useful

At the shipped G=4 / NG=32 that is 32 loads each wasting 87.5% of a sector, so
G=8 should halve the loads and double the useful fraction -- ~2x less L1 traffic,
about the size of the whole gap. This probe was written to measure that.

*** DO NOT TRUST THE NUMBERS BELOW YET. ***

The probe is not validated and behaves like the round-1 microbenchmarks this
study was burned by:

  - Changing only the ACCUMULATOR dtype (int32 -> int64), which is not part of
    the trade under test, moved the G=4 baseline from 1898 us to 996 us. A
    baseline that halves under an irrelevant edit is measuring codegen, not the
    format.
  - With single wide loads, every larger group measures SLOWER (G=8: 0.57-0.69x,
    G=16: 0.87x), contradicting the sector argument above with no mechanism that
    explains it. Changing NG also changes the tile shape [BLOCK_N, NG] and hence
    Triton's layout and vectorisation choices, so the comparison is not clean.

Two readings remain open: either the sector model is wrong (bytes-per-sector is
not what binds), or the probe confounds group size with tile layout. Resolving it
needs ncu sector counts per configuration, not timings -- if sectors halve for
G=8 while time does not improve, the sector model is the thing that is wrong.

Until that is done this is a research note, not a recommendation, and no format
change should be argued from it. Note also that G=4 is assumed by
`vq_codebook.py:170` (codewords packed into int32) and that larger groups
quantise worse at equal KC, so even a confirmed win here is only the decode-side
half of the trade.

  docker exec -e CUDA_VISIBLE_DEVICES=0 oscar-ab bash -lc 'cd <repo> && \
    /opt/venv-oscar/bin/python pipelines/throughput/kernel_study/probe_group_size.py'
"""import os

import torch
import triton
import triton.language as tl

L = 128                      # head_dim, fixed
KC = 256                     # codewords per group, fixed
GRID = int(os.environ.get("GRID", 1536))   # ~ the served CTA count (64*8*3)
TOKENS = int(os.environ.get("TOKENS", 8192))


@triton.jit
def _gather(Idx, CB, Out, S,
            NG: tl.constexpr, G: tl.constexpr, KC: tl.constexpr,
            ELEM: tl.constexpr, BLOCK_N: tl.constexpr):
    """Gather NG codewords of G bytes per token and reduce, the shipped kernel's
    access pattern with the group geometry as a parameter. The codeword is read
    as G/4 int32 words, matching how a packed codebook is actually laid out."""
    pid = tl.program_id(0)
    # Words per codeword in the element type CB was passed as. A G=8 codeword
    # fetched as TWO int32 loads is still two lookups, so it buys nothing; it has
    # to be ONE 8-byte load for the lookup count to actually halve.
    W: tl.constexpr = G // ELEM
    Idx += pid * NG * S
    CB += pid * NG * KC * W
    offs_g = tl.arange(0, NG)
    acc = tl.zeros([BLOCK_N, NG], dtype=tl.int64)
    for start_n in range(0, S, BLOCK_N):
        n = start_n + tl.arange(0, BLOCK_N)
        idx = tl.load(Idx + n[:, None] * NG + offs_g[None, :]).to(tl.int32)
        base = CB + offs_g[None, :] * (KC * W) + idx * W
        for w in tl.static_range(W):
            acc += tl.load(base + w).to(tl.int64)
    tl.store(Out + pid * NG + offs_g, tl.sum(acc, 0))


def bench(fn, reps=50):
    fn(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps):
        fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


def main() -> int:
    dev = "cuda"
    print(f"head_dim={L} KC={KC} grid={GRID} tokens/CTA={TOKENS}   "
          f"(coords decoded per CTA held fixed at {TOKENS * L:,})")
    print(f"\n{'NG':>4} {'G':>3} {'sub-table':>10} {'codeword':>9} "
          f"{'loads/tok':>12} {'us':>9} {'vs G=4':>8} {'useful/sector':>14}")
    base = None
    for ng, g, elem in ((32, 4, 4), (16, 8, 4), (16, 8, 8), (8, 16, 8)):
        assert ng * g == L
        w = g // elem
        dt = torch.int32 if elem == 4 else torch.int64
        idx = torch.randint(0, KC, (GRID, TOKENS, ng), device=dev, dtype=torch.uint8)
        cb = torch.randint(-(2 ** 30), 2 ** 30, (GRID, ng, KC, w), device=dev,
                           dtype=dt)
        out = torch.empty(GRID, ng, device=dev, dtype=torch.int64)
        fn = lambda: _gather[(GRID,)](idx, cb, out, TOKENS, NG=ng, G=g, KC=KC,
                                      ELEM=elem, BLOCK_N=64, num_warps=2,
                                      num_stages=3)
        try:
            ms = bench(fn)
        except Exception as e:
            print(f"{ng:>4} {g:>3}  FAILED {type(e).__name__}: "
                  f"{str(e).splitlines()[0][:70]}")
            continue
        if base is None:
            base = ms
        print(f"{ng:>4} {g:>3} {KC*g:>9}B {g:>8}B {ng*w:>12} "
              f"{ms*1e3:>9.1f} {base/ms:>7.2f}x {g:>10}/32 B")

    print("\nINCONCLUSIVE -- see the module docstring. The G=4 baseline moved 2x\n"
          "under an unrelated accumulator-dtype change, so these timings are\n"
          "measuring codegen as much as format. Resolve with ncu sector counts\n"
          "per configuration before arguing any format change from this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
