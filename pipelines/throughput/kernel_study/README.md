# vq2 decode-kernel study (2026-07-27)

Investigation into the vq2-vs-OSCAR-INT2 decode gap. **No net kernel win was
achieved.** This directory exists so the measurements are not re-derived: it
records what the bottleneck actually is, and which plausible fixes are dead.

Two things *did* land in the engine from this work, both outside this directory:
the pool-accounting fix (`pool_configurator.py`, +40% vq2 capacity) and the
batch-adaptive launch config (`decode_attention.py`, 1.13× at bs=64).

## The bottleneck

Ablating the real Triton kernel at the served shape (bs=64, ctx=30k, engine-derived splits):

| ablation | time | implies |
|---|---|---|
| full | 1934 µs | |
| no codebook gather | 925 µs | **gather = 1009 µs (52%)** |
| no join-interleave | 1837 µs | join = 96 µs (5%) |

int2 at the same shape is 1634 µs. **Remove the gather and vq2 beats int2** —
the entire gap is the codebook gather.

Why it is slow: `isel` is `[BLOCK_N, NG]`, so Triton maps threads along NG and a
warp's 32 lanes hit 32 *different* group sub-tables (`g*KC + isel`, 1 KB apart),
spanning the whole 32 KB codebook. Maximum address divergence — measured
487 G lookups/s in-kernel, matching ~600 G/s for isolated scattered gathers.

## Dead ends (all measured, do not retry)

| idea | result |
|---|---|
| `tl.gather` (Triton on-chip gather) | **1.7–4.3× SLOWER**; lowers to warp shuffles, 34 instr/element, 512-select tree |
| shared-memory codebook staging, fused | 3.58× in isolation, **0.40× fused** — its shared footprint costs more than the gather it saves |
| smaller codebook (K=256→16) | only 1.28×, for half the bit rate |
| split-K changes | vq2 pinned at ~588 tok/s across splits 1→8 |
| deeper pipelining (`num_stages` 2/3/4) | flat to 0.2% |
| DRAM/compute overlap | already achieved — index traffic (237 µs) fully hidden inside a 922 µs gather |
| eliminating `tl.trans(kg)` | **0.86× (slower)** — Triton folds it into `ldmatrix.trans`, it was never a real transpose |
| int2's large-batch config (32/4/1) on vq2 | worse (538 vs 593 tok/s); vq2's dependent gather needs more warps in flight than int2's bit-unpack |
| full TileLang rewrite | 0.40× at the served shape (see below) |

## The one lever not yet tried

Flip `isel` to **group-major `[NG, BLOCK_N]`**: a warp then reads 32 tokens
within *one* group's 1 KB sub-table — a **32× reduction in per-warp address
spread**, and it needs no shared memory, so it works in Triton.

Blocker: `tl.join` only interleaves the trailing axis, so group-major planes come
out `(g, n, m)` where `(g, m, n)` is needed. It requires the **strided coordinate
layout** the V-side VQ path already uses (`decode_attention.py`, V_VQ branch) —
group `g` holding coords `{g, g+NG, g+2NG, g+3NG}` — under which the four planes
concatenate with no interleave. That relabeling is an exact, lossless permutation
that folds offline into `forward` and `q_map`, but needs a new codebook bundle.

## Files

- `bench_stage1.py` — **the most reusable thing here.** 96 launch configs in ~20 s at the *served* shape, using the engine's own split heuristic. Build this before optimising anything.
- `tl_vq2_stage1.py` — fused vq2 stage-1 in TileLang. **Numerically correct** (rel 3.4e-04 vs Triton) but 0.40× at the served shape. Needs `.venv-tilelang` (gitignored; `pip install tilelang`).
- `test_tl_vq2.py` — validates TileLang against Triton. Two-process (`--triton` / `--tilelang`): the engine venv ships a TVM that collides with TileLang's.
- `probe_regs_int2_vs_vq2.py` — compiled-artifact comparison. vq2: 255 regs, 2 spills, 30 KB shared, 2 CTAs/SM. int2: 246 regs, 0 spills, 3.7 KB, 8 CTAs/SM.
- `probe_gather_ab.py`, `probe_shared_gather.cu`, `probe_tilelang_gather.py` — gather-rate ladder: Triton 600 → CUDA 1232 → CUDA+shared 1649 → TileLang+shared+vectorized 2147 G lookups/s.
- `probe_tl_fp8.py` — e5m2→f16 is exactly `byte << 8` (verified over all 256 values).
- `probe_tl_frag_gemm.py` — TileLang `T.gemm` accepts a fragment A operand.
- `probe_overlap.py`, `sweep_vq2_cfg.py` — overlap and config sweeps.

## Methodological warning

Five separate times a microbenchmark's ranking **inverted** when measured
end-to-end, because its regime differed from the served one: int32 indices
(4× the real DRAM traffic), `GRID=1` (one CTA, latency-bound), `BATCH=2` (128
CTAs on 132 SMs), the isolated gather (no tile materialisation, 64× more tokens
per CTA), and hardcoded `splits=8` (the engine derives splits from occupancy).

Every one over-promised. `bench_stage1.py` is the corrected form — real shape,
engine-derived splits — and its ranking does agree with end-to-end. Validate any
new probe against one known end-to-end pair before trusting it.
