# vq2 decode-kernel study

Investigation into the vq2-vs-OSCAR-INT2 decode gap on H100 (Qwen3-8B geometry:
`H_Q=32, H_KV=8, L=128, NG=32, KC=256`), at the served shape `bs=64, ctx=30000`
with engine-derived split-K.

**Status: closed.** The shipped Triton kernel runs at **1.21x int2**. A CUDA
kernel with the codebook in shared memory runs at **0.94-1.05x int2** and
**0.75-0.83x Triton** across four shapes -- i.e. the vq2 decode-kernel penalty
against OSCAR-INT2 is gone. Two engine fixes did land
from this work (elsewhere): the pool-accounting fix (`pool_configurator.py`,
+40% vq2 capacity) and the batch-adaptive launch config (`decode_attention.py`,
1.13x at bs=64).

Round 2 (2026-07-27) replaced the round-1 diagnosis, which was wrong.

**Narrative write-up:** `notes/page_quant/studies/vq2_decode_kernel_report.md`.
This file stays the measurement log; that one explains the reasoning.

## The bottleneck, measured

Round 1 inferred the bottleneck by ablation (delete a load, diff the runtime)
because Nsight Compute appeared unavailable. It concluded "maximum address
divergence". **Hardware counters say otherwise.** `profile_ncu.py` gets counters
by running ncu in a *privileged sibling container* -- the driver ships with
`RmProfilingAdminOnly: 1` so counters need CAP_SYS_ADMIN, which the normal
`oscar-ab` container lacks, but a sibling container gets them without touching
the host driver.

| metric | int2 | vq2 | group-major vq2 |
|---|---|---|---|
| time | 1968 us | 2386 us | 2473 us |
| global ld requests | 5.3 M | **20.2 M** | 20.2 M |
| global ld sectors | 94.1 M | **392.4 M** | 570.5 M |
| **sectors / request** | 17.77 | **19.43** | 28.19 |
| **L1 throughput** | 30.4% | **73.8%** | 67.5% |
| L1 hit rate | 32.7% | 51.2% | 65.3% |
| L2 sectors | 100.2 M | 250.6 M | 249.0 M |
| DRAM read | 1.46 GB | 2.20 GB | 2.59 GB |
| warps active | 9.9% | 11.9% | 12.1% |

Three things follow, and they overturn round 1:

1. **Sectors per request are nearly identical** (17.8 vs 19.4). Address
   divergence is *not* what separates the two kernels. Lanes already share
   sectors (0.75 sectors/lane, not the 1.0 the round-1 model predicted).
2. **vq2 is L1-sector-throughput bound** -- 4.2x int2's sector count, at 74% of
   L1 peak. The codebook gather is ~94% of that traffic.
3. The traffic cascades: 392 M L1 sectors -> 251 M L2 sectors (~4.2 TB/s, near
   L2 peak) -> **0.74 GB more DRAM than int2 despite identical bytes/token**.

Both kernels are also register-limited to ~8 warps/SM (12% occupancy): int2 runs
1 warp/CTA x 8 CTAs, vq2 2 warps/CTA x 4 CTAs.

The decisive control is group-major: it has the **best L1 hit rate (65.3%) and
is the slowest**. Hit rate is not the lever; sector count is.

## Why nothing in Triton fixes it

32 lanes fetching 4 B each out of a 32 KB table cost ~24 sectors per warp
instruction *no matter how the table is addressed*. Every Triton-level lever
leaves that number alone, and all of them measure as no-ops or regressions:

| idea | result |
|---|---|
| group-major `isel` `[NG, BLOCK_N]` | **1.13-1.15x SLOWER**; best L1 hit rate of any variant, most sectors |
| four K=NG dots, no `kg` tile | **1.19x slower**; registers unchanged at 249, so `kg` was not the register driver |
| `eviction_policy` evict_first/evict_last (9 combos) | 0.99-1.00x -- **and the PTX is byte-identical to no hints; Triton drops them, so this measured nothing** (dump_triton_asm.py) |
| `cache_modifier` `.cg` / `.cs` (L1 bypass) | 1.01-1.03x -- same: dropped before PTX |
| launch-config sweep (96 configs) | **wrong** -- it capped num_stages at 3. The gather is 32 cp.async per token, so num_stages is the pipeline depth; at stages>=4 the optimum moves to BLOCK_H=4 and the kernel goes from 1.02-1.34x int2 to 0.97-1.10x |
| `tl.gather` (Triton on-chip gather) | **1.7-4.3x SLOWER**; lowers to warp shuffles, 34 instr/element |
| shared-memory codebook staging, fused | 3.58x in isolation, **0.40x fused** |
| smaller codebook (KC 256->16) | only 1.28x, for half the bit rate |
| split-K changes | vq2 pinned at ~588 tok/s across splits 1->8 |
| deeper pipelining (`num_stages` 2/3/4) | flat to 0.2% |
| eliminating `tl.trans(kg)` | **0.86x (slower)** -- Triton folds it into `ldmatrix.trans` |
| int2's large-batch config (32/4/1) on vq2 | worse (538 vs 593 tok/s) |
| full TileLang rewrite | 0.40x at the served shape |

The round-1 note claimed group-major was blocked by `tl.join` needing a
strided-coordinate codebook bundle. **That blocker does not exist**: decomposing
`qk[h,n] = sum_m dot(q_m, p_m)` with `q_m[h,g] = Q[h, 4g+m]` consumes the planes
in their native layout using four K=NG dots, needing no bundle change. It was
implemented, validated (7.16e-07 vs shipped) -- and it is simply slower.

## What does work: the codebook in shared memory

Shared memory has no sectors. `probe_cuda_qk.py` isolates exactly the contested
piece (gather + score, no V/softmax/PV) and runs identical work both ways:

| | time | vs Triton |
|---|---|---|
| Triton gather+qk | 1134 us | 1.00x |
| CUDA, shared codebook, fp32 | 753 us | 1.51x |
| CUDA, shared codebook, half2 + `prmt` | **684 us** | **1.66x** |

Confirmed by counters: global ld sectors **331 M -> 37.4 M (8.8x)**. Two
optimisations mattered beyond the staging itself:

- **`prmt` byte-permute** decodes two e5m2 bytes to one `half2` in ONE
  instruction (`__byte_perm(cw, 0, 0x1404)` builds `[0, b0, 0, b1]`, which is
  exactly the pair of `<<8` shifts e5m2->fp16 needs). The scalar path costs 8
  instructions for the same 4 coords.
- **Thread coarsening (TPT)** amortises the `q` broadcast loads across tokens.
  At TPT=1 those broadcasts are 4/5 of all shared loads and L1 sits at 93%.

`half2` accumulates the score in fp16: 2.3e-03 relative on qk, 3.2e-04 on the
final stage-1 output (softmax damps it). The fp32 path is kept for comparison.

## The full CUDA kernel: the gap to int2 is closed

`cuda_vq2_stage1.py` is a complete, validated stage-1 (rel out 3.2e-04, rel lse
1.0e-06 against the shipped kernel). Measured across four shapes, not just the
one it was tuned on:

| shape | int2 | Triton vq2 | **CUDA vq2** | vs int2 | vs Triton |
|---|---|---|---|---|---|
| bs=64 ctx=30k | 1602 us | 1942 us (1.21x) | **1604 us** | 1.00x | **0.83x** |
| bs=32 ctx=60k | 1593 us | 1934 us (1.21x) | **1503 us** | **0.94x** | **0.78x** |
| bs=16 ctx=30k | 368 us | 501 us (1.36x) | **375 us** | 1.02x | **0.75x** |
| bs=64 ctx=8k | 402 us | 536 us (1.33x) | **423 us** | 1.05x | **0.79x** |

Config: `PV=3`, half2, `TPT=1`, `THR=128`, `GUNR=32`, `PUNR=4`.

### What got it there, in order of contribution

1. **Codebook in shared memory** (Triton cannot express this): global load
   sectors 331 M -> 37.4 M.
2. **PV register tiling (`PV=3`)** -- 2452 -> 1949 us. Tokens partitioned across
   warps so each token is touched by ONE warp; each lane owns the four quarters
   of ONE packed V byte (`{lane, lane+32, lane+64, lane+96}` are exactly that
   byte's four 2-bit fields), so one byte plus one `float4` of `p` feeds 16
   FMAs; and the (scale, zero) affine folded out of the inner loop via
   `sum p*((q2-z)*s) = sum (p*s)*q2 - sum (p*s)*z`.
3. **FULL unroll of the 32-group score loop** (`GUNR` 4 -> 32) -- 1843 -> 1630 us.
   The single biggest win after the shared codebook. The score accumulators are
   a serial dependency chain; without unrolling there is almost no ILP.
4. **q half2 pair as ONE 8 B load** -- 1977 -> 1843 us. Loading the two adjacent
   `__half2` separately costs 256 shared loads/token against the fp32 path's 128
   `float4` loads, which is exactly why half2 won in the isolated probe (FMA
   bound) but LOST in the full kernel (shared bound).
5. **Unrolling the PV loop 4x** (`PUNR=4`) -- 1630 -> 1608 us. Same ILP problem;
   its dynamic bound stops the compiler unrolling it by default.
6. **Dropping `vsc_s`/`vze_s`** -- write-only once the affine is folded -- takes
   shared from 47104 to 46080 B, crossing the 5-CTA/SM threshold (227 KB / 5).

### Phase decomposition (ablations `PV=6/7/8/9`, at the tuned config)

| phase | time | share |
|---|---|---|
| gather + score | 937 us | 58% |
| PV loop | 452 us | 28% |
| V global load | 216 us | 13% |
| softmax | 25 us | 2% |

### Measured worse -- do not retry

| idea | result | why |
|---|---|---|
| bulk-stage the tile through shared | 2609 vs 2032 us | extra shared write+read and two more barriers cost more than the latency hidden |
| software-pipeline block bi+1 into registers | 1966 vs 1930 us | doubled register state costs more occupancy than it buys |
| 256 threads/CTA | 1718-2310 vs 1604 us | `BLOCK_N = THR*TPT` grows the tile, so shared grows too |
| cross-warp reduction via shared `atomicAdd` (to shrink 8 KB -> 2 KB) | 2562 vs 1934 us, twice | occupancy went UP and the kernel got 30% worse |
| q-reuse kernel (`PV=5`): decouple the TPT group from the staging block | 2565 vs 2011 us | per-sub-pass barriers and `iw[TPT][8]` register pressure outweigh the saved broadcasts |
| LUT kernel (`PV=10`): fold q into the table | 1887 vs 1604 us | 707 M instructions vs 852 M and 49.7 M shared loads vs 80.6 M, but the 64 KB table drops occupancy to 3 CTAs/SM and 8 B elements give 1.30 bank conflicts/load vs 0.48 |

### Tensor cores: the PV bound was right, the QK bound was not

Two ablations, and the difference between them is the lesson. `PV=4` removes 3
of every 4 PV FMAs; `PV=9` removes 3 of every 4 SCORE FMAs (and their q loads):

| ablation | time | implies |
|---|---|---|
| PV=3 (full) | 1879 us | |
| PV=4 (3/4 of PV FMAs gone) | 1882 us -> saves 46 us | all PV FMAs cost **~61 us** |
| PV=9 (3/4 of score FMAs gone) | 1026 us -> saves 853 us | score work costs **~1137 us** |

So the PV arithmetic is essentially free (already overlapped with memory), while
the score arithmetic is fully exposed. An earlier round of this study measured
only the PV ablation and concluded "tensor cores cannot help" -- that conclusion
was drawn from the wrong phase. What actually recovered the score cost was not
tensor cores but **ILP**: fully unrolling the group loop.

## Why the TileLang attempt loses (0.40x)

Profiled rather than guessed at. It is not fighting the hardware, it is fighting
the lowering:

| metric | Triton vq2 | TileLang |
|---|---|---|
| time | 1951 us | 4846 us |
| global ld requests | 20.2 M | **36.8 M** |
| shared bank conflicts | -- | **227.8 M** on 102.3 M shared loads |
| inst executed | 893 M | **1517 M** (1.7x) |
| stall long_scoreboard | 0.33 | **3.12** |

It *does* stage the codebook in shared and *does* get the best L1 hit rate
(78.5%), but it reloads `K_Idx` per coordinate rather than per group (36.8 M
global requests, more than Triton's 20.2 M), and pays 2.2 bank conflicts per
shared load. `BLOCK_H` is padded to 16 for the MMA when only `VALID_H=4` rows are
real, so 3/4 of the MMA work is wasted.

## Files

- `profile_ncu.py` -- **start here.** ncu comparison table (sectors/request, L1
  throughput, hit rates, spills, bank conflicts, stall reasons). Needs the
  privileged sibling container; the docstring has the `docker run` line.
- `bench_stage1.py` -- 96 launch configs in ~20 s at the *served* shape using the
  engine's own split heuristic.
- `vq2_variants.py` -- group-major, four-dot, and cache-policy variants, each
  validated against the shipped kernel before timing.
- `probe_cuda_qk.py` -- the decisive gather+qk probe, CUDA vs Triton.
- `cuda_vq2_stage1.py` -- the full CUDA stage-1. `PV=3` is the fast warp-tiled
  path; `PV=0/1` are the earlier assignments kept as measurements; `PV=2`/`PV=4`
  are FMA ablations bounding the tensor-core ceiling for each structure.
- `probe_group_size.py` -- **INCONCLUSIVE**, read its docstring before citing it.
  Asks whether a larger codebook group (G=8/16) cuts sector volume; its baseline
  moved 2x under an unrelated dtype change, so it is currently measuring codegen
  as much as format.
- `tl_vq2_stage1.py`, `test_tl_vq2.py` -- the TileLang kernel and its two-process
  validator (the engine venv's TVM collides with TileLang's).
- `probe_regs_int2_vs_vq2.py`, `probe_gather_ab.py`, `probe_shared_gather.cu`,
  `probe_tilelang_gather.py`, `probe_tl_fp8.py`, `probe_tl_frag_gemm.py`,
  `probe_overlap.py`, `sweep_vq2_cfg.py` -- round-1 probes.

## Methodological warning

Round 1 had five microbenchmarks whose ranking **inverted** when measured
end-to-end, because their regime differed from the served one: int32 indices
(4x the real DRAM traffic), `GRID=1` (one CTA, latency-bound), `BATCH=2` (128
CTAs on 132 SMs), the isolated gather (no tile materialisation), and hardcoded
`splits=8` (the engine derives splits from occupancy).

Round 2 adds a sixth and more expensive failure mode: **inferring a bottleneck by
ablation when counters were one container away.** The round-1 conclusion
("maximum address divergence") sent the study after addressing layouts for a
long time; the counters showed in one run that sectors/request were nearly
identical between the two kernels and the real difference was sector *volume*.
Check whether ncu is reachable before reasoning about memory behaviour from
timings.

`probe_group_size.py` is a live instance of the same hazard, left in the tree
deliberately and labelled: it disagrees with a first-principles sector argument
and its baseline is unstable, so it is recorded as an open question rather than
a finding.

Two CUDA-specific traps, both caught by counters and both worth 3-5x:

- `reinterpret_cast<const uint8_t*>(&local_uint4)` to index bytes makes nvcc
  demote the variable to **local memory** (150 M local loads, 9.5 GB DRAM, 5x
  slower than Triton). Extract bytes by shifting instead. The same trap bit again
  in the pipelining attempt: indexing a register buffer with a RUNTIME slot
  (`buf[cur][t]`) demotes it too, and cost 43% until the buffers were named and
  rotated by register copy.
- An online-softmax rescale applied AFTER the block it should precede makes the
  kernel fast and silently wrong (rel 4.2e-01). `corr` is accumulated in phase B,
  so it must be rescaled before phase B, not with the accumulators after it.
- Fully unrolling a 512-FMA body makes the compiler hoist all 512 `q` values
  into registers and spill (255 regs). `#pragma unroll 4` fixes it.
