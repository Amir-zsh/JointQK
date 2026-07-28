# Closing the vq2 decode-kernel gap against OSCAR-INT2

**H100 (sm90), Qwen3-8B / Qwen3-4B-Thinking-2507 geometry: `H_Q=32, H_KV=8,
head_dim=128, NG=32, KC=256, G=4`.**

Outcome: the vq2 decode penalty against OSCAR-INT2 is gone, by two independent
routes.

1. A **CUDA stage-1** staging the codebook in shared memory: 1.65-1.79× the
   shipped Triton kernel and 1.01-1.13× OSCAR-INT2 end-to-end, on both models at
   every context length. Wired in behind `SGLANG_VQ2_CUDA=1` (default-off).

   **This win is fp16-accumulate only.** The kernel defaults to `half2`
   (`SGLANG_VQ2_CUDA_FP32=0`, rel 3.2e-04 on stage-1 output). With fp32
   accumulation -- the precision OSCAR-INT2 and shipped vq2 Triton both use --
   the advantage disappears entirely. Measured on one box, bs=64/ctx=30000,
   full stage-1: int2 1636-1650 µs, CUDA half2 **1618 µs (0.98×)**, CUDA fp32
   **1827 µs (1.11×)**, retuned Triton 1803 µs (1.10×). So at equal numerics
   nothing here beats OSCAR-INT2; every fp32 path lands at 1.10-1.11×.
   §10.4 records what is left to try.
2. **Retuning the Triton launch ladder** -- no kernel code at all -- takes the
   Triton kernel from 1.02-1.34× to **0.97-1.10× int2** (§6b). This was found
   last, by reading the compiled PTX, and it recovers most of what the CUDA
   rewrite bought.

Read §6b before deciding between those two: on that evidence the retuned Triton
kernel was the honest recommendation, with the CUDA path reserved for the
last ~5%.

**§10 adds a third route, with the same fp32 caveat.** A TileLang kernel
rebuilt with the CUDA kernel's structure measures 1.20× faster than the CUDA
kernel on the contested piece -- **but only in half2**. At fp32 it is 0.85× the
CUDA kernel (883 vs 751 µs) and 1.29× shipped Triton. Its value is portability:
it JIT-specialises per geometry instead of being gated to one by `supports()`,
at a measured fp32 cost of ~18% against CUDA. It is a probe, not a complete
stage-1 -- softmax, PV tiling and the stage-2 interface remain.

**Bottom line under an fp32 constraint: nothing in this report beats
OSCAR-INT2.** All three vq2 routes land at 1.10-1.11×. See §10.4.

Artefacts: `pipelines/throughput/kernel_study/` (probes, variants, profiler
harness), `vendor/OSCAR-vq/.../triton_ops/vq2_cuda_stage1.py` (the kernel),
`notes/throughput_benchmark.png` (the chart).

---

## 1. The problem

vq2 stores each K row as `NG=32` uint8 indices into a per-(layer, kv-head)
codebook of `KC=256` packed-fp8 codewords, plus a per-token fp8 scale. Against
OSCAR-INT2 it is **byte-for-byte identical in cache footprint** — both cost 80 B
per (token, kv-head) — so after the pool-accounting fix the two arms get the same
KV pool and admit the same batch size. Any remaining difference is pure kernel
cost.

And there was one. At the served shape (`bs=64, ctx=30000`, engine-derived
split-K):

| | time |
|---|---|
| `_fwd_grouped_kernel_stage1_quant_int2` | 1602 µs |
| `_fwd_grouped_kernel_stage1_quant_vq2` | 1926 µs (**1.20×**) |

End-to-end this was worse than the kernel ratio, because decode attention
dominates a long-context step: vq2 reached only 569 tok/s at 30k/bs=max against
int2's 926 (**1.63×**). The capacity handicap had been fixed; this was the
kernel handicap, and it was what kept vq2 uninteresting despite compressing K
just as hard as int2.

The structural difference is one line. int2 unpacks K with shifts and masks —
pure ALU on data it already loaded. vq2 must **gather**: read 32 indices, then
issue 32 *dependent*, data-dependent loads into a 32 KB table.

---

## 2. Root-cause analysis

### 2.1 The first diagnosis was wrong

An earlier round of this study concluded the bottleneck was *"maximum address
divergence"*: `isel` is laid out `[BLOCK_N, NG]`, Triton maps a warp's 32 lanes
along the trailing axis, so each lane lands in a different group's 1 KB
sub-table — 32 guaranteed-distinct L1 sectors per warp instruction, the worst
case for the LSU. That conclusion was reached by *ablation* (delete a load, diff
the runtime), because Nsight Compute appeared unavailable.

It was reachable. The driver ships `RmProfilingAdminOnly: 1`, so counters need
`CAP_SYS_ADMIN`, which the normal container lacks — but a **privileged sibling
container** gets them without touching the host driver:

```
docker run -d --name oscar-prof --gpus all --privileged \
  -e NVIDIA_DRIVER_CAPABILITIES=all -v <repo>:<repo> -w <repo> oscar-cu13:latest sleep infinity
```

### 2.2 What the counters actually say

| metric | int2 | vq2 | group-major vq2 |
|---|---|---|---|
| time | 1968 µs | 2386 µs | 2473 µs |
| global ld **requests** | 5.3 M | **20.2 M** | 20.2 M |
| global ld **sectors** | 94.1 M | **392.4 M** | 570.5 M |
| **sectors / request** | 17.77 | **19.43** | 28.19 |
| **L1 throughput** | 30.4% | **73.8%** | 67.5% |
| L1 hit rate | 32.7% | 51.2% | **65.3%** |
| L2 sectors | 100.2 M | 250.6 M | 249.0 M |
| DRAM read | 1.46 GB | 2.20 GB | 2.59 GB |
| warps active | 9.9% | 11.9% | 12.1% |

Three things follow, and the first two overturn the old diagnosis:

1. **Sectors per request are nearly identical** (17.8 vs 19.4). Address
   divergence is not what separates the kernels. Lanes already share sectors —
   0.75 sectors per lane, not the 1.0 the divergence model predicted.
2. **vq2 is bound by L1 sector VOLUME**, at 74% of L1 peak against int2's 30%.
   The codebook gather is ~94% of that traffic.
3. The traffic **cascades**: 392 M L1 sectors → 251 M L2 sectors (~4.2 TB/s, near
   L2 peak) → **0.74 GB more DRAM than int2 despite identical bytes/token**. The
   gather's misses are manufacturing DRAM traffic out of nothing.

The decisive control is the group-major variant: it has the **best L1 hit rate of
any variant (65.3%) and is the slowest**. Hit rate is not the lever. Volume is.

### 2.3 Why volume cannot be fixed by addressing

A codeword is 4 bytes. A sector is 32 bytes. Every lookup drags in a full sector
to use 4 bytes of it — **12.5% sector efficiency** — and 32 lanes with random
indices cannot be made to share sectors, because the indices are data. Rearranging
*which* sub-table a lane reads changes the address spread but not the number of
sectors touched. That is why every Triton-level lever below measures as a no-op.

The only way out is to move the gather off the sector-based path entirely.
**Shared memory has no sectors**; a shared gather costs only bank conflicts
(`bank = idx % 32`, ~3.4-way for random indices, versus ~24 sectors). Triton
*can* express the lookup -- `tl.gather` fits it exactly and is bit-exact -- but
its only available lowering at this shape is a warp-shuffle compare-and-select
network, measured **13.4× slower** than the scattered global loads it replaces;
Gluon's `shared_memory_descriptor` indexes only by uniform scalar, and upstream
`ttg.local_gather` is not in 3.6.0 (§6b).

Triton's codebook load *does* lower to `cp.async.ca.shared.global`, which is
easy to misread as "Triton already does the CUDA kernel's staging, just at a
different granularity". It does not, and the distinction is the whole report.
`cp.async` there is the **pipeliner staging the gathered results** on their way
to `tl.dot`; the *source* of all 32 async copies per token is still a global
gather at a data-dependent address. Every sector counted above is still paid.
The CUDA kernel instead stages the **table itself** -- one coalesced 32 KB copy
per CTA -- after which every lookup is `LDS` and no sector is touched at all.
Staging results is not staging the table. See §6b, where reading the emitted PTX
did overturn two genuine §3.1 "no-op" findings -- but not this one.

---

## 3. What was tried

### 3.1 Triton-level levers — all measured, all dead

Each was implemented and validated against the shipped kernel before timing.

| approach | rationale | result |
|---|---|---|
| **group-major `isel`** `[NG, BLOCK_N]` | put a warp inside ONE 1 KB sub-table; expected ~1.57× fewer wavefronts | **1.13-1.15× SLOWER**. Best hit rate, most sectors. |
| **four K=NG dots**, no `kg` tile | `kg` is `[BLOCK_N, L]` fp16, ~64 regs/thread; dropping it should raise occupancy | **1.19× slower**; registers unchanged at 249, so `kg` was never the register driver |
| `eviction_policy` evict_first/last (9 combos) | keep the 32 KB table L1-resident against streaming K/V | 0.99-1.00× — noise |
| `cache_modifier` `.cg` / `.cs` (L1 bypass) | stronger, instruction-level version of the above | 1.01-1.03× — noise |
| launch-config sweep (96 configs) | maybe it is just mistuned | 64/8/2/3 already optimal |
| `tl.gather` | Triton's on-chip gather | **1.7-4.3× slower**; lowers to warp shuffles, 34 instr/element |
| smaller codebook (KC 256→16) | fewer sectors per sub-table | only 1.28×, for half the bit rate |
| split-K, `num_stages` | scheduling | flat |
| removing `tl.trans(kg)` | looked like a free transpose to delete | **0.86× (slower)** — Triton had folded it into `ldmatrix.trans`; it was never a transpose |
| full **TileLang** rewrite | TileLang *can* express a shared gather | correct, **0.40×** |

Two of these deserve a note.

**The group-major "blocker" did not exist.** The earlier round recorded
group-major as the one untried lever, blocked by `tl.join` only interleaving the
trailing axis, and therefore needing a retrained strided-coordinate codebook
bundle. That is avoidable: don't rebuild the `[BLOCK_N, L]` tile at all. With
plane *m* holding coord `4g+m`,

```
qk[h,n] = Σ_c q[h,c]·k[n,c] = Σ_m Σ_g q[h,4g+m]·p_m[g,n] = Σ_m dot(q_m, p_m)
```

so four `K=NG` dots consume the planes in their native layout, with `q_m` a
strided slice of `q` hoisted out of the loop. No bundle change. Implemented,
validated to 7.16e-07 — and simply slower.

**TileLang was profiled, not guessed at.** It does stage the codebook in shared
and does achieve the best L1 hit rate of anything tried (78.5%). It loses on the
lowering: **36.8 M global requests** against Triton's 20.2 M (it reloads `K_Idx`
per *coordinate* rather than per group), **227.8 M bank conflicts** on 102.3 M
shared loads, 1517 M instructions against 893 M, and `stall long_scoreboard` 3.12
against 0.33. `BLOCK_H` is padded to 16 for the MMA when only 4 rows are real, so
3/4 of the MMA work is wasted.

### 3.2 Dropping to CUDA

A focused probe isolated exactly the contested piece — gather + score, no V, no
softmax, no PV — running identical work in CUDA and Triton at the served shape.

The first CUDA attempt was **5× slower than Triton**, from two self-inflicted
bugs that counters found immediately:

- `reinterpret_cast<const uint8_t*>(&local_uint4)` to index bytes makes nvcc
  demote the variable to **local memory**: 150 M local loads, 255 registers,
  9.47 GB DRAM read. Extract bytes by shifting instead.
- Fully unrolling the 512-FMA body made the compiler hoist all 512 `q` values
  into registers and spill. `#pragma unroll 4` fixed it.

With those fixed the probe confirmed the thesis:

| | time | vs Triton |
|---|---|---|
| Triton gather+qk | 1134 µs | 1.00× |
| CUDA, shared codebook, fp32 | 753 µs | 1.51× |
| CUDA, shared codebook, half2 + `prmt` | **684 µs** | **1.66×** |

Counters confirmed the mechanism directly: **global load sectors 331 M → 37.4 M
(8.8×)**.

Two micro-optimisations mattered beyond the staging itself:

- **`prmt` (byte permute)** decodes two e5m2 bytes into one `half2` in a *single*
  instruction. `__byte_perm(cw, 0, 0x1404)` builds `[0x00, b0, 0x00, b1]`, which
  is exactly the pair of `<<8` shifts that e5m2→fp16 requires (they share sign
  and a 5-bit exponent with the same bias). The scalar path costs 8 instructions
  for the same 4 coords.
- **Thread coarsening** amortises the `q` broadcast loads across tokens.

### 3.3 The full CUDA kernel, and the ablations that steered it

The first *complete* CUDA stage-1 was **1.27× slower than Triton** despite the
1.66× gather. Phase ablations (`PV=6/7/8/9` in `cuda_vq2_stage1.py`) decomposed
it, and two FMA ablations settled the most attractive wrong turn:

| ablation | result | implication |
|---|---|---|
| `PV=4`: remove 3 of 4 **PV** FMAs | saves 46 µs | all PV FMAs cost **~61 µs** — already overlapped with memory |
| `PV=9`: remove 3 of 4 **score** FMAs | saves 853 µs | score work costs **~1137 µs** — fully *exposed* |

An earlier round had measured only the *PV* ablation and concluded "tensor cores
cannot help". That was the wrong phase. The score arithmetic is the exposed one —
and what recovered it was not tensor cores but **ILP**.

What actually worked, in order of contribution:

| change | effect | why |
|---|---|---|
| **codebook in shared memory** | global sectors 331 M → 37.4 M | the root cause |
| **PV register tiling** | 2452 → 1949 µs | tokens partitioned across warps so each token is touched by ONE warp; each lane owns the four quarters of **one packed V byte** (`{lane, lane+32, lane+64, lane+96}` are exactly that byte's four 2-bit fields), so one byte + one `float4` of `p` feeds 16 FMAs; and the (scale, zero) affine folded out of the inner loop via `Σ p·((q2−z)·s) = Σ (p·s)·q2 − Σ (p·s)·z`. 2 shared loads/token instead of 8. |
| **`q` half2 pair as ONE 8-byte load** | 1977 → 1843 µs | two adjacent `__half2` loads cost 256 shared loads/token against the fp32 path's 128 `float4` loads. This is why half2 *won* in the isolated probe (FMA-bound) but *lost* in the full kernel (shared-bound). |
| **FULL unroll of the 32-group score loop** | 1843 → 1630 µs | **the single biggest win after the shared codebook.** The score accumulators are a serial dependency chain; at `unroll 4` there is almost no ILP. |
| PV loop unroll 4× | 1630 → 1608 µs | same ILP problem; its dynamic bound stops the compiler unrolling it |
| drop write-only `vsc_s`/`vze_s` | → 1604 µs | dead once the affine is folded; frees 1 KB, crossing the 5-CTA/SM threshold |

### 3.4 CUDA approaches that measured worse — do not retry

| idea | result | why |
|---|---|---|
| bulk-stage the tile through shared | 2609 vs 2032 µs | extra shared write+read and two more barriers cost more than the latency hidden |
| software-pipeline the next block into registers | 1966 vs 1930 µs | doubled register state costs more occupancy than it buys. Also: indexing a register buffer with a **runtime** slot (`buf[cur][t]`) demotes it to local memory — 43% slower until the buffers were named and rotated by register copy |
| 256 threads/CTA | 1718-2310 vs 1604 µs | `BLOCK_N = THR × TPT` grows the tile, so shared grows with it |
| cross-warp reduction via shared `atomicAdd` | 2562 vs 1934 µs, reproduced twice | shrank the buffer 8 KB → 2 KB and *raised* occupancy, and the kernel got 30% worse |
| q-reuse kernel: decouple the reuse group from the staging block | 2565 vs 2011 µs | per-sub-pass barriers and `iw[TPT][8]` register pressure outweigh the saved broadcasts |
| **LUT kernel**: fold `q` into the table so one 8 B load returns all 4 heads' partial scores | 1887 vs 1604 µs | it delivers exactly what it promised — 707 M instructions vs 852 M, 49.7 M shared loads vs 80.6 M — but the 64 KB table drops occupancy to 3 CTAs/SM and its 8 B elements give 1.30 bank conflicts/load vs 0.48 |

---

## 4. The solution

`vendor/OSCAR-vq/.../triton_ops/vq2_cuda_stage1.py`. One CTA per
`(batch, kv_head, split)`, 128 threads.

**Stage the codebook in shared.** `cb_s[NG*KC]` = 32 KB, loaded cooperatively
once per CTA. Every subsequent lookup is a shared access: no tag lookup, no
sector, only `bank = idx % 32`. This is the whole point, and it is precisely what
Triton has no *efficient* form of: `tl.gather` expresses the same lookup
correctly but lowers to warp shuffles here, 13.4x slower (§6b).

**Score with maximum ILP.** The 32-group loop is fully unrolled so the four
per-head accumulator chains interleave; `q` is read as one 8-byte `uint2` per
`(head, group)`; codewords are decoded with `prmt` into `half2` and consumed by
`__hfma2`.

**PV by register tiling.** Warp-strided tokens; each lane owns the four quarters
of one packed V byte; the int2 affine folded out of the inner loop into a
per-head correction. Two shared loads per token.

**How this fixes the Triton kernel's problem.** The Triton kernel's cost is L1
sector volume, and sector volume is a property of gathering 4-byte codewords from
global memory — not of how the gather is addressed. That is why nine different
addressing, caching and scheduling changes each moved it by less than 3%. Moving
the table to shared memory removes the sectors entirely (**331 M → 37.4 M global
sectors, 8.8×**), which converts the kernel from L1-bound to instruction-bound;
the unrolling and layout work then attacks *that* bound. The result is a kernel
doing the same arithmetic on the same data with an order of magnitude less cache
traffic.

Standalone, across four shapes: **0.94-1.05× int2 and 0.75-0.83× the Triton
kernel**, at rel 3.2e-04 against the shipped kernel.

---

## 5. Integrating it — and the bug that nearly got published

Wiring it into sglang exposed three gaps between benchmark and engine, all
handled by `supports()` or the kernel: `att_out`/`att_lse` are **non-contiguous
slices** of a `[bs, heads, total_splits, L]` scratch (head stride is
`total_splits*L`); `q` is **bf16**, not fp16; and `kv_indices` is **int64**, not
int32.

Then the arm measured **1.6× at bs=1 and 8.3× at bs=64** end-to-end. The bs=64
number was impossible on its face — 5,195 tok/s is 12.3 ms/step, while 36 layers
× ~1.5 ms/layer is ~55 ms of stage-1 alone — so it was chased rather than
charted. Five hypotheses died by measurement (GPU placement, launch failure,
paging locality, KV occupancy, server args), and a liveness probe settled it:
making the kernel **1.9× slower per call** left throughput completely unchanged.

The torch profiler over sglang's `/start_profile` gave the answer in one line:

| | vq2_s8 | vq2_cuda (broken) |
|---|---|---|
| `_fwd_grouped_kernel_stage1_quant_vq2` | 282.39 ms / 2304 calls | **absent** |
| the CUDA `vq2_stage1` | — | **absent** |
| total CUDA kernel time | 813.58 ms | 530.91 ms |

The 282.67 ms difference is exactly the cost of the quant stage-1. **Neither
kernel was running.** The dispatch skipped the Triton launch, and the CUDA kernel
was never captured — decode was silently skipping long-context attention, which
is why it looked 8× faster and why slowing the kernel changed nothing.

Cause: the kernel launched as `<<<grid, THR, sm>>>` — the **legacy default
stream**. sglang captures decode into CUDA graphs on a side stream, and work on
the default stream is not captured. Every eager unit test passed throughout,
because the default stream synchronises implicitly. Fixed by launching on
`at::cuda::getCurrentCUDAStream()`; the profiler then shows the kernel with
**2304 calls** (64 steps × 36 layers) at 113.8 µs against Triton's 122.6 µs.

---

## 6. End-to-end result

bs=max, aggregate decode tok/s, prefill excluded, CUDA graphs enabled throughout,
18/18 cells passing every harness gate:

| model | length | bf16 | int2 | vq2_s8 (Triton) | **vq2_cuda** | vs int2 | vs vq2_s8 |
|---|---|---|---|---|---|---|---|
| Qwen3-8B | 30k | 467 | 926 | 569 | **938** | 1.01× | 1.65× |
| Qwen3-8B | 60k | 178 | 481 | 287 | **501** | 1.04× | 1.74× |
| Qwen3-8B | 90k | 91 | 282 | 184 | **318** | 1.13× | 1.73× |
| 4B-Thinking | 30k | 370 | 970 | 588 | **976** | 1.01× | 1.66× |
| 4B-Thinking | 60k | 217 | 506 | 306 | **513** | 1.01× | 1.68× |
| 4B-Thinking | 90k | 95 | 332 | 192 | **344** | 1.03× | 1.79× |

Against bf16 at 90k the speedup goes from 2.03× (Triton vq2) to **3.50×**. The
margin over int2 grows with context length, which is expected: decode attention's
share of the step grows with KV size.

---

## 6b. Postscript: part of the gap was reachable from Triton, the core was not

Asked whether the CUDA optimisations could be ported back, the first answer was
"no, the shared-memory gather is the whole win and Triton cannot express it".
Reading the compiled PTX corrected two *specific* claims underneath it -- that
cache hints did nothing, and that the launch config was already optimal -- and
those corrections were worth a real retune. The headline claim survived, but the
wording was imprecise and is restated here: Triton *can* express the gather, and
`tl.gather` will even stage the 32 KB table in shared -- it just gathers from it
with warp shuffles, 13.4x slower than the global loads it replaces. The sectors
are therefore unavoidable in practice, and the retune plateaus against int2
instead of passing it.

Two things in this section were previously overstated and are corrected below:
that seeing `.shared` in the PTX meant Triton was staging the table, and that
the per-lane shared gather was simply inexpressible. Neither survived a look at
the emitted SASS -- which is the same lesson as §8, applied to my own fix.

**Gluon exists and does not help.** Triton 3.6.0 ships
`triton.experimental.gluon` with `allocate_shared_memory` and a
`shared_memory_descriptor` -- so the blanket claim "Triton has no user-managed
shared memory" was wrong. The surface, checked on the installed 3.6.0, is
`dtype, index, layout, load, numel, permute, rank, reshape, shape, slice,
store`. `load(layout)` takes no index and reads the whole descriptor;
`index(i)` and `slice(start, length, dim)` return another *descriptor* -- a
view -- not values. A per-lane gather has to return a tensor of 32 different
values; nothing here accepts an index tensor, so it is not expressible. Same
wall, one layer down.

**`tl.gather` can express the lookup, and is 13x slower.** This is the one that
had to be tested rather than argued, because the earlier "`tl.gather` is 4x
slower" note never recorded *which* lowering it measured. Triton's
`GatherLoweringHelper` picks between two -- warp shuffles when
`isWarpLocal()`, a shared-memory scratch path otherwise -- and both symbols are
present in our 3.6.0 binary. A slow result is the signature of the shuffle path
and says nothing about the shared one. Upstream `main` additionally has a
`ttg.local_gather` op (gather from a shared descriptor by index *tensor*); it is
absent from 3.6.0 and not reachable from the Python frontend.

The lookup does fit `tl.gather`'s signature, which the first pass missed:

    cb_tile = [KC, NG]                 codebook, loop-invariant
    isel    = [BLOCK_N, NG]            per-token group indices
    tl.gather(cb_tile, isel, axis=0) -> out[n, g] = cb_tile[isel[n, g], g]

ranks match and the non-axis dim matches. `probe_tl_gather.py` builds both at
the real decode shape, gather-bound, and checks the SASS:

| | LDS | SHFL | ISETP | SEL | time |
|---|---|---|---|---|---|
| global gather (shipped) | 4 | 1 | -- | -- | **20.2 µs** |
| `tl.gather` from tile | 30 | **516** | 453 | 257 | **271.5 µs** |

Bit-exact against the global gather, and **13.4x slower**. It took the shuffle
path -- a compare-and-select network across the warp -- and the 32 KB of shared
it allocates is the tile's storage, not gather scratch. Triton is free to choose
a layout that makes the gather warp-local, and it does.

So the wall is not "inexpressible" but "expressible and much worse", which is a
stronger result: the operation exists, it is correct, and its only available
lowering at this shape is the wrong one. The remaining untested door is forcing
a non-warp-local layout via Gluon's explicit layout control so the scratch path
is selected instead. That is speculative, a large lift, and would still be
Triton's generic scratch lowering rather than the CUDA kernel's single
coalesced stage -- recorded as an open question, not a plan.

**Triton stages the gathered RESULTS through shared -- not the table.** The
codebook load lowers to **32 x `cp.async.ca.shared.global` per token**, which an
earlier revision of this report read as "Triton does the same global->shared
staging the CUDA kernel does, just once per TOKEN instead of once per CTA."
That reading was wrong and is corrected here.

`cp.async` on that path is the pipeliner moving *already-gathered codewords*
into shared on their way to `tl.dot`. The source operand is still
`cb_head_base + offs_g * KC + isel` (`decode_attention.py:2710`) -- a global
pointer at a data-dependent address -- so every one of those 32 copies per token
pays its own sector. The CUDA kernel stages the **table**: one coalesced 32 KB
copy per CTA (`vq2_cuda_stage1.py:130`), after which each lookup is `LDS` and no
sector is touched. Staging results is not staging the table; that is why the
global-load sector counts still differ 392 M vs 37.4 M despite both kernels
showing `.shared` traffic in their PTX.

The consequence is that the sector volume **is** a fixed property of the Triton
kernel, not a tunable of pipeline depth. `num_stages` changes how well the
latency is hidden, never how many sectors are fetched -- which is exactly the
shape of the retune result below (a real but bounded win that plateaus against
int2 rather than passing it).

It also explains a null result that had been recorded as a finding: the earlier
`eviction_policy` / `cache_modifier` sweep produced PTX **byte-identical** to no
hints. Triton pins `.ca` on that path (a 4-byte `cp.async` cannot take `.cg`)
and silently drops the user hint, so that sweep measured nothing. Note Triton
*does* use `.cg` for the 16-byte `isel` load -- its codegen is better than the
earlier round assumed.

**And it made `num_stages` the first-class knob.** `num_stages` is the `cp.async`
pipeline depth. The earlier config sweep capped it at 3 and therefore sat on the
wrong side of an interaction: at stages=3 `BLOCK_H=8` wins, but at stages>=4
`BLOCK_H=4` wins -- and `BLOCK_H=4` exactly matches `kv_group_num`, so the qk
tile carries no padded rows. Retuning the launcher ladder (no kernel code
changed):

| bs | int2 | vq2 before | vq2 retuned |
|---|---|---|---|
| 1 | 98 us | 1.09x | **0.97x** |
| 4 | 147 us | 1.02x | 1.03x |
| 16 | 380 us | 1.34x | **1.10x** |
| 32 | 972 us | -- | **0.98x** |
| 64 | 1649 us | 1.18x | **1.09x** |

So the shipped Triton kernel goes from 1.02-1.34x int2 to **0.97-1.10x**, against
the CUDA kernel's 0.94-1.05x. Most of what the CUDA rewrite bought was reachable
by configuration.

That reframes the CUDA kernel's value: it is now worth roughly 5% over retuned
Triton at bs=16-64, and carries a `cpp_extension` JIT build, a stream-capture
hazard, and hard-coded geometry. **Retuned Triton is the better default**; the
CUDA path is for cases where that last 5% is worth the operational cost.

The lesson is the same one this study keeps relearning: read what the compiler
actually emitted before theorising about the hardware. Two conclusions here --
"cache hints do nothing" and "the launch config is already optimal" -- were both
artefacts of not looking.

## 7. Caveats and open items

- **The reference arms were measured on an older engine tree.** The `vq2_cuda`
  cells are current; bf16 / int2 / vq2_s8 / vq2_s48 date from the original sweep,
  before `SGLANG_MIXED_KV_EXACT_CHUNKED_PREFILL` landed. That flag defaults to
  OFF and allocates nothing when disabled, so it should not matter — but a strict
  apples-to-apples chart would re-run all four.
- **Runs must be serialised.** Two harness servers concurrently, or one alongside
  another user's job, produced OOMs and hard server kills with no traceback that
  looked like arm-specific failures and were not: the identical config passes 9/9
  alone on a free GPU.
- **Accuracy is untested for this kernel.** The throughput benchmark uses
  synthetic codebooks and is accuracy-invalid by construction. The kernel matches
  the Triton reference to 3.2e-04 on stage-1 output, but no NIAH/LongBench run has
  been done with it.
- The kernel is specialised to `NG=32, KC=256, head_dim=128, KVG=4`, e5m2, INT2 V,
  no logit-cap, no xai-temperature. `supports()` gates every one; anything else
  falls back to Triton.
- Untested lever: codebook **group size** `G`. The gather cost is proportional to
  `NG`, and a 4-byte codeword uses 4 of every 32-byte sector, so `G=8` should halve
  both. `probe_group_size.py` exists but is **explicitly labelled INCONCLUSIVE** —
  its baseline moved 2× under an unrelated dtype change. It is a format question
  (a retrained codebook), not a kernel one.

---

## 8. Methodological notes

The expensive failures in this study were all measurement failures, not
engineering ones.

- **Check whether a profiler is reachable before inferring hardware behaviour
  from timings.** The round-1 diagnosis ("address divergence") sent the study
  after addressing layouts for a long time; counters falsified it in one run by
  showing sectors/request were nearly identical.
- **An end-to-end speedup larger than the component can explain is a bug report.**
  Layers × per-layer time versus step time falsified the 8.3× before any
  profiling. The arithmetic is cheap; do it first.
- **With synthetic codebooks every accuracy signal is blind.** Output is garbage
  by design, so a kernel that silently does nothing passes every gate the harness
  has. `no_nan` and `full_length` cannot detect omitted attention.
- **Eager unit tests cannot catch stream-capture bugs.** Validate a decode kernel
  by kernel *count* inside a running server, not only by numerics offline.
- Round 1 recorded five microbenchmarks whose ranking inverted end-to-end because
  their regime differed from the served one. Round 2 adds two more traps, both
  costing 3-5×: `reinterpret_cast` on a local's address, and runtime-indexed
  register arrays — each silently demotes to local memory.
- `--disable-cuda-graph` roughly halves throughput, so any A/B run without graphs
  understates a decode kernel's effect. One such run briefly convinced me the
  kernel was dead; it wasn't.

---

## 9. Postscript: reviewing a co-author's independent optimization pass

A co-author (Samuel) optimized vq2 on a clone several commits behind this tree
and sent `vq2_kernel_changes_for_amir.md`. Four of his findings were tested
here. Two were already ours, one is an H100 regression, one is real and small.

**Microbenchmark method note.** The first pass reported ~2-3x for everything and
a suspicious ~16.7 us floor identical across n=8 and n=512 and across every
block width. That floor was CPU dispatch, not GPU time: torch's einsum path is
three ops and the fused path is one, so a Python-loop benchmark reports the
op-count ratio and calls it a speedup. These maps run *inside* sglang's captured
graph, where launch cost is already amortised, so every number below is measured
under `CUDAGraph.replay()`. Same conclusions, honest magnitudes -- and one
conclusion (the column split) flipped from "ties" to "no reason to exist".

| his finding | verdict here |
|---|---|
| fused per-head **q** map | already in this tree as `vq_map_q_fused` / `SGLANG_VQ_OPT_QMAP`; 2.7-3.0x over torch bmm at T=4..64 |
| **column-split** grid `(rows, H, D/BLOCK_E)` | no gain: 3.0-3.2 us split vs 3.1 us unsplit. His 3.56x/4.06x is against the *unfused* torch path, which is the same win we already have by another route |
| fused **K** map | real and absent here: 2.0-2.8x (8.1 -> 3.2 us at T=8, 53.9 -> 26.3 at T=8192). Landed as `SGLANG_VQ_OPT_KMAP` |
| `quant_split_cap = cap // bs` | **regression on H100.** bs=4: 188 vs 147 us (+28%); bs=16: 591 vs 405 (+46%). The shipped heuristic is within 1-9% of the measured optimum at every batch |

### 9.1 The fused K map

`vq_map_k` was still `torch.einsum(...).contiguous()` -- three kernels over a
small tensor. It reuses `_vq_qmap_kernel` with `HAS_MEAN` and `GRP=1`, since the
K map differs only by the centering term and the absent GQA group.

It is bit-identical to the einsum at bf16 and fp16 across T=1..2048, contiguous
and strided. It was *not* initially: `tl.dot` defaults to TF32 on fp32 inputs,
which made the flag a silent ~8e-4 numerics change on any fp32 path. Pinning
`input_precision="ieee"` for fp32 inputs makes all three dtypes exact. The
served path is bf16, so this never bit us -- but a flag that changes numerics
only at one dtype is exactly the kind of thing that surfaces two studies later.

### 9.2 What it is worth end-to-end, and why that was predictable

The map runs once per layer per decode step on the HP write
(`unified_kv_pool.py:1319` -> `_prepare_hp_kv_tensors`), with T = batch. So the
ceiling is 36 layers x ~5.5 us ~= **200 us/step** -- ~2% of a bs=1 step and
~0.2% of a bs=64 one. That arithmetic was done before the A/B, and the A/B
landed on it.

`artifacts/throughput/vq2_kmap_ab/`, Qwen3-8B, 30k, 1024 out. Arms are doubled
(`kmap_a`/`kmap_b`, `base_a`/`base_b`) so the identical-pair spread measures the
noise floor; a gap smaller than that spread is not a result.

| bs | kmap (a / b) | base (a / b) | delta | pair spread |
|---|---|---|---|---|
| 1 | 83.14 / 82.70 | 82.27 / 82.06 | **+0.92%** | 0.26-0.53% |
| 4 | 275.76 / 273.11 | 274.47 / 272.97 | +0.26% | 0.55-0.97% |
| 64 | 663.32 / 659.15 | 663.53 / 661.40 | -0.19% | 0.32-0.63% |

All 12 cells passed every gate. Only bs=1 separates cleanly (both kmap runs beat
both base runs); bs=4 and bs=64 overlap and are noise. Keep the flag -- it is
free and bit-identical -- but it is not a headline number, and the doubled-arm
design is what licenses saying so rather than reporting +0.26% as a win.

### 9.3 His question about the CUDA kernel's lane mapping

He asks whether a warp's 32 lanes read 32 tokens at one group, or one token
across 32 groups -- a 4x L1-line-traffic difference he could not control from
Triton, and his best guess at where our gains come from.

Ours is the former: `const int n = base + tid` makes one thread own one token,
and the group loop `for (int g = 0; g < NG; ++g)` is sequential, so at any
instant 32 lanes read 32 distinct tokens at the same group.

But that is not the mechanism. `cb_s` is *shared* memory -- the head's entire
32 KB table staged once per block behind the >48 KB dynamic-shared opt-in -- so
the gather is `LDS`, bank-limited, and the L1-line argument does not apply to it
at all. This also resolves why his own shared-memory LUT attempt measured 0.53x:
full-table residency and the lane mapping only pay off together.

### 9.4 Two of his protocol gotchas, checked against this tree

- **`MEM_FRAC` is ignored** -- real here too: `serve_oscar.sh:18` is
  `MEM_FRAC=0.78`, not `${MEM_FRAC:-0.78}`. The throughput harness escapes it by
  passing `--mem-frac` (`run_throughput.py:161`), which is why its logs read 0.85
  and the pool matches the 0.85 prediction. His absolute numbers are at 0.78 and
  are not comparable to these.
- **"the quantized path gets ZERO radix-cache hits"** -- fixed in this tree.
  The vq2 arms above show `cached=28,408` of 30,000 at bs=1. His conclusion that
  prefix warm-up is unreproducible for int2/vq2, and that all arms must therefore
  be benchmarked with sharing off, does not apply here.

### 9.5 Three dead hypotheses for the server-vs-microbenchmark gap

Unrelated to his document, but closed while the A/B ran. vq2 stage-1 costs
2311 us/call in the server against 1803 in the microbenchmark (+28%) while int2
matches (1571 vs 1649). Newly eliminated:

- **Arena layout.** The per-layer VQ index view `_vq_k_idx_big[l]` is contiguous
  with strides identical to the benchmark's standalone tensor. Ruled out by
  reading the allocation, not by measuring.
- **Cold L2.** A 256 MB flush between launches (timed separately and subtracted)
  moves neither arm: 1.00x vq2, 0.99x int2.
- **SM clock throttling.** Plausible on the shape of the evidence -- vq2 is
  ALU-heavy and int2 bandwidth-bound, so only vq2 would feel a clock drop. But
  the serving GPUs sample at 1980 MHz (= max), 318-333 W against a 700 W limit,
  38-44 C. No throttling.

The gap stands unexplained, with a shorter candidate list.

---

## 10. TileLang, rebuilt: the CUDA structure in a portable language

The CUDA kernel's weakness is that `supports()` gates one geometry,
`(NG, KC, L, KVG) == (32, 256, 128, 4)`. TileLang JIT-specialises per geometry,
so it would give that flexibility back -- if it could be made fast. Round 2
measured a full TileLang rewrite at **0.40x** and set it aside.

That verdict was wrong, and the reason is in this report's own §3.1 table: ncu
attributed the 0.40x to four *lowering* problems, none of them the gather --
`K_Idx` refetched per coordinate (36.8 M global requests vs 20.2 M), 227.8 M
bank conflicts, 1.7x instruction bloat, and `BLOCK_H` padded to 16 to feed an
MMA when only 4 rows are real. Meanwhile the isolated gather probe had TileLang
at **2149 G lookups/s against CUDA's 2116** -- the primitive was already there.
The kernel was simply not shaped like the CUDA one.

### 10.1 The rebuild

`probe_tilelang_qk.py` re-runs the *same* contested piece as `probe_cuda_qk.py`
-- gather + score, no V, no softmax, no PV -- with the CUDA kernel's structure:

* one thread per token, tokens strided by THREADS, so the 32 index bytes arrive
  as **8 vectorized int32 words once per token** (kills failure 1);
* every lane sits at the same group at the same instant, so a warp reads inside
  ONE 1 KB sub-table and conflicts only on `idx % 32` (kills failure 2);
* **scalar FMA into per-thread registers, no MMA** -- with KVG = 4 any
  tensor-core tile discards 3/4 of its rows (kills failure 4);
* codebook and `q` staged in shared (failure 3).

Two TileLang-specific notes. Layout inference rejects a fragment addressed as
`(t, h)` by two different unrolled `h` loops, so the accumulators are
`alloc_local` registers -- which is what the CUDA kernel uses anyway. And
`T.vectorlow`/`T.vectorhigh` exist in the Python API but have **no CUDA
codegen**, so `half2` lanes are extracted through an `int32` reinterpret.

The `half2` step is where it pulls ahead. Rather than pre-decoding the codebook
to fp16 -- which needs 64 KB, cutting CTAs/SM from 7 to 3, and gives 2-way bank
conflicts where packed int32 gives 1-way -- the four e5m2 bytes are reassembled
into two `half2` words with shifts, emulating the CUDA kernel's `prmt`. The
table stays 32 KB and the 512 scalar FMAs per token become 256 `HFMA2`.

### 10.2 Result

All three measured in one session on the same H100, same shape
(BS=64, CTX=30000), validated against an independent fp32 torch reference.

| kernel | splits=3 (matched) | splits=8 | accuracy |
|---|---|---|---|
| Triton (shipped) | -- | 1140.7 µs | 3.6e-07 |
| CUDA fp32, best tpt | -- | 751.1 µs | 3.6e-07 |
| **CUDA half2 tpt=2 (best CUDA)** | -- | **687.0 µs** | 2.3e-03 |
| TileLang fp32 TPT=2 | 923.8 µs | 882.8 µs | 2.1e-07 |
| **TileLang half2 TPT=2** | **572.9 µs** | **551.3 µs** | **1.7e-03** |

**Read this table by accumulator precision, not by language.** Grouped that way
it says something different from the headline it first suggested:

* at **fp32** -- the precision OSCAR and shipped vq2 both use -- CUDA is
  **1.18x faster** than TileLang (751 vs 883 µs). TileLang still beats shipped
  Triton by 1.29x.
* at **fp16 (half2)** TileLang is **1.20x faster** than CUDA at matched split
  geometry (1.24x at each one's own best), with slightly better error.

So the ordering *flips with the numerics choice*, and the half2 comparison is
only the relevant one if fp16 accumulation is acceptable. It is not, by default:
`_fwd_grouped_kernel_stage1_quant_int2` declares every accumulator
`dtype=tl.float32` (`e_max`, `e_sum`, all four PV accumulators) and its `qk`
comes from `tl.dot`, which accumulates fp32 for fp16 inputs. The shipped vq2
Triton kernel matches. **half2 is therefore a new accuracy regression against
both shipped arms, not a trade the project has already accepted** -- which is
why the CUDA kernel keeps `SGLANG_VQ2_CUDA_FP32=1` as an escape hatch.

Under an fp32 constraint the honest statement is: TileLang is 1.29x shipped
Triton and 0.85x the CUDA kernel.

The split geometry was checked precisely because it could have faked the result:
`probe_cuda_qk.py`'s engine heuristic picks 3 splits, so only 3 of its 8
z-blocks do work (1536 active blocks) while the TileLang probe at SPLITS=8 runs
all 4096 at 1/8 the size -- better load balance across 132 SMs for the same
total token work. Forcing TileLang to splits=3 costs only 3% (551 -> 573 µs), so
the 1.20x is the kernel, not the schedule.

### 10.3 Status and what this changes

This is the **contested piece only**. A complete TileLang stage-1 still needs
online softmax, the PV register tiling (the CUDA kernel's second-largest win,
2452 -> 1949 µs), and the stage-2 interface.

But it inverts the recommendation recorded in §7. The earlier projection --
that TileLang would land near 0.8x CUDA, so the portability would cost ~20%
throughput -- was extrapolated from the fp32 variant before the `half2` step
existed. On the measured evidence the portable kernel is *faster* than the
hand-written CUDA one, and the trade the project thought it faced (flexibility
vs speed) does not exist on this piece.

Two caveats before this is treated as settled. The `half2` variants of both
kernels accumulate in fp16 -- 1.7e-03 relative here against 2.3e-03 for CUDA,
small and softmax-damped, but they are not the fp32 kernels and should not be
compared against fp32 numbers elsewhere in this report. And a probe is not a
kernel: nothing here proves the ordering survives the full stage-1, where
register pressure from the PV accumulators is exactly what forced several CUDA
design choices.

### 10.4 The fp32 constraint, and the one lever left

Requiring fp32 accumulation (OSCAR-INT2's precision, and shipped vq2 Triton's)
collapses the ranking. Full stage-1, bs=64 ctx=30000, one session:

| kernel | time | vs int2 |
|---|---|---|
| **int2 (OSCAR)** | **1636-1650 µs** | -- |
| vq2 CUDA, half2 | 1618 µs | **0.98x** |
| vq2 Triton, retuned | 1803 µs | 1.10x |
| vq2 CUDA, fp32 | 1827 µs | 1.11x |

The CUDA kernel's entire advantage over OSCAR was fp16 accumulation. At fp32 it
is indistinguishable from the retuned Triton kernel, and **nothing built so far
beats OSCAR-INT2 at equal numerics**.

What has not been tried is the obvious way to get fp32 accumulation at
fp16-multiply speed: **tensor cores, with tokens on the M axis.**

Every MMA attempt in this study put *heads* on M. `tl.dot(q_full, k_full)` in
both the shipped Triton kernel and the round-2 TileLang rewrite has
M = BLOCK_H = 4, padded to 16 -- **4x of the tensor-core work discarded**, which
is why §3.1 recorded MMA as a dead end and why the CUDA kernel deliberately uses
scalar FFMA instead.

Computing the transpose instead, `qk^T = k_hat^T · q^T`, puts BLOCK_N tokens on
M (hundreds available, no padding) and the 4 heads on N (padded to 8 for
`m16n8k16`) -- **2x waste instead of 4x**. Tensor cores consume fp16 operands and
accumulate in fp32 natively, so this is exactly "half2-speed multiply, fp32
accumulate": the thing the half2 trick bought, without the accuracy cost.

**Tested, and it loses.** `qk_max_tl_mma` in `probe_tilelang_qk.py` implements
exactly that: `T.gemm(k_s, qt_s, transpose_B=True)` with BN tokens on M and the
4 heads padded to N=8. Two TileLang constraints surfaced on the way -- the MMA
requires `N % 8 == 0` (hence the padding), and WGMMA requires the B operand
N-major, so `q` is stored `[8, L]` and transposed in the GEMM rather than in
memory.

| variant (fp32 accumulate) | time |
|---|---|
| **scalar FMA, TPT=2** | **883-889 µs** |
| MMA, BN=64 | 1345.7 µs |
| MMA, BN=128 | 1119-1136 µs |
| MMA, BN=256 | 1361.1 µs |

The tensor cores are ~1.28x SLOWER than scalar FMA. The cause is already in
§5's ablation table: the MMA needs a materialised `[BN, L]` fp16 K tile in
shared, and "bulk-stage the tile through shared" was measured there at 2609 vs
2032 µs for the CUDA kernel. Feeding tensor cores costs a full shared
write+read round-trip that the register-resident scalar path never pays, and
that round-trip exceeds what the MMA saves.

Pipelining does not rescue it: `T.Pipelined(ntile, num_stages=n)` to overlap the
next tile's gather with the current tile's GEMM measures 1136 / 1134.6 / 1134.5
/ 1135.4 µs at n = 0 / 2 / 3 / 4 -- completely flat. The shared round-trip is a
throughput cost, not a latency to hide.

So the lever is spent. Under the fp32 constraint the standing is:

| fp32 kernel, contested piece | time | vs CUDA |
|---|---|---|
| CUDA fp32 (tpt=8) | 751 µs | 1.00x |
| TileLang fp32 (TPT=2) | 883 µs | 0.85x |
| Triton (shipped) | 1141 µs | 0.66x |

TileLang is 1.29x shipped Triton but does not match CUDA, and no fp32 route
beats OSCAR-INT2 at the full-kernel level. Remaining untried: holding the K
tile in MMA-layout *fragments* instead of shared, which would avoid the
round-trip -- but the gather writes are data-dependent per token, and mapping
them onto a WGMMA fragment layout is precisely the hard part.

### 10.5 The full TileLang stage-1: built, exact, and not competitive

Pushed to settle the projection argument, the complete stage-1 was built in
TileLang (`tilelang_stage1_full.py`): scalar QK gather core + online softmax +
int2-V PV + split-K partials/Lse, in SPMD style (`get_thread_binding`, explicit
`sync_threads`, two thread-role phases per tile). Validated bit-for-bit-class
against the shipped Triton kernel's own outputs (golden dumps, rel 1e-05 small
shape / 4e-05 served shape), so every number below is from a PROVEN-equivalent
kernel.

| full stage-1, fp32, bs=64/ctx=30k | time |
|---|---|
| OSCAR int2 (Triton) | 1636-1650 µs |
| vq2 Triton retuned | 1803 µs |
| vq2 CUDA fp32 | 1827 µs |
| **vq2 TileLang v1 (scalar)** | 3168 µs |
| vq2 TileLang v2 (T.gemm PV) | 3735 µs |
| **vq2 TileLang best (v3 + minBlocks=5)** | **2995 µs** |

The contested-piece advantage (883 vs Triton's 1141) did not compose, and the
ablation ledger says precisely why. Phase knockouts at the served shape:
full 3168; -PV 2178; -QK 2089; -softmax-update 3065; -V-staging / -score-writes
/ -l-scan each ~3165 (no change); empty shell 964; shell minus index loads 665;
minus barriers too 643. Three structural facts fall out:

1. **The costs are non-additive** -- QK "costs" 1080 and PV 990 against a shell
   of only ~650, and micro-optimising the instruction count (vectorised q and p
   reads, folded dequant, combined shifts) moved the total just 2%
   (3168 -> 3103). The kernel is LATENCY-bound: the barrier-separated scalar
   phases idle the memory pipeline that Triton keeps saturated with
   cp.async + MMA overlap.
2. **TileLang's own machinery for that overlap is unusable here.** `T.gemm` in
   the dynamic-length tile loop made the kernel *slower* (v2, 3735), and
   `T.Pipelined` is a measured no-op on these loops (1134-1136 µs flat across
   num_stages 0/2/3/4 in the MMA probe). The two facilities that make Triton's
   remainder cost 662 µs are exactly the ones that do not engage.
3. **Default codegen caps occupancy.** TileLang emits
   `__launch_bounds__(threads, 1)`; `T.annotate_min_blocks_per_sm(5)` recovered
   3.5%. Real, and nowhere near the 1.7x needed.

**Verdict: at fp32, TileLang cannot beat the retuned Triton kernel on this
shape with the current TileLang version.** Best full kernel: 2995 µs = 1.66x
Triton, 1.8x OSCAR. The piece-level win was real but is worth 258 µs, while
the machinery gap costs ~1500 µs.

What WOULD beat OSCAR at fp32, in order of plausibility:
* **CUDA + `mma.sync`**: the union of the shared-codebook gather (CUDA has it,
  Triton cannot express it) and tensor-core dots with fp32 accumulation
  (Triton has it, the scalar CUDA kernel skipped it because half2 covered the
  gap). fp16-operand MMA with fp32 accumulate has NO accuracy cost -- exact
  products, fp32 sums -- it is what OSCAR itself does. Hand-written, one
  geometry, but the only identified route below 1636 µs.
* **Triton, if/when `ttg.local_gather` ships and is frontend-reachable**: the
  same union from the other side, keeping Triton's pipeline.
* **TileLang, if dynamic-loop pipelining and gemm overhead are fixed
  upstream**: the portable route, currently blocked by (2).

### 10.6 The profile-driven rounds: every design point measured

Pushed further on the demand that a 2100 µs remainder "cannot make sense", the
full kernel was profiled with ncu (privileged sibling container) and each fix
the profile motivated was built and measured. Parity vs the Triton golden
outputs held at every step.

**What ncu says about the scalar kernel** (best variant, 2995 µs): 84
regs/thread, **zero** local-memory traffic (no spills), occupancy limited to 5
blocks by shared+regs, warps active 27.9% -- and **issue_active 74.2%** with
**2.137 G instructions executed** against Triton's 893 M for the same math. So
it is not latency-bound after all: it is **instruction-throughput bound**. The
root is arithmetic density: a scalar warp FMA retires 32 MACs; an HMMA retires
2048. Stage-1 is ~15.8 G MACs; as scalar warp instructions that is a ~494 M
instruction floor before a single byte moves. No scalar kernel closes a 2.4x
instruction deficit at 74% issue.

**The fix the profile dictates** -- both dots on tensor cores (fp16 operands,
fp32 accumulate: exact products, no accuracy change; it is what Triton itself
does) -- was built as v4: pure tile style (block ops only, no SPMD), Triton's
exact structure with BLOCK_H=16 padding, gather feeding an f16 K tile from the
shared codebook, fragment softmax, two T.gemms. It compiles, passes parity at
rel 1.1e-05... and measures **6046 µs**. The machinery around the gemms
(fragment layout passes with replicate-4 elementwise ops, the p16
fragment->shared roundtrip forced by incompatible accumulator/operand layouts,
per-tile gemm setup) multiplies the per-tile cost.

**The last hypothesis** -- that `T.Pipelined` fails only because the tile loop
has a dynamic trip count -- was tested by compiling a static count (bench
shapes are uniform; masked padding) and sweeping stages:

| variant | full stage-1 |
|---|---|
| scalar SPMD, best (MINB=5) | **2995 µs** |
| scalar + T.gemm PV (v2) | 3735 µs |
| pure-tile MMA, dynamic loop (v4) | 6046 µs |
| pure-tile MMA, static NTS=79 + Pipelined stages 0/2/3 | 10030 / 10051 / 10040 µs |
| Triton retuned | 1803 µs |
| OSCAR int2 | 1636-1650 µs |

Flat across stages even with a static bound: the software pipeliner cannot
stage a producer that contains a data-dependent shared-memory gather under
control flow. That is not a tuning gap; it is the framework's boundary on this
kernel shape, established by direct experiment.

**Final standing.** TileLang's expressible-and-fast region does not contain
this kernel: the scalar style is instruction-bound 1.66x above Triton, and the
MMA style pays unpipelined machinery costs of 2-5x. The one design that beats
OSCAR at fp32 remains unbuilt in any framework: shared-codebook gather +
`mma.sync` fp16/fp32-acc + hand cp.async pipeline -- i.e. CUDA.

### 10.7 The CUDA mma.sync kernel: built, iterated, and the ~10% that remains

The union kernel proposed in 10.4-10.6 -- shared-codebook gather + m16n8k16
`mma.sync` (f16 operands, f32 accumulators: exact products, no accuracy change)
-- was built as `cuda_mma_stage1.py` and optimised through profile-driven
rounds. The novel part is the fused gather: each lane decodes, from the shared
codebook, exactly the 8 halves its A-fragment position owns (groupID/threadID
arithmetic maps a fragment slot to one `prmt` of one codeword), so scores never
touch a materialised k_hat tile. QK computes S^T with tokens on M (2x head
padding, not the 4x that killed earlier MMA attempts); PV computes acc^T =
V^T.p with int2 V dequantised straight into A-fragments. Parity vs the Triton
goldens: rel 1.1e-05 / 9.7e-06 at every step below.

| iteration | full stage-1 (splits=3) | driven by |
|---|---|---|
| first correct version | 2190.6 µs | -- |
| + 48 B padded shared strides | **1876.6 µs** | ncu: 54% bank-conflict rate on 32 B rows |
| + cp.async double-buffered staging | 1878.6 µs | long_scoreboard 17.5->11%, but shared 2x -> 3 CTAs/SM: net zero |
| + transposed V arena | 2214.7 µs | REVERTED: 32 scattered STS + sync V load cost more than halved reads saved |
| + in-place V byte repack (prmt) + hoisted PV loads | 1900.5 µs | mio_throttle 14->9; PV shared loads 128->32/warp-tile; net slightly negative |
| + THR=256 / CHUNKS=1 (24 warps/SM) | 1925.7 µs | REVERTED: 256-thread barriers + duplicated per-warp staging |
| + dual even/odd MMA accumulator chains | **1847.2 µs** | `wait` 16%: 8 MMAs serialised on one C-register set |
| best, splits=8 | **1815.7 µs** | grid-level parallelism (stage-2 cost rises with splits) |

Final standing at fp32, same box, bs=64/ctx=30k:

| kernel | splits=3 | splits=8 |
|---|---|---|
| OSCAR int2 | 1636-1650 µs | -- |
| vq2 Triton retuned | 1803 µs | 1783 µs |
| vq2 CUDA scalar | 1827 µs | -- |
| **vq2 CUDA mma.sync** | **1847 µs** | **1816 µs** |
| vq2 TileLang best | 2995 µs | -- |

**The decisive profile.** The final kernel's global-load sectors are
**99.4 M** -- versus 392 M for the shipped Triton vq2 and 94 M for int2 -- and
its instruction count is 740 M versus Triton's 893 M. Both original bottlenecks
of this study (sector volume, then instruction density) are SOLVED, at int2
parity or better. What remains is `short_scoreboard` 26.8% at 18% occupancy:
the codebook gather now lives as an LDS dependency chain in the MMA operand
path. int2 has no such chain -- its dequant is shift/mask on data already in
registers from the streaming load. At the occupancy this kernel can reach
(shared-limited by the 32 KB table + tile arenas), ~30-cycle shared latency in
the operand critical path is worth almost exactly the ~10% observed.

**Conclusion of the fp32 campaign.** Four independent implementations --
Triton (MMA + cp.async, global gather), CUDA scalar (shared gather, FFMA),
CUDA mma.sync (shared gather, tensor cores, cp.async), TileLang -- now
converge at 1.10-1.13x OSCAR-INT2, each bounded by a different resource
(sectors / instruction issue / operand-path LDS latency). When every corner of
the design space lands on the same number, that number is the format's price:
**at fp32 accuracy, 32 data-dependent codebook lookups per (token, head) cost
~10% over int2's shift-mask dequant, and no kernel engineering removes it.**
The routes that DO remove it change the problem, not the kernel: fp16
accumulation (CUDA half2: 0.98x int2, rel 3.2e-04), or the format itself --
larger G / smaller KC shrinks the table and the chain (task #5, open).

### 10.8 Adaptive split selection and gpt-oss geometry support

**Adaptive splits** (`SGLANG_VQ2_ADAPTIVE_SPLITS=1`, requires `KV_SPLITS` >=
the top rung). The fp32 chart pinned KV_SPLITS=8 on every arm for cross-arm
comparability, and that pin is exactly what the CUDA kernel's bs=1 deficit was:
at bs=1 the grid is 1 x 8 heads x 8 splits = 64 CTAs on 132 SMs. The measured
per-batch stage-1 optimum (H100, ctx=30k) is 48/32/16/8/8/4 splits for
bs 1/2/4/8-16/32 -- 3.0x at bs=1 (105 -> 35 us) -- while the engine's cap=48
heuristic over-splits mid-batch by 9-20%. The ladder is wired as a CAP on the
heuristic in `triton_backend.get_num_kv_splits`, so per-sequence adaptation to
context length is preserved below it, and under CUDA-graph capture the batch
(and hence the cap) is static per captured graph.

End-to-end (Qwen3-8B, 30k, vq2_cuda_fp32, `vq2_adaptive_ab`):

| cell | pinned s=8 | adaptive | int2 @ s=8 |
|---|---|---|---|
| bs=1 | 85 tok/s | **109 (+28%)** | 94 |
| bs=4 | 310 | **325 (+5%)** | 314 |
| bs=64 | 862 | 855 (-1%) | 922 |

With adaptive splits the fp32 CUDA arm moves AHEAD of OSCAR-INT2 at bs=1 and
bs=4 (int2 itself still pinned at its shipped s=8; its own optimum at bs=1 is
also higher-split, so the bs=1 comparison is best-config-vs-shipped-config).

**gpt-oss-20b geometry.** The kernel source is now parameterized on
(NG, KC, L, KVG) via `VQ2_GEOM_*` build defines (bare `-DL=` clobbers torch
template parameters -- found the hard way), with `_SUPPORTED_GEOMS =
{(32,256,128,4), (16,256,64,8)}`, geometry-keyed extension builds, and
`SGLANG_VQ2_CUDA_GEOM` to prebuild a non-default geometry outside capture.
The structural invariant that made this clean: a packed V byte always holds 4
quarters with dim = byte + (L/4)*quarter, so only bytes/token (NBY), index
words (NIW) and p-chunks (KV4) vary; lanes >= NBY idle in PV for L < 128.

Verification at (16, 256, 64, 8), vs the geometry-generic Triton kernel on
identical inputs (no goldens exist for this geometry; two independent
implementations agreeing IS the check):

| bs | parity (rel) | Triton | CUDA fp32 | ratio |
|---|---|---|---|---|
| 1 | 4.4e-05 | 77 us | 107 us | 0.72x |
| 32 | 5.3e-05 | 1156 us | 974 us | **1.19x** |
| 64 | 6.2e-05 | 2894 us | 1912 us | **1.51x** |

The CUDA kernel wins MORE at gpt-oss geometry than at Qwen's: the 16 KB
codebook halves the shared floor and KVG=8 doubles arithmetic per gathered
byte. bs=1 wants the adaptive split ladder, which applies unchanged.

Default-geometry regression after the templating: parity byte-identical
(rel 5.762e-05 before and after), perf within the day's noise band
(1852 vs 1815-1852 us across the session).

**Open engine-side caveat:** the gpt-oss e2e vq2 path is unverified upstream of
the kernel -- in `throughput_gptoss20b_mxfp4` only the bf16 arm ever produced
cells (int2/vq2 arms have no results), so hybrid-SWA/mxfp4 quant serving is its
own workstream. The kernel is ready for it; the engine may not be.
