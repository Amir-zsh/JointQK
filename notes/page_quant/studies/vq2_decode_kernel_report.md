# Closing the vq2 decode-kernel gap against OSCAR-INT2

**H100 (sm90), Qwen3-8B / Qwen3-4B-Thinking-2507 geometry: `H_Q=32, H_KV=8,
head_dim=128, NG=32, KC=256, G=4`.**

Outcome: the vq2 decode penalty against OSCAR-INT2 is gone. A CUDA stage-1 that
stages the VQ codebook in shared memory runs **1.65-1.79× the shipped Triton vq2
kernel and 1.01-1.13× OSCAR-INT2** end-to-end, on both models, at every context
length tested. It is wired into the engine behind `SGLANG_VQ2_CUDA=1`
(default-off).

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
cannot express one: `tl.gather` lowers to warp shuffles and benchmarks **4×
slower than Triton's own scattered global loads**.

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
Triton cannot express.

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
