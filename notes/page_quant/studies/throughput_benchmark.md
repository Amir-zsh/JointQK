# Controlled decode-throughput benchmark

Built 2026-07-27. Harness: `pipelines/throughput/`. Chart: `notes/throughput_benchmark.png`.

## Why this exists

Reproducing OSCAR's Figure 4 never converged, because the published workload is
under-specified — prefix-sharing structure alone moved one measurement between
1.08× and 5.66×, and their repo contains no script that produces the figure
(checked against their live GitHub, latest commit `41ebcdb`). This benchmark
replaces guesswork about someone else's protocol with one whose every knob is
pinned and recorded.

## Protocol

Three operating points per (model, arm, input length): **bs=1** (equal-setting
latency), **bs=4** (scaling), **bs=max** (largest batch whose KV fits that arm's
own pool).

| knob | value |
|---|---|
| lengths / output | 30k, 60k, 90k / 1024 tokens, `ignore_eos` |
| models | Qwen3-8B, Qwen3-4B-Thinking-2507, both TP=1, H100 |
| arms | bf16, oscar_int2, vq2 at split-K 8 and 48 |
| `mem_fraction_static` | 0.85, identical across every arm |
| cross-request prefix sharing | none — every prompt a distinct random token stream |
| per-request prefix warmup | yes, `RADIX_CACHE=1`, warm pass then measured pass |
| chunked prefill | 4096 |
| calibration artifacts | **synthetic** (random rotations + random codebook) |

**Synthetic artifacts** (`make_synthetic_artifacts.py`) generate rotations and a
VQ codebook from a model's `(L, H, D)` alone. Throughput does not depend on their
contents — the kernel issues the same gathers and GEMM shapes — and this removes
the per-model calibration hurdle entirely. It is what let Qwen3-4B-Thinking run a
vq2 arm at all; no calibrated codebook exists for it, so the Figure 4 study could
only give it bf16 + int2. **These runs are throughput-valid and accuracy-invalid.**

**Warmup is load-bearing, not a refinement.** Cold, a bs=max cell never forms:
prefilling 64 × 30k ≈ 2M tokens takes minutes while 1024 decode tokens take ~30 s,
so early requests finish generating before late ones start decoding. The warm pass
collapses TTFT to roughly one decode step, and costs almost no extra wall clock —
it does the prefill the cold run would have done anyway, separated from the decode
rather than interleaved with it.

## Engine bug found and fixed: vq2 pool double-count

`pool_configurator.py:265-269` added `head_dim//4` to `bytes_per_head` for the vq2
K-index arena, on top of `_get_unified_mixed_kv_bytes_per_quant_token` (`:128-151`)
whose 64-byte term **already covers K+V**. That term is written as a page-sharing
identity, `(head_dim + v_head_dim) * hp_bytes // n_q`, which is what hid the
double-count. The block's own comment said "**instead of** head_dim/4 packed int2
bytes" — replacement was intended, `+=` was written. `_create_arenas`
(`unified_kv_pool.py:501-536`) allocates the VQ arena in the `else`-branch *place*
of the int2 K arena, so only one K arena ever exists, and at G=4 both cost
`NG = head_dim/4 = 32` B. The correct adjustment was exactly zero.

Confirmed three independent ways before touching anything:

1. Pool ratio `1705776/2388088` = **0.714286** = exactly `80/112`, six digits, on two models.
2. Predicting the arenas from **80** B/head/token reproduces the logged "KV Cache is allocated" line exactly for both arms (int2 52.66 GiB, vq2 38.02 GiB). At the charged 112 B, vq2 would have allocated 52.66 GiB — it allocated 38.02.
3. The difference was idle HBM: phantom reservation 14.64 GiB vs observed post-pool free-memory delta 14.59 GiB.

| Qwen3-8B, mem_frac 0.85 | before | after |
|---|---|---|
| vq2 pool tokens | 1,705,776 | **2,389,000** (+40.1%) |
| int2 pool tokens | 2,388,088 | 2,389,000 |
| vq2 KV allocated | 38.02 GB | 52.68 GB |
| vq2 idle HBM | 24.96 GB | 10.33 GB |

Fixed unconditionally. **Prior vq2 artifacts no longer reproduce bit-for-bit** —
any vq2 run before this used a pool 28.6% smaller than its budget allowed.

## Results — aggregate decode throughput, prefill excluded

Speedup is each arm's best achievable vs bf16's best at the same length; batch
sizes differ by arm, which is the point.

**Qwen3-8B**

| length | bf16 | oscar_int2 | vq2 (s8) | vq2 (s48) | **vq2_cuda** |
|---|---|---|---|---|---|
| 30k | 467 (b12) | 926 (b64) 1.98× | 569 (b64) 1.22× | 568 (b64) 1.22× | **938 (b64) 2.01×** |
| 60k | 178 (b6) | 481 (b39) 2.70× | 287 (b39) 1.61× | 286 (b39) 1.61× | **501 (b39) 2.81×** |
| 90k | 91 (b4) | 282 (b26) 3.11× | 184 (b26) 2.03× | 191 (b26) 2.11× | **318 (b26) 3.50×** |

**Qwen3-4B-Thinking-2507**

| length | bf16 | oscar_int2 | vq2 (s8) | vq2 (s48) | **vq2_cuda** |
|---|---|---|---|---|---|
| 30k | 370 (b13) | 970 (b64) 2.62× | 588 (b64) 1.59× | 585 (b64) 1.58× | **976 (b64) 2.64×** |
| 60k | 217 (b7) | 506 (b45) 2.34× | 306 (b45) 1.41× | 294 (b45) 1.36× | **513 (b45) 2.37×** |
| 90k | 95 (b4) | 332 (b30) 3.49× | 192 (b30) 2.02× | 197 (b30) 2.07× | **344 (b30) 3.61×** |

Speedup grows with context length, as expected: the capacity advantage compounds
as each request's KV gets larger. int2 beats vq2 throughout — vq2's decode kernel
remains the bottleneck (its dependent codebook gather, documented previously), and
the pool fix removed vq2's capacity handicap without touching that.

Split-K is a genuine trade: at bs=1 s48 wins (98 vs 83 tok/s at 30k on the 8B),
at bs=4 it loses (249 vs 273). At bs=max the two converge.

### Why int2 beats vq2 — it is the kernel, not a missing flag

Checked both ways. The `SGLANG_VQ_OPT_{QMAP,FLUSH,PREFILL}` flags have live call
sites (`triton_backend.py:1861`, `unified_kv_pool.py:1042`, `:970`/`:1007`), and
an explicit A/B with them forced to 0 confirms they are active and worth real
time (Qwen3-8B, 30k, `artifacts/throughput/vq_opt_ab/`):

| batch | opts OFF | opts ON | gain | oscar_int2 | int2/vq2 |
|---|---|---|---|---|---|
| 1 | 81 | 83 | +2.5% | 95 | 1.14× |
| 4 | 250 | 274 | +9.6% | 316 | 1.15× |
| 16 | 414 | 477 | +15.2% | 740 | **1.55×** |

The remaining gap is **batch-dependent**: 1.14–1.15× at bs=1/4, 1.55× at bs=16,
1.63× at bs=64. That matches the profiled mechanism — vq2's decode attention
issues a dependent gather of 4096 scattered 4-byte codebook lookups per BLOCK_N
iteration (11.278 ms/step vs int2's 7.872 ms), a cost that scales with tokens
decoded per step, while int2's bit-unpack does not. The pool fix removed vq2's
*capacity* handicap; this is the *kernel* handicap and is what remains to attack.

## Metric choice, and a bias it exposed

Headline is **the sum of per-request decode rates**, each excluding that request's
own prefill. The obvious alternative — `bs × out / (latency − last_ttft)` — divides
*all* tokens by only the post-`last_ttft` window, crediting tokens emitted earlier
to a shorter window. Measured overstatement: **+5–7%** at the largest batches,
0 when entry is synchronous. Both are recorded per cell (`decode_tok_s` and
`decode_tok_s_total`) along with their gap, so the correction is auditable.

## Gates

Every cell records pass/fail on: exact output length (no early EOS), no
retraction, warmup effective, batch concurrency reached, no NaN, and agreement
with the server's own `last_gen_throughput`. Things they caught:

- **EOS**: verified `finish_reason: length` and exactly 1024 tokens on every request. With random codebooks the model emits garbage, so EOS could otherwise have fired at different rates per arm.
- **Warmup depth**: measured via `last_ttft / warm_s` (bf16 ≈ 0.03, vq2 ≈ 0.11), not via `#cached-token`, which is logged *per prefill batch* and therefore sums across co-scheduled requests — comparing it to a single `ctx` passes trivially at bs>4.
- **mem_frac 0.90 was unsafe.** It passed the calibration boot but OOM'd every quant arm partway through the 90k cells: HP arena 4.56 GiB + bs=64 CUDA graphs + 90k prefill activations exceeded the 7.9 GiB reserve. Dropped to 0.85 (uniformly) and shrank the pinned HP-prefix pool.

## Engine fix 2: vq2 launch config never scaled with batch

`_decode_grouped_att_m_fwd_quant_vq2` pinned `BLOCK_N=128, BLOCK_H=8, warps=4` at
every batch size. int2's launcher has always had a batch-adaptive ladder; vq2
never did, and that asymmetry is a large part of why the gap *widens* with batch
(1.14x at bs=1 -> 1.63x at bs=64). Measured at the served shape with the engine's
own split heuristic (`pipelines/throughput/kernel_study/bench_stage1.py`):

| batch | adaptive | old fixed 128/8/4 | int2 | vq2/int2 |
|---|---|---|---|---|
| 1 | 111.7 us | 109.3 us | 97.6 | 1.14x (was 1.12x) |
| 4 | 156.5 us | 153.7 us | 147.0 | 1.06x (was 1.05x) |
| 64 | **1961.7 us** | 2210.4 us | 1657.1 | **1.18x (was 1.33x)** |

1.13x at bs=64, unchanged below bs=16. End-to-end this measured +4.7% (620 vs
592 tok/s), moving the served gap ~1.63x -> ~1.55x. Note vq2 wants a *larger*
config than int2 at the same batch -- its codebook gather is a dependent load
needing more warps in flight -- so int2's 32/4/1 large-batch tier measures worse
for vq2 (2171 us). It gets its own ladder rather than copying int2's.

## Kernel investigation: the gap to int2 is closed (in a CUDA kernel)

Full record in `pipelines/throughput/kernel_study/README.md`.

**Result.** The shipped Triton vq2 kernel runs at 1.21x int2. A hand-written CUDA
stage-1 that stages the codebook in **shared memory** runs at **0.94-1.05x int2**
and **0.75-0.83x Triton**, validated to rel 3.2e-04 against the shipped kernel:

| shape | int2 | Triton vq2 | CUDA vq2 | vs int2 | vs Triton |
|---|---|---|---|---|---|
| bs=64 ctx=30k | 1602 us | 1942 us | **1604 us** | 1.00x | 0.83x |
| bs=32 ctx=60k | 1593 us | 1934 us | **1503 us** | **0.94x** | 0.78x |
| bs=16 ctx=30k | 368 us | 501 us | **375 us** | 1.02x | 0.75x |
| bs=64 ctx=8k | 402 us | 536 us | **423 us** | 1.05x | 0.79x |

This is a *study artifact*, not an engine change: nothing in `vendor/OSCAR-vq` was
modified, and wiring a CUDA extension into sglang's attention backend is a
separate decision.

**Round 2 corrected round 1's diagnosis.** Round 1 inferred the bottleneck by
ablation and blamed "maximum address divergence". Nsight Compute was in fact
reachable -- the driver sets `RmProfilingAdminOnly: 1`, so counters need
CAP_SYS_ADMIN, which a privileged *sibling* container provides without touching
the host driver (`profile_ncu.py`). The counters showed sectors *per request* are
nearly identical between the two kernels (17.8 vs 19.4); what differs is sector
**volume** (94 M vs 392 M), and vq2 sits at 74% of L1 peak. The decisive control:
group-major `isel` -- round 1's "one untried lever" -- has the *best* L1 hit rate
of any variant and is the *slowest*. It also never needed the strided-coordinate
codebook bundle round 1 assumed.

**What actually closed it**, in order: the shared-memory codebook (global sectors
331 M -> 37.4 M); PV register tiling; **fully unrolling the 32-group score loop**
(worth 1843 -> 1630 us on its own -- the score accumulators are a serial
dependency chain with no ILP otherwise); loading the q half2 pair as one 8-byte
access; and unrolling the PV loop.

**A correction worth keeping.** An earlier round measured a *PV-side* FMA ablation
(~61 us) and concluded tensor cores could not help. The *score-side* ablation
tells a different story (~1137 us) -- the PV arithmetic is overlapped with
memory, the score arithmetic is exposed. The right fix turned out to be ILP, not
tensor cores.

**Methodological caution**: round 1 had five microbenchmarks whose ranking
inverted end-to-end because their regime differed from the served one; only
`bench_stage1.py` agrees. Round 2 adds a sixth failure mode -- inferring a
bottleneck by ablation when counters were one container away -- and leaves
`probe_group_size.py` in the tree explicitly labelled INCONCLUSIVE as a live
example of the same hazard.

## vq2_cuda arm: the CUDA stage-1, and the bug that nearly got charted

Full write-up: `notes/page_quant/studies/vq2_decode_kernel_report.md`
(problem, root cause, every approach tried, and the final design).

The CUDA stage-1 is wired in as an opt-in arm (`SGLANG_VQ2_CUDA=1`,
`triton_ops/vq2_cuda_stage1.py`, dispatched from
`_decode_grouped_att_m_fwd_quant_vq2`). Default-off, and `supports()` gates every
assumption, so the Triton path is untouched for anything else.

### The bug: launching on the wrong stream

The arm first measured **1.6x at bs=1 and 8.3x at bs=64** end-to-end. The bs=64
number was impossible on its face -- 5,195 tok/s is 12.3 ms/step, while 36 layers
x the measured ~1.5 ms/layer is ~55 ms of stage-1 alone -- so it was chased rather
than published. Five hypotheses were tested and killed by measurement (GPU
placement, launch failure, paging locality, KV occupancy, server args), and a
liveness probe settled it: `SGLANG_VQ2_CUDA_REP=8` makes the kernel 1.9x slower
per call, the server demonstrably loaded that binary, and throughput did not move.

The torch profiler over sglang's `/start_profile` endpoint
(`kernel_study/profile_server_decode.py`) showed the answer in one line. Comparing
total CUDA kernel time for 64 decode steps:

| | vq2_s8 | vq2_cuda (broken) |
|---|---|---|
| `_fwd_grouped_kernel_stage1_quant_vq2` | 282.39 ms / 2304 calls | **absent** |
| the CUDA `vq2_stage1` | -- | **absent** |
| total kernel time | 813.58 ms | 530.91 ms |

The 282.67 ms difference is exactly the cost of the quant stage-1. **Neither
kernel was running**: the dispatch skipped the Triton launch, and the CUDA kernel
was never captured, so decode was silently skipping long-context attention
altogether. That is why it looked 8x faster and why making the kernel slower
changed nothing.

Cause: the kernel launched as `<<<grid, THR, sm>>>`, i.e. on the **legacy default
stream**. sglang captures decode into CUDA graphs on a side stream, and work on
the default stream is not captured. Eager-mode tests all passed because the
default stream synchronises implicitly -- which is precisely why a unit test
could not catch it. Fixed by launching on `at::cuda::getCurrentCUDAStream()`.

After the fix the profiler shows `vq2_stage1<__nv_bfloat16, long, 128, true>` with
**2304 calls** (64 steps x 36 layers) at 113.8 us against Triton's 122.6 us.

### Final result (18/18 cells pass every gate)

bs=max, aggregate decode tok/s, CUDA graphs enabled throughout:

| model | length | int2 | vq2_s8 | **vq2_cuda** | vs int2 | vs vq2_s8 |
|---|---|---|---|---|---|---|
| Qwen3-8B | 30k | 926 | 569 | **938** | 1.01x | 1.65x |
| Qwen3-8B | 60k | 481 | 287 | **501** | 1.04x | 1.74x |
| Qwen3-8B | 90k | 282 | 184 | **318** | 1.13x | 1.73x |
| 4B-Thinking | 30k | 970 | 588 | **976** | 1.01x | 1.66x |
| 4B-Thinking | 60k | 506 | 306 | **513** | 1.01x | 1.68x |
| 4B-Thinking | 90k | 332 | 192 | **344** | 1.03x | 1.79x |

Two caveats on these numbers:

- **The runs must be serialised.** Running two servers concurrently, or alongside
  another user's job, produced OOMs and hard server kills (no traceback) that
  looked like arm-specific failures and were not: the identical config passes
  9/9 when run alone on a free GPU. Earlier "vq2_cuda OOMs at 60k" conclusions
  were contention artefacts.
- **The vq2_cuda cells were measured on a newer engine tree** than the bf16 /
  int2 / vq2_s8 / vq2_s48 cells, which date from the original sweep. The
  intervening change (`SGLANG_MIXED_KV_EXACT_CHUNKED_PREFILL`) defaults to OFF
  and allocates nothing when disabled, so it should not affect the comparison --
  but the four reference arms have not been re-measured on the current tree, and
  a strict apples-to-apples chart would re-run them.

### Lessons worth keeping

- **A kernel-level speedup that exceeds what the kernel can explain is a bug
  report, not a result.** The arithmetic (layers x per-layer time vs step time)
  falsified it before any profiling.
- **With synthetic codebooks the output is garbage by design, so every accuracy
  signal is blind.** A kernel that silently does nothing passes every gate the
  harness has. `no_nan` and `full_length` cannot catch omitted attention.
- **Eager unit tests cannot catch stream-capture bugs.** Validate a decode kernel
  by kernel COUNT inside a running server, not only by numerics offline.
- Two harness issues surfaced: the bs=64 cell reported `ttft_spread_frac` 0.76
  (vq2_s8: 0.08) while the headline metric sums per-request rates, which is only
  stagger-robust when requests overlap; and `--disable-cuda-graph` roughly halves
  throughput, so any A/B run without graphs understates a decode kernel's effect.

## Known limitations

- The quant arms cannot cache the final partial prefill chunk (the reuse fix commits up to the last *complete* chunk minus the recent window), so they warm to ~95% where bf16 reaches 100%. At `chunked_prefill_size=4096` the residual is ~5%; at 8192 it is 19%.
- `bs_max_ceiling=64` binds at 30k for every quant arm — their pools would admit ~77. Those cells are capacity-limited by the ceiling, not by memory.
- Single run per cell; no repeat-variance measurement yet.
- capacity.py assumes a uniform full-attention pool, so its prediction is an upper bound for hybrid-SWA models (gpt-oss). B_max is always taken from the server's logged pool, which is authoritative regardless.

## gpt-oss-20B integration (partial)

Official `openai/gpt-oss-20b` (MXFP4 weights), TP=2, `configs/gptoss20b_mxfp4.json`.
Driver gained TP support: GPU groups, `TP=n`, `--gpu 0,1`, per-group free-waiting.
Geometry `L24 H8 D64`, 64 q-heads, hybrid SWA (12 of 24 layers full-attention).

**Three boot blockers, two fixed:**

1. **MoE routing** — `NameError: name 'routing' is not defined` (`topk.py:331`). The image's `triton_kernels` has no `.routing` module (the API moved; it ships `topk.py` instead), the import at `topk.py:35` is inside `try/except ImportError: pass`, and the MXFP4 auto-path selects `TRITON_KERNEL` regardless. **Fixed** by pinning `--moe-runner-backend triton`.
2. **fa3 prefill** — `Boolean value of Tensor with more than one value is ambiguous` via `hybrid_attn_backend.py:189` → `flash_attention_v3.py:131`. fa3 does not handle this model's hybrid-SWA path. **Fixed** by using triton prefill for gpt-oss only; the Qwen configs keep fa3.
3. **Quant arms still blocked** — `oscar_int2` and `vq2` die in `_forward_extend_quantized_dense` (`triton_backend.py:654`) at `o.copy_(result.view_as(o))`: `shape '[4, 4096]' is invalid for input of size 262144`. Reproduces identically at TP=1 (`[4,4096]`/262144) and TP=2 (`[4,2048]`/131072), so **it is not TP sharding**. Diagnosis: the learned-attention-sink correction (`:645-651`) assumes `lse` is `[num_heads, total_q]`, but `sgl_kernel/flash_attn.py:158` documents `softmax_lse` as `(batch_size, nheads, seqlen)`. With the observed `[64,64]` lse, `result[4,64,64] * scale[64,64,1]` broadcasts to `64×64×64 = 262144` — exactly the reported size. Not patched speculatively: a wrong lse reshape would silently corrupt attention instead of crashing.

**bf16 KV baseline measured** (pool 2,228,556 tokens):

| length | bs=1 | bs=4 |
|---|---|---|
| 30k | 204 tok/s | 617 |
| 60k | 144 | 531 |
| 90k | 111 | 402 |

bs=max cells were **rejected**: `cached=0` and TTFT spread 2.7–6.7, i.e. the warm
pass did not survive to the measured pass at large batch — the entries are evicted
in between. Qwen at the same batch cached fine, so this is specific to the hybrid
pool. bs=max on gpt-oss needs that resolved before it means anything.

**NVFP4 is not runnable here at all** — `modelopt_quant.py:1521-1526` raises
"Please use Blackwell and above"; these are sm90 cards. MXFP4 is the 4-bit format
that works on Hopper (`mxfp4.py:172`).

**Open**: there is no official bf16 gpt-oss checkpoint. Prior repo work used
`unsloth/gpt-oss-20b-BF16`; not pulled, pending a decision on non-official sources.
