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

| length | bf16 | oscar_int2 | vq2 (s8) | vq2 (s48) |
|---|---|---|---|---|
| 30k | 467 (b12) | **926 (b64) 1.98×** | 569 (b64) 1.22× | 568 (b64) 1.22× |
| 60k | 178 (b6) | **481 (b39) 2.70×** | 287 (b39) 1.61× | 286 (b39) 1.61× |
| 90k | 91 (b4) | **282 (b26) 3.11×** | 184 (b26) 2.03× | 191 (b26) 2.11× |

**Qwen3-4B-Thinking-2507**

| length | bf16 | oscar_int2 | vq2 (s8) | vq2 (s48) |
|---|---|---|---|---|
| 30k | 370 (b13) | **970 (b64) 2.62×** | 588 (b64) 1.59× | 585 (b64) 1.58× |
| 60k | 217 (b7) | **506 (b45) 2.34×** | 306 (b45) 1.41× | 294 (b45) 1.36× |
| 90k | 95 (b4) | **332 (b30) 3.49×** | 192 (b30) 2.02× | 197 (b30) 2.07× |

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

## Kernel investigation: no further win

Full record in `pipelines/throughput/kernel_study/README.md`. The headline:
ablating the real kernel shows the **codebook gather is 52% of it** (1009 of
1934 us at bs=64); remove it and vq2 runs at 925 us against int2's 1634 us. The
whole gap is the gather.

Eight approaches were measured and rejected -- `tl.gather` (1.7-4.3x slower),
shared-memory staging fused (0.40x), smaller codebooks (1.28x for half the rate),
split-K, pipelining, DRAM overlap (already achieved), removing `tl.trans` (it was
never a real transpose -- Triton folds it into `ldmatrix.trans`), and a full
TileLang rewrite (correct, 0.40x). The one untried lever is group-major `isel`,
which needs a strided-coordinate codebook bundle.

**Methodological caution recorded there too**: five microbenchmarks inverted
their ranking when measured end-to-end, each because its regime differed from the
served one. Only `bench_stage1.py` (real shape, engine-derived splits) agrees.

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
