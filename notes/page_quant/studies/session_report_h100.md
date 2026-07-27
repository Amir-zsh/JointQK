# H100 session report — vq2 optimization, an OSCAR prefix-cache bug, and the Figure 4 reproduction

Scope: everything changed in this session, what was measured, what was rejected,
and what is still open. Nothing here is committed.

---

## 1. Summary

Three separate lines of work:

1. **vq2 decode optimization.** Three optimizations adopted, four rejected. Net on
   the served benchmark: median TPOT 36.26 → 32.84 ms (−9.4%), mean TTFT 10119 →
   7327 ms (−27.6%).
2. **A real bug in OSCAR's released code**: mixed-KV prefix reuse is capped at the
   first prefill chunk. Root-caused, fixed behind a flag, validated on 800 paired
   NIAH rows.
3. **OSCAR Figure 4 reproduction.** Left panel reproduces on both Qwen models.
   Right panel does not, and we established *why* — it is a serving-workload
   property, not a stack deficiency.

**Every flag we added defaults to OFF.** With no env set, the engine behaves as
it did before this session, except for one deliberate revert described in §5.

---

## 2. Performance optimizations

Measured on Qwen3-8B, H100, in=32768 out=512 conc=8, KV_SPLITS=48, unless noted.
Baseline = plain vq2, 36.26 ms median TPOT.

### 2.1 Adopted

| flag | what it does | effect |
|---|---|---|
| `SGLANG_VQ_OPT_QMAP` | fused Triton per-head query map, replacing `vq_map_q`'s permute → bmm → permute → `contiguous()` chain | **bit-identical**; −1.1% |
| `SGLANG_VQ_OPT_FLUSH` | fused nearest-centroid encode for the decode aging flush; the torch path materialised an `[L,n,H,NG,K]` fp32 score tensor (~76 MB at n=64) and made 3 passes over it | 3.5–8.2× isolated; QMAP+FLUSH together 36.26 → 33.68 (−7.1%) |
| `SGLANG_VQ_OPT_PREFILL` | routes prefill/extend VQ writes through that same fused kernel via a new single-layer adapter `vq_encode_single`; the unfused path built a `[chunk,H,NG,K]` fp32 score tensor (537 MB at a 2048-token chunk) | **TTFT −27.6%** (10108 → 7327 ms), TPOT −2.5% |

Files: `mem_cache/vq_codebook.py` (`vq_map_q_fused`, `_vq_encode_kernel`,
`vq_encode_fused`, `vq_encode_single`), `mem_cache/unified_kv_pool.py`
(call sites `_vq_write_k` / `_vq_write_v`), `layers/attention/triton_backend.py`
(qmap dispatch).

Validation: QMAP is bit-identical. FLUSH and PREFILL are **not** bit-exact — the
fused kernel accumulates differently, so ~0.2% of centroid indices differ, always
at near-ties (max score gap 4e-3..9e-3, scales identical to 0). NIAH-64K over 800
rows: 77.69 → 78.09, unchanged within noise.

### 2.2 Rejected — all four removed from the tree

| idea | result | why it failed |
|---|---|---|
| `CB16` — gather pre-dequantised fp16 centroids instead of packed int32 + a 4-pass fp8 decode | **+27% slower** | doubles gather bytes (8 B vs 4 B); the codeword gather is L1-sector/LSU-bound, not ALU-bound |
| on-chip codebook via `tl.gather` + 4-dot restructure | **2.2–5.7× slower** | probe in `scratchpad/probe_gather.py`. `tl.gather` lowers fine (`emitGatherInShared`, no per-iteration re-store) — it is simply worse. The 4-dot form is also 15–19% worse because Triton inserts a shared-memory round-trip per dot operand |
| `SPLITK` — bound the stage-2 reduction by actual per-request split counts | **+0.4%** | unused splits were *already* skipped by the `-inf` LSE check, so there was nothing to save |
| `ADAPTSPLIT` — derive the split cap from batch size | −4% but **unsafe** | CUDA graphs capture at a bucket size and replay at the real batch size, so the cap baked into the captured grid can be smaller than the cap the replay path clamps to → silently dropped KV at long context |
| `KSCALE1` — drop the dead zero slot from the vq2 K scale arena | **7.6% slower** | bit-identical and +3.70% pool capacity, but costs decode. Kept as a documented negative, not in the tree |
| sm90 launch-config retune (36 configs) | no change | the A100 default `BLOCK_N=128 / warps=4 / stages=3` is the best of all 36 on H100 too |

**One finding worth keeping** from the ADAPTSPLIT work: the *total* split count
(`max_hp_kv_splits + quant cap`) must be divisible by 4. Measured median TPOT by
total: 10→35.94, 12→32.29, 13→38.86, 14→35.77, 16→32.45, 20→32.72, 24→32.88,
56→33.76. Every total divisible by 4 is fast; every other one costs 10–20%.
Anyone hand-setting `KV_SPLITS` such that `8+KV` is not a multiple of 4 silently
loses ~20%.

### 2.3 Why the attention kernel could not be improved

Trace attribution (bs=8, ctx≈34K, per decode step, 36 layers): the vq2 stage-1
attention kernel is 11.278 ms vs int2's 7.872 ms — **+3.4 ms, ~100% of the
remaining gap** once QMAP and FLUSH landed. Both formats read near-identical DRAM
bytes per token (76 vs 80 B), so it is not bandwidth. The cost is a *dependent*
gather: 4096 scattered 4-byte lookups per BLOCK_N=128 iteration into a 32 KiB
per-(layer,head) codebook, at dependency depth 3 vs int2's 2. Every attempt to
remove or restructure that gather made things worse. **This cost is intrinsic to
the VQ format**, not an implementation defect.

---

## 3. The OSCAR prefix-cache bug

**This is OSCAR's code, not ours.** `mem_cache/common.py` and
`mem_cache/radix_cache.py` are unmodified in our tree apart from the gated fix;
the last upstream commit on `radix_cache.py` is their own `e2c6544`.

### Symptom

Send an identical prompt twice to a warm server with the radix cache enabled. The
second request should be served from cache. On the mixed-KV pool the reuse caps
at `chunked_prefill_size − hp_recent_tokens`, regardless of prompt length:

| mode | chunk | cached of 30,000 | warm TTFT |
|---|---|---|---|
| int2 | 8192 (default) | **7,936** | 984 ms |
| int2 | 32768 | 29,728 | 111 ms |
| bf16 | 8192 | 29,999 | 166 ms |

bf16 on identical flags does not cap, so this is mixed-KV specific.

### Root cause

`common.py:543` folds **two different constraints** into one variable,
`req.mixed_kv_quant_slack_cutoff_len`, via a running `min`:

* **A** — the HP-recent tail bound on a non-final chunk. Purely positional; it
  *grows* with `seq_len` and has no persistent ownership behind it.
* **B** (`:563`, `:590`) — partial-quant-page ownership, backed by accumulating
  `mixed_kv_quant_slack_indices`. Persisting this **is** correct: a partially
  owned page must not be donated to the radix tree or `alloc_quant` will reissue
  it to a second request and it gets double-freed.

Because A is min-accumulated with B, it latches at chunk 1's value and never
advances. It is only reset in `_with_mixed_quant_slack`, i.e. at request
completion — never between chunks.

One mechanism predicts all three measurements: with a chunk larger than the
prompt there is no *non-final* chunk, so the latching branch never runs.

### Fix

`SGLANG_MIXED_KV_PREFIX_REUSE_ACROSS_CHUNKS` (default **OFF**). Constraint A gets
its own per-step field, assigned fresh each chunk and cleared on the final chunk.
B's variable, accumulation and reset are untouched. Files: `environ.py`,
`managers/schedule_batch.py`, `mem_cache/common.py`, `mem_cache/radix_cache.py`,
`mem_cache/chunk_cache.py`.

The read site (`_mixed_kv_slack_insert_limit`) mins both bounds and needs no flag
check: with the fix off the new field is never written, stays `None`, and is
filtered out.

### Validation

* **Inert when off** — flag-off reproduces the pre-patch numbers exactly on all
  three cells (7,936 / 29,728 / 29,999).
* **Works when on** — 7,936 → 29,728 at the default chunk; warm TTFT 984 → 109 ms
  (**9.0×**).
* **Accuracy** — NIAH-64K, 800 paired rows, all 8 subtasks, greedy, radix cache
  on, default chunk size. Paired exact sign test: control better on 45 rows,
  treatment on 34, **p = 0.260**. Overall 29.50% → 28.66%. No significant
  difference; disagreements balanced, consistent with cache-order numerical flips
  (reusing a *quantised* cache changes reduction order, so greedy is not
  deterministic across arms).

Why it matters beyond us: it costs OSCAR prefix caching on any prompt longer than
one chunk, which is their headline serving scenario.

**Not touched:** the `cache_finished_req` early-return (`radix_cache.py:571`).
It guards a documented use-after-free (~0.5% gibberish under concurrent retract)
and I could not verify its failure mechanism, so it was left alone.

---

## 4. Figure 4 reproduction

### Left panel — reproduces

decode throughput speedup vs BF16 at bs=1 (`decode_tok_s` = 1000/TPOT, which
excludes the request's own prefill by construction):

| context | paper OSCAR | ours OSCAR-INT2 | ours vq2 (tuned splits) |
|---|---|---|---|
| **Qwen3-4B-Thinking** 30k / 60k / 100k | 1.98 / 2.52 / 3.08 | 2.12 / 2.64 / 3.01 | 2.34 / 3.23 / 3.81 |
| **Qwen3-8B** 30k / 60k / 100k | 1.84 / 2.29 / 2.88 | 2.02 / 2.36 / 2.70 | 2.03 / 2.82 / 3.29 |

Within +7%/+5%/−2% (4B) and +10%/+3%/−6% (8B). Reproduced twice, under both
triton and fa3 prefill backends, which agree to ~1%.

**vq2 at its tuned split-K beats OSCAR at every context on both models.**

### Right panel — does not reproduce, and we know why

The panel's value depends entirely on the serving workload, which the paper does
not specify. Three regimes, all measured:

| regime | advantage from | behaviour vs batch | measured (8B) |
|---|---|---|---|
| neither arm thrashes | decode rate | **shrinks** | 2.75 → 1.44 → 1.06 |
| only bf16 thrashes | KV capacity | **grows** | 2.55 → 4.15 → 6.09 |
| both thrash | neither | compresses | 1.93 at BS=32 |

The paper's curve grows monotonically to BS=32, so their workload keeps bf16
over-subscribed while OSCAR stays resident. Our two workloads bracket their
number: total prefix sharing gives 1.41× at BS=8, zero sharing gives 6.09×, the
paper says 3.35×.

The capacity cliff is visible directly. Qwen3-8B, bf16 pool 373,892 tokens
(≈3.7 × 100k prompts):

| BS | tokens needed | bf16 tok/s | bf16 TTFT | prefill_share | ratio |
|---|---|---|---|---|---|
| 2 | 202k (fits) | 42.0 | 330 ms | 0.007 | 2.55× |
| 4 | 404k (8% over) | **38.6** | **16,719 ms** | 0.259 | **4.15×** |
| 8 | 808k (2.2× over) | 39.9 | 74,489 ms | 0.58 | 6.09× |

At BS=4 bf16 throughput *falls* while the batch doubles. Qwen3-4B, with ~8 GB
smaller weights and therefore a larger KV pool, does **not** thrash at BS=4
(TTFT 621 ms, ratio 2.07×) — the model with more spare memory thrashes later,
exactly as the capacity account predicts.

### Cross-check that settles the mechanism

Their own **Table 8** (per-decode kernel profiling, Qwen3-8B, 1×H100) gives
BF16 23.4 ms vs OSCAR 15.5 ms at B=32 = **1.51× at the kernel level**, with BF16
"capacity limited" only from B=64. Figure 4 claims 3.44× at BS=32. So roughly
2.3× of that panel is **serving-layer admission/queueing**, not kernel speed.
That is precisely why the left panel (a decode-rate measure) reproduces cleanly
and the right panel does not.

### What OSCAR ships

Verified against the live repo (`github.com/FutureMLS-Lab/OSCAR`, latest commit
`41ebcdb` 2026-06-28, README-only since June):

* `rotation/` — 19 scripts, all rotation calibration or GPQA **accuracy** eval.
  No mention of throughput, tok/s, or `bench_serving`.
* `benchmark/kernels/quantization/bench_decode_int_kv.py` (+ `.example.yaml`) —
  the **only** OSCAR-authored perf script, and it is kernel-level. Its example
  config defaults to `batch_size: 128`, `profiling_mode: torch_profiler`, and
  seq_lens 8k/16k/32k/64k/128k — i.e. Table 8's shape, not Figure 4's.
* Everything else with `bench_serving`/`throughput` in the name appears in *both*
  their fork and the vanilla sglang copy they vendor alongside it → inherited.

**No script reproduces Figure 4.** It came from tooling that was not released.

---

## 5. Flags and defaults

### Ours — all default OFF

| flag | default | status |
|---|---|---|
| `SGLANG_VQ_OPT_QMAP` | OFF | adopted; enable for vq2 runs |
| `SGLANG_VQ_OPT_FLUSH` | OFF | adopted; enable for vq2 runs |
| `SGLANG_VQ_OPT_PREFILL` | OFF | adopted; enable for vq2 runs |
| `SGLANG_MIXED_KV_PREFIX_REUSE_ACROSS_CHUNKS` | OFF | validated; enable for throughput work |

Reproduce the adopted vq2 stack with:
`SGLANG_VQ_OPT_QMAP=1 SGLANG_VQ_OPT_FLUSH=1 SGLANG_VQ_OPT_PREFILL=1`

### One deliberate default change

`SGLANG_VQ_FP8_FMT` now defaults to **`e5m2`** (`vq_codebook.py:resolve_vq_fp8_fmt`).

Mid-session it defaulted to `auto`, which resolves to **e4m3 on sm90** — a silent
numerics change versus upstream that our runs never saw because they all pin
e5m2. e4m3 is genuinely better where available (rel L2 centroid error 2.643% vs
5.668% on the shipped bundle; 0 of 1.57M values saturate against its 448 ceiling)
but downstream it measured neutral (NIAH +0.22 = noise, 0.00 speed). The deciding
factor: **A100 physically cannot use e4m3** — Triton gates `fp8e4nv` at
capability ≥ 8.9 (`triton/backends/nvidia/compiler.py:188`) and A100 is sm80 — so
every A100 baseline is e5m2 by construction, and defaulting to e4m3 would break
comparability with them. `e4m3` and `auto` remain opt-in.

### serve_oscar.sh additions

| knob | default | purpose |
|---|---|---|
| `RADIX_CACHE` | `0` (cache disabled) | `=1` keeps the prefix cache on for reuse benchmarks |
| `PREFILL_BACKEND` | `triton` | `=fa3` matches OSCAR's own `eval_oscar_gpqa.sh` |
| `DECODE_BACKEND` | `triton` | unchanged from before |

Note fa3 **works** but is *inert* on the quant path: when `kv_cache_dtype ==
"int2"` both backends delegate to `quantized_kv_prefill`
(`flashattention_backend.py:687`), so an A/B measured 1192 vs 1185 ms. Quant
prefill is bound by rotate/clip/pack, not attention.

### One upstream bug fix, unconditional

`QuantKernel/fused_hadamard_int2_kv.py` (+5 lines) passes `LLOYD_MAX` to
`_quantized_set_kv_int2_kernel`, which declares it with no default. Only the
clipped writer passed it; the pretransformed path did not, so it raised. Not
flag-gated because the code does not run without it.

---

## 6. New tooling

| file | purpose |
|---|---|
| `pipelines/oscar_e2e/run_decode_speed.py` | config-driven serving-throughput driver: arms × (input_len, batch_size, groups) grid, boots per-GPU, records `provenance.json` per arm, aggregates. Cell sharding across spare GPUs; per-cell server health check |
| `pipelines/oscar_e2e/configs/*.json` | Figure 4 panels for both models, plus replay and prefix-group sweeps |
| `pipelines/oscar_e2e/repro_chunk_prefix_cache.py` | minimal 3-cell repro of the prefix-cache bug with an explicit verdict; also validates the fix via `--fix` |
| `pipelines/oscar_e2e/verify_chunk_cutoff_fix.py` | correctness probe on real NIAH content |
| `pipelines/eval/paired_niah_ab.py` | paired sign test on identical rows; scoring cross-checked against `SCORER_REGISTRY` and refuses to report on mismatch |
| `pipelines/eval/plot_fig4.py` | renders the reproduction with published values overlaid |
| `pipelines/oscar_e2e/run_fig4_all.sh`, `run_fig4_replay.sh` | end-to-end chains |

Measurement discipline built into the harness, each after being burned once:

* `provenance.json` per arm records what actually loaded (`kv_cache_dtype`, split
  count, pool size, whether a VQ codebook loaded). A "control" arm that silently
  ran the wrong mode produced a plausible-looking result before this existed.
* `prefill_share` gate, **workload-aware**: it rejects prefill-dominated cells for
  the shared-prefix workload, but *reports* them under replay, where a high share
  is the finding rather than a fault.
* Per-cell `/health` probe: a dead server used to turn one crash into N
  "no throughput line" rows that read as independent failures.
* Port-scoped `pkill` and port-tagged log names: concurrent runs were killing each
  other's servers and parsing each other's logs.
* Logs under `$ROOT` (bind-mounted at an identical path in the container), never
  the scratchpad, which the container sees at a different path.

---

## 7. Corrections made during the session

Recorded because the pattern matters more than any single item: several
confident claims were wrong and were caught only by re-measuring.

* "The quant pool never contributes cached tokens" — wrong; it caches up to one
  chunk. Concluded from a `tail -2` of one log.
* "fa3 is missing from this build" — wrong; I probed `sgl_kernel` when sglang
  imports it from `sglang.jit_kernel.flash_attention`.
* "CB16 shows the gather is DRAM-bandwidth-bound" — wrong; it is L1-sector bound.
* A `plan_quant_split_cap` refactor added "for safety" cached a value across
  forward types and cost **25%**, invisible for hours because I stopped
  re-measuring after adopting it.
* A paired-analysis scorer used `any(ref in pred)` where the official scorer is
  `sum(found)/len(refs)`; it reported multiquery at 81% vs the official 38.2%,
  and initially returned 0/800 for both arms — a broken test that *confirms* the
  hoped-for answer.
* A guard that refused to report 0%/100% arms would have suppressed a genuine
  catastrophic regression. Replaced with cross-validation against the
  authoritative scorer — verify the method, never gate on the answer.
* The shared-prefix workload choice for Figure 4 right, which erased the very
  capacity effect the panel measures.

---

## 8. Open items

* **Figure 4 right** is unreproduced. The prefix-group sweep at BS=32 is landing
  in the paper's range (3.52× and 4.14× against their 3.44×) but has not been
  mapped to specific group counts yet.
* `notes/fig4_reproduction.png` predates the replay/BS-fill data and should be
  re-rendered.
* The BS=1/8 replay points are at `mem_frac 0.85` while the group sweep is at
  `0.70`. Pool size *is* the independent variable in the capacity story, so those
  two sets should not be plotted on one axis without saying so.
* `KSCALE1` (+3.70% pool capacity, bit-identical, 7.6% slower) was removed. If
  capacity ever binds harder than decode speed, it is worth revisiting.
* Nothing is committed.

---

## 9. Final result: Figure 4 right is workload-determined

Prefix-sharing sweep, BS=32, 100k input, `mem_frac 0.70`. Identical code,
hardware, batch size and context — only the number of DISTINCT shared prefixes
(`--gsp-num-groups`) varies:

| groups | Qwen3-8B ratio | bf16 prefill_share | Qwen3-4B ratio |
|---|---|---|---|
| 1 (all share one prompt) | **1.08x** | 0.09 | **1.09x** |
| 8 | **5.66x** | 0.89 | **3.99x** |
| 16 | -- | -- | 3.22x |
| **paper** | **3.44x** | | **6.17x** |

At g=1 bf16 is not stressed and the advantage collapses to decode rate alone.
At g=8 bf16 thrashes (prefill_share 0.89) while OSCAR stays resident (0.28).

**The same stack produces 1.08x to 5.66x purely by changing prefix-sharing
structure**, and the paper's 3.44x falls inside that range. Since Figure 4's
workload is unspecified and no script was released, the number is not
reproducible in the strict sense: it is a point on a curve whose x-axis the paper
does not report. Our two extremes bracket it; that is the strongest statement the
evidence supports.

Sweep coverage is sparse (several oscar cells at g=2/4 lost to server OOM at
BS=32), so the g=1 and g=8 endpoints are solid and the middle is not.

### Two harness defects found here, both fixed

* The `prefill_share <= 0.20` gate was rejecting every thrashing cell, which is
  exactly where the finding lives. It now applies only where a full cache hit is
  expected (single shared prefix) and reports elsewhere. It had silently emptied
  the entire sweep table.
* `pkill -f launch_server` does not match sglang's `sglang::scheduler` children.
  Three orphans held 190 GB across three GPUs after their launchers exited. Reap
  by scheduler name; this is the same orphan pattern behind the GPU contamination
  seen earlier in the session.

### For the next benchmark

* Pool size is the independent variable in any capacity comparison, so `mem_frac`
  must be fixed across an entire sweep. Our 0.85 and 0.70 sets are not comparable.
* Record `decode_tok_s` (prefill-free) and `output_tok_s` (prefill-inclusive)
  together; they answer different questions and diverge under load.
* State the prefix-sharing structure explicitly. It moved a single measurement
  between 1.08x and 5.66x.
* At BS=32 with 100k contexts a cell takes 5-15 minutes and servers OOM
  unpredictably; per-cell isolation, health checks and orphan reaping have to be
  designed in, not bolted on.
