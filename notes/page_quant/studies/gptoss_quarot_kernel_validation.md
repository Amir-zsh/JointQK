# Validating the plain-int2 (QuaRot/Naive) kernel on gpt-oss

*QuaRot (`serve_oscar.sh --int2plain`, `HADAMARD_ORDER`, `k/v_clip_ratio=1.0`)
produces degenerate garbage on gpt-oss-20B — repetition loops, punctuation
salad, `cap_hit_rate=1.0` (never terminates naturally). Eight independent
checks across the pool/dispatch code, a new bypass-quantization diagnostic,
and a kernel-vs-reference numerical audit find no implementation bug at any
layer.
The collapse is a genuine, inherent consequence of unclipped 2-bit min-max
quantization on gpt-oss's K/V distributions, not a defect in this port.*

Model: `openai/gpt-oss-20b` (24 layers, hybrid sliding-window/full attention,
head_dim 64, 8 KV heads / 64 Q heads, learned attention sinks, native MXFP4
MoE). Engine: `vendor/OSCAR-vq` (branch `vq2-longhorizon`), TP=2, A100-80GB.
Investigated 2026-07-28.

---

## 1. The problem

`quarot` is a new arm added to `gptoss20b_mxfp4_v1.json` this session — OSCAR's
own calibration-free Table-2 baseline, added by literally reusing
`serve_oscar.sh`'s existing `--int2plain` mode (`HADAMARD_ORDER=128` default,
no K/V clipping). Every prior use of `--int2plain` in this repo is
Llama-3.1-8B or Qwen3-8B (head_dim 128, dense); this was the first time it was
ever pointed at gpt-oss's hybrid-SWA, head_dim-64 architecture.

First symptom: an immediate crash —
`ValueError: head_dim (64) must be divisible by hadamard_order (128)` — a
real, simple bug: the protocol copied Qwen/Llama's `hadamard_order=128`
without checking it against gpt-oss's head_dim. Fixed to `hadamard_order=64`
(the correct "full Hadamard over head_dim" value for *this* model — not a
downgrade).

That fix cleared the crash but not the real problem: the resulting output is
degenerate garbage, not merely inaccurate:

```
' back. \nThere blue.  The sun is a bright. The sky is a. The green, the green...'
' i. I thavI, i u t or I, t w em the or your m - c and sur I h m,  I em inc c...'
```

`niah_smoke` scored **0.0 on all 8 subtasks**, `cap_hit_rate=1.0` (every
response burns the full token budget without a natural stop). The question
this note answers: is that a bug in our plain-int2-pool integration for
gpt-oss, or an inherent property of the method?

---

## 2. Structural hypotheses ruled out (code + live-config tracing)

Four candidate gpt-oss-specific gaps were checked and eliminated before
touching any diagnostic tooling:

1. **`SGLANG_OSCAR_ABSORB_V_ROTATION`.** Initially suspected as "the hybrid-SWA
   fix `--int2plain` never sets." Wrong: `models/utils.py`'s absorb helper
   opens with `if not envs.SGLANG_OSCAR_ABSORB_V_ROTATION.get(): return False`
   — it's a no-op unless *explicitly enabled*, and every gpt-oss arm
   (including the working `oscar_int2`/`vq2`) explicitly sets `ABSORB_V_ROT=0`,
   matching the default. Identical no-op state across every arm; not the
   differentiator.
2. **SWA/full-attention layer split (`SWAKVPool`).** Which layers get
   quantized is driven by `self.kv_cache_dtype == "int2"` in
   `model_runner_kv_cache_mixin.py` — generic, identical logic regardless of
   mixed-pool vs plain-pool mode. Sliding-window layers stay unquantized bf16
   in both.
3. **`kv_cache_quant_group_size`.** Confirmed live via
   `resolved_server_info.json` on a real boot: `"kv_cache_quant_group_size":
   null` — correctly resolves to per-head single-group
   (`group_size = head_dim`), matching the "required for hybrid-SWA models"
   contract, independent of arm.
4. **The Hadamard transform itself.** Traced both the write-side blocked FWHT
   (`_fwht_blocked_segments_tensor`, parameterized by `head_dim`/`log_n`) and
   the read-side `apply_segmented_hadamard_transform` — both generically
   handle any `(head_dim, hadamard_order)` pair satisfying the divisibility
   constraint; no hardcoded 128. **Decisive check:** reran with
   `SGLANG_INT2_NO_HADAMARD=1` (rotation forced to identity — OSCAR's
   "Naive-INT2" baseline). Produced the **identical flavor of garbage**
   (repetition loops, punctuation salad). This conclusively rules out rotation
   as the cause — the bug (if any) is in the shared int2 quantize/dequantize
   path, not the Hadamard code.
5. **Learned attention sinks.** gpt-oss's sink correction (`sinks` parameter)
   was traced through every branch of `forward_decode`'s dispatch in
   `triton_backend.py` — consistently threaded to
   `decode_attention_fwd_quantized` (the plain-pool path) exactly as it is to
   the mixed-pool and vq2 paths. Not silently dropped.

## 3. The comparison that reframed the question

Before assuming "bug," compared `quarot`'s garbage against `oscar_int2`'s own
degraded-but-scored-0.0 outputs (aime25, niah) — same architecture, same
dispatch code, calibrated+clipped instead of raw. `oscar_int2`'s responses
are **fully coherent, on-topic mathematical reasoning** (`finish_reason:
'stop'`, natural completion) that simply lands on the wrong final answer —
categorically different from `quarot`'s token-level incoherence. This proves
gpt-oss *can* run coherently under int2 KV quantization on this codebase (the
mixed pool does it), which reframes the question from "does this architecture
support int2 KV compression at all" to "what specifically differs between the
mixed pool and the plain pool."

## 4. The shadow-dequant diagnostic

Built a new, purpose-built diagnostic (`SGLANG_INT2_SHADOW_DEQUANT=1`,
`vq_codebook.py`/`memory_pool.py`/`triton_backend.py`) that exercises the
*exact same* plain-pool code — indices, dispatch, sink handling, SWA/full
split — while bypassing only the int2 pack/unpack step:

- `set_kv_buffer` writes raw K/V into a parallel dense bf16 shadow buffer
  instead of quantizing, and returns early (no real quantize call at all).
- `forward_decode`'s quantized-KV branch, when the shadow flag is set, reads
  that shadow buffer through the *standard* (non-quantized) attention kernel
  (`decode_attention_fwd`) instead of dequantizing.
- Scope: decode-path only (no chunked-prefill `dequantize_prefix_kv`
  support) — valid for single-chunk prompts (< `chunked_prefill_size=4096`
  tokens), verified for the test prompts used (max 391 tokens).

**A bug was found and fixed in the diagnostic itself before trusting any
result.** `forward_extend` (prefill) pre-rotates K/V via
`prepare_quantized_extend_qkv` before calling `set_kv_buffer` with
`already_hadamard_transformed=True`; `forward_decode` calls it with raw,
never-rotated K/V. Without correcting for that, the shadow buffer held
*rotated* values for prompt tokens and *unrotated* values for generated
tokens, while the decode-side read always used unrotated Q — a real
inconsistency that would have corrupted the test. Fixed by applying the
self-inverse Hadamard transform to un-rotate whenever
`already_hadamard_transformed=True`, so every shadow entry is consistently
raw.

### Results

**Toy test** (3 short completions, greedy): `"The capital of France is"` →
`" Paris..."`, `"The sky is"` → `" blue..."` — correct factual answers,
grammatically coherent English. Categorically different from the garbage
above.

**Rigorous test** (50 real `math500` problems, real `math_verify` scorer,
greedy decoding, `bf16` run in parallel on the same 50 problems for a direct
comparison):

| | accuracy | cap_hit_rate | mean completion tokens |
|---|---|---|---|
| oracle (shadow-dequant) | **0.94** (47/50) | 0.02 | 1937 |
| bf16 | **0.92** (46/50) | 0.04 | 2480 |

Statistically indistinguishable from bf16, and critically, `cap_hit_rate` is
low for both — natural completion, not truncation/repetition-loop garbage.
**Bypassing only the int2 pack/unpack step, with everything else about the
plain pool held identical, fully restores bf16-quality output on a real,
scored benchmark.** This is the strongest single piece of evidence: whatever
degrades the output lives specifically in the quantize/dequantize math, not
anywhere else in the plain-pool machinery.

## 5. Kernel vs. the original upstream OSCAR implementation

`vendor/OSCAR` (the unmodified authors' repository, distinct from our
`OSCAR-vq` fork) is checked out locally. Diffed
`sglang/srt/mem_cache/kv_quant_kernels.py` between the two:

```
val_range = tl.maximum(val_max - val_min, 1e-8)
scale = val_range / 3.0
zero = -val_min / scale
q0 = (vals0 / scale + zero + 0.5).to(tl.uint8)   # ...q1, q2, q3
```

**Byte-for-byte identical** in both repos. The only diff in `OSCAR-vq` is an
added optional `LLOYD_MAX` branch (the unrelated TurboQuant baseline) wrapped
around this exact, unchanged original logic — `quarot` runs with
`lloyd_max=False` and hits the identical `else:` branch shown above. **This
core quantize logic was never modified during the gpt-oss integration work;
it is the original authors' implementation, unedited.**

## 6. Kernel vs. the documented formula, numerically

Standalone script (no live server) calling the real Triton functions
(`quantized_set_kv_int2_triton`, `dequantize_kv_int2_triton`) directly on
synthetic K/V data — including planted outliers per (token, head), matching
QuaRot's worst-case no-clip scenario — and comparing element-wise against an
independent hand-written reference of the documented formula (per-head
min-max, `scale = range/3`, `zero = -min/scale`, round-half-up, `dequant =
(q - zero) * scale`), using the kernel's own stored scale/zero (isolating
purely the quantize/pack step, strict float32 throughout to avoid a
precision-mismatch confound):

| | value |
|---|---|
| elements tested | 1,024,000 |
| significant divergences (>0.4 quant levels) | 50 (0.0049%) |
| max abs diff | 1.10 |
| mean abs diff | 0.00004 |

Every one of the 50 divergences occurs where the pre-floor quantization index
computes to **exactly** an integer (2.000000, 3.000000, 1.000000, ...) — the
signature of ordinary floating-point tie-breaking between two independently
computed implementations (Triton's parallel reduction vs. sequential
PyTorch), not a systematic or structural error. **The kernel correctly
implements the formula it claims to implement.**

## 7. Conclusion

No bug found at any layer tested:

| layer | check | result |
|---|---|---|
| ABSORB_V_ROTATION | code trace | no-op on every arm, not a gap |
| SWA/full split | code trace | generic, dtype-driven |
| quant_group_size | live config | correctly `null` (per-head) |
| Hadamard rotation | code trace + Naive-baseline test | generic; ruled out (no-rotation still garbage) |
| sinks | dispatch trace | consistently threaded everywhere |
| pool/dispatch/indices | shadow-dequant, 50 real problems | matches bf16 (0.94 vs 0.92) |
| quantize/dequantize kernel | diff vs. upstream OSCAR | byte-identical, unmodified |
| quantize/dequantize kernel | numerical vs. documented formula | 99.995% exact, rest is float tie noise |

One real, separate bug *was* found and fixed: `hadamard_order=128` (the
Qwen/Llama value) doesn't divide gpt-oss's `head_dim=64`, causing an
immediate boot crash. That fix (`hadamard_order=64`) is necessary but not
sufficient — it clears the crash, not the garbage.

The garbage output is a genuine, inherent property of the method as
literally defined: unclipped (`clip_ratio=1.0`) 2-bit min-max quantization,
applied per-(token, head) over gpt-oss's K/V distributions. A single outlier
in even one of 64 dimensions dominates the min-max range and crushes the
remaining ~63 dimensions into effectively one usable quantization level —
catastrophic, but *expected* behavior for naive min-max int2 without
clipping. OSCAR's own calibrated clip ratios (K=0.96, V=0.92) exist
specifically to prevent this failure mode, which is exactly why `oscar_int2`
degrades gracefully instead of collapsing. No config change was made to
"fix" QuaRot's numbers — the point of this investigation was solely to
confirm the *implementation* is correct, not to make the *method* perform
better.

## Artifacts

- `pipelines/runpod/protocols/gptoss20b_mxfp4_v1{,_ctx131k}.json` — `quarot`
  arm definition (`hadamard_order=64`, `k/v_clip_ratio=1.0`).
- `pipelines/runpod/run_cell.sh` — `hadamard_order`/`k_clip_ratio`/
  `v_clip_ratio` arm-level plumbing.
- `vendor/OSCAR-vq/sglang-research/python/sglang/srt/environ.py` —
  `SGLANG_INT2_SHADOW_DEQUANT` declaration.
- `vendor/OSCAR-vq/sglang-research/python/sglang/srt/mem_cache/memory_pool.py`
  — shadow-write bypass in `set_kv_buffer` (with the prefill/decode
  rotation-consistency fix).
- `vendor/OSCAR-vq/sglang-research/python/sglang/srt/layers/attention/triton_backend.py`
  — shadow-read dispatch in `forward_decode`.
- `/tmp/.../scratchpad/verify_int2_kernel.py` (session scratch, not
  committed) — the kernel-vs-formula numerical audit script (§6).
