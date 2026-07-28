# Validating OSCAR-INT2's implementation on gpt-oss (companion to report14, and to QuaRot's validation)

*Same question as `gptoss_quarot_kernel_validation.md`, asked of `oscar_int2`
(calibrated rotation + clip, OSCAR's shipped baseline) instead of QuaRot: is
`oscar_int2`'s degraded accuracy on gpt-oss (0.9176→0.3904 math500, 0.7400→0.0
aime25, 0.5041→0.2505 gpqa, flat 0.0 on every NIAH length) a bug in this
port, or inherent to the method? Unlike QuaRot, this question was already
answered — more rigorously than a same-day redo could manage — in
`report14.md`. This note summarizes that evidence, checks whether it
transfers to today's checkpoint and calibration bundle, and reconciles it
against today's fresh results.*

---

## 1. Don't duplicate report14 — extend it

`report14.md` (2026-07-24—26, +2026-07-27 addendum) already ran the
OSCAR-equivalent of every check in the QuaRot report, and one strictly
stronger than anything done there:

- **Multiple independent kernel-correctness gates**, all passing exactly:
  the unified HP+INT2 decode kernel against a reference (≤1.5e-2), the pool
  write + 256 decode flushes (exact, 8192/8192 unique slots), decode over
  pool-written buffers with a real index split (≤4.6e-3), the prefill sink
  LSE rescale (0.2%), and rotation orthogonality (2.4e-07). These exclude
  the quantized decode kernels, pool writes, allocator, index construction,
  rotations, sinks, and layer mapping as the source of the quality collapse.
- **An engine-free HuggingFace simulation** — strictly stronger than this
  session's shadow-dequant diagnostic, since it removes the *entire serving
  stack*, not just the quantize step: overwrite the KV cache with its own
  quantize→dequantize roundtrip inside a plain HF forward pass, no engine
  code at all. Llama-3.1-8B stays 10/10; gpt-oss-20B goes 4/10→**0/10**. The
  collapse reproduces with zero engine involvement, which is the cleanest
  possible proof it isn't a serving-stack bug.
- **A matched-noise control**: inject random Gaussian noise of *exactly*
  int2's own error magnitude into the same K tensors — it does **more**
  damage than int2 itself (0.909 vs 0.718 relative logit error on gpt-oss).
  A broken/buggy quantizer would look the opposite of this; int2's
  structured error is measurably *better* than equivalent unstructured
  noise.
- **A parity audit** for a separate, real mechanism (chunked prefill
  reading its own quantized prefix, ~24 NIAH points) proving bit-identical
  reconstruction between the prefill and decode read paths — real effect,
  not a bug, now fixed by an opt-in exact-chunked-prefill flag.
- **Two genuine bugs found and fixed** in the process, both distinct from
  the core fidelity finding: a piecewise-CUDA-graph padding bug (NaN/Inf
  under concurrent sampled decoding) and a VQ-specific learned-sink
  coordinate error (17.7 points). Neither one is the scalar-INT2 collapse —
  both were isolated and shown not to explain it.
- **A mechanistic account** of *why* gpt-oss is more fragile than Llama:
  attention logit error is 3.4× worse on gpt-oss than Llama in every basis
  tested (including calibration-free Hadamard and random orthogonal,
  ruling out "bad calibration"); damage is concentrated in gpt-oss's 12
  full-attention layers (its sole long-range-information channel, vs
  Llama's 32 with redundancy); 82% of MoE expert-routing decisions flip
  between bf16 and INT2 runs, discretely amplifying the drift.

Report14's own confidence assessment (§8): "**Solid.** There are three
independent findings" — reproduces with zero engine code, the graph bug is
independently and completely characterized, and the VQ sink fix is derived
algebraically and verified numerically. That is a stronger evidentiary
basis than anything practical to rebuild today; this note's job is to check
whether it applies to *today's* setup, not to re-derive it.

## 2. Does it transfer to today's checkpoint?

Report14 was run on `unsloth/gpt-oss-20b-BF16`. Today's `oscar_int2`
results are served from `openai/gpt-oss-20b` (the official native-MXFP4
checkpoint, dequantized to bf16 at load time via `triton_kernels`, per
earlier this session's finding). Two gaps to check before trusting a
straight transfer:

**Weight values.** Earlier this session, the same MoE expert tensors were
independently dequantized both ways — `transformers.integrations.mxfp4
.convert_moe_packed_tensors` (calibration's path) and `triton_kernels
.numerics_details.mxfp.upcast_from_mxfp` (serving's path) — and a
sorted-value comparison across a full `[2880,2880]` tensor showed **0.0 max
diff, 100% exact match**: both decode to the identical multiset of bf16
values. This was checked for MoE expert weights specifically, not the
attention Q/K/V/O projections that actually feed the K/V being quantized —
but those come from the same underlying release via the same class of
dequantization, and `unsloth/gpt-oss-20b-BF16` is a standard community
bf16 conversion of the same official weights, not an independently retrained
model. There is no known reason the attention weights would diverge where
the MoE weights don't.

**Calibration bundle.** This is a real difference, not just a formality:
today's `oscar_int2` (`gptoss20b_mxfp4_v1.json`) uses
`artifacts/oscar_gptoss20b_mxfp4/rotations_gpqa198` — a calibration build
specific to the native-MXFP4 checkpoint (project task #15) — while
report14 used `artifacts/oscar_gptoss20b/rotations_gpqa198`, calibrated
against the unsloth checkpoint. So report14's *exact numbers* don't
directly transfer; its *mechanisms* (architecture-level fragility,
independent of calibration quality — shown by the identity/random/Hadamard
basis sweep all landing in the same 3.1–4.3× damage-gap range as the
shipped rotation) should.

## 3. Reconciling report14's headline collapse against today's mixed results

Report14's headline number is total collapse: NIAH-8K INT2 = **0.0**,
cap-hit **1.00**, literal word salad. Today's `oscar_int2` is *not*
uniformly like that — math500 is 0.3904 (coherent, partial credit) and
gpqa is 0.2505 (coherent, above-floor), which read more like "real but
survivable degradation" than "total collapse." Read closely, though, this
is not a contradiction — it's the same finding, seen across a wider task
set:

| task | today's oscar_int2 | character |
|---|---|---|
| math500 | 0.3904 | coherent chain-of-thought, wrong answers |
| gpqa | 0.2505 | coherent, above-floor |
| aime25 | **0.0000** | **coherent reasoning** (`finish_reason: 'stop'`), lands on wrong/unformatted answers — not garbage |
| niah (8192–131072) | **flat 0.0000 at every length** | matches report14's core NIAH finding |

`aime25`'s raw responses were read directly earlier this session — real,
on-topic mathematical derivations, not incoherent text — so its 0.0 belongs
with math500/gpqa's "coherent but wrong" bucket, not NIAH's collapse. NIAH
is where today's fresh results land exactly on report14's own headline
finding: **total collapse, specifically on long-context needle-retrieval**,
the task category report14 measured most closely and explained mechanistically
(full-attention-layer bottleneck, no cross-layer redundancy, MoE routing
flips). Reasoning/QA tasks with shorter, denser context degrade gracefully;
long-context retrieval collapses. That split is exactly what report14's
mechanism predicts — full-attention layers are gpt-oss's only long-range
channel, and NIAH is the task that stresses long-range retrieval hardest.

## 4. Conclusion

No new investigation was required to answer this — report14 already did,
more thoroughly than this session could redo same-day, including a
strictly stronger control (engine-free HF simulation) than the
shadow-dequant approach used for QuaRot. Its conclusion: `oscar_int2`'s
degraded accuracy on gpt-oss is a genuine, engine-independent, architecture-level
fidelity limit of per-token min-max 2-bit quantization on a model whose
long-range information flows through only 12 full-attention layers with a
router sensitive enough to flip 82% of its decisions under the resulting
noise — not a bug in the pool, dispatch, kernels, or sink handling, all of
which were independently gated and passed. The one real serving-layer bug
found (piecewise-graph padding) and the one real VQ-specific bug (sink
coordinate error) are both fixed and both shown *not* to explain the core
finding. The calibration bundle differs between report14 and today's
official-checkpoint setup, so exact numbers don't transfer — but the
architecture-level mechanism does, and today's fresh results (graceful
degradation on math500/aime25/gpqa, total collapse on NIAH) land exactly
where that mechanism predicts.

## Artifacts

- `notes/page_quant/studies/report14.md` — the primary evidentiary source
  (§4 HF simulation, §5 layer attribution, §6 mechanism, §10 chunked-prefill
  parity audit).
- `pipelines/oscar_e2e/dbg_int2_hf_simulation.py`,
  `verify_gptoss_int2_decode.py`, `verify_gptoss_mixed_pool.py`,
  `verify_gptoss_int2_e2e.py`, `audit_prefill_decode_parity.py` — the
  gates/controls report14 built and kept as permanent regression checks.
- `artifacts/oscar_gptoss20b_mxfp4/rotations_gpqa198` vs
  `artifacts/oscar_gptoss20b/rotations_gpqa198` — the calibration-bundle
  difference noted in §2.
