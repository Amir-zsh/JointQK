# plan7 (pre-registration): pgq7 — efficient implementation (OSCAR-class Triton decode path)

**Registered:** 2026-07-11, branch `pgq7` (cut from `pgq6` @ 743c59f).
**Status: REGISTERED.** User directive: build an efficient implementation
"similar to OSCAR" before the long-generation study; kernel tests target
**Qwen3-8B** (user directive 2026-07-11). Engineering study — the format is
frozen (pgq4 F-4 constants + pgq6 mrg default which disengages at 2.0 b/c);
no new allocation choices, so no selection/F1 freeze machinery beyond the
pre-registered K3 speed bar and K4 parity check.

## 0. Thesis

pgq4's fold identity was designed for this port: with k̂ = (s⊙ĉ)G + μ the
decode score is q·k̂ = ((qGᵀ)⊙s)·ĉ + q·μ, and q·μ cancels in softmax — basis
AND calibration-static scales fold into the query once per (layer, kv-group,
step); the kernel reads only integer codes + a 3-bit rung id per token
(vs OSCAR's per-group (scale, zero) = 32 bit/token metadata loaded inside the
KV loop). At 2.13 b/c the K read is ~34 B/token vs 256 B/token BF16 (~7.5×);
OSCAR reports ~3× bs=1 decode speedup from the same bandwidth argument. Ours
reads strictly less per token than OSCAR INT2 at equal-or-better measured
quality (pgq4/pgq5/pgq6 evidence). Deliverable: quantized-RESIDENT decoding
with measured latency/memory, not just proxy quality.

## 1. Architecture to mirror (vendor/OSCAR/sglang-research, explored)

Pure Triton, FlashDecoding split-K: per-tier stage-1 kernels write disjoint
split ranges of one partial buffer (`_fwd_grouped_kernel_stage1_quant_int2`,
decode_attention.py:1363), one tier-agnostic stage-2 online-softmax reduce
(`_fwd_kernel_stage2_unified`, :2265). Two-arena slot space (INT2 paged +
BF16 HP sink/recent, dispatch `slot >= HP_OFFSET`, unified_kv_pool.py:122)
maps 1:1 onto our sink escape + recency window + Mode-B' fp16 ring. Their
tuning table targets exactly our K4 geometry (Qwen3-8B 32Q/8KV d=128).
Microbench harness to mirror: benchmark/kernels/quantization/
bench_decode_int_kv.py (correctness-check, then time).

## 2. Scope (v1)

- **Model:** Qwen3-8B, bundle cd6a3d41 (36L, 32Q/8KV, d=128, GQA 4, layers
  ≥1; prof_share=head → per-(head, rung) width table, 8×8×4 entries/layer in
  smem; rung segments are (head, rung)-homogeneous so token stride is still
  fixed per stage-1 segment launch). Llama (per-layer profiles, constexpr
  widths) = simpler deferred variant.
- **Codec:** `pgq_proflmrw_rdo@2.0` exactly (the shipping config; mrg
  disengages at 2.0). Merge rows (per-page counts + log-c bias into the
  online softmax) designed into the layout, implemented as K5 stretch.
- **K-only in the attention math; V stays BF16** in correctness runs (v7
  keeps V = TurboQuant-2b dequantized). Microbench additionally reports
  ours+V-INT2 (OSCAR's V path unchanged) for an honest end-to-end story.
- Encode (pack) is PyTorch v1 (page-seal amortized); GPU flush = stretch.
- References for K4 parity: committed Qwen Stage-B cells (proflmrw@2.0
  F1 48.29, FP 50.70, TQ 42.13) + Qwen heldout gates.

## 3. Byte layout (frozen by K1)

Per (layer, head), rung-segmented pages:
- Header: 3-bit rung id/token (original order), 2-bit M-class reserved,
  per-rung contiguous token segments recorded as encode-side permutation
  (tokens within a page stay contiguous per rung).
- Token payload = 4 × 32-coord block segments; block at width w packs 32
  codes little-endian into 4w bytes (w ∈ {2,3,4,6} → 8/12/16/24 B,
  byte-aligned by construction). Stride constexpr GIVEN the rung → ≤8
  index-list segment launches per (l, h), like OSCAR's tier split but finer.
- Codes are LUT indices; dequant = LUT_w[code] (unit level) × code_std_j
  folded into q̃. Uniform top width (6) is LUT-representable too (levels
  (code−lim)·α_w) — one code path. LUTs {4, 8, 16} entries in registers,
  width-6 (64) in smem.
- Sink rows (positions 0-3, page 0): 8-bit absolute segment; second folded
  query q̃_sink = (qGᵀ)⊙sink_scale (4 rows only).
- fp16 segments (recency pages + Mode-B' ring + layer-0) go through the BF16
  stage-1 exactly like OSCAR's HP tier.

## 4. Stages

- **K0** — branch `pgq7`, plan7.md (this file). Done at registration.
- **K1 — packed format + golden vectors** (CPU). `kvq/kernels/pgq_pack.py`:
  pack/unpack in PyTorch mirroring §3; reference = `FoldedScalarPagedCompressor
  .roundtrip` with an ADDITIVE `emit` kwarg exposing (assign, per-coord codes,
  sink codes) — no behavior change, pgq4/pgq6 suites pin it. Packed→unpacked
  reconstruction bit-identical to the compressor's r̂ (same LUT snap), on
  synthetic + the 4 real Qwen selection rows (pgq5_qwen_compact8_16 roles,
  bundle cd6a3d41). `tests/test_pgq_pack.py`.
- **K2 — Triton decode kernel** (the core). `kvq/kernels/pgq_decode_attn.py`:
  rung-segment stage-1 (grid = batch × kv-group × split; unpack per 32-block
  at constexpr width; dot with pre-folded q̃ slice; online-softmax partials),
  sink + BF16 segment launches, single unified stage-2. Query folding = one
  (32,128)@(128,128) GEMM per (l, group) per step in PyTorch. Gates:
  golden-vector logit parity; full attention-output parity ≤1e-2 rel
  (fp16 accumulation); determinism across runs.
- **K3 — microbench** (GPU-cheap). `pipelines/kernels/bench_pgq_decode.py`
  mirroring OSCAR's harness: arms = BF16-Triton baseline, OSCAR INT2 (their
  kernel, fabricated buffers), ours (@2.0 rung mix from real Qwen cell
  telemetry), ours+V-INT2. Sweep context {8k, 32k, 80k, 128k} × batch
  {1, 8, 32}, Qwen geometry. Report ms/step, GB/step, speedup.
  **Pre-registered bar: ≥2.5× stage-1 latency vs BF16 Triton at 80k bs=1,
  K-only.** Miss → profile, ONE optimization iteration, report honestly.
- **K4 — end-to-end demo.** Keep PACKED K resident after prefill (not
  dequantized fp16); route decode attention layers ≥1 through the kernel
  (HF eager attention override, `torch.compiler.disable` like OSCAR's
  backend). Qwen3-8B, lcc + hotpotqa (50 rows): F1 parity vs committed
  Stage-B Mode-A cells (row-paired, CI ∋ 0 expected — identical math up to
  fp16 accumulation); decode tokens/s + peak memory vs (a) fp16-resident
  emulation, (b) full-fp16 no-press, at native lengths + one synthetic
  64k stress row.
- **K5 — stretch (explicit go/no-go with user):** mrg rows (bias add +
  per-page row counts), GPU pack + Mode-B' flush kernels (prereq for the
  efficient long-gen study), SGLang fork integration (unified_kv_pool.py +
  triton_backend.py as templates).

## 5. Verification

K1: bit-identity vs compressor roundtrip (synthetic + real rows); pgq4/pgq6
suites stay green (emit kwarg additive). K2: golden-vector + output parity +
determinism. K3: OSCAR-style correctness check before every timing row;
JSON artifact + report7.md table. K4: row-paired parity bootstrap (fixed
tool); memory via torch.cuda.max_memory_allocated + nvidia-smi cross-check.
Fingerprint 262/262 before any commit; per-commit approval as always.

## 6. Budget & risks

GPU ≈ 3-6 GPU-h (engineering-dominated). Toolchain verified: triton 3.6.0,
A100-SXM4-40GB, GPUs 0-6 open (yield to live user jobs). Risks: (R1)
register pressure with width-6 smem LUT + 4-width unpack in one kernel →
split stage-1 launches per rung (constexpr width per launch; OSCAR already
launches per-tier); (R2) rung-segmented gather costs random access vs
contiguous pages → segments built per PAGE (encode-side permutation keeps
same-rung tokens contiguous within a page); (R3) HF-eager integration
friction in K4 → fallback: K3 kernel-level results + standalone generate()
demo instead of full bench-stack integration.
