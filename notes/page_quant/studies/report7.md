# report7 — pgq7: OSCAR-class decode path (K1-K3 status, in progress)

**Study:** plan7.md (registered 2026-07-11) + plan8 §4 row-semantics update
(the format to serve is pgq8's DCT coefficient rows). Engineering study;
correctness gates + one pre-registered speed bar.

## Status by stage

- **K1 — packed format + golden vectors: DONE.** `kvq/kernels/pgq_pack.py`:
  little-endian width-block packing, per-page-contiguous segment lists,
  sink segment; bit-identical pack→unpack→dequant vs both compressors
  (pgq4 token rows AND pgq8 coefficient rows via the emitted per-row sigma
  maps), pinned on synthetic + real Qwen selection rows.
- **K2 — fused Triton decode kernel: correctness DONE, perf shaping OPEN.**
  Stage-1 v0 (per-page programs) and v1 (rows-as-lanes tiles, tensor-core
  dots for q·K̂, the page-local inverse DCT, and p·V; grid = splits × KV
  heads; padded 16-row q tile) with split-K online-softmax partials; torch
  stage-2 reduce; sink/partial/fp16 tier folded in as an extra split in
  μ-dropped logit space. Parity vs fp32 full-softmax references: 9 GPU
  tests green (DCT/identity/mixed-rw/real-bundle; split-invariance);
  bench-time parity gate 3.3e-4.
- **K3 — microbench vs BF16 + OSCAR: RUN, bar MISSED, diagnosis clear.**

## K3 results (A100, Qwen geometry: 8 KV heads × group 4, d=128, bs=1,
one layer-step; pgq arm = K packed at the real dctlmrw@2.0 rung mix +
V fp16; OSCAR arm = vendored `_fwd_grouped_kernel_stage1_quant_int2`,
K AND V INT2 + scales; quantized arms AND SDPA under CUDA graphs — the
standard decode-serving configuration; artifacts/kernels/
bench_pgq_decode.json)

| context | pgq v2 | pgq v1 | dense fp16 | SDPA flash | OSCAR INT2 | ×dense | ×SDPA | ×OSCAR |
|---|---|---|---|---|---|---|---|---|
| 8k | 0.160 | 0.270 | 0.894 | 0.114 | 0.042 | 5.59 | 0.72 | 0.26 |
| 32k | 0.261 | 0.683 | 0.971 | 0.392 | 0.078 | 3.72 | 1.50 | 0.30 |
| 80k | **0.622** | 1.107 | 2.645 | 0.990 | 0.217 | **4.25** | **1.59** | 0.35 |
| 128k | 0.966 | 1.805 | 3.946 | 1.545 | 0.260 | 4.08 | 1.60 | 0.27 |

**Pre-registered bar (≥2.5× vs dense fp16 @80k bs=1): PASS at 4.25×.**
From 32k up the quantized-resident path also beats flash SDPA fp16 by
1.5-1.6×.

## The iteration that got there (recorded per plan)

v1 (per-row payload pointers) missed the bar at 2.03×: ~22 GB/s effective
— gather-bound, since rows of one page carry different rungs and hence
different strides, scattering every unpack tile. The parallelism sweep
made it worse (negative result recorded). The fix was plan7's R2 design:
**K2c' two-phase kernel** — phase A per rung with the pack-time
rung-segment lists: dense (rows × constexpr-stride) payload tiles,
constexpr widths and LUT bases (one compiled variant per rung, OSCAR's
per-tier pattern), coalesced loads, tensor-core q-dots, scatter into a
u-buffer; phase B per page: u tile → page DCT → online softmax → V →
split partials. Phase A's 7 small launches made raw v2 latency-bound at
short context; CUDA graph capture (SGLang's own decode configuration;
applied to baselines too) removed it: 0.16-0.97 ms across 8k-128k.

## vs OSCAR, honestly

Their kernel remains ~3× faster. Two known factors: (1) **V**: they read
INT2 V (+scales) ≈ 8 MB at 128k where we read fp16 V ≈ 33.5 MB — V now
dominates our traffic (K payload is 4.3 MB); (2) maturity (tuned block
sizes, single fused stage-1). The format-side answer is the V-INT2 tier
(their V path can be adopted unchanged — V is not part of the pgq K
format); with it our per-step bytes drop to ~9 MB vs their ~12 MB
(K-side: 34 vs 36 B/token, ~10× less metadata). Closing the remaining
kernel-maturity gap is ordinary tuning work, no format risk.

## Next steps

1. V-INT2 tier in phase B (adopt OSCAR's V dequant pattern) → re-run K3
   for the apples-to-apples long-context number.
2. K4 end-to-end quantized-resident Qwen demo (F1 parity row-paired,
   tokens/s, peak memory) — the kernel side is now fast enough to be
   worth wiring in.
3. Kernel tuning iteration (block sizes, fuse phase A/B via persistent
   programs) if the OSCAR gap matters after V-INT2.
