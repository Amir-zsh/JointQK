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
K AND V INT2 + scales; artifacts/kernels/bench_pgq_decode.json)

| context | pgq v1 | dense fp16 | SDPA flash | OSCAR INT2 | ×dense | ×OSCAR |
|---|---|---|---|---|---|---|
| 8k | 0.290 | 0.904 | 0.119 | 0.141 | 3.12 | 0.49 |
| 32k | 0.683 | 0.968 | 0.395 | 0.081 | 1.42 | 0.12 |
| 80k | 1.109 | 2.255 | 0.972 | 0.173 | **2.03** | 0.16 |
| 128k | 1.803 | 3.360 | 1.516 | 0.263 | 1.86 | 0.15 |

**Pre-registered bar (≥2.5× vs dense fp16 @80k bs=1): MISS (2.03×;
passes at 8k with 3.12×).** OSCAR's kernel is 6-8× faster than v1 at long
context.

## Diagnosis (why, and why it's fixable)

At 80k the pgq arm reads ~24 MB in 1.11 ms — ~22 GB/s effective on a
~1.5 TB/s part: the kernel is **gather-bound, not bandwidth-bound**. The
per-row payload pointers (rows of one page have different rungs, hence
different strides) turn every (64-row × 32-coord) unpack tile into
scattered uint8 gathers, while OSCAR's crumb layout reads 32 contiguous
bytes per token. The one in-plan optimization iteration (finer split
parallelism, warps sweep) made things worse — parallelism is not the
lever; memory layout is. The designed fix is exactly plan7's R2
mitigation, not yet implemented: **rung-segmented payload within each
page** (encode-side permutation already exists in pack_sequence's segment
lists) so each segment has constexpr width and stride → contiguous
vectorized loads. Byte arithmetic says the format then has OSCAR-class
traffic on K (2.13 b/c ≈ 34 B/token vs their 36) with ~10× less metadata;
V-side parity additionally needs a V-INT2 tier (their V path, unchanged).

## Honest reading

The quality story (pgq8: sub-wall proxy, incumbent F1 at 12% fewer bits,
+6 F1 over TurboQuant on Qwen) is fully banked; the serving story is NOT
yet: correctness of the fused path is proven end-to-end, the format's
bandwidth advantage is real on paper, but the current kernel does not yet
realize it. Beating OSCAR's kernels requires the coalesced repack (est.
the dominant fix), tensor-core-friendly LUT dequant, and a V-INT2 tier —
concrete, scoped engineering, no science risk.

## Next steps (K2c'/K3'/K4)

1. Rung-segmented coalesced payload repack + per-segment constexpr-width
   stage-1 (the OSCAR per-tier launch pattern, one tier per rung).
2. Re-run K3; then add the ours+V-INT2 arm (their V path) for the
   apples-to-apples long-context number.
3. K4 end-to-end quantized-resident Qwen demo (F1 parity row-paired,
   tokens/s, peak memory) once K3' clears the bar.
