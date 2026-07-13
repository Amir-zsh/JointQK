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
| 8k | 0.185 | 0.270 | 0.885 | 0.114 | 0.040 | 4.80 | 0.62 | 0.22 |
| 32k | 0.301 | 0.683 | 0.973 | 0.394 | 0.078 | 3.24 | 1.31 | 0.26 |
| 80k | **0.737** | 1.107 | 2.512 | 0.973 | 0.173 | **3.41** | **1.32** | 0.23 |
| 128k | 1.144 | 1.805 | 3.625 | 1.530 | 0.295 | 3.17 | 1.34 | 0.26 |

**Pre-registered bar (≥2.5× vs dense fp16 @80k bs=1): PASS at 3.41×.**
From 32k up the quantized-resident path also beats flash SDPA fp16 by
~1.3×. (An earlier constexpr-width variant measured 0.62 ms @80k / 4.25×,
but was only correct for SHARED profiles — Llama's prof_share=layer regime;
the shipped kernel loads per-(rung, head) tables, the K4 bug-fix below, at
a ~15% latency cost. Both artifacts are in bench_pgq_decode.json history.)

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

## V-INT2 tier (built, parity-tested — and a measured negative)

Phase B gained an INT2-V variant (quarter-interleaved codes, per-token
scale/zero folded algebraically: sum_t p_t(s_t c_t + z_t) = (p*s)@c +
(sum p_t z_t)*1 — V never materialized; parity 12/12 incl. cross-check vs
the fp16 tier on identical v_hat). Benchmarked: **SLOWER than fp16-V at
the current operating point** — 0.882 vs 0.622 ms @80k, 1.370 vs 0.964
@128k — despite reading ~26 MB less. Diagnosis: phase B is not
bandwidth-bound (the whole step runs ~40 GB/s effective), so removing V
bytes buys nothing while the in-register unpack + 4 narrow dots add
compute. This REVISES the OSCAR-gap attribution: the remaining ~3x is
kernel compute structure — leading term: GQA group 4 padded to 16-row
tensor-core tiles wastes 4x of every dot (OSCAR's grouped kernel packs
real query heads) — not format bytes. V-INT2 stays in the tree for when
the compute side is tuned to bandwidth.

## K4 — end-to-end quantized-resident decode on Qwen3-8B: DONE

`pipelines/kernels/k4_decode_demo.py` (plan7 R3 standalone-loop form):
after a normal HF prefill, the prompt's K cache (layers >= 1) is
compressed with the SHIPPING codec (pgq_dctlmrw_rdo@2.0, bundle 2cf29a8a),
packed into the kernel layout, and the fp16 prefix K is FREED; decode
attention runs the fused kernel on packed bytes merged with an fp16 ring
(layer 0, sink page, partial tail, generated tokens) via softmax partials.
Wired through transformers 5.x's attention registry + a ring Cache
subclass — the packed prefix never materializes. 6011-token prompt,
48 greedy tokens (artifacts/kernels/k4_decode_demo.json):

- **G1 (plumbing): 1.000** — the custom loop reproduces HF generate
  token-for-token with an uncompressed ring.
- **Self-check: 1.5e-4** — kernel vs exact torch on the served rows, real
  Qwen weights/keys/queries.
- **G2 (the kernel IS the format): 1.000** — the quantized-resident arm is
  token-identical to the Mode-A control (exact attention over dequantized
  K-hat), i.e. the committed F1 numbers price this path exactly.
- Memory: prefix K 443 MB -> 67 MB packed+ring (**6.6x**); peak 20 GB.
- Quantized vs full-fp16 greedy agreement at 6k/48: 0.458 (1.000 at 1.4k)
  — pure quantization effect (G2 pins the kernel blameless); greedy chains
  amplify small logit shifts, while the row-paired F1 evidence (report8:
  +0.52 tie at 2.0, 48.80 vs 48.29) is the calibrated quality statement.
- tokens/s: fp16 loop 25.0 | dequant control 15.7 | kernel-resident 14.7 —
  the demo loop is python-glue-bound (35 layers x per-step partial merges,
  no CUDA graphs in the loop); kernel-level latency is the K3 story.
- Page-seal (encode+pack, torch, one-off): 1492 s for 6k x 35 layers —
  GPU pack kernels remain the recorded K5 item.

A real bug this stage caught: Qwen's per-HEAD profiles (prof_share=head)
were mis-served by the segmented layout (head 0's width tables used for
all heads — G2 was 0.02). Every synthetic test used shared profiles; the
fix (per-(rung, head) stride/width/LUT tables as program-uniform scalars)
kept coalescing and all 12 kernel tests green. Lesson recorded: the
kernel test matrix must span BOTH prof_share regimes.

## Diagnosis revision (2026-07-13, byte-ledger accounting)

Computing effective bandwidth per arm from the bench artifact corrects the
earlier attribution. At 128k: pgq 298 MB / 1.14 ms = 261 GB/s vs OSCAR
84 MB / 0.30 ms = 285 GB/s — the kernels run at ESSENTIALLY EQUAL memory
efficiency; OSCAR's ~3.9x latency win is ~3.5x BYTES (our fp16 V = 268 of
298 MB) x ~1.1x efficiency. The padded-GQA compute waste flagged earlier is
real but small (~30 us of tensor work in a 1.14 ms step). The V-INT2 arm is
the decisive datum: fewest bytes of all (72 MB) yet 49 GB/s — 5x below our
own fp16-V efficiency — i.e. the V-INT2 CONCEPT is right and the phase-B
implementation (4 narrow dots + dual fp32 sigma loads/row) is the problem;
at nominal efficiency that arm computes to ~0.28 ms, FASTER than OSCAR.
Both kernels sit at ~20% of A100 peak — no one is at a physical wall.
Conclusion: no fundamental limit; matching/beating OSCAR = (1) their V
pattern done properly, (2) halve/quarter sigma traffic (fp16, single-map),
(3) fuse phase A/B + q-head packing, (4) tuning tables.

## Next steps (post-K4)

1. Kernel compute-structure iteration if the OSCAR gap matters: pack real
   query heads across KV heads into full tiles (removes the 4x pad waste),
   fuse phase A/B via persistent programs, then re-evaluate V-INT2 (it
   pays only near the bandwidth limit).
2. GPU page-seal kernels (encode is 25 min in torch for 6k x 35 layers).
3. CUDA-graph the demo loop / SGLang integration (K5, user go/no-go).
