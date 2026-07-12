# probe8 — token-axis transform probe (pgq8 pre-study)

**Date:** 2026-07-11. **Question:** would a page-level transform along the
TOKEN axis (quantize DCT/KLT coefficient rows of the 64×128 page instead of
token rows) buy anything over per-token coding? Decides go/no-go for pgq8
before any implementation. Tool: `pipelines/page_quant/probe_token_axis.py`;
artifacts `artifacts/page_quant2/token_axis_probe_{llama31_8b,qwen3_8b}.json`.

**Data discipline:** fit-pool rows only (bundle selection rows excluded —
selection stays unburned for pgq8 screening). qpca_unc code space from the
frozen bundles (Llama ff390304 40r-pool: 18 raws on disk; Qwen cd6a3d41: 12),
sinks dropped, layer 0 excluded. Llama 534k pages, Qwen 530k pages.

## Verdict: GO — every arm positive, on both models

| metric (mean over cells, layers 1+) | Llama | Qwen |
|---|---|---|
| token-lag correlation, lag 1 | 0.70 | 0.79 |
| lag 16 | 0.48 | 0.63 |
| DCT coding gain G (AM/GM of coeff-row energies) | 2.50 | 5.96 |
| pooled-KLT gain (best fixed transform) | 2.54 | 6.07 |
| Haar gain | 2.33 | 5.47 |
| per-page-KLT oracle (unrealizable bound) | 16.2 | 26.8 |
| high-rate bits saved, DCT vs identity (b/coord) | 0.66 | 1.29 |
| DC row energy fraction | 49% | 64% |
| top-16-of-64 rows energy (identity → DCT) | 25% → 76% | 25% → 83% |
| **matched-rate sim, SE ratio DCT/identity @1.0** | **0.583** | **0.552** |
| @1.5 | **0.634** | **0.595** |
| @2.0 | 0.649 | 0.614 |
| sim KLT/identity @1.5 | 0.616 | 0.568 |

Sim = identical allocator (per-(row, 32-block) λ-bisection over widths
{0,2,3,4,6}) and identical grids (bundle LM cents + covering uniform top,
per-(row,coord) calibrated scales) on 24 sampled pages/cell; only the row
basis differs. It is the direct predictor of a pgq8 screening outcome.

## Findings in detail

1. **The correlation is large and long-range.** Lag-1 code-space correlation
   0.70/0.79; decay to lag 16 is shallow (0.48/0.63). Slow decay is exactly
   the regime where a graded transform beats hard 2:1 merging — pgq6's merge
   could only harvest adjacent near-duplicates; most of this structure lives
   in smooth long-range components (DC row alone: 49–64% of energy).
2. **DCT ≈ KLT: no learned transform needed.** The pooled-KLT (best fixed
   transform, would need calibration + sideband-free) beats DCT by only
   1.5–1.9% in gain; the token-axis covariance is Toeplitz to within
   cv ≈ 0.015. A fixed DCT-II, hardcoded, zero calibration, captures
   ~98.5% of the fixed-transform opportunity. Haar (cheaper in-kernel) gives
   up ~7–8% of the gain — a legitimate fallback if the kernel wants it.
3. **Predicted screening outcome (Llama, from sim ratios × measured pgq6
   curve):** proflm/mrg logit_err 0.00483 @1.5 × 0.634 ≈ **0.0031 ≈ the ecu
   closure bar 0.00318** — the first mechanism in the arc that plausibly
   REACHES the entropy-coding wall, consistent with theory: the wall is a
   per-token-independent-coding bound, and token mixing is outside that
   class. At 1.0: 0.01233 × 0.583 ≈ 0.0072 vs ecu 0.00646 (close, above).
   Treat as directional — the sim allocator is unconstrained (no ≤8-rung
   menu, no monotone profiles) on both arms.
4. **Uniform across content, layers, cells.** Per-task DCT gain 2.1–2.7 on
   Llama (max: passage_retrieval; min: repobench/multi_news) — same
   content-independence as pgq6 merging. Every layer ≥ 2.0; early layers
   higher. Cell p10 = 1.85 — no cell loses. Qwen more heterogeneous
   (qmsum 8.7, per-layer max 29.7) and stronger everywhere.
5. **RoPE scrambles surprisingly little.** Raw-key-space check (4 rows,
   analytic de-rotation): post-RoPE DCT gain 3.80/3.96 vs pre-RoPE 4.41/4.32
   — ~85–90% of the token-axis structure survives RoPE. A pre-RoPE variant
   (quantize before rotation, RoPE at decode) is a deferred upside, not a
   prerequisite.
6. **Per-page oracle gap.** The per-page-KLT bound (G 16–27) is far above
   any fixed transform — there is large page-local structure a fixed T
   cannot reach (it needs per-page sideband). The per-page width RDO
   recovers some of this adaptively; the rest is recorded as a deferred
   direction (e.g. tiny per-page transform dictionaries).

## Caveats (recorded before any screening)

- Sim SE ratios at 2.0 (0.61–0.65) will likely NOT translate to F1 — pgq4/6
  showed F1 saturates near 2 b/c (ecu's 1.8× lower proxy at 2.0 was an F1
  tie). The F1-relevant claim is the LOW-rate regime: OSCAR-class quality at
  ~1.0–1.5 b/c.
- Total code-space SE is preserved by orthogonality, but the transform
  redistributes error across tokens within a page; per-token protections
  (sink, window) must stay OUTSIDE transformed pages (they already are,
  positionally), and real logit-err screening on selection rows remains the
  gate.
- Partial pages (< 64 tokens) stay identity-coded; Mode-B' ring pages too.
- Decode cost: fold identity survives — page logits z = Dᵀ(Ŷ q̃ᵀ), one
  64-point inverse transform per (page, query head) per step on top of the
  existing dots; rung-0 coefficient rows skip their dot exactly like merged
  rows skipped theirs.

## Proposed pgq8 (pending user approval)

P0: `PageDCTCompressor` — fixed DCT-II over tokens per interior page,
per-coefficient-row rung RDO (existing machinery, rows ↔ coefficients),
per-(row,coord) scale map from the fit pool (stats refit only, no basis
change), sink/rw/partial pages identity. P1: selection-row screening at
{1.0, 1.25, 1.5, 2.0} vs the pgq3 closure bars + proflm + mrg controls;
pre-registered bars: beat ecu 0.00318 @1.5 (sub-wall claim), ≥ proflm at
every rate (dominance). P2: F1 wave at the lowest admitted rate + 2.0
parity. Sequencing note: pgq7 K2 (Triton kernel) should wait for the P1
verdict — the packed format's row semantics change if pgq8 lands.
