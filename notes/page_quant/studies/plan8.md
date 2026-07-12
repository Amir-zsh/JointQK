# plan8 (pre-registration): pgq8 — page-level token-axis DCT codec

**Registered:** 2026-07-11, branch `pgq8` (cut from `pgq7`, user-approved
after the probe8 GO verdict). Protocol v7 (Llama-3.1-8B-Instruct primary,
5-task LongBench, layer-0 fp16, V = TurboQuant 2b, honest all-in rates,
compact8-TRAIN fitting, row-paired 10k bootstraps). Probe:
`notes/page_quant/studies/probe8_token_axis.md` (bars below were stated
there before this plan was written; selection rows untouched by the probe).

## 0. Thesis

User direction (2026-07-11): apply the transformation at PAGE level so the
transform output mixes information about every token inside the page, then
quantize that. Formally: per interior page, replace the 64 token rows of the
code matrix R (64×d, qpca_unc space) by coefficient rows Y = D·R with D the
orthonormal DCT-II along the token axis; run the existing per-row width-rung
RDO on coefficient rows; decode R̂ = Dᵀ·Ŷ.

Why it can work where pgq6 merging could not: token-axis correlation is
large (lag-1 0.70/0.79) and LONG-RANGE (lag-16 0.48/0.63); a graded
transform harvests smooth long-range components a hard 2:1 merge cannot.
Why it may reach sub-wall rates: the pgq3 ~1.5 b/c closure bars bound
per-token-INDEPENDENT coders; token mixing is outside that class. Probe
matched-rate sim (same allocator, same grids, only the row basis changed):
SE ratio DCT/identity 0.58/0.63/0.65 at 1.0/1.5/2.0 (Llama), 0.55/0.60/0.61
(Qwen). DCT ≈ pooled KLT (gap < 2%, Toeplitz cv 0.015) → fixed transform,
no sideband. Fold identity survives: z_page = Dᵀ(Ŷ q̃ᵀ) — one 64-point
inverse transform per (page, query head) per step.

**Signal policy unchanged (phase-symmetric only).** No new allocation
signals; the DCT is a fixed data-independent matrix.

## 1. Format (P0)

`kvq/compression/pgq8_dct.py::PageDCTCompressor` (loader family `pgq_dct*`,
e.g. `pgq_dctlm_rdo`, `pgq_dctlmrw_rdo`) — v1 contract:

- Interior FULL pages (64 tokens) transform: Y = D·R_page. Pages that stay
  IDENTITY: page 0 when it contains sinks (positions 0-3 keep the absolute
  8-bit escape), rw-forced recency pages, the partial last page, and all
  decode-ring pages (Mode-B' unaffected).
- Rungs = the frozen pgq4 width profiles, assigned per COEFFICIENT ROW by
  the same per-page λ-bisection + greedy refinement under the same all-in
  budget (header + 3-bit id per row — rate semantics identical to pgq4, so
  honest rates are directly comparable).
- Grids: LM cents / covering-uniform top as in pgq4, but scaled by a
  per-(coefficient-row, coord) std map `dct_std` (L,H,64,d) fit on the SAME
  fit pool (stats refit only — basis, profiles, alphas, ladder, targets all
  frozen from ff390304). Identity rows keep code_std. alphas carried
  unchanged (coverage checked by the G-gates).
- No new metadata: rung ids double as coefficient-row ids; the transform is
  hardcoded. 2-bit page-class flag reserved (transform on/off is position-
  derivable in v1: full interior page ⇒ transformed).
- Decode/eval: reconstruct R̂ = Dᵀ Ŷ per transformed page, then k̂ = r̂G+μ —
  same-shape roundtrip, no press change (mirrors pgq6's drop-in property).

## 2. P1 — screening (selection rows, Llama bundle ff390304 + dct_std)

Arms: `pgq_dctlm_rdo` at per-token rates {1.0, 1.25, 1.5, 2.0}. Controls:
`pgq_proflm_rdo` and `pgq_mrglm_rdo` at the same rates (existing numbers,
re-used). Harness: `score_pgq_candidates.py` unchanged.

**Pre-registered bars (from the probe note, frozen before any run):**
- **B1 (sub-wall):** dctlm@1.5 logit_err < ecu closure bar **0.00318**.
- **B2 (dominance):** dctlm ≤ proflm logit_err at EVERY rate (transform
  never hurts; ties allowed within 2%).
- **B3 (gates):** held-out rate within ±2%, overflow < 1%, sink code-relerr
  ≤ 0.01, |normR−1| ≤ 0.05 at 1.5 and 2.0 — pgq4's G1/G2'/G3' verbatim.
- Secondary (descriptive): dctlm@1.0 vs ecu@1.0 0.00646; margin to the
  per-page-KLT oracle.

## 3. P2 — F1 wave (fires only if B1 & B2 & B3 pass)

Cells (Mode A, launch_pgq_longbench.sh, labels carry bundle sha8):
1. `pgq_dctlmrw_rdo` at the LOWEST rate whose logit_err ≤ proflm@2.0's
   0.00272 (the "same quality, fewer bits" headline — aim (a)); expected
   from sim: ~1.25.
2. `pgq_dctlmrw_rdo:2.0` — parity/dominance arm.
3. Incumbent control `pgq_proflmrw_rdo:2.0` — existing cells.
Refs: FP, TQ, ecu, rvq, oscar — all existing. Row-paired 10k bootstraps
(fixed parser); per-task + pooled + non-lcc; report8.md.
Decision tree: (a) low-rate arm F1 within CI of proflmrw@2.0 ⇒ headline
"OSCAR-class quality at ~60% of the bits"; (b) 2.0-parity tie ⇒ transform
adopted as format default like mrg; (c) both fail ⇒ SE→F1 exchange-rate
lesson recorded, format stays pgq6.
Qwen transfer: only if the Llama wave confirms (machinery ready via
--model-tag; dct_std refit from the 12-row Qwen pool).

## 4. Interaction with pgq7 (kernel port)

K2 (Triton stage-1) HOLDS until the P1 verdict: if pgq8 lands, stage-1 dots
run against coefficient rows and stage-2 gains a per-page 64-point inverse
DCT on the logit tile; the K1 packed format's payload/segment layout is
unchanged (rows are rows), only row semantics change. K1 golden vectors
regenerate mechanically.

## 5. Budget & hygiene

Fit pass ~0.2 GPU-h; P1 ≈ 3-4 GPU-h; P2 ≈ 10-15 GPU-h if fired. Logs
`logs/pgq8_*.log` + heartbeats; fingerprint 262/262 before any commit;
per-commit approval. Deferred (recorded): pre-RoPE variant (+10-15%
correlation headroom, probe §5); per-page adaptive transforms (oracle gap
16-27× vs fixed 2.5-6×); Haar kernel fallback; transform-aware ω.
