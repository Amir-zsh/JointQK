# pgq2: can a random-access (no-entropy-coding) codec match EC? — No.

**Date:** 2026-07-07/08 · **Branch:** `paged_quant` · **Model:** Llama-3.1-8B-Instruct
**Protocol:** v7 (Mode A, layer-0 fp16, V=v_turboquant@2b, fraction 1.0, compact8
exclusions) · **Tasks:** lcc/musique/2wikimqa + qasper/hotpotqa (new; FP/TQ baselines
from `downstream_v7`, ecu references benched fresh) · **Plan:** approved pgq2 plan
(two arms in isolation, then +ω; all hyperparameters frozen on TRAIN selection rows
before any F1 — `artifacts/page_quant2/frozen_choices.json`).

## TL;DR — the viable-solution statement

1. **Entropy coding owns quality at ≤1.5 b/c, decisively and significantly.** The best
   random-access arm (norm + 4-stage residual VQ, per-token stage RDO) loses to
   token-uniform EC by **−4.2 F1 at 1.5 b/c (95% CI [−5.8, −2.6])**, −12 at 1.0, −22
   at 0.75 (5-task paired bootstraps). Combined with pgq1: EC-in-fixed-pages @1.5 =
   43.98 (trio) ≈ unconstrained EC — *pages are free; the entropy model over
   coordinate values is the necessary ingredient.*
2. **Deployable architecture (data-settled):** compress K at prefill with the
   EC-family codec and let the cache hold reconstructions (the Mode-A pattern every
   F1 number here already uses — no decode-path codec required); EC pages for
   offload/prefix/storage tiers (decode-once, PCIe-bound); the fully-compressed
   *resident* cache below ~2 b/c stays open — this study maps its frontier.
3. **ω (importance-weighted allocation) replicated in a third representation family,
   with a sharp regime law.** Arm A (norm + scalar directions): **+3.7 mean SIG**
   (hotpotqa +8.6 SIG; lcc negative again — fourth sighting of the OOD-code
   exception). Arm B (fine-grained VQ): **tie**. Refined claim: *ω pays exactly when
   the ladder is coarse enough that unweighted RDO starves attended tokens; ladders
   with fine rate granularity self-protect.*
4. **The representation stack at ~1.14 b/c (all-in, matched):** uniform scalar
   directions 12.9 → +per-token RDO 18.0 → +ω 23.1 → residual VQ 33.0 (+13.0 SIG
   vs scalar) → EC ~45. Expressiveness dominates; allocation refines.
5. **Sink physics, now fully characterized (3 gate-caught bugs, 0 GPU-h wasted):**
   sink *norms* are outliers (fp16 norms fix), sink *directions* are outliers
   (std-scaled codebooks and 138k-token codebooks both miss them; an absolute
   [−1,1] mid-tread grid is outlier-proof for any unit vector), and grids without a
   zero level corrupt norms coherently (third sighting). Raw-k sink relerr is
   inherently amplified by qpca_unc's non-orthonormal inverse — the attention-true
   gate metric is code-space relerr (0.015 ✓) + sink-mass shift.
6. Scalar shaped-direction coding (Arm A) is a clean negative even at abundant rate:
   36.1 at 2.14 b/c vs TurboQuant 45.7 at 2.125 — TQ's isotropic Beta-matched
   construction beats per-coord std-shaping of normalized directions.

## Full wave-1 grid (5-task mean | trio mean, honest all-in rates)

| config | rate | 5-task | trio |
|---|---|---|---|
| full_precision | 16 | 48.40 | 44.99 |
| turboquant_k2_v2 | 2.125 | 45.73 | 41.52 |
| ecu @1.5 / 1.0 / 0.75 | — | 46.80 / 45.22 / 43.76 | 43.02 / 41.25 / 40.01 |
| pgq_rvq_rdo @1.5 / 1.0 / 0.75 | 1.496/0.996/0.746 | 42.62 / 32.97 / 18.76 | 37.63 / 29.14 / 18.36 |
| pgq_rvq_ea @1.0 | 0.996 | 32.93 | 28.49 |
| pgq_rvq_uni (2 stages) | 1.141 | 37.03 | 32.23 |
| pgq_nd_ea / nd_rdo @1.0 | 0.996 | 23.05 / 17.96 | 20.05 / 16.63 |
| pgq_nd_rdo @1.5 | 1.489 | 27.55 | 23.72 |
| pgq_nd_uni w1 / w2 | 1.141/2.140 | 12.92 / 36.10 | 12.22 / 31.35 |

Zero page overflow in every pgq2 cell (structural), rates exact, all gates logged in
`artifacts/page_quant2/pgq2_heldout_report.json`.

## Pre-registered bootstrap contrasts (row-paired, 10k, 5-task)

- rvq@1.5 − ecu@1.5: **−4.16 [−5.80, −2.57] SIG** (deployment gate: fail)
- rvq@1.0 − nd@1.0: **+13.01 [+10.94, +15.11] SIG** (VQ ≫ shaped scalar)
- nd_ea − nd_rdo @1.0: **+3.68 [+2.25, +5.20] SIG** (ω replicates, 3rd family)
- rvq_ea − rvq_rdo @1.0: −1.00 [−2.65, +0.62] tie (ω neutral under fine ladders)
- rvq@0.75 − ecu@0.75: −21.73 SIG (fixed-page starvation, confirmed)

## Big-pool retrain (pre-registered addendum): data-starved or fundamental?

The compact8 pool keeps raw K for test rows only (leakage), so 40 fresh TRAIN rows
were captured (`artifacts/calibration_splits/pgq2_bigfit_40/`,
run `pgq2_bigfit_llama31_8b`, disjoint from the EC 26). RVQ codebooks retrained on
~5× the tokens, frozen τ/θ carried unchanged; single F1 config `rvq_rdo@1.5`.

**Result: a wash.** Retrained rvq_rdo@1.5 = **42.06** (lcc 39.14 / musique 26.83 /
2wiki 44.43 / qasper 42.07 / hotpotqa 57.84) vs 42.62 with 138k-token codebooks —
within noise, and the overfit gate still fired 161 stage-level fallbacks at 5× data.
**Per the pre-registered rule: the VQ→EC gap is FUNDAMENTAL** — it is the entropy
model over coordinate values, not codebook quality. No-EC parity at ≤1.5 b/c
requires a different idea class (learned transforms with lattice/trellis codes), or
accepting the Mode-A architecture as the answer. Fresh capture:
`artifacts/calibration/pgq2_bigfit_llama31_8b/` (40 TRAIN rows, 15 min GPU).

## ω-gate arm (wave 2b): negative

nd_eag@1.0 = 23.03 vs nd_ea 23.05 (lcc 27.41 vs 27.75). The within-page-m-spread
confidence gate (θ=0.25, frozen) fires too rarely to matter and does NOT repair the
OOD-code regression. The lcc/ω interaction needs a different trigger (page-level
code/text discrimination or per-head predictor-quality weights) — future work, not
patched mid-study.

## Artifacts & code

Bundles `artifacts/page_quant2/pgq2_bundle__qpca_unc__*.pt` (v3);
`frozen_choices.json` (τ=0.5 all rates, θ=0.25, sha-stamped, + retrain addendum);
cells `artifacts/bench_pgq/llama31_8b/pgq__pgq_{nd,rvq}_*`;
code `kvq/compression/{norm_direction,rvq}.py` (+ loader in page_quant.py),
`pipelines/page_quant/{fit_pgq2_bundle,select_pgq2_hparams}.py`,
tests `tests/test_{norm_direction,rvq}.py` (16 new, 41 total green).
