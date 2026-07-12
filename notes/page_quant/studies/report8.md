# report8 — pgq8: page-level token-axis DCT codec

**Study:** plan8.md (registered 2026-07-11, branch `pgq8`), protocol v7.
User direction: transform at PAGE level so the output mixes information
about every token in the page, then quantize that. Format:
`PageDCTCompressor` (`pgq_dct*`) — fixed orthonormal DCT-II across the 64
tokens of each interior page in qpca_unc code space, coefficient rows
through the frozen pgq4 width-rung page RDO, per-(coefficient-row, coord)
scale maps (`dct_std`, the ONLY refit; bundle 9d2527b5 = pgq4 ff390304 +
dct_std). Pre-test: probe8_token_axis.md (GO on every arm; its matched-rate
sim predicted the screening ratios within a few points).

## Verdicts (registered tree, plan8 §2-3)

1. **B1 sub-wall: PASS — the entropy-coding closure is broken on the proxy
   axis, at 1.5 AND 1.0 b/c.** Held-out selection-row logit_err:
   dctlm@1.5 **0.00313** < ecu bar 0.00318; dctlm@1.0 **0.00621** < ecu@1.0
   0.00646. The pgq3 wall bounds per-token-INDEPENDENT coders; the token-
   mixing transform is the first arm outside that class, and it is the
   first arm in the arc to sit on the EC frontier's good side.
2. **B2 dominance: PASS.** dctlm beats proflm at EVERY rate — err ratios
   0.50 / 0.59 / 0.65 / 0.69 at 1.0 / 1.25 / 1.5 / 2.0 (probe sim
   predicted 0.58 / — / 0.63 / 0.65).
3. **B3 gates: PASS on every arm.** Rates exact (1.5000, 1.9999; rw arms
   1.8868 @1.75, 2.1288 @2.0), ovf 0, sinkCE 0.002, normR 0.972-0.985.
4. **F1 parity at matched rate: TIE with positive point — the transform is
   free.** dctlmrw@2.0 − proflmrw@2.0 = +0.56 [−0.77, +1.86]; no task
   hurt. Adopted into the format default (branch (b) of the tree).
5. **F1 headline (aim a): incumbent quality at 11% fewer honest bits.**
   dctlmrw@1.75 (honest 1.8868) vs proflmrw@2.0 (honest 2.1288):
   −0.50 [−1.80, +0.76] — the registered CI∋0 criterion met. Crossover
   located by screening (dctlm@1.6 0.00281 just above the 0.00272
   criterion; 1.75 = 0.00239 passes).
6. **vs entropy coding at F1:** dctlmrw@2.0 ties ecu@2.0
   (−0.78 [−2.37, +0.82]); dctlmrw@1.75 loses SIG to ecu@2.0
   (−1.84 [−3.61, −0.07]). The proxy sub-wall does NOT convert to F1
   superiority in the saturated regime (probe-note caveat confirmed);
   EC keeps a small F1 edge concentrated in musique/2wikimqa.
7. Rate response is real: dctlmrw 1.75 vs 2.0 = −1.06 [−2.12, −0.01] SIG,
   qasper-driven (−2.95 SIG).

## P1 screening (selection rows, real queries; logit_err)

| rate | dctlm | proflm (pgq4) | ratio | closure bars |
|---|---|---|---|---|
| 1.0 | **0.00621** | 0.01233 | 0.50 | ecu 0.00646 → beaten |
| 1.25 | 0.00433 | 0.00728 | 0.59 | — |
| 1.5 | **0.00313** | 0.00483 | 0.65 | ecu 0.00318, rvq 0.00405 → both beaten |
| 1.6 | 0.00281 | — | — | crossover probe |
| 1.75 | 0.00239 | — | — | ≤ proflm@2.0 (0.00272) → headline arm |
| 2.0 | 0.00188 | 0.00272 | 0.69 | ecu@2.0 0.00155 stands |

## W1 F1 (Llama v7, 5 tasks, honest rates; row-paired 10k bootstraps)

| task | dctlmrw@1.75 (1.887) | dctlmrw@2.0 (2.129*) | proflmrw@2.0 (2.129) | ecu@2.0 |
|---|---|---|---|---|
| lcc | 50.92 | 50.08 | 50.05 | 49.76 |
| musique | 30.90 | 30.63 | 30.97 | 33.10 |
| 2wikimqa | 44.87 | 47.33 | 45.11 | 49.10 |
| qasper | 40.94 | 43.89 | 42.92 | 44.55 |
| hotpotqa | 59.25 | 60.23 | 60.34 | 59.57 |
| **mean5** | 45.38 | **46.43** | 45.88 | 47.22 |

*dctlmrw@2.0 heldout rate matches the incumbent's 2.1288 convention.
Contrasts in `pgq8_bootstraps.json` (fixed parser).

## Interpretation

The user's page-transform idea does exactly what the probe predicted at the
proxy level and lands two format facts at F1: (1) mixing tokens before
quantization costs nothing at the shipping rate and buys a large low-rate
extension — the R-D curve at 1.0 b/c improves 2× over the width-only codec
and 1.3× over merging (0.00621 vs 0.00894 vs 0.01233); (2) at F1 the win
cashes out as RATE, not score: incumbent-parity at 1.887 vs 2.129 honest
b/c. The EC frontier is now beaten on the axis where EC was provably
optimal for independent tokens — the remaining F1 edge of ecu@2.0 (musique/
2wikimqa) is the next open question, plausibly an interaction between EC's
per-token adaptive rates and retrieval-style tasks rather than a rate-axis
effect. Merging's roles (fewer rows, page-size classes) are orthogonal:
DCT reduces BYTES at fixed rows; mrg reduces ROWS. A combined arm (merge
levels on coefficient rows) is a registered deferred direction.

## Artifacts

`artifacts/page_quant2/`: pgq8_bundle__llama31_8b.pt (9d2527b5),
phaseA_pgq8.json, phaseA_pgq8_crossover.json, pgq8_heldout_report.json,
pgq8_heldout_rw175.json, pgq8_bootstraps.json, token_axis_probe_{llama31_8b,
qwen3_8b}.json; `artifacts/bench_pgq/llama31_8b/pgq__pgq_dctlmrw_rdo__b{1.75,
2.0}__9d2527b5__*` (10 cells). Code: kvq/compression/pgq8_dct.py, loader
family pgq_dct* + dct_std validation (page_quant.py), lm_codes 2-D scale
support (pgq4_folded.py), fit_pgq8_stats.py, probe_token_axis.py,
launch_pgq_longbench.sh PGQ8_BUNDLE routing, fit_pgq4_bundle.py --bundle
eval override, tests/test_pgq8.py (9). Suites green: pgq4/pgq6/pgq8/pack/
page_quant 44/44.

## Appendix: Qwen3-8B transfer (2026-07-12, bundle 2cf29a8a)

Same recipe, zero new choices: dct_std refit from the 12-row Qwen fit pool,
everything else frozen. Screening (selection rows): dctlm dominates proflm
at every rate (0.00293/0.00231/0.00188 vs 0.00441/0.00336/0.00262 at
1.5/1.75/2.0); the crossover criterion (≤ proflm@2.0 = 0.00262) again
selects **1.75**. Gates clean (rates 1.8175/2.0636, ovf 0, sinkCE 0.003,
normR 0.978/0.983). W1 (10 cells, row-paired vs the committed pgq5
incumbent cd6a3d41; pgq8_bootstraps_qwen.json):

| task | dctlmrw@1.75 (1.818) | dctlmrw@2.0 (2.064) | proflmrw@2.0 (2.064) |
|---|---|---|---|
| lcc | 63.18 | 63.43 | 63.41 |
| musique | 31.31 | 31.77 | 32.49 |
| 2wikimqa | 44.85 | 46.00 | 44.17 |
| qasper | 39.65 | 40.00 | 37.76 |
| hotpotqa | 61.48 | 62.82 | 63.60 |
| **mean5** | 48.09 | **48.80** | 48.29 |

- parity at 2.0: +0.52 [−0.66, +1.73] tie (positive point — free, again);
- **headline replicates: dctlmrw@1.75 vs incumbent −0.19 [−1.46, +1.10]
  tie at 12% fewer honest bits** (1.8175 vs 2.0636);
- **vs TurboQuant K2V2: +5.97 [+3.72, +8.22] SIG at 14% fewer bits than
  TQ's 2.125** (the @2.0 arm: +6.68 [+4.38, +9.00]);
- FP retention at 1.82 b/c: 0.949 (−2.61 [−4.23, −1.04]).

The format story is now two-model consistent at every level: proxy
dominance ratios, the 1.75 crossover, parity-freeness, and the
rate-not-score cash-out.

## Deferred (recorded)

Qwen transfer of dct (machinery ready: fit_pgq8_stats --model-tag qwen3_8b
+ existing --model-tag bench path; plan8 fires it only on user go);
dct × mrg combined arm; pre-RoPE variant (+10-15% correlation headroom);
per-page adaptive transforms (oracle gap); EC's residual musique/2wikimqa
F1 edge; kernel port row-semantics update (pgq7 K2 resumes against
coefficient rows; z_page = Dᵀ(Ŷ q̃ᵀ) stage-2 tile op).
