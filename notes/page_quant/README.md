# page_quant — findings organized by approach

The knowledge from the pgq1→pgq3 arc (2026-07-07 → 07-09), organized by **approach**
rather than by study. Each approach doc is self-contained: idea → what was built →
results (with CIs) → verdict → artifacts → open directions.

## Approaches

| # | approach | verdict |
|---|---|---|
| [1](approaches/01_calibrated_basis.md) | Calibrated query-aware basis (Σ_Q / qpca_unc) | **foundation — validated twice** |
| [2](approaches/02_entropy_coded_pages.md) | Entropy-coded fixed-byte pages + Mode-A serving | **deployed method** (codec inherited from ec_k2v2; this arc added format + allocation) |
| [3](approaches/03_importance_omega.md) | ω importance weighting (mean-logit signal) | **works — as a regime law** |
| [4](approaches/04_fixed_width_codecs.md) | Fixed-width random-access codecs (scalar → RVQ → TCQ/E8/sparse/learned) | **closed with data** |
| [5](approaches/05_page_selection.md) | Page selection / eviction | **real signal, niche defined** |
| [6](approaches/06_serving_and_kernels.md) | Serving architecture & kernels | **measured; port targets known** |

## Chronological records (`studies/`)

`plan.md` (pgq1 design v1→v3) · `report.md` (pgq1) · `report2.md` (pgq2) ·
`plan3.md` (pgq3 pre-registration + OSCAR amendments) · `report3.md` (pgq3) ·
`plan4.md` (pgq4 pre-registration + A1/A2) · `report4.md` (pgq4: profile rungs
+ recency window tie EC at 2 b/c, SIG over rvq, kernel-ready format).
Bug ledger: [`fixes_to_apply.md`](fixes_to_apply.md) (root-caused chains, pgq_fixed +
pgq3-1..5).

## Presentation (`presentation/`)

`final_report.md` / `final_report.html` (shareable consolidated report) ·
`method_explainer.html` (newcomer walkthrough with math + interactive RDO demo).

## Protocol constants (all results)

Llama-3.1-8B-Instruct · LongBench 5-task (lcc, musique, 2wikimqa, qasper, hotpotqa),
FP = 48.40 · v7 protocol (layer-0 fp16, V = TurboQuant 2b) for codec results; kvpress
conventions for eviction results (NOT cross-comparable) · rates honest all-in ·
row-paired 10k bootstraps · calibration compact8-TRAIN only.
