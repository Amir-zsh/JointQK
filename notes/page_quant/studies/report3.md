# pgq3: ω page selection + the no-EC door, closed against the full idea class

**Study window:** 2026-07-08 → 07-09 · **Branch:** `paged_quant` · **Model:** Llama-3.1-8B-Instruct
**Pre-registration:** `plan3.md` (incl. OSCAR amendments + documented gate recalibration).
Code implemented remotely (Ultraplan), integrated as commits `276e77c`/`89d07d3` after a
local audit that fixed two blocking bugs pre-run. Bug ledger: `fixes_to_apply.md` pgq3-1..5.

## Headline results

**Thrust A (ω page selection):** the calibrated mean-logit signal is REAL for page
eviction but loses to in-context statistics. At compression ratio 0.50, five tasks,
row-paired bootstraps:

| contrast (omega_page minus) | Δ F1 | 95% CI | verdict |
|---|---|---|---|
| its random-page control | **+2.52** | [+0.58, +4.52] | SIG — signal beats structure |
| streaming_llm | **+4.54** | [+2.42, +6.68] | SIG |
| snapkv | −1.38 | [−3.14, +0.33] | tie |
| expected_attention | **−2.39** | [−4.47, −0.31] | SIG against us |

5-task means: FP 48.40 · expected_attention 46.98 · snapkv 45.66 · **omega_page 43.33**
· random_page-control 41.04 · streaming_llm 38.99 · knorm 38.45 · random 6.46.

Pre-registered outcomes: primary bar (≥ expected_attention) **failed**; A2.2 expansion
not triggered; the A1 probe's quest gate did not fire (true-query Quest boxes 42.7 vs
static calibrated 43.8 recall@25% — query-aware A3 dropped by rule). The claim ladder
resolves to its pre-agreed fallback: **page selection from calibration statistics alone
— no runtime hidden states — captures most of the achievable signal and beats every
zero-signal baseline; its niche is offloaded pages, prefix caches, and cold tiers where
in-context statistics do not exist.** (The probe's `incontext_mu` scorer, 50.0 vs 44.3
recall, predicted exactly this ordering offline before any F1 was spent.)

**Thrust B (no-EC reopening):** **closed, comprehensively, at zero F1 cost.** All four
pre-registered families — page-reset TCQ (with the compander as its 1-state control and
fixed-K sparse-significance rungs in its ladder), E8 Voronoi product codes, the
OSCAR-emulation anchor, and the learned linear pair — fail G-P3 proxy screening against
the pgq2 long-vector RVQ incumbent at matched honest rates:

| method @1.5 b/c | top1 | logit_err |
|---|---|---|
| ecu (EC reference) | 0.495 | 0.0032 |
| **pgq_rvq_rdo (incumbent, F1 42.62)** | 0.461 | **0.0041** |
| nd (pgq2 scalar) | 0.428 | 0.0171 |
| tcq (qpca_unc basis) | 0.415 | 0.0174 |
| tcq (learned basis, family d) | 0.394 | 0.0298 |
| e8 | 0.746* | 0.0400 |

(*e8's top1 is proxy pathology, third documented sighting: heavy eviction protects sink
argmaxes while destroying everything else — `logit_err` is the honest column.)

Three findings inside the negative:

1. **Trellis, warped tables, and sparse-K rungs bought nothing over plain scalar**
   (tcq 0.0174 vs pgq2's nd 0.0171). The fixed-width zero-coding hypothesis (family e)
   fails: sparse rungs do not recover the entropy model's near-free zero coordinates.
2. **The transform is not the constraint** (family d): 248/248 heads trained without
   overfit, the STE objective improved, and the resulting codec got WORSE
   (logit_err +71%, sinkCE 0.015→0.112 — the learned basis broke qpca_unc's
   sink-alignment). RVQ dominating under the SAME frozen basis had predicted this.
3. **What matters is effective vector dimension.** At 1.0–1.5 b/c the angular-fidelity
   ranking EC > 64-dim VQ > 8-dim lattice ≈ per-coord scalar held across three
   independent measurements (pgq2 F1, gate physics, proxies). The no-EC door stays
   closed below ~1.5 b/c, now against the full idea class the pgq2 report named.

## Gate physics learned along the way (three sweeps, each fix verified)

- **Direction-code shrinkage** (sweep 1): sparse-K zeroing and coarse tables return
  short vectors; with sinks kept exact, softmax reallocates mass to sinks — G2 failing
  as a G3 symptom. Fix: raw-domain norm transmission with decode-side renormalization
  (`r̂ = n16·û/‖û@G‖`) → raw normR = 0.998–1.000 **by construction**, better than the
  incumbent's own 0.963.
- **LS gain ≠ norm restoration** (sweep 2): the least-squares gain is a shrinkage
  estimator (MSE-optimal, norm-destroying) — it lowered E8's normR 0.91→0.85. The
  softmax cares about scale, not MSE; another instance of the project's core theme.
- **Gate calibration**: plan3's original absolute G2/G3 thresholds were stricter than
  anything ever measured at these rates, including the incumbent; recalibrated
  incumbent-relative (documented in plan3 §gates before any F1; F1 bars untouched).
- **OSCAR's windows replicated in reverse**: the emulation without (S0=64, W=256) BF16
  windows corrupts Llama sinks (sinkCE 0.076, mass −0.083); with them, sinks are exact.
  Their ablation knee is a *requirement*, not a tuning choice.

## Where this leaves the project

The deployment recommendation of `final_report.md` §8 is unchanged and now maximally
supported: **Mode-A entropy coding at prefill for the resident tier, EC fixed-byte pages
for cold tiers.** For decode-time page *selection*, use in-context scorers
(expected_attention-class) when hidden states are available; the calibrated ω scorer is
the tool for offloaded/cold pages. A fused-decode kernel would only ever be justified
for an RVQ-class codec (−4.2 vs EC), and nothing cheaper reaches even that.

What would genuinely reopen the door (recorded for completeness): codes with effective
vector dimension ≫8 that stay table-free at decode (e.g. trellis-coded VQ over long
blocks, or learned implicit codebooks evaluated in-kernel) — a different engineering
class than anything tried here.

## Artifacts

- Probe: `artifacts/page_quant2/page_selection_probe.json` (+ `omega_stats_llama31_8b.pt`)
- Eviction cells: `artifacts/bench_evict/llama31_8b/` (35 cells), summary
  `artifacts/bench_evict/bench_summary.json`
- Bundles: `artifacts/page_quant2/pgq3_bundle__qpca_unc__compact8train60r400{,__scalarctl,__lin}.pt`
  (+ `linear_pair_llama31_8b.pt`, `linear_pair_report.json`, `frozen_choices_pgq3.json`)
- Gates: `pgq3_heldout_report.json` (three sweeps in log history:
  `logs/pgq3_fits{,2,3}.log`, `logs/pgq3_lin_chain.log`)
- Screening: `artifacts/page_quant/phaseA_pgq3.json`, `phaseA_pgq2_refs.json`,
  `phaseA_pgq3_lin.json`
- Code: `kvq/compression/{tcq,e8,oscar_arm}.py`, `kvq/presses/omega_page_press.py`,
  `pipelines/page_quant/{probe_page_selection,fit_pgq3_bundle,train_linear_pair}.py`,
  `pipelines/bench/launch_evict_longbench.sh`, `pipelines/eval/aggregate_evict.py`
