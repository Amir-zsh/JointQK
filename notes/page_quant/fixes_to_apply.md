# page_quant bug tracker

## FIXED: pgq_fixed F1 collapse — three stacked root causes (one real lesson)

- **Symptom:** pgq_fixed/pgq_fixed_ea F1 collapsed to ~0–14 (whitespace-only
  generations) across three successive fit configurations, while per-head proxies
  (top1 0.476, k_mse ~1.0, no NaN, sane norms) stayed unremarkable each time.
- **Diagnosis chain** (each stage verified with a targeted probe, not proxies):
  1. *Mid-rise grids have no zero level* → every near-zero coord inflates to ±s/2 →
     all key norms grow coherently → softmax corrupted network-wide. Ranking
     proxies are blind to uniform norm inflation. Fix: mid-tread symmetric grids
     (zero level), widths {0,2,4,8} (1-bit mid-tread is degenerate).
  2. *"Tail-safe" q999 scales are not safe for sinks*: attention-sink keys (tokens
     0–3, carrying 19–84% of realized mass per head) sit 2.7–5.4× BEYOND the
     per-coord 99.9% quantile, so every width clipped them equally → the
     Lagrangian saw no distortion gain from wider widths → assigned sinks w=0 →
     sink mass 0.84→0.007 → generation destroyed. Probe:
     `sink mass ref/fixed/ec = 0.843/0.007/0.836`. Fix: w4/w8 scales cover the
     per-head fit-data max |r| (recovered from the finest EC rung's alphabets;
     live-coord filtered, ratio clamped to 8).
  3. *Even covering 4-bit grids are too coarse for sinks and the Lagrangian
     rationally refuses to buy w8 at tight budgets* → forced w=8 for sequence
     positions 0–3 (position-based ⇒ decoder-derivable, zero sideband; page 0's
     budget pays; standard practice — KIVI keeps sinks high precision). After the
     fix: sink relerr 0.02–0.04, rate exactly 0.996.
- **The lesson (report-worthy):** entropy-coded deadzone quantizers get sink safety
  FOR FREE (unbounded calibration-range alphabets + zero bin). Fixed-width
  saturating grids must handle the zero level and the sink tail EXPLICITLY, and
  per-head reconstruction metrics + argmax proxies cannot detect either failure —
  both corrupt the softmax denominator, not the ranking.
- **Residual (known, not a bug):** at b=1.0 the width mix is eviction-heavy (~50%
  w0), which shrinks non-sink competitor logits and overshoots sink mass
  (0.84→0.96). Rate-regime property; judged by F1.
- **Verification:** unit tests 8/8; sink-gate probe PASS on relerr; F1 rerun3
  (sha post-patch) in flight; quarantined cells: `*__BROKEN_pre_scalefix`
  (MSE-fit mid-rise), `*__BROKEN_midrise` (q999 mid-rise), `*__BROKEN_stale`
  (pre-sink-fix).

## pgq3-1: probe frozen rule could select non-deployable scorer (caught in audit, pre-run)
- **Root cause:** `probe_page_selection.py` put `incontext_mu` (built from the row's own
  realized decode queries) in `STATIC_SCORERS`, which fed both the frozen score_mode
  argmax and the quest-gate baseline; it is not an `OmegaPagePress` score mode, so a win
  would have frozen an unusable rule and crashed the evict launcher.
- **Fix:** `DEPLOYABLE_SCORERS` (omega_max/omega_mean/quest_mu) now defines the argmax
  and gate baseline; `incontext_mu` is measurement-only; rule string records this.
- **Verification:** final probe run froze `omega_mean` (44.3) while `incontext_mu`
  measured 50.0 — the exclusion demonstrably bound.

## pgq3-2: probe recall metric floored by forced sink/recent mass
- **Root cause:** `recall_at` counted force-kept pages (sink page 0 + recent page) in
  numerator and denominator; sinks hold 19–84% of attention mass, flooring every scorer
  near ~85%+ (random@50% = 89.3 raw) and voiding both the budget-line assert and the
  ±10-pt quest-gate scale. The probe's own sanity assert caught it on the first run.
- **Fix:** contested-mass recall — forced pages excluded from both sides; budget still
  counts them (matches press accounting); `meta.recall_basis` records the change.
- **Verification:** rerun passes all asserts (random 23.8@25%, 49.2@50%); oracle 53.0,
  monotone curves, oracle-dominance intact.

## pgq3-3: direction-code shrinkage — G3 normR 0.87-0.93, reactive G2 sink-mass rise (first gate sweep)
- **Root cause:** TCQ/sparse-K/E8 reconstructions come back systematically short
  (sparse-K zeros non-top-K coords; coarse warped-LM tables underfit tails; E8's
  wrap-to-zero guard biases toward 0). Sinks are kept nearly exact, so shrunken bulk
  logits reallocate softmax mass onto sinks: sinkΔ +0.075..+0.13 with sinkCE 0.015 —
  G2 failing as a *symptom* of G3.
- **Fix:** (a) TCQ family: decoder-side unit renormalization of every decoded direction
  (norm is transmitted; zero rows stay zero), Dmat prices the renormalized snap;
  (b) E8: per-(l,h,rung) least-squares gains fit on TRAIN (E[<rm,rh>]/E[||rh||²]),
  stored in bundle (`e8_gains`), zero per-token bits.
- **Verification:** unit tests updated to the new decoder contract (30 green); held-out
  G1-G3 re-gate pending refit.

## pgq3-4: OSCAR emulation without its BF16 windows corrupts sinks (first gate sweep)
- **Root cause:** family (f) omitted the published recipe's (S0=64, W=256) BF16
  sink/recent protection; per-token min-max scales do NOT isolate sink outliers on
  Llama (sinkCE 0.076, sinkΔ -0.083 at INT2) — independently replicates why OSCAR
  ships those windows.
- **Fix:** fp16 passthrough windows (64, 256) in oscar_arm, charged 16 b/c, honest at
  short contexts; window-off contract kept testable via ctor fields.
- **Verification:** test_oscar_windows_fp16_and_charged green; re-gate pending refit.

## pgq3-5: gate thresholds mis-transcribed + E8 LS-gain was a shrinkage estimator (second gate sweep)
- **Root cause (a):** plan3's absolute G2/G3 thresholds (shift ≤0.01, normR ∈[0.98,1.02])
  are stricter than the pgq2 record-holder itself achieved (rvq_rdo@1.5: 0.054 / 0.963) —
  unpassable at these rates. **(b):** the E8 LS gain E[<y,ŷ>]/E[||ŷ||²] minimizes MSE by
  shrinking toward zero — it LOWERED normR (0.91→0.85). **(c):** TCQ's code-domain unit
  renorm left raw normR at 0.98–1.05 (non-orthogonal inverse map).
- **Fix:** gates recalibrated incumbent-relative in plan3 (documented, F1 bar untouched);
  both families redesigned to transmit the RAW-domain norm and renormalize after the
  inverse map (r̂ = n16·û/||û@G||): raw normR = 1 by construction. E8 restructured to
  norm+direction (lattice codes the unit direction; +16b/rung, all-zero rung = free
  evict); LS gains removed.
- **Verification:** 30 tcq/e8 tests green on the new contracts; third gate sweep pending.
