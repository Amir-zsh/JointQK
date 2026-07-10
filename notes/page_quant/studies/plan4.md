# plan4 (pre-registration): pgq4 — decode-resident page-selective quantization

**Registered:** 2026-07-09, branch `pgq4` (cut from `paged_quant` @ 429e515).
**Status: REGISTERED.** All tunables frozen on selection rows before any F1
(`frozen_choices_pgq4.json`, sha8-labeled cells). Protocol v7 (Llama-3.1-8B-Instruct,
5-task LongBench {lcc, musique, 2wikimqa, qasper, hotpotqa}, FP = 48.40, layer-0 fp16,
V = TurboQuant 2b, honest all-in rates, compact8-TRAIN-only fitting, row-paired 10k
bootstraps). Gates incumbent-relative per the plan3 recalibration.

## 0. Thesis and honest framing

pgq3 closed the no-EC door **below ~1.5 b/c**: effective vector dimension binds there.
This study does not relitigate that. It targets the band where the door is ajar —
**2.0–2.5 b/c**, where TurboQuant-class scalar hits 45.73 @2.125 and the EC-vs-scalar
gap demonstrably shrinks — with codecs whose every operation maps to a named OSCAR
kernel primitive, so decode runs directly on the quantized resident cache (real
GPU-memory savings, unlike Mode A), plus page-selective allocation (fewer bits to
unimportant tokens), which OSCAR's own limitations name as open.

**The pivotal untested claim:** a calibration-static folded-scalar codec with
per-token rungs matches TurboQuant/OSCAR-class quality while carrying strictly less
per-token metadata than either (2–18 bits/token vs OSCAR's 32 bits/(token,group)),
and survives decode-time compression of its own generated tokens.

**The kernel identity the study rests on (family A):**
score = q·k̂ = (q @ Gᵀ diag(s))·i + q·μ, with q·μ constant per head → cancels in
softmax. The non-orthogonal qpca_unc inverse AND the calibration-static per-coord
scales fold into ONE tiny per-head matmul on q per decode step; the segment kernel is
INT-unpack → dot → × per-rung constant. Verified by unit test before any GPU spend.

Fake-quant fp16 storage remains the harness convention; kernel-implementability is a
**design gate** (every family maps to OSCAR's per-width index lists → per-width
stage-1 launches → single tier-agnostic online-softmax stage-2), not a deliverable.

## 1. Families

All: 64-token pages in a calibrated basis, d = 128, sinks (positions 0–3) forced to
top rung position-derivably (zero sideband), exact zero level in every grid,
HEADER_BITS = 96/page, rung-id bits/token, snap-aware distortion, allocation via
`_paged_lambda_assign` + greedy refinement.

### A — `pgq_fold_*` (folded-scalar; primary kernel-tier family)
qpca_unc per-(layer,head); r = (k − μ)@F; per-coord calibration-static scale s_j;
rung = width w ∈ {0, 2, 3, 4} (id 2 bits); symmetric mid-tread grids with exact zero.
Sub-variants (ONE frozen at P2):
- **A-static:** no per-token scalar. R_t = w·d.
- **A-gain:** fp16 raw-domain gain g_t = n_t/‖û@G‖ per token. R_t = w·d + 16.
- Grid: uniform INT (round+clamp to ±(2^{w−1}−1)) vs Lloyd-Max-Gaussian LUT
  (dequant-then-fp32 only; never per-centroid in-graph — OSCAR's LM cautionary tale).
ω arm `_ea`: D-weighting with τ carried from pgq2 `omega_tau_by_rate` (no
re-selection), run at the tightest admitted rate (regime law: ω pays where rate binds).

### B — `pgq_prof_*` (profile rungs = waterfilling + prefix truncation)
Rung = frozen per-coord width profile p (≤8 profiles, id 3 bits), built by
`water_fill(code_std², B_target)` + largest-remainder rounding to {0,2,3,4}, forced
monotone non-increasing in qpca coordinate order, width changes only at 32-coord
block boundaries. Prefix-truncation profiles (w uniform on first r coords, 0 after)
included — the fixed-width, zero-index-bit analogue of EC tail-zero coding.
Per-layer-shared profile shapes IF the per-head→per-layer TRAIN distortion penalty
< 3% (else per-head, and B is demoted to quality-only — kernel story degrades).
Gain/static choice inherited from A's freeze.

### C — `pgq_fold_pgr` (page-rung control; kernel-trivial)
One width per whole page; cross-page allocation via `_paged_lambda_assign` at page
granularity (D_page = Σ_t ω_t·D_t with omega_mean page score; budget = sequence
budget). Side = header + 2 bits/page. Answers: is within-page per-token allocation
worth anything at an INT ladder, or does OSCAR-granularity suffice?

### D — basis ablation (screening-only, promotion-gated)
Identical A-quantizer under: (i) qpca_unc per-head [default], (ii) OSCAR per-layer
U_Q·Hadamard·bit-reversal-perm (parity-checked against
`vendor/OSCAR/rotation/compute_kv_rotation.py`), (iii) r_sym (orthogonal; waterfill
weights diag(R_symᵀ Σ_Q R_sym)). **Promotion rule:** alternative basis gets 5 F1
cells iff its logit_err @2.0 is within 3% relative of (i) — a tie is a WIN for the
cheaper per-layer kernel.

### References (in-wave, not new families)
ecu@2.0 (b2.0 uniform EC bundle — fit in P0), pgq_rvq_rdo@2.0 (pgq2 v2 bundle),
oscar_uni INT2 (honest ≈2.28 b/c incl. 32 b/group + fp16 (64,256) windows; built in
pgq3, never F1'd), existing TurboQuant@2.125 v7 cells.

### Per-token metadata & kernel map

| family | per-token metadata | b/c overhead (d=128) | kernel primitive |
|---|---|---|---|
| A-static | 2-bit rung id | 0.016 (+0.012 header) | per-width index lists; scales folded into q̃ |
| A-gain | id + fp16 gain | 0.14 | + one scalar fetch/token (< OSCAR 32 b/group) |
| B | 3-bit id (+gain if inherited) | 0.023 | per-profile lists; 32-coord constexpr blocks |
| C | 2 bits/page | 0.0005 | page-granular width (OSCAR-native) |
| oscar_uni | 32 b/(token,group) + windows | 0.25+ | the vendored system itself |

## 2. Rate ladder

{1.5, 2.0, 2.5} b/c honest all-in; **2.0 primary**; 2.5 = FP-parity attempt; 1.5 =
stretch (honest re-test of the pgq3 closure with the post-lessons codec — extraordinary
claim needs SIG > 42.62). Layer-0 fp16 and V = TurboQuant 2b retained for
cross-study comparability (challenged, deliberately kept; noted in report4).

## 3. Decode-time arm (Mode-B', the "works during decoding" claim)

`jointqk_press.py` gains default-off flags `decode_chunk=8`, `decode_recent=W`:
decode tokens buffer fp16 in a recent ring; tokens aging past W are quantized in
chunks of 8 (OSCAR's flush granularity) and spliced back; `start_pos` passed to
`roundtrip` so sink forcing applies only to true positions 0–3; window tokens charged
16 b/c in the honest rate. Arms: winner @2.0 × W ∈ {0, 32} × 5 tasks (10 cells).
W=0 is the adversarial case. **Tier-D bar:** ModeB'(32) − ModeA row-paired CI-lower
≥ −1.0 (W=0 reported; power caveat R2 acknowledged — claim worded as robustness).

## 4. Gates (frozen; incumbent-relative to pgq_rvq_rdo at matched rate)

Held-out (blocking, before F1; selection rows; G2 layers [1,8,16,24,31], heads
(0,3,7)); rvq's own @2.0/@2.5 values measured in the same run and written to
`pgq4_heldout_report.json` BEFORE any F1:
- **G1:** realized honest rate within ±2% of nominal; page overflow < 1%.
- **G2:** sink code-relerr ≤ 0.05 AND sink mass shift ≤ shift_rvq(rate) + 0.02.
- **G3:** |normR − 1| ≤ |normR_rvq(rate) − 1| + 0.02.
- **G-P4 (F1 admission per family×rate):** logit_err ≤ 1.15× logit_err(rvq_rdo @ same
  rate). top1 reported, never gating (proxy pathology, 3 sightings).
- **G0-perf:** ≤ 2× rvq press wall-clock on fraction-0.05 smoke; Mode-B' ≤ 1.3× its
  own Mode-A.

F1 bars (row-paired 10k bootstraps):
- **Tier-K (primary):** best kernel-family arm at honest ≤ 2.125 b/c: F1 ≥ 45.73 with
  Δ-vs-TurboQuant CI-lower > −1.0, AND SIG > rvq_rdo@2.0. Success = pre-agreed
  trigger for the OSCAR kernel-port decision.
- **Tier-E (headline):** Δ(arm@2.0 − ecu@2.0) CI ∋ 0 or > 0.
- **Tier-FP (@2.5, report-grade):** Δ vs 48.40 CI-lower ≥ −1.5.
- **Stretch @1.5:** SIG > 42.62 (would amend the pgq3 closure).
- **Tier-D:** §3.

## 5. Phases (GPUs 0–3; logs/pgq4_study.log + heartbeat; target ≤75 GPU-h, cap 120)

| phase | what | cost |
|---|---|---|
| P0 | fit_pgq4_bundle (3 bases, scales, profiles, ω carry) + ec b2.0 uniform bundle | ~3 h |
| P1 | screening: {A-static, A-gain, A-LM, B, B-prefix, C}×{1.5,2.0,2.5} + D×3 @2.0 + rvq/ecu refs | ~2 GPU-h |
| P2 | FREEZE frozen_choices_pgq4.json (A variant, grid, B profiles/sharing, basis, ω rule). No F1 before its sha8 exists. | CPU |
| P3 | held-out G1–G3 → pgq4_heldout_report.json (blocking) | ~1 GPU-h |
| P4 | fraction-0.05 smokes + G0-perf per admitted arm | ~2 GPU-h |
| P5 | W1 (Mode A): A@{2.0,2.5}, A_ea@tightest, B@2.0, C@2.0, ecu@2.0, rvq@2.0, oscar_uni@2.28, (A@1.5 if admitted) ≤45 cells | 20–29 GPU-h |
| P6 | W2 (decode): winner × Mode-B' W∈{0,32} = 10 cells | 5–7 GPU-h |
| P7 | W3 (conditional, cap 20): recent-window prefill ablation (winner+ecu, last 4 prompt pages → top rung, 10), D promotion (5), B@2.5 (5) | ≤13 GPU-h |
| P8 | bootstraps + report4.md + fingerprint + commit proposal | CPU |

Kill-switch economy: if nothing passes G-P4 at any rate, F1 spend = references only
(15 cells, ~15 GPU-h) and the study closes as "the 2 b/c band is TQ/rvq-bound."

## 6. Decision tree (pre-agreed)

1. No family passes G-P4 anywhere → refs-only negative close.
2. A passes, B fails → drop B; promote A@1.5 + D-promotion into W1's freed cells.
3. A-static fails screening, A-gain passes → A-gain is the family; headline becomes
   "one fp16 gain/token suffices" (still < OSCAR metadata).
4. Tier-K met @2.0 → W2 decode arm runs (decode claim only ever made for a tier-K
   codec).
5. Tier-K met, Tier-E missed → headline is metadata/kernel economics + TQ-parity.
6. C ties A @2.0 (Δ CI ∋ 0) → "page granularity suffices at 2 b/c; per-token rungs
   are a 1.5-band tool" — useful negative for the kernel port.
7. ω arm harmful at 2.0 → regime-law confirmation (rate doesn't bind), not failure.

## 7. Pre-registered bootstrap contrasts

A@2.0 − {TQ@2.125, rvq@2.0, ecu@2.0, oscar_uni@2.28}; B@2.0 − A@2.0; C@2.0 − A@2.0;
A_ea − A (ω on INT ladder); ModeB'(W) − ModeA (winner); window-forced − plain
(winner, ecu); A@2.5 − FP. Per-task + 5-task mean + non-lcc mean; realized telemetry
rates printed next to every F1.

## 8. Highest-risk assumptions

- **R1:** "scalar ties EC at 2.0–2.5" is the untested extrapolation this study rests
  on. For: TQ 45.73@2.125 is scalar-class. Against: TQ has per-token scales; A-static
  removes them. Mitigation: proxy admission before F1; A-gain fallback; 15-cell
  worst case.
- **R2:** decode-arm power — LongBench generations are short; W=32 leaves many decode
  tokens fp16. W=0 adversarial arm is the real test; claim worded as robustness.
- **R3:** B per-layer profile sharing penalty > 3% → B demoted to quality-only.
- **R4 (minor):** ω transfer to the INT ladder at 2.0 — rate may not bind; ω placed
  at tightest admitted rate by frozen rule.

## Amendments

- **A1 (2026-07-09, pre-P3, selection rows only).** P1 screening: family A fails
  G-P4 everywhere (0.0159 @2.0 vs bar 0.00274; gain variant WORSE — raw-norm
  rescaling through the non-orthogonal inverse distorts the Q-weighted direction),
  C fails (0.0218 @2.0 — page granularity insufficient at an INT ladder), D bases
  fail (oscar 0.0160, r_sym 0.0201 @2.0 — no promotion). Family B is the study:
  5x better than A at matched rate; with LM grids (`proflm`) it PASSES G-P4 @2.5
  (0.00226 <= 0.00233) and misses @2.0 by 12% (0.00306 vs 0.00274). Diagnosis: the
  w<=4 cap binds on leading qpca coords (B trails ecu ~2x exactly where the
  spectrum is steep). **Amendment: extend the family-B profile width set to
  {0, 2, 3, 4, 6}** — 32-coord blocks, <=8 profiles, 3 id bits, monotone rule,
  sink escape all unchanged; w=6 is a bulk width (top width keeps the covering
  uniform grid; LM LUTs now cover w in {2,3,4}). Kernel map unchanged: one more
  constexpr unpack width. A/px ladders stay {0,2,3,4} for comparability. No F1
  data was seen before this amendment (none exists yet).
- **A2 (2026-07-09, pre-P3, selection rows only).** Post-A1 screening: proflm
  passes G-P4 @2.5 with margin (0.00184 vs 0.00233, now beating rvq outright),
  misses @2.0 by 3.6% (0.00284 vs 0.00274) and @1.5 by 4.7%; the omega arm hurts
  the proxy at every rate (regime-law-consistent; static D stays frozen for
  screening). Two fit-time implementation details the registration never pinned
  are upgraded: (i) profile TUPLES are now selected by the MEASURED per-block
  quantizer distortion (LM grids for bulk widths, covering uniform top —
  matching the winning arm) instead of the 4^-w Gaussian proxy; (ii) the rung
  ladder targets densify around the operating points: [0,128,192,224,256,288,
  320,768] bits/token (the RDO mixes hull-adjacent rungs, so hull density at
  192/256/320 is what it can use). Codec, rates, gates, block structure all
  unchanged. One iteration only; the P2 freeze happens on whatever this yields.
