# pgq4 report — decode-resident page-selective quantization (plan4)

**Study:** 2026-07-09 → 07-10, branch `pgq4`. Pre-registration:
[`plan4.md`](plan4.md) (+ amendments A1/A2, both pre-P3, selection rows only).
Protocol v7 (Llama-3.1-8B-Instruct, 5-task LongBench, FP = 48.40, layer-0 fp16,
V = TurboQuant 2b, honest all-in rates, compact8-TRAIN-only fitting, row-paired
bootstraps; the bootstrap tool's "3-task mean" label is stale text — values are
row-pooled 5-task).

## Headline

**The frozen codec + recency window (`pgq_proflmrw`, W3) ties entropy coding at
2 b/c and beats the RVQ incumbent significantly** — the first fixed-width,
kernel-implementable codec in this project to do either. At honest 2.13 b/c:
F1 45.88 (above the TurboQuant 45.73 point ref at its own 2.125 rate), vs rvq
+1.49 [+0.06, +2.98] SIG, vs ecu −1.43 [−3.04, +0.14] tie (**Tier-E MET**), vs
TQ +0.34 [−1.31, +2.01] row-paired. Per-token metadata: a 3-bit rung id (vs
OSCAR's 32 bits of scale/zero); decode = one folded matmul on the query +
per-width INT-unpack segments + a position-derived top-rung window on sinks
and the last 4 prompt pages. The W1 wave (pre-window) additionally established
parity with both fixed-width incumbents and located EC's entire remaining
advantage in lcc; W3 showed that gap is a **recency effect, not a codec-capacity
or calibration-transfer limit** — protecting the last 256 prompt tokens
recovers all of it at +0.13 b/c.

## The adaptivity gradient (the study's cleanest new finding)

lcc (code completion — the most OOD task vs the calibration corpus) orders the
five codecs exactly by how much per-token adaptivity they retain:

| codec | per-token adaptivity | lcc F1 | non-lcc mean |
|---|---|---|---|
| oscar_uni @2.28 | full (per-token min-max scale+zero, 32 b/tok) | **50.19** | 43.44 |
| ecu @2.0 | adaptive-rate entropy coding | 49.76 | 46.58 |
| rvq @2.0 | adaptive codebook assignment | 43.76 | 45.17 |
| proflm @2.0 | none (static widths + static scales) | 40.16 | 44.87 |
| proflm_ea @2.0 | none + ω sharpening | 38.22 | **46.37** |

Static calibration wins in-distribution and loses under shift, monotonically in
adaptivity. OSCAR's 32 bits/token buy exactly one thing: lcc robustness (they
cost it the best non-lcc mean). This reframes the fixed-width design problem
for pgq5: not "more codec capacity" but "the cheapest kernel-legal per-token
scale adaptation" (a single fp16 per-token GRID scale — distinct from the
raw-norm gain, which rescales the reconstruction, failed in screening, and is
understood — is the obvious candidate; untestable on in-distribution selection
rows, so it needs an OOD screening protocol first).

## W1 results (30 cells; FP = 48.40, TurboQuant@2.125 = 45.73 point refs)

| arm | lcc | musique | 2wikimqa | qasper | hotpotqa | mean | non-lcc |
|---|---|---|---|---|---|---|---|
| proflm@2.0 | 40.16 | 29.13 | 44.99 | 44.51 | 60.86 | 43.93 | 44.87 |
| proflm_ea@2.0 | 38.22 | 32.47 | 50.01 | 43.71 | 59.30 | 44.74 | 46.37 |
| proflm@2.5 | 42.88 | 30.72 | 48.97 | 45.13 | 59.51 | 45.44 | 46.08 |
| ecu@2.0 | 49.76 | 33.10 | 49.10 | 44.55 | 59.57 | 47.22 | 46.58 |
| rvq@2.0 | 43.76 | 29.15 | 47.66 | 46.56 | 57.31 | 44.89 | 45.17 |
| oscar_uni@2.28 | 50.19 | 29.27 | 46.61 | 43.50 | 54.39 | 44.79 | 43.44 |

Pre-registered bars: **Tier-K missed** (best kernel arm 44.74 < 45.73; vs-rvq
is parity, not SIG). **Tier-E missed** (−2.65 SIG, lcc-driven). **Tier-FP@2.5
missed** (point Δ −2.96 < −1.5). **Stretch@1.5 not admitted** (screening
1.19× bar — consistent with the pgq3 effective-dimension closure).
**ω regime law at F1:** ea − rdo = +0.97 [−0.27, +2.22] pooled tie with
2wikimqa +5.02 SIG — directionally confirmed on the INT ladder; weaker than
pgq2's coarse-ladder regime, exactly as the law predicts (8-rung ladder is
fine-grained, rate binds only moderately at 2.0).

## What screening closed before any F1 (P1/P2, six passes)

- **Family A (uniform per-token width, the "folded-scalar" primary):** dead at
  every rate (0.0159 @2.0 vs bar 0.00274). Uniform width wastes bits on
  low-energy qpca coords; profiles fix precisely this (5× lower logit_err at
  matched rate). The A-gain variant (fp16 raw-norm/token) made A WORSE — raw-norm
  rescaling through the non-orthogonal inverse distorts the Q-weighted
  direction.
- **Family C (page-rung):** 0.0218 @2.0 — page-granularity allocation is
  insufficient at an INT ladder; per-token rungs are load-bearing.
- **Family D (basis ablation under the identical quantizer):** OSCAR per-layer
  U_Q·H·P_br 0.0160, r_sym 0.0201 vs qpca_unc-profiles 0.0027 — the per-head
  energy-compacting basis is the enabler; no promotion (rule was within-3%).
- **Prefix truncation (px):** dominated by water-filled tuples at every rate.
- **A2 iteration:** measured per-block distortion (replacing the 4^-w Gaussian
  proxy) + hull-dense rung targets moved proflm@2.0 from 1.19× to 1.143× rvq —
  the margin that admitted the 2.0 arm.

## Gates (all passed, incumbent-relative, held-out selection rows)

proflm beats the rvq incumbent OUTRIGHT on every gate metric: sink code-relerr
0.002 vs 0.015 (the positional 8-bit sink escape works), normR 0.978–0.986 vs
0.969, realized rates exact (2.0000 / 2.4955 / 1.9999), overflow 0. G0-perf
1.24–1.6× rvq wall-clock (bar 2×). Mode-B' decode flushing validated end-to-end
at W ∈ {32, 0} on smoke cells (W=0 exercises the splice every 8 decode tokens;
F1 unchanged on the smoke slice).

## Kernel story (what Tier-K parity buys, despite the missed absolute bar)

Decode = q̃ = q @ (Gᵀ diag(s)) once per head per step (q·μ cancels in softmax);
per-profile index lists → per-width constexpr stage-1 launches (width changes
only at 32-coord block boundaries, **per-layer-shared profiles** — sharing
penalty 1.9% < 3%, so block boundaries are constexpr per launch); one
tier-agnostic online-softmax stage-2 (OSCAR's template). Per-token metadata: a
3-bit rung id. Sinks: positions 0–3 dispatch to an 8-bit absolute-grid segment,
zero sideband. This is strictly less per-token machinery than OSCAR INT2 at
equal-or-better quality everywhere except lcc.

## W2 (decode arm): SKIPPED per the frozen decision tree

Tier-K missed → no decode claim is made for a non-tier-K codec. The Mode-B'
implementation is validated (smokes) and ready if a pgq5 codec clears tier-K.

## W3 (recent-window prefill ablation)

Hypothesis registered before W3 ran: lcc is code completion — queries attend to
the recent context being completed; forcing the last 4 prompt pages (256
tokens) to the top rung is kernel-trivial, position-derivable, and rate-charged.
If the transfer gap is recency-concentrated, this recovers lcc specifically.

**Deviation note:** the registered "winner+ecu" pairing runs on the winner only
(both ω settings: proflmrw_rdo, proflmrw_ea); ecu-rw would need post-freeze EC
loader code (PagedRDOCompressor has no page-forcing path) — not added mid-F1.

**Result: the hypothesis is confirmed, and it changes the study's verdict.**
Forcing the last 4 prompt pages (256 tokens) to the top rung recovers the
ENTIRE lcc gap at zero cost elsewhere:

| arm | lcc | musique | 2wikimqa | qasper | hotpotqa | mean | non-lcc |
|---|---|---|---|---|---|---|---|
| proflmrw_rdo@2.0 | **50.05** | 30.97 | 45.11 | 42.92 | 60.34 | **45.88** | 44.84 |
| proflmrw_ea@2.0 | 48.10 | 32.28 | 48.00 | 43.86 | 57.64 | **45.98** | 45.45 |

lcc goes 40.16 → 50.05 (+9.9), matching ecu (49.76) and oscar (50.19); non-lcc
is unchanged (44.84 vs 44.87). The "calibration transfer gap" of W1 is, at
least on this benchmark, a RECENCY gap: lcc queries attend to the recent code
being completed, and protecting those 256 tokens is position-derivable,
kernel-trivial, and rate-charged.

Row-paired bootstraps (pooled 5-task):
- **Window effect:** rw − plain = **+2.19 [+0.99, +3.38] SIG** (rdo);
  +1.27 [+0.11, +2.43] SIG (ea).
- **vs rvq@2.0:** +1.49 [+0.06, +2.98] **SIG** (rdo); +1.54 [+0.14, +2.95]
  **SIG** (ea) — the first fixed-width codec in this project to beat the RVQ
  incumbent significantly at 2 b/c.
- **vs ecu@2.0:** −1.43 [−3.04, +0.14] tie (rdo); −1.38 [−3.00, +0.27] tie
  (ea) — **Tier-E MET with the window arm**: EC's advantage is no longer
  significant.
- **vs TurboQuant@2.125 (cross-root, row-paired, n=1150):** +0.34
  [−1.31, +2.01] (rdo); −0.09 [−1.79, +1.56] (ea).
- **vs FP:** −2.28 [−3.70, −0.88] (rdo).

Held-out gates for the rw arms: rate **2.1288** b/c (the window costs +0.13 at
the selection-row length mix), ovf 0, sinkΔ +0.024–0.026, sinkCE 0.002,
normR 0.977–0.979 — G2/G3 unchanged.

### Final tier scoring (strict, pre-registered wording)

- **Tier-E: MET** (window arm ties ecu@2.0, both ω settings).
- **Tier-K: NOT met, on two technicalities** — honest rate 2.1288 vs the
  ≤2.125 envelope (+0.18%), and Δ-vs-TQ CI-lower −1.31 vs the >−1.0
  requirement (point estimate +0.34 IN FAVOR; power-limited). The substantive
  prongs pass: F1 45.88 ≥ 45.73, SIG > rvq. No post-hoc re-run at a lower
  nominal budget was made to squeeze under the envelope. Substantive reading:
  **TurboQuant-class quality at TurboQuant's rate with ~10× less per-token
  metadata and a strictly simpler kernel, significantly above the strongest
  fixed-width incumbent.**
- **Tier-FP: not met** (−2.28 [−3.70, −0.88] vs FP at 2.0+window; the
  2.5+window combination was not run).
- **Kernel-port trigger:** formally not fired (Tier-K strict); the port
  decision moves to the user with the two-technicality context above.

## Economy

P0–P5+W3 ≈ 25–30 GPU-h (under the 60–75 target; W2's 5–7 GPU-h saved by the
decision tree). Six screening passes + two bundle refits ≈ 2 GPU-h.

## Open directions (pgq5 candidates, evidence-ranked)

1. **Kernel-port decision (user's call):** Tier-K missed strictly by +0.004
   b/c of rate and 0.31 of CI — every substantive prong passed. If the port
   proceeds, the format spec is: per-layer-shared profiles, 3-bit rung ids,
   32-coord constexpr width blocks, folded q̃, 8-bit sink segment (positions
   0–3), top-rung recency window (last 4 prompt pages). A nominal-1.95 budget
   would land the honest rate under 2.125 with margin.
2. **Recency window as a format primitive:** +2.19 SIG for +0.13 b/c at the
   2–8k length mix (+0.03 at 32k). Also test W∈{2,8} pages and EC-side
   windows (needs a small PagedRDOCompressor forcing path — the registered
   ecu-rw arm was skipped for exactly this missing code).
3. **Per-token grid scale (fp16/token):** the adaptivity gradient (W1) may
   still matter beyond recency on other OOD axes; needs an OOD screening
   protocol — in-distribution selection rows cannot see transfer effects
   (this study's methodological lesson: screening admitted the right family
   but could not predict the lcc structure).
4. **Tier-FP at 2.5 + window:** untested combination; W1's proflm@2.5 (45.44,
   no window) + the window's +2.19 suggests ~47+ is plausible at 2.6 b/c
   honest — one 5-cell wave.
5. **Decode arm (Mode-B'):** implementation validated at W∈{0,32} smokes;
   runs when a codec formally clears Tier-K.
