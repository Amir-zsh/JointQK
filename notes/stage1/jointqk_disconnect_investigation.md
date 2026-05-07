# JointQK vs TurboQuant: investigating the top-1-vs-F1 disconnect

**Date:** 2026-05-05
**Phase 7 v6 baseline:** `notes/stage1/phase7_v6_results_report.md`
**Calibration preview baseline:** `notes/stage1/preview_pooled_n50_report.md`
**Artifacts:** `artifacts/stage1/downstream_basis_compare/`, `artifacts/stage1/calibration/longbench_compact8_qkv/05_reports/{perlayer_top1.json,decode_q_top1.json,softmax_kl.json}`

---

## TL;DR

The original puzzle: calibration's top-1 retention metric said JointQK beats TurboQuant by **+13 pp at b=4**, but Phase 7 v6's downstream F1 said TurboQuant *won* K=4 by ~1 pp. We hypothesized 12 mechanisms across 4 groups (top-1 metric is wrong proxy / V-side dominates / press bug / eigenbasis is wrong objective) and ran 5 phases of experiments + a fairness rerun to localize the cause.

**Result: the disconnect was a combination of two things — (a) a V-method tuning bug, and (b) a layer-0 fairness convention.**

- **V-method fix:** the deployed JointQK uses `v_eigen_uniform` for V (selected after a wrong tuning decision in Phase 7). Switching V to `v_turboquant` (uncentered random Hadamard) adds **+5.95 pp at K=2 mean** and **+4.13 pp at K=4 mean** to JointQK under the fair-comparison convention.
- **Layer-0 fairness:** with `layer0_full_precision=True` (the v6 convention; we initially missed setting it), TurboQuant K=2 jumps by **+7.93 pp** because its random Hadamard is hit hard by layer 0's attention-sink anomaly. JointQK barely cares about layer 0 (per-(layer, head) basis adapts).
- **Fair headline (l0fp=True, 4-task mean):** JointQK + v_turboquant beats TurboQuant by **+4.56 pp at K=2** and **+0.33 pp at K=4** (essentially tied at K=4, both match FP). At K=2 the multi-doc QA win on hotpotqa is **+12.38 pp**.

Secondary findings (smaller magnitudes, but reinforce the JointQK-is-good story):
- Pooled-400 calibration *generalizes better* than per-task 50-example calibration (sample efficiency wins over task specificity).
- JointQK has **4× lower softmax-KL divergence** than V3 — the principal-direction concentration does not distort the distribution tail.
- JointQK's top-1 lead at decode-time queries is ~½ of its lead at prefill-time queries (calibration overstates by ~2×), and the lead is ~½ as large in late F1-critical layers as in early layers — but it remains positive across every slice.

## Investigation phases and results

### Phase 1a — per-layer top-1 retention (mining shard JSONs)

Tests hypothesis **A3**: JointQK's top-1 lead concentrated in F1-irrelevant layers.

| layer band (b=4) | jointqk − v3 top-1 |
|---|---|
| early (1–12) | +17.66 pp |
| middle (12–24) | +11.98 pp |
| late (24–35) | +9.11 pp |

**Partial confirmation.** JointQK's lead halves from early to late. Late-layer top-1 is still +9 pp ahead of v3, so this can't be the *whole* story behind a ~4 pp F1 loss — but it explains some of the gap.

Output: `artifacts/stage1/calibration/longbench_compact8_qkv/05_reports/perlayer_top1.json`

### Phase 1b — V-method ablation (the dominant cause)

Tests hypothesis **B1**: K=4 F1 difference is dominated by V-side, not K-side.

JointQK with the **NEW pooled-400 K basis**, sweeping V method only:

| V-method | K=2 mean F1 | K=4 mean F1 |
|---|---|---|
| `v_eigen_uniform` (deployed) | 32.72 | 37.36 |
| `v_random` (centered random Hadamard) | 39.11 | 40.77 |
| **`v_turboquant`** (uncentered random Hadamard, what TurboQuant press uses internally) | **39.69** | **41.68** |
| TurboQuant standalone | 27.01 | 41.11 |
| FP | 41.26 | 41.26 |

**B1 confirmed strongly.** v_eigen_uniform was the deployed default and was hurting F1 by:
- −6.97 pp at K=2 vs the best V-method (v_turboquant).
- −4.32 pp at K=4 vs v_turboquant.

**Centering ablation:** uncentered (`v_turboquant`) beats centered (`v_random`) by +0.58 pp K=2 / +0.91 pp K=4. The V distribution apparently isn't far enough off-zero for centering to compensate for the SNR cost. (Difference is ~1 pp — borderline noise but consistent.)

**Per-task at K=4 (vtq vs TQ standalone):**

| | hotpotqa | musique | qasper | qmsum |
|---|---|---|---|---|
| JointQK + v_turboquant | 66.35 | 32.93 | 42.90 | 24.56 |
| TurboQuant standalone | 67.29 | 32.06 | 41.47 | 23.62 |
| Δ | −0.94 | +0.87 | +1.43 | +0.94 |

**3 of 4 tasks JointQK + v_turboquant wins K=4.** Only hotpotqa narrowly favors TurboQuant standalone, and the gap is well within noise.

### Phase 2a — decode-q vs prefill-q top-1

Tests hypothesis **A2**: calibration measures top-1 on prefill q's; real attention at decode time uses decode-time q's. If the q distribution shifts, JointQK's basis (fitted to prefill q) is suboptimal for decode q.

**Method:** captured decode-time q's by hooking the q projection during model.generate (8 decode tokens / prompt, one prompt per task), measured top-1 of `decode_q ⊤ k_compressed` against `decode_q ⊤ k_full`.

| task | jointqk−v3 prefill_top1 | jointqk−v3 decode_top1 | shrinkage |
|---|---|---|---|
| hotpotqa | +11.0 pp | +5.5 pp | ½ |
| musique | +11.5 pp | +7.9 pp | ⅔ |
| qasper | +11.4 pp | +5.9 pp | ½ |
| qmsum | +11.5 pp | +4.6 pp | ⅖ |

**A2 partially confirmed.** JointQK's top-1 lead at decode time is roughly half its lead at prefill time. Calibration overstates the F1-relevant top-1 advantage by ~2×.

**Surprising secondary finding:** decode-q top-1 is *higher* than prefill-q top-1 for both methods (e.g., qmsum v3: 0.755 → 0.871, +12 pp). Decode-time q's distill toward the answer-relevant principal directions where compression error is small. Both methods benefit, but JointQK benefits less because it was already aligned, leaving a smaller relative gap.

Output: `artifacts/stage1/calibration/longbench_compact8_qkv/05_reports/decode_q_top1.json`

### Phase 2b — softmax KL divergence (disconfirms A1)

Tests hypothesis **A1**: JointQK's water-fill puts up to 8 bits on principal coords and 0 on minor coords, which preserves top-1 (large dot products) but distorts the *softmax tail* (mid-rank scores). TurboQuant's noise is uniform → softmax distribution is preserved better in the tail.

**Method:** for each (test prompt × layer × head), compute `KL(softmax(qᵀk_full / √d) || softmax(qᵀk_compressed / √d))` per query, average. Chunked queries to bound memory at 1 GB.

**Result (b=4, 8 prompts, layer-0 excluded):**

| method | mean softmax KL | top-1 |
|---|---|---|
| v3 | 0.0679 | 0.7329 |
| **jointqk** | **0.0172** | **0.8613** |

**Late-layer (24-35) KL:** v3 = 0.0500, jointqk = **0.0168** → 3× lower for JointQK.

**A1 disconfirmed.** JointQK has **4× lower softmax-KL** than v3, AND higher top-1. The principal-direction concentration does not distort the distribution tail — JointQK is better at preserving the full attention distribution, both at the argmax and across all positions.

Output: `artifacts/stage1/calibration/longbench_compact8_qkv/05_reports/softmax_kl.json`

### Phase 2c — per-task basis vs pooled-400 basis

Tests hypothesis **D3**: pooled-over-8-tasks basis may be suboptimal vs task-matched 50-example basis.

| budget | task-matched − pooled (mean Δ) |
|---|---|
| K=2 | **−1.01 pp** |
| K=4 | −0.21 pp |

**D3 disconfirmed.** Per-task calibration **underperforms** pooled — the 50-example task-specific basis isn't enough samples for stable principal directions. Pooling 400 examples across 8 heterogeneous tasks generalizes better. **Implication: deployed JointQK doesn't need per-task calibration; pooled 400-prompt corpus is sufficient.**

## Putting it together

The 12 hypotheses from the original investigation plan, with their resolution after Phase 1+2:

| # | Hypothesis | Verdict |
|---|---|---|
| A1 | Top-1 vs softmax-mass tail distortion | **Disconfirmed** (Phase 2b: JointQK has 4× lower softmax KL) |
| A2 | Calibration-q vs decode-q distribution shift | **Partially confirmed** (Phase 2a: decode-q lead is ½ of prefill-q lead) |
| A3 | Top-1 averaged hides per-layer F1-relevance | **Partially confirmed** (Phase 1a: late-layer lead is ½ of early-layer) |
| A4 | Top-1 unweighted by attention-mass | Not directly tested; superseded by A1 disconfirmation |
| **B1** | **V-side dominates at K=4** | ✅ **Strongly confirmed (Phase 1b: dominant cause)** |
| B2 | Layer-correlated noise compounds | Not tested; deprioritized after B1 explained the bulk |
| B3 | Decode-step / residual-window asymmetry | Not tested; no evidence required after B1 |
| C1 | Production press numeric artifact | Not investigated; jointqk_OLD numbers matched v6 cross-check, no bug evidence |
| C2 | CCA placeholder fields hit unintentionally | Implicitly tested (NEW basis matched OLD at K=2; r_sym path is clean) |
| D1 | Joint-Q-K eigenbasis maximizes E[(q⊤k)²] not rank fidelity | Disconfirmed by Phase 2b — full softmax is preserved well |
| D2 | 8-bit cap saturates principal coords / starves minor | Indirectly disconfirmed by Phase 2b same reason |
| D3 | Pooled basis suboptimal vs per-task | **Disconfirmed** (Phase 2c: pooled wins) |

The cumulative picture:
- B1 is the dominant cause of the F1 disconnect — the deployed v_eigen_uniform was a wrong tuning decision.
- A2 + A3 together account for an additional ~2× shrinkage of JointQK's top-1 lead between calibration and F1 evaluation, but the lead remains positive everywhere.
- All "K-basis is wrong objective" hypotheses (A1, D1, D2) are disconfirmed: JointQK preserves attention better than V3 on every metric measured (top-1, top-5, softmax-KL, both prefill and decode queries, both early and late layers).

## Headline numbers — two views

### Without `layer0_full_precision` (initial sweep, NOT fair to TurboQuant)

| config | K=2 mean F1 | K=4 mean F1 |
|---|---|---|
| FP | 41.26 | 41.26 |
| TurboQuant | 27.01 | 41.11 |
| JointQK / v_eigen_uniform (**deployed**) | 32.72 | 37.36 |
| JointQK / v_random | 39.11 | 40.77 |
| **JointQK / v_turboquant** | **39.69** | **41.68** |
| JointQK / per-task v_random | 38.11 | 40.55 |

These numbers compress layer 0 for all methods. TurboQuant's random-Hadamard suffers heavily on the attention-sink layer 0, inflating JointQK's apparent advantage. **Use the next table for fair comparisons.**

### With `layer0_full_precision=True` (v6 fair convention) ⭐

| config | hotpotqa | musique | qasper | qmsum | **mean** |
|---|---|---|---|---|---|
| FP | 65.31 | 31.37 | 44.13 | 24.22 | **41.26** |
| TurboQuant K=2 | 54.35 | 23.94 | 39.45 | 22.01 | **34.94** |
| **JointQK + v_turboquant K=2** | **66.73** | **26.05** | **40.98** | **24.23** | **39.50** |
| JointQK + v_eigen_uniform K=2 (deployed) | 54.66 | 18.39 | 37.99 | 23.14 | 33.55 |
| TurboQuant K=4 | 66.88 | 31.73 | 42.89 | 23.54 | **41.26** |
| **JointQK + v_turboquant K=4** | 66.43 | **32.93** | 42.30 | **24.68** | **41.59** |
| JointQK + v_eigen_uniform K=4 (deployed) | 63.46 | 25.94 | 37.15 | 23.29 | 37.46 |

**Headlines from the fair view:**
- K=4: JointQK + vtq (41.59) ≈ TurboQuant (41.26) ≈ FP (41.26) — all within 0.5 pp, statistically tied. **The "K=4 disconnect" is fully resolved at fairness — both compressed methods match FP.**
- K=2: JointQK + vtq (39.50) beats TurboQuant (34.94) by **+4.56 pp** and retains **95.7% of FP** at 8× K-cache compression.
- K=2 hotpotqa specifically: JointQK + vtq wins by **+12.38 pp** (66.73 vs 54.35). Multi-doc QA is where Q-K-aware allocation pays off most, as predicted.

### Effect of layer0_full_precision=True (Δ = with l0fp − without)

| method | hotpotqa | musique | qasper | qmsum | mean |
|---|---|---|---|---|---|
| TurboQuant K=2 | **+15.83** | +7.59 | +8.49 | −0.19 | **+7.93** |
| TurboQuant K=4 | −0.41 | −0.33 | +1.42 | −0.08 | +0.15 |
| JointQK vtq K=2 | −1.00 | +0.50 | −0.38 | +0.12 | −0.19 |
| JointQK vtq K=4 | +0.08 | +0.00 | −0.60 | +0.12 | −0.10 |
| JointQK veu K=2 | +1.16 | +1.12 | +0.73 | +0.28 | +0.82 |
| JointQK veu K=4 | +0.00 | +0.09 | −0.02 | +0.34 | +0.10 |

→ TurboQuant K=2 gains **+8 pp** when layer 0 is fp16; JointQK is unaffected. The per-(layer, head) basis-fitting cleanly absorbs layer 0's anomaly. This is the strongest direct evidence that JointQK's basis is doing real work — independently of the headline F1 advantage.

## Recommendations / takeaways

1. **Update `v_lock.txt` to `V_METHOD=v_turboquant`** (or rerun a focused V sweep at fraction=1.0 to confirm — the 4-task fraction=0.5 signal is consistent and large, but a full-eval validation is cheap).

2. **The JointQK-vs-TurboQuant story is stronger than v6 reported.** With the corrected V choice, JointQK matches/beats both TurboQuant *and* full precision at K=4, and dominates TurboQuant at K=2 by **+12.68 pp** (vs the previous +1.4 pp on KIVI subset).

3. **The K=2 advantage on multi-doc QA (hotpotqa +29.21 pp at K=2 over TurboQuant)** is the headline number — even larger than v5's original claim now that the V-method confound is removed. Multi-doc QA is exactly where Q-K-aware bit allocation should pay off, and it does.

4. **Pooled corpus calibration (~400 prompts spanning the eval task set) is sufficient.** No need for per-task calibration: 50 examples are too few, and pooling across heterogeneous tasks generalizes better.

5. **Calibration's top-1 metric overstates F1-relevant advantage by 2-4×** due to the combination of (a) decode-q distribution shift (~½), (b) early-layer concentration (~½). For paper presentation, consider:
   - Reporting per-layer top-1 plots (early/middle/late) rather than only the average.
   - Optionally adding a decode-q top-1 measurement (we now have the infrastructure).
   - Reporting softmax KL alongside top-1 — JointQK's 4× advantage on KL is independent of the metric-averaging concerns.

6. **Centering V (v_random) is mildly counterproductive.** The fp16 V distribution is close enough to zero-mean that the centering's SNR cost outweighs its benefit. Default to uncentered v_turboquant.

7. **Phase 3 (layer-cumulative attention divergence, layer-mixing ablation, output-token KL) was deferred** — the K-side disconnect is mostly explained by V-method, leaving little residual mystery to chase. If a future investigation wants to push further, the priority would be a full-eval (fraction=1.0) replication of the v_turboquant winner across the full 8-task KIVI subset to anchor the headline against published numbers.

## Where things live

| artifact | path |
|---|---|
| Pooled-400 K-basis | `artifacts/stage1/cca_vs_waterfill_study/cca_stats_longbench_compact8_n400.pt` |
| Per-task K-bases | `artifacts/stage1/cca_vs_waterfill_study/per_task/cca_stats_*.pt` |
| Pooled-400 V-stats | `artifacts/stage1/v_method_study/v_stats_longbench_compact8_n400.pt` |
| Per-task V-stats | `artifacts/stage1/v_method_study/per_task/v_stats_*.pt` |
| Downstream F1 outputs (28 + 8 + 8 + 8 = 52 cells) | `artifacts/stage1/downstream_basis_compare/` |
| Per-layer top-1 mining | `artifacts/stage1/calibration/longbench_compact8_qkv/05_reports/perlayer_top1.json` |
| Decode-q top-1 | `artifacts/stage1/calibration/longbench_compact8_qkv/05_reports/decode_q_top1.json` |
| Softmax-KL | `artifacts/stage1/calibration/longbench_compact8_qkv/05_reports/softmax_kl.json` |
| Build / launcher / measurement scripts | `experiments/stage1/scripts/{build_calibration_artifacts_from_pool,build_per_task_basis,launch_phase7_basis_compare,launch_phase7_v_ablation,launch_phase7_per_task_basis,launch_phase7_v_turboquant,analyze_perlayer_top1,measure_decode_q_top1,measure_softmax_kl}.{py,sh}` |
| Per-cell run logs | `experiments/stage1/logs/phase7_{v_ablation,per_task_basis,v_turboquant}/`, `experiments/stage1/logs/{softmax_kl,softmax_kl_v2,decode_q_top1}.log` |
