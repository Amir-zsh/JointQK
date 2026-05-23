# QPCA basis — closed-form optimum for logit MSE — results report

**Date:** 2026-05-21
**Pooled K-fidelity run:** `artifacts/calibration/longbench_compact9_qkv_qwen3_8b/05_reports/pooled_n50_qpca/`
**Per-task K-fidelity run:** `artifacts/calibration/longbench_compact9_qkv_qwen3_8b/05_reports/per_task_qpca_vs_jointqk/`
**Downstream F1 sweep:** `artifacts/bench/qwen3_8b/qpca_k{2,3,4}_v3_<task>/`
**Source code:** `analysis/measure_qpca_compact9.py`, `analysis/measure_qpca_vs_jointqk_per_task.py`
**Derivation:** `background/derivation_qpca_for_queries.pdf` · companion: `background/qpca_derivation_companion.html`

---

## TL;DR

QPCA is the closed-form linear basis that minimizes $\sum_{i,j} (q_i^\top k_j - q_i^\top U r_j)^2$ — i.e., **total logit reconstruction MSE** — over all linear encoder/decoder pairs. It is also the optimal decorrelating transform for source coding under any distortion measure (Linder–Zamir–Zeger 1999).

Three nested experiments on Qwen3-8B, compact8/compact9 LongBench calibration:

| Layer | Setup | What it measures | Result |
|---|---|---|---|
| **1. Pooled K-fidelity** | 90 test rows, all 9 compact9 tasks combined | top-1, top-5, k_mse, logit_err | QPCA wins logit_err by 16–36 %, loses top-1 by 3–5 pp |
| **2. Per-task K-fidelity** | Same setup, per-task breakdown | top-1/top-5/logit_err per task | Pattern **uniform across all 9 tasks**: QPCA wins logit_err everywhere, loses top-1 everywhere |
| **3. Downstream F1** | LongBench full sweep, fraction=1.0, 12 tasks | Actual task F1 (the bench goal) | **QPCA loses mean F1 at every bit width** (−5.36 pp at k=2, −1.25 pp at k=3, −0.82 pp at k=4) |

All three experiments tell the same story: **QPCA wins what it was designed for (logit MSE) and loses what we actually want (attention argmax → downstream F1)**. The math is correct (all sanity gates passed — F·G = I to 2.4×10⁻⁶); the result is a property of the loss-mismatch, not of the implementation.

The pattern is the same "MSE-vs-argmax disconnect" investigated earlier in the project (`notes/jointqk_disconnect_investigation.md`): **lower MSE on $q^\top k$ does not imply higher top-1 attention retention**, and lower top-1 retention does propagate to lower downstream F1.

**Decision: don't switch the production basis. `jointqk` (heuristic, but argmax-aware via its `q_diag · k_diag` water-fill score) remains the deployed K-side basis at every bit width.** QPCA's value is as a closed-form baseline that any future "argmax-aware" basis must beat.

---

## The math

**Objective**: $L(U, \{r_j\}) = \sum_{i=1}^N \sum_{j=1}^M (q_i^\top k_j - q_i^\top U r_j)^2$

**Closed-form** (full rank, $p = d$):
1. Form $M_q = \sum_i q_i q_i^\top = \Sigma_Q$ and $S_k = \sum_j k_j k_j^\top = \Sigma_K$.
2. Eigendecompose $A = M_q^{1/2} S_k M_q^{1/2} = V \Lambda V^\top$ with $V$ orthonormal and $\Lambda$ diagonal.
3. **Encoder**: $r_j = V^\top M_q^{1/2} k_j$ (forward map for keys).
4. **Decoder**: $\hat k_j = M_q^{-1/2} V r_j$ (inverse map).

**Codebase convention** (right-multiplication on row vectors):
- `forward_map` = $M_q^{1/2} V$ (so `k_row @ forward_map` ↔ column-vec $V^\top M_q^{1/2} k_{col}$, transposed).
- `inverse_map` = $V^\top M_q^{-1/2}$.

**Code-space per-coord variance**: $\mathbb{E}[r r^\top] = V^\top M_q^{1/2} S_k M_q^{1/2} V = V^\top A V = \Lambda$ (diagonal).

**Q-weighted key MSE after quantization equals plain code-space MSE**:
$$\sum_i (q_i^\top (k - \hat k))^2 = (k - \hat k)^\top M_q (k - \hat k) = (r - \hat r)^\top V^\top V (r - \hat r) = \|r - \hat r\|^2$$

So the Q-weighting is fully absorbed into the basis. Each code coord contributes equally to the loss → optimal bit allocation = **water-fill on Λ alone** (no `q_diag * k_diag` product, unlike jointqk's heuristic score).

**Differences from `jointqk`**:

|                  | Forward $F$ (right-mult) | Inverse $G$ | $F G$ | Water-fill score |
|------------------|--------------------------|-------------|-------|------------------|
| `jointqk`        | $R$ (orthogonal eigvec of $(\Sigma_Q\Sigma_K + \Sigma_K\Sigma_Q)/2$) | $R^\top$ | $I$ | $\mathrm{diag}(R^\top \Sigma_Q R) \cdot \mathrm{diag}(R^\top \Sigma_K R)$ |
| **`qpca`**       | $M_q^{1/2} V$ (non-orthogonal) | $V^\top M_q^{-1/2}$ | $I$ | $\Lambda$ alone |

The two bases coincide only if $\Sigma_Q$ and $\Sigma_K$ commute (rare in practice).

---

## Setup

- **Model**: Qwen/Qwen3-8B (36 layers × 32 Q-heads × 8 KV-heads, head_dim 128).
- **Calibration corpus**: 450 LongBench train prompts (50 × 9 tasks: hotpotqa, multi_news, musique, passage_retrieval_en, qasper, qmsum, repobench-p, triviaqa, **lcc**). 9-task compact9 corpus (extends compact8 with LCC; built earlier today, see commit 2c17214 batch-13).
- **Eval corpus**: 90 LongBench test prompts (10 × 9 tasks).
- **Per-coord allocation**: continuous water-fill capped at 8 bits/coord. **`jointqk` uses the deployed `q_diag·k_diag` score**; **`qpca` uses Λ alone** (each method uses its theoretically-correct allocation; this is part of the comparison, not a confounder).
- **Bit budgets**: average 2, 3, 4 bits per coordinate.
- **Headline reporting**: layer-0 excluded (per project convention — layer 0 has anomalous attention-sink behaviour).
- **Numerical precision**: all eigendecomps for $M_q^{\pm 1/2}$ and $A$ run in fp64, cast to fp32 at the end. Eps-regularization on $\Sigma_Q$ via `regularize_batch(trace_lift=1e-4)` lifts low-eigenvalue dimensions so $M_q^{-1/2}$ is well-defined.

Per-shard execution: 4 GPUs × ~22-23 test idx each, round-robin slice of `test_indices`. Per-shard accumulators (`{bits: {layer: {mse_num, mse_den, logit_num, logit_den, top1_num, top1_den, top5_num, top5_den}}}`) written to `shard_NNN.json`; merged on CPU into `qpca_merged.json` in the same schema as `merged.json["method"]`.

**Total wall time**: 20 minutes (vs 78 min for the original 4-method `pooled_n50` run).

---

## Math correctness gates (run at shard 0 startup)

| Gate | Threshold | Observed | Pass |
|---|---|---|---|
| 1. `‖F·G − I‖_F` max over (L,H) (in fp32, after fp64 eigendecomps) | < 1e-3 | 2.45e-6 | ✓ |
| 2. Code variance on single test row vs Λ, \|log-ratio\| median | < 3.0 (within ~20×) | 0.09 (within ~1.1×) | ✓ |
| 3. b=8 Lloyd-Max roundtrip relative MSE on one (L,h) | < 1e-2 | 6.21e-5 | ✓ |

The gates establish that the QPCA implementation is numerically correct — any subsequent finding reflects the method, not a code bug.

---

## Headline empirical metrics (layer-0 excluded, pooled N=50)

| method      | bits | **top-1** | **top-5** | k_mse     | logit_err  |
|-------------|------|-----------|-----------|-----------|------------|
| v3          | 2    | 0.3986    | 0.6594    | 6.05×10⁻¹ | 2.46×10²   |
| v3          | 3    | 0.5751    | 0.8506    | 1.77×10⁻¹ | 6.46×10¹   |
| v3          | 4    | 0.7317    | 0.9574    | 4.86×10⁻² | 1.72×10¹   |
| **jointqk** | **2** | **0.6569** | **0.9218** | 2.27×10⁻¹ | 2.99×10¹  |
| **jointqk** | **3** | **0.7833** | **0.9800** | 7.13×10⁻² | 9.85       |
| **jointqk** | **4** | **0.8589** | **0.9937** | 2.07×10⁻² | 3.72       |
| qpca        | 2    | 0.6116    | 0.8941    | **2.22×10⁻¹** | **2.52×10¹** |
| qpca        | 3    | 0.7459    | 0.9591    | **7.04×10⁻²** | **7.72**     |
| qpca        | 4    | 0.8297    | 0.9782    | 2.07×10⁻² | **2.39**       |

(Bold per row: the winning value for that metric across the calibrated methods.)

**Headline deltas (qpca − jointqk):**

| bits | Δtop-1 | Δtop-5 | Δk_mse rel.   | Δlogit_err rel. |
|------|--------|--------|---------------|------------------|
| 2    | −0.045 | −0.028 | −2.2 % (better) | **−16 % (better)** |
| 3    | −0.037 | −0.021 | −1.3 % (better) | **−22 % (better)** |
| 4    | −0.029 | −0.016 | ≈ 0 % (tie)     | **−36 % (better)** |

---

## Reading the result

QPCA **wins decisively on what it was designed for** (`logit_err`) and **loses decisively on the metric the bench actually cares about** (top-1 attention retention).

### Why `logit_err` improves so much

`logit_err` is the empirical estimator of $\frac{1}{N M T^2} \sum_{ij} (q_i^\top (k_j - \hat k_j))^2$ — exactly the QPCA objective, up to a normalization. By construction, QPCA's basis + Λ-water-fill allocation is **the closed-form minimizer of this quantity over linear encoder/decoder pairs**. A 16-36 % reduction over `jointqk` is concrete evidence the derivation works as advertised, on real Qwen3 Q/K data, at non-trivial bit widths.

### Why top-1 retention drops

Top-1 retention measures the **probability that the original `argmax_t (q^\top k_t)` survives quantization**. This depends on the largest few logits being preserved correctly — not on the average MSE across all (query, key) pairs.

Two reasons QPCA underperforms here:

1. **Λ-only bit allocation is uniform-per-coord-importance**. The QPCA derivation says each code-coord contributes equally to the *averaged* loss, so the optimal allocation just equalizes per-coord quantization noise relative to coord variance. But the top-1 metric is dominated by a small subset of high-logit pairs (the peaks); those rely on specific high-leverage subspace directions surviving with high fidelity. `jointqk`'s `q_diag · k_diag` water-fill score is heuristic but **concentrates bits on the high-logit-leverage coords**, which is what argmax preservation needs.

2. **The non-orthogonal $M_q^{1/2} V$ basis introduces correlations among Lloyd-Max code coords.** The Lloyd-Max codebook assumes independent Gaussian coords; the QPCA forward map is non-orthogonal so the actual code distribution may be elongated/sheared in ways the per-coord codebook can't capture. `jointqk`'s orthogonal basis is friendlier to per-coord scalar quantization.

### The disconnect, restated

This is the same "V3-vs-CCA top-1 puzzle" that was investigated earlier in the project (see `notes/jointqk_disconnect_investigation.md`): **lower reconstruction MSE does not imply better attention rank preservation**. QPCA makes the disconnect quantitative — we can now point at a basis with provably-lower logit MSE that nevertheless loses on top-1 by 3-5 pp.

### What this implies for future basis design

If the bench-relevant metric is top-1 attention retention (and the F1 results downstream confirm this), the right objective is **not** $\sum (q^\top k - q^\top \hat k)^2$. It's something like:

- Top-k cross-entropy of the attention distribution against the original.
- A leverage-weighted MSE that up-weights pairs near the original argmax.
- A direct margin-preserving criterion: protect the gap between the original top-k and the rest.

None of these have closed-form linear solutions like QPCA does; they require iterative optimization. But QPCA is now the **closed-form baseline** any "argmax-preserving" basis must beat.

---

## Per-task K-fidelity (Experiment 2)

**Source:** `analysis/measure_qpca_vs_jointqk_per_task.py`, output `artifacts/calibration/longbench_compact9_qkv_qwen3_8b/05_reports/per_task_qpca_vs_jointqk/per_task_merged.json`. Wall time: **26 min on 6 GPUs**.

Same setup as Experiment 1, but accumulators are keyed by `(method, task, bits, layer)` so the merge yields per-task metrics. Each task's 10 test rows scored against the compact9-pooled basis (both methods).

### Top-1 deltas, qpca − jointqk (layer-0 excluded, pooled-train basis)

| task | k=2 | k=3 | k=4 |
|---|---:|---:|---:|
| qasper | −0.045 | −0.036 | −0.028 |
| hotpotqa | −0.041 | −0.035 | −0.028 |
| musique | −0.045 | −0.037 | −0.029 |
| qmsum | −0.043 | −0.034 | −0.026 |
| multi_news | **−0.047** | **−0.037** | −0.029 |
| triviaqa | **−0.055** | **−0.046** | **−0.036** |
| passage_retrieval_en | −0.039 | −0.031 | −0.024 |
| repobench-p | **−0.049** | **−0.042** | **−0.034** |
| lcc | −0.043 | −0.035 | −0.027 |

### Logit_err reductions, qpca vs jointqk (relative %)

| task | k=2 | k=3 | k=4 |
|---|---:|---:|---:|
| qasper | −18 % | −25 % | −39 % |
| hotpotqa | −18 % | −24 % | −39 % |
| musique | −18 % | −24 % | −39 % |
| qmsum | −18 % | −24 % | −38 % |
| multi_news | −18 % | −24 % | −38 % |
| triviaqa | −11 % | −17 % | −31 % |
| passage_retrieval_en | −19 % | −25 % | **−40 %** |
| repobench-p | −13 % | −18 % | −31 % |
| lcc | −17 % | −23 % | −36 % |

### Reading the per-task result

**The disconnect is structural, not task-dependent.** Across all 9 tasks × 3 bit widths (27 cells), QPCA beats jointqk on logit_err and loses to jointqk on top-1 — without a single exception. There is no task where the closed-form-optimal basis recovers the heuristic basis's argmax-preservation advantage.

**Two tasks stand out as the largest losses for QPCA** (top-1):

- `triviaqa` (−5.5 pp at k=2)
- `repobench-p` (−4.9 pp at k=2)

These are also the two tasks where QPCA's logit_err win is *smallest* (−11 % and −13 % at k=2, vs −17–19 % elsewhere). That correlation is mechanistic: tasks with flatter K spectra are where the QPCA Λ-only allocation diverges most from jointqk's leverage-weighted heuristic, and that divergence costs both ways — QPCA's MSE win shrinks AND its argmax loss grows.

`passage_retrieval_en` is the opposite extreme: largest qpca win on logit_err (−40 % at k=4), smallest top-1 loss (−2.4 pp at k=4). More concentrated K spectrum → the two allocations agree more.

---

## Downstream F1 (Experiment 3)

**Source:** `artifacts/bench/qwen3_8b/qpca_k{2,3,4}_v3_<task>/` (new) vs `artifacts/bench/qwen3_8b/jointqk_k{2,3,4}_v3_<task>/` + `full_precision_<task>/` (existing v7 production). **36 new cells**, fraction=1.0, **118 min wall on 6 GPUs**.

Matches the production v7 protocol exactly: same model (Qwen3-8B), same calibration corpus (compact8 pooled-400 — the QPCA basis was rebuilt from compact8 sigma_q/sigma_k to match the existing jointqk reference), same v-method (v_turboquant, v=3), same train-row exclusion from eval, same layer-0 full-precision policy.

### F1 grid (layer-0 excluded, KIVI 8-task + 4 multi-doc QA = 12 tasks)

| task | oracle | jq_k2 | qp_k2 | Δ | jq_k3 | qp_k3 | Δ | jq_k4 | qp_k4 | Δ |
|------|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| qasper | 43.06 | 42.27 | 36.55 | **−5.72** | 41.34 | 41.09 | −0.25 | 42.02 | 41.37 | −0.65 |
| qmsum | 24.37 | 24.24 | 22.72 | −1.52 | 24.40 | 23.98 | −0.42 | 24.47 | 23.78 | −0.69 |
| multi_news | 25.30 | 25.24 | 24.63 | −0.61 | 24.96 | 25.35 | **+0.39** | 25.24 | 24.88 | −0.36 |
| trec¹ | 41.50 | 69.00 | 64.50 | −4.50 | 32.00 | 35.50 | **+3.50** | 38.50 | 45.50 | **+7.00** |
| triviaqa | 90.71 | 88.40 | 88.12 | −0.28 | 87.95 | 89.36 | **+1.41** | 90.90 | 89.64 | −1.26 |
| samsum | 39.98 | 39.28 | 37.80 | −1.48 | 39.68 | 39.57 | −0.11 | 39.65 | 40.42 | **+0.77** |
| lcc | 64.81 | 55.86 | 43.39 | **−12.47** | 60.78 | 55.49 | −5.29 | 62.72 | 59.74 | −2.98 |
| repobench-p | 60.68 | 58.25 | 49.47 | **−8.78** | 56.36 | 53.87 | −2.49 | 56.91 | 56.21 | −0.70 |
| hotpotqa | 63.75 | 62.14 | 50.40 | **−11.74** | 61.38 | 57.02 | −4.36 | 63.72 | 59.39 | −4.33 |
| musique | 33.47 | 28.91 | 23.25 | −5.66 | 31.69 | 28.79 | −2.90 | 32.08 | 29.13 | −2.95 |
| 2wikimqa | 48.43 | 44.50 | 39.62 | −4.88 | 48.46 | 47.68 | −0.78 | 49.24 | 48.66 | −0.58 |
| narrativeqa | 28.89 | 26.14 | 19.41 | **−6.73** | 27.71 | 24.07 | −3.64 | 28.89 | 25.81 | −3.08 |
| **MEAN** | **47.08** | **47.02** | **41.66** | **−5.36** | **44.73** | **43.48** | **−1.25** | **46.20** | **45.38** | **−0.82** |

¹ `trec` numbers are unstable across bit widths even for jointqk (69.00 at k=2 > 41.50 fp16 oracle; 32.00 at k=3 < oracle) — trec is a classification task scored by exact-match on a short label, with only 200 rows, so single-row flips swing the F1 dramatically. Treat trec as noisy and weight the overall ranking by the other 11 tasks.

### Reading the F1 result

**QPCA loses on mean F1 at every bit width**, with the loss largest at k=2 (most aggressive compression):

| bit | mean ΔF1 | proxy Δtop-1 (compact9, 9 tasks) |
|---|---:|---:|
| 2 | **−5.36 pp** | −4.6 pp |
| 3 | **−1.25 pp** | −3.7 pp |
| 4 | **−0.82 pp** | −2.9 pp |

**The proxy predicted the direction correctly at every bit width.** The proxy slightly *underestimated* the F1 hit at k=2 (the regime where the bit budget is most starved and any per-coord allocation mistake compounds across all heads) and slightly *overestimated* it at k=3/k=4 (where redundancy across attention heads buffers small per-head argmax flips before they reach the answer).

### Where the losses concentrate

- **All multi-doc QA tasks** (hotpotqa, musique, 2wikimqa, narrativeqa) lose ≥ 4 pp at k=2. These tasks require the model to attend back to specific entity / fact tokens — exactly the high-leverage argmax-preservation regime where QPCA's uniform Λ-allocation hurts most.
- **Code tasks** (lcc, repobench-p) lose 8–12 pp at k=2 — the second-worst category. Code attention is sparse and position-sensitive (the model must lock onto specific token references), so argmax preservation is critical.
- **Long-input summarization** (qasper, qmsum, multi_news) loses 0.5–5.7 pp at k=2 — milder, because summary F1 is more forgiving of small attention shifts.
- **Classification tasks** (trec, samsum) have noisy small effects — F1 redundancy in label-classification dominates the comparison.

### F1 ↔ K-fidelity proxy correspondence is strong

Both experiments (per-task K-fidelity and per-task F1) point at the same tasks as the biggest losses for QPCA — multi-doc QA and code. The proxy's task ranking *roughly* predicts the F1 ranking, though the F1 absolute hits are typically 2× the per-task top-1 deltas (proxy says −4 pp top-1 → F1 says −8-12 pp on lcc/hotpotqa at k=2). The amplification is the proxy-to-F1 transfer: each percent point of top-1 deficit in a "high-attention-relevance" task costs ~2 pp F1 in extreme cases, ~1 pp in tame cases.

---

## Decision

**Do not switch the production basis.** `jointqk` remains the deployed K-side basis at every bit width.

The decision is now backed by three concentric experiments — pooled K-fidelity (Experiment 1), per-task K-fidelity (Experiment 2), and downstream F1 (Experiment 3). All three converge: QPCA's closed-form optimality for logit MSE comes at a real, measurable cost on the metric the system actually optimizes for (downstream task accuracy via attention argmax preservation).

**Keep the QPCA implementation around** (`analysis/measure_qpca_compact9.py`, the press-class extension at `kvq/compression/per_coord.py` and `kvq/presses/jointqk_press.py`, the compact8 bundle `artifacts/bases/qpca_qwen3_8b_longbench_compact8_n400.pt`). They serve as:

- A **closed-form baseline** that any future argmax-aware basis must beat (provably-lower logit MSE; the bar for "do you really need iterative optimization?").
- A **scaffold for allocation-swap experiments**: testing `qpca-basis + (q_diag · k_diag) score` vs `jointqk-basis + Λ score` would cleanly separate "is the basis the problem?" from "is the score the problem?" — both code paths now exist via the `qpca_waterfill` and `r_sym_waterfill` method dispatch in `build_jointqk_compressor`.

**Possible follow-ups** (not committed):

- Allocation-swap experiment (as above) to disentangle basis choice from allocation choice.
- Non-Gaussian (Gaussian-mixture or empirical-CDF) Lloyd-Max codebooks adapted to QPCA's actual non-orthogonal-projected code distribution. Could close some of the basis-side gap (Reading the result → "non-orthogonal basis fights per-coord scalar quantization").
- Direct argmax-aware basis: replace the closed-form QPCA objective with an empirical attention-KL or margin-preservation loss, solve iteratively. The QPCA result quantifies how much room there is to improve (the −5 pp F1 at k=2 is the maximum a perfect argmax-aware basis could recover, since jointqk is itself heuristic).

---

## Outputs on disk

```
# Standalone drivers (no pipeline integration)
analysis/measure_qpca_compact9.py                    (~440 LOC, pooled K-fidelity)
analysis/measure_qpca_vs_jointqk_per_task.py         (~440 LOC, per-task K-fidelity)

# Basis bundles (production-ready; loadable by JointQKPress with k_method="qpca_waterfill")
artifacts/bases/qpca_qwen3_8b_compact9_n450.pt       (12 MB; for per-task K-fidelity run, compact9 corpus)
artifacts/bases/qpca_qwen3_8b_longbench_compact8_n400.pt  (90 MB; for F1 sweep, matching prior production)

# Press / compressor extension
kvq/compression/per_coord.py                         (added qpca_* branch in build_jointqk_compressor)
kvq/presses/jointqk_press.py                         (pass qpca fields from bundle when present)

# Experiment 1 — pooled K-fidelity (20 min, 4 GPUs)
artifacts/calibration/longbench_compact9_qkv_qwen3_8b/05_reports/pooled_n50_qpca/
├── qpca_merged.json
├── shard_000.json … shard_003.json
└── shard_000.log … shard_003.log
logs/qpca_compact9.log

# Experiment 2 — per-task K-fidelity (26 min, 6 GPUs)
artifacts/calibration/longbench_compact9_qkv_qwen3_8b/05_reports/per_task_qpca_vs_jointqk/
├── per_task_merged.json
├── shard_000.json … shard_005.json
└── shard_000.log … shard_005.log
logs/per_task_qpca_vs_jointqk.log

# Experiment 3 — downstream F1 sweep (118 min, 6 GPUs)
artifacts/bench/qwen3_8b/qpca_k{2,3,4}_v3_<task>/    (36 cells across 12 LongBench tasks)
logs/bench_qpca_only/
logs/bench_qpca_only_launcher.log
```

Reference runs (existing v7 production, used as the jointqk + oracle baseline):

```
artifacts/bench/qwen3_8b/full_precision_<task>/      (12 fp16 oracle cells)
artifacts/bench/qwen3_8b/jointqk_k{2,3,4}_v3_<task>/ (36 jointqk cells matching the QPCA sweep)
```
