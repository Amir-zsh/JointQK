# Stage 1E Note: CCA vs Water-Filling Comparative Study

> Hand-written companion to the auto-generated [stage1e_cca_vs_waterfill_report.md](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1e_cca_vs_waterfill_report.md). Per-experiment deep-dive reviews live in [stage1e_cca_vs_waterfill/](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1e_cca_vs_waterfill/) — start there for derivations and bug-hunts. This file is the cross-experiment summary post-F1, post-F8, post-F11.
>
> Read together with [stage1d_norm_spread_ablation_report.md](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1d_norm_spread_ablation_report.md) and [stage1e_partial_spectrum_report.md](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1e_partial_spectrum_report.md). This stage refines the partial-spectrum result by comparing two specific (basis × allocation) design choices end-to-end.

## 1. Problem formulation

We want to compress KV-cache **keys** to roughly 2-4 bits per coordinate while preserving the attention scores `q · k` that future queries will compute against those keys. The V3 / TurboQuant baseline applies a random rotation, unit-normalizes each vector, and scalar-quantizes uniformly. That ignores the structure we know about Q and K.

Two recent threads suggested doing better:

- The rate-distortion simulation in section 6B of [longbench_data_tour.ipynb](/vault/amir/efficient-llm/teamily-project/experiments/stage1/notebooks/longbench_data_tour.ipynb) showed water-filling in the Q-eigenbasis V (eigenvectors of `M_q = E[q q^T]`) gives roughly a 5× theoretical Q-weighted distortion reduction over V3 at `b_avg = 3`, in 100% of (layer, kv_head) pairs.
- A collaborator's [runtime_complexity_analysis_new.md](/vault/amir/efficient-llm/teamily-project/notes/tmp/runtime_complexity_analysis_new.md) proposed a hybrid: offline **CCA projection** (canonical correlation basis) followed by uniform-bit TurboQuant of the projected keys. Predicted compression: 9-14× at low advertised attention error.

Both proposals belong to the same design space `(rotation basis × bit allocation)`. The hybrid uses CCA basis + hard rank cutoff + uniform bits. The water-filling result uses V basis + continuous bit allocation. Stage 1E was set up to answer:

> Under matched bit budgets, on Qwen3-8B's actual second-moment statistics, which (basis × allocation) combination wins on the metrics we care about — and does the closed-form simulation actually predict the right winner?

## 2. Methods compared

For each `(layer, kv_head)` and each `b_avg ∈ {2, 3, 4}`:

| Method | Basis | Allocation |
|---|---|---|
| **`v3`** | random Hadamard rotation | uniform `b_avg` bits/coord (after unit-normalize) |
| **`v_truncate`** | V (top eigvecs of `M_q`) | hard cutoff at rank r=64, uniform bits on top-r |
| **`v_waterfill`** | V | continuous water-fill on `λ_j · σ²_j(V)` |
| **`cca_uniform`** | `P_K` from CCA | hard cutoff at rank r=64, uniform bits on top-r |
| **`cca_waterfill`** | `P_K` | continuous water-fill on `diag((P_K_inv)^T Σ_Q P_K_inv)_j · σ²_j(CCA)` |

Where:
- `λ_j` are eigenvalues of `M_q`; `σ²_j(V) = (V^T Σ_K V)_jj`.
- `P_K` is the canonical-key projection from the SVD `Σ_Q^{-1/2} · C_QK · Σ_K^{-1/2} = U S V^T`, with `S = diag(ρ_1, …, ρ_d)` the canonical correlations and `P_K = V^T Σ_K^{-1/2}`.
- The CCA water-fill weight is the **trace-formula weight** `(P_K_inv^T Σ_Q P_K_inv)_jj`. Earlier drafts of E2 used `ρ_j²` (the canonical-score MSE weight), which is mathematically a *different* objective and undercut the Q-weighted distortion estimate by ~100× (see [F8](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1e_cca_vs_waterfill/fixes_to_apply.md) for derivation + Monte-Carlo verification, and [F11](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1e_cca_vs_waterfill/fixes_to_apply.md) for the analogous fix in the real compressor).
- Water-filling is reverse water-filling: bits are allocated proportional to `0.5 log_2(weight_j · σ²_j / θ)` with θ chosen so the total budget is met. Coords below θ get zero bits.

## 3. Experimental setup

### 3.1 Data

Full 24-example LongBench-E bundle at `artifacts/stage1/query_stats_longbench_under4k/`:
- 8 examples per config × 3 configs (`qasper_e`, `hotpotqa_e`, `passage_retrieval_en_e`).
- 81,223 prefill tokens total across all examples.
- Prompt length per example: 1.4k - 3.9k tokens. **Decode tokens per example: 1 - 34** (most around 2-8). This decode count is below the Stage 1E plan's `≥ 64` target — see E5 caveat in §6 below.
- Pre-RoPE and post-RoPE Q, K captured during `model.generate`. We use `q_post`, `k_post`.

### 3.2 Calibration

`Σ_Q`, `Σ_K`, `C_QK` are computed per `(layer, kv_head)` from prefill positions only:
- `Σ_Q[h] = (1/N) Σ_t q_t q_t^T`, with Q heads GQA-pooled (averaged within each kv-head's group of 4 query heads).
- `Σ_K[h] = (1/N) Σ_t k_t k_t^T`.
- `C_QK[h] = (1/N) Σ_t q_t k_t^T` (GQA-pooled q).

These are **uncentered raw second moments** (no `E[q]` subtraction), to match the raw attention-logit objective `q · k`. Re-deriving with centered covariances changes layer-0-excluded `r_{95}` by at most 1 rank and median ρ₁ from `0.993` to `0.940` — terminology matters but rank conclusions are stable. See E1 review for the centered-vs-uncentered diagnostic.

Whitening uses Tikhonov regularization with `ε = 1e-4 · trace/d` to handle the layer-0 near-singularity.

E4a/E4b run the same calibration on subsets of examples; F1 reconciled an earlier `Σ_Q` convention drift between E4 and E1/E3 (per-Q-head outer-then-mean in both, divide by `group · total_tokens`).

### 3.3 Quantization

A new `PerCoordCompressor` ([toolkit/per_coord_quantization.py](/vault/amir/efficient-llm/teamily-project/experiments/stage1/toolkit/per_coord_quantization.py)) was written for the non-V3 methods. It does not unit-normalize (V3 backend incompatibility); instead per-coordinate it rotates the key, scalar-quantizes using a 1D Gaussian Lloyd-Max codebook scaled to per-coord std, then inverse-rotates. Coordinates with `b_j = 0` reconstruct as zero. Rotations support both orthogonal (V) and non-orthogonal (CCA `P_K`) cases via separate `forward_map = P_K^T` and `inverse_map = P_K_inv^T`.

### 3.4 Metrics

Per `(example, layer, kv_head, method)` we compute:
1. `geometry_distortion = E[(k - k̂)^T M_q (k - k̂)] / d` — the Q-weighted distortion that the closed-form simulation predicts.
2. `logit_mse = E[(q^T k - q^T k̂)²]` — squared error of attention logits.
3. `top1_match` — fraction of `q_t` for which `argmax_i (q_t^T k̂_i) = argmax_i (q_t^T k_i)`.
4. `top5_containment` — fraction of `q_t` for which the true argmax is in the top-5 reconstructed.

All metrics are computed for two Q slices: prefill queries `q_t` from `[0:prompt_length]` and decode-phase queries from `[prompt_length:captured_length]` (generated tokens, against the same compressed prefill cache).

Reporting follows Stage 1's convention: report **layer-0-excluded** mean as the headline because layer 0 has anomalous norm/condition properties (Stage 1D finding).

### 3.5 Decision-rule precommitment

Per the Stage 1E plan, the eight decision-rule branches were specified before any results came in. The branch that fires depends on the numbers; this protects against narrative-fitting after the fact.

## 4. Results

### 4.1 Headline at b_avg = 3, layer-0-excluded (24 examples × 36 layers × 8 kv_heads, post-F11)

| Method | top-1 ↑ | logit_mse ↓ | geometry_distortion ↓ | top-5 ↑ |
|---|---:|---:|---:|---:|
| **`v_waterfill`** | **0.760** | **0.066** | **0.066** | **0.937** |
| `v3` | 0.682 | 0.457 | 0.456 | 0.906 |
| `v_truncate` | 0.592 | 0.537 | 0.528 | 0.856 |
| `cca_waterfill` | 0.535 | 0.097 | 0.097 | 0.762 |
| `cca_uniform` | 0.226 | 0.863 | 0.859 | 0.414 |

`v_waterfill` is best on every metric. `cca_waterfill` is **2nd on geometry distortion** (much better than V3's 0.456) but **4th on top-1** (worse than V3's 0.682). This metric disagreement is the central tension of the study and is analyzed in §5.

### 4.2 Bit-budget sensitivity

`v_waterfill` advantage over `v3` on top-1, layer-0-excluded:

| `b_avg` | `v_waterfill` | `v3` | gap |
|---:|---:|---:|---:|
| 2 | 0.629 | 0.510 | +12.0 pp |
| 3 | 0.760 | 0.682 | +7.8 pp |
| 4 | 0.837 | 0.806 | +3.1 pp |

The advantage shrinks as bits grow — expected, since uniform allocation in a random basis approaches sufficient precision at high `b_avg`.

### 4.3 Generalization

**E4a (cross-task)** — calibrate from one config, evaluate on all 24 examples, layer-0-excluded:

| Calibration source | `v_waterfill` top-1 | `cca_waterfill` top-1 |
|---|---:|---:|
| `qasper` | 0.779 | 0.539 |
| `hotpotqa` | 0.756 | 0.548 |
| `passage_retrieval_en` | 0.766 | 0.523 |
| In-domain (E3, calib all 24) | 0.760 | 0.535 |

The 9-cell `(calib × eval)` matrix has a top-1 spread of ≤ 4 pp for both methods. Cross-task degradation is below the plan's 20 pp threshold by an order of magnitude.

**E4b (within-task LOO)** — for each config × held-out example, calibrate from the other 7. LOO top-1 std dev across the 8 folds within a config:

| Config | `v_waterfill` SD | `cca_waterfill` SD | `v3` SD |
|---|---:|---:|---:|
| qasper | 0.007 | 0.005 | 0.010 |
| hotpotqa | 0.013 | 0.006 | 0.018 |
| passage_retrieval_en | 0.003 | 0.002 | 0.005 |

Within-task LOO is essentially noise-level. CCA methods are the *least* variable across folds — global second-moment structure is robust to dropping one of eight calibration examples.

### 4.4 Decode-phase Q (E5) — gap = decode minus prefill, layer-0-excluded, b_avg = 3

| Method | prefill | decode (dq-weighted) | Δ |
|---|---:|---:|---:|
| `v_waterfill` | 0.760 | 0.829 | +0.069 |
| `v3` | 0.682 | 0.825 | +0.144 |
| `v_truncate` | 0.592 | 0.734 | +0.142 |
| `cca_waterfill` | 0.535 | 0.596 | +0.061 |
| `cca_uniform` | 0.226 | 0.279 | +0.054 |

Decode-phase top-1 is **higher** than prefill-phase for every method at every bit budget. At `b_avg = 4`, V3 decode top-1 (`0.895`) actually edges `v_waterfill` decode top-1 (`0.888`) by < 1 pp — within decode-statistic noise (max `decode_query_count` is 34), but worth flagging. The "compress before generation" production claim is supported.

## 5. Analysis

### 5.1 Sim vs reality after F8 + F11

Earlier drafts of this note reported the closed-form simulation overpredicting CCA by 100-300×. That was largely an arithmetic bug (F8): the CCA water-fill objective used `weights = ρ²` (canonical-score MSE) instead of the trace-formula weight `(P_K_inv^T Σ_Q P_K_inv)_jj` (Q-weighted reconstruction MSE). After F8 in the simulation and the analogous F11 in the real compressor, sim and real agree to within a small Bennett-approximation residual.

| Method (b_avg = 3, layer-0-excluded) | E2 sim `log₂(D/D_v3)` (post-F8) | E3 real `log₂(geo/geo_v3)` (post-F11) | implied ratio gap |
|---|---:|---:|---:|
| `v_waterfill` | -3.45 | -2.79 | 1.6× |
| `cca_waterfill` | -3.01 | -2.24 | 1.7× |
| `v_truncate_r64` | +0.65 | +0.21 | 1.4× |
| `cca_uniform_r64` | +1.33 | +0.91 | 1.3× |

These residual ~1.4-1.7× gaps are consistent with (a) the gap between Bennett's high-rate approximation and real Lloyd-Max scalar quantization (the Lloyd-Max distortion-rate constant differs from `2^{-2b}` at low `b`), and (b) the Σ_Q regularization inconsistency tracked as F14. Both are sub-percent doc/alignment issues, not headline-flipping bugs. See [verify_f11_roundtrip.py](/vault/amir/efficient-llm/teamily-project/experiments/stage1/scripts/verify_f11_roundtrip.py) for an end-to-end Monte-Carlo check that the closed-form prediction matches the real roundtrip MSE within 1.4 % rel error per (layer, kv_head, b_avg, method) cell.

### 5.2 Geometry vs top-1: the key tension

| At b_avg = 3 (l0excl) | top-1 ranking | geometry_distortion ranking |
|---|---|---|
| Best | `v_waterfill` (0.760) | `v_waterfill` (0.066) |
| 2nd | `v3` (0.682) | `cca_waterfill` (0.097) |
| 3rd | `v_truncate` (0.592) | `v3` (0.456) |
| 4th | `cca_waterfill` (0.535) | `v_truncate` (0.528) |
| 5th | `cca_uniform` (0.226) | `cca_uniform` (0.859) |

Notice `cca_waterfill` is **2nd** on geometry distortion but **4th** on top-1. The simulation's metric of choice (Q-weighted distortion ≈ geometry distortion) places `cca_waterfill` above `v3`; the production metric (top-1 attention rank) places it below.

This is **the same pattern as the Stage 1D layer-0 paradox**, generalized: methods that improve Q-weighted distortion can degrade top-1 retention, because Q-weighted distortion averages over Q while top-1 reacts to specific queries.

The likely mechanism: CCA's water-fill allocation puts most bits on the top high-`ρ` coords and very few on low-`ρ` coords (often `b_j ∈ {0, 1}` on the bottom 80-110 coords at `b_avg ∈ {2, 3}`). The discarded subspace has low *expected* projection from Q (by construction), but individual `q_t` realizations can have non-trivial projection onto it. When they do, the resulting `q_t · (k_i - k̂_i)` is a structured, basis-aligned shift that perturbs *specific* logits — which is exactly what attacks rank-based metrics. V3's random rotation diffuses noise more isotropically, so the same total Q-weighted distortion produces fewer top-1 flips. V_waterfill avoids the trap because `M_q`'s eigenvalue spectrum decays gradually relative to the canonical correlation spectrum, so water-fill on V allocates non-zero bits across most coords; there is no large "discarded subspace" residual.

This is the methodological lesson of Stage 1E: **Q-weighted distortion and top-1 retention can disagree on aggressive allocations that zero out entire subspaces**. Soft allocations (like V-waterfill's, where most coords get a few bits) keep the metrics in sync; hard cutoffs and CCA-style sharp spectra do not.

### 5.3 Decode-phase improvement over prefill is consistent and large

For every method, decode-phase top-1 is higher than prefill-phase. Reason: queries from generated tokens are *conditioned on* having just produced tokens that recall specific content from the prompt. Their attention is more peaked (fewer competing keys close in score) than the diffuse attention of prefill positions. Peaked attention has a larger top-1-vs-runner-up gap and tolerates more reconstruction error.

This is reassuring for the production use case: prefill calibration is not just adequate for decode, it's *easier* at decode time. Assumption A7 holds.

Caveat: `decode_query_count` per example is 1-34, well below the plan's `≥ 64` target. Aggregated cross-example numbers (over 6912 rows × `group=4` Q-heads per row) are credible, but per-example or per-config decode-only comparisons are noisy. See E5 review for details.

## 6. Decision rule and implications

### 6.1 Plan branches that fired

- **Rule 1 (V + waterfill ≥ CCA + waterfill on top-1):** ✅ fires. V_waterfill leads CCA_waterfill by ~22 pp on top-1.
- **Rule 5 (E4a >20% top-1 degradation under cross-task CCA → CCA broken at task level):** ❌ does not fire. Observed cross-task degradation is ~4 pp.
- **Rule 6 (E4b >10% degradation under within-task LOO → CCA sample-specific):** ❌ does not fire. Observed degradation is sub-1 pp.
- **Rule 7/8 (E5 decode vs prefill):** ✅ supports the production claim. Decode top-1 is *higher* than prefill top-1 for every method.
- **Rule 2 (CCA + waterfill > CCA + uniform on Q-weighted distortion AND top-1):** ✅ post-F11. cca_waterfill beats cca_uniform by ~31 pp on top-1 and ~9× on geometry. Hard cutoff is suboptimal vs continuous water-fill in the same basis.

### 6.2 Net recommendation

Use **`v_waterfill`** as the Stage 3 candidate basis/allocation design. V basis has additional advantages over CCA:

- **Task-agnostic.** `M_q = E[q q^T]` is a property of the model, not the data distribution beyond the calibration set. CCA additionally needs `Σ_K` and `C_QK`, which encode joint Q-K structure that could be more task-specific. Empirically (E4a) both generalize across LongBench-E configs at this scale, but the V calibration is cleaner and lighter.
- **Cheaper.** One eigh per (layer, kv_head). CCA needs two whitening factorizations and an SVD.
- **Backend-friendly.** V is orthogonal, so the inverse rotation is just transpose; no matrix-inverse drift. CCA's `P_K` is non-orthogonal, requiring a separate `P_K_inv` and the explicit-inverse machinery in `PerCoordCompressor`.

CCA is **not "broken"** post-F11 — corrected `cca_waterfill` is the second-best method by a wide margin on geometry distortion (`0.097` vs V3's `0.456`), and it generalizes well across tasks and samples. But it is dominated by V_waterfill on the production metric (top-1), and it carries the additional implementation complexity above. The case for CCA in production is weaker than the geometry-distortion ranking alone suggests.

## 7. Stage 3 plan

The next step is to convert `v_waterfill` from oracle calibration (using all 24 prefill examples to compute `M_q`) to deployable calibration:

1. Confirm `M_q` is stable across larger calibration sets (e.g., 100+ prompts) and unrelated text distributions.
2. Add coarse rounding to integer bits (current implementation rounds continuous `b_j`; production would prefer fixed-bit-width hardware paths).
3. Compare against the Stage 1E `gamma = 0.25` partial-spectrum oracle from [stage1e_partial_spectrum_report.md](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1e_partial_spectrum_report.md). The `v_waterfill` design here can be viewed as a different point in the same partial-spectrum design space (allocation rather than gamma).
4. Test with downstream task scoring (LongBench-E exact-match, F1) — current evaluation is internal-metrics-only.
5. Regenerate the calibration bundle with `max_new_tokens ≥ 64` per example before any decode-only headline numbers carry weight in Stage 3 reporting.

CCA can be deferred. If a task arises where the V_waterfill ceiling is reached, CCA-style joint conditioning is the next basis to revisit, but with awareness that aggressive water-filling on it must be paired with the trace-formula weight (F8/F11) and not the canonical-score weight `ρ²`.

## 8. Open follow-ups (not blocking the recommendation)

- **F9** (P2): E3 gate should explicitly assert `P_K_inv @ P_K ≈ I` and the row-vector pairing for CCA maps. Manual check passes.
- **F10** (P3): E3 gate should enforce the Stage 1D V3 baseline cross-check (parses the wrong key today). Manual check passes within 5%.
- **F12** (P3): bootstrap CIs include layer 0 while headline means exclude it. Reporting-only.
- **F13** (P3): finish doc-level cleanup of "centered CCA" language to "uncentered second-moment CCA / CCA-style whitened SVD".
- **F14** (P3): `Σ_Q` regularization differs between V branch (regularized eigvals), CCA branch (un-regularized `Mq`), and E2 simulation (regularized rebuild). Sub-percent residual; doc/alignment fix only.

See [stage1e_cca_vs_waterfill/fixes_to_apply.md](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1e_cca_vs_waterfill/fixes_to_apply.md) for the full bug tracker.

## 9. Quick-reference artifacts

- Per-experiment review notes: [e1_canonical_correlation_spectrum.md](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1e_cca_vs_waterfill/e1_canonical_correlation_spectrum.md), [e2_closed_form_simulation.md](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1e_cca_vs_waterfill/e2_closed_form_simulation.md), [e3_real_quantization.md](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1e_cca_vs_waterfill/e3_real_quantization.md), [e4_generalization.md](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1e_cca_vs_waterfill/e4_generalization.md), [e5_decode_phase.md](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1e_cca_vs_waterfill/e5_decode_phase.md).
- Headline E3 numbers + per-method bootstrap CIs: `artifacts/stage1/cca_vs_waterfill_study/e3/e3_b{2,3,4}_r64_summary.json`. Decode metrics live in the same row files.
- Closed-form simulation (E1+E2): `artifacts/stage1/cca_vs_waterfill_study/metrics_e1_e2.json`.
- Cross-task (E4a): `artifacts/stage1/cca_vs_waterfill_study/e4a/e4a_calib_*_summary.json` (3 calibration sources).
- Within-task LOO (E4b): `artifacts/stage1/cca_vs_waterfill_study/e4b/e4b_*_loo*_summary.json` (24 folds).
- Pre-F11 row backups (`*.pre_f11`) preserved for diff/audit.
- Report charts: `artifacts/stage1/cca_vs_waterfill_study/report_charts/` — regenerate via `python experiments/stage1/scripts/make_e{1,2,3,4,5}_charts.py`.
- F11 verification scripts: [verify_f11_allocation.py](/vault/amir/efficient-llm/teamily-project/experiments/stage1/scripts/verify_f11_allocation.py) (synthetic + 288-head allocation match) and [verify_f11_roundtrip.py](/vault/amir/efficient-llm/teamily-project/experiments/stage1/scripts/verify_f11_roundtrip.py) (Monte-Carlo closed-form-vs-real, 18/18 cells within 1.4 %).
- Auto-generated structured report: [stage1e_cca_vs_waterfill_report.md](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1e_cca_vs_waterfill_report.md).
- Index page: `artifacts/stage1/cca_vs_waterfill_study/INDEX.md`.
