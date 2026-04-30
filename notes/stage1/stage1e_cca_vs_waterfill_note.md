# Stage 1E Note: CCA vs Water-Filling Comparative Study

> Hand-written companion to the auto-generated [stage1e_cca_vs_waterfill_report.md](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1e_cca_vs_waterfill_report.md).
>
> Read together with [stage1d_norm_spread_ablation_report.md](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1d_norm_spread_ablation_report.md) and [stage1e_partial_spectrum_report.md](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1e_partial_spectrum_report.md). This stage refines the partial-spectrum result by comparing two specific (basis × allocation) design choices end-to-end.

## 1. Problem formulation

We want to compress KV-cache **keys** to roughly 2-4 bits per coordinate while preserving the attention scores `q · k` that future queries will compute against those keys. The V3 / TurboQuant baseline applies a random rotation, unit-normalizes each vector, and scalar-quantizes uniformly. That ignores the structure we know about Q and K.

Two recent threads suggested doing better:

- The rate-distortion simulation in section 6B of [longbench_data_tour.ipynb](/vault/amir/efficient-llm/teamily-project/experiments/stage1/notebooks/longbench_data_tour.ipynb) showed water-filling in the Q-eigenbasis V (eigenvectors of `M_q = E[q q^T]`) gives roughly a 5× theoretical Q-weighted distortion reduction over V3 at `b_avg = 3`, in 100% of (layer, kv_head) pairs.
- A collaborator's [runtime_complexity_analysis_new.md](/vault/amir/efficient-llm/teamily-project/notes/tmp/runtime_complexity_analysis_new.md) proposed a hybrid: offline **CCA projection** (canonical correlation basis) followed by uniform-bit TurboQuant of the projected keys. Predicted compression: 9-14× at low advertised attention error.

Both proposals belong to the same design space `(rotation basis × bit allocation)`. The hybrid uses CCA basis + hard rank cutoff + uniform bits. The water-filling result uses V basis + continuous bit allocation. Stage 1E was set up to answer:

> Under matched bit budgets, on Qwen3-8B's actual second-moment statistics, which (basis × allocation) combination wins on the metrics we care about — and does the closed-form simulation actually predict the right winner?

## 2. Proposed solutions and methods compared

For each `(layer, kv_head)` and each `b_avg ∈ {2, 3, 4}`:

| Method | Basis | Allocation |
|---|---|---|
| **`v3`** | random Hadamard rotation | uniform `b_avg` bits/coord (after unit-normalize) |
| **`v_truncate`** | V (top eigvecs of `M_q`) | hard cutoff at rank r=64, uniform bits on top-r |
| **`v_waterfill`** | V | continuous water-fill on `λ_j · σ²_j(V)` |
| **`cca_uniform`** | `P_K` from CCA | hard cutoff at rank r=64, uniform bits on top-r |
| **`cca_waterfill`** | `P_K` | continuous water-fill on `ρ_j² · σ²_j(CCA)` |

Where:
- `λ_j` are eigenvalues of `M_q`; `σ²_j(V) = (V^T Σ_K V)_jj`.
- `P_K` is the canonical-key projection from the SVD `Σ_Q^{-1/2} · C_QK · Σ_K^{-1/2} = U S V^T`, with `S = diag(ρ_1, ..., ρ_d)` the canonical correlations and `P_K = V^T Σ_K^{-1/2}`.
- Water-filling is reverse water-filling: bits are allocated proportional to `0.5 log_2(weight_j · σ²_j / θ)` with θ chosen so the total budget is met. Coords below θ get zero bits.

## 3. Details of the experimental setup

### 3.1 Data

Full 24-example LongBench-E bundle at `artifacts/stage1/query_stats_longbench_under4k/`:
- 8 examples per config × 3 configs (`qasper_e`, `hotpotqa_e`, `passage_retrieval_en_e`).
- 81,223 prefill tokens total across all examples.
- Prompt length per example: 1.4k - 3.9k tokens. Decode tokens per example: 2 - 35 (most around 4-9).
- Pre-RoPE and post-RoPE Q, K captured during model.generate. We use `q_post`, `k_post`.

### 3.2 Calibration

`Σ_Q`, `Σ_K`, `C_QK` are computed per `(layer, kv_head)` from prefill positions only:
- `Σ_Q[h] = (1/N) Σ_t q_t q_t^T`, with Q heads GQA-pooled (averaged within each kv-head's group of 4 query heads).
- `Σ_K[h] = (1/N) Σ_t k_t k_t^T`.
- `C_QK[h] = (1/N) Σ_t q_t k_t^T` (GQA-pooled q).

Whitening uses Tikhonov regularization with `ε = 1e-4 · trace/d` to handle the layer-0 near-singularity.

### 3.3 Quantization

A new `PerCoordCompressor` ([toolkit/per_coord_quantization.py](/vault/amir/efficient-llm/teamily-project/experiments/stage1/toolkit/per_coord_quantization.py)) was written for the non-V3 methods. It does not unit-normalize (V3 backend incompatibility); instead per-coordinate it rotates the key, scalar-quantizes using a 1D Gaussian Lloyd-Max codebook scaled to per-coord std, then inverse-rotates. Coordinates with `b_j = 0` reconstruct as zero. Rotations support both orthogonal (V) and non-orthogonal (CCA `P_K`) cases via separate `forward_map` and `inverse_map`.

### 3.4 Metrics

Per `(example, layer, kv_head, method)` we compute:
1. `geometry_distortion = E[(k - k̂)^T M_q (k - k̂)] / d` — the Q-weighted distortion that the closed-form simulation predicts.
2. `logit_mse = E[(q^T k - q^T k̂)²]` — squared error of attention logits.
3. `top1_match` — fraction of `q_t` for which `argmax_i (q_t^T k̂_i) = argmax_i (q_t^T k_i)`.
4. `top5_containment` — fraction of `q_t` for which the true argmax is in the top-5 reconstructed.

All metrics are computed against full `(prompt_length, head_dim)` keys for each `q_t` from `[0:prompt_length]` (prefill queries) and from `[prompt_length:total_length]` (decode-phase queries from generated tokens, evaluated against the same compressed prefill cache).

Reporting follows Stage 1's convention: report **layer-0-excluded** mean as the headline because layer 0 has anomalous norm/condition properties (Stage 1D finding).

### 3.5 Decision-rule precommitment

Per the Stage 1E plan, the eight decision-rule branches were specified before any results came in. The branch that fires depends on the numbers; this protects against narrative-fitting after the fact.

## 4. Results

### 4.1 Headline numbers at b_avg = 3, layer-0-excluded (24 examples × 36 layers × 8 kv_heads)

| Method | top-1 ↑ | logit_mse ↓ | geometry_distortion ↓ | top-5 ↑ |
|---|---:|---:|---:|---:|
| **`v_waterfill`** | **0.760** | **0.066** | **0.066** | **0.937** |
| `v3` | 0.682 | 0.457 | 0.456 | 0.906 |
| `v_truncate` | 0.592 | 0.537 | 0.528 | 0.856 |
| `cca_waterfill` | 0.370 | 0.309 | 0.308 | 0.593 |
| `cca_uniform` | 0.226 | 0.863 | 0.859 | 0.415 |

`v_waterfill` is best on every metric.

### 4.2 Bit-budget sensitivity

`v_waterfill` advantage over `v3` on top-1, layer-0-excluded:

| `b_avg` | `v_waterfill` | `v3` | gap |
|---:|---:|---:|---:|
| 2 | 0.629 | 0.510 | +12.0 pp |
| 3 | 0.760 | 0.682 | +7.8 pp |
| 4 | 0.837 | 0.806 | +3.1 pp |

The advantage shrinks as bits grow — expected, since uniform allocation in a random basis approaches sufficient precision at high `b_avg`.

### 4.3 Generalization

**E4a (cross-task)** — calibrate from one config, evaluate on all 24 examples:

| Calibration source | `v_waterfill` top-1 (l0excl) |
|---|---:|
| `qasper` | 0.771 |
| `hotpotqa` | 0.754 |
| `passage_retrieval_en` | 0.759 |
| In-domain (E3 pooled, all 24) | 0.760 |

Cross-task degradation is 0-1 pp.

**E4b (within-task LOO)** — for each config × held-out example, calibrate from the other 7:

| Method | hotpotqa | passage_retrieval_en | qasper | overall (n=24) |
|---|---:|---:|---:|---:|
| `v_waterfill` | 0.761 | 0.752 | 0.776 | 0.763 |
| `v3` | 0.691 | 0.671 | 0.660 | 0.682 |

Within-task LOO is essentially noise-level.

### 4.4 Decode-phase Q (E5) — gap = decode minus prefill, layer-0-excluded

| `b_avg` | Method | prefill | decode | Δ |
|---:|---|---:|---:|---:|
| 3 | `v_waterfill` | 0.760 | 0.828 | +0.068 |
| 3 | `v3` | 0.682 | 0.824 | +0.143 |
| 3 | `cca_waterfill` | 0.370 | 0.423 | +0.053 |
| 3 | `cca_uniform` | 0.226 | 0.279 | +0.054 |
| 3 | `v_truncate` | 0.592 | 0.732 | +0.140 |

Decode-phase top-1 is **higher** than prefill-phase for every method at every bit budget. The "compress before generation" production claim is supported.

## 5. Analysis

### 5.1 Simulation overpredicted by 100-300× — but in a structured way

The closed-form simulation (Bennett's high-rate distortion approximation, `D_j ≈ σ²_j · 2^{-2 b_j}`) predicted at `b_avg = 3`:

| Method | Predicted log₂(D / D_v3) | Predicted ratio | Real ratio (geometry distortion) |
|---|---:|---:|---:|
| `v_waterfill` | -3.45 | 0.092× | 0.144× (real geo_dist 0.066 / V3 0.456) |
| `cca_waterfill` | -8.40 | 0.003× | **0.676×** (0.308 / 0.456) |
| `cca_uniform` (r=64) | -4.56 | 0.042× | **1.881×** (0.859 / 0.456 — *worse* than V3) |

For `v_waterfill`, simulation directionally agrees with reality (~1.5× overprediction). For `cca_waterfill`, simulation overpredicted by ~225×; for `cca_uniform`, simulation said it would *beat* V3 but in reality it loses by 1.9×.

The reason is Bennett. Its `D = σ² · 2^{-2b}` assumes a smooth scalar quantizer with cells small relative to the source distribution. CCA's water-fill allocation puts essentially all bits on the top ~20 high-`ρ` coords (15-20 bits each, effectively full precision) and **zero bits on ~110 low-`ρ` coords**. At `b_j = 0` there is no quantizer; the coord is discarded. Bennett's smoothness assumption is violated. The discarded mass has full variance `σ²_j`, but it's *not* the smooth-noise residual Bennett models — it's a deterministic projection onto the discarded subspace.

In Q-weighted distortion, that projection is small *in expectation* over Q (because the discarded subspace has low canonical correlation by construction). But:

1. The expectation is taken over Q's distribution; individual `q_t` realizations have variance around their mean.
2. The discarded subspace has low expected projection from Q, but particular `q_t` vectors can have non-trivial projection. When they do, `q_t · (k_i - k̂_i)` is a structured shift that perturbs *specific* logits.
3. Top-1 retention depends on whether the perturbation exceeds the gap between true argmax and the runner-up. Structured perturbations preferentially flip specific keys; isotropic perturbations from random rotations average out across keys.

So CCA achieves what it was designed to achieve — low Q-weighted distortion in expectation — but at the cost of distortion that is structured (concentrated in specific subspaces), and structured distortion is what attacks rank-based metrics.

### 5.2 Metrics disagree on CCA but agree on V_waterfill

| At b_avg = 3 (l0excl) | top-1 ranking | geometry_distortion ranking |
|---|---|---|
| Best | `v_waterfill` | `v_waterfill` |
| 2nd | `v3` | `cca_waterfill` |
| 3rd | `v_truncate` | `v3` |
| 4th | `cca_waterfill` | `v_truncate` |
| 5th | `cca_uniform` | `cca_uniform` |

Notice `cca_waterfill` is **2nd** on geometry distortion but **4th** on top-1. This is the simulation-vs-reality gap surfacing as a metric-vs-metric disagreement. The simulation's metric of choice (Q-weighted distortion ≈ geometry distortion) places `cca_waterfill` above `v3`; the production metric (top-1 attention rank) places it below.

This is **the same pattern as the Stage 1D layer-0 paradox**, generalized: methods that improve Q-weighted distortion can degrade top-1 retention, because Q-weighted distortion averages over Q while top-1 reacts to specific queries.

`v_waterfill` avoids this trap because:
- `M_q`'s eigenvalue spectrum decays gradually (much less than `ρ_j²` for CCA), so water-filling allocates non-zero bits across most coords.
- No `b_j = 0` discards → no structured residual → Bennett's smoothness assumption approximately holds → simulation directionally matches reality → metrics agree.

### 5.3 Decode-phase improvement over prefill is consistent and large

For every method, decode-phase top-1 is higher than prefill-phase. Reason: queries from generated tokens are *conditioned on* having just produced tokens that recall specific content from the prompt. Their attention is more peaked (fewer competing keys close in score) than the diffuse attention of prefill positions. Peaked attention has a larger top-1-vs-runner-up gap and tolerates more reconstruction error.

This is reassuring for the production use case: prefill calibration is not just adequate for decode, it's *easier* at decode time. Assumption A7 holds.

## 6. Implications and decision rule

The fired branch is **Rule 1**: V + waterfill ≥ CCA + waterfill on top-1 → use V basis.

V basis has additional advantages over CCA:
- **Task-agnostic.** `M_q = E[q q^T]` is a property of the model, not the data distribution beyond the calibration set. CCA additionally needs `Σ_K` and `C_QK`, which encode joint Q-K structure that could be more task-specific. Empirically (E4a) both generalize across LongBench-E configs at this scale, but the V calibration is cleaner and lighter.
- **Cheaper.** One eigh per (layer, kv_head). CCA needs two whitening factorizations and an SVD.
- **Backend-friendly.** V is orthogonal, so the inverse rotation is just transpose; no matrix-inverse drift. CCA's `P_K` is non-orthogonal, requiring a separate `P_K_inv`.

CCA is not "broken" — its closed-form ranking on Q-weighted distortion does correspond to a real (smaller) improvement on geometry distortion in real quantization. But the gain is much smaller than predicted, and it doesn't translate to top-1 retention.

The most important takeaway is methodological: **the closed-form Bennett simulation must not be used to pick allocations that include `b_j = 0` discards**. Bennett does not model truncation. Soft allocations (where most coords get a few bits) are within Bennett's regime; aggressive allocations (where most coords get zero) are not.

## 7. Stage 3 plan

The next step is to convert `v_waterfill` from oracle calibration (using all 24 prefill examples to compute `M_q`) to deployable calibration:

1. Confirm `M_q` is stable across larger calibration sets (e.g., 100 prompts) and unrelated text distributions.
2. Add coarse rounding to integer bits (current implementation rounds continuous `b_j`; production would prefer fixed-bit-width hardware paths).
3. Compare against the Stage 1E `gamma = 0.25` partial-spectrum oracle from [stage1e_partial_spectrum_report.md](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1e_partial_spectrum_report.md). The `v_waterfill` design here can be viewed as a different point in the same partial-spectrum design space (allocation rather than gamma).
4. Test with downstream task scoring (LongBench-E exact-match, F1) — current evaluation is internal-metrics-only.

CCA can be deferred. If a task arises where the V_waterfill ceiling is reached, CCA-style joint conditioning is the next basis to revisit, but with awareness that aggressive water-filling on it must be replaced by a smoother allocation.

## 8. Quick-reference artifacts

- Headline numbers + per-method bootstrap CIs: `artifacts/stage1/cca_vs_waterfill_study/e3/e3_b{2,3,4}_r64_summary.json`
- Closed-form simulation (E1+E2): `artifacts/stage1/cca_vs_waterfill_study/metrics_e1_e2.json`
- Cross-task (E4a): `artifacts/stage1/cca_vs_waterfill_study/e4a/e4a_calib_*_summary.json`
- Within-task LOO (E4b): `artifacts/stage1/cca_vs_waterfill_study/e4b/e4b_*_loo*_summary.json`
- Figures: `artifacts/stage1/cca_vs_waterfill_study/figures/`
- Auto-generated structured report: [stage1e_cca_vs_waterfill_report.md](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1e_cca_vs_waterfill_report.md)
- Index page: `artifacts/stage1/cca_vs_waterfill_study/INDEX.md`
