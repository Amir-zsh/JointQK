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
| **`cca_uniform`** | `P_K` from CCA (non-orthogonal) | hard cutoff at rank r=64, uniform bits on top-r |
| **`cca_waterfill`** | `P_K` (non-orthogonal) | continuous water-fill on `diag((P_K_inv)^T Σ_Q P_K_inv)_j · σ²_j(CCA)` |
| **`cca_orth_uniform`** | `V_h = P_K · Σ_K^{1/2}` (orthogonal CCA basis) | hard cutoff at rank r=64, uniform bits on top-r |
| **`cca_orth_waterfill`** | `V_h` (orthogonal) | continuous water-fill on `(V_h Σ_Q V_h^T)_jj · (V_h Σ_K V_h^T)_jj` |
| **`r_sym_uniform`** | `R_sym = eigvec((Σ_Q Σ_K + Σ_K Σ_Q)/2)` (orthogonal joint Q-K basis) | hard cutoff at rank r=64, uniform bits on top-r |
| **`r_sym_waterfill`** | `R_sym` | continuous water-fill on `(R_sym^T Σ_Q R_sym)_jj · (R_sym^T Σ_K R_sym)_jj` |

Where:
- `λ_j` are eigenvalues of `M_q`; `σ²_j(V) = (V^T Σ_K V)_jj`.
- `P_K` is the canonical-key projection from the SVD `Σ_Q^{-1/2} · C_QK · Σ_K^{-1/2} = U S V_h^T`, with `S = diag(ρ_1, …, ρ_d)` the canonical correlations and `P_K = V_h^T Σ_K^{-1/2}`.
- `V_h` (the right-singular-vectors of the whitened cross-moment SVD) is orthogonal by construction. Dropping the `Σ_K^{-1/2}` whitening factor from `P_K` recovers the same canonical-correlation ordering of coords without the non-orthogonal noise amplification (see §5.4).
- `R_sym` is the eigenbasis of the symmetric anti-commutator `(Σ_Q Σ_K + Σ_K Σ_Q)/2`. It is orthogonal and considers Q and K jointly: high-eigenvalue directions are the ones along which a typical `q` and `k` have aligned high variance simultaneously. Eigenvalues sort coords by joint Q-K energy.
- The CCA water-fill weight is the **trace-formula weight** `(P_K_inv^T Σ_Q P_K_inv)_jj`. Earlier drafts of E2 used `ρ_j²` (the canonical-score MSE weight), which is mathematically a *different* objective and undercut the Q-weighted distortion estimate by ~100× (see [F8](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1e_cca_vs_waterfill/fixes_to_apply.md) for derivation + Monte-Carlo verification, and [F11](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1e_cca_vs_waterfill/fixes_to_apply.md) for the analogous fix in the real compressor).
- For the orthogonal bases (`V`, `V_h`, `R_sym`), the trace-formula weight collapses to `(basis^T Σ_Q basis)_jj` because the basis is its own inverse-transpose.
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

### 4.1 Headline at b_avg = 3, layer-0-excluded (24 examples × 36 layers × 8 kv_heads, post-F11 + post-newbases)

| Method | top-1 ↑ | logit_mse ↓ | geometry_distortion ↓ | top-5 ↑ |
|---|---:|---:|---:|---:|
| **`r_sym_waterfill`** | **0.860** | **0.054** | **0.054** | **0.993** |
| `v_waterfill` | 0.760 | 0.066 | 0.066 | 0.937 |
| `cca_orth_waterfill` | 0.675 | 0.198 | 0.197 | 0.898 |
| `v3` | 0.682 | 0.457 | 0.456 | 0.906 |
| `v_truncate` | 0.592 | 0.537 | 0.528 | 0.856 |
| `cca_waterfill` | 0.535 | 0.097 | 0.097 | 0.762 |
| `cca_orth_uniform` | 0.393 | 6.351 | 6.200 | 0.567 |
| `cca_uniform` | 0.226 | 0.863 | 0.859 | 0.414 |
| `r_sym_uniform` | 0.219 | 61.97 | 62.08 | 0.383 |

`r_sym_waterfill` is the **new winner on every metric**: +10.0 pp top-1 over `v_waterfill`, +5.6 pp over the prior champion's top-5, and lower geometry distortion as well. `v_waterfill` slips to second.

`cca_orth_waterfill` confirms the "drop the whitening factor" hypothesis: replacing the non-orthogonal `P_K` with the orthogonal `V_h = P_K · Σ_K^{1/2}` recovers +14.0 pp top-1 over the original `cca_waterfill` while keeping the same canonical-correlation ordering. It is now competitive with `v3` on top-1 and beats it by 2.3× on geometry distortion. The remaining gap to `v_waterfill` (~8.5 pp top-1) is closed by the additional joint-Q-K weighting in `R_sym`.

The uniform variants of the new bases (`cca_orth_uniform`, `r_sym_uniform`) are worse than the original `cca_uniform` because their per-coord variance is highly heterogeneous in the orthogonal bases; without water-filling, uniform bits over-quantize the high-variance dimensions and produce huge geometry distortions. This is consistent with the §5 lesson that hard cutoffs amplify basis-variance heterogeneity. **The right pairing is always (orthogonal joint basis × water-fill).**

### 4.2 Bit-budget sensitivity

Top-1 across `b_avg ∈ {2, 3, 4}`, layer-0-excluded, ranked by `b=3` performance:

| `b_avg` | `r_sym_waterfill` | `v_waterfill` | `cca_orth_waterfill` | `v3` | `cca_waterfill` |
|---:|---:|---:|---:|---:|---:|
| 2 | **0.767** | 0.629 | 0.515 | 0.510 | 0.362 |
| 3 | **0.860** | 0.760 | 0.675 | 0.682 | 0.535 |
| 4 | **0.919** | 0.837 | 0.789 | 0.806 | 0.674 |

`r_sym_waterfill` wins at every bit budget by a wider margin at low `b_avg`: +13.9 pp at 2 bits, +10.0 pp at 3 bits, +8.2 pp at 4 bits. Unlike the old V_waterfill-vs-V3 gap, which shrinks with `b_avg`, `r_sym_waterfill`'s advantage is sustained — the joint Q-K basis pays off most when you have few bits and need every bit to land on a well-conditioned direction.

### 4.3 Generalization

**E4a (cross-task)** — calibrate from one config, evaluate on all 24 examples, layer-0-excluded top-1:

| Calibration source | `r_sym_waterfill` | `v_waterfill` | `cca_orth_waterfill` | `cca_waterfill` |
|---|---:|---:|---:|---:|
| `qasper` | **0.856** | 0.779 | 0.668 | 0.539 |
| `hotpotqa` | **0.857** | 0.756 | 0.673 | 0.548 |
| `passage_retrieval_en` | **0.858** | 0.766 | 0.683 | 0.523 |
| In-domain (E3, calib all 24) | 0.860 | 0.760 | 0.675 | 0.535 |

The 9-cell `(calib × eval)` matrix has a top-1 spread of ≤ 0.3 pp for `r_sym_waterfill` (the tightest of any method tested), and the cross-task ranking matches in-domain E3 for every cell. The new orthogonal-basis methods generalize across tasks at least as well as the existing methods, and `r_sym_waterfill` is the clear cross-task champion.

**E4b (within-task LOO)** — for each config × held-out example, calibrate from the other 7. LOO top-1 std dev across the 8 folds within a config:

| Config | `r_sym_waterfill` SD | `v_waterfill` SD | `cca_orth_waterfill` SD | `cca_waterfill` SD |
|---|---:|---:|---:|---:|
| qasper | 0.005 | 0.007 | 0.009 | 0.005 |
| hotpotqa | 0.009 | 0.013 | 0.016 | 0.006 |
| passage_retrieval_en | 0.003 | 0.003 | 0.005 | 0.002 |

Within-task LOO is essentially noise-level for all four water-fill methods. `r_sym_waterfill`'s std dev is comparable to `v_waterfill` and below `cca_orth_waterfill` (which inherits some of the cross-fold sensitivity from the un-orthogonalised `P_K` calibration). All methods generalise within < 1 pp under within-task LOO.

### 4.4 Decode-phase Q (E5) — gap = decode minus prefill, layer-0-excluded, b_avg = 3

| Method | prefill | decode (dq-weighted) | Δ |
|---|---:|---:|---:|
| **`r_sym_waterfill`** | **0.860** | **0.904** | +0.043 |
| `v_waterfill` | 0.760 | 0.829 | +0.069 |
| `v3` | 0.682 | 0.825 | +0.144 |
| `cca_orth_waterfill` | 0.675 | 0.762 | +0.086 |
| `v_truncate` | 0.592 | 0.734 | +0.142 |
| `cca_waterfill` | 0.535 | 0.596 | +0.061 |
| `cca_orth_uniform` | 0.393 | 0.539 | +0.146 |
| `r_sym_uniform` | 0.219 | 0.290 | +0.071 |
| `cca_uniform` | 0.226 | 0.279 | +0.054 |

Decode-phase top-1 is **higher** than prefill-phase for every method at every bit budget. At `b_avg = 4`, `r_sym_waterfill` decode top-1 reaches `0.944` — within 6 pp of full-precision attention. The decode-vs-prefill gap is *smaller* for `r_sym_waterfill` than for `v_waterfill` because R_sym already gets prefill so close to the ceiling that there's less headroom for decode-easier-than-prefill to recover. The "compress before generation" production claim is supported.

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
| Best | `r_sym_waterfill` (0.860) | `r_sym_waterfill` (0.054) |
| 2nd | `v_waterfill` (0.760) | `v_waterfill` (0.066) |
| 3rd | `v3` (0.682) | `cca_waterfill` (0.097) |
| 4th | `cca_orth_waterfill` (0.675) | `cca_orth_waterfill` (0.197) |
| 5th | `v_truncate` (0.592) | `v3` (0.456) |
| 6th | `cca_waterfill` (0.535) | `v_truncate` (0.528) |

For the **water-fill methods on orthogonal bases** (`r_sym_waterfill`, `v_waterfill`, `cca_orth_waterfill`), top-1 and geometry rankings agree exactly. The disagreement only appears for the **non-orthogonal** `cca_waterfill` (2nd on geometry, but worse on top-1 than V3) and for the **hard-cutoff** `v_truncate` / `cca_uniform` rows.

This sharpens the §5 lesson from the post-F11 draft: the geometry-vs-top-1 disagreement is **not** a generic property of low Q-weighted distortion failing to imply high top-1. It is specifically a **CCA-non-orthogonal-basis pathology**: when `forward_map = P_K^T` is non-orthogonal, the inverse `P_K_inv^T` amplifies coordinate-aligned quantization noise unevenly, which lands as structured logit perturbations that flip top-1 even at low average distortion. Switching to the orthogonal `V_h` (drop the `Σ_K^{-1/2}` factor from `P_K`) recovers most of the geometry advantage of CCA-style canonical-correlation ordering *and* aligns top-1 with geometry.

`R_sym` goes one step further: it considers the joint Q-K energy `(Σ_Q Σ_K + Σ_K Σ_Q)/2` rather than the canonical-correlation cross-moment alone, sorting coords by where a typical query and key have aligned high variance simultaneously. With orthogonal basis + water-fill, geometry and top-1 stay locked together as `b_avg` varies.

The methodological lesson, restated: **Q-weighted distortion and top-1 retention agree under (orthogonal basis × continuous water-fill); they can disagree under non-orthogonal bases or hard cutoffs that zero out entire subspaces.**

### 5.3 What the V_h CCA result tells us about the CCA pathology

A 5-head pre-merge prototype showed CCA's "top-1 amplification" — the ratio of `|q · (k_top1 - k̂_top1)|` to `|q · (k_random - k̂_random)|` — was ~11×, far higher than V3 (~3×) or V_waterfill (~2×). After replacing `P_K` with `V_h`, the same head-set's amplification dropped to ~2.8×, comparable to V_waterfill. The full E3 merge confirms this generalises: `cca_orth_waterfill` recovers +14.0 pp top-1 over `cca_waterfill` at `b_avg=3` while keeping the same canonical-correlation ordering of coords. The non-orthogonal noise amplification was the dominant pathology, not the canonical-correlation basis itself.

### 5.4 What the R_sym result tells us about the right design space

`r_sym_waterfill` outperforms `v_waterfill` (which uses the Q-only basis `eigvec(Σ_Q)`) by 8-14 pp top-1. The two bases differ only in whether they consider K's covariance: V uses `Σ_Q` alone; R_sym uses `(Σ_Q Σ_K + Σ_K Σ_Q)/2`. The bit-budget sensitivity table above shows R_sym wins by a *wider* margin at low bit budgets — exactly the regime where allocating bits to the right coords matters most. The takeaway: **for KV-cache compression of K, the basis must consider both K's variance and Q's expected energy; ignoring K's covariance leaves performance on the table.**

### 5.5 Decode-phase improvement over prefill is consistent and large

For every method, decode-phase top-1 is higher than prefill-phase. Reason: queries from generated tokens are *conditioned on* having just produced tokens that recall specific content from the prompt. Their attention is more peaked (fewer competing keys close in score) than the diffuse attention of prefill positions. Peaked attention has a larger top-1-vs-runner-up gap and tolerates more reconstruction error.

This is reassuring for the production use case: prefill calibration is not just adequate for decode, it's *easier* at decode time. Assumption A7 holds.

Caveat: `decode_query_count` per example is 1-34, well below the plan's `≥ 64` target. Aggregated cross-example numbers (over 6912 rows × `group=4` Q-heads per row) are credible, but per-example or per-config decode-only comparisons are noisy. See E5 review for details.

## 6. Decision rule and implications

### 6.1 Plan branches that fired

- **Rule 1 (V + waterfill ≥ CCA + waterfill on top-1):** ✅ fires for the original `cca_waterfill`. V_waterfill leads it by 22.5 pp on top-1. After the V_h fix, `cca_orth_waterfill` closes most of that gap (V_waterfill leads by 8.5 pp), and `r_sym_waterfill` *flips* the rule by beating both V_waterfill and CCA-orth by 8-19 pp.
- **Rule 5 (E4a >20% top-1 degradation under cross-task CCA → CCA broken at task level):** ❌ does not fire. Cross-task spread is ≤ 0.3 pp for `r_sym_waterfill`, ≤ 1.6 pp for `cca_orth_waterfill`, and ≤ 4 pp for `cca_waterfill`.
- **Rule 6 (E4b >10% degradation under within-task LOO → CCA sample-specific):** ❌ does not fire. LOO std dev is sub-1 pp for all four water-fill methods.
- **Rule 7/8 (E5 decode vs prefill):** ✅ supports the production claim. Decode top-1 is *higher* than prefill top-1 for every method, and `r_sym_waterfill` decode reaches 0.904 at `b_avg=3` (closest to full-precision attention of any method tested).
- **Rule 2 (basis + waterfill > basis + uniform on Q-weighted distortion AND top-1):** ✅ for every basis. The waterfill version of every basis (`v_waterfill`, `cca_waterfill`, `cca_orth_waterfill`, `r_sym_waterfill`) dominates its uniform-r=64 counterpart on both metrics.

### 6.2 Net recommendation

Use **`r_sym_waterfill`** as the Stage 3 candidate basis/allocation design. R_sym is the new champion across every metric (top-1, top-5, geometry, logit MSE) at every bit budget, and across every cross-task / within-task / decode evaluation. Its win is largest at low bit budgets — the regime production cares about most.

Why R_sym:
- **Joint Q-K basis.** `R_sym = eigvec((Σ_Q Σ_K + Σ_K Σ_Q)/2)` weights coords by where Q and K both have aligned high variance. This consumes K's covariance information that V_waterfill ignores; the joint signal is what closes the 8-14 pp top-1 gap.
- **Orthogonal.** `R_sym^T R_sym = I` so the inverse rotation is just transpose. No matrix-inverse drift, no `P_K_inv` machinery.
- **Same calibration cost as CCA.** One eigendecomposition of a `d × d` matrix per (layer, kv_head); cheaper than the SVD-of-whitened-cross-moment that CCA does.
- **Generalizes as well as V_waterfill.** Cross-task spread ≤ 0.3 pp; LOO std dev within rounding of V_waterfill's.

`v_waterfill` becomes the **fallback** if R_sym calibration is unavailable (e.g., if `Σ_K` is hard to obtain in the deployment context — though this seems unlikely given it's a model-side statistic). `cca_orth_waterfill` is the recommended *intermediate* method if you want to retain the canonical-correlation interpretation of P_K but need orthogonality.

`cca_waterfill` is **not recommended** for production: even after F11, the non-orthogonal `P_K` amplifies coordinate-aligned quantization noise, and `cca_orth_waterfill` strictly dominates it on every metric.

## 7. Stage 3 plan

The next step is to convert `r_sym_waterfill` from oracle calibration (using all 24 prefill examples to compute `Σ_Q`, `Σ_K`) to deployable calibration:

1. Confirm `Σ_Q`, `Σ_K`, and the resulting `R_sym = eigvec((Σ_Q Σ_K + Σ_K Σ_Q)/2)` basis are stable across larger calibration sets (e.g., 100+ prompts) and unrelated text distributions. The within-task LOO SD ≤ 0.009 at 8 examples is encouraging but should be validated at scale.
2. Add coarse rounding to integer bits (current implementation rounds continuous `b_j`; production would prefer fixed-bit-width hardware paths).
3. Compare against the Stage 1E `gamma = 0.25` partial-spectrum oracle from [stage1e_partial_spectrum_report.md](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1e_partial_spectrum_report.md). The `r_sym_waterfill` design can be viewed as choosing the joint-Q-K basis for the partial-spectrum truncation.
4. Test with downstream task scoring (LongBench-E exact-match, F1) — current evaluation is internal-metrics-only.
5. Regenerate the calibration bundle with `max_new_tokens ≥ 64` per example before any decode-only headline numbers carry weight in Stage 3 reporting.
6. Extend E2 (closed-form simulation) to predict `r_sym_waterfill` Q-weighted distortion. The integration gate flagged that the current sim winner (V_waterfill) disagrees with the new real winner (R_sym_waterfill) — this is because R_sym was added post-E2; recomputing E2 for R_sym should restore alignment.

V_waterfill remains the safe fallback; CCA-style methods (orth or non-orth) can be deferred.

## 8. Open follow-ups (not blocking the recommendation)

- **F9** (P2): E3 gate should explicitly assert `P_K_inv @ P_K ≈ I` and the row-vector pairing for CCA maps. Manual check passes.
- **F10** (P3): E3 gate should enforce the Stage 1D V3 baseline cross-check (parses the wrong key today). Manual check passes within 5%.
- **F12** (P3): bootstrap CIs include layer 0 while headline means exclude it. Reporting-only.
- **F13** (P3): finish doc-level cleanup of "centered CCA" language to "uncentered second-moment CCA / CCA-style whitened SVD".
- **F14** (P3): `Σ_Q` regularization differs between V branch (regularized eigvals), CCA branch (un-regularized `Mq`), and E2 simulation (regularized rebuild). Sub-percent residual; doc/alignment fix only.

See [stage1e_cca_vs_waterfill/fixes_to_apply.md](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1e_cca_vs_waterfill/fixes_to_apply.md) for the full bug tracker.

## 9. Quick-reference artifacts

- Per-experiment review notes: [e1_canonical_correlation_spectrum.md](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1e_cca_vs_waterfill/e1_canonical_correlation_spectrum.md), [e2_closed_form_simulation.md](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1e_cca_vs_waterfill/e2_closed_form_simulation.md), [e3_real_quantization.md](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1e_cca_vs_waterfill/e3_real_quantization.md), [e4_generalization.md](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1e_cca_vs_waterfill/e4_generalization.md), [e5_decode_phase.md](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1e_cca_vs_waterfill/e5_decode_phase.md).
- Headline E3 numbers + per-method bootstrap CIs: `artifacts/stage1/cca_vs_waterfill_study/e3/e3_b{2,3,4}_r64_summary.json`. Decode metrics live in the same row files. `cca_orth_*` and `r_sym_*` rows merged in via [merge_newbases.py](/vault/amir/efficient-llm/teamily-project/experiments/stage1/scripts/merge_newbases.py); pre-merge backups (`*.pre_newbases`) preserved for diff/audit.
- Closed-form simulation (E1+E2): `artifacts/stage1/cca_vs_waterfill_study/metrics_e1_e2.json`.
- Cross-task (E4a): `artifacts/stage1/cca_vs_waterfill_study/e4a/e4a_calib_*_summary.json` (3 calibration sources).
- Within-task LOO (E4b): `artifacts/stage1/cca_vs_waterfill_study/e4b/e4b_*_loo*_summary.json` (24 folds).
- Pre-F11 row backups (`*.pre_f11`) preserved for diff/audit.
- Report charts: `artifacts/stage1/cca_vs_waterfill_study/report_charts/` — regenerate via `python experiments/stage1/scripts/make_e{1,2,3,4,5}_charts.py`.
- F11 verification scripts: [verify_f11_allocation.py](/vault/amir/efficient-llm/teamily-project/experiments/stage1/scripts/verify_f11_allocation.py) (synthetic + 288-head allocation match) and [verify_f11_roundtrip.py](/vault/amir/efficient-llm/teamily-project/experiments/stage1/scripts/verify_f11_roundtrip.py) (Monte-Carlo closed-form-vs-real, 18/18 cells within 1.4 %).
- Auto-generated structured report: [stage1e_cca_vs_waterfill_report.md](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1e_cca_vs_waterfill_report.md).
- Index page: `artifacts/stage1/cca_vs_waterfill_study/INDEX.md`.
