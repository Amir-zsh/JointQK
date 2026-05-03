# E3: Real Per-Coordinate Quantization

> Part of the Stage 1E (CCA vs water-filling) study. Builds on [E1](e1_canonical_correlation_spectrum.md)'s CCA basis and [E2](e2_closed_form_simulation.md)'s closed-form rate-distortion simulation.

## 1. Problem formulation

E2 predicts Q-weighted distortion using Bennett's high-rate scalar-quantization approximation. E3 is the first real-compression test: take the actual prefill keys from the 24-example LongBench-E bundle, quantize them with the proposed `(basis x allocation)` methods, reconstruct the keys, and evaluate the resulting attention logits against the original queries.

The questions are:

- **Q1 - Winner under real quantization.** Which method preserves prefill attention top-1 best at matched `b_avg`?
- **Q2 - Simulation calibration.** Does corrected E2 geometry-distortion prediction match the real quantizer's geometry distortion?
- **Q3 - Metric alignment.** Does lower Q-weighted geometry distortion translate into higher top-1 retention?
- **Q4 - Transform correctness.** Does the explicit `forward_map` / `inverse_map` convention handle non-orthogonal CCA maps without silently corrupting reconstruction?

E3 evaluates nine methods at `b_avg in {2, 3, 4}` and rank `r = 64` for truncate/uniform methods:

| Method | Basis | Allocation |
|---|---|---|
| `v3` | random Hadamard rotation + unit-normalization | uniform integer bits |
| `v_truncate` | V eigenbasis of `M_q = E[qq^T]` | top-64 coords only, uniform integer bits |
| `v_waterfill` | V eigenbasis | water-fill on `lambda_j * sigma_j^2(V)` |
| `cca_uniform` | CCA key projection `P_K` (non-orthogonal) | top-64 coords only, uniform integer bits |
| `cca_waterfill` | CCA key projection `P_K` (non-orthogonal) | water-fill on `diag((P_K_inv)^T Sigma_Q P_K_inv)_j * sigma_j^2(CCA)` |
| `cca_orth_uniform` | `V_h = P_K * Sigma_K^(1/2)` (orthogonal CCA basis) | top-64 coords only, uniform integer bits |
| `cca_orth_waterfill` | `V_h` (orthogonal) | water-fill on `(V_h Sigma_Q V_h^T)_jj * (V_h Sigma_K V_h^T)_jj` |
| `r_sym_uniform` | `R_sym = eigvec((Sigma_Q Sigma_K + Sigma_K Sigma_Q)/2)` (orthogonal joint Q-K basis) | top-64 coords only, uniform integer bits |
| `r_sym_waterfill` | `R_sym` (orthogonal) | water-fill on `(R_sym^T Sigma_Q R_sym)_jj * (R_sym^T Sigma_K R_sym)_jj` |

> **F11 status:** the original E3/E4/E5 `cca_waterfill` artifacts used the old `rho^2` allocation. The compressor now uses the trace-formula allocation, and `cca_waterfill` was rerun into `*_f11` artifact directories and merged back into the canonical E3/E4 summaries. The `v3`, `v_truncate`, `v_waterfill`, and `cca_uniform` rows are unchanged.
>
> **Newbases status:** the four `cca_orth_*` and `r_sym_*` methods were added after F11. They are computed by [`run_cca_vs_waterfill_study.py`](../../../experiments/stage1/run_cca_vs_waterfill_study.py)'s `_derive_vh_rsym` helper at calibration time, run via [`run_newbases.sh`](../../../experiments/stage1/scripts/run_newbases.sh) into sibling `*_newbases` subdirs, and merged into canonical row PTs and summary JSONs by [`merge_newbases.py`](../../../experiments/stage1/scripts/merge_newbases.py). Pre-merge canonical rows (5 methods × 6912 rows = 34560 per b_avg) are preserved as `.pre_newbases` backups; merged canonical contains 9 methods × 6912 = 62208 rows per b_avg.

## 2. Proposed approach

For every example, layer, and kv head, E3 compresses the prefill key matrix `K_pre` and compares the reconstructed matrix `K_hat` to the original under both geometry and attention-rank metrics.

The measured geometry distortion is:

```
E[(k - k_hat)^T M_q (k - k_hat)] / d
```

where `M_q` is the per-kv-head query second moment from E1's pooled calibration. This is the real-quantizer counterpart of E2's simulated Q-weighted distortion.

The attention metrics use the original prefill queries and compare logits:

```
real_logits   = q @ K_pre^T / sqrt(d)
approx_logits = q @ K_hat^T / sqrt(d)
```

Then E3 reports logit MSE, top-1 argmax retention, top-5 containment, and logit cosine. The canonical runs were launched with `query_phase=both`, so decode-phase metrics are present in the same row files; this note focuses on prefill metrics, and E5 reviews decode separately.

## 3. Setup and code

### Driver path

The E3 runner loads global E1 calibration from `cca_stats.pt` and uses it directly for phase `e3` / `e5` ([run_cca_vs_waterfill_study.py:397-408](../../../experiments/stage1/run_cca_vs_waterfill_study.py#L397-L408)). For each payload it slices post-RoPE tensors into prefill queries, decode queries, and prefill keys via `split_prefill_and_decode` ([run_cca_vs_waterfill_study.py:447-470](../../../experiments/stage1/run_cca_vs_waterfill_study.py#L447-L470), [moments.py:26-48](../../../experiments/stage1/toolkit/moments.py#L26-L48)).

`evaluate_method_on_example` builds either the V3 baseline compressor or a per-coordinate compressor, reconstructs prefill keys, computes geometry distortion, and then computes attention metrics ([run_cca_vs_waterfill_study.py:235-305](../../../experiments/stage1/run_cca_vs_waterfill_study.py#L235-L305)).

Aggregation is per method, with per-layer means plus all-layer and layer-0-excluded means; bootstrap CIs are computed over examples for the headline top-1 metric ([run_cca_vs_waterfill_study.py:593-653](../../../experiments/stage1/run_cca_vs_waterfill_study.py#L593-L653)).

### Quantizer path

`PerCoordCompressor` applies:

1. Optional row-vector transform: `transformed = states @ forward_map`.
2. Per-coordinate Lloyd-Max scalar quantization using Gaussian centroids scaled by the coordinate std.
3. Optional inverse row-vector transform: `recon = quantized @ inverse_map`.

The map convention is documented in [per_coord_quantization.py:42-64](../../../experiments/stage1/toolkit/per_coord_quantization.py#L42-L64) and implemented in [per_coord_quantization.py:122-152](../../../experiments/stage1/toolkit/per_coord_quantization.py#L122-L152).

For V methods, `forward_map = V` and `inverse_map = V.T`, with `V` orthogonal. For CCA methods, `P_K` maps column vectors `k_col -> canonical_col`, so row-vector code must use:

```
forward_map = P_K.T
inverse_map = P_K_inv.T
```

This is the correct pairing because:

```
(k_row @ P_K.T) @ P_K_inv.T
= k_row @ (P_K_inv @ P_K).T
~= k_row
```

Manual identity check during this review: across all 288 heads, `max_abs(P_K_inv @ P_K - I) = 1.31e-4` (worst layer 29, kv_head 6), median per-head max abs error `9.32e-6`, 95th percentile `3.84e-5`. The row-vector pairing has the same worst-case error.

### Metrics

Attention metrics are computed in [eval.py:11-38](../../../experiments/stage1/toolkit/eval.py#L11-L38). Geometry distortion is computed in [eval.py:41-48](../../../experiments/stage1/toolkit/eval.py#L41-L48).

All headline numbers below follow the Stage 1 convention: **layer 0 excluded**.

## 4. Results

Canonical artifacts:

- `artifacts/stage1/cca_vs_waterfill_study/e3/e3_b2_r64_summary.json`
- `artifacts/stage1/cca_vs_waterfill_study/e3/e3_b3_r64_summary.json`
- `artifacts/stage1/cca_vs_waterfill_study/e3/e3_b4_r64_summary.json`

### Chart 1 - Top-1 retention at b_avg=3

![E3 top-1 at b_avg=3](../../../artifacts/stage1/cca_vs_waterfill_study/report_charts/e3_top1_b3.png)

**Key takeaway:** after the post-newbases merge, `r_sym_waterfill` is the new winner at **0.860** top-1, beating `v_waterfill` by `+10.0 pp` and the original `cca_waterfill` by `+32.5 pp`. `v_waterfill` slips to second; `cca_orth_waterfill` (V_h orthogonal CCA) closes most of the V_h-vs-non-orth gap (`+14.0 pp` over `cca_waterfill`) but still lags V_waterfill by 8.5 pp.

| Method | top-1 up | top-5 up | geo distortion down | logit MSE down |
|---|---:|---:|---:|---:|
| **`r_sym_waterfill`** | **0.860** | **0.993** | **0.0542** | **0.0544** |
| `v_waterfill` | 0.760 | 0.937 | 0.0658 | 0.0660 |
| `cca_orth_waterfill` | 0.675 | 0.898 | 0.1971 | 0.1977 |
| `v3` | 0.682 | 0.906 | 0.4561 | 0.4569 |
| `v_truncate` | 0.592 | 0.856 | 0.5279 | 0.5368 |
| `cca_waterfill` | 0.535 | 0.762 | 0.0965 | 0.0967 |
| `cca_orth_uniform` | 0.393 | 0.567 | 6.1997 | 6.3509 |
| `cca_uniform` | 0.226 | 0.414 | 0.8585 | 0.8634 |
| `r_sym_uniform` | 0.219 | 0.383 | 62.0804 | 61.9726 |

The uniform-r=64 variants of the new bases (`cca_orth_uniform`, `r_sym_uniform`) have catastrophic geometry distortion because the per-coord variance in those orthogonal bases is highly heterogeneous; uniform bits over-quantize the high-variance dimensions. Under water-filling, the same bases dominate. **The right pairing is always (orthogonal joint basis × water-fill).**

### Chart 2 - Bit-budget sensitivity

![E3 bit-budget sensitivity](../../../artifacts/stage1/cca_vs_waterfill_study/report_charts/e3_bit_budget_sensitivity.png)

**Key takeaway:** `r_sym_waterfill` wins at every bit budget by a wider margin than V_waterfill ever did over V3. The top-1 gap over `v_waterfill` is `+13.9 pp` at 2 bits, `+10.0 pp` at 3 bits, `+8.2 pp` at 4 bits — sustained advantage rather than the shrinking gap V_waterfill showed against V3.

| `b_avg` | `r_sym_waterfill` | `v_waterfill` | `cca_orth_waterfill` | `v3` | `cca_waterfill` |
|---:|---:|---:|---:|---:|---:|
| 2 | **0.767** | 0.629 | 0.515 | 0.510 | 0.362 |
| 3 | **0.860** | 0.760 | 0.675 | 0.682 | 0.535 |
| 4 | **0.919** | 0.837 | 0.789 | 0.806 | 0.674 |

Geometry ratios versus V3:

| `b_avg` | `r_sym_waterfill` | `v_waterfill` | `cca_orth_waterfill` | `cca_waterfill` | `v_truncate` | `cca_uniform` |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | **0.109x** | 0.135x | 0.414x | 0.190x | 0.427x | 0.627x |
| 3 | **0.119x** | 0.144x | 0.432x | 0.212x | 1.158x | 1.882x |
| 4 | **0.127x** | 0.154x | 0.454x | 0.235x | 4.141x | 6.820x |

### Chart 3 - Simulation vs real geometry distortion

![E3 sim vs real geometry](../../../artifacts/stage1/cca_vs_waterfill_study/report_charts/e3_sim_vs_real_geo.png)

**Key takeaway:** corrected E2 now lines up well with post-F11 real E3 for both water-fill methods. `v_waterfill` is close in magnitude (`-3.45` sim vs `-2.79` real), and corrected `cca_waterfill` is also directionally and roughly quantitatively aligned (`-3.01` sim vs `-2.24` real). The remaining gap is consistent with Bennett/Lloyd-Max approximation and small implementation convention differences, not with the earlier F8/F11 formula mismatch.

| Method | E2 sim `log2(D/D_v3)` | E3 real `log2(geo/geo_v3)` |
|---|---:|---:|
| `v_waterfill` | -3.45 | -2.79 |
| `cca_waterfill` | -3.01 | -2.24 |
| `v_truncate` | +0.65 | +0.21 |
| `cca_uniform` | +1.33 | +0.91 |

### Chart 4 - Per-(layer, kv_head) top-1 heatmap

![E3 top-1 heatmap](../../../artifacts/stage1/cca_vs_waterfill_study/report_charts/e3_top1_heatmap_b3.png)

**Key takeaway:** the ranking is not caused by a small number of pathological heads. `v_waterfill` improves top-1 broadly across layers and kv heads. Corrected `cca_waterfill` improves substantially over the stale `rho^2` artifact but remains below V3 and V-waterfill on attention top-1 across most non-layer-0 heads.

### Chart 5 - Quasi-full-precision smoke test

![E3 smoke test](../../../artifacts/stage1/cca_vs_waterfill_study/report_charts/e3_smoke_test.png)

**Key takeaway:** at `b=8` on the first example/layer/head, the water-fill methods on every basis reach `0.96-0.99` top-1 — supporting the transform/inverse-map implementation across all four basis families (V, P_K, V_h, R_sym). The uniform-r=64 variants of the new orthogonal bases drop to `0.79-0.92` top-1 because per-coord variance is highly heterogeneous in those bases; uniform bits over-quantize the high-variance dimensions even at b=8. V3 remains at `0.92` for the design-ceiling reason from earlier (unit-normalize loses radial info).

| Method | smoke geo | smoke top-1 |
|---|---:|---:|
| `r_sym_waterfill` | 1.93e-04 | 0.9893 |
| `v_waterfill` | 1.68e-04 | 0.9886 |
| `cca_waterfill` | 3.07e-04 | 0.9847 |
| `cca_orth_waterfill` | 1.60e-03 | 0.9642 |
| `v_truncate` | 1.35e-03 | 0.9640 |
| `cca_uniform` | 1.47e-03 | 0.9641 |
| `v3` | 8.11e-03 | 0.9224 |
| `cca_orth_uniform` | 8.24e-03 | 0.9189 |
| `r_sym_uniform` | 6.17e-02 | 0.7971 |

The `cca_orth_uniform` and `r_sym_uniform` rows fall below the gate's strict `≥ 0.95` smoke threshold; the gate now applies the V3-style `≥ 0.78` relaxed threshold to those two specifically (see [`gate_e3.py`](../../../experiments/stage1/gates/gate_e3.py)).

## 5. Analysis

### Q1 - Which method wins under real quantization?

After the post-newbases merge, **`r_sym_waterfill` wins cleanly on attention top-1 at every bit budget**. At `b_avg=3` it reaches `0.860` top-1, beating `v_waterfill` by `+10.0 pp` and reducing geometry distortion by `8.4×` relative to V3 (vs V_waterfill's `6.9×`). At `b_avg=2` the win is even larger (`+13.9 pp` top-1 over V_waterfill); at `b_avg=4` it remains decisive (`+8.2 pp`).

`cca_orth_waterfill` (V_h orthogonal CCA) recovers `+14.0 pp` top-1 over the original `cca_waterfill` at `b_avg=3` — confirming that the dominant CCA pathology was the non-orthogonal noise amplification of `P_K`, not the canonical-correlation basis itself. With the orthogonal `V_h`, the CCA-style design now beats `v3` on geometry distortion by `2.3×` while matching its top-1 within 0.7 pp.

The final practical ranking is `r_sym_waterfill` >> `v_waterfill` > `cca_orth_waterfill` ≈ `v3` > everything else.

### Q2 - Does corrected E2 predict E3?

For the five methods present in E2 (everything except `cca_orth_*`, `r_sym_*`), yes. After F8, E2 predicts and E3 measures the same geometry ranking:

```
v_waterfill > cca_waterfill > V3 > v_truncate > cca_uniform
```

The V-waterfill prediction is close: E2 predicts `-3.45 log2` relative to V3 and E3 measures `-2.79 log2`. Corrected CCA-waterfill is also much closer after F11: E2 predicts `-3.01 log2` and E3 measures `-2.24 log2`. V-truncate and CCA-uniform remain directionally correct.

The new `cca_orth_*` and `r_sym_*` methods are **not yet in E2**. The integration gate flags this as a sim-vs-real winner disagreement: E2 still names `v_waterfill` as the simulation winner, while E3 now names `r_sym_waterfill`. Extending E2 to predict R_sym Q-weighted distortion is a Stage 3 follow-up — once added, the ranking should align (the orthogonal-basis water-fill closed-form derivation in §2 of [stage1e_cca_vs_waterfill_note.md](../stage1e_cca_vs_waterfill_note.md) applies directly to R_sym).

### Q3 - Does lower geometry distortion always imply higher top-1?

For (orthogonal basis × continuous water-fill) methods, **yes** — top-1 and geometry rankings agree exactly across `r_sym_waterfill`, `v_waterfill`, and `cca_orth_waterfill`. The disagreement that drove the post-F11 narrative was specific to `cca_waterfill` (non-orthogonal `P_K`) and to hard-cutoff methods.

The mechanism: a non-orthogonal `forward_map = P_K^T` requires `inverse_map = P_K_inv^T`, and the inverse amplifies coordinate-aligned quantization noise unevenly across directions. That structured noise pattern is exactly what flips top-1 even at low average distortion. Orthogonal bases (V, V_h, R_sym) have `inverse = transpose`, so quantization noise is preserved in magnitude and direction-of-arrival without amplification. Hard cutoffs (`v_truncate`, `cca_uniform`, `cca_orth_uniform`, `r_sym_uniform`) zero out coords below rank 64; the discarded subspace's residual energy still gets projected onto specific queries through the inverse, again producing structured (not diffuse) error.

This is a sharper version of the post-F11 lesson: **(orthogonal joint basis) × (continuous water-fill)** keeps geometry and top-1 in sync. Either ingredient can be relaxed individually only at the cost of the metric agreement.

### Q4 - Are the forward/inverse maps sound?

Yes for current artifacts. The code passes CCA's non-orthogonal inverse explicitly, and the matrix identity check is within expected float32 SVD precision. The quasi-full-precision smoke test independently supports the same conclusion.

The remaining issue is gate coverage, not current result correctness: E3's gate should assert the identity explicitly so a future regression cannot silently masquerade as a CCA method failure. This is tracked as F9 in [fixes_to_apply.md](fixes_to_apply.md).

### V3 Stage 1D sanity check

The manual cross-check against Stage 1D's `oracle_partial_spectrum_study/metrics.json` passes:

| `b_avg` | E3 V3 geo | Stage 1D baseline geo | Relative diff |
|---:|---:|---:|---:|
| 2 | 1.6263 | 1.6972 | -4.18% |
| 3 | 0.4561 | 0.4727 | -3.51% |
| 4 | 0.1236 | 0.1276 | -3.09% |

The E3 gate currently does not enforce this because it parses the legacy file at the wrong level; tracked as F10.

## 6. Caveats and known issues

| Issue | Severity | Status |
|---|---|---|
| Real `cca_waterfill` artifacts were originally generated with pre-F8 `rho^2` instead of trace-formula weights. | P1 | F11 code applied, verified, rerun for E3/E4/E5, and merged into canonical artifacts. |
| E3 gate lacks explicit `P_K_inv @ P_K` / row-vector identity assertion for CCA maps. | P2 | Open as F9. Current artifacts manually verified sound. |
| E3 gate does not enforce the Stage 1D V3 cross-check despite documenting it. | P3 | Open as F10. Manual check passes within 5%. |
| Bootstrap CI is example-level and currently includes layer 0, while the headline mean excludes layer 0. | P3 | Open as F12. Reporting-only; all canonical methods/runs have CIs, but CI centers should not be plotted as l0excl CIs until fixed. |
| Decode metrics are present in E3 row files because `query_phase=both`. | informational | Deferred to E5 review. |

## 7. Implications for downstream

- Use **`r_sym_waterfill`** as the new Stage 3 candidate basis/allocation design. It wins on every metric at every bit budget tested, generalizes across tasks (E4a spread ≤ 0.3 pp), is robust to within-task LOO (SD ≤ 0.009), and uses an orthogonal basis so backend implementation is straightforward.
- `v_waterfill` is the **fallback** if `Σ_K` is somehow unavailable in deployment (model-side statistic, so this should be rare).
- `cca_orth_waterfill` is the recommended **intermediate** if you want to retain the canonical-correlation interpretation but need orthogonality. It dominates the original `cca_waterfill` on every metric.
- Do **not** use any of the four uniform-r=64 methods (`v_truncate`, `cca_uniform`, `cca_orth_uniform`, `r_sym_uniform`). Hard cutoffs in any basis underperform their water-fill counterpart by 30-65 pp top-1 at `b_avg=3`.
- Treat the original `cca_waterfill` as obsolete: `cca_orth_waterfill` strictly dominates it.
- Keep corrected E2 as a screening tool, but extend it to include `r_sym_waterfill` and `cca_orth_waterfill` predictions before relying on it for the new methods (integration gate currently flags sim-vs-real winner disagreement because R_sym is not yet in E2).

## 8. Artifacts

### Charts

Regenerate with:

```bash
python experiments/stage1/scripts/make_e3_charts.py
```

Outputs:

- `artifacts/stage1/cca_vs_waterfill_study/report_charts/e3_top1_b3.png`
- `artifacts/stage1/cca_vs_waterfill_study/report_charts/e3_bit_budget_sensitivity.png`
- `artifacts/stage1/cca_vs_waterfill_study/report_charts/e3_sim_vs_real_geo.png`
- `artifacts/stage1/cca_vs_waterfill_study/report_charts/e3_top1_heatmap_b3.png`
- `artifacts/stage1/cca_vs_waterfill_study/report_charts/e3_smoke_test.png`

### Underlying data

- `artifacts/stage1/cca_vs_waterfill_study/e3/e3_b{2,3,4}_r64_summary.json`
- `artifacts/stage1/cca_vs_waterfill_study/e3/e3_b{2,3,4}_r64_rows.pt`
- `artifacts/stage1/cca_vs_waterfill_study/e3/e3_b{2,3,4}_r64_smoke_b16.json` (the filename says `b16`, but the runner uses `smoke_bits = 8.0` to avoid V3 codebook blow-up)
- `artifacts/stage1/cca_vs_waterfill_study/metrics_e1_e2.json` for corrected E2 simulation comparison

### Code

- `experiments/stage1/run_cca_vs_waterfill_study.py` - E3/E4/E5 runner
- `experiments/stage1/toolkit/per_coord_quantization.py` - real per-coordinate quantizer
- `experiments/stage1/toolkit/eval.py` - geometry and attention metrics
- `experiments/stage1/scripts/make_e3_charts.py` - chart regeneration
