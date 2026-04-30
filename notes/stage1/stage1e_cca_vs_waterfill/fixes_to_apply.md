# Stage 1E: Fixes to Apply

> Tracking document for issues found during the per-experiment review of Stage 1E (E1 → E2 → E3 → E4 → E5). Fixes are NOT applied immediately — we batch them once the review pass is complete.
>
> Each entry includes severity, where it was found, what it affects, and the proposed fix. New fixes are appended; status is updated in place.

## Status legend

- **open** — known, not yet fixed.
- **applied** — fix landed in a commit.
- **deferred** — known but explicitly chosen not to fix (with reason).

## Severity legend

- **P0**: invalidates a headline conclusion.
- **P1**: makes some reported numbers wrong but doesn't flip the headline.
- **P2**: correctness/robustness improvement; protects against future bugs.
- **P3**: cosmetic / docs.

---

## Resolved in this session

### F11 — Real CCA-waterfill allocation still used pre-F8 `rho^2` weight  *[E3/E4/E5]*  **MAJOR — APPLIED + RERUN MERGED**
- **Severity:** P1
- **Surfaced in:** E3 review
- **Where:** Pre-fix, [per_coord_quantization.py](../../../experiments/stage1/toolkit/per_coord_quantization.py) used `weights = rho^2` for CCA methods. The corrected E2 simulation uses the trace-formula weight in [run_cca_diagnostics.py:205-218](../../../experiments/stage1/run_cca_diagnostics.py#L205-L218).
- **Description:** F8 fixed the CCA water-fill objective in the closed-form E2 simulation, replacing `rho^2` with `diag((P_K_inv)^T Sigma_Q P_K_inv)`. The real E3/E4/E5 `PerCoordCompressor` path still water-filled CCA coordinates with `rho^2`, so the original real `cca_waterfill` artifacts were not evaluating the same method as post-F8 E2's corrected `cca_waterfill` prediction.
- **Why it matters:** This affected all previously reported real `cca_waterfill` numbers in E3/E4/E5. It does not affect `v3`, `v_truncate`, `v_waterfill`, or `cca_uniform`. After rerun, the final E3 conclusion is that corrected `cca_waterfill` is much better on geometry than the stale artifact but still worse than V3 and V-waterfill on top-1.
- **Allocation mismatch diagnostic:** Over all 288 heads, comparing current `rho^2` integer allocations to trace-formula integer allocations:

  | `b_avg` | Mean abs bit diff / coord | Median per-head L1 bit diff | Mean zero-bit coords current | Mean zero-bit coords trace |
  |---:|---:|---:|---:|---:|
  | 2 | 0.767 | 96 | 22.40 | 1.53 |
  | 3 | 0.876 | 110 | 10.24 | 0.13 |
  | 4 | 0.926 | 116 | 4.98 | 0.01 |

  No head had identical current-vs-trace allocation at any bit budget.
- **Fix applied as code:** In the CCA branch of `build_method_compressor`, replaced:
  ```python
  rho = cca["rho"].clamp(min=0.0, max=1.0)
  weights = (rho.float() ** 2).clamp_min(1e-30)
  ```
  with:
  ```python
  metric_in_canonical = (
      cca["P_K_inv"].transpose(-1, -2)
      @ sigma_q_for_head
      @ cca["P_K_inv"]
  )
  weights = metric_in_canonical.diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
  ```
  keeping `sigma_k_diag = diag(P_K Sigma_K P_K^T)` as the source variance term.
- **Verification:** Added and ran `experiments/stage1/scripts/verify_f11_allocation.py`.
  - Synthetic diagonal case: fixed compressor allocates `[3, 1, 0, 0]`, old `rho^2` would allocate `[1, 1, 1, 1]`.
  - Real Qwen3-8B all 288 heads: fixed compressor exactly matches independent trace-formula integer allocations at `b_avg in {2,3,4}`.
  - Old-vs-new mean per-head L1 bit differences: 98.24 (b=2), 112.12 (b=3), 118.58 (b=4); zero heads identical.
  - Tiny E3 runner smoke (`f11_smoke`, 1 example x 1 layer x 8 heads, CPU) completed successfully.
- **Rerun/merge:** Reran only `cca_waterfill` with the fixed allocator into temporary artifact directories, then merged only those rows into the canonical summaries with `experiments/stage1/scripts/merge_f11_cca_waterfill.py`.
  - E3/E5: `artifacts/stage1/cca_vs_waterfill_study/e3_f11/`, b_avg ∈ {2,3,4}, `query_phase=both`.
  - E4a: `artifacts/stage1/cca_vs_waterfill_study/e4a_f11/`, three calibration-source runs.
  - E4b: `artifacts/stage1/cca_vs_waterfill_study/e4b_f11/`, 24 leave-one-out runs.
  - Canonical files were backed up once with `.pre_f11` suffixes before row replacement.
- **Post-rerun validation:** `gate_e3.py`, `gate_e4.py`, and `gate_e5.py` all pass on the merged canonical artifacts. `make_e2_charts.py` and `make_e3_charts.py` were regenerated without stale-F11 labels.
- **Post-F11 E3 headline at b_avg=3:** corrected `cca_waterfill` top-1 = 0.535, top-5 = 0.762, geometry = 0.0965, real log₂(geo/geo_v3) = -2.24. This is a large improvement over stale `rho^2` geometry/top-1, but remains below V3 top-1 = 0.682 and V-waterfill top-1 = 0.760.
- **Files affected:** `experiments/stage1/toolkit/per_coord_quantization.py`; verification script `experiments/stage1/scripts/verify_f11_allocation.py`; merge script `experiments/stage1/scripts/merge_f11_cca_waterfill.py`; canonical E3/E4 row/summary artifacts and E2/E3 charts.
- **Status:** **applied, rerun, merged, and gate-verified**.

## Open fixes

### F9 — E3 gate does not explicitly validate per-head reconstruction identity  *[E3]*
- **Severity:** P2
- **Surfaced in:** E3 review
- **Where:** E3 relies on `P_K` / `P_K_inv` when constructing CCA compressors in [per_coord_quantization.py:212-260](../../../experiments/stage1/toolkit/per_coord_quantization.py#L212-L260), then uses the resulting reconstruction in [run_cca_vs_waterfill_study.py:266-284](../../../experiments/stage1/run_cca_vs_waterfill_study.py#L266-L284). The existing E3 gate checks finite metrics and smoke-test behavior, but not the transform identity in [gate_e3.py:22-104](../../../experiments/stage1/gates/gate_e3.py#L22-L104).
- **Description:** `PerCoordCompressor` correctly supports non-orthogonal CCA maps via separate `forward_map = P_K.T` and `inverse_map = P_K_inv.T`, but E3 has no explicit gate that asserts `(x @ P_K.T) @ P_K_inv.T ≈ x` for all heads. A future regression back to `inverse_map = forward_map.T` would silently inflate CCA geometry distortion and could look like a method failure rather than a code-path failure.
- **Manual check during review:** Current artifacts are sound: over all 288 `(layer, kv_head)` pairs, `max_abs(P_K_inv @ P_K - I) = 1.31e-4` (worst at layer 29, kv_head 6), median per-head max abs error `9.32e-6`, 95th percentile `3.84e-5`. Row-vector pairing has the same max error.
- **Proposed fix:** Add an E3 gate check loading `cca_stats.pt`, asserting both `P_K_inv @ P_K ≈ I` and `P_K.T @ P_K_inv.T ≈ I` with a threshold around `1e-3`, reporting the worst `(layer, kv_head)` on failure. Optionally add a tiny no-quantization synthetic roundtrip test for `PerCoordCompressor` with explicit CCA maps.
- **Files affected:** `experiments/stage1/gates/gate_e3.py` (and optionally a small helper test script).
- **Rerun cost:** Zero for experiment artifacts; gate-only.
- **Status:** **open** — do not apply until the review batch is ready.

### F10 — E3 gate does not enforce the Stage 1D V3 baseline cross-check  *[E3]*
- **Severity:** P3
- **Surfaced in:** E3 review
- **Where:** [gate_e3.py:57-69](../../../experiments/stage1/gates/gate_e3.py#L57-L69)
- **Description:** The gate comments say V3 geometry distortion should match the Stage 1D `oracle_partial_spectrum_study/metrics.json` baseline within about 5%, but the code only scans top-level keys containing `"baseline"` and `"3"`. The actual Stage 1D file stores the relevant values under `layer0_excluded_summary`, so the gate never performs the comparison.
- **Manual check during review:** The current E3 V3 layer-0-excluded geometry distortion matches Stage 1D `baseline_raw` within tolerance:

  | `b_avg` | E3 V3 geo | Stage 1D `baseline_raw` geo | Relative diff |
  |---:|---:|---:|---:|
  | 2 | 1.6263 | 1.6972 | −4.18% |
  | 3 | 0.4561 | 0.4727 | −3.51% |
  | 4 | 0.1236 | 0.1276 | −3.09% |

- **Proposed fix:** Parse `legacy["layer0_excluded_summary"][str(b_avg)]["baseline_raw"]["mean_geometry_distortion"]` for `b_avg ∈ {2,3,4}`, compare against E3's `aggregated["v3"]["geometry_distortion"]["l0excl_mean"]`, and fail if absolute relative error exceeds 5%.
- **Files affected:** `experiments/stage1/gates/gate_e3.py`.
- **Rerun cost:** Zero for experiment artifacts; gate-only.
- **Status:** **open** — do not apply until the review batch is ready.

### F12 — E3 bootstrap CI is computed over all layers while headline is layer-0-excluded  *[E3]*
- **Severity:** P3
- **Surfaced in:** E1-E3 re-review
- **Where:** [run_cca_vs_waterfill_study.py:628-651](../../../experiments/stage1/run_cca_vs_waterfill_study.py#L628-L651)
- **Description:** `aggregate_results` computes `bootstrap_ci` from per-example means over all rows for the method, including layer 0. The headline means in the review and tables are `l0excl_mean`, which exclude layer 0. This makes the CI center inconsistent with the headline estimator.
- **Why it matters:** The issue is reporting-only; it does not affect stored per-layer metrics or headline `l0excl_mean` values. It can mislead if CI bands are plotted around layer-0-excluded bars. At `b_avg=3`, the CI mean differs from the headline by:

  | Method | CI mean (all layers) | Headline l0excl top-1 | Difference |
  |---|---:|---:|---:|
  | `v_waterfill` | 0.7592 | 0.7601 | −0.0009 |
  | `v3` | 0.6653 | 0.6818 | −0.0165 |
  | `v_truncate` | 0.5899 | 0.5918 | −0.0020 |
  | `cca_waterfill` | 0.3768 | 0.3697 | +0.0071 |
  | `cca_uniform` | 0.2317 | 0.2258 | +0.0059 |

- **Proposed fix:** In `aggregate_results`, compute bootstrap examples from rows with `layer > 0` whenever the report headline is `l0excl_mean`, and store metadata like `"layer_filter": "exclude_0"`. Optionally keep a separate all-layer CI if needed.
- **Files affected:** `experiments/stage1/run_cca_vs_waterfill_study.py`; existing summary JSON can be regenerated from row files without recompressing keys if a small postprocessor is written, otherwise rerun E3/E4 summaries.
- **Rerun cost:** Zero for raw artifacts; summary-only regeneration from row files should be sufficient. Full E3 rerun not required for this fix alone.
- **Status:** **open** — do not apply until the review batch is ready.

### F14 — Σ_Q regularization is inconsistent between V branch, CCA branch, and E2 simulation  *[E2/E3]*
- **Severity:** P3
- **Surfaced in:** F11 re-review (deeper Monte-Carlo verification pass)
- **Where:**
  - [run_cca_vs_waterfill_study.py:213-230](../../../experiments/stage1/run_cca_vs_waterfill_study.py#L213-L230) — `mq_eigvals` / `mq_eigvecs` come from regularized `sym_mq_reg = sym_mq + ε·trace/d·I`; the returned `Mq` is the un-regularized `sym_mq`.
  - [per_coord_quantization.py:204-211, 224-229](../../../experiments/stage1/toolkit/per_coord_quantization.py#L204-L229) — V branch uses `weights = mq_eigvals` (regularized); CCA branch uses `sigma_q_for_head = Mq` (un-regularized) in the trace formula.
  - [run_cca_diagnostics.py:191, 212](../../../experiments/stage1/run_cca_diagnostics.py#L191) — E2 simulation rebuilds `sigma_q_full` from regularized `sigma_q_eigvals`/`sigma_q_eigvecs`, so its trace formula uses the regularized Σ_Q.
- **Description:** The V real branch and the E2 simulation effectively use Σ_Q + ε·trace/d·I; the CCA real branch uses Σ_Q. Differences:
  - V branch true `(V.T M_q_unreg V)_jj` vs `mq_eigvals`: median abs diff `2.07e-4`, max `7.63e-4`. Median rel diff `0.027%`; max rel `324%` (only on near-zero tail eigvals).
  - At our operating points (b_avg ∈ {2, 3, 4}), water-fill assigns 0 bits to those tail coords anyway, so the allocation is unaffected.
  - Does explain part of the residual E2-vs-E3 sim-vs-real gap that remains after F8/F11.
- **Why it matters:** Doc/alignment. The F8 + F11 corrections collapsed the headline gap; F14 is a sub-percent residual. Listing it for audit-trail completeness so the simulation and real path are provably the same modulo rounding.
- **Diagnostic:** `experiments/stage1/scripts/verify_f11_roundtrip.py` shows the closed-form trace prediction matches the real compressor roundtrip MSE within 1.4% across 3 layers × 3 b_avg × {CCA, V} water-fill on Gaussian samples drawn from real Σ_Q, Σ_K. The remaining gap is consistent with regularization plus Lloyd-Max constant approximation.
- **Proposed fix:** Pick one convention and apply everywhere. Simplest is to drop V's regularization step entirely, since `sym_mq` is empirically PSD (sum of outer products) and `eigh` is stable on it; or, equivalently, rebuild `Mq` as the regularized version so all three paths see the same Σ_Q. Verify with the existing scripts after.
- **Files affected:** `experiments/stage1/run_cca_vs_waterfill_study.py`, `experiments/stage1/run_cca_diagnostics.py`, `experiments/stage1/toolkit/per_coord_quantization.py`. No artifact rerun needed (sub-percent effect).
- **Rerun cost:** Zero compute; would refresh `cca_stats.pt` and re-derive E2 numbers (~30 sec from cache).
- **Status:** **open** — flagged for completeness; do not apply until the review batch is ready.

### F13 — E1 review/report terminology says centered CCA, but implementation uses uncentered second moments  *[E1 docs]*
- **Severity:** P3
- **Surfaced in:** E1-E3 re-review
- **Where:** E1 calibration uses `Σ_Q = E[qq^T]`, `Σ_K = E[kk^T]`, and `C_QK = E[qk^T]` in [run_cca_diagnostics.py:83-136](../../../experiments/stage1/run_cca_diagnostics.py#L83-L136), then passes those raw second moments into [compute_cca_basis](../../../experiments/stage1/toolkit/metric_transform.py#L66-L128). The E1 note previously used classical "centered random vectors" wording.
- **Description:** The method is better described as **uncentered / second-moment CCA** or a CCA-style whitening/SVD diagnostic for attention logits, not classical centered covariance CCA. This is intentional for the attention objective because logits use raw `q^T k`, but the wording matters.
- **Diagnostic during review:** Recomputing a centered-covariance variant changed layer-0-excluded `r95` only mildly (uncentered min/median/max `48/68/109`; centered `49/69/109`), but reduced median `rho1` outside layer 0 from `0.993` to `0.940`. Rank conclusions are stable; absolute "canonical correlation" values are not exactly classical centered correlations.
- **Proposed fix:** Update remaining docs/code comments/report labels to say "uncentered second-moment CCA" where precision matters. Keep `rho` naming if convenient, but define it as a singular value of the whitened raw cross-moment.
- **Files affected:** Review notes, auto report text, possibly comments in `run_cca_diagnostics.py` / `metric_transform.py`.
- **Rerun cost:** None.
- **Status:** **open** — E1/E2/E3 review notes have been partially corrected; broader docs/report cleanup remains.

---

## Applied fixes (this batch)

### F8 — E2 simulation uses wrong per-coord weight for CCA methods  *[E2]*  **MAJOR — APPLIED**
- **Severity:** P1
- **Surfaced in:** E2 review
- **Where:** [run_cca_diagnostics.py:193-217](../../../experiments/stage1/run_cca_diagnostics.py#L193-L217)
- **Description:** Pre-fix, `simulate_method`'s CCA branch used `weights = ρ²` and computed `D = sum_j ρ²_j · σ²_j(CCA) · 2^{−2 b_j}`. This is the canonical-score-MSE quantity, NOT the Q-weighted reconstruction MSE that E3 actually measures. The correct formula uses the trace-formula per-coord weight `((P_K_inv)^T Σ_Q P_K_inv)_jj`.
- **Verification (`scripts/verify_f8_bug.py`):**
  - Synthetic Σ_Q = Σ_K = I, C_QK = diag(ρ): trace formula matches analytic answer to 6e-08; buggy formula off by 3.74×.
  - Synthetic random PSD case + Monte-Carlo (n=500k samples): trace formula matches MC within **0.38%**; buggy formula off by **99.89%**.
  - V-basis case: trace formula reduces to current V code to fp64 precision (V case unaffected).
  - Real Qwen3-8B (layer=1, head=0) + Monte-Carlo (n=200k): trace formula matches MC within **0.61%**; buggy formula under-predicts by **193.5×**.
- **Fix applied:** In `simulate_method`'s CCA branch:
  ```python
  P_K = cca["P_K"]
  P_K_inv = cca["P_K_inv"]
  sigma_q_full = (sigma_q_eigvecs * sigma_q_eigvals.unsqueeze(-2)) @ sigma_q_eigvecs.transpose(-1, -2)
  metric_in_canonical = P_K_inv.transpose(-1, -2) @ sigma_q_full @ P_K_inv
  weights = metric_in_canonical.diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
  Sk_in_cca = P_K @ sigma_k @ P_K.transpose(-1, -2)
  sigma_k_diag = Sk_in_cca.diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
  ```
  V branch unchanged. Updated docstring with the derivation and the verify_f8_bug.py reference.
- **Empirical impact on E2 simulation at b_avg=3, layer-0-excluded:**

  | Method | Pre-fix log₂(D/D_v3) | Post-fix log₂(D/D_v3) |
  |---|---:|---:|
  | `v_waterfill` | −3.45 | **−3.45** (unchanged) |
  | `cca_waterfill` | −8.40 | **−3.01** |
  | `cca_uniform_r96` | −7.02 | **+0.54** (now loses to V3) |
  | `cca_uniform_r64` | −4.56 | **+1.33** (now loses to V3) |
  | `cca_uniform_r48` | −3.43 | **+1.77** (now loses to V3) |
  | `cca_uniform_r32` | −2.44 | **+2.25** (now loses to V3) |

  Headline: corrected sim now picks `v_waterfill` over `cca_waterfill` and predicts all CCA-uniform variants lose to V3. Post-F11 real geometry now matches the corrected CCA-waterfill simulation directionally.
- **Files modified:** `experiments/stage1/run_cca_diagnostics.py`. Verification script: `experiments/stage1/scripts/verify_f8_bug.py`.
- **Rerun cost:** E1+E2 ~30 sec (cached). At the time F8 was applied, E3/E4 were treated as unaffected because they do not call `simulate_method`; the later E3 review found and fixed F11, the analogous stale `rho^2` allocation in the real `PerCoordCompressor` path. Gate `gate_e1_e2.py` PASSES with the new numbers.
- **Status:** **applied (in working tree)**. The E2 markdown report has been updated to reflect the corrected ranking and to caveat the F11 CCA-waterfill real-artifact mismatch.

---

---

## Applied fixes (this batch)

### F1 — Σ_Q convention inconsistency between E1/E3-global and E4 calibration  *[E4]*
- **Severity:** P1
- **Surfaced in:** E1 review + Codex review of E4
- **Where:** [run_cca_vs_waterfill_study.py:159-160](../../../experiments/stage1/run_cca_vs_waterfill_study.py#L159-L160)
- **Description:** E1 (`load_pooled_stats`) and the global E3 calibration used `Σ_Q[h] = mean_g E[q_g q_g^T]` — treat each query head as a sample. E4's `_accumulate_calibration_stats` used `Σ_Q[h] = E[(mean_g q_g)(mean_g q_g)^T]` — average then outer. These differ by within-group cross terms.
- **Empirical impact** (from `experiments/stage1/scripts/diagnose_f1_impact.py` on one example, 288 (layer, head) pairs):
  - `trace(A) / trace(B)` median **0.71** (range 0.40–0.91) — A is ~30% smaller in trace than B.
  - `‖A − B‖_F / ‖B‖_F` median **0.13** (max 0.77).
  - Top-64 V subspace alignment median **0.83** (range 0.65–0.95) — V basis genuinely rotates between conventions; some heads have alignment as low as 0.65.
  - CCA `ρ` differs structurally: e.g., `ρ_{30}` typically 0.84 under A vs 0.55 under B — ~30pp gap.
  - `r_{95}` shifts up by ~15 ranks under A relative to B.
- **Why it matters:** This is **not** a benign scale-only difference. The eigenstructure changes, so the V basis and CCA basis used in E4 differed from those used in E3. The fact that V_waterfill landed at ~0.76 in both E3 (convention B) and E4a/E4b (convention A) was *despite* the basis differing — because water-fill on `λ_j · σ²_j(V)` is scale-invariant, and the resulting per-coordinate variances happen to give similar total bit allocations across the two bases. Per-(layer, head) allocations can still shift noticeably.
- **Diagnostic script:** [scripts/diagnose_f1_impact.py](../../../experiments/stage1/scripts/diagnose_f1_impact.py).
- **Fix applied (in working tree):** `_accumulate_calibration_stats` now does:
  ```python
  q_per_head_outer = torch.einsum("lhsd,lhse->lhde", q, q)  # (n_layers, n_q_heads, d, d)
  sq_acc += q_per_head_outer.view(n_layers, n_kv_heads, group, head_dim, head_dim).sum(dim=2)
  ```
  and divides `sigma_q` by `(group * total_tokens)` instead of `total_tokens`. C_QK still uses GQA-pooled q (linear in q). Verified by direct comparison: fixed `sigma_q` matches convention B to fp32 zero (`max abs diff = 0`, `relative Frobenius = 0`) on the qasper-8 calibration set.
- **Files affected:** `experiments/stage1/run_cca_vs_waterfill_study.py`.
- **Rerun cost:** E4a (~2 hr) + E4b (~12 min). Total ~2.2 hr.
- **Status:** **applied (code only)** — E4a/E4b artifacts still reflect pre-fix numbers; rerun pending.

### F2 — Decode-only `--query-phase decode` path passes empty tensor → NaN  *[E5 codepath]*  **APPLIED**
- **Severity:** P2 (never triggered in our actual runs; we always used `prefill` or `both`)
- **Surfaced in:** Codex review
- **Fix applied:** Removed `decode` from CLI `--query-phase` choices. `both` covers what `decode` was meant to do. Option-(a) from the original plan.
- **Files modified:** `experiments/stage1/run_cca_vs_waterfill_study.py` (argparse choices + help text).
- **Rerun cost:** Zero (existing runs used `prefill` or `both`).
- **Status:** applied (in working tree).

### F3 — Water-fill `clamp_max(max_bits)` discards budget instead of redistributing  *[E2/E3]*  **APPLIED**
- **Severity:** P2
- **Surfaced in:** Codex review
- **Fix applied:** Rewrote `water_fill` as iterative loop: solve unconstrained water-fill, identify saturated coords (>= max_bits), fix them at max_bits, recompute on remaining coords with reduced budget. Loop bounded by `d + 1` iterations. Extracted helper `_water_fill_unconstrained_row` for the inner solve. Verified with 5 test cases including saturation, batch shape, and over-budget capping.
- **Empirical impact at our operating points (b_avg ∈ {2, 3, 4}, head_dim 128):** **None**. Re-running E1+E2 produced identical Q-weighted distortion numbers for v_waterfill and cca_waterfill — the saturation case (>16 bits/coord) does not trigger at these bit budgets on Qwen3-8B's spectra. The bug was real but inert at the operating points we evaluate.
- **Files modified:** `experiments/stage1/toolkit/metric_transform.py` (`water_fill` rewrite + new helper).
- **Rerun cost incurred:** E1+E2 (~30 sec, cached). E3/E4/E5 not rerun — bit allocations match pre-fix at our budgets.
- **Status:** applied (in working tree).

### F4 — Gate `gate_e1_e2.py` doesn't verify `P_K_inv · P_K ≈ I`  *[E1]*  **APPLIED**
- **Severity:** P2
- **Fix applied:** Added batch-matmul identity check across all 288 (layer, kv_head) heads with threshold 1e-3 max abs err. Reports the worst (layer, head) on failure.
- **Validation:** PASS with max abs err = 1.3e-4 (well under threshold; expected from float32 SVD precision).
- **Files modified:** `experiments/stage1/gates/gate_e1_e2.py`.
- **Status:** applied.

### F5 — Gate cross-check is single-(layer, head); doesn't catch head-specific bugs  *[E1]*  **APPLIED**
- **Severity:** P3
- **Fix applied:** Sample 3 random `(layer, kv_head)` pairs via seeded RNG (seed=42), cross-check all of them, fail on the worst case. Reports the chosen pairs and max relative error.
- **Validation:** PASS with max rel err = 7.4e-4 across pairs `(4, 4), (4, 5), (4, 6)`.
- **Files modified:** `experiments/stage1/gates/gate_e1_e2.py`.
- **Status:** applied.

### F6 — Histogram in `e1_r95_distribution.png` uses raw counts, undersells layer-0  *[E1]*  **APPLIED**
- **Severity:** P3
- **Fix applied:** Two-panel chart: left panel keeps raw counts (with note that layer 0 has fewer pairs by design); right panel adds density-normalized comparison so each group integrates to 1, allowing a fair shape comparison.
- **Files modified:** `experiments/stage1/scripts/make_e1_charts.py` (`plot_r95_distribution`).
- **Status:** applied (chart regenerated).

### F7 — Headline `r_{95}` table is missing layer-0 max breakdown  *[E1]*  **APPLIED**
- **Severity:** P3
- **Fix applied:** Headline table now has separate rows for `r_{95}` overall (28/68/109), layer 0 (28/45/70), and layers 1–35 (48/68/109), plus an explicit row pointing to the `(layer=8, kv_head=5)` location of the overall max.
- **Files modified:** `notes/stage1/stage1e_cca_vs_waterfill/e1_canonical_correlation_spectrum.md` (section 4 headline table).
- **Status:** applied.

---

## Earlier applied fixes (pre-batch)

### A1 — Layer-0 spectrum description had direction inverted  *[E1]*
- **Severity:** P1 (analysis claim was backwards in the original note)
- **Surfaced in:** E1 chart generation revealed the actual numbers
- **Description:** Original E1 note claimed layer 0 had a *flatter* spectrum and *higher* `r_{95}`. Actual data shows layer 0 has a *steeper* spectrum (sink-dominated) and *lower* `r_{95}` (median 47 vs 68 for layers 1–35).
- **Fix applied:** Rewrote section 5 (Q1, Q2 analysis), updated chart titles in `make_e1_charts.py`, regenerated charts.
- **Status:** applied.

---

## Issues to watch for in upcoming E2–E5 reviews

When we get to those experiments, specifically check:

- **(E2)** Bennett's `D = σ² · 2^{-2b}` reduces to `σ²` at b=0. Verify whether the simulation treats this as "Gaussian residual with variance σ²" vs the actual quantizer (which outputs 0). For pure Q-weighted distortion these match, but worth confirming.
- **(E3)** Whether `evaluate_method_on_example` validates the reconstruction identity for non-V3 methods. If `PerCoordCompressor`'s forward/inverse maps disagree, geometry distortion would be artificially high.
- **(E4)** Beyond F1: whether the calibration uses prefill positions only (it does — sliced at `:prompt_length`).
- **(E5)** Whether decode-phase Q comes from the post-RoPE tensor at the correct positions (i.e., `q_post[:, :, prompt_length:total_length, :]` with no off-by-one).
- **All experiments:** whether the Stage 1D V3 cross-check actually compares numbers within ±5% on geometry distortion (gate_e3's INFO log mentions this but doesn't enforce it).

---

## Naming convention for fix commits

When applying fixes, reference the ID(s) in the commit message:
> `Stage 1E: fix Σ_Q convention in E4 calibration (F1)`

If multiple fixes ship together: `Stage 1E: apply F1, F4, F5 (Σ_Q convention + gate hardening)`.
