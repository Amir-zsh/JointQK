# Resume Prompt — Stage 1E Review (E4, E5, then fixes/reruns)

> Paste this entire file (or its contents) into a fresh chat to continue the per-experiment review of Stage 1E. Working directory: `/vault/amir/efficient-llm/teamily-project`.

---

## Context

This is part of a KV-cache compression research project. Stage 1E is the **CCA vs water-filling comparative study**: five experiments (E1-E5) comparing five `(basis x allocation)` methods — `v3`, `v_truncate`, `v_waterfill`, `cca_uniform`, `cca_waterfill` — at bit budgets `b_avg in {2, 3, 4}`, on Qwen3-8B with the LongBench-E 24-example bundle (3 configs x 8 examples; 81,223 prefill tokens).

We are reviewing each experiment carefully:

1. Re-derive the problem formulation, proposed approach, and code path.
2. Generate result charts in addition to the auto-generated diagnostic figures.
3. Write a self-contained review note per experiment with embedded charts and "Key takeaway" lines.
4. Track bugs in `fixes_to_apply.md`.
5. Do **not** apply fixes unless explicitly asked; the workflow is review -> batch fixes -> apply together -> rerun affected experiments.
6. When fixes are applied, verify them rigorously before claiming correctness.

**Current state:** E1, E2, and E3 have been reviewed and re-reviewed. F11 is now fixed, rerun, merged, and gate-verified for E3/E4/E5 `cca_waterfill`. E4/E5 narrative review is still the next documentation task, but the stale-value blocker is gone.

---

## Required Reading Before Starting

Read these files first, in order:

1. `notes/stage1/stage1e_cca_vs_waterfill/fixes_to_apply.md`
   - Current bug tracker. Open fixes currently include F9-F13.
2. `notes/stage1/stage1e_cca_vs_waterfill/e1_canonical_correlation_spectrum.md`
   - E1 review. Updated to describe the method as **uncentered / second-moment CCA** rather than classical centered CCA.
3. `notes/stage1/stage1e_cca_vs_waterfill/e2_closed_form_simulation.md`
   - E2 review. Updated post-F8/post-F11: corrected `cca_waterfill` sim-vs-real is now apples-to-apples and reasonably aligned on geometry.
4. `notes/stage1/stage1e_cca_vs_waterfill/e3_real_quantization.md`
   - E3 review. Reports post-F11 merged E3 artifacts; corrected `cca_waterfill` is no longer stale.
5. `notes/stage1/stage1e_cca_vs_waterfill_note.md`
   - Cross-experiment hand-written summary. This still needs a post-E4/E5 narrative refresh.

Also useful:

- `experiments/stage1/run_cca_diagnostics.py` — E1/E2 driver.
- `experiments/stage1/run_cca_vs_waterfill_study.py` — E3/E4/E5 runner.
- `experiments/stage1/toolkit/per_coord_quantization.py` — real per-coordinate quantizer; F11 lives here.
- `experiments/stage1/toolkit/metric_transform.py` — `compute_cca_basis`, `water_fill`, `whitening_factor`.
- `experiments/stage1/gates/gate_e1_e2.py`, `experiments/stage1/gates/gate_e3.py`.

---

## What Has Been Done

### Reviews Completed

- **E1 — CCA-style second-moment spectrum diagnostic**
  - Review note: `notes/stage1/stage1e_cca_vs_waterfill/e1_canonical_correlation_spectrum.md`
  - Charts: `experiments/stage1/scripts/make_e1_charts.py` -> `artifacts/stage1/cca_vs_waterfill_study/report_charts/e1_*.png`
  - Important clarification: implementation uses uncentered raw second moments `E[qq^T]`, `E[kk^T]`, `E[qk^T]`, intentionally matching raw attention logits. F13 tracks broader docs cleanup.

- **E2 — closed-form rate-distortion simulation**
  - Review note: `notes/stage1/stage1e_cca_vs_waterfill/e2_closed_form_simulation.md`
  - Charts: `experiments/stage1/scripts/make_e2_charts.py` -> `report_charts/e2_*.png`
  - F8 fixed the CCA simulation weight: use `diag((P_K_inv)^T Sigma_Q P_K_inv)`, not `rho^2`.
  - E2 is internally corrected, and its `cca_waterfill` comparison to real E3 is now post-F11/apples-to-apples.

- **E3 — real per-coordinate quantization**
  - Review note: `notes/stage1/stage1e_cca_vs_waterfill/e3_real_quantization.md`
  - Charts: `experiments/stage1/scripts/make_e3_charts.py` -> `report_charts/e3_*.png`
  - Main post-F11 result: `v_waterfill` remains the best method at all bit budgets.
  - Corrected `cca_waterfill` now has much better geometry than the stale artifact but still underperforms V3 and V-waterfill on top-1.

### Verification Already Run

- `python experiments/stage1/gates/gate_e1_e2.py` — passes.
- `python experiments/stage1/gates/gate_e3.py` — passes.
- `python experiments/stage1/scripts/make_e1_charts.py` — passes.
- `python experiments/stage1/scripts/make_e2_charts.py` — passes.
- `python experiments/stage1/scripts/make_e3_charts.py` — passes.
- Manual identity check: `max_abs(P_K_inv @ P_K - I) = 1.31e-4` over all 288 heads.
- Manual Stage 1D V3 cross-check: E3 V3 layer-0-excluded geometry matches Stage 1D `baseline_raw` within 5% at b=2/3/4.
- Centered-vs-uncentered diagnostic: centered recompute changes layer-0-excluded `r95` from `48/68/109` to `49/69/109`, so E1 rank conclusions are stable; median `rho1` changes from `0.993` to `0.940`, so terminology matters.

---

## Current Fix Tracker Summary

Open fixes:

- **F11 (P1, E3/E4/E5):** real `cca_waterfill` artifacts used pre-F8 `rho^2`.
  - File: `experiments/stage1/toolkit/per_coord_quantization.py`.
  - Code is now fixed and verified to use `diag((P_K_inv)^T Sigma_Q P_K_inv)`.
  - E3/E4/E5 `cca_waterfill` artifacts have now been rerun into `e3_f11`, `e4a_f11`, `e4b_f11` and merged into canonical summaries.
  - Does **not** affect `v3`, `v_truncate`, `v_waterfill`, or `cca_uniform`.
  - Post-merge gates passed: `gate_e3.py`, `gate_e4.py`, `gate_e5.py`.
  - Post-F11 E3 b=3: `cca_waterfill` top-1 = 0.535, geo = 0.0965, real log₂(geo/geo_v3) = -2.24; `v_waterfill` still wins top-1 at 0.760.
  - Two verification scripts exist:
    - `experiments/stage1/scripts/verify_f11_allocation.py` — synthetic + all-288-heads allocation match.
    - `experiments/stage1/scripts/verify_f11_roundtrip.py` — Monte-Carlo closed-form-vs-real roundtrip MSE; 18/18 within 1.4% rel err across 3 layers × {2,3,4} bits × {CCA, V} water-fill. Confirms allocation, codebook scaling, forward/inverse maps, and noise propagation are all internally consistent.

- **F14 (P3, E2/E3 docs):** Σ_Q regularization is inconsistent between V real branch (regularized eigvals), CCA real branch (un-regularized `Mq`), and E2 simulation (regularized rebuild). Sub-percent effect; doc/alignment fix only. See `fixes_to_apply.md` for details.

- **F9 (P2, E3 gate):** E3 gate does not explicitly assert `P_K_inv @ P_K ≈ I` and row-vector pairing for CCA maps.
  - Current artifacts manually pass.
  - Gate-only fix.

- **F10 (P3, E3 gate):** E3 gate does not enforce the Stage 1D V3 baseline cross-check.
  - Manual check passes.
  - Gate-only fix.

- **F12 (P3, E3 reporting):** E3 bootstrap CI is computed over all layers, while headline means are layer-0-excluded.
  - Reporting-only; raw/headline metrics unaffected.
  - Can likely regenerate summaries from row files rather than rerun compression.

- **F13 (P3, docs):** terminology says classical centered CCA in some places, but implementation uses uncentered second moments.
  - E1/E2/E3 review notes partially corrected.
  - Broader docs/report/code-comment cleanup remains.

Applied fixes in working tree include F1-F8 plus A1. See `fixes_to_apply.md` for details.

Important: F11 code has been applied and the affected artifacts have been rerun/merged. Do not apply other fixes unless the user asks.

---

## What Remains

### 1. Check E4 Rerun Status

E4 is split into:

- **E4a:** cross-task calibration: calibrate from one of `qasper` / `hotpotqa` / `passage_retrieval_en`, evaluate on all 24.
- **E4b:** within-task leave-one-out: 24 folds, hold out one example and calibrate from the rest.

F1 fixed `_accumulate_calibration_stats` to match E1/E3's `Sigma_Q` convention, but the E4 artifacts must be post-F1 before reviewing.

Check:

```bash
ls -la /vault/amir/efficient-llm/teamily-project/experiments/stage1/logs/gate_e4_post_f1.log
bash /vault/amir/efficient-llm/teamily-project/experiments/stage1/scripts/watch_runs.sh --once
tail -n 80 /vault/amir/efficient-llm/teamily-project/experiments/stage1/logs/launcher_e4a_post_f1.log
tail -n 80 /vault/amir/efficient-llm/teamily-project/experiments/stage1/logs/launcher_e4b_post_f1.log
```

If the marker is absent, do not touch these files while the rerun might be active:

- `experiments/stage1/run_cca_vs_waterfill_study.py`
- `experiments/stage1/toolkit/metric_transform.py`
- `experiments/stage1/toolkit/per_coord_quantization.py`
- `experiments/stage1/gates/gate_e4.py`
- `experiments/stage1/write_stage1e_report.py`

### 2. E4 Review

Write:

- `notes/stage1/stage1e_cca_vs_waterfill/e4_generalization.md`
- `experiments/stage1/scripts/make_e4_charts.py`

Review checks:

- Compare pre-F1 vs post-F1 numbers for `v_waterfill` cross-task / LOO.
- Check E4a "in-domain diagonal" against corresponding E3 subsets.
- Check E4b LOO variance across folds.
- Use the merged post-F11 `cca_waterfill` E4 rows; canonical E4 artifacts now include the corrected trace-formula allocation.

Useful charts:

- Cross-task heatmap: calibration source x eval target.
- LOO fold variance plot.
- Comparison to in-domain E3 numbers.
- Allocation-shift diagnostic if available.

### 3. E5 Review

Write:

- `notes/stage1/stage1e_cca_vs_waterfill/e5_decode_phase.md`
- `experiments/stage1/scripts/make_e5_charts.py`

E5 piggybacks on E3 via `query_phase=both`: decode-phase queries are positions after `prompt_length`, evaluated against compressed prefill keys.

Checks:

- Verify decode slicing: `q_post[:, :, prompt_length:, :]` from captured tensors. Be careful about `captured_length = total_length - 1`; use actual captured tensor length, not a naive `total_length` bound.
- Compare decode vs prefill top-1 across methods and bit budgets.
- Decode lengths are small, so pooling/weighting matters.
- Use the merged post-F11 decode `cca_waterfill` rows from canonical E3 summaries.

Useful charts:

- Decode-vs-prefill top-1 scatter.
- Gap distribution per method.
- Decode query count histogram.

### 4. After E4/E5

- Refresh `notes/stage1/stage1e_cca_vs_waterfill_note.md` with post-F8, post-F1, and post-F11 conclusions.
- Confirm `gate_integration` passes with the relevant artifacts.
- If the user asks to apply remaining fixes, F9/F10/F12/F13/F14 are still candidates; F11 is already applied, rerun, and merged.
- Do not commit unless the user explicitly approves.

---

## Conventions To Follow

1. Per-experiment review files go under `notes/stage1/stage1e_cca_vs_waterfill/eN_*.md`.
2. Each review should include:
   - Problem formulation
   - Proposed approach
   - Setup and code with file/line refs
   - Results with embedded charts and "Key takeaway" lines
   - Analysis
   - Caveats / known issues
   - Implications
   - Artifacts / regen commands
3. Dedicated chart scripts go under `experiments/stage1/scripts/make_eN_charts.py`.
4. Track bugs in `fixes_to_apply.md`; new IDs continue from F13.
5. Do not apply fixes during review unless explicitly asked.
6. Report layer-0-excluded headline numbers; mention layer 0 separately.
7. Use "uncentered second-moment CCA" or "CCA-style second-moment spectrum" for E1 unless discussing classical CCA as background.
8. For `cca_waterfill`, always state whether the artifact is as-run old `rho^2` allocation or corrected trace-formula allocation.
9. Do not commit without explicit approval.

---

## Suggested First Message In A New Chat

> Continue the Stage 1E review from the updated resume prompt. Read `fixes_to_apply.md`, the E1/E2/E3 review notes, then start the E4 generalization review and chart script from the merged post-F1/post-F11 canonical artifacts. Do not apply additional fixes or touch active rerun files unless I explicitly ask.

---

## Compact Status Snapshot

- Date last updated: 2026-04-30
- Working dir: `/vault/amir/efficient-llm/teamily-project`
- Branch: `master`
- Last known commit: `9c8c458 Stage 1E: CCA vs water-filling end-to-end study`
- Working tree: dirty; E1/E2/E3 review notes, chart scripts, report charts, diagnostics, and fixes are in the working tree.
- Reviews done: E1, E2, E3.
- Reviews remaining: E4, E5, then cross-experiment summary refresh.
- Critical pending rerun: none. F11 `cca_waterfill` real allocator mismatch is fixed, rerun, merged, and gate-verified.
- Do not commit without user approval.
