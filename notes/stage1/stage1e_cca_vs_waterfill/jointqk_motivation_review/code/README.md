# Empirical-verification code for `jointqk_motivation.md` §10

This directory contains the analysis script that produces the numbers used in
§10 of [`../jointqk_motivation.md`](../jointqk_motivation.md). The script
enforces the **train/test split discipline** the math-review forum identified
as missing in the previous numbers: bases are extracted from train data only,
all evaluation uses held-out test data.

## Files

- `run_empirical_verification.py` — main script; produces
  `../results/empirical_results.json` and `../results/empirical_run.log`.
- `README.md` — this file.

## Data sources

Fixed by the script's defaults; no need to specify unless reproducing on a
different bundle.

| Role | Path | Origin |
|---|---|---|
| Train bases | `artifacts/stage1/cca_vs_waterfill_study/cca_stats_longbench_compact8_n400.pt` | Built by `experiments/stage1/scripts/build_calibration_artifacts_from_pool.py` from the **400 LongBench train prompts**. Contains `Σ_Q^train`, `Σ_K^train`, `V_Q` (= `mq_eigvecs`), `V_h`, `R_sym`, `P_K`, `P_K_inv`. |
| Test moments | `artifacts/stage1/calibration/longbench_compact8_qkv/02_stats/aggregate.pt` (group `pooled\|test`, plus per-task groups) | Built by `experiments/stage1/calibration/` pipeline from the **80 held-out test prompts** (10 per task × 8 LongBench tasks). |
| Test raw Q/K | `artifacts/stage1/calibration/longbench_compact8_qkv/01_raw/shard_*/longbench__<task>__row<NNNNN>__test.pt` | Same pipeline; per-row `q_post`, `k_post`, `v` tensors. |

The 400 train prompts and 80 test prompts are completely disjoint per
`00_split/manifest.json`.

## Run

From the project root, with the project venv:

```bash
./.venv/bin/python notes/stage1/stage1e_cca_vs_waterfill/jointqk_motivation_review/code/run_empirical_verification.py
```

Wall time: ~25–40 min on one A100 (Phase B dominates). For a quick smoke run
that skips real quantization:

```bash
./.venv/bin/python notes/stage1/stage1e_cca_vs_waterfill/jointqk_motivation_review/code/run_empirical_verification.py --skip-quant
```

Output: `../results/empirical_results.json`, `../results/empirical_run.log`.

## What the script computes

**Phase A — pure-statistics (predicted-only).** No quantization, just
linear-algebra on calibration moments.

- **§10.1 geomean predictions.** For each of {`V_Q`, `V_K`, `V_h`, `R_sym`,
  random orth} (bases extracted from train), compute the per-(layer, kv_head)
  product geomean `(Π_j w_j σ_j²)^{1/d}` evaluated on **test** Σ_Q, Σ_K. Take
  the layer-0-excluded mean across heads.
- **§8.2 headroom.** R_sym product geomean / Hadamard floor on test.
- **§10.4 cross-task spread.** Per-task predicted geomean for each method,
  using each test task's pooled moments.
- **§5 active-set sizes.** Reverse-water-fill on R_sym water-fill weights
  evaluated on test moments; min/median/max active count across 288 heads at
  `b_avg ∈ {2, 3, 4}`.
- **§7 / §8 M-indefiniteness.** Eigendecomposition of
  `M = ½(Σ_Q^train Σ_K^train + Σ_K^train Σ_Q^train)` across all 288 heads;
  count of negative eigenvalues, min/max ratio per head.

**Phase B — real quantization (slow).** Loads `n_test_files_quant` raw test
files (default 16, round-robin across the 8 tasks), runs the actual
`PerCoordCompressor` (and `Stage1MSECompressor` for V3) per (layer, kv_head)
using the train basis, computes:

- **Q-weighted geometry distortion** = `(1/L) Σ_t (Δk_t)^⊤ Σ_Q^test Δk_t / d`
  per (layer, kv_head, file), then aggregated.
- **Top-1 retention** = fraction of (test query, prefill position) pairs whose
  argmax against the compressed K matches the argmax against full-precision K.

Methods evaluated at `b_avg = 3`, `r = 64`:
`v3`, `v_truncate`, `v_waterfill`, `cca_uniform`, `cca_waterfill`,
`cca_orth_uniform`, `cca_orth_waterfill`, `r_sym_uniform`, `r_sym_waterfill`.

Then computes:
- **§10.2 predicted vs measured.** Predicted geomean ratio
  (R_sym / V_Q etc.) vs measured geo-distortion ratio.
- **§10.2 per-layer Pearson** between predicted geomean and measured
  geo-distortion across the 35 non-zero layers.

## Output schema

`empirical_results.json` is a single JSON object with:

```
{
  "calibration_train": {...},
  "evaluation_test": {...},
  "config": {...},
  "section_10_1_geomean_predictions": {<basis>: {Q_side, K_side, product}},
  "section_8_2_headroom": {R_sym_geomean, hadamard_floor, R_sym_over_floor},
  "section_10_2_predicted_ratios": {R_sym_over_V_Q, V_Q_over_TQ_proxy, ...},
  "section_8_M_indefinite": {min_eigenvalue_overall, n_heads_with_negative, ...},
  "section_5_active_set_R_sym": {<b_avg>: {min, median, max, mean}},
  "section_10_4_cross_task_predicted_spread": {<basis>: {per_task_geomeans, spread_relative, ...}},
  "phase_B_real_quantization": {
    "b_avg": 3,
    "methods": {<method>: {geo_l0excl_mean, top1_l0excl_mean, ...}}
  },
  "section_10_2_predicted_vs_measured": {...},
  "section_10_2_per_layer_pearson": {...}
}
```

## Reproducibility

The script is deterministic given the same `--seed` (default 42); the only
non-determinism is in CUDA float-summation, which affects results in the
last 2–3 decimal places.
