# Stage 1E CCA vs Water-Filling — Review Packet

**Status:** SUCCESS
**Generated:** 2026-04-29 07:39:27

## Headline (b_avg = 3, layer-0-excluded)

**See E3 summaries**: `/vault/amir/efficient-llm/teamily-project/artifacts/stage1/cca_vs_waterfill_study/e3/*_summary.json` (search for `top1_prefill[l0excl_mean]`).

## E1+E2 — Closed-form simulation
- Spectrum diagnostic: `figures/spectrum_overlay.png`, `figures/r95_heatmap.png`, `figures/spectrum_per_layer.png`
- Pareto frontier: `figures/sim_pareto.png`
- Per-layer log-ratios: `figures/sim_per_layer_lines.png`
- Per-method heatmaps: `figures/sim_log_ratio_*_b3.png`
- Metrics: `metrics_e1_e2.json`

## E3 — Real quantization (b_avg ∈ {2, 3, 4}, r=64)
- `e3_b2_r64_summary.json`
- `e3_b3_r64_summary.json`
- `e3_b4_r64_summary.json`
- `e3_smoke_summary.json`
- `e3_timing_summary.json`

## E4a — Cross-task generalization (calibrate on one config, eval on three)
- `e4a_calib_hotpotqa_b3_r64_summary.json`
- `e4a_calib_passage_retrieval_en_b3_r64_summary.json`
- `e4a_calib_qasper_b3_r64_summary.json`

## E4b — Within-task LOO (24 folds, 8 per config)
- `e4b_hotpot_loo10_b3_r64_summary.json`
- `e4b_hotpot_loo11_b3_r64_summary.json`
- `e4b_hotpot_loo12_b3_r64_summary.json`
- `e4b_hotpot_loo13_b3_r64_summary.json`
- `e4b_hotpot_loo14_b3_r64_summary.json`
- `e4b_hotpot_loo15_b3_r64_summary.json`
- `e4b_hotpot_loo8_b3_r64_summary.json`
- `e4b_hotpot_loo9_b3_r64_summary.json`
- `e4b_passage_loo16_b3_r64_summary.json`
- `e4b_passage_loo17_b3_r64_summary.json`
- `e4b_passage_loo18_b3_r64_summary.json`
- `e4b_passage_loo19_b3_r64_summary.json`
- `e4b_passage_loo20_b3_r64_summary.json`
- `e4b_passage_loo21_b3_r64_summary.json`
- `e4b_passage_loo22_b3_r64_summary.json`
- `e4b_passage_loo23_b3_r64_summary.json`
- `e4b_qasper_loo0_b3_r64_summary.json`
- `e4b_qasper_loo1_b3_r64_summary.json`
- `e4b_qasper_loo2_b3_r64_summary.json`
- `e4b_qasper_loo3_b3_r64_summary.json`
- `e4b_qasper_loo4_b3_r64_summary.json`
- `e4b_qasper_loo5_b3_r64_summary.json`
- `e4b_qasper_loo6_b3_r64_summary.json`
- `e4b_qasper_loo7_b3_r64_summary.json`
- `e4b_smoke_summary.json`

## E5 — Decode-phase Q (pulled from E3 outputs with --query-phase both)

Search the e3 `*_summary.json` for `top1_decode[l0excl_mean]` vs `top1_prefill[l0excl_mean]`.
A small gap means decode-phase queries are well-served by prefill-time calibration.

## Logs
- Pipeline: `/vault/amir/efficient-llm/teamily-project/experiments/stage1/logs/pipeline.log`
- Per-run: `/vault/amir/efficient-llm/teamily-project/experiments/stage1/logs/<run_name>.log` (with `*.summary.json` on success or `*.FAILED` on failure)
- Registry: `/vault/amir/efficient-llm/teamily-project/experiments/stage1/logs/_registry.tsv`

## What to look at first (5 minutes)

1. `/vault/amir/efficient-llm/teamily-project/artifacts/stage1/cca_vs_waterfill_study/figures/sim_pareto.png` — closed-form Pareto across methods.
2. `/vault/amir/efficient-llm/teamily-project/artifacts/stage1/cca_vs_waterfill_study/e3/e3_b3.0_r64_summary.json` — headline real-quantization results.
3. The Stage 1E report under `notes/stage1/stage1e_cca_vs_waterfill_report.md` (if pipeline succeeded).
