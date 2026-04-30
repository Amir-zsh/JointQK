#!/bin/bash
# Rerun E4 (cross-task + LOO) after applying F1 (Σ_Q convention fix).
# Total wall clock: ~2.2 hours on 4 GPUs.
#
# Stages:
#   1. E4a cross-task (3 calibration sources × 24 examples, 3-way parallel on GPUs 0,1,2)        ~2 hr
#   2. E4b within-task LOO (24 folds, 4-way parallel × 6 batches on GPUs 0,1,2,3)               ~12 min
#   3. gate_e4 (validates summaries)                                                             seconds
#   4. write_stage1e_report (regenerates notes/stage1/stage1e_cca_vs_waterfill_report.md)        seconds
#
# Usage (in tmux):
#   bash experiments/stage1/scripts/rerun_e4_post_f1.sh
#
# Per-stage logs:
#   experiments/stage1/logs/launcher_e4a_post_f1.log
#   experiments/stage1/logs/launcher_e4b_post_f1.log
#   experiments/stage1/logs/gate_e4_post_f1.log
#   experiments/stage1/logs/write_report_post_f1.log
#
# Live monitoring (in another tmux pane):
#   watch -n 5 bash experiments/stage1/scripts/watch_runs.sh --once
#
# Retry just the failed runs:
#   bash experiments/stage1/scripts/launch_cca_study.sh --phase e4a --only-failed
#   bash experiments/stage1/scripts/launch_cca_study.sh --phase e4b --only-failed

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO}"

LOGS="${REPO}/experiments/stage1/logs"
mkdir -p "${LOGS}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Rerun E4 post-F1: starting ==="
echo "Repo: ${REPO}"
echo "Logs: ${LOGS}"
echo

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [1/4] E4a cross-task (3-way parallel, ~2 hr)"
bash experiments/stage1/scripts/launch_cca_study.sh --phase e4a --gpus 0,1,2 2>&1 | tee "${LOGS}/launcher_e4a_post_f1.log"

echo
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [2/4] E4b within-task LOO (4-way parallel × 6 batches, ~12 min)"
bash experiments/stage1/scripts/launch_cca_study.sh --phase e4b --gpus 0,1,2,3 2>&1 | tee "${LOGS}/launcher_e4b_post_f1.log"

echo
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [3/4] gate_e4"
python -u -m experiments.stage1.gates.gate_e4 2>&1 | tee "${LOGS}/gate_e4_post_f1.log"

echo
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [4/4] write_stage1e_report (regenerate auto report)"
python -u -m experiments.stage1.write_stage1e_report 2>&1 | tee "${LOGS}/write_report_post_f1.log"

echo
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Rerun E4 post-F1: complete ==="
echo "Refreshed report: notes/stage1/stage1e_cca_vs_waterfill_report.md"
echo "Updated artifacts: artifacts/stage1/cca_vs_waterfill_study/{e4a,e4b}/"
