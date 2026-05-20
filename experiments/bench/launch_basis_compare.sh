#!/bin/bash
# Phase 7 basis comparison: deployed cca_stats.pt (24-example calibration) vs
# new cca_stats_longbench_compact8_n400.pt (400-example pooled calibration).
#
# Tasks: 4 LongBench (multi-doc QA + single-doc QA + summarization).
# Configs: oracle, jointqk@OLD@K∈{2,4}V=3, jointqk@NEW@K∈{2,4}V=3,
#          turboquant@K∈{2,4}V=3 (calibration-independent baseline).
# Fraction: 0.5 (≈100 samples/task) — sufficient to resolve ≥1.5pp F1 deltas.
#
# 4 tasks × 7 configs = 28 cells. ~30-60 min wall on 6 GPUs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

GPUS="${GPUS:-0,1,2,3,4,5}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
FRACTION="${EVAL_FRACTION:-0.5}"

MODEL="Qwen/Qwen3-8B"
CCA_OLD="${REPO_ROOT}/artifacts/bases/cca_stats.pt"
CCA_NEW="${REPO_ROOT}/artifacts/bases/cca_stats_longbench_compact8_n400.pt"
VST_OLD="${REPO_ROOT}/artifacts/v_bases/v_stats.pt"
VST_NEW="${REPO_ROOT}/artifacts/v_bases/v_stats_longbench_compact8_n400.pt"

V_LOCK_FILE="${REPO_ROOT}/artifacts/v_bases/v_lock.txt"
V_METHOD=$(grep -oP 'V_METHOD=\K\S+' "$V_LOCK_FILE")
V_BITS=$(grep -oP 'V_BITS=\K\d+' "$V_LOCK_FILE")
COMPRESS_DECODE=False

OUT_BASE="${REPO_ROOT}/artifacts/downstream_basis_compare"
LOG_DIR="${REPO_ROOT}/experiments/logs/phase7_basis_compare"

echo "[$(date '+%H:%M:%S')] Phase 7 basis compare: V_METHOD=$V_METHOD V_BITS=$V_BITS FRACTION=$FRACTION"
echo "[$(date '+%H:%M:%S')] OLD cca: $CCA_OLD"
echo "[$(date '+%H:%M:%S')] NEW cca: $CCA_NEW"
echo "[$(date '+%H:%M:%S')] OLD v:   $VST_OLD"
echo "[$(date '+%H:%M:%S')] NEW v:   $VST_NEW"

mkdir -p "$OUT_BASE" "$LOG_DIR"
CMDS="$LOG_DIR/commands.txt"
: > "$CMDS"

VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"
PY_PATH="${REPO_ROOT}:${REPO_ROOT}/vendor/kvpress"
ALLOC_CONF='PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'

# Multi-doc QA (most likely to reward Q-K-aware allocation per the v6 report's
# observation about v5's narrativeqa / hotpotqa lead) + single-doc QA + summary.
# qasper and qmsum overlap with the KIVI 8-task subset — direct comparability
# to v6 numbers in `notes/phase7_v6_results_report.md`.
TASKS=(hotpotqa musique qasper qmsum)

emit() {
    local press="$1" json="$2" label="$3" task="$4"
    local cmd="cd ${REPO_ROOT}/vendor/kvpress/evaluation && ${ALLOC_CONF} PYTHONPATH=${PY_PATH} ${VENV_PYTHON} -u evaluate.py \
        --press_name=${press} \
        --press_kwargs='${json}' \
        --model='${MODEL}' \
        --dataset=longbench \
        --data_dir=${task} \
        --fraction=${FRACTION} \
        --output_dir=${OUT_BASE}/${label}_${task} # label=${label}_${task}"
    echo "$cmd" >> "$CMDS"
}

emit_oracle() {
    local task="$1"
    local cmd="cd ${REPO_ROOT}/vendor/kvpress/evaluation && ${ALLOC_CONF} PYTHONPATH=${PY_PATH} ${VENV_PYTHON} -u evaluate.py \
        --press_name=no_press \
        --compression_ratio=0.0 \
        --model='${MODEL}' \
        --dataset=longbench \
        --data_dir=${task} \
        --fraction=${FRACTION} \
        --output_dir=${OUT_BASE}/full_precision_${task} # label=oracle_${task}"
    echo "$cmd" >> "$CMDS"
}

for task in "${TASKS[@]}"; do
    emit_oracle "$task"
    for kb in 2 4; do
        json_jq_old="{\"cca_stats_path\": \"${CCA_OLD}\", \"v_stats_path\": \"${VST_OLD}\", \"v_method\": \"${V_METHOD}\", \"k_bits\": ${kb}, \"v_bits\": ${V_BITS}, \"compress_decode\": ${COMPRESS_DECODE}, \"quantize_k\": True, \"quantize_v\": True}"
        emit "jointqk" "$json_jq_old" "jointqk_OLD_k${kb}_v${V_BITS}" "$task"

        json_jq_new="{\"cca_stats_path\": \"${CCA_NEW}\", \"v_stats_path\": \"${VST_NEW}\", \"v_method\": \"${V_METHOD}\", \"k_bits\": ${kb}, \"v_bits\": ${V_BITS}, \"compress_decode\": ${COMPRESS_DECODE}, \"quantize_k\": True, \"quantize_v\": True}"
        emit "jointqk" "$json_jq_new" "jointqk_NEW_k${kb}_v${V_BITS}" "$task"

        json_tq="{\"k_bits\": ${kb}, \"v_bits\": ${V_BITS}, \"compress_decode\": ${COMPRESS_DECODE}}"
        emit "turboquant" "$json_tq" "turboquant_k${kb}_v${V_BITS}" "$task"
    done
done

n_jobs=$(wc -l < "$CMDS")
echo "[$(date '+%H:%M:%S')] queued $n_jobs jobs in $CMDS"

if [[ "${1:-}" == "--dry-run" ]]; then
    head -5 "$CMDS"
    echo "..."
    echo "[$(date '+%H:%M:%S')] dry run — not executing"
    exit 0
fi

${VENV_PYTHON} experiments/bench/parallel_launcher.py \
    --commands-file "$CMDS" \
    --log-dir "$LOG_DIR" \
    --gpus "$GPUS" \
    --jobs-per-gpu "$JOBS_PER_GPU" \
    --label-from-comment
