#!/bin/bash
# Phase 2c: per-task basis evaluation.
# Each task uses its OWN cca_stats.pt + v_stats.pt fitted on its 50 train examples.
# Compare to pooled-400 (NEW) basis to test whether task-matched calibration helps.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

GPUS="${GPUS:-0,1,2,3,4,5}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
FRACTION="${EVAL_FRACTION:-0.5}"

MODEL="Qwen/Qwen3-8B"
PER_TASK_CCA_DIR="${REPO_ROOT}/artifacts/bases/per_task"
PER_TASK_VST_DIR="${REPO_ROOT}/artifacts/v_bases/per_task"

# Override v_lock — v_random is the actual best V for these tasks (per Phase 1b
# V-method ablation, which gave +6 pp at K=2 over the v_eigen_uniform default).
V_METHOD=v_random
V_BITS=3
COMPRESS_DECODE=False

OUT_BASE="${REPO_ROOT}/artifacts/downstream_basis_compare"
LOG_DIR="${REPO_ROOT}/experiments/logs/phase7_per_task_basis"

echo "[$(date '+%H:%M:%S')] Phase 2c per-task basis: V_METHOD=$V_METHOD V_BITS=$V_BITS FRACTION=$FRACTION"

mkdir -p "$OUT_BASE" "$LOG_DIR"
CMDS="$LOG_DIR/commands.txt"
: > "$CMDS"

VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"
PY_PATH="${REPO_ROOT}:${REPO_ROOT}/kvpress"
ALLOC_CONF='PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'

TASKS=(hotpotqa musique qasper qmsum)

emit() {
    local press="$1" json="$2" label="$3" task="$4"
    local cmd="cd ${REPO_ROOT}/kvpress/evaluation && ${ALLOC_CONF} PYTHONPATH=${PY_PATH} ${VENV_PYTHON} -u evaluate.py \
        --press_name=${press} \
        --press_kwargs='${json}' \
        --model='${MODEL}' \
        --dataset=longbench \
        --data_dir=${task} \
        --fraction=${FRACTION} \
        --output_dir=${OUT_BASE}/${label}_${task} # label=${label}_${task}"
    echo "$cmd" >> "$CMDS"
}

for task in "${TASKS[@]}"; do
    cca_path="${PER_TASK_CCA_DIR}/cca_stats_${task}.pt"
    vst_path="${PER_TASK_VST_DIR}/v_stats_${task}.pt"
    if [[ ! -f "$cca_path" ]]; then
        echo "ERROR: missing $cca_path" >&2
        exit 1
    fi
    for kb in 2 4; do
        json_jq="{\"cca_stats_path\": \"${cca_path}\", \"v_stats_path\": \"${vst_path}\", \"v_method\": \"${V_METHOD}\", \"k_bits\": ${kb}, \"v_bits\": ${V_BITS}, \"compress_decode\": ${COMPRESS_DECODE}, \"quantize_k\": True, \"quantize_v\": True}"
        emit "jointqk" "$json_jq" "jointqk_TASK_k${kb}_v${V_BITS}" "$task"
    done
done

n_jobs=$(wc -l < "$CMDS")
echo "[$(date '+%H:%M:%S')] queued $n_jobs jobs"

${VENV_PYTHON} experiments/bench/parallel_launcher.py \
    --commands-file "$CMDS" \
    --log-dir "$LOG_DIR" \
    --gpus "$GPUS" \
    --jobs-per-gpu "$JOBS_PER_GPU" \
    --label-from-comment
