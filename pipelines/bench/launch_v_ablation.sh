#!/bin/bash
# Phase 1b: V-method ablation on JointQK.
# Tests whether the K=4 F1 loss is V-side rather than K-side.
# JointQK currently uses v_eigen_uniform (per v_lock.txt). This sweep replaces
# V with v_random (TurboQuant's V method) at K∈{2,4}, holding the K basis fixed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

GPUS="${GPUS:-0,1,2,3,4,5}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
FRACTION="${EVAL_FRACTION:-0.5}"

MODEL="Qwen/Qwen3-8B"
CCA_NEW="${REPO_ROOT}/artifacts/bases/jointqk_longbench_compact8_n400.pt"
VST_NEW="${REPO_ROOT}/artifacts/v_bases/v_stats_longbench_compact8_n400.pt"

V_BITS=3
COMPRESS_DECODE=False

OUT_BASE="${REPO_ROOT}/artifacts/downstream_basis_compare"
LOG_DIR="${REPO_ROOT}/logs/phase7_v_ablation"

echo "[$(date '+%H:%M:%S')] Phase 1b V ablation: V_BITS=$V_BITS FRACTION=$FRACTION"

mkdir -p "$OUT_BASE" "$LOG_DIR"
CMDS="$LOG_DIR/commands.txt"
: > "$CMDS"

VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"
PY_PATH="${REPO_ROOT}:${REPO_ROOT}/vendor/kvpress"
ALLOC_CONF='PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'

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

for task in "${TASKS[@]}"; do
    for kb in 2 4; do
        # JointQK NEW basis with v_random (instead of v_eigen_uniform)
        json_jq_vrand="{\"cca_stats_path\": \"${CCA_NEW}\", \"v_stats_path\": \"${VST_NEW}\", \"v_method\": \"v_random\", \"k_bits\": ${kb}, \"v_bits\": ${V_BITS}, \"compress_decode\": ${COMPRESS_DECODE}, \"quantize_k\": True, \"quantize_v\": True}"
        emit "jointqk" "$json_jq_vrand" "jointqk_NEW_vrand_k${kb}_v${V_BITS}" "$task"
    done
done

n_jobs=$(wc -l < "$CMDS")
echo "[$(date '+%H:%M:%S')] queued $n_jobs jobs"

${VENV_PYTHON} pipelines/bench/parallel_launcher.py \
    --commands-file "$CMDS" \
    --log-dir "$LOG_DIR" \
    --gpus "$GPUS" \
    --jobs-per-gpu "$JOBS_PER_GPU" \
    --label-from-comment
