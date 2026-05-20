#!/bin/bash
# JointQK K-side + TurboQuant's actual V compressor (v_turboquant).
# Distinct from v_random (centered random Hadamard) — v_turboquant is uncentered,
# matching what the standalone TurboQuant press uses internally.
# 4 tasks × K∈{2,4} × V=3 = 8 cells.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

GPUS="${GPUS:-0,1,2,3,4,5}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
FRACTION="${EVAL_FRACTION:-0.5}"

MODEL="Qwen/Qwen3-8B"
CCA_NEW="${REPO_ROOT}/artifacts/bases/cca_stats_longbench_compact8_n400.pt"

V_METHOD=v_turboquant
V_BITS=3
COMPRESS_DECODE=False

OUT_BASE="${REPO_ROOT}/artifacts/downstream_basis_compare"
LOG_DIR="${REPO_ROOT}/experiments/logs/phase7_v_turboquant"

echo "[$(date '+%H:%M:%S')] Phase 1b-extra v_turboquant ablation: V_BITS=$V_BITS FRACTION=$FRACTION"

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

# v_turboquant doesn't use v_stats_path (compressor is internal). We still pass
# one to satisfy the press constructor; it gets ignored when v_method=v_turboquant.
VST_NEW="${REPO_ROOT}/artifacts/v_bases/v_stats_longbench_compact8_n400.pt"

for task in "${TASKS[@]}"; do
    for kb in 2 4; do
        json="{\"cca_stats_path\": \"${CCA_NEW}\", \"v_stats_path\": \"${VST_NEW}\", \"v_method\": \"${V_METHOD}\", \"k_bits\": ${kb}, \"v_bits\": ${V_BITS}, \"compress_decode\": ${COMPRESS_DECODE}, \"quantize_k\": True, \"quantize_v\": True}"
        emit "jointqk" "$json" "jointqk_NEW_vtq_k${kb}_v${V_BITS}" "$task"
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
