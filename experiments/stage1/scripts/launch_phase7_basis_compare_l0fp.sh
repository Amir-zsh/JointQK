#!/bin/bash
# Fair-comparison rerun: layer0_full_precision=True for ALL methods.
# Mirrors the v6 convention so absolute numbers are comparable.
# Configs: TurboQuant K∈{2,4} V=3, JointQK NEW + v_turboquant K∈{2,4} V=3,
#          JointQK NEW + v_eigen_uniform K∈{2,4} V=3 (to anchor against v6 numbers).
# 4 tasks × 6 configs = 24 cells.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

GPUS="${GPUS:-0,1,2,3,4,5}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
FRACTION="${EVAL_FRACTION:-0.5}"

MODEL="Qwen/Qwen3-8B"
CCA_NEW="${REPO_ROOT}/artifacts/stage1/cca_vs_waterfill_study/cca_stats_longbench_compact8_n400.pt"
VST_NEW="${REPO_ROOT}/artifacts/stage1/v_method_study/v_stats_longbench_compact8_n400.pt"
COMPRESS_DECODE=False
V_BITS=3
L0FP=True

OUT_BASE="${REPO_ROOT}/artifacts/stage1/downstream_basis_compare_l0fp"
LOG_DIR="${REPO_ROOT}/experiments/stage1/logs/phase7_basis_compare_l0fp"

echo "[$(date '+%H:%M:%S')] Phase 7 basis compare with layer0_full_precision=True"

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

emit_oracle() {
    local task="$1"
    local cmd="cd ${REPO_ROOT}/kvpress/evaluation && ${ALLOC_CONF} PYTHONPATH=${PY_PATH} ${VENV_PYTHON} -u evaluate.py \
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
        # TurboQuant with l0fp=True
        json_tq="{\"k_bits\": ${kb}, \"v_bits\": ${V_BITS}, \"compress_decode\": ${COMPRESS_DECODE}, \"layer0_full_precision\": ${L0FP}}"
        emit "turboquant" "$json_tq" "turboquant_l0fp_k${kb}_v${V_BITS}" "$task"

        # JointQK NEW basis + v_turboquant V + l0fp=True
        json_jq_vtq="{\"cca_stats_path\": \"${CCA_NEW}\", \"v_stats_path\": \"${VST_NEW}\", \"v_method\": \"v_turboquant\", \"k_bits\": ${kb}, \"v_bits\": ${V_BITS}, \"compress_decode\": ${COMPRESS_DECODE}, \"layer0_full_precision\": ${L0FP}, \"quantize_k\": True, \"quantize_v\": True}"
        emit "jointqk" "$json_jq_vtq" "jointqk_NEW_vtq_l0fp_k${kb}_v${V_BITS}" "$task"

        # JointQK NEW basis + v_eigen_uniform V + l0fp=True (anchor against v6 numbers)
        json_jq_veu="{\"cca_stats_path\": \"${CCA_NEW}\", \"v_stats_path\": \"${VST_NEW}\", \"v_method\": \"v_eigen_uniform\", \"k_bits\": ${kb}, \"v_bits\": ${V_BITS}, \"compress_decode\": ${COMPRESS_DECODE}, \"layer0_full_precision\": ${L0FP}, \"quantize_k\": True, \"quantize_v\": True}"
        emit "jointqk" "$json_jq_veu" "jointqk_NEW_veu_l0fp_k${kb}_v${V_BITS}" "$task"
    done
done

n_jobs=$(wc -l < "$CMDS")
echo "[$(date '+%H:%M:%S')] queued $n_jobs jobs"

${VENV_PYTHON} experiments/stage1/scripts/parallel_launcher.py \
    --commands-file "$CMDS" \
    --log-dir "$LOG_DIR" \
    --gpus "$GPUS" \
    --jobs-per-gpu "$JOBS_PER_GPU" \
    --label-from-comment
