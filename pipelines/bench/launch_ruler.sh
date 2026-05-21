#!/bin/bash
# Phase 7.3: RULER NIAH on both models at ctx ∈ {4096, 8192, 16384} × 4 conditions.
# 2 models × 3 ctx × (1 oracle + JointQK k=4 + TurboQuant k=4 + KIVI int4) = 24 jobs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

GPUS="0,1,2,3,4,5"
KS="4"   # comma-separated K bits to test
CTXS="4096,8192,16384"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS="$2"; shift 2 ;;
        --ks) KS="$2"; shift 2 ;;
        --ctxs) CTXS="$2"; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

V_LOCK_FILE="${REPO_ROOT}/artifacts/v_bases/v_lock.txt"
V_METHOD=$(grep -oP 'V_METHOD=\K\S+' "$V_LOCK_FILE")
V_BITS=$(grep -oP 'V_BITS=\K\d+' "$V_LOCK_FILE")
# Prefill-only compression for all methods. Decode-step KV stays fp16 so the
# RULER comparison does not depend on method-specific generation-time behavior.
COMPRESS_DECODE=False

LOG_DIR="${REPO_ROOT}/logs/phase7_ruler"
mkdir -p "$LOG_DIR"
CMDS="$LOG_DIR/commands.txt"
: > "$CMDS"

CONDA_ACTIVATE='source ~/miniconda3/etc/profile.d/conda.sh && conda activate efficient-llm'
PY_PATH="${REPO_ROOT}:${REPO_ROOT}/vendor/kvpress"

IFS=',' read -ra CTX_ARR <<< "$CTXS"
IFS=',' read -ra K_ARR <<< "$KS"

emit() {
    local model="$1" press="$2" json="$3" out_dir="$4" label="$5" ctx="$6"
    local cmd="${CONDA_ACTIVATE} && cd ${REPO_ROOT}/vendor/kvpress/evaluation && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=${PY_PATH} python -u evaluate.py \
        --press_name=${press} \
        --press_kwargs='${json}' \
        --model='${model}' \
        --dataset=ruler \
        --data_dir=${ctx} \
        --output_dir=${out_dir} # label=${label}"
    echo "$cmd" >> "$CMDS"
}

emit_oracle() {
    local model="$1" out_dir="$2" label="$3" ctx="$4"
    local cmd="${CONDA_ACTIVATE} && cd ${REPO_ROOT}/vendor/kvpress/evaluation && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=${PY_PATH} python -u evaluate.py \
        --press_name=no_press \
        --compression_ratio=0.0 \
        --model='${model}' \
        --dataset=ruler \
        --data_dir=${ctx} \
        --output_dir=${out_dir} # label=${label}"
    echo "$cmd" >> "$CMDS"
}

for MODEL_TAG in qwen3_8b llama31_8b; do
    case "$MODEL_TAG" in
        qwen3_8b)
            MODEL="Qwen/Qwen3-8B"
            CCA="${REPO_ROOT}/artifacts/bases/jointqk.pt"
            VST="${REPO_ROOT}/artifacts/v_bases/v_stats.pt"
            ;;
        llama31_8b)
            MODEL="meta-llama/Llama-3.1-8B-Instruct"
            CCA="${REPO_ROOT}/artifacts/bases/llama31_8b/jointqk.pt"
            VST="${REPO_ROOT}/artifacts/v_bases/v_stats_llama31_8b.pt"
            ;;
    esac
    OUT="${REPO_ROOT}/artifacts/downstream/${MODEL_TAG}"

    for ctx in "${CTX_ARR[@]}"; do
        emit_oracle "$MODEL" "${OUT}/ruler_full_${ctx}" "${MODEL_TAG}_full_${ctx}" "$ctx"
        for kb in "${K_ARR[@]}"; do
            json_jq="{\"cca_stats_path\": \"${CCA}\", \"v_stats_path\": \"${VST}\", \"v_method\": \"${V_METHOD}\", \"k_bits\": ${kb}, \"v_bits\": ${V_BITS}, \"compress_decode\": ${COMPRESS_DECODE}, \"quantize_k\": True, \"quantize_v\": True}"
            emit "$MODEL" "jointqk" "$json_jq" "${OUT}/ruler_jointqk_k${kb}_${ctx}" "${MODEL_TAG}_jointqk_k${kb}_${ctx}" "$ctx"

            json_tq="{\"k_bits\": ${kb}, \"v_bits\": ${V_BITS}, \"compress_decode\": ${COMPRESS_DECODE}}"
            emit "$MODEL" "turboquant" "$json_tq" "${OUT}/ruler_turboquant_k${kb}_${ctx}" "${MODEL_TAG}_turboquant_k${kb}_${ctx}" "$ctx"
        done
        json_kv="{\"k_bits\": 4, \"v_bits\": 4, \"group_size\": 128, \"compress_decode\": ${COMPRESS_DECODE}}"
        emit "$MODEL" "kivi" "$json_kv" "${OUT}/ruler_kivi_${ctx}" "${MODEL_TAG}_kivi_${ctx}" "$ctx"
    done
done

n_jobs=$(wc -l < "$CMDS")
echo "[$(date '+%H:%M:%S')] Phase 7 RULER: $n_jobs jobs queued"

python "${REPO_ROOT}/pipelines/bench/parallel_launcher.py" \
    --commands-file "$CMDS" \
    --log-dir "$LOG_DIR" \
    --gpus "$GPUS" \
    --label-from-comment
