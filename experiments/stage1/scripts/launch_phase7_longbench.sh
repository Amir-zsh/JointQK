#!/bin/bash
# Phase 7.1/7.2: full LongBench sweep on Qwen or Llama.
# 8 configs (1 oracle + JointQK@K∈{2,3,4} + TurboQuant@K∈{2,3,4} + KIVI int4) × 12 tasks = 96 jobs/model.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

GPUS="0,1,2,3,4,5"
FRACTION="${EVAL_FRACTION:-0.5}"
MODEL_TAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS="$2"; shift 2 ;;
        --fraction) FRACTION="$2"; shift 2 ;;
        --model) MODEL_TAG="$2"; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$MODEL_TAG" ]]; then
    echo "ERROR: --model {qwen3_8b,llama31_8b} required" >&2
    exit 1
fi

case "$MODEL_TAG" in
    qwen3_8b)
        MODEL="Qwen/Qwen3-8B"
        CCA="${REPO_ROOT}/artifacts/stage1/cca_vs_waterfill_study/cca_stats.pt"
        VST="${REPO_ROOT}/artifacts/stage1/v_method_study/v_stats.pt"
        ;;
    llama31_8b)
        MODEL="meta-llama/Llama-3.1-8B-Instruct"
        CCA="${REPO_ROOT}/artifacts/stage1/cca_vs_waterfill_study/llama31_8b/cca_stats.pt"
        VST="${REPO_ROOT}/artifacts/stage1/v_method_study/v_stats_llama31_8b.pt"
        ;;
    *) echo "Unknown model tag: $MODEL_TAG"; exit 1 ;;
esac

V_LOCK_FILE="${REPO_ROOT}/artifacts/stage1/v_method_study/v_lock.txt"
DECODE_FILE="${REPO_ROOT}/artifacts/stage1/downstream/qwen3_8b/decode_scope/decode_decision.txt"
OUT_BASE="${REPO_ROOT}/artifacts/stage1/downstream/${MODEL_TAG}"
LOG_DIR="${REPO_ROOT}/experiments/stage1/logs/phase7_longbench_${MODEL_TAG}"

V_METHOD=$(grep -oP 'V_METHOD=\K\S+' "$V_LOCK_FILE")
V_BITS=$(grep -oP 'V_BITS=\K\d+' "$V_LOCK_FILE")
WINNER=$(grep -oP 'WINNER=\K[AB]' "$DECODE_FILE" 2>/dev/null || echo "A")
COMPRESS_DECODE=$([ "$WINNER" = "B" ] && echo True || echo False)
echo "[$(date '+%H:%M:%S')] Phase 7 LongBench $MODEL_TAG: V_METHOD=$V_METHOD V_BITS=$V_BITS COMPRESS_DECODE=$COMPRESS_DECODE FRACTION=$FRACTION"

mkdir -p "$OUT_BASE" "$LOG_DIR"
CMDS="$LOG_DIR/commands.txt"
: > "$CMDS"

CONDA_ACTIVATE='source ~/miniconda3/etc/profile.d/conda.sh && conda activate efficient-llm'
PY_PATH="${REPO_ROOT}:${REPO_ROOT}/kvpress"
ALLOC_CONF='PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'

# All LongBench-E task names
TASKS=(narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p)

emit() {
    local press="$1" json="$2" label_prefix="$3" task="$4"
    local cmd="${CONDA_ACTIVATE} && cd ${REPO_ROOT}/kvpress/evaluation && ${ALLOC_CONF} PYTHONPATH=${PY_PATH} python -u evaluate.py \
        --press_name=${press} \
        --press_kwargs='${json}' \
        --model='${MODEL}' \
        --dataset=longbench \
        --data_dir=${task} \
        --fraction=${FRACTION} \
        --output_dir=${OUT_BASE}/${label_prefix}_${task} # label=${label_prefix}_${task}"
    echo "$cmd" >> "$CMDS"
}

emit_oracle() {
    local task="$1"
    local cmd="${CONDA_ACTIVATE} && cd ${REPO_ROOT}/kvpress/evaluation && ${ALLOC_CONF} PYTHONPATH=${PY_PATH} python -u evaluate.py \
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
    for kb in 2 3 4; do
        json_jq="{\"cca_stats_path\": \"${CCA}\", \"v_stats_path\": \"${VST}\", \"v_method\": \"${V_METHOD}\", \"k_bits\": ${kb}, \"v_bits\": ${V_BITS}, \"compress_decode\": ${COMPRESS_DECODE}, \"quantize_k\": True, \"quantize_v\": True}"
        emit "jointqk" "$json_jq" "jointqk_k${kb}_v${V_BITS}" "$task"
        json_tq="{\"k_bits\": ${kb}, \"v_bits\": ${V_BITS}, \"compress_decode\": ${COMPRESS_DECODE}}"
        emit "turboquant" "$json_tq" "turboquant_k${kb}_v${V_BITS}" "$task"
    done
    json_kv="{\"k_bits\": 4, \"v_bits\": 4, \"group_size\": 128, \"compress_decode\": ${COMPRESS_DECODE}}"
    emit "kivi" "$json_kv" "kivi_int4" "$task"
done

n_jobs=$(wc -l < "$CMDS")
echo "[$(date '+%H:%M:%S')] Phase 7 LongBench $MODEL_TAG: $n_jobs jobs queued"

python "${REPO_ROOT}/experiments/stage1/scripts/parallel_launcher.py" \
    --commands-file "$CMDS" \
    --log-dir "$LOG_DIR" \
    --gpus "$GPUS" \
    --label-from-comment
