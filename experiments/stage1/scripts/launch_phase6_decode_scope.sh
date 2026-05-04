#!/bin/bash
# Phase 6: decode-scope ablation (Mode A vs Mode B) on JointQK at K∈{2,3,4} × V_locked.
# 2 tasks × 2 modes × 3 K bits = 12 jobs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

GPUS="0,1,2,3,4,5"
FRACTION="0.3"
MODEL="Qwen/Qwen3-8B"
CCA="${REPO_ROOT}/artifacts/stage1/cca_vs_waterfill_study/cca_stats.pt"
VST="${REPO_ROOT}/artifacts/stage1/v_method_study/v_stats.pt"
V_LOCK_FILE="${REPO_ROOT}/artifacts/stage1/v_method_study/v_lock.txt"
OUT_BASE="${REPO_ROOT}/artifacts/stage1/downstream/qwen3_8b/decode_scope"
LOG_DIR="${REPO_ROOT}/experiments/stage1/logs/phase6"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS="$2"; shift 2 ;;
        --fraction) FRACTION="$2"; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

V_METHOD=$(grep -oP 'V_METHOD=\K\S+' "$V_LOCK_FILE")
V_BITS=$(grep -oP 'V_BITS=\K\d+' "$V_LOCK_FILE")
echo "[$(date '+%H:%M:%S')] Phase 6: V_METHOD=$V_METHOD, V_BITS=$V_BITS"

mkdir -p "$OUT_BASE" "$LOG_DIR"
CMDS="$LOG_DIR/commands.txt"
: > "$CMDS"

CONDA_ACTIVATE='source ~/miniconda3/etc/profile.d/conda.sh && conda activate efficient-llm'
PY_PATH="${REPO_ROOT}:${REPO_ROOT}/kvpress"

for task in qasper narrativeqa; do
  for mode in False True; do
    for kb in 2 3 4; do
      tag_mode=$([ "$mode" = "True" ] && echo "modeB" || echo "modeA")
      out_subdir="${task}_${tag_mode}_k${kb}"
      json="{\"cca_stats_path\": \"${CCA}\", \"v_stats_path\": \"${VST}\", \"v_method\": \"${V_METHOD}\", \"k_bits\": ${kb}, \"v_bits\": ${V_BITS}, \"compress_decode\": ${mode}, \"quantize_k\": True, \"quantize_v\": True}"
      cmd="${CONDA_ACTIVATE} && cd ${REPO_ROOT}/kvpress/evaluation && PYTHONPATH=${PY_PATH} python -u evaluate.py \
        --press_name=jointqk \
        --press_kwargs='${json}' \
        --model='${MODEL}' \
        --dataset=longbench \
        --data_dir=${task} \
        --fraction=${FRACTION} \
        --output_dir=${OUT_BASE}/${out_subdir} # label=${out_subdir}"
      echo "$cmd" >> "$CMDS"
    done
  done
done

n_jobs=$(wc -l < "$CMDS")
echo "[$(date '+%H:%M:%S')] Phase 6: $n_jobs jobs queued"

python "${REPO_ROOT}/experiments/stage1/scripts/parallel_launcher.py" \
    --commands-file "$CMDS" \
    --log-dir "$LOG_DIR" \
    --gpus "$GPUS" \
    --label-from-comment
