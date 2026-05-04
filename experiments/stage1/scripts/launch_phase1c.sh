#!/bin/bash
# Phase 1C: combined K+V sanity check at K ∈ {2,3,4} × locked V (read from v_lock.txt).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

GPUS="0,1,2"
FRACTION="0.3"
TASK="qasper"
MODEL="Qwen/Qwen3-8B"
CCA="${REPO_ROOT}/artifacts/stage1/cca_vs_waterfill_study/cca_stats.pt"
VST="${REPO_ROOT}/artifacts/stage1/v_method_study/v_stats.pt"
V_LOCK_FILE="${REPO_ROOT}/artifacts/stage1/v_method_study/v_lock.txt"
OUT_BASE="${REPO_ROOT}/artifacts/stage1/v_method_study/sweep_combined"
LOG_DIR="${REPO_ROOT}/experiments/stage1/logs/phase1c"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS="$2"; shift 2 ;;
        --fraction) FRACTION="$2"; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -f "$V_LOCK_FILE" ]]; then
    echo "ERROR: $V_LOCK_FILE not found. Run aggregate_phase1.py first." >&2
    exit 1
fi

V_METHOD=$(grep -oP 'V_METHOD=\K\S+' "$V_LOCK_FILE")
V_BITS=$(grep -oP 'V_BITS=\K\d+' "$V_LOCK_FILE")
echo "[$(date '+%H:%M:%S')] Phase 1C: locked V_METHOD=$V_METHOD, V_BITS=$V_BITS"

mkdir -p "$OUT_BASE" "$LOG_DIR"
CMDS="$LOG_DIR/commands.txt"
: > "$CMDS"

CONDA_ACTIVATE='source ~/miniconda3/etc/profile.d/conda.sh && conda activate efficient-llm'
PY_PATH="${REPO_ROOT}:${REPO_ROOT}/kvpress"

for kb in 2 3 4; do
    json="{\"cca_stats_path\": \"${CCA}\", \"v_stats_path\": \"${VST}\", \"v_method\": \"${V_METHOD}\", \"v_bits\": ${V_BITS}, \"k_method\": \"r_sym_waterfill\", \"k_bits\": ${kb}, \"quantize_k\": True, \"quantize_v\": True, \"compress_decode\": False}"
    cmd="${CONDA_ACTIVATE} && cd ${REPO_ROOT}/kvpress/evaluation && PYTHONPATH=${PY_PATH} python -u evaluate.py \
        --press_name=jointqk \
        --press_kwargs='${json}' \
        --model='${MODEL}' \
        --dataset=longbench \
        --data_dir=${TASK} \
        --fraction=${FRACTION} \
        --output_dir=${OUT_BASE}/jointqk_k${kb}_v${V_BITS} # label=combined_k${kb}_v${V_BITS}"
    echo "$cmd" >> "$CMDS"
done

n_jobs=$(wc -l < "$CMDS")
echo "[$(date '+%H:%M:%S')] Phase 1C: $n_jobs jobs queued in $CMDS"

python "${REPO_ROOT}/experiments/stage1/scripts/parallel_launcher.py" \
    --commands-file "$CMDS" \
    --log-dir "$LOG_DIR" \
    --gpus "$GPUS" \
    --label-from-comment
