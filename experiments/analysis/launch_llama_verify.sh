#!/bin/bash
# Focused Llama-3.1-8B re-verification of the phase7 v7 F1 numbers reported in
# notes/bench_llama31_8b_results_report.md (which were produced on a
# remote machine). Re-runs a focused 30-cell subset locally at fraction=1.0 with
# identical kwargs, so we can compare F1 numbers and confirm the JointQK vs
# TurboQuant disconnect on Llama is reproducible.
#
# Subset:
#   - 6 tasks: hotpotqa, lcc, repobench-p, qasper, qmsum, 2wikimqa
#     (covers disconnect cells, ties, and a JQ-wins task — 2wikimqa)
#   - 5 configs each: FP, jointqk_k2_v3, turboquant_k2_v3, jointqk_k4_v3, turboquant_k4_v3
#
# Output: artifacts/bench_llama_verify/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

GPUS="${GPUS:-0,1,2,3}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
MAX_RETRIES="${MAX_RETRIES:-5}"
FRACTION="${EVAL_FRACTION:-1.0}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS="$2"; shift 2 ;;
        --fraction) FRACTION="$2"; shift 2 ;;
        --jobs-per-gpu) JOBS_PER_GPU="$2"; shift 2 ;;
        --max-retries) MAX_RETRIES="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

MODEL="meta-llama/Llama-3.1-8B-Instruct"
CCA="${REPO_ROOT}/artifacts/bases/cca_stats_llama31_8b_longbench_compact8_n400.pt"
VST="${REPO_ROOT}/artifacts/v_bases/v_stats_llama31_8b_longbench_compact8_n400.pt"
[[ ! -f "$CCA" ]] && { echo "ERROR: missing $CCA" >&2; exit 1; }
[[ ! -f "$VST" ]] && { echo "ERROR: missing $VST" >&2; exit 1; }

V_LOCK_FILE="${REPO_ROOT}/artifacts/v_bases/v_lock.txt"
V_METHOD=$(grep -oP '^V_METHOD=\K\S+' "$V_LOCK_FILE")
echo "[$(date '+%H:%M:%S')] llama_verify | v_lock V_METHOD=$V_METHOD"

EXCLUDE_INDICES_FILE="${REPO_ROOT}/artifacts/calibration_splits/longbench_compact8_60_seed20260504_2k32k/exclude_train_indices_for_eval.json"
[[ ! -f "$EXCLUDE_INDICES_FILE" ]] && { echo "ERROR: missing $EXCLUDE_INDICES_FILE" >&2; exit 1; }

OUT_BASE="${REPO_ROOT}/artifacts/bench_llama_verify"
LOG_DIR="${REPO_ROOT}/experiments/logs/phase7_v7_llama_verify"
mkdir -p "$OUT_BASE" "$LOG_DIR"
CMDS="$LOG_DIR/commands.jsonl"
: > "$CMDS"

# Focused task subset
TASKS=(hotpotqa lcc repobench-p qasper qmsum 2wikimqa)

emit_oracle() {
    local task="$1"
    .venv/bin/python - <<PY >> "$CMDS"
import json
print(json.dumps({
    "_label": "oracle_${task}",
    "press_name": "no_press",
    "compression_ratio": 0.0,
    "dataset": "longbench",
    "data_dir": "${task}",
    "fraction": $FRACTION,
    "exclude_indices_file": "$EXCLUDE_INDICES_FILE",
    "output_dir": "$OUT_BASE/full_precision_${task}",
}))
PY
}

emit_jq() {
    local task="$1" kb="$2" vb="$3"
    .venv/bin/python - <<PY >> "$CMDS"
import json
print(json.dumps({
    "_label": "jointqk_k${kb}_v${vb}_${task}",
    "press_name": "jointqk",
    "press_kwargs": {
        "cca_stats_path": "$CCA",
        "v_stats_path": "$VST",
        "v_method": "$V_METHOD",
        "k_bits": ${kb},
        "v_bits": ${vb},
        "compress_decode": False,
        "layer0_full_precision": True,
        "quantize_k": True,
        "quantize_v": True,
    },
    "dataset": "longbench",
    "data_dir": "${task}",
    "fraction": $FRACTION,
    "exclude_indices_file": "$EXCLUDE_INDICES_FILE",
    "output_dir": "$OUT_BASE/jointqk_k${kb}_v${vb}_${task}",
}))
PY
}

emit_tq() {
    local task="$1" kb="$2" vb="$3"
    .venv/bin/python - <<PY >> "$CMDS"
import json
print(json.dumps({
    "_label": "turboquant_k${kb}_v${vb}_${task}",
    "press_name": "turboquant",
    "press_kwargs": {
        "k_bits": ${kb},
        "v_bits": ${vb},
        "compress_decode": False,
        "layer0_full_precision": True,
    },
    "dataset": "longbench",
    "data_dir": "${task}",
    "fraction": $FRACTION,
    "exclude_indices_file": "$EXCLUDE_INDICES_FILE",
    "output_dir": "$OUT_BASE/turboquant_k${kb}_v${vb}_${task}",
}))
PY
}

for task in "${TASKS[@]}"; do
    emit_oracle "$task"
    for kb in 2 4; do
        emit_jq "$task" "$kb" 3
        emit_tq "$task" "$kb" 3
    done
done

n_jobs=$(wc -l < "$CMDS")
echo "[$(date '+%H:%M:%S')] queued $n_jobs jobs in $CMDS"
echo "  tasks (${#TASKS[@]}): ${TASKS[*]}"
echo "  configs/task: FP + 2 K(2,4) × 2 methods (JQ, TQ) at V=3 = 5"
echo "  fraction: $FRACTION"
echo "  output: $OUT_BASE"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "=== first 4 cmds ==="
    head -4 "$CMDS"
    echo "[$(date '+%H:%M:%S')] dry run — not executing"
    exit 0
fi

.venv/bin/python experiments/bench/worker.py \
    --model "$MODEL" \
    --commands-file "$CMDS" \
    --log-dir "$LOG_DIR" \
    --gpus "$GPUS" \
    --jobs-per-gpu "$JOBS_PER_GPU" \
    --max-retries "$MAX_RETRIES"
