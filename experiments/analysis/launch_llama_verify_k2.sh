#!/bin/bash
# K=2-only focused Llama verify: only the missing cells from the prior run.
# Reuses output_dir paths; worker.py skips if metrics.json exists.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

GPUS="${GPUS:-0,1,2,3}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
MAX_RETRIES="${MAX_RETRIES:-5}"
FRACTION="${EVAL_FRACTION:-1.0}"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
CCA="${REPO_ROOT}/artifacts/bases/cca_stats_llama31_8b_longbench_compact8_n400.pt"
VST="${REPO_ROOT}/artifacts/v_bases/v_stats_llama31_8b_longbench_compact8_n400.pt"

V_LOCK_FILE="${REPO_ROOT}/artifacts/v_bases/v_lock.txt"
V_METHOD=$(grep -oP '^V_METHOD=\K\S+' "$V_LOCK_FILE")

EXCLUDE_INDICES_FILE="${REPO_ROOT}/artifacts/calibration_splits/longbench_compact8_60_seed20260504_2k32k/exclude_train_indices_for_eval.json"

OUT_BASE="${REPO_ROOT}/artifacts/bench_llama_verify"
LOG_DIR="${REPO_ROOT}/experiments/logs/phase7_v7_llama_verify_k2"
mkdir -p "$LOG_DIR"
CMDS="$LOG_DIR/commands.jsonl"
: > "$CMDS"

# K=2-only: 6 tasks × {FP, JQ-K2, TQ-K2} = 18 cells (most already done; skip-if-exists handles).
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
    local task="$1"
    .venv/bin/python - <<PY >> "$CMDS"
import json
print(json.dumps({
    "_label": "jointqk_k2_v3_${task}",
    "press_name": "jointqk",
    "press_kwargs": {
        "cca_stats_path": "$CCA",
        "v_stats_path": "$VST",
        "v_method": "$V_METHOD",
        "k_bits": 2,
        "v_bits": 3,
        "compress_decode": False,
        "layer0_full_precision": True,
        "quantize_k": True,
        "quantize_v": True,
    },
    "dataset": "longbench",
    "data_dir": "${task}",
    "fraction": $FRACTION,
    "exclude_indices_file": "$EXCLUDE_INDICES_FILE",
    "output_dir": "$OUT_BASE/jointqk_k2_v3_${task}",
}))
PY
}

emit_tq() {
    local task="$1"
    .venv/bin/python - <<PY >> "$CMDS"
import json
print(json.dumps({
    "_label": "turboquant_k2_v3_${task}",
    "press_name": "turboquant",
    "press_kwargs": {
        "k_bits": 2,
        "v_bits": 3,
        "compress_decode": False,
        "layer0_full_precision": True,
    },
    "dataset": "longbench",
    "data_dir": "${task}",
    "fraction": $FRACTION,
    "exclude_indices_file": "$EXCLUDE_INDICES_FILE",
    "output_dir": "$OUT_BASE/turboquant_k2_v3_${task}",
}))
PY
}

for task in "${TASKS[@]}"; do
    emit_oracle "$task"
    emit_jq "$task"
    emit_tq "$task"
done

n_jobs=$(wc -l < "$CMDS")
echo "[$(date '+%H:%M:%S')] queued $n_jobs K=2 jobs (already-done cells will be skipped)"

.venv/bin/python experiments/bench/worker.py \
    --model "$MODEL" \
    --commands-file "$CMDS" \
    --log-dir "$LOG_DIR" \
    --gpus "$GPUS" \
    --jobs-per-gpu "$JOBS_PER_GPU" \
    --max-retries "$MAX_RETRIES"
