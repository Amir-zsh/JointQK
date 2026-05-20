#!/bin/bash
# Re-run JointQK K=2 V=3 on Llama with two new calibrations:
#   - compact9: pooled basis fit on compact8 ∪ {lcc} (~450 train prompts)
#   - lcconly:  single-task basis fit on lcc-only train prompts (~50 train prompts)
#
# Tasks: 4 chosen to span the F1-disconnect spectrum (see plan):
#   lcc           — biggest disconnect, OOD vs compact8
#   repobench-p   — code, in compact8 calibration
#   hotpotqa      — QA, in compact8 calibration
#   2wikimqa      — OOD, JQ wins control
#
# 4 tasks × 2 bases = 8 cells. FP/TQ baselines re-use prior verify-run cells.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
CCA_COMPACT9="${REPO_ROOT}/artifacts/stage1/cca_vs_waterfill_study/cca_stats_llama31_8b_compact9_n450.pt"
CCA_LCCONLY="${REPO_ROOT}/artifacts/stage1/cca_vs_waterfill_study/cca_stats_llama31_8b_lcc_only_n50.pt"
VST="${REPO_ROOT}/artifacts/stage1/v_method_study/v_stats_llama31_8b_longbench_compact8_n400.pt"

V_LOCK_FILE="${REPO_ROOT}/artifacts/stage1/v_method_study/v_lock.txt"
V_METHOD=$(grep -oP '^V_METHOD=\K\S+' "$V_LOCK_FILE")

# IMPORTANT: with compact9 calibration, we excluded a NEW set of train rows from eval
# (the 450 train rows of compact9). For fair comparison, use the compact9 exclude file.
EXCLUDE_FILE="${REPO_ROOT}/artifacts/stage1/calibration_splits/longbench_compact9_60_seed20260504_2k32k/exclude_train_indices_for_eval.json"

OUT_COMPACT9="${REPO_ROOT}/artifacts/stage1/downstream_v7_llama_compact9"
OUT_LCCONLY="${REPO_ROOT}/artifacts/stage1/downstream_v7_llama_lcconly"
LOG_DIR="${REPO_ROOT}/experiments/stage1/logs/phase7_v7_llama_compact9_lcconly"
mkdir -p "$OUT_COMPACT9" "$OUT_LCCONLY" "$LOG_DIR"
CMDS="$LOG_DIR/commands.jsonl"
: > "$CMDS"

TASKS=(hotpotqa lcc repobench-p 2wikimqa)

emit_jq() {
    local task="$1" cca="$2" tag="$3" outdir="$4"
    .venv/bin/python - <<PY >> "$CMDS"
import json
print(json.dumps({
    "_label": "jointqk_k2_v3_${tag}_${task}",
    "press_name": "jointqk",
    "press_kwargs": {
        "cca_stats_path": "$cca",
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
    "fraction": 1.0,
    "exclude_indices_file": "$EXCLUDE_FILE",
    "output_dir": "$outdir/jointqk_k2_v3_${task}",
}))
PY
}

for t in "${TASKS[@]}"; do
    emit_jq "$t" "$CCA_COMPACT9" "compact9" "$OUT_COMPACT9"
    emit_jq "$t" "$CCA_LCCONLY"  "lcconly"  "$OUT_LCCONLY"
done

n_jobs=$(wc -l < "$CMDS")
echo "[$(date '+%H:%M:%S')] queued $n_jobs JointQK K=2 V=3 cells (4 tasks × 2 bases)"

.venv/bin/python experiments/stage1/scripts/phase7_worker.py \
    --model "$MODEL" \
    --commands-file "$CMDS" \
    --log-dir "$LOG_DIR" \
    --gpus 0,1,2,3 \
    --jobs-per-gpu 1 \
    --max-retries 5
