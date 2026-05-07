#!/bin/bash
# Phase 7 v7 — re-execution of v6 with all findings from the JointQK-disconnect
# investigation applied to maximize JointQK performance:
#
#   1. Calibration corpus: pooled-400 LongBench-compact8 (4.5M tokens) instead of
#      the deployed cca_stats.pt (24 examples, 81k tokens). 50× more tokens.
#   2. V method: v_turboquant (uncentered random Hadamard + uniform Lloyd-Max),
#      replacing the v_eigen_uniform that was selected on a wrong tuning decision.
#      v_lock.txt has been updated to reflect this change.
#   3. layer0_full_precision=True is now the press default — explicitly passed
#      here for clarity / forward compatibility.
#   4. Tasks expanded from KIVI 8 to 12 (adds 4 multi-doc QA: hotpotqa, musique,
#      2wikimqa, narrativeqa) — these are where Q-K-aware bit allocation is
#      expected to win, missing from v6.
#
# Configs (16 per task × 12 tasks = 192 cells):
#   1 full_precision oracle
#   6 JointQK / NEW pool / v_turboquant: K∈{2,3,4} × V∈{2,3}
#   6 TurboQuant: K∈{2,3,4} × V∈{2,3}
#   3 KIVI: int2, int3, int4
#
# All compressed methods: layer0_full_precision=True, compress_decode=False (Mode A).
#
# See `notes/stage1/jointqk_disconnect_investigation.md` for the analysis driving
# these v7 changes.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

GPUS="${GPUS:-0,1,2,3,4,5}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"            # 2 jobs/GPU; OOM-retried up to MAX_RETRIES.
MAX_RETRIES="${MAX_RETRIES:-10}"             # OOM requeue count (phase7_worker).
FRACTION="${EVAL_FRACTION:-1.0}"

# Model selection. Default Qwen3-8B; pass --model {qwen3_8b,llama31_8b} to switch.
MODEL_TAG="qwen3_8b"
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_TAG="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --fraction) FRACTION="$2"; shift 2 ;;
        --jobs-per-gpu) JOBS_PER_GPU="$2"; shift 2 ;;
        --max-retries) MAX_RETRIES="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

case "$MODEL_TAG" in
    qwen3_8b)
        MODEL="Qwen/Qwen3-8B"
        CCA_SUFFIX="longbench_compact8_n400"
        ;;
    llama31_8b)
        MODEL="meta-llama/Llama-3.1-8B-Instruct"
        CCA_SUFFIX="llama31_8b_longbench_compact8_n400"
        ;;
    *) echo "Unknown --model tag: $MODEL_TAG (expected qwen3_8b or llama31_8b)" >&2; exit 1 ;;
esac

# v7 calibration: pooled-N=400 LongBench-compact8 train prompts.
CCA="${REPO_ROOT}/artifacts/stage1/cca_vs_waterfill_study/cca_stats_${CCA_SUFFIX}.pt"
VST="${REPO_ROOT}/artifacts/stage1/v_method_study/v_stats_${CCA_SUFFIX}.pt"
if [[ ! -f "$CCA" ]]; then
    echo "ERROR: missing calibration $CCA" >&2
    echo "Build it with: .venv/bin/python experiments/stage1/scripts/build_calibration_artifacts_from_pool.py --run-id longbench_compact8_qkv_${MODEL_TAG} --output-suffix ${CCA_SUFFIX}" >&2
    exit 1
fi

# v_lock now points to v_turboquant; we still read it so the lock stays the
# single source of truth and any future re-tune lands automatically. Note that
# v_turboquant is calibration-free, so $VST is not actually loaded by the press
# in this configuration — it's still required by the press API surface though.
V_LOCK_FILE="${REPO_ROOT}/artifacts/stage1/v_method_study/v_lock.txt"
V_METHOD=$(grep -oP '^V_METHOD=\K\S+' "$V_LOCK_FILE")
V_BITS_LOCK=$(grep -oP '^V_BITS=\K\d+' "$V_LOCK_FILE")
echo "[$(date '+%H:%M:%S')] $MODEL_TAG | v_lock V_METHOD=$V_METHOD V_BITS=$V_BITS_LOCK"

COMPRESS_DECODE=False
LAYER0_FP=True

# Exclude calibration train rows from the F1 evaluation (option A from the
# disconnect investigation, train-split-only scope). For the 7 tasks shared
# with the calibration corpus, this drops 50/200 ≈ 25% of LongBench rows so
# JointQK is evaluated on prompts its basis was NOT fit on. Tasks not in the
# calibration (2wikimqa, lcc, narrativeqa, samsum, trec) are unaffected.
EXCLUDE_INDICES_FILE="${REPO_ROOT}/artifacts/stage1/calibration_splits/longbench_compact8_60_seed20260504_2k32k/exclude_train_indices_for_eval.json"
if [[ ! -f "$EXCLUDE_INDICES_FILE" ]]; then
    echo "ERROR: missing $EXCLUDE_INDICES_FILE" >&2
    echo "Generate it with: .venv/bin/python -c \"import json; from pathlib import Path; m=json.loads(Path('artifacts/stage1/calibration_splits/longbench_compact8_60_seed20260504_2k32k/manifest.json').read_text()); excl={}; [excl.setdefault(r['config'],[]).append(int(r['row_index'])) for r in m['rows'] if r['split']=='train']; Path('$EXCLUDE_INDICES_FILE').write_text(json.dumps(excl, indent=2))\"" >&2
    exit 1
fi

OUT_BASE="${REPO_ROOT}/artifacts/stage1/downstream_v7/${MODEL_TAG}"
LOG_DIR="${REPO_ROOT}/experiments/stage1/logs/phase7_v7_${MODEL_TAG}"

# 12-task eval mix. KIVI 8 = direct v6 comparability. Multi-doc QA = the
# regime where v5 found JointQK's largest advantages (and v6 missed because
# its task subset had no multi-doc QA).
TASKS_KIVI=(qasper qmsum multi_news trec triviaqa samsum lcc repobench-p)
TASKS_MULTIDOC=(hotpotqa musique 2wikimqa narrativeqa)
TASKS=("${TASKS_KIVI[@]}" "${TASKS_MULTIDOC[@]}")

# Config grid (16 configs/task). K and V budgets mirror v6 main sweep.
K_BITS=(2 3 4)
V_BITS=(2 3)

mkdir -p "$OUT_BASE" "$LOG_DIR"
# JSONL for phase7_worker.py: one EvaluationConfig dict per line.
CMDS="$LOG_DIR/commands.jsonl"
: > "$CMDS"

VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"

# Common fields used by every cell.
EXCLUDE_REL="$EXCLUDE_INDICES_FILE"

emit_oracle() {
    local task="$1"
    .venv/bin/python <<PY >> "$CMDS"
import json
print(json.dumps({
    "_label": f"oracle_{$(printf %s "$task" | python3 -c 'import sys,json;print(json.dumps(sys.stdin.read()))')}",
    "press_name": "no_press",
    "compression_ratio": 0.0,
    "dataset": "longbench",
    "data_dir": "$task",
    "fraction": $FRACTION,
    "exclude_indices_file": "$EXCLUDE_REL",
    "output_dir": "$OUT_BASE/full_precision_$task",
}))
PY
}

emit_jq() {  # task k_bits v_bits
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
        "k_bits": $kb,
        "v_bits": $vb,
        "compress_decode": False,
        "layer0_full_precision": True,
        "quantize_k": True,
        "quantize_v": True,
    },
    "dataset": "longbench",
    "data_dir": "$task",
    "fraction": $FRACTION,
    "exclude_indices_file": "$EXCLUDE_REL",
    "output_dir": "$OUT_BASE/jointqk_k${kb}_v${vb}_${task}",
}))
PY
}

emit_tq() {  # task k_bits v_bits
    local task="$1" kb="$2" vb="$3"
    .venv/bin/python - <<PY >> "$CMDS"
import json
print(json.dumps({
    "_label": "turboquant_k${kb}_v${vb}_${task}",
    "press_name": "turboquant",
    "press_kwargs": {
        "k_bits": $kb,
        "v_bits": $vb,
        "compress_decode": False,
        "layer0_full_precision": True,
    },
    "dataset": "longbench",
    "data_dir": "$task",
    "fraction": $FRACTION,
    "exclude_indices_file": "$EXCLUDE_REL",
    "output_dir": "$OUT_BASE/turboquant_k${kb}_v${vb}_${task}",
}))
PY
}

emit_kivi() {  # task bits
    local task="$1" b="$2"
    .venv/bin/python - <<PY >> "$CMDS"
import json
print(json.dumps({
    "_label": "kivi_int${b}_${task}",
    "press_name": "kivi",
    "press_kwargs": {
        "k_bits": $b,
        "v_bits": $b,
        "group_size": 128,
        "compress_decode": False,
        "layer0_full_precision": True,
    },
    "dataset": "longbench",
    "data_dir": "$task",
    "fraction": $FRACTION,
    "exclude_indices_file": "$EXCLUDE_REL",
    "output_dir": "$OUT_BASE/kivi_int${b}_${task}",
}))
PY
}

for task in "${TASKS[@]}"; do
    emit_oracle "$task"
    for kb in "${K_BITS[@]}"; do
        for vb in "${V_BITS[@]}"; do
            emit_jq "$task" "$kb" "$vb"
            emit_tq "$task" "$kb" "$vb"
        done
    done
    for kb in 2 3 4; do
        emit_kivi "$task" "$kb"
    done
done

n_jobs=$(wc -l < "$CMDS")
echo "[$(date '+%H:%M:%S')] queued $n_jobs jobs in $CMDS"
echo "  tasks (${#TASKS[@]}): ${TASKS[*]}"
echo "  configs/task: 1 oracle + ${#K_BITS[@]}×${#V_BITS[@]}=$((${#K_BITS[@]}*${#V_BITS[@]})) JointQK + same TurboQuant + 3 KIVI = $((1 + 2*${#K_BITS[@]}*${#V_BITS[@]} + 3))"
echo "  fraction: $FRACTION  (Mode A, layer0_full_precision=True)"
echo "  CCA: $CCA"
echo "  VST: $VST"
echo "  V_METHOD: $V_METHOD"
echo "  EXCLUDE_INDICES_FILE: $EXCLUDE_INDICES_FILE"
echo "  jobs-per-gpu: $JOBS_PER_GPU  max-retries (OOM): $MAX_RETRIES"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo
    echo "=== first 4 jsonl commands ==="
    head -4 "$CMDS"
    echo "..."
    echo "[$(date '+%H:%M:%S')] dry run — not executing"
    exit 0
fi

${VENV_PYTHON} experiments/stage1/scripts/phase7_worker.py \
    --model "$MODEL" \
    --commands-file "$CMDS" \
    --log-dir "$LOG_DIR" \
    --gpus "$GPUS" \
    --jobs-per-gpu "$JOBS_PER_GPU" \
    --max-retries "$MAX_RETRIES"
