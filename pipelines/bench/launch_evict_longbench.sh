#!/bin/bash
# plan3 Thrust A bench: page-selection eviction on the pgq protocol tasks,
# Llama-3.1-8B, fraction=1.0, compact8 exclusions. PURE EVICTION cells — no
# K/V quantization anywhere (a different method class than the pgq sweeps;
# FP baselines come from downstream_v7, never re-run).
#
# Cells are "<press>:<ratio>" pairs, ratio = compression_ratio (fraction of
# tokens evicted):
#   omega_page:0.50          OmegaPagePress, score_mode frozen by the A1
#                            probe (read from page_selection_probe.json)
#   omega_page_random:0.50   the same press, score_mode=random_page control
#   expected_attention / snapkv / knorm / streaming_llm / random : stock
#                            kvpress registry presses
#
# REGISTRY PREREQ (one-time, local vendor mod — vendor/ is gitignored):
#   vendor/kvpress/evaluation/evaluate_registry.py must register
#     "omega_page": OmegaPagePress   (from kvq.presses.omega_page_press)
#   as a CLASS entry, exactly like the existing JointQKPress local mod.
#
#   bash pipelines/bench/launch_evict_longbench.sh --cells omega_page:0.50
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPUS="${GPUS:-0,1,2,3}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
MAX_RETRIES="${MAX_RETRIES:-10}"
FRACTION="${EVAL_FRACTION:-1.0}"
CELLS=""
TASKS_CSV="${TASKS_CSV:-lcc,musique,2wikimqa,qasper,hotpotqa}"
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cells) CELLS="$2"; shift 2 ;;
        --tasks) TASKS_CSV="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --fraction) FRACTION="$2"; shift 2 ;;
        --jobs-per-gpu) JOBS_PER_GPU="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done
[[ -n "$CELLS" ]] || { echo "ERROR: --cells required" >&2; exit 1; }

MODEL="meta-llama/Llama-3.1-8B-Instruct"
EXCLUDE_INDICES_FILE="${REPO_ROOT}/artifacts/calibration_splits/longbench_compact8_60_seed20260504_2k32k/exclude_train_indices_for_eval.json"
OMEGA_STATS="${REPO_ROOT}/artifacts/page_quant2/omega_stats_llama31_8b.pt"
PROBE_JSON="${REPO_ROOT}/artifacts/page_quant2/page_selection_probe.json"
OUT_BASE="${REPO_ROOT}/artifacts/bench_evict/llama31_8b"
LOG_DIR="${REPO_ROOT}/logs/bench_evict_llama31_8b"

for f in "$EXCLUDE_INDICES_FILE"; do
    [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done

IFS=',' read -ra TASKS <<< "$TASKS_CSV"
mkdir -p "$OUT_BASE" "$LOG_DIR"
STAMP=$(date '+%H%M%S')
CMDS="$LOG_DIR/commands_${STAMP}.jsonl"
: > "$CMDS"

sha8() { sha256sum "$1" | cut -c1-8; }

IFS=',' read -ra CELL_ARR <<< "$CELLS"
for cell in "${CELL_ARR[@]}"; do
    press="${cell%%:*}"; ratio="${cell##*:}"
    SCORE_MODE=""
    if [[ "$press" == omega_page || "$press" == omega_page_random ]]; then
        [[ -f "$OMEGA_STATS" ]] || { echo "ERROR: missing $OMEGA_STATS (run probe_page_selection.py)" >&2; exit 1; }
        [[ -f "$PROBE_JSON" ]] || { echo "ERROR: missing $PROBE_JSON (frozen score_mode)" >&2; exit 1; }
        if [[ "$press" == omega_page_random ]]; then
            SCORE_MODE="random_page"
        else
            SCORE_MODE=$(.venv/bin/python -c "import json;print(json.load(open('$PROBE_JSON'))['frozen']['score_mode'])")
        fi
        SHA=$(sha8 "$OMEGA_STATS")
        label_press="omega_page_${SCORE_MODE}__${SHA}"
        PRESS_NAME="omega_page"
    else
        label_press="$press"
        PRESS_NAME="$press"
    fi
    for task in "${TASKS[@]}"; do
        label="evict__${label_press}__r${ratio}__${task}"
        [[ "$FRACTION" == "1.0" ]] || label="${label}__f${FRACTION}"
        PRESS_NAME="$PRESS_NAME" SCORE_MODE="$SCORE_MODE" ratio="$ratio" \
        task="$task" label="$label" FRACTION="$FRACTION" \
        OMEGA_STATS="$OMEGA_STATS" OUT_BASE="$OUT_BASE" \
        EXCLUDE_INDICES_FILE="$EXCLUDE_INDICES_FILE" \
        .venv/bin/python - <<'PY' >> "$CMDS"
import json, os
e = os.environ
cmd = {
    "_label": e["label"],
    "press_name": e["PRESS_NAME"],
    "compression_ratio": float(e["ratio"]),
    "dataset": "longbench",
    "data_dir": e["task"],
    "fraction": float(e["FRACTION"]),
    "exclude_indices_file": e["EXCLUDE_INDICES_FILE"],
    "output_dir": f'{e["OUT_BASE"]}/{e["label"]}',
}
if e["SCORE_MODE"]:
    cmd["press_kwargs"] = {
        "omega_stats_path": e["OMEGA_STATS"],
        "score_mode": e["SCORE_MODE"],
        "page_size": 64,
    }
print(json.dumps(cmd))
PY
    done
done

n_jobs=$(wc -l < "$CMDS")
echo "[$(date '+%H:%M:%S')] queued $n_jobs eviction cells in $CMDS (cells: $CELLS, fraction: $FRACTION)"

if [[ "$DRY_RUN" -eq 1 ]]; then
    head -2 "$CMDS"
    echo "dry run — not executing"
    exit 0
fi

.venv/bin/python pipelines/bench/worker.py \
    --model "$MODEL" \
    --commands-file "$CMDS" \
    --log-dir "$LOG_DIR" \
    --gpus "$GPUS" \
    --jobs-per-gpu "$JOBS_PER_GPU" \
    --max-retries "$MAX_RETRIES"
