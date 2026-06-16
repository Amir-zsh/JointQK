#!/bin/bash
# EC (entropy-coded K) bench: lcc / musique / 2wikimqa at K=2, V=2 on
# Llama-3.1-8B, against the existing v7 baselines (full_precision,
# turboquant_k2_v2, jointqk_k2_v2 — already on disk, not re-run).
#
# Eval protocol is identical to the v7 sweep: fraction=1.0, compact8 train
# rows excluded, layer0_full_precision=True, Mode A (no decode compression).
# V side is v_turboquant @ 2 bits — the same V quantizer as jointqk_k2_v2 and
# (by seed rule) TurboQuantV3's value compressor, so the K side is the only
# difference under test.
#
#   bash pipelines/bench/launch_ec_longbench.sh --variants r_sym:0.375,hadamard:0.375
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPUS="${GPUS:-0,1,2,3}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
MAX_RETRIES="${MAX_RETRIES:-10}"
FRACTION="${EVAL_FRACTION:-1.0}"
VARIANTS="r_sym:0.375,hadamard:0.375"
DRY_RUN=0
MODE=percoord   # percoord (ell-weighted, *.pt) | uniform (single scalar/head, *__uniform.pt)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --variants) VARIANTS="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --fraction) FRACTION="$2"; shift 2 ;;
        --jobs-per-gpu) JOBS_PER_GPU="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done
case "$MODE" in
    percoord) BUNDLE_GLOB_SUFFIX="compact8train*[0-9].pt"; LABEL_SUFFIX="" ;;
    uniform)  BUNDLE_GLOB_SUFFIX="compact8train*__uniform.pt"; LABEL_SUFFIX="_uniform" ;;
    *) echo "Unknown --mode: $MODE (expected percoord|uniform)" >&2; exit 1 ;;
esac

MODEL="meta-llama/Llama-3.1-8B-Instruct"
CCA="${REPO_ROOT}/artifacts/bases/jointqk_llama31_8b_longbench_compact8_n400.pt"
VST="${REPO_ROOT}/artifacts/v_bases/v_stats_llama31_8b_longbench_compact8_n400.pt"
EXCLUDE_INDICES_FILE="${REPO_ROOT}/artifacts/calibration_splits/longbench_compact8_60_seed20260504_2k32k/exclude_train_indices_for_eval.json"
EC_DIR="${REPO_ROOT}/artifacts/ec/llama31_8b"
OUT_BASE="${REPO_ROOT}/artifacts/bench_ec/llama31_8b"
LOG_DIR="${REPO_ROOT}/logs/bench_ec_llama31_8b"

for f in "$CCA" "$VST" "$EXCLUDE_INDICES_FILE"; do
    [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done

TASKS=(lcc musique 2wikimqa)
mkdir -p "$OUT_BASE" "$LOG_DIR"
CMDS="$LOG_DIR/commands.jsonl"
: > "$CMDS"

IFS=',' read -ra VAR_ARR <<< "$VARIANTS"
for var in "${VAR_ARR[@]}"; do
    basis="${var%%:*}"; dz="${var##*:}"
    BUNDLE=$(ls "$EC_DIR"/ec_bundle__${basis}__b*__dz${dz}__${BUNDLE_GLOB_SUFFIX} 2>/dev/null | head -1)
    [[ -n "$BUNDLE" ]] || { echo "ERROR: no $MODE bundle for $basis dz=$dz in $EC_DIR" >&2; exit 1; }
    for task in "${TASKS[@]}"; do
        label="ec_${basis}_dz${dz}${LABEL_SUFFIX}_k2_v2_${task}"
        BUNDLE="$BUNDLE" basis="$basis" dz="$dz" task="$task" label="$label" \
        CCA="$CCA" VST="$VST" FRACTION="$FRACTION" OUT_BASE="$OUT_BASE" \
        EXCLUDE_INDICES_FILE="$EXCLUDE_INDICES_FILE" \
        .venv/bin/python - <<'PY' >> "$CMDS"
import json, os
e = os.environ
print(json.dumps({
    "_label": e["label"],
    "press_name": "jointqk",
    "press_kwargs": {
        "cca_stats_path": e["CCA"],
        "v_stats_path": e["VST"],
        "k_method": "ec_" + e["basis"],
        "ec_bundle_path": e["BUNDLE"],
        "v_method": "v_turboquant",
        "k_bits": 2,
        "v_bits": 2,
        "compress_decode": False,
        "layer0_full_precision": True,
        "quantize_k": True,
        "quantize_v": True,
    },
    "dataset": "longbench",
    "data_dir": e["task"],
    "fraction": float(e["FRACTION"]),
    "exclude_indices_file": e["EXCLUDE_INDICES_FILE"],
    "output_dir": f'{e["OUT_BASE"]}/{e["label"]}',
}))
PY
    done
done

n_jobs=$(wc -l < "$CMDS")
echo "[$(date '+%H:%M:%S')] queued $n_jobs EC cells in $CMDS (variants: $VARIANTS)"

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
