#!/bin/bash
# Phase 1AB launcher: V-only sweep (4 methods × 3 budgets) + K-only sweep (3 budgets) + 1 oracle.
#
# Runs JointQKPress with quantize_k/quantize_v flags via the parallel launcher.
# All jobs run on Qwen3-8B against LongBench/qasper at fraction 0.3.
#
# Usage:
#   bash experiments/calibration_stability/launch_ab.sh [--gpus 0,1,2,3,4,5] [--fraction 0.3]
#   tail -f experiments/logs/phase1ab/_overview.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

GPUS="0,1,2,3,4,5"
FRACTION="0.3"
TASK="qasper"
MODEL="Qwen/Qwen3-8B"
CCA="${REPO_ROOT}/artifacts/bases/cca_stats.pt"
VST="${REPO_ROOT}/artifacts/v_bases/v_stats.pt"
OUT_BASE="${REPO_ROOT}/artifacts/v_bases/sweep"
LOG_DIR="${REPO_ROOT}/experiments/logs/phase1ab"
DRY_RUN=0
ONLY_VMETHOD=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS="$2"; shift 2 ;;
        --fraction) FRACTION="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --only-vmethod) ONLY_VMETHOD="$2"; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$OUT_BASE" "$LOG_DIR"
CMDS="$LOG_DIR/commands.txt"
: > "$CMDS"

# Each command runs from the repo root with PYTHONPATH set; the underlying
# kvpress/evaluation/evaluate.py needs to run with cwd=kvpress/evaluation/ for
# its `from benchmarks.* import ...` style. We wrap each command in a subshell.

# Each command: activate efficient-llm conda env, cd to kvpress/evaluation/,
# set PYTHONPATH so kvpress and project root are importable, then run evaluate.py.
CONDA_ACTIVATE='source ~/miniconda3/etc/profile.d/conda.sh && conda activate efficient-llm'
PY_PATH="${REPO_ROOT}:${REPO_ROOT}/kvpress"

# Helper: emit one command line + label
emit_cmd() {
    local press_name="$1"
    local press_kwargs_json="$2"
    local out_subdir="$3"
    local label="$4"

    local cmd="${CONDA_ACTIVATE} && cd ${REPO_ROOT}/kvpress/evaluation && PYTHONPATH=${PY_PATH} python -u evaluate.py \
        --press_name=${press_name} \
        --press_kwargs='${press_kwargs_json}' \
        --model='${MODEL}' \
        --dataset=longbench \
        --data_dir=${TASK} \
        --fraction=${FRACTION} \
        --output_dir=${OUT_BASE}/${out_subdir} # label=${label}"
    echo "$cmd" >> "$CMDS"
}

if [[ -z "$ONLY_VMETHOD" ]]; then
    # 1) Full-precision oracle (uses no_press)
    echo "${CONDA_ACTIVATE} && cd ${REPO_ROOT}/kvpress/evaluation && PYTHONPATH=${PY_PATH} python -u evaluate.py \
        --press_name=no_press \
        --compression_ratio=0.0 \
        --model='${MODEL}' \
        --dataset=longbench \
        --data_dir=${TASK} \
        --fraction=${FRACTION} \
        --output_dir=${OUT_BASE}/full_precision # label=full_precision" >> "$CMDS"
fi

# 2) V-only sweep: calibrated V builders + TurboQuant V3 value compressor.
for vmethod in v_random v_eigen_uniform v_eigen_waterfill v_turboquant; do
    if [[ -n "$ONLY_VMETHOD" && "$vmethod" != "$ONLY_VMETHOD" ]]; then
        continue
    fi
    for vb in 2 3 4; do
        if [[ "$vmethod" == "v_turboquant" ]]; then
            json="{\"v_method\": \"${vmethod}\", \"v_bits\": ${vb}, \"quantize_k\": False, \"quantize_v\": True, \"compress_decode\": False}"
        else
            json="{\"v_stats_path\": \"${VST}\", \"v_method\": \"${vmethod}\", \"v_bits\": ${vb}, \"quantize_k\": False, \"quantize_v\": True, \"compress_decode\": False}"
        fi
        emit_cmd "jointqk" "$json" "vonly_${vmethod}_v${vb}" "vonly_${vmethod}_v${vb}"
    done
done

if [[ -z "$ONLY_VMETHOD" ]]; then
    # 3) K-only sweep: 3 budgets = 3 jobs
    for kb in 2 3 4; do
        json="{\"cca_stats_path\": \"${CCA}\", \"k_method\": \"r_sym_waterfill\", \"k_bits\": ${kb}, \"quantize_k\": True, \"quantize_v\": False, \"compress_decode\": False}"
        emit_cmd "jointqk" "$json" "konly_k${kb}" "konly_k${kb}"
    done
fi

n_jobs=$(wc -l < "$CMDS")
echo "[$(date '+%H:%M:%S')] Phase 1AB: $n_jobs jobs queued in $CMDS"
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[$(date '+%H:%M:%S')] DRY RUN: commands written to $CMDS, dispatcher SKIPPED"
    exit 0
fi
echo "[$(date '+%H:%M:%S')] Dispatching across GPUs $GPUS, logs in $LOG_DIR/"

python "${REPO_ROOT}/experiments/bench/parallel_launcher.py" \
    --commands-file "$CMDS" \
    --log-dir "$LOG_DIR" \
    --gpus "$GPUS" \
    --label-from-comment
