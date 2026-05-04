#!/bin/bash
# Phase 5: Llama-3.1-8B Stage-1E reproduction (E3 + E5 only).
# Uses the existing launch_cca_study.sh as a template, hardcoded for Llama bundle/output.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

GPUS="0,1,2,3,4,5"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS="$2"; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

LOG_DIR="${REPO_ROOT}/experiments/stage1/logs/phase5"
mkdir -p "$LOG_DIR"

LLAMA_BUNDLE="${REPO_ROOT}/artifacts/stage1/query_stats_longbench_under4k_llama31_8b"
LLAMA_CCA="${REPO_ROOT}/artifacts/stage1/cca_vs_waterfill_study/llama31_8b/cca_stats.pt"
OUT_SUBDIR="llama31_8b"

# launch_cca_study.sh hardcodes --phase e3 with --query-phase both, which produces
# both prefill (E3) and decode (E5) metrics in one summary.json per b_avg. So we
# only need a single phase=e3 dispatch.
METHODS="v3,v_truncate,v_waterfill,cca_uniform,cca_waterfill,cca_orth_uniform,cca_orth_waterfill,r_sym_uniform,r_sym_waterfill"

echo "[$(date '+%H:%M:%S')] Phase 5: dispatching Llama E3+E5 (gpus $GPUS)"
bash "${REPO_ROOT}/experiments/stage1/scripts/launch_cca_study.sh" \
    --phase e3 \
    --gpus "$GPUS" \
    --b-avgs 2,3,4 \
    --rank 64 \
    --methods "$METHODS" \
    --query-phase both \
    --output-subdir "$OUT_SUBDIR" \
    --extra-args "--bundle ${LLAMA_BUNDLE} --cca-stats ${LLAMA_CCA}" \
    2>&1 | tee "${LOG_DIR}/llama_e3_dispatcher.log"

echo "[$(date '+%H:%M:%S')] Phase 5: Llama Stage-1E complete"
