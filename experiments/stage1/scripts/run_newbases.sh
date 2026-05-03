#!/bin/bash
# Stage 1E: run only the new-bases methods (cca_orth_*, r_sym_*) through E3/E4a/E4b.
# Outputs into dedicated subdirs that will be merged into canonical via merge_newbases.py.
#
# Usage:
#   experiments/stage1/scripts/run_newbases.sh                  # all phases
#   experiments/stage1/scripts/run_newbases.sh --phase e3       # only E3
#   experiments/stage1/scripts/run_newbases.sh --phases e3,e4a  # subset
#
# GPU pool defaults to 0,1,2,3 (the user's allocation).

set -e
set -o pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LAUNCHER="${REPO_ROOT}/experiments/stage1/scripts/launch_cca_study.sh"
LOGS_DIR="${REPO_ROOT}/experiments/stage1/logs"
mkdir -p "${LOGS_DIR}"

PHASES="e3,e4a,e4b"
GPUS="0,1,2,3"
METHODS="cca_orth_uniform,cca_orth_waterfill,r_sym_uniform,r_sym_waterfill"
RUN_SUFFIX="_newbases"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phases) PHASES="$2"; shift 2 ;;
        --phase)  PHASES="$2"; shift 2 ;;
        --gpus)   GPUS="$2"; shift 2 ;;
        --methods) METHODS="$2"; shift 2 ;;
        *)
            echo "Unknown flag: $1" >&2
            exit 1
            ;;
    esac
done

IFS=',' read -ra PHASE_ARR <<< "${PHASES}"

run_phase() {
    local phase="$1"
    local subdir="${phase}_newbases"
    local extra=""
    case "${phase}" in
        e3)  extra="--b-avgs 2,3,4 --query-phase both" ;;
        e4a) extra="" ;;
        e4b) extra="" ;;
        e5)  extra="--b-avgs 3 --query-phase both" ;;
        *) echo "Unsupported phase ${phase}" >&2; return 1 ;;
    esac
    echo "[$(date '+%F %T')] === phase=${phase} subdir=${subdir} ==="
    "${LAUNCHER}" \
        --phase "${phase}" \
        --gpus "${GPUS}" \
        --methods "${METHODS}" \
        --output-subdir "${subdir}" \
        --run-suffix "${RUN_SUFFIX}" \
        ${extra}
}

EXIT=0
for ph in "${PHASE_ARR[@]}"; do
    if ! run_phase "${ph}"; then
        echo "[$(date '+%F %T')] phase=${ph} FAILED"
        EXIT=1
    fi
done

echo "[$(date '+%F %T')] run_newbases.sh done; exit=${EXIT}"
exit "${EXIT}"
