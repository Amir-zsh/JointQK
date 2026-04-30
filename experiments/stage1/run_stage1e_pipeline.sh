#!/bin/bash
# Stage 1E end-to-end pipeline orchestrator.
#
# Phases (in order):
#   1. E1+E2 diagnostics (run_cca_diagnostics, single-process; ~3-5 min cold, <1 min cached)
#   2. gate_e1_e2
#   3. E3 (real quantization, 3-way parallel on GPUs 0,1,2; query-phase=both → E5 piggybacks)
#   4. gate_e3
#   5. E4a cross-task (3-way parallel)
#   6. E4b within-task LOO (24 folds, 4-way parallel × 6 batches)
#   7. gate_e4
#   8. gate_e5
#   9. gate_integration
#  10. write_stage1e_report → stage1e_cca_vs_waterfill_report.md + INDEX.md
#
# Markers:
#   STAGE1E_DONE         (success)
#   STAGE1E_FAILED       (any gate failed)
#
# Flags:
#   --gpus 0,1,2,3       GPU pool (default 0,1,2,3)
#   --skip-e1-e2         skip E1+E2 if metrics_e1_e2.json already exists
#   --skip-e3            skip E3 if all e3 summaries present
#   --skip-e4            skip E4 (a+b) if summaries present
#   --only PHASE         run only one phase {e1, e3, e4a, e4b, gates}
#   --force              ignore --skip-* and rerun everything

set -e
set -o pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ARTIFACT_BASE="${REPO_ROOT}/artifacts/stage1/cca_vs_waterfill_study"
LOGS_DIR="${REPO_ROOT}/experiments/stage1/logs"
DONE_MARKER="${ARTIFACT_BASE}/STAGE1E_DONE"
FAIL_MARKER="${ARTIFACT_BASE}/STAGE1E_FAILED"
mkdir -p "${ARTIFACT_BASE}" "${LOGS_DIR}"

GPUS="0,1,2,3"
ONLY=""
FORCE=0
SKIP_E1_E2=0
SKIP_E3=0
SKIP_E4=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS="$2"; shift 2 ;;
        --only) ONLY="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --skip-e1-e2) SKIP_E1_E2=1; shift ;;
        --skip-e3) SKIP_E3=1; shift ;;
        --skip-e4) SKIP_E4=1; shift ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

cd "${REPO_ROOT}"

log_phase() { echo -e "\n[$(date '+%Y-%m-%d %H:%M:%S')] === $1 ===" | tee -a "${LOGS_DIR}/pipeline.log"; }
write_fail() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: $1" | tee -a "${LOGS_DIR}/pipeline.log"
    {
        echo "Stage 1E pipeline failed at: $1"
        echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "See logs at ${LOGS_DIR}/"
    } > "${FAIL_MARKER}"
    bash "${REPO_ROOT}/experiments/stage1/scripts/build_index.sh" failed "$1" >> "${LOGS_DIR}/pipeline.log" 2>&1 || true
    exit 1
}

run_gate() {
    local gate_name="$1"
    log_phase "${gate_name}"
    python -u -m "experiments.stage1.gates.${gate_name}" 2>&1 | tee -a "${LOGS_DIR}/pipeline.log" || write_fail "${gate_name}"
}

if [[ "${ONLY}" == "gates" ]]; then
    run_gate "gate_e1_e2"
    run_gate "gate_e3"
    run_gate "gate_e4"
    run_gate "gate_e5"
    run_gate "gate_integration"
    echo -e "\n[$(date '+%Y-%m-%d %H:%M:%S')] Stage 1E gates OK"
    exit 0
fi

rm -f "${DONE_MARKER}" "${FAIL_MARKER}"

# Phase 1: E1+E2
if [[ "${ONLY}" == "" ]] || [[ "${ONLY}" == "e1" ]] || [[ "${ONLY}" == "all" ]]; then
    if [[ "${FORCE}" -eq 0 ]] && [[ "${SKIP_E1_E2}" -eq 1 ]] && [[ -f "${ARTIFACT_BASE}/metrics_e1_e2.json" ]]; then
        log_phase "E1+E2 skipped by flag"
    elif [[ "${FORCE}" -eq 1 ]] || [[ ! -f "${ARTIFACT_BASE}/metrics_e1_e2.json" ]]; then
        log_phase "E1+E2 diagnostics"
        python -u -m experiments.stage1.run_cca_diagnostics 2>&1 | tee -a "${LOGS_DIR}/pipeline.log" || write_fail "run_cca_diagnostics"
    else
        log_phase "E1+E2 already done (cached)"
    fi
    log_phase "gate_e1_e2"
    python -u -m experiments.stage1.gates.gate_e1_e2 2>&1 | tee -a "${LOGS_DIR}/pipeline.log" || write_fail "gate_e1_e2"
fi

# Phase 2: E3 (with E5 piggyback via --query-phase both)
if [[ "${ONLY}" == "" ]] || [[ "${ONLY}" == "e3" ]] || [[ "${ONLY}" == "all" ]]; then
    if [[ "${FORCE}" -eq 0 ]] && [[ "${SKIP_E3}" -eq 1 ]] \
        && [[ -f "${ARTIFACT_BASE}/e3/e3_b2_r64_summary.json" ]] \
        && [[ -f "${ARTIFACT_BASE}/e3/e3_b3_r64_summary.json" ]] \
        && [[ -f "${ARTIFACT_BASE}/e3/e3_b4_r64_summary.json" ]]; then
        log_phase "E3 skipped by flag"
    else
        log_phase "E3 (real quantization, 3-way parallel)"
        bash "${REPO_ROOT}/experiments/stage1/scripts/launch_cca_study.sh" --phase e3 --gpus "${GPUS}" --b-avgs 2,3,4 --query-phase both 2>&1 | tee -a "${LOGS_DIR}/pipeline.log" || write_fail "E3 launcher"
    fi
    log_phase "gate_e3"
    python -u -m experiments.stage1.gates.gate_e3 2>&1 | tee -a "${LOGS_DIR}/pipeline.log" || write_fail "gate_e3"
    log_phase "gate_e5 (decode-phase pulled from E3 outputs)"
    python -u -m experiments.stage1.gates.gate_e5 2>&1 | tee -a "${LOGS_DIR}/pipeline.log" || write_fail "gate_e5"
fi

# Phase 3: E4a cross-task
if [[ "${ONLY}" == "" ]] || [[ "${ONLY}" == "e4a" ]] || [[ "${ONLY}" == "all" ]]; then
    if [[ "${FORCE}" -eq 0 ]] && [[ "${SKIP_E4}" -eq 1 ]] && [[ $(find "${ARTIFACT_BASE}/e4a" -name 'e4a_calib_*_b*_r*_summary.json' 2>/dev/null | wc -l) -ge 3 ]]; then
        log_phase "E4a skipped by flag"
    else
        log_phase "E4a cross-task (3-way parallel)"
        bash "${REPO_ROOT}/experiments/stage1/scripts/launch_cca_study.sh" --phase e4a --gpus "${GPUS}" 2>&1 | tee -a "${LOGS_DIR}/pipeline.log" || write_fail "E4a launcher"
    fi
fi

# Phase 4: E4b LOO
if [[ "${ONLY}" == "" ]] || [[ "${ONLY}" == "e4b" ]] || [[ "${ONLY}" == "all" ]]; then
    if [[ "${FORCE}" -eq 0 ]] && [[ "${SKIP_E4}" -eq 1 ]] && [[ $(find "${ARTIFACT_BASE}/e4b" -name 'e4b_*_loo*_b*_r*_summary.json' 2>/dev/null | wc -l) -ge 24 ]]; then
        log_phase "E4b skipped by flag"
    else
        log_phase "E4b LOO (24 folds, 4-way parallel × 6 batches)"
        bash "${REPO_ROOT}/experiments/stage1/scripts/launch_cca_study.sh" --phase e4b --gpus "${GPUS}" 2>&1 | tee -a "${LOGS_DIR}/pipeline.log" || write_fail "E4b launcher"
    fi
    log_phase "gate_e4"
    python -u -m experiments.stage1.gates.gate_e4 2>&1 | tee -a "${LOGS_DIR}/pipeline.log" || write_fail "gate_e4"
fi

# Phase 5: integration + report
if [[ "${ONLY}" == "" ]] || [[ "${ONLY}" == "report" ]] || [[ "${ONLY}" == "all" ]]; then
    log_phase "gate_integration"
    python -u -m experiments.stage1.gates.gate_integration 2>&1 | tee -a "${LOGS_DIR}/pipeline.log" || write_fail "gate_integration"
    log_phase "write_stage1e_report"
    python -u -m experiments.stage1.write_stage1e_report 2>&1 | tee -a "${LOGS_DIR}/pipeline.log" || write_fail "write_stage1e_report"
    bash "${REPO_ROOT}/experiments/stage1/scripts/build_index.sh" success "" 2>&1 | tee -a "${LOGS_DIR}/pipeline.log" || true
    {
        echo "Stage 1E pipeline completed."
        echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Review packet: ${ARTIFACT_BASE}/INDEX.md"
        echo "Report: notes/stage1/stage1e_cca_vs_waterfill_report.md"
    } > "${DONE_MARKER}"
fi

echo -e "\n[$(date '+%Y-%m-%d %H:%M:%S')] Stage 1E pipeline OK"
