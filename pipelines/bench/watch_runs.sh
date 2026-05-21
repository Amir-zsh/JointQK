#!/bin/bash
# Single-screen status dashboard for parallel Stage 1E runs.
#
# Usage:  bash watch_runs.sh
#
# Reads logs/_registry.tsv and shows, per run:
#   - run_name, gpu, pid, started, status
#   - last progress line from log
#   - heartbeat age (seconds)
#   - GPU utilization for that GPU (from nvidia-smi)
#
# Loops with watch -n 5; ctrl+c to exit.

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LOGS_DIR="${REPO_ROOT}/logs"
REGISTRY="${LOGS_DIR}/_registry.tsv"

render_dashboard() {
    if [[ ! -f "${REGISTRY}" ]]; then
        echo "(no registry yet at ${REGISTRY})"
        return
    fi
    printf "Stage 1E run dashboard — %s\n" "$(date '+%Y-%m-%d %H:%M:%S')"
    printf "Registry: %s\n\n" "${REGISTRY}"
    if command -v nvidia-smi >/dev/null 2>&1; then
        printf "%-6s %-6s %-7s %-7s %-7s\n" "GPU" "Util%" "Mem%" "MemUsedMB" "MemTotMB"
        nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | awk '{printf "%-6s %-6s %-7s %-7s %-7s\n", $1, $2, $3, $4, $5}'
        printf "\n"
    fi
    printf "%-30s %-4s %-8s %-7s %-19s %-7s %s\n" "run_name" "gpu" "pid" "status" "started" "hb_age" "last_progress"
    printf -- "----------------------------------------------------------------------------------------------------------------------------------\n"
    tail -n +2 "${REGISTRY}" | while IFS=$'\t' read -r name gpu pid args started; do
        # Status
        if [[ -f "${LOGS_DIR}/${name}.FAILED" ]]; then
            status="FAIL"
        elif [[ -f "${LOGS_DIR}/${name}.summary.json" ]]; then
            status="DONE"
        else
            if kill -0 "${pid}" 2>/dev/null; then
                status="RUN"
            else
                status="DEAD"
            fi
        fi
        # Heartbeat age in seconds (or "-" if missing)
        if [[ -f "${LOGS_DIR}/${name}.heartbeat" ]]; then
            hb_t=$(stat -c %Y "${LOGS_DIR}/${name}.heartbeat" 2>/dev/null || stat -f %m "${LOGS_DIR}/${name}.heartbeat")
            now=$(date +%s)
            hb_age="$((now - hb_t))s"
        else
            hb_age="-"
        fi
        # Last progress line
        last_line=$(grep -E "^\[.* progress: " "${LOGS_DIR}/${name}.log" 2>/dev/null | tail -1 | sed 's/^\[.*\] //' | head -c 70)
        if [[ -z "${last_line}" ]]; then
            last_line=$(tail -1 "${LOGS_DIR}/${name}.log" 2>/dev/null | sed 's/^\[.*\] //' | head -c 70)
        fi
        printf "%-30s %-4s %-8s %-7s %-19s %-7s %s\n" "${name:0:30}" "${gpu}" "${pid}" "${status}" "${started:0:19}" "${hb_age}" "${last_line}"
    done
}

if [[ "${1:-}" == "--once" ]]; then
    render_dashboard
else
    while true; do
        clear
        render_dashboard
        sleep 5
    done
fi
