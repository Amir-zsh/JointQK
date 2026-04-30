#!/bin/bash
# Launch Stage 1E E3/E4/E5 runs in parallel on a GPU pool.
# Default GPU pool: 0,1,2,3.
#
# Usage:
#   launch_cca_study.sh --phase e3
#   launch_cca_study.sh --phase e4a
#   launch_cca_study.sh --phase e4b
#   launch_cca_study.sh --phase e5
#   launch_cca_study.sh --phase e3 --gpus 0,1,2 --b-avgs 2,3,4
#
# Writes:
#   experiments/stage1/logs/<run_name>.log              streamed stdout/stderr
#   experiments/stage1/logs/<run_name>.heartbeat        touched periodically by runner
#   experiments/stage1/logs/<run_name>.summary.json     on clean exit
#   experiments/stage1/logs/<run_name>.FAILED           on non-zero exit
#   experiments/stage1/logs/_registry.tsv               run_name TAB gpu TAB pid TAB cmd TAB started_at
#   experiments/stage1/logs/_registry.tsv.lock          short-lived lockfile
#
# Waits for all backgrounded children before exit. Exit code:
#   0 if all runs reported clean .summary.json
#   1 if any .FAILED present after wait

set -e
set -o pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LOGS_DIR="${REPO_ROOT}/experiments/stage1/logs"
mkdir -p "${LOGS_DIR}"

PHASE=""
GPUS="0,1,2,3"
B_AVGS="2,3,4"
RANK="64"
METHODS="v3,v_truncate,v_waterfill,cca_uniform,cca_waterfill"
QUERY_PHASE="both"
ONLY_FAILED=0
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase) PHASE="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --b-avgs) B_AVGS="$2"; shift 2 ;;
        --rank) RANK="$2"; shift 2 ;;
        --methods) METHODS="$2"; shift 2 ;;
        --query-phase) QUERY_PHASE="$2"; shift 2 ;;
        --only-failed) ONLY_FAILED=1; shift ;;
        --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
        *)
            echo "Unknown flag: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${PHASE}" ]]; then
    echo "ERROR: --phase {e3,e4a,e4b,e5} is required" >&2
    exit 1
fi

IFS=',' read -ra GPU_ARR <<< "${GPUS}"
N_GPU=${#GPU_ARR[@]}

# Build the list of (run_name, cli_args) tuples for this phase.
RUNS=()
case "${PHASE}" in
    e3|e5)
        IFS=',' read -ra B_ARR <<< "${B_AVGS}"
        for B in "${B_ARR[@]}"; do
            run_name="${PHASE}_b${B}_r${RANK}"
            args="--phase e3 --b-avg ${B} --rank ${RANK} --methods ${METHODS} --query-phase ${QUERY_PHASE} --output-subdir ${PHASE} --run-name ${run_name} --full-precision-smoke-test"
            RUNS+=("${run_name}|${args}")
        done
        ;;
    e4a)
        # 3 calibration sources; each evaluated on all 24 examples.
        for CFG in qasper hotpotqa passage_retrieval_en; do
            run_name="e4a_calib_${CFG}_b3_r${RANK}"
            args="--phase e4a --b-avg 3.0 --rank ${RANK} --methods ${METHODS} --query-phase prefill --output-subdir e4a --run-name ${run_name} --calibration-config ${CFG}"
            RUNS+=("${run_name}|${args}")
        done
        ;;
    e4b)
        # 24 LOO folds (3 configs × 8 examples). Indices in manifest: 0-7 qasper, 8-15 hotpotqa, 16-23 passage_retrieval_en.
        for IDX in $(seq 0 7); do
            run_name="e4b_qasper_loo${IDX}_b3_r${RANK}"
            args="--phase e4b --b-avg 3.0 --rank ${RANK} --methods ${METHODS} --query-phase prefill --output-subdir e4b --run-name ${run_name} --loo-config qasper --loo-index ${IDX}"
            RUNS+=("${run_name}|${args}")
        done
        for IDX in $(seq 8 15); do
            run_name="e4b_hotpot_loo${IDX}_b3_r${RANK}"
            args="--phase e4b --b-avg 3.0 --rank ${RANK} --methods ${METHODS} --query-phase prefill --output-subdir e4b --run-name ${run_name} --loo-config hotpotqa --loo-index ${IDX}"
            RUNS+=("${run_name}|${args}")
        done
        for IDX in $(seq 16 23); do
            run_name="e4b_passage_loo${IDX}_b3_r${RANK}"
            args="--phase e4b --b-avg 3.0 --rank ${RANK} --methods ${METHODS} --query-phase prefill --output-subdir e4b --run-name ${run_name} --loo-config passage_retrieval_en --loo-index ${IDX}"
            RUNS+=("${run_name}|${args}")
        done
        ;;
    *)
        echo "ERROR: unsupported phase ${PHASE}" >&2
        exit 1
        ;;
esac

# If --only-failed, filter to runs that produced a .FAILED file.
if [[ "${ONLY_FAILED}" -eq 1 ]]; then
    FILTERED=()
    for entry in "${RUNS[@]}"; do
        name="${entry%%|*}"
        if [[ -f "${LOGS_DIR}/${name}.FAILED" ]]; then
            FILTERED+=("${entry}")
        fi
    done
    RUNS=("${FILTERED[@]}")
    if [[ ${#RUNS[@]} -eq 0 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] No FAILED runs to retry; exiting cleanly."
        exit 0
    fi
fi

REGISTRY="${LOGS_DIR}/_registry.tsv"

launch_one() {
    local run_name="$1"
    local cli_args="$2"
    local gpu_id="$3"
    local started_at; started_at="$(date '+%Y-%m-%d %H:%M:%S')"
    rm -f "${LOGS_DIR}/${run_name}.FAILED" "${LOGS_DIR}/${run_name}.summary.json"
    : > "${LOGS_DIR}/${run_name}.log"
    : > "${LOGS_DIR}/${run_name}.heartbeat"
    (
        cd "${REPO_ROOT}"
        # Cap BLAS thread pool to avoid 4-way oversubscription thrash on shared cores.
        # Each process gets ~8 threads; with 4 processes that's 32 cores total.
        OMP_NUM_THREADS=8 \
        MKL_NUM_THREADS=8 \
        OPENBLAS_NUM_THREADS=8 \
        NUMEXPR_NUM_THREADS=8 \
        TORCH_NUM_THREADS=8 \
        CUDA_VISIBLE_DEVICES="${gpu_id}" python -u -m experiments.stage1.run_cca_vs_waterfill_study \
            ${cli_args} \
            --device cuda \
            > "${LOGS_DIR}/${run_name}.log" 2>&1
    ) &
    local pid=$!
    echo -e "${run_name}\t${gpu_id}\t${pid}\t${cli_args}\t${started_at}" >> "${REGISTRY}"
    echo "[${started_at}] launched run=${run_name} on gpu=${gpu_id} pid=${pid}"
}

# Initialize registry with a header (overwrite for new launch).
> "${REGISTRY}"
echo -e "run_name\tgpu\tpid\tcli_args\tstarted_at" >> "${REGISTRY}"

# Pool launching: assign runs to GPUs in batches of N_GPU at a time.
i=0
PIDS_BATCH=()
for entry in "${RUNS[@]}"; do
    name="${entry%%|*}"
    args="${entry#*|}"
    gpu="${GPU_ARR[$((i % N_GPU))]}"
    launch_one "${name}" "${args}" "${gpu}"
    PIDS_BATCH+=($!)
    i=$((i + 1))
    # When batch is full, wait for it to drain before launching more.
    if (( i % N_GPU == 0 )); then
        wait
        PIDS_BATCH=()
    fi
done
# Drain any remaining
wait

# Final status check: mark FAILED if non-zero exit or missing summary.
EXIT_CODE=0
for entry in "${RUNS[@]}"; do
    name="${entry%%|*}"
    if [[ -f "${LOGS_DIR}/${name}.FAILED" ]] || [[ ! -f "${LOGS_DIR}/${name}.summary.json" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: ${name}"
        EXIT_CODE=1
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] OK:     ${name}"
    fi
done

exit ${EXIT_CODE}
