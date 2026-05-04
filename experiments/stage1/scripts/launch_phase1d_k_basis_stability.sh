#!/bin/bash
# Phase 1D: offline K-basis calibration stability study.
#
# This does not run model inference. It recomputes Q/K second moments from the
# 24-example calibration bundle, fits R_sym bases from progressively larger
# calibration subsets, and measures basis/allocation/distortion stability.
#
# Usage:
#   bash experiments/stage1/scripts/launch_phase1d_k_basis_stability.sh --gpus 6,7
#   tail -f experiments/stage1/logs/phase1d_k_basis_stability.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

GPUS="6"
BUNDLE="${REPO_ROOT}/artifacts/stage1/query_stats_longbench_under4k"
OUT="${REPO_ROOT}/artifacts/stage1/v_method_study/k_basis_stability"
LOG="${REPO_ROOT}/experiments/stage1/logs/phase1d_k_basis_stability.log"
SAMPLE_SIZES="1,2,4,8,16,25"
REPETITIONS="20"
RANKS="16,32,64,96"
K_BITS="2,3,4"
INCLUDE_LOO=0
MOMENTS_CACHE=""
HOLDOUT_PER_CONFIG=""
HOLDOUT_SEED=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPUS="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --bundle) BUNDLE="$2"; shift 2 ;;
        --output-dir) OUT="$2"; shift 2 ;;
        --sample-sizes) SAMPLE_SIZES="$2"; shift 2 ;;
        --repetitions) REPETITIONS="$2"; shift 2 ;;
        --ranks) RANKS="$2"; shift 2 ;;
        --k-bits) K_BITS="$2"; shift 2 ;;
        --include-loo) INCLUDE_LOO=1; shift ;;
        --moments-cache) MOMENTS_CACHE="$2"; shift 2 ;;
        --holdout-per-config) HOLDOUT_PER_CONFIG="$2"; shift 2 ;;
        --holdout-seed) HOLDOUT_SEED="$2"; shift 2 ;;
        --log) LOG="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$(dirname "$LOG")" "$OUT"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
N_SHARDS="${#GPU_ARR[@]}"

echo "[$(date '+%H:%M:%S')] Phase 1D K-basis stability"
echo "  GPUs=${GPUS} (shards=${N_SHARDS})"
echo "  bundle=${BUNDLE}"
echo "  output=${OUT}"
echo "  sample_sizes=${SAMPLE_SIZES} repetitions=${REPETITIONS} ranks=${RANKS} k_bits=${K_BITS}"
echo "  include_loo=${INCLUDE_LOO}"
if [[ -n "$MOMENTS_CACHE" ]]; then
    echo "  moments_cache=${MOMENTS_CACHE}"
fi
if [[ -n "$HOLDOUT_PER_CONFIG" ]]; then
    echo "  holdout_per_config=${HOLDOUT_PER_CONFIG}"
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY RUN:"
    for shard_id in "${!GPU_ARR[@]}"; do
        gpu="${GPU_ARR[$shard_id]}"
        rows_name="phase1d_k_basis_stability_rows.shard${shard_id}.jsonl"
        loo_arg=""
        if [[ "$INCLUDE_LOO" -eq 1 ]]; then
            loo_arg="--include-loo"
        fi
        cache_arg=""
        if [[ -n "$MOMENTS_CACHE" ]]; then
            cache_arg="--moments-cache ${MOMENTS_CACHE}"
        fi
        holdout_arg=""
        if [[ -n "$HOLDOUT_PER_CONFIG" ]]; then
            holdout_arg="--holdout-per-config ${HOLDOUT_PER_CONFIG}"
        fi
        if [[ -n "$HOLDOUT_SEED" ]]; then
            holdout_arg="${holdout_arg} --holdout-seed ${HOLDOUT_SEED}"
        fi
        echo "CUDA_VISIBLE_DEVICES=${gpu} ${REPO_ROOT}/.venv/bin/python ${REPO_ROOT}/experiments/stage1/scripts/run_phase1d_k_basis_stability.py --device cuda --bundle ${BUNDLE} --output-dir ${OUT} --sample-sizes ${SAMPLE_SIZES} --repetitions ${REPETITIONS} --ranks ${RANKS} --k-bits ${K_BITS} --num-shards ${N_SHARDS} --shard-id ${shard_id} --rows-filename ${rows_name} ${loo_arg} ${cache_arg} ${holdout_arg}"
    done
    exit 0
fi

echo "[$(date '+%H:%M:%S')] writing log to ${LOG}"
: > "$LOG"

pids=()
rows_files=()
for shard_id in "${!GPU_ARR[@]}"; do
    gpu="${GPU_ARR[$shard_id]}"
    rows_name="phase1d_k_basis_stability_rows.shard${shard_id}.jsonl"
    rows_files+=("${OUT}/${rows_name}")
    cmd="CUDA_VISIBLE_DEVICES=${gpu} ${REPO_ROOT}/.venv/bin/python ${REPO_ROOT}/experiments/stage1/scripts/run_phase1d_k_basis_stability.py \
        --device cuda \
        --bundle ${BUNDLE} \
        --output-dir ${OUT} \
        --sample-sizes ${SAMPLE_SIZES} \
        --repetitions ${REPETITIONS} \
        --ranks ${RANKS} \
        --k-bits ${K_BITS} \
        --num-shards ${N_SHARDS} \
        --shard-id ${shard_id} \
        --rows-filename ${rows_name}"
    if [[ "$INCLUDE_LOO" -eq 1 ]]; then
        cmd="${cmd} --include-loo"
    fi
    if [[ -n "$MOMENTS_CACHE" ]]; then
        cmd="${cmd} --moments-cache ${MOMENTS_CACHE}"
    fi
    if [[ -n "$HOLDOUT_PER_CONFIG" ]]; then
        cmd="${cmd} --holdout-per-config ${HOLDOUT_PER_CONFIG}"
    fi
    if [[ -n "$HOLDOUT_SEED" ]]; then
        cmd="${cmd} --holdout-seed ${HOLDOUT_SEED}"
    fi
    echo "[$(date '+%H:%M:%S')] shard ${shard_id}/${N_SHARDS} gpu=${gpu}" | tee -a "$LOG"
    bash -c "$cmd" >> "$LOG" 2>&1 &
    pids+=("$!")
done

fail=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        fail=1
    fi
done
if [[ "$fail" -ne 0 ]]; then
    echo "[$(date '+%H:%M:%S')] Phase 1D FAILED; see ${LOG}" | tee -a "$LOG"
    exit 1
fi

"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/experiments/stage1/scripts/aggregate_phase1d_k_basis_stability.py" \
    --rows "${rows_files[@]}" \
    --output-dir "$OUT" 2>&1 | tee -a "$LOG"

echo "[$(date '+%H:%M:%S')] Phase 1D complete" | tee -a "$LOG"
