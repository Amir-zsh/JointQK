#!/bin/bash
# Run GPT-OSS math reasoning cells through the OSCAR SGLang stack.
#
# Prompt/cap protocol follows the corrected Qwen long-reasoning rows:
#   math500: first 200 rows, K=4, max_new=32768
#   aime25 :  30 rows, K=4, max_new=32768
#   gpqa   : 198 rows, K=4, max_new=32768
#
# Sampling follows OpenAI's gpt-oss recommendation:
#   temperature=1.0, top_p=1.0. OpenAI does not recommend a top_k; in SGLang
#   top_k=-1 disables top-k filtering.
#
# GPT-OSS requires TP=2 on this box, so this runner is intentionally sequential
# over methods and takes a comma-separated GPU pair.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

GPU_PAIR="${GPU_PAIR:-3,6}"
PORT="${PORT:-30920}"
OUT_ROOT="${OUT_ROOT:-artifacts/oscar_gptoss20b/math_grid}"
MODEL="${MODEL:-unsloth/gpt-oss-20b-BF16}"
ROT_DIR="${ROT_DIR:-$ROOT/artifacts/oscar_gptoss20b/rotations_gpqa198}"
VQ_CODEBOOK="${VQ_CODEBOOK:-$ROOT/artifacts/oscar_gptoss20b/vqa_gptoss20b_G4_strat_flat_ptn_gpqacc128k_fp8.pt}"
ARMS_CSV="${ARMS:-bf16,int2,vq2,old_vq2}"
TASKS_CSV="${TASKS:-math500,aime25}"
MATH_ROWS="${MATH_ROWS:-artifacts/prompt_rows/math500_gptoss_32k_n200.jsonl}"
AIME_ROWS="${AIME_ROWS:-artifacts/prompt_rows/aime25_gptoss_32k.jsonl}"
GPQA_ROWS="${GPQA_ROWS:-artifacts/prompt_rows/gpqa_diamond_gptoss_32k.jsonl}"
SAMPLES="${SAMPLES:-4}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
MATH_THREADS="${MATH_THREADS:-4}"
AIME_THREADS="${AIME_THREADS:-4}"
GPQA_THREADS="${GPQA_THREADS:-4}"
# The server runs in the oscar container; the host .venv-oscar raises
# "Either a revision or a version must be specified" out of transformers'
# hub_kernels on this box. Set SERVER_CONTAINER= (empty) to serve on the host.
# The client stays on the host either way -- see run_cell.
SERVER_CONTAINER="${SERVER_CONTAINER-oscar-ab}"

mkdir -p logs "$OUT_ROOT"

SERVER_PID=""

ensure_port_free() {
  if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "port $PORT is already in use; refusing to kill an external process" >&2
    lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >&2 || true
    return 1
  fi
}

stop_server() {
  if [[ -n "$SERVER_CONTAINER" ]]; then
    # docker exec -d leaves no host PID, so scope the kill by port rather than
    # by pattern -- a bare launch_server pattern would take down concurrent runs.
    docker exec "$SERVER_CONTAINER" bash -lc \
      "pkill -9 -f '[l]aunch_server.*--port $PORT'" >/dev/null 2>&1 || true
    for _ in $(seq 1 60); do
      lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 || break
      sleep 2
    done
  elif [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}

wait_ready() {
  local log="$1"
  for _ in $(seq 1 180); do
    grep -qE "The server is fired up and ready to roll|Uvicorn running on http://127.0.0.1:$PORT" "$log" && return 0
    grep -qE "Received sigquit|CUDA out of memory|Not enough memory|Traceback" "$log" && {
      tail -80 "$log"
      return 1
    }
    sleep 5
  done
  tail -80 "$log"
  return 1
}

boot_arm() {
  local arm="$1"
  # Port-scoped: two concurrent runs of the SAME arm (e.g. a radix-cache
  # on/off control pair) would otherwise share one log, and wait_ready would
  # read the other run's "fired up" line and proceed against a dead port.
  local log="logs/gptoss_math_${arm}_${PORT}_server.log"
  ensure_port_free
  : > "$log"

  local senv=(
    TP=2
    "MODEL=$MODEL"
    ABSORB_V_ROT=0
    QUANT_GROUP_SIZE=0
    "ROT_DIR=$ROT_DIR"
    "MAX_REQS=${MAX_REQS:-4}"
    "CUDA_GRAPH_BS=${CUDA_GRAPH_BS:-4}"
    "KV_SPLITS=${KV_SPLITS:-48}"
  )
  # Forward the knobs a caller may want to vary without teaching this script
  # about each one; unset names are simply skipped.
  local k
  for k in RADIX_CACHE MAX_TOKENS PREFILL_BACKEND SERVE_EXTRA \
           SGLANG_SWA_POOL_TOKENS SGLANG_MIXED_KV_PREFIX_REUSE_ACROSS_CHUNKS \
           SGLANG_MIXED_KV_HP_PREFIX_POOL_TOKENS \
           SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE \
           SGLANG_VQ2_CUDA SGLANG_VQ2_CUDA_FP32 SGLANG_VQ2_CUDA_THR \
           SGLANG_VQ2_CUDA_GEOM SGLANG_VQ_OPT_QMAP SGLANG_VQ_OPT_KMAP \
           SGLANG_VQ_OPT_FLUSH SGLANG_VQ_OPT_PREFILL SGLANG_VQ_FP8_FMT \
           SGLANG_CHECK_MIXED_KV_SWA; do
    [[ -n "${!k:-}" ]] && senv+=("$k=${!k}")
  done

  local args=(--gpu "$GPU_PAIR" --port "$PORT" --model "$MODEL" --ctx "${CTX:-73728}" --mem-frac "${MEM_FRAC:-0.78}")
  case "$arm" in
    bf16)
      args+=(--bf16)
      ;;
    int2)
      ;;
    vq2)
      args+=(--vq2 --vq-codebook "$VQ_CODEBOOK")
      senv+=(SGLANG_VQ_DISABLE_SINK_CORRECTION=0)
      ;;
    old_vq2)
      args+=(--vq2 --vq-codebook "$VQ_CODEBOOK")
      senv+=(SGLANG_VQ_DISABLE_SINK_CORRECTION=1)
      ;;
    *)
      echo "unknown arm: $arm" >&2
      return 2
      ;;
  esac

  echo "[$(date -Is)] boot arm=$arm gpu_pair=$GPU_PAIR port=$PORT container=${SERVER_CONTAINER:-<host>}" | tee -a "$log"
  if [[ -n "$SERVER_CONTAINER" ]]; then
    local eargs=()
    for k in "${senv[@]}"; do eargs+=(-e "$k"); done
    docker exec -d "${eargs[@]}" "$SERVER_CONTAINER" bash -lc \
      "cd $ROOT && bash pipelines/oscar_e2e/serve_oscar.sh ${args[*]} >> $ROOT/$log 2>&1"
    SERVER_PID=""
  else
    nohup env "${senv[@]}" bash pipelines/oscar_e2e/serve_oscar.sh "${args[@]}" >> "$log" 2>&1 &
    SERVER_PID=$!
  fi
  wait_ready "$log"
}

run_cell() {
  local arm="$1"
  local name="$2"
  local rows="$3"
  local threads="$4"
  local out="$OUT_ROOT/$arm/$name"
  if [[ -f "$out/metrics.json" ]]; then
    echo "[$(date -Is)] skip complete $arm/$name"
    return 0
  fi
  # The client only speaks HTTP, so it can run wherever; the server may not.
  # Overridable because .venv/bin/python's interpreter symlink points outside
  # the oscar-ab bind mount, so it resolves on the host but not in the container
  # -- run the whole script in the container and this line is what breaks.
  "${CLIENT_PYTHON:-.venv/bin/python}" pipelines/oscar_e2e/run_prompts_client.py \
    --rows "$rows" \
    --port "$PORT" \
    --timeout 7200 \
    --samples "$SAMPLES" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --threads "$threads" \
    --out "$out"
}

aggregate_if_ready() {
  local dataset="$1"
  local cell="$2"
  local specs=()
  local have_anchor=0
  IFS=',' read -r -a arms <<< "$ARMS_CSV"
  for arm in "${arms[@]}"; do
    if [[ "$arm" == "old_vq2" && "$cell" != "math500" ]]; then
      continue
    fi
    [[ -f "$OUT_ROOT/$arm/$cell/metrics.json" ]] || return 0
    [[ "$arm" == "bf16" ]] && have_anchor=1
    specs+=("$arm=$OUT_ROOT/$arm/$cell")
  done
  [[ "${#specs[@]}" -gt 0 ]] || return 0
  [[ "$have_anchor" -eq 1 ]] || return 0
  .venv/bin/python pipelines/eval/aggregate_acck.py \
    --cells "${specs[@]}" \
    --anchor bf16 \
    --out "$OUT_ROOT/${dataset}_acck_summary.json"
}

trap 'stop_server' EXIT

IFS=',' read -r -a arms <<< "$ARMS_CSV"
IFS=',' read -r -a tasks <<< "$TASKS_CSV"
for arm in "${arms[@]}"; do
  boot_arm "$arm"
  for task in "${tasks[@]}"; do
    if [[ "$arm" == "old_vq2" && "$task" != "math500" ]]; then
      echo "[$(date -Is)] skip old_vq2/$task (old VQ baseline is math500-only)"
      continue
    fi
    case "$task" in
      math500)
        run_cell "$arm" math500 "$MATH_ROWS" "$MATH_THREADS"
        ;;
      aime25)
        run_cell "$arm" aime25 "$AIME_ROWS" "$AIME_THREADS"
        ;;
      gpqa)
        run_cell "$arm" gpqa "$GPQA_ROWS" "$GPQA_THREADS"
        ;;
      *)
        echo "unknown task: $task" >&2
        exit 2
        ;;
    esac
  done
  stop_server
done

for task in "${tasks[@]}"; do
  aggregate_if_ready "$task" "$task"
done

echo "[$(date -Is)] GPT-OSS math run complete -> $OUT_ROOT"
