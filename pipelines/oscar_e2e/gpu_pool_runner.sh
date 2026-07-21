#!/bin/bash
# GPU-pool worker for sharded eval jobs. N workers (one per GPU) pull from a
# shared flock-protected queue; a worker keeps its arm's server alive across
# consecutive same-arm jobs (arm-affinity claiming) and only reboots on an
# arm change — so sharding doesn't multiply server-boot cost.
#
# Queue file (TSV): status \t arm \t out_dir \t rows_file \t extra_args...
#   status: TODO | RUN:<gpu> | DONE | FAIL:<rc>
# Arms: bf16 | int2 | vq2 | quarot | naive      (Llama-3.1-8B, ROT_DIR set)
#       qwen-vq2 | qwen-vqv                     (Qwen3-8B, gpqacc64k codebook)
#
#   bash gpu_pool_runner.sh <queue.tsv> <gpu>     # one worker
set -u
ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
QUEUE="${1:?queue.tsv}"; GPU="${2:?gpu}"
PORT=$((30850 + GPU))
LOCK="$QUEUE.lock"
WLOG="logs/pool_gpu${GPU}.log"
mkdir -p logs
exec >>"$WLOG" 2>&1
log(){ echo "[$(date '+%F %T')] gpu$GPU $*"; }

CUR_ARM=""
SERVER_PID=""

teardown(){
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
  local P=$(lsof -t -i :$PORT 2>/dev/null); [ -n "$P" ] && kill $P 2>/dev/null
  sleep 5
  # orphaned scheduler on this GPU (ours only)
  local UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $GPU)
  for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | awk -F', ' -v u="$UUID" '$2==u{print $1}'); do
    [ "$(ps -o user= -p $p 2>/dev/null | tr -d ' ')" = "$(whoami)" ] && kill -9 $p 2>/dev/null
  done
  CUR_ARM=""; SERVER_PID=""
}
trap teardown EXIT

boot_arm(){ # arm
  local arm="$1"
  teardown
  local MODEL="meta-llama/Llama-3.1-8B-Instruct"
  local SERVE=(--gpu "$GPU" --port "$PORT")
  export ROT_DIR="$ROOT/artifacts/oscar_llama31_8b/rotations"
  unset SGLANG_VQ_V_CODEBOOK_PATH SGLANG_INT2_NO_HADAMARD HADAMARD_ORDER SGLANG_LLOYD_MAX 2>/dev/null
  case "$arm" in
    bf16)   SERVE+=(--bf16 --model "$MODEL") ;;
    int2)   SERVE+=(--model "$MODEL") ;;
    vq2)    SERVE+=(--vq2 --model "$MODEL" --vq-codebook artifacts/oscar_llama31_8b/vqa_llama31_8b_G4_strat_flat_ptn_gpqacc64k_fp8.pt) ;;
    vq2mix) SERVE+=(--vq2 --model "$MODEL" --vq-codebook artifacts/oscar_llama31_8b/vqa_llama31_8b_G4_strat_flat_ptn_mixed64k_fp8.pt) ;;
    quarot) SERVE+=(--int2plain --model "$MODEL"); export HADAMARD_ORDER=128 ;;
    naive)  SERVE+=(--int2plain --model "$MODEL"); export HADAMARD_ORDER=16 SGLANG_INT2_NO_HADAMARD=1 ;;
    turbo)  SERVE+=(--int2plain --model "$MODEL"); export HADAMARD_ORDER=128 SGLANG_LLOYD_MAX=1 ;;
    qwen-vq2)
      export ROT_DIR="$ROOT/artifacts/oscar_e2e/rotzoo/Qwen3-8B/seq20000_prompt83_group128"
      SERVE+=(--vq2 --vq-codebook third_party/samuel_vq/codebooks/vqa_G4_strat_flat_ptn_gpqacc64k_fp8.pt) ;;
    qwen-vqv)
      export ROT_DIR="$ROOT/artifacts/oscar_e2e/rotzoo/Qwen3-8B/seq20000_prompt83_group128"
      export SGLANG_VQ_V_CODEBOOK_PATH="$ROOT/third_party/samuel_vq/codebooks/vqv_G4_strided_gpqa_engine.pt"
      SERVE+=(--vq2 --vq-codebook third_party/samuel_vq/codebooks/vqa_G4_strat_flat_ptn_gpqacc64k_fp8.pt) ;;
    *) log "unknown arm $arm"; return 1 ;;
  esac
  # Full-capacity concurrency: short-context cells leave the 140K-token pool
  # mostly idle at the default 8 requests; 24 concurrent requests + matching
  # CUDA-graph batch keep decode batches fat. Long-context cells self-limit
  # via per-job client threads (the scheduler admits what fits the pool).
  export MAX_REQS="${POOL_MAX_REQS:-24}" CUDA_GRAPH_BS="${POOL_MAX_REQS:-24}"
  local BLOG="logs/pool_gpu${GPU}_server.log"
  : > "$BLOG"   # truncate SYNCHRONOUSLY: nohup's redirect races the grep below,
                # which otherwise matches the previous boot's stale "fired up"
  nohup bash pipelines/oscar_e2e/serve_oscar.sh "${SERVE[@]}" >> "$BLOG" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 120); do
    grep -q "The server is fired up and ready to roll" "$BLOG" && { CUR_ARM="$arm"; log "server up arm=$arm"; return 0; }
    grep -qE "Received sigquit|CUDA out of memory|Not enough memory" "$BLOG" && break
    sleep 5
  done
  log "BOOT FAILED arm=$arm"; tail -5 "$BLOG"; teardown; return 1
}

claim_job(){ # prefer CUR_ARM; echo "lineno<TAB>line" or nothing
  ( flock -x 9
    local ln line
    if [ -n "$CUR_ARM" ]; then
      ln=$(grep -n -P "^TODO\t$CUR_ARM\t" "$QUEUE" | head -1 | cut -d: -f1)
    fi
    [ -z "${ln:-}" ] && ln=$(grep -n -P "^TODO\t" "$QUEUE" | head -1 | cut -d: -f1)
    [ -z "${ln:-}" ] && return 1
    line=$(sed -n "${ln}p" "$QUEUE")
    sed -i "${ln}s/^TODO/RUN:$GPU/" "$QUEUE"
    echo "${ln}"$'\t'"$line"
  ) 9>"$LOCK"
}

mark_job(){ # lineno newstatus
  ( flock -x 9; sed -i "${1}s/^RUN:$GPU/${2}/" "$QUEUE" ) 9>"$LOCK"
}

log "worker start (queue=$QUEUE)"
# Wait for the GPU to free (a monolithic-era server/client may still own it;
# their cells complete, then we take over).
for i in $(seq 1 720); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
  [ "$USED" -lt 2000 ] && break
  [ $((i % 24)) -eq 0 ] && log "waiting for gpu (used=${USED}MiB)"
  sleep 10
done
while true; do
  CLAIM=$(claim_job) || { log "queue drained"; break; }
  LN=$(cut -f1 <<<"$CLAIM")
  IFS=$'\t' read -r _st ARM OUT ROWS EXTRA <<<"$(cut -f2- <<<"$CLAIM")"
  log "job L$LN arm=$ARM out=$OUT"
  if [ "$ARM" != "$CUR_ARM" ]; then
    boot_arm "$ARM" || { mark_job "$LN" "FAIL:boot"; continue; }
  fi
  # shellcheck disable=SC2086
  .venv/bin/python pipelines/oscar_e2e/run_prompts_client.py \
    --rows "$ROWS" --port "$PORT" --timeout 3600 $EXTRA --out "$OUT"
  RC=$?
  if [ $RC -eq 0 ]; then mark_job "$LN" "DONE"; log "job L$LN DONE"
  else mark_job "$LN" "FAIL:$RC"; log "job L$LN FAIL rc=$RC"
    # server may have died with the job; force a fresh boot next claim
    teardown
  fi
done
teardown
log "worker exit"
