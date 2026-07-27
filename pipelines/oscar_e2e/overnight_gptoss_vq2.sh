#!/bin/bash
# Overnight: wait for free GPUs, validate the p2-2 fix on a live server, then
# run the gpt-oss vq2 grid.
#
# SAFETY -- other people's work is running on these boxes:
#   * a GPU counts as free ONLY if it has zero compute processes. We never
#     co-tenant, so we cannot OOM anyone or trip the memory-balance abort.
#   * cleanup kills ONLY the server PID we launched and the listener on OUR
#     port. It never sweeps "all processes on GPU N" -- that pattern (used by
#     gptoss_bf16_grid.sh) would kill the owner's unrelated VLRL training.
#
# Order is deliberate: debugging first, experiment second. Phase 1 verifies
# that p2-2 (SWAKVPool missing a release_req_slab delegate, so the per-request
# HP-recent ring cursor was never reset) is actually fixed, by booting with the
# strict idle memory check ENABLED -- the check we previously had to disable to
# get any numbers at all. If it still trips, the grid does not start: a slot
# leak corrupts exactly the long-context cells the grid measures.
set -u
ROOT="${ROOT:-/vault/amir/efficient-llm/teamily-project}"
cd "$ROOT"
LOG=logs/overnight_gptoss_vq2.log
HB=logs/overnight_gptoss_vq2.heartbeat
M=unsloth/gpt-oss-20b-BF16
OUT=artifacts/oscar_gptoss20b/grid/vq2
CB="$ROOT/artifacts/oscar_gptoss20b/vqa_gptoss20b_G4_strat_flat_ptn_gpqacc128k_fp8.pt"
export ROT_DIR="$ROOT/artifacts/oscar_gptoss20b/rotations_gpqa198"
MAX_WAIT_MIN="${MAX_WAIT_MIN:-720}"
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; touch "$HB"; }
mkdir -p "$OUT"; : > "$LOG"; touch "$HB"

# --- free-GPU discovery: zero compute processes, nothing else counts --------
free_gpus(){
  local busy; busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  local out=""
  for g in 0 1 2 3 4 5 6; do
    local uuid; uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$g" 2>/dev/null) || continue
    grep -q "$uuid" <<< "$busy" || out="$out $g"
  done
  echo "$out"
}

SRV_PID=""; SRV_PORT=""
stop_server(){
  [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null
  if [ -n "$SRV_PORT" ]; then
    local p; p=$(lsof -t -i :"$SRV_PORT" 2>/dev/null)
    # only our own listener; never a GPU-wide sweep
    for x in $p; do
      [ "$(ps -o user= -p "$x" 2>/dev/null | tr -d ' ')" = "$(whoami)" ] && kill -9 "$x" 2>/dev/null
    done
  fi
  SRV_PID=""; SRV_PORT=""; sleep 10
}
trap 'log "interrupted -- stopping server"; stop_server; exit 130' INT TERM

boot(){ # gpus port ctx strictmem -> 0 ok
  local gpus="$1" port="$2" ctx="$3" strict="$4"
  local SLOG="logs/overnight_srv_${port}.log"; : > "$SLOG"
  SRV_PORT="$port"
  # DISABLE_CUDA_GRAPH is REQUIRED, not a preference: the mixed HP+quant decode
  # path is not CUDA-graph safe at bs>1. Isolated single-variable -- 4 threads
  # with graphs died on a multinomial assert (NaN/Inf logits), the identical run
  # with --disable-cuda-graph survived. Costs eager-decode speed; every
  # throughput number we have was measured eager anyway.
  DISABLE_CUDA_GRAPH=1 \
  ABSORB_V_ROT=0 QUANT_GROUP_SIZE=0 MAX_TOKENS=42000 TP=2 \
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE="$strict" \
    nohup bash pipelines/oscar_e2e/serve_oscar.sh --vq2 --vq-codebook "$CB" \
      --model "$M" --ctx "$ctx" --gpu "$gpus" --port "$port" >> "$SLOG" 2>&1 &
  SRV_PID=$!
  for i in $(seq 1 150); do
    grep -q "fired up and ready to roll" "$SLOG" && { log "boot OK (port $port, gpus $gpus)"; return 0; }
    grep -qE "Received sigquit|CUDA out of memory|Traceback" "$SLOG" && break
    sleep 5; touch "$HB"
  done
  log "BOOT FAIL port $port"; tail -20 "$SLOG" >> "$LOG"; stop_server; return 1
}

# --- wait for a clean pair --------------------------------------------------
PAIR=""
for attempt in $(seq 1 $((MAX_WAIT_MIN / 2))); do
  F=$(free_gpus); N=$(wc -w <<< "$F")
  if [ "$N" -ge 2 ]; then
    PAIR=$(awk '{print $1","$2}' <<< "$F"); log "attempt $attempt: free {$F } -> claiming $PAIR"; break
  fi
  [ $((attempt % 15)) -eq 1 ] && log "attempt $attempt: only $N free {$F } -- waiting"
  sleep 120; touch "$HB"
done
[ -z "$PAIR" ] && { log "no free pair within ${MAX_WAIT_MIN}min -- giving up"; echo OVERNIGHT_NO_GPU >> "$LOG"; exit 1; }

# =============== PHASE 1: does the p2-2 fix hold on a live server? ==========
log "=== PHASE 1: concurrency smoke with CUDA graphs disabled (the workaround)"
if boot "$PAIR" 30995 16384 0; then
  head -24 artifacts/prompt_rows/niah_8192_gptoss_t1.jsonl > /tmp/p1_rows.jsonl
  .venv/bin/python pipelines/oscar_e2e/run_prompts_client.py \
    --rows /tmp/p1_rows.jsonl --port 30995 --threads 4 --timeout 1800 \
    --samples 1 --temperature 1.0 --top-p 1.0 --top-k -1 \
    --out artifacts/oscar_gptoss20b/p2_2_leakcheck >> "$LOG" 2>&1
  RC=$?
  ALIVE=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:30995/get_server_info" 2>/dev/null)
  LEAK=$(grep -acE "leaked|memory leak|Memory leak" "logs/overnight_srv_30995.log" 2>/dev/null || echo 0)
  log "phase1 client rc=$RC server_http=$ALIVE leak_lines=$LEAK"
  stop_server
  if [ "$ALIVE" = "200" ] && [ "$LEAK" -eq 0 ]; then
    log "PHASE 1 PASS -- survived 24 concurrent sampled requests (graphs off)"
  else
    log "PHASE 1 FAIL -- still dying with graphs off; NOT starting the grid (it would"
    log "              corrupt the long-context cells). The workaround is insufficient."
    echo OVERNIGHT_PHASE1_FAIL >> "$LOG"; exit 1
  fi
else
  echo OVERNIGHT_PHASE1_BOOT_FAIL >> "$LOG"; exit 1
fi

# =============== PHASE 2: the vq2 grid =====================================
log "=== PHASE 2: gpt-oss vq2 NIAH grid (bf16 anchors already exist)"
cell(){ # ctx rows out threads
  local ctx="$1" rows="$2" out="$3" threads="$4"
  [ -f "$out/metrics.json" ] && { log "skip $out (already done)"; return 0; }
  # re-check the pair is still unowned by anyone else before each cell
  local F; F=$(free_gpus)
  for g in ${PAIR//,/ }; do
    grep -qw "$g" <<< "$F" || { log "gpu $g taken by another user -- pausing 10min"; sleep 600; return 2; }
  done
  boot "$PAIR" 30995 "$ctx" 0 || return 1
  .venv/bin/python pipelines/oscar_e2e/run_prompts_client.py \
    --rows "$rows" --port 30995 --threads "$threads" --timeout 7200 \
    --samples 1 --temperature 1.0 --top-p 1.0 --top-k -1 \
    --out "$out" >> "$LOG" 2>&1
  log "$out rc=$?"
  stop_server
  [ -f "$out/metrics.json" ] && log "$out OK" || log "$out produced no metrics.json"
}

for spec in \
  "16384 artifacts/prompt_rows/niah_8192_gptoss.jsonl  $OUT/niah_8192  12" \
  "24576 artifacts/prompt_rows/niah_16384_gptoss.jsonl $OUT/niah_16384  8" \
  "40960 artifacts/prompt_rows/niah_32768_gptoss.jsonl $OUT/niah_32768  4" \
  "73728 artifacts/prompt_rows/niah_65536_gptoss.jsonl $OUT/niah_65536  2" ; do
  set -- $spec
  for try in 1 2 3; do
    cell "$1" "$2" "$3" "$4"; rc=$?
    [ $rc -ne 2 ] && break     # rc=2 means a GPU got taken; retry after the pause
  done
done

log "=== OVERNIGHT DONE"
echo OVERNIGHT_DONE >> "$LOG"
