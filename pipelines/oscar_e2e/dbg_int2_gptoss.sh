#!/bin/bash
# Isolated int2 gpt-oss capture: boot one server (persistent log, no
# truncation), reproduce the S1 sequence — 512-token gen (triggers quant
# flush), idle gap, then one 8K NIAH request — checking liveness after each
# so the crash traceback is preserved in the srv log.
set -u
ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"; cd "$ROOT"
GPUS="${1:-2,3}"; PORT="${2:-30931}"
SRV="$ROOT/logs/dbg_int2_srv.log"; LOG="$ROOT/logs/dbg_int2.log"
: > "$SRV"; : > "$LOG"
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
export ROT_DIR="$ROOT/artifacts/oscar_gptoss20b/rotations_gpqa198"

log "boot int2 on $GPUS"
env ABSORB_V_ROT=0 QUANT_GROUP_SIZE=0 MAX_TOKENS=42000 TP=2 DISABLE_CUDA_GRAPH=1 \
  nohup bash pipelines/oscar_e2e/serve_oscar.sh --model unsloth/gpt-oss-20b-BF16 \
  --ctx 16384 --gpu "$GPUS" --port "$PORT" >> "$SRV" 2>&1 &
SPID=$!
for i in $(seq 1 240); do
  grep -q "fired up and ready to roll" "$SRV" && break
  grep -qE "Traceback|CUDA out of memory|Received sigquit" "$SRV" && { log "boot error"; break; }
  sleep 5
done
grep -q "fired up" "$SRV" || { log "BOOT FAIL"; tail -25 "$SRV" >>"$LOG"; exit 1; }
log "boot OK"

alive(){ curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/get_server_info" 2>/dev/null; }

log "step1: 512-token gen (flush trigger)"
.venv/bin/python - "$PORT" >>"$LOG" 2>&1 <<'PY'
import json,sys,urllib.request
p={"text":"Write a long essay about the ocean.","sampling_params":{"temperature":1.0,"max_new_tokens":512,"ignore_eos":True}}
r=urllib.request.Request(f"http://127.0.0.1:{sys.argv[1]}/generate",json.dumps(p).encode(),{"Content-Type":"application/json"})
print("gen512 ok, ntok:",json.loads(urllib.request.urlopen(r,timeout=600).read()).get("meta_info",{}).get("completion_tokens"))
PY
log "after gen512: server http=$(alive)"
sleep 8
log "after 8s idle: server http=$(alive)"

log "step2: one 8K NIAH request"
.venv/bin/python - "$PORT" >>"$LOG" 2>&1 <<'PY'
import json,sys,urllib.request
row=json.loads(open("artifacts/prompt_rows/niah_8192_gptoss_t1.jsonl").readline())
txt=row.get("text") or row.get("prompt") or row.get("input")
p={"text":txt,"sampling_params":{"temperature":1.0,"top_p":1.0,"top_k":-1,"max_new_tokens":128}}
r=urllib.request.Request(f"http://127.0.0.1:{sys.argv[1]}/generate",json.dumps(p).encode(),{"Content-Type":"application/json"})
print("niah1 ok:",json.loads(urllib.request.urlopen(r,timeout=600).read())["text"][:80])
PY
log "after niah1: server http=$(alive)"

log "=== DBG_DONE (srv log preserved at $SRV)"
kill $SPID 2>/dev/null; P=$(lsof -t -i :$PORT 2>/dev/null); [ -n "$P" ] && kill $P
for g in ${GPUS//,/ }; do UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $g); for pp in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | awk -F', ' -v u="$UUID" '$2==u{print $1}'); do [ "$(ps -o user= -p $pp 2>/dev/null|tr -d ' ')" = amir ] && kill -9 $pp 2>/dev/null; done; done
echo DBG_DONE >> "$LOG"
