#!/bin/bash
# OSCAR baseline reproduction on the PLAIN int2 pool (no mixed-KV band = no MP):
#   Naive-INT2  = HADAMARD_ORDER=1  (identity, no rotation)
#   QuaRot-INT2 = HADAMARD_ORDER=128 (full Hadamard over head_dim)
# Per-token int2, no band, no OSCAR learned rotation. Robust config + hard teardown.
# Usage: run_baseline_plain.sh <name> <gpu> <hadamard_order> <evals>
set -u
NAME="$1"; GPU="$2"; ORDER="$3"; EVALS="$4"; PORT=$((30980+GPU))
ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
JQ=/vault/samuel/efficient-llm/JointQK
AR=/vault/amir/efficient-llm/teamily-project
PY=$ROOT/.venv/bin/python
NIAH=$AR/artifacts/prompt_rows/niah_32768_qwen.jsonl
HE=$ROOT/artifacts/prompt_rows_code/humaneval_qwen.jsonl
declare -A ACC=(
  [gpqa]="$AR/artifacts/prompt_rows/gpqa_diamond_think_qwen.jsonl"
  [aime25]="$AR/artifacts/prompt_rows/aime25_think_qwen.jsonl"
  [math500]="$AR/artifacts/prompt_rows/math500_think16k_qwen.jsonl"
)
LOG=$ROOT/logs/base_plain_${NAME}.log
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
cd "$ROOT"

free_gpu(){
  pkill -9 -f "port $PORT" 2>/dev/null || true; sleep 4
  local U; U=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $GPU | tr -d ' ')
  for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null | grep "$U" | cut -d, -f1); do
    ps -o cmd= -p "$p" 2>/dev/null | grep -q sglang && kill -9 "$p" 2>/dev/null || true
  done
  sleep 6
}

free_gpu
# ORDER=0 -> Naive (no rotation, SGLANG_INT2_NO_HADAMARD); ORDER>=2 -> QuaRot (Hadamard of that order)
if [ "$ORDER" = "0" ]; then HADENV="SGLANG_INT2_NO_HADAMARD=1 HADAMARD_ORDER=16"; else HADENV="HADAMARD_ORDER=$ORDER"; fi
log "serving $NAME (plain int2, $HADENV, no MP) on GPU$GPU port$PORT"
env $HADENV MAX_TOKENS=350000 MAX_REQS=4 MEM_FRAC=0.85 \
  setsid bash pipelines/oscar_e2e/serve_oscar.sh --int2plain --gpu $GPU --port $PORT \
  > "$ROOT/logs/serve_plain_${NAME}.log" 2>&1 < /dev/null &
for i in $(seq 1 150); do [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/get_model_info 2>/dev/null)" = "200" ] && break; sleep 10; done
if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/get_model_info 2>/dev/null)" != "200" ]; then
  log "$NAME server never came up"; tail -25 "$ROOT/logs/serve_plain_${NAME}.log" >> "$LOG"; free_gpu; exit 1; fi
log "$NAME server up; evals=[$EVALS]"

for e in $EVALS; do
  out=$ROOT/artifacts/oscar_e2e/base_${NAME}
  case "$e" in
    niah)    out=${out}_niah32k;   args=(--rows "$NIAH" --threads 2 --timeout 3600) ;;
    he)      out=${out}_humaneval; args=(--rows "$HE" --temperature 0.6 --top-p 0.95 --top-k 20 --samples 1 --threads 4 --timeout 3600) ;;
    gpqa|aime25|math500) out=${out}_${e}; args=(--rows "${ACC[$e]}" --samples 4 --temperature 1.0 --top-p 0.95 --top-k 40 --threads 4 --timeout 3600) ;;
    *) log "skip $e"; continue ;;
  esac
  mkdir -p "$out"
  PYTHONPATH="$AR" $PY pipelines/oscar_e2e/run_prompts_client.py "${args[@]}" --port $PORT --out "$out" \
    >> "$ROOT/logs/client_plain_${NAME}_${e}.log" 2>&1
  log "$NAME $e -> $(head -c 120 $out/metrics.json 2>/dev/null | tr -d '\n')"
done
free_gpu
log "$NAME ALL DONE"; echo BASE_PLAIN_${NAME}_DONE >> "$LOG"
