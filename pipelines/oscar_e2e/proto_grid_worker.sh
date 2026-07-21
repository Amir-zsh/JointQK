#!/bin/bash
# One GPU worker for the OSCAR-EXACT-protocol reasoning grid.
# Serves one arm, runs its tasks at T=0.6/top-p0.95/top-k20/5-seeds (max-gen 32768
# baked in rows), scores each cell vs the OSCAR paper. Resume-safe per cell.
# Usage: proto_grid_worker.sh <gpu> <arm> <task1,task2,...>
#   arm  = bf16 | oscar_int2 | vq2 | quarot_int2 | naive_int2
#   task = gpqa | aime25 | math500
set -u
GPU="$1"; ARM="$2"; IFS=',' read -ra TASKS <<< "$3"; PORT=$((30960+GPU))
ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
AR=/vault/amir/efficient-llm/teamily-project
JQ=/vault/samuel/efficient-llm/JointQK
PY=$ROOT/.venv/bin/python
ROT8B="$ROOT/artifacts/oscar_e2e/rotzoo/Qwen3-8B/seq20000_prompt83_group128"
CB8B="$JQ/entropy_coding/vqa_G4_strat_flat_ptn_gpqacc64k_fp8.pt"
declare -A ROWS=(
  [gpqa]="$AR/artifacts/prompt_rows/gpqa_diamond_think_qwen.jsonl"
  [aime25]="$AR/artifacts/prompt_rows/aime25_think_qwen.jsonl"
  [math500]="$ROOT/artifacts/prompt_rows_proto/math500_think32k_qwen.jsonl"
)
declare -A TGT=(  # OSCAR paper Table 2, Qwen3-8B (mean +/- std)
  [bf16_gpqa]="56.67+/-2.30" [bf16_aime25]="70.00+/-3.33" [bf16_math500]="92.59+/-0.62"
  [oscar_int2_gpqa]="55.05+/-1.47" [oscar_int2_aime25]="66.67+/-3.33" [oscar_int2_math500]="92.22+/-0.83"
  [quarot_int2_gpqa]="14.98+/-0.63" [quarot_int2_aime25]="2.22+/-1.57" [quarot_int2_math500]="23.13+/-1.88"
  [naive_int2_gpqa]="0.00" [naive_int2_aime25]="0.00" [naive_int2_math500]="0.00"
)
LOG=$ROOT/logs/proto_${ARM}_gpu${GPU}.log
HB=$ROOT/logs/proto_${ARM}_gpu${GPU}.heartbeat
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; touch "$HB"; }
cd "$ROOT"

free_gpu(){
  pkill -9 -f "port $PORT" 2>/dev/null || true; sleep 4
  local U; U=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $GPU | tr -d ' ')
  for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null | grep "$U" | cut -d, -f1); do
    ps -o cmd= -p "$p" 2>/dev/null | grep -q sglang && kill -9 "$p" 2>/dev/null || true
  done
  sleep 6
}

serve(){
  local env=""; local args=(--gpu $GPU --port $PORT)
  case "$ARM" in
    bf16)        args+=(--bf16 --model Qwen/Qwen3-8B) ;;
    oscar_int2)  env="ROT_DIR=$ROT8B"; args+=(--model Qwen/Qwen3-8B) ;;
    vq2)         env="ROT_DIR=$ROT8B SGLANG_VQ_CODEBOOK_PATH=$CB8B"; args+=(--vq2 --model Qwen/Qwen3-8B --vq-codebook "$CB8B") ;;
    quarot_int2) env="HADAMARD_ORDER=128"; args+=(--int2plain) ;;
    naive_int2)  env="SGLANG_INT2_NO_HADAMARD=1 HADAMARD_ORDER=16"; args+=(--int2plain) ;;
    *) log "unknown arm $ARM"; exit 1 ;;
  esac
  free_gpu
  env $env MEM_FRAC=0.85 MAX_REQS="${MAX_REQS:-8}" \
    setsid bash pipelines/oscar_e2e/serve_oscar.sh "${args[@]}" \
    > "$ROOT/logs/serve_proto_${ARM}_gpu${GPU}.log" 2>&1 < /dev/null &
  for i in $(seq 1 180); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/get_model_info 2>/dev/null)" = "200" ] && return 0
    touch "$HB"; sleep 10
  done
  return 1
}

: > "$LOG"
log "arm=$ARM gpu=$GPU port=$PORT tasks=${TASKS[*]}"
if ! serve; then log "SERVER FAILED"; tail -20 "$ROOT/logs/serve_proto_${ARM}_gpu${GPU}.log" >> "$LOG"; free_gpu; exit 1; fi
log "server up"

for t in "${TASKS[@]}"; do
  OUT=$ROOT/artifacts/oscar_e2e/proto/${ARM}/${t}
  mkdir -p "$OUT"
  log "running $t (rows=$(basename ${ROWS[$t]}))"
  PYTHONPATH="$AR" $PY pipelines/oscar_e2e/run_prompts_client.py \
    --rows "${ROWS[$t]}" --samples 5 --temperature 0.6 --top-p 0.95 --top-k 20 \
    --threads 8 --timeout 3600 --port $PORT --out "$OUT" \
    >> "$ROOT/logs/client_proto_${ARM}_${t}.log" 2>&1
  $PY - "$OUT/metrics.json" "${ARM}_${t}" "${TGT[${ARM}_${t}]:-?}" <<'PY' >> "$LOG" 2>&1
import json,sys,statistics as st
d=json.load(open(sys.argv[1])); pk=d.get("per_k",[])
a=[x["accuracy"]*100 for x in pk]
mu=st.mean(a); sd=st.stdev(a) if len(a)>1 else 0.0
print(f"RESULT {sys.argv[2]}: {mu:.2f}+/-{sd:.2f} n={len(a)} seeds={[round(x,1) for x in a]}  | OSCAR={sys.argv[3]}")
PY
  log "$t done -> $(grep RESULT $LOG | tail -1)"
done
free_gpu
log "ARM $ARM DONE"; echo PROTO_${ARM}_GPU${GPU}_DONE >> "$LOG"
