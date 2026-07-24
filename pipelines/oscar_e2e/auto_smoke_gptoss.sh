#!/bin/bash
# Autonomous retry-launcher for the gpt-oss quant smoke on lambda7.
# Waits for >=3 idle GPUs (0 compute procs), grabs a pair leaving >=1 free,
# runs the smoke; retries on churn-induced boot failure until the int2 AND
# vq2 NIAH-8K slices land (or the attempt cap is hit).
set -u
ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"; cd "$ROOT"
LOG="$ROOT/logs/auto_smoke_gptoss.log"; HB="$ROOT/logs/auto_smoke_gptoss.heartbeat"
: > "$LOG"; log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; touch "$HB"; }
INT2_OUT="artifacts/oscar_gptoss20b/grid/int2/niah_8192_smoke/metrics.json"
VQ2_OUT="artifacts/oscar_gptoss20b/grid/vq2/niah_8192_smoke/metrics.json"
MAXATT="${MAXATT:-30}"

idle_gpus(){ local busy; busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
  for i in 0 1 2 3 4 5 6; do local u; u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $i)
    echo "$busy" | grep -q "$u" || printf "%s " "$i"; done; }

log "=== auto-smoke start (need int2+vq2 NIAH slices)"
for att in $(seq 1 $MAXATT); do
  [ -f "$INT2_OUT" ] && [ -f "$VQ2_OUT" ] && { log "already complete"; break; }
  idle=$(idle_gpus); n=$(echo $idle | wc -w)
  if [ "$n" -lt 3 ]; then log "attempt $att: only $n idle {$idle} (<3, leave-1) — wait"; sleep 120; continue; fi
  pair=$(echo $idle | tr ' ' '\n' | head -2 | paste -sd,)
  log "attempt $att: idle={$idle} -> using pair $pair (leaving $((n-2)) free)"
  # fresh outputs so a partial prior attempt doesn't count as success
  rm -rf artifacts/oscar_gptoss20b/grid/int2/niah_8192_smoke artifacts/oscar_gptoss20b/grid/vq2/niah_8192_smoke
  DISABLE_CUDA_GRAPH=1 bash pipelines/oscar_e2e/smoke_gptoss_quant.sh "$pair" 30931 >> "$LOG" 2>&1
  if [ -f "$INT2_OUT" ] && [ -f "$VQ2_OUT" ]; then log "SUCCESS on attempt $att (pair $pair)"; break; fi
  log "attempt $att incomplete (boot race / churn) — retry after 60s"
  sleep 60
done
log "=== AUTO_SMOKE_DONE int2=$([ -f "$INT2_OUT" ] && echo Y || echo N) vq2=$([ -f "$VQ2_OUT" ] && echo Y || echo N)"
echo AUTO_SMOKE_DONE >> "$LOG"
