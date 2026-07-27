#!/bin/bash
# Wait for a genuinely empty two-GPU pair, then run the resume-safe GPT-OSS
# math grid. This script never kills external processes; run_gptoss_math.sh
# also refuses to start if its port is already occupied.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PORT="${PORT:-30920}"
INTERVAL="${INTERVAL:-60}"
MEM_FREE_MB="${MEM_FREE_MB:-100}"
ARMS="${ARMS:-bf16,int2,vq2,old_vq2}"
OUT_ROOT="${OUT_ROOT:-artifacts/oscar_gptoss20b/math_grid}"

echo "[$(date -Is)] waiting for two empty GPUs on $(hostname 2>/dev/null || echo unknown-host)"
while true; do
  mapfile -t free_gpus < <(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
      awk -F, -v lim="$MEM_FREE_MB" '{
        gsub(/ /, "", $1);
        gsub(/ /, "", $2);
        if ($2 <= lim) print $1;
      }'
  )

  if [[ "${#free_gpus[@]}" -ge 2 ]]; then
    pair="${free_gpus[0]},${free_gpus[1]}"
    if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
      echo "[$(date -Is)] found free pair $pair but port $PORT is busy; continuing"
    else
      echo "[$(date -Is)] starting GPT-OSS math grid on GPU_PAIR=$pair"
      env GPU_PAIR="$pair" PORT="$PORT" ARMS="$ARMS" OUT_ROOT="$OUT_ROOT" \
        bash pipelines/oscar_e2e/run_gptoss_math.sh
      rc=$?
      echo "[$(date -Is)] gptoss math grid exited rc=$rc"
      exit "$rc"
    fi
  else
    echo "[$(date -Is)] free_gpus=${free_gpus[*]:-none}"
  fi

  sleep "$INTERVAL"
done
