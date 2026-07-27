#!/bin/bash
# Launch remaining GPT-OSS follow-up cells when their current GPU lanes free.
# This script is intentionally conservative: it never kills processes and only
# starts a cell if its output is missing, the selected port is unused, and the
# selected GPUs have negligible memory use.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

LOG="${LOG:-logs/gptoss_followups_watch.log}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/tmp/codex_lambda_known_hosts)
REMOTE="${REMOTE:-lambda-server6}"
MAX_MEM_MB="${MAX_MEM_MB:-100}"
SLEEP_SEC="${SLEEP_SEC:-60}"

mkdir -p logs

log() {
  echo "[$(date -Is)] $*" | tee -a "$LOG"
}

pid_alive() {
  local pid_file="$1"
  [[ -f "$pid_file" ]] || return 1
  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

local_pair_free() {
  local g0="$1" g1="$2"
  local used
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g0","$g1" 2>/dev/null | awk '{gsub(/ /,""); print}')"
  [[ -n "$used" ]] || return 1
  while read -r m; do
    [[ "$m" =~ ^[0-9]+$ ]] || return 1
    (( m <= MAX_MEM_MB )) || return 1
  done <<< "$used"
}

remote_pair_free() {
  local g0="$1" g1="$2"
  ssh "${SSH_OPTS[@]}" "$REMOTE" \
    "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g0,$g1" \
    2>/dev/null | awk -v max="$MAX_MEM_MB" '
      BEGIN { ok=1; n=0 }
      { gsub(/ /,""); if ($1 !~ /^[0-9]+$/ || $1 > max) ok=0; n++ }
      END { exit !(ok && n == 2) }'
}

port_free_local() {
  ! lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

port_free_remote() {
  ssh "${SSH_OPTS[@]}" "$REMOTE" \
    "if lsof -nP -iTCP:$1 -sTCP:LISTEN >/dev/null 2>&1; then exit 1; fi" \
    >/dev/null 2>&1
}

launch_local_vq2_gpqa() {
  local pid_file="logs/gptoss_math_driver_server7_vq2_gpqa.pid"
  [[ -f artifacts/oscar_gptoss20b/math_grid/vq2/aime25/metrics.json ]] || return 0
  [[ ! -f artifacts/oscar_gptoss20b/math_grid/vq2/gpqa/metrics.json ]] || return 0
  pid_alive "$pid_file" && return 0
  port_free_local 30923 || return 0
  local_pair_free 4 5 || return 0
  log "launch local vq2/gpqa on lambda-server7 GPUs 4,5 port 30923"
  setsid nohup bash -lc \
    'cd /vault/amir/efficient-llm/teamily-project && env GPU_PAIR=4,5 PORT=30923 ARMS=vq2 TASKS=gpqa OUT_ROOT=artifacts/oscar_gptoss20b/math_grid bash pipelines/oscar_e2e/run_gptoss_math.sh' \
    > logs/gptoss_math_driver_server7_vq2_gpqa.log 2>&1 < /dev/null &
  echo $! > "$pid_file"
}

launch_remote_int2_gpqa() {
  local rcmd='cd /vault/amir/efficient-llm/teamily-project &&
    test -f artifacts/oscar_gptoss20b/math_grid/int2/aime25/metrics.json &&
    test ! -f artifacts/oscar_gptoss20b/math_grid/int2/gpqa/metrics.json &&
    if [ -f logs/gptoss_math_driver_server6_int2_gpqa.pid ]; then
      pid=$(cat logs/gptoss_math_driver_server6_int2_gpqa.pid 2>/dev/null || true)
      if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then exit 0; fi
    fi &&
    if lsof -nP -iTCP:30924 -sTCP:LISTEN >/dev/null 2>&1; then exit 0; fi
    setsid nohup bash -lc "cd /vault/amir/efficient-llm/teamily-project && env GPU_PAIR=0,1 PORT=30924 ARMS=int2 TASKS=gpqa OUT_ROOT=artifacts/oscar_gptoss20b/math_grid bash pipelines/oscar_e2e/run_gptoss_math.sh" > logs/gptoss_math_driver_server6_int2_gpqa.log 2>&1 < /dev/null &
    echo $! > logs/gptoss_math_driver_server6_int2_gpqa.pid
    echo launched'
  local launched
  launched="$(ssh "${SSH_OPTS[@]}" "$REMOTE" "$rcmd" 2>/dev/null || true)"
  if [[ "$launched" == "launched" ]]; then
    log "requested remote int2/gpqa launch on $REMOTE GPUs 0,1 port 30924"
  fi
}

log "watch start"
while true; do
  launch_local_vq2_gpqa
  if remote_pair_free 0 1; then
    launch_remote_int2_gpqa
  fi

  vq_done=0
  int_done=0
  [[ -f artifacts/oscar_gptoss20b/math_grid/vq2/gpqa/metrics.json ]] && vq_done=1
  ssh "${SSH_OPTS[@]}" "$REMOTE" \
    'test -f /vault/amir/efficient-llm/teamily-project/artifacts/oscar_gptoss20b/math_grid/int2/gpqa/metrics.json' \
    >/dev/null 2>&1 && int_done=1 || true
  (( vq_done == 1 && int_done == 1 )) && break
  sleep "$SLEEP_SEC"
done
log "watch complete"
