#!/bin/bash
# One-shot status: GPUs on both boxes + whatever overnight job is running.
#
# "free" means ZERO compute processes. A memory threshold alone wrongly counts
# co-tenants' small-footprint live jobs as free, which is how a co-tenanted
# boot tripped the memory-balance abort earlier.
#
#   bash pipelines/oscar_e2e/gpu_status.sh
set -u
ROOT="${ROOT:-/vault/amir/efficient-llm/teamily-project}"
L6=10.137.32.78          # lambda-server6 ("lambda6" does not resolve)

show(){ # label   ssh-prefix ("" for local)
  local label="$1"; shift
  local pre=("$@")
  echo "=== $label"
  "${pre[@]}" bash -s <<'EOS' 2>/dev/null || echo "   unreachable"
nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader > /tmp/_apps.csv
free=0
while IFS=, read -r g u m t util; do
  gi=$(echo "$g"|xargs); uu=$(echo "$u"|xargs)
  mu=$(echo "$m"|xargs|tr -d ' MiB'); tt=$(echo "$t"|xargs|tr -d ' MiB')
  own=$(awk -F', ' -v k="$uu" '$2==k{print $1}' /tmp/_apps.csv | while read -r p; do
          printf '%s ' "$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')"; done | tr ' ' '\n' | sort -u | tr '\n' ' ')
  if [ -z "${own// /}" ]; then own="FREE"; free=$((free+1)); fi
  printf "  gpu%s  free %6s MiB  util %3s%%  %s\n" "$gi" "$((tt-mu))" "$(echo "$util"|xargs|tr -d ' %')" "$own"
done < <(nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader)
echo "  -> $free GPU(s) with zero compute processes"
EOS
}

show "lambda7 (local)"
show "lambda6 ($L6)" ssh -o BatchMode=yes -o ConnectTimeout=10 "$L6"

echo "=== overnight job (lambda6)"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$L6" "
  cd $ROOT 2>/dev/null || exit 0
  if [ -f logs/overnight_gptoss_vq2.log ]; then
    age=\$(( \$(date +%s) - \$(stat -c %Y logs/overnight_gptoss_vq2.heartbeat 2>/dev/null || echo 0) ))
    # pgrep -f matches this very ssh command line, which reported a
    # 9h-dead job as RUNNING. Match the bash process, and treat a stale
    # heartbeat as dead regardless of what pgrep says.
    alive=\$(pgrep -f 'bash pipelines/oscar_e2e/overnight_gptoss_vq2.sh' >/dev/null && echo RUNNING || echo 'NOT RUNNING')
    [ \"\$age\" -gt 900 ] && alive=\"\$alive (STALE)\"
    echo \"  \$alive · heartbeat \${age}s ago\"
    grep -aE 'PHASE|OK\$|rc=|DONE|FAIL|NO_GPU' logs/overnight_gptoss_vq2.log | tail -8 | sed 's/^/  /'
  else
    echo '  no overnight log'
  fi
  echo '  --- vq2 grid cells with metrics.json ---'
  for d in artifacts/oscar_gptoss20b/grid/vq2/*/; do
    [ -f \"\$d/metrics.json\" ] && echo \"  done: \$d\"
  done 2>/dev/null | head -10
" 2>/dev/null || echo "  unreachable"
