#!/bin/bash
# plan11 C1 vq2 arm chain: wait for the in-flight NIAH-800 (our-codebook V2
# datapoint) to finish, reboot the vq2 server with Samuel's gpqacc64k
# codebook (the A10-1 repro winner: 95.9/77.6 Mode-A vs ours 87.6/65.8) +
# pytorch sampling, run the 100-row NIAH V2 gate, then the full wave.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
LOG=logs/lh_vq2_chain.log
HB=logs/lh_vq2_chain.heartbeat
exec >>"$LOG" 2>&1

CODEBOOK="third_party/samuel_vq/codebooks/vqa_G4_strat_flat_ptn_gpqacc64k_fp8.pt"
echo "=== $(date -Is) vq2 chain start (codebook=$CODEBOOK)"

# 1. Wait for the our-codebook NIAH-800 run to finish (client writes
#    metrics.json last).
for i in $(seq 1 240); do
    [[ -f artifacts/oscar_e2e/vq2/niah_32768/metrics.json ]] && break
    sleep 30; touch "$HB"
done
echo "$(date -Is) our-codebook NIAH-800 done (or timed out); metrics:"
cat artifacts/oscar_e2e/vq2/niah_32768/metrics.json 2>/dev/null || echo "(missing)"

# 2. Kill the old vq2 server (flashinfer sampling would crash on T=1.0).
PIDS=$(lsof -t -i :30801 2>/dev/null)
[[ -n "$PIDS" ]] && { echo "killing old vq2 server: $PIDS"; kill $PIDS; sleep 10; }

# 3. Boot with Samuel's codebook.
BOOTLOG=logs/lh_vq2_server.log
nohup bash pipelines/oscar_e2e/serve_oscar.sh --vq2 --gpu 1 --port 30801 \
    --vq-codebook "$CODEBOOK" >"$BOOTLOG" 2>&1 &
SERVER_PID=$!
for i in $(seq 1 120); do
    grep -q "The server is fired up and ready to roll" "$BOOTLOG" && break
    grep -qE "Received sigquit|CUDA out of memory" "$BOOTLOG" && { echo "BOOT FAILED"; tail -30 "$BOOTLOG"; exit 2; }
    sleep 5; touch "$HB"
done
grep -q "The server is fired up" "$BOOTLOG" || { echo "BOOT TIMEOUT"; exit 2; }
echo "$(date -Is) vq2 server (gpqacc64k) ready pid=$SERVER_PID"

# 4. V2 gate: 100-row NIAH-32K slice, greedy, vs Mode-A 95.9 (same codebook,
#    f=0.25). Informational log; the wave proceeds either way and the morning
#    report applies the 5-pt bar.
head -100 artifacts/prompt_rows/niah_32768_qwen.jsonl > artifacts/prompt_rows/niah_32768_qwen_n100.jsonl
.venv/bin/python pipelines/oscar_e2e/run_prompts_client.py \
    --rows artifacts/prompt_rows/niah_32768_qwen_n100.jsonl \
    --port 30801 --threads 6 --out artifacts/oscar_e2e/lh/v2_vq2gpqacc_niah100
echo "$(date -Is) V2 gate metrics:"
cat artifacts/oscar_e2e/lh/v2_vq2gpqacc_niah100/metrics.json

# 5. Full wave on the live server.
bash pipelines/oscar_e2e/run_longhorizon_wave.sh --arm vq2 --gpu 1 --port 30801 --reuse-server
RC=$?

echo "$(date -Is) wave rc=$RC; killing server pid=$SERVER_PID"
kill "$SERVER_PID" 2>/dev/null; sleep 5
PIDS=$(lsof -t -i :30801 2>/dev/null); [[ -n "$PIDS" ]] && kill $PIDS 2>/dev/null
echo "=== $(date -Is) vq2 chain done rc=$RC"
