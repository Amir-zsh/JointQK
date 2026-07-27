#!/bin/bash
# Figure 4 RIGHT panel under the replay workload (BS 1/8/32 at 100k).
# Distinct prompts per request, set run twice, second measured -- so KV capacity
# binds and the batch panel measures what the paper's does.
set -uo pipefail
ROOT=/raid/amir/quantization/teamily-project; cd "$ROOT"
CFG=pipelines/oscar_e2e/configs; PY=.venv/bin/python; HB=logs/fig4_replay.heartbeat
docker exec oscar-ab bash -lc "pkill -9 -f '[l]aunch_server'" >/dev/null 2>&1
for _ in $(seq 1 60); do
  b=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000{n++} END{print n+0}')
  [ "$b" -eq 0 ] && break; touch "$HB"; sleep 10
done
$PY -u pipelines/oscar_e2e/run_decode_speed.py --config $CFG/fig4_right_qwen3_8b_replay.json \
    --gpus 0,1,2,3 --base-port 31000 --shard-cells --parallel-measure > logs/fig4_replay_8b.log 2>&1 &
A=$!; sleep 20
$PY -u pipelines/oscar_e2e/run_decode_speed.py --config $CFG/fig4_right_qwen3_4b_thinking_replay.json \
    --gpus 4,5,6,7 --base-port 31300 --shard-cells --parallel-measure > logs/fig4_replay_4b.log 2>&1 &
B=$!
while kill -0 $A 2>/dev/null || kill -0 $B 2>/dev/null; do touch "$HB"; sleep 20; done
docker exec oscar-ab bash -lc "pkill -9 -f '[l]aunch_server'" >/dev/null 2>&1
echo "=== 8B REPLAY ==="; tail -14 logs/fig4_replay_8b.log
echo "=== 4B REPLAY ==="; tail -14 logs/fig4_replay_4b.log
echo "=== REPLAY DONE $(date -Is) ==="
