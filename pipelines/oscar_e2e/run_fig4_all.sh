#!/bin/bash
# Figure 4 reproduction for both Qwen models, end to end.
#
# Two driver instances run per phase on DISJOINT GPU sets with DISTINCT base
# ports -- the driver's kill is port-scoped, so overlapping ports would let the
# two runs tear down each other's servers (a mistake made twice already today).
#
# Phase 1: left panels  (bs=1, input 30k/60k/100k)      8B: 4 arms, 4B: 4 arms
# Phase 2: right panels (100k input, BS 1/8/32)         8B: 3 arms, 4B: 3 arms
# Phase 3: render charts
set -uo pipefail
ROOT=/raid/amir/quantization/teamily-project
cd "$ROOT"
CFG=pipelines/oscar_e2e/configs
PY=.venv/bin/python
HB=logs/fig4_all.heartbeat

wait_all_free () {
    for _ in $(seq 1 120); do
        busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
               | awk '$1>2000{n++} END{print n+0}')
        [ "$busy" -eq 0 ] && return 0
        touch "$HB"; sleep 10
    done
    echo "WARN: GPUs still busy after 20 min"
}

phase () {                       # name  cfgA gpusA portA   cfgB gpusB portB
    local name=$1 ca=$2 ga=$3 pa=$4 cb=$5 gb=$6 pb=$7
    echo "=== $name $(date -Is) ==="
    docker exec oscar-ab bash -lc "pkill -9 -f '[l]aunch_server'" >/dev/null 2>&1
    wait_all_free
    $PY -u pipelines/oscar_e2e/run_decode_speed.py --config "$ca" --gpus "$ga" \
        --base-port "$pa" --shard-cells --parallel-measure > "logs/${name}_a.log" 2>&1 &
    local A=$!
    sleep 20
    $PY -u pipelines/oscar_e2e/run_decode_speed.py --config "$cb" --gpus "$gb" \
        --base-port "$pb" --shard-cells --parallel-measure > "logs/${name}_b.log" 2>&1 &
    local B=$!
    while kill -0 $A 2>/dev/null || kill -0 $B 2>/dev/null; do touch "$HB"; sleep 20; done
    echo "--- $name done $(date -Is)"
    tail -14 "logs/${name}_a.log"; tail -14 "logs/${name}_b.log"
}

phase fig4_left \
    $CFG/fig4_left_qwen3_8b.json          0,1,2,3 31000 \
    $CFG/fig4_left_qwen3_4b_thinking.json 4,5,6,7 31200

phase fig4_right \
    $CFG/fig4_right_qwen3_8b.json          0,1,2,3 31400 \
    $CFG/fig4_right_qwen3_4b_thinking.json 4,5,6,7 31600

docker exec oscar-ab bash -lc "pkill -9 -f '[l]aunch_server'" >/dev/null 2>&1
echo "=== rendering charts $(date -Is) ==="
$PY pipelines/eval/plot_fig4.py --out notes/fig4_reproduction.png
echo "=== ALL DONE $(date -Is) ==="
