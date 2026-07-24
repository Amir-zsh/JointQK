#!/bin/bash
# P2 smoke: gpt-oss-20B quantized KV on the hybrid SWA pool (A100, TP=2).
#
# Stage S1: int2 (unified mixed HP+int2 on the 12 full-attention layers,
#           SWA layers bf16) — boot, greedy gen probe, NIAH-8K slice.
# Stage S2: vq2 (same pool, K tier = group-VQ via the gpqacc128k codebook).
#
# gpt-oss specifics vs the llama serve config:
#   ABSORB_V_ROT=0     weight-folding walks contiguous global layers; the
#                      hybrid rotation bundle is dense over full-attn layers.
#   QUANT_GROUP_SIZE=0 head_dim=64 -> per-head single scale group.
#   T=1.0 sampling for NIAH (greedy degenerates on this model).
#
# Usage: smoke_gptoss_quant.sh <gpuA,gpuB> [port]
set -u
ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
GPUS="${1:?gpu pair, e.g. 2,3}"
PORT="${2:-30931}"
MODEL="unsloth/gpt-oss-20b-BF16"
LOG="$ROOT/logs/smoke_gptoss_quant.log"
HB="$ROOT/logs/smoke_gptoss_quant.heartbeat"
SRV="$ROOT/logs/smoke_gptoss_quant_srv.log"
ROWS="artifacts/prompt_rows/niah_8192_gptoss_t1.jsonl"
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; touch "$HB"; }
: > "$LOG"

export ROT_DIR="$ROOT/artifacts/oscar_gptoss20b/rotations_gpqa198"
CB="$ROOT/artifacts/oscar_gptoss20b/vqa_gptoss20b_G4_strat_flat_ptn_gpqacc128k_fp8.pt"

boot_and_probe(){ # mode-label serve-extra-args...
    local label="$1"; shift
    : > "$SRV"
    env ABSORB_V_ROT=0 QUANT_GROUP_SIZE=0 MAX_TOKENS=42000 TP=2 \
        DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-}" \
        nohup bash pipelines/oscar_e2e/serve_oscar.sh --model "$MODEL" \
        --ctx 16384 "$@" --gpu "$GPUS" --port "$PORT" >> "$SRV" 2>&1 &
    local pid=$!
    for i in $(seq 1 240); do
        grep -q "fired up and ready to roll" "$SRV" && break
        grep -qE "Traceback|CUDA out of memory|Received sigquit" "$SRV" && break
        sleep 5; touch "$HB"
    done
    if ! grep -q "fired up" "$SRV"; then
        log "$label BOOT FAIL"; tail -30 "$SRV" >> "$LOG"
        kill $pid 2>/dev/null; return 1
    fi
    log "$label boot OK"
    # greedy short-gen probe (coherence) + decode-throughput probe
    .venv/bin/python - "$PORT" >> "$LOG" 2>&1 <<'PY'
import json, sys, time, urllib.request
port = sys.argv[1]
def gen(text, max_new, temp=0.0):
    p = {"text": text, "sampling_params": {"temperature": temp, "max_new_tokens": max_new, "ignore_eos": True}}
    r = urllib.request.Request(f"http://127.0.0.1:{port}/generate", json.dumps(p).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=600).read())
out = gen("The capital of France is", 16)
print("gen probe:", out["text"][:100])
# decode tok/s: 512 forced tokens, bs=1 (warm, after the probe above)
t0 = time.time(); out = gen("Write a long essay about the ocean.", 512, temp=1.0)
dt = time.time() - t0
ct = out.get("meta_info", {}).get("completion_tokens", 512)
print(f"decode throughput bs=1: {ct} tok in {dt:.1f}s = {ct/dt:.1f} tok/s")
PY
    log "$label gen + throughput probe done"
    return 0
}

run_niah_slice(){ # out-dir
    local SLICE="$ROOT/artifacts/prompt_rows/niah_8192_gptoss_t1_head48.jsonl"
    [ -f "$SLICE" ] || head -48 "$ROWS" > "$SLICE"
    .venv/bin/python pipelines/oscar_e2e/run_prompts_client.py \
        --rows "$SLICE" --port "$PORT" --threads 6 --timeout 1800 \
        --samples 1 --temperature 1.0 --top-p 1.0 --top-k -1 \
        --out "$1" >> "$LOG" 2>&1
    log "niah slice rc=$? -> $1"
}

stop_server(){
    P=$(lsof -t -i :$PORT 2>/dev/null); [ -n "$P" ] && kill $P; sleep 8
    for g in ${GPUS//,/ }; do
        UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $g)
        for pp in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | awk -F', ' -v u="$UUID" '$2==u{print $1}'); do
            [ "$(ps -o user= -p $pp 2>/dev/null | tr -d ' ')" = "$(whoami)" ] && kill -9 $pp 2>/dev/null
        done
    done
    sleep 3
}

log "=== S0 bf16 reference (throughput anchor) ==="
if boot_and_probe "S0-bf16" --bf16 ; then
    :
fi
stop_server

log "=== S1 int2-mixed (hybrid) ==="
if boot_and_probe "S1-int2" ; then
    run_niah_slice artifacts/oscar_gptoss20b/grid/int2/niah_8192_smoke
fi
stop_server

log "=== S2 vq2 (hybrid, gpqacc128k) ==="
if boot_and_probe "S2-vq2" --vq2 --vq-codebook "$CB" ; then
    run_niah_slice artifacts/oscar_gptoss20b/grid/vq2/niah_8192_smoke
fi
stop_server

log "=== SMOKE_GPTOSS_QUANT_DONE"
echo SMOKE_GPTOSS_QUANT_DONE >> "$LOG"
