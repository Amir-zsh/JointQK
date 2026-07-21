#!/bin/bash
# Llama-3.1-8B-Instruct onboarding for the OSCAR-vq engine (plan: lambda6).
# Adapted from Samuel's Qwen3-4B/32B build recipes (logs/build_4b_*.sh,
# build_qwen3_32b.sh in his oscar_vq2 tree) with our paths + disk guards.
#
# Stages (each gated; log + heartbeat in logs/):
#   G0  K integrity gates with the EXISTING (pre-ptn) Llama codebook
#   S1  bf16 serve smoke (--model flag, no rotations needed)
#   R   GPQA QKV dump (50 prompts) -> OSCAR rotations (qqt_sst, r_h_pbr)
#       -> orthogonality validation -> dump DELETED (disk)
#   C   64K-concat GPQA capture -> stratified/flat/ptn codebook (bpc 2)
#       -> fp8-e5m2 bundle (the Llama analogue of gpqacc64k)
#   V   gates on the NEW bundle + int2 smoke (new ROT_DIR) + vq2 smoke
#
# Usage: build_llama31_8b.sh <gpuA> <gpuB>   (R on gpuA, C on gpuB, parallel)
set -u
ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
GPU_A="${1:?gpuA}"; GPU_B="${2:?gpuB}"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
OUT="$ROOT/artifacts/oscar_llama31_8b"
DUMP="$OUT/qkv_dump"; ROT="$OUT/rotations"
BAS="$OUT/basis_moments"; POOL="$OUT/query_stats"
CBRAW="$OUT/vqa_llama31_8b_G4_strat_flat_ptn_gpqacc64k.pt"
CBFP8="$OUT/vqa_llama31_8b_G4_strat_flat_ptn_gpqacc64k_fp8.pt"
CSV="$ROOT/artifacts/prompt_rows/gpqa_diamond.csv"
OLD_BUNDLE="$ROOT/third_party/samuel_vq/codebooks/vqa_G4_strat_flat_llama_fp8.pt"
PY="$ROOT/.venv/bin/python"
LOG="$ROOT/logs/build_llama31_8b.log"; HB="$ROOT/logs/build_llama31_8b.heartbeat"
mkdir -p "$OUT" logs
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; touch "$HB"; }
space_guard(){ local free=$(df --output=avail -BG /vault | tail -1 | tr -dc 0-9)
  [ "$free" -ge "${1:-15}" ] || { log "DISK GUARD: only ${free}G free (<${1}G) — abort"; exit 3; }; }
log "=== build start gpus=$GPU_A,$GPU_B model=$MODEL"
space_guard 20

# --- G0: K gates on the existing Llama bundle (32 layers -> probe 0 5 18 31)
log "G0 gates (existing pre-ptn Llama bundle)"
PYTHONPATH="$ROOT/vendor/OSCAR-vq/sglang-research/python" CUDA_VISIBLE_DEVICES=$GPU_A \
  timeout 600 "$ROOT/.venv-oscar/bin/python" pipelines/oscar_e2e/verify_vq_engine.py \
  --bundle "${OLD_BUNDLE#$ROOT/}" --v-bundle /nonexistent --layers 0 5 18 31 \
  >> "$LOG" 2>&1
grep -q "ALL GATES PASS" <(tail -5 "$LOG") || { log "G0 FAILED"; exit 1; }
log "G0 PASS"

# --- S1: bf16 serve smoke
log "S1 bf16 smoke"
nohup bash pipelines/oscar_e2e/serve_oscar.sh --bf16 --model "$MODEL" \
  --gpu "$GPU_A" --port 30830 > logs/llama_bf16_smoke.log 2>&1 &
SPID=$!
for i in $(seq 1 90); do
  grep -q "The server is fired up and ready to roll" logs/llama_bf16_smoke.log && break
  grep -qE "Received sigquit|CUDA out of memory|Not enough memory" logs/llama_bf16_smoke.log && break
  sleep 5; touch "$HB"
done
if grep -q "fired up" logs/llama_bf16_smoke.log; then
  python3 - <<'PYEOF' >> "$LOG" 2>&1
import json, urllib.request
p = {"text": "The capital of France is", "sampling_params": {"temperature": 0.0, "max_new_tokens": 16}}
r = urllib.request.Request("http://127.0.0.1:30830/generate", json.dumps(p).encode(), {"Content-Type": "application/json"})
print("S1 gen:", json.loads(urllib.request.urlopen(r, timeout=180).read())["text"][:80])
PYEOF
  log "S1 PASS"
else
  log "S1 FAILED (boot)"; tail -12 logs/llama_bf16_smoke.log >> "$LOG"; kill $SPID 2>/dev/null; exit 1
fi
kill $SPID 2>/dev/null; PIDS=$(lsof -t -i :30830 2>/dev/null); [ -n "$PIDS" ] && kill $PIDS; sleep 8


# --- R (GPU_A): QKV dump (50 prompts) -> rotations -> validate -> cleanup
log "R1 qkv dump (50 prompts, gpu $GPU_A)"
if [ ! -f "$ROT/k_rotation_qqt_r_h_pbr.pt" ]; then
  if [ ! -d "$DUMP/layer_0" ]; then
    CUDA_VISIBLE_DEVICES=$GPU_A $PY -u third_party/samuel_vq/capture_qkv_dump.py \
      --model "$MODEL" --csv "$CSV" --num-prompts 50 --out "$DUMP" >> "$LOG" 2>&1
  fi
  [ -d "$DUMP/layer_0" ] || { log "R1 FAILED"; exit 1; }
  NL=$(ls -d "$DUMP"/layer_* | wc -l); log "R1 done layers=$NL"
  [ "$NL" -eq 32 ] || { log "expected 32 layers got $NL; abort"; exit 1; }
  log "R2 rotations (qqt_sst, r_h_pbr, head_dim 128)"
  mkdir -p "$ROT"
  OMP_NUM_THREADS=32 $PY vendor/OSCAR-vq/rotation/compute_kv_rotation.py \
    --dump-path "$DUMP" --output-dir "$ROT" --head-dim 128 \
    --method qqt_sst --composition r_h_pbr --chunk-id all >> "$LOG" 2>&1
  [ -f "$ROT/k_rotation_qqt_r_h_pbr.pt" ] || { log "R2 FAILED"; exit 1; }
fi
$PY - <<PYEOF >> "$LOG" 2>&1
import torch
for f in ["k_rotation_qqt_r_h_pbr", "v_rotation_sst_r_h_pbr"]:
    d = torch.load("$ROT/%s.pt" % f, map_location="cpu", weights_only=False); L = d["layers"]
    R = L[sorted(L)[0]]["rotation"].double()
    e = (R @ R.T - torch.eye(128)).abs().max().item()
    print("ROTVAL", f, "n_layers", len(L), "orth_err %.1e" % e)
    assert len(L) == 32 and e < 1e-4
PYEOF
grep -q "ROTVAL v_rotation" <(tail -6 "$LOG") || { log "R3 validation FAILED"; exit 1; }
log "R DONE; deleting dump (disk)"
rm -rf "$DUMP"

# --- C (sequential, BOTH gpus): 64K-concat capture -> ptn codebook -> fp8.
# 8B fp16 weights + a 65536-token prefill OOM one A100-40GB; device_map=auto
# shards weights across both cards (R has finished by now, GPU_A is free).

  log "C1 concat-64k capture start (gpus $GPU_A,$GPU_B, sharded)"
  if [ ! -f "$BAS/basis_moments.pt" ]; then
    mkdir -p "$BAS"
    CUDA_VISIBLE_DEVICES=$GPU_A,$GPU_B PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u \
      third_party/samuel_vq/capture_gpqa_concat.py --model "$MODEL" --csv "$CSV" \
      --target-ctx 65536 --n-sequences 8 --pool-stride 4 \
      --out-basis "$BAS/basis_moments.pt" --out-pool "$POOL" >> "$LOG" 2>&1
  fi
  [ -f "$BAS/basis_moments.pt" ] || { log "C1 FAILED"; exit 1; }
  NEX=$(ls "$POOL/examples" 2>/dev/null | wc -l); log "C1 done pool examples=$NEX"
  space_guard 15
  log "C2 codebook train (stratified flat ptn bpc2)"
  IDX=$(seq 0 $((NEX-1)) | paste -sd' ')
  CUDA_VISIBLE_DEVICES=$GPU_B $PY -u third_party/samuel_vq/train_group_vq_alloc.py \
    --basis-moments "$BAS/basis_moments.pt" --data-root "$POOL" --code-idx $IDX \
    --grouping stratified --allocation flat --bpc 2 --pertoken-norm \
    --out "$CBRAW" >> "$LOG" 2>&1
  [ -f "$CBRAW" ] || { log "C2 FAILED"; exit 1; }
  $PY third_party/samuel_vq/make_fp8.py --in "$CBRAW" --out "$CBFP8" --fmt e5m2 >> "$LOG" 2>&1
  [ -f "$CBFP8" ] || { log "C3 fp8 FAILED"; exit 1; }
  log "C DONE -> $CBFP8"

# --- V: gates on the NEW bundle, int2 smoke, vq2 smoke
log "V1 gates on new ptn bundle"
PYTHONPATH="$ROOT/vendor/OSCAR-vq/sglang-research/python" CUDA_VISIBLE_DEVICES=$GPU_A \
  timeout 600 "$ROOT/.venv-oscar/bin/python" pipelines/oscar_e2e/verify_vq_engine.py \
  --bundle "${CBFP8#$ROOT/}" --v-bundle /nonexistent --layers 0 5 18 31 >> "$LOG" 2>&1
grep -q "ALL GATES PASS" <(tail -5 "$LOG") || { log "V1 FAILED"; exit 1; }
log "V1 PASS"

smoke_serve(){ # <mode-args...> ; port 30830, gpu A
  local name="$1"; shift
  nohup bash pipelines/oscar_e2e/serve_oscar.sh "$@" --gpu "$GPU_A" --port 30830 \
    > "logs/llama_${name}_smoke.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 90); do
    grep -q "The server is fired up and ready to roll" "logs/llama_${name}_smoke.log" && break
    grep -qE "Received sigquit|CUDA out of memory|Not enough memory" "logs/llama_${name}_smoke.log" && break
    sleep 5; touch "$HB"
  done
  if grep -q "fired up" "logs/llama_${name}_smoke.log"; then
    python3 -c "
import json, urllib.request
p={'text':'The capital of France is','sampling_params':{'temperature':0.0,'max_new_tokens':16}}
r=urllib.request.Request('http://127.0.0.1:30830/generate', json.dumps(p).encode(), {'Content-Type':'application/json'})
print('$name gen:', json.loads(urllib.request.urlopen(r,timeout=180).read())['text'][:80])" >> "$LOG" 2>&1 \
      && log "$name smoke PASS" || log "$name smoke GEN-FAIL"
  else
    log "$name smoke BOOT-FAIL"; tail -8 "logs/llama_${name}_smoke.log" >> "$LOG"
  fi
  kill $pid 2>/dev/null; local P=$(lsof -t -i :30830 2>/dev/null); [ -n "$P" ] && kill $P; sleep 8
}
export ROT_DIR="$ROT"
log "V2 int2 smoke (llama rotations)"
smoke_serve int2 --model "$MODEL"
log "V3 vq2 smoke (new ptn codebook)"
smoke_serve vq2 --vq2 --model "$MODEL" --vq-codebook "${CBFP8#$ROOT/}"

log "=== BUILD_LLAMA_DONE rotations=$ROT codebook=$CBFP8"
echo BUILD_LLAMA_DONE >> "$LOG"
