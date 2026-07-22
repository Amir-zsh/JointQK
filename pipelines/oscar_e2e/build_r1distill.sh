#!/bin/bash
# DeepSeek-R1-Distill-Llama-8B onboarding — identical architecture to
# Llama-3.1-8B (head_dim 128, 8 KV heads, full attention), so the entire
# validated stack reuses as-is. Calibration = the unified 198-prompt GPQA
# corpus, both deliveries (per-prompt dump -> rotations; 8x64K concat ->
# ptn codebook). Purpose: the long-CoT reasoning axis the Llama grid can't
# show (R1-distill thinks for thousands of decode tokens).
#
# Usage: build_r1distill.sh <gpuA> <gpuB>
set -u
ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
GPU_A="${1:?gpuA}"; GPU_B="${2:?gpuB}"
MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
OUT="$ROOT/artifacts/oscar_r1d_llama8b"
DUMP="$OUT/qkv_dump"; ROT="$OUT/rotations_gpqa198"
CORPUS="$ROOT/artifacts/oscar_llama31_8b/gpqa_only_corpus.jsonl"
BAS="$OUT/basis_moments"; POOL="$OUT/query_stats"
CBRAW="$OUT/vqa_r1d_llama8b_G4_strat_flat_ptn_gpqacc64k.pt"
CBFP8="$OUT/vqa_r1d_llama8b_G4_strat_flat_ptn_gpqacc64k_fp8.pt"
CSV="$ROOT/artifacts/prompt_rows/gpqa_diamond.csv"
PY="$ROOT/.venv/bin/python"
LOG="$ROOT/logs/build_r1distill.log"; HB="$ROOT/logs/build_r1distill.heartbeat"
mkdir -p "$OUT" logs
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; touch "$HB"; }
space_guard(){ local free=$(df --output=avail -BG /vault | tail -1 | tr -dc 0-9)
  [ "$free" -ge "${1:-25}" ] || { log "DISK GUARD: only ${free}G free (<${1}G) — abort"; exit 3; }; }
log "=== r1distill build start gpus=$GPU_A,$GPU_B"
[ -f "$CORPUS" ] || { log "corpus missing"; exit 1; }
space_guard 30

# --- R: rotations from the 198-prompt dump (R1 chat template via tokenizer)
if [ ! -f "$ROT/k_rotation_qqt_r_h_pbr.pt" ]; then
  log "R1 qkv dump (198 prompts)"
  if [ ! -d "$DUMP/layer_0" ]; then
    CUDA_VISIBLE_DEVICES=$GPU_A $PY -u third_party/samuel_vq/capture_qkv_dump.py \
      --model "$MODEL" --csv "$CSV" --num-prompts 198 --gpu 0 --out "$DUMP" >> "$LOG" 2>&1
  fi
  [ -d "$DUMP/layer_0" ] || { log "R1 dump FAILED"; exit 1; }
  NL=$(ls -d "$DUMP"/layer_* | wc -l); log "R1 dump done layers=$NL"
  [ "$NL" -eq 32 ] || { log "expected 32 layers got $NL"; exit 1; }
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
    print("ROTVAL r1d", f, "n_layers", len(L), "orth_err %.1e" % e)
    assert len(L) == 32 and e < 1e-4
PYEOF
grep -q "ROTVAL r1d v_rotation" <(tail -6 "$LOG") || { log "R3 validation FAILED"; exit 1; }
log "R DONE; deleting dump"
rm -rf "$DUMP"

# --- C: 8x64K concat capture -> ptn codebook -> fp8 (gpqacc64k recipe)
log "C1 concat-64k capture"
if [ ! -f "$BAS/basis_moments.pt" ]; then
  mkdir -p "$BAS"
  CUDA_VISIBLE_DEVICES=$GPU_A,$GPU_B PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u \
    pipelines/oscar_e2e/capture_mixed_concat.py --model "$MODEL" --corpus "${CORPUS#$ROOT/}" \
    --target-ctx 65536 --n-sequences 8 --pool-stride 4 \
    --out-basis "$BAS/basis_moments.pt" --out-pool "$POOL" >> "$LOG" 2>&1
fi
[ -f "$BAS/basis_moments.pt" ] || { log "C1 FAILED"; exit 1; }
NEX=$(ls "$POOL/examples" 2>/dev/null | wc -l); log "C1 done pool examples=$NEX"
space_guard 15
log "C2 codebook train (stratified flat ptn bpc2)"
if [ ! -f "$CBRAW" ]; then
  IDX=$(seq 0 $((NEX-1)) | paste -sd' ')
  CUDA_VISIBLE_DEVICES=$GPU_B $PY -u third_party/samuel_vq/train_group_vq_alloc.py \
    --basis-moments "$BAS/basis_moments.pt" --data-root "$POOL" --code-idx $IDX \
    --grouping stratified --allocation flat --bpc 2 --pertoken-norm \
    --out "$CBRAW" >> "$LOG" 2>&1
fi
[ -f "$CBRAW" ] || { log "C2 FAILED"; exit 1; }
$PY third_party/samuel_vq/make_fp8.py --in "$CBRAW" --out "$CBFP8" --fmt e5m2 >> "$LOG" 2>&1
[ -f "$CBFP8" ] || { log "C3 FAILED"; exit 1; }
log "C DONE -> $CBFP8; deleting pool"
rm -rf "$POOL"

# --- V: gates + serve smokes (bf16 / int2 / vq2), thinking-mode gen check
log "V1 gates on r1d bundle"
PYTHONPATH="$ROOT/vendor/OSCAR-vq/sglang-research/python" CUDA_VISIBLE_DEVICES=$GPU_A \
  timeout 600 "$ROOT/.venv-oscar/bin/python" pipelines/oscar_e2e/verify_vq_engine.py \
  --bundle "${CBFP8#$ROOT/}" --v-bundle /nonexistent --layers 0 5 18 31 >> "$LOG" 2>&1
grep -q "ALL GATES PASS" <(tail -5 "$LOG") || { log "V1 FAILED"; exit 1; }
log "V1 PASS"

export ROT_DIR="$ROT"
smoke(){ # name serve-extra...
  local name="$1"; shift
  : > "logs/r1d_${name}_smoke.log"
  nohup bash pipelines/oscar_e2e/serve_oscar.sh --model "$MODEL" "$@" \
    --gpu "$GPU_A" --port 30830 >> "logs/r1d_${name}_smoke.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 90); do
    grep -q "fired up and ready to roll" "logs/r1d_${name}_smoke.log" && break
    grep -qE "Received sigquit|CUDA out of memory" "logs/r1d_${name}_smoke.log" && break
    sleep 5; touch "$HB"
  done
  if grep -q "fired up" "logs/r1d_${name}_smoke.log"; then
    python3 -c "
import json, urllib.request
p={'text':'<｜begin▁of▁sentence｜><｜User｜>What is 17*23?<｜Assistant｜><think>\n','sampling_params':{'temperature':0.6,'top_p':0.95,'max_new_tokens':300}}
r=urllib.request.Request('http://127.0.0.1:30830/generate', json.dumps(p).encode(), {'Content-Type':'application/json'})
t=json.loads(urllib.request.urlopen(r,timeout=300).read())['text']
print('$name gen ok, has_think_close:', '</think>' in t, '| tail:', t[-80:].replace(chr(10),' '))" >> "$LOG" 2>&1 \
      && log "$name smoke PASS" || log "$name smoke GEN-FAIL"
  else
    log "$name smoke BOOT-FAIL"; tail -6 "logs/r1d_${name}_smoke.log" >> "$LOG"
  fi
  kill $pid 2>/dev/null; local P=$(lsof -t -i :30830 2>/dev/null); [ -n "$P" ] && kill $P; sleep 8
}
smoke bf16 --bf16
smoke int2
smoke vq2 --vq2 --vq-codebook "${CBFP8#$ROOT/}"
log "=== BUILD_R1D_DONE codebook=$CBFP8"
echo BUILD_R1D_DONE >> "$LOG"
