#!/bin/bash
# gpqacc128k for R1-Distill-Llama-8B: K codebook calibrated on the SAME 198-prompt GPQA corpus
# as gpqacc64k, cycled into 4 x 131072-token sequences instead of 8 x 65536 —
# identical total token budget (524K), identical trainer flags, identical
# rotations. The 64k-vs-128k comparison isolates RoPE-position coverage.
#
# Usage: build_r1d_128k_codebook.sh <gpuA> <gpuB> <gpuC>   (capture shards
# across all three; 128K prefill captures need the headroom)
set -u
ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
GPU_A="${1:?gpuA}"; GPU_B="${2:?gpuB}"; GPU_C="${3:-$2}"
# CUDA_VISIBLE_DEVICES rejects duplicate ordinals (Error 101 -> silent CPU
# fallback under device_map=auto), so dedupe the capture GPU list.
CAP_GPUS=$(printf '%s\n' "$GPU_A" "$GPU_B" "$GPU_C" | awk '!s[$0]++' | paste -sd, -)
MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
OUT="$ROOT/artifacts/oscar_r1d_llama8b"
CORPUS="$ROOT/artifacts/oscar_llama31_8b/gpqa_only_corpus.jsonl"
BAS="$OUT/basis_moments_128k"; POOL="$OUT/query_stats_128k"
CBRAW="$OUT/vqa_r1d_llama8b_G4_strat_flat_ptn_gpqacc128k.pt"
CBFP8="$OUT/vqa_r1d_llama8b_G4_strat_flat_ptn_gpqacc128k_fp8.pt"
PY="$ROOT/.venv/bin/python"
LOG="$ROOT/logs/build_r1d_128k.log"; HB="$ROOT/logs/build_r1d_128k.heartbeat"
mkdir -p logs
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; touch "$HB"; }
space_guard(){ local free=$(df --output=avail -BG /vault | tail -1 | tr -dc 0-9)
  [ "$free" -ge "${1:-25}" ] || { log "DISK GUARD: only ${free}G free (<${1}G) — abort"; exit 3; }; }
log "=== gpqacc128k build start gpus=$CAP_GPUS"
space_guard 11
CUDA_VISIBLE_DEVICES=$CAP_GPUS $PY -c "import torch; assert torch.cuda.is_available()" \
  || { log "ABORT: CUDA unavailable under CVD=$CAP_GPUS (would fall back to CPU)"; exit 1; }

# --- C0: GPQA-only segment corpus (same template as capture_gpqa_concat)
if [ ! -f "$CORPUS" ]; then
  $PY - "$CORPUS" <<'PYEOF' >> "$LOG" 2>&1
import json, sys
import pandas as pd
TMPL = ("Answer the following multiple choice question. The last line of your response should be "
        "of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. "
        "Think step by step before answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}")
df = pd.read_csv("artifacts/prompt_rows/gpqa_diamond.csv")
with open(sys.argv[1], "w") as fh:
    for _, r in df.iterrows():
        fh.write(json.dumps({"domain": "gpqa", "text": TMPL.format(
            Question=r["Question"], A=r["Correct Answer"], B=r["Incorrect Answer 1"],
            C=r["Incorrect Answer 2"], D=r["Incorrect Answer 3"])}) + "\n")
print("corpus segments:", sum(1 for _ in open(sys.argv[1])))
PYEOF
fi
[ -f "$CORPUS" ] || { log "C0 FAILED"; exit 1; }

# --- C1: 4 x 128K concat capture (3-GPU sharded weights + activations)
log "C1 concat-128k capture"
if [ ! -f "$BAS/basis_moments.pt" ]; then
  mkdir -p "$BAS"
  CUDA_VISIBLE_DEVICES=$CAP_GPUS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u \
    pipelines/oscar_e2e/capture_mixed_concat.py --model "$MODEL" --corpus "${CORPUS#$ROOT/}" \
    --target-ctx 131072 --n-sequences 4 --pool-stride 4 \
    --out-basis "$BAS/basis_moments.pt" --out-pool "$POOL" >> "$LOG" 2>&1
fi
[ -f "$BAS/basis_moments.pt" ] || { log "C1 FAILED"; exit 1; }
NEX=$(ls "$POOL/examples" 2>/dev/null | wc -l); log "C1 done pool examples=$NEX"
space_guard 8

# --- C2/C3: train + fp8 (identical flags to gpqacc64k)
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
[ -f "$CBFP8" ] || { log "C3 fp8 FAILED"; exit 1; }
log "C DONE -> $CBFP8; deleting pool (disk)"
rm -rf "$POOL"

# --- V: gates + vq2 smoke on the new bundle
log "V1 gates on 128k bundle"
PYTHONPATH="$ROOT/vendor/OSCAR-vq/sglang-research/python" CUDA_VISIBLE_DEVICES=$GPU_A \
  timeout 600 "$ROOT/.venv-oscar/bin/python" pipelines/oscar_e2e/verify_vq_engine.py \
  --bundle "${CBFP8#$ROOT/}" --v-bundle /nonexistent --layers 0 5 18 31 >> "$LOG" 2>&1
grep -q "ALL GATES PASS" <(tail -5 "$LOG") || { log "V1 FAILED"; exit 1; }
log "V1 PASS"

export ROT_DIR="$OUT/rotations_gpqa198"
log "V2 vq2 smoke (128k codebook)"
: > logs/llama_vq2_128k_smoke.log
nohup bash pipelines/oscar_e2e/serve_oscar.sh --vq2 --model "$MODEL" \
  --vq-codebook "${CBFP8#$ROOT/}" --gpu "$GPU_A" --port 30830 \
  >> logs/llama_vq2_128k_smoke.log 2>&1 &
SPID=$!
for i in $(seq 1 90); do
  grep -q "The server is fired up and ready to roll" logs/llama_vq2_128k_smoke.log && break
  grep -qE "Received sigquit|CUDA out of memory|Not enough memory" logs/llama_vq2_128k_smoke.log && break
  sleep 5; touch "$HB"
done
if grep -q "fired up" logs/llama_vq2_128k_smoke.log; then
  python3 -c "
import json, urllib.request
p={'text':'The capital of France is','sampling_params':{'temperature':0.0,'max_new_tokens':16}}
r=urllib.request.Request('http://127.0.0.1:30830/generate', json.dumps(p).encode(), {'Content-Type':'application/json'})
print('vq2-128k gen:', json.loads(urllib.request.urlopen(r,timeout=180).read())['text'][:80])" >> "$LOG" 2>&1 \
    && log "V2 smoke PASS" || { log "V2 smoke GEN-FAIL"; exit 1; }
else
  log "V2 smoke BOOT-FAIL"; tail -8 logs/llama_vq2_128k_smoke.log >> "$LOG"; exit 1
fi
kill $SPID 2>/dev/null; P=$(lsof -t -i :30830 2>/dev/null); [ -n "$P" ] && kill $P; sleep 8

log "=== BUILD_128K_DONE codebook=$CBFP8"
echo BUILD_128K_DONE >> "$LOG"
