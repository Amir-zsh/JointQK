#!/bin/bash
# gpqacc128k codebook for gpt-oss-20b: same corpus/budget/flags as the 64k
# build, 4 x 131072 concat (prefill-chunk 256 keeps eager attention fp32
# weights ~8.6GB at 128K). Companion to build_gptoss.sh; rotations reused.
set -u
ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
GPU_A="${1:?}"; GPU_B="${2:?}"; GPU_C="${3:?}"
MODEL="unsloth/gpt-oss-20b-BF16"
OUT="$ROOT/artifacts/oscar_gptoss20b"
BAS="$OUT/basis_moments_128k"; POOL="$OUT/query_stats_128k"
CBRAW="$OUT/vqa_gptoss20b_G4_strat_flat_ptn_gpqacc128k.pt"
CBFP8="$OUT/vqa_gptoss20b_G4_strat_flat_ptn_gpqacc128k_fp8.pt"
PY="$ROOT/.venv/bin/python"
LOG="$ROOT/logs/build_gptoss_128k.log"; HB="$ROOT/logs/build_gptoss_128k.heartbeat"
mkdir -p logs
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; touch "$HB"; }
space_guard(){ local free=$(df --output=avail -BG /vault | tail -1 | tr -dc 0-9)
  [ "$free" -ge "${1:-10}" ] || { log "DISK GUARD: ${free}G — abort"; exit 3; }; }
log "=== gptoss gpqacc128k build start gpus=$GPU_A,$GPU_B,$GPU_C"
space_guard 10
if [ ! -f "$BAS/basis_moments.pt" ]; then
  mkdir -p "$BAS"
  CUDA_VISIBLE_DEVICES=$GPU_A,$GPU_B,$GPU_C PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u \
    pipelines/oscar_e2e/gptoss_calibrate.py --model "$MODEL" concat \
    --target-ctx 131072 --n-sequences 4 --pool-stride 4 --prefill-chunk 256 \
    --out-basis "$BAS/basis_moments.pt" --out-pool "$POOL" >> "$LOG" 2>&1
fi
[ -f "$BAS/basis_moments.pt" ] || { log "C1 FAILED"; exit 1; }
NEX=$(ls "$POOL/examples" 2>/dev/null | wc -l); log "C1 done examples=$NEX"
log "C2 train"
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
rm -rf "$POOL"
$PY - <<PYEOF >> "$LOG" 2>&1
import torch
b = torch.load("$CBFP8", map_location="cpu", weights_only=False)
F = b["forward"]
print("BUNDLECHK128", tuple(F.shape), len(b["bounds"]), b["codebooks"][(0,0)][0].shape[0])
assert F.shape == (12, 8, 64, 64) and len(b["bounds"]) == 16
PYEOF
log "=== BUILD_GPTOSS_128K_DONE $CBFP8"
echo BUILD_GPTOSS_128K_DONE >> "$LOG"
