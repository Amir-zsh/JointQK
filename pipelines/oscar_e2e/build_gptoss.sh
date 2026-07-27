#!/bin/bash
# gpt-oss-20b P1 calibration: full-attention layers only (12 of 24; SWA
# layers stay bf16 in the serving policy). Unified 198-prompt GPQA corpus,
# both deliveries. head_dim 64 -> 64x64 rotations, NG=16 codebook groups.
# P2 (engine layer-policy wiring for quantized serving) is a separate step;
# this build produces + validates the calibration artifacts.
#
# Usage: build_gptoss.sh <gpuA> <gpuB> <gpuC>
set -u
ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
GPU_A="${1:?}"; GPU_B="${2:?}"; GPU_C="${3:?}"
MODEL="${MODEL:-unsloth/gpt-oss-20b-BF16}"
OUT="${OUT:-$ROOT/artifacts/oscar_gptoss20b}"
DUMP="$OUT/qkv_dump"; ROT="$OUT/rotations_gpqa198"
BAS="$OUT/basis_moments"; POOL="$OUT/query_stats"
CBRAW="$OUT/vqa_gptoss20b_G4_strat_flat_ptn_gpqacc64k.pt"
CBFP8="$OUT/vqa_gptoss20b_G4_strat_flat_ptn_gpqacc64k_fp8.pt"
PY="$ROOT/.venv/bin/python"
LOG="${LOG:-$ROOT/logs/build_gptoss.log}"; HB="${HB:-$ROOT/logs/build_gptoss.heartbeat}"
mkdir -p "$OUT" logs
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; touch "$HB"; }
space_guard(){ local free=$(df --output=avail -BG "${DISK_GUARD_PATH:-$OUT}" | tail -1 | tr -dc 0-9)
  [ "$free" -ge "${1:-25}" ] || { log "DISK GUARD: ${free}G free — abort"; exit 3; }; }
log "=== gptoss P1 build start gpus=$GPU_A,$GPU_B,$GPU_C"
# head_dim 64 x 12 layers: dump ~3G, pool ~1.6G — the whole build fits in
# <10G (guards sized accordingly; /vault is a shared volume that runs hot).
space_guard 12

# --- R: rotation dump (2 GPUs; short prompts) -> rotations (head_dim 64)
if [ ! -f "$ROT/k_rotation_qqt_r_h_pbr.pt" ]; then
  log "R1 dump 198 prompts (full-attention layers)"
  if [ ! -d "$DUMP/layer_0" ]; then
    CUDA_VISIBLE_DEVICES=$GPU_A,$GPU_B $PY -u pipelines/oscar_e2e/gptoss_calibrate.py \
      --model "$MODEL" dump --num-prompts 198 --out "$DUMP" >> "$LOG" 2>&1
  fi
  [ -d "$DUMP/layer_11" ] || { log "R1 FAILED"; exit 1; }
  log "R2 rotations (qqt_sst r_h_pbr head-dim 64)"
  mkdir -p "$ROT"
  cp "$DUMP/layer_map.json" "$ROT/" 2>/dev/null
  OMP_NUM_THREADS=32 $PY vendor/OSCAR-vq/rotation/compute_kv_rotation.py \
    --dump-path "$DUMP" --output-dir "$ROT" --head-dim 64 \
    --method qqt_sst --composition r_h_pbr --chunk-id all >> "$LOG" 2>&1
  [ -f "$ROT/k_rotation_qqt_r_h_pbr.pt" ] || { log "R2 FAILED"; exit 1; }
fi
$PY - <<PYEOF >> "$LOG" 2>&1
import torch
for f in ["k_rotation_qqt_r_h_pbr", "v_rotation_sst_r_h_pbr"]:
    d = torch.load("$ROT/%s.pt" % f, map_location="cpu", weights_only=False); L = d["layers"]
    R = L[sorted(L)[0]]["rotation"].double()
    e = (R @ R.T - torch.eye(R.shape[0])).abs().max().item()
    print("ROTVAL gptoss", f, "n_layers", len(L), "dim", R.shape[0], "orth_err %.1e" % e)
    assert len(L) == 12 and R.shape[0] == 64 and e < 1e-4
PYEOF
grep -q "ROTVAL gptoss v_rotation" <(tail -6 "$LOG") || { log "R3 validation FAILED"; exit 1; }
log "R DONE; deleting dump"
rm -rf "$DUMP"

# --- C: 8x64K concat capture (3 GPUs) -> codebook -> fp8
log "C1 concat-64k capture"
if [ ! -f "$BAS/basis_moments.pt" ]; then
  mkdir -p "$BAS"
  CUDA_VISIBLE_DEVICES=$GPU_A,$GPU_B,$GPU_C PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u \
    pipelines/oscar_e2e/gptoss_calibrate.py --model "$MODEL" concat \
    --target-ctx 65536 --n-sequences 8 --pool-stride 4 \
    --out-basis "$BAS/basis_moments.pt" --out-pool "$POOL" >> "$LOG" 2>&1
fi
[ -f "$BAS/basis_moments.pt" ] || { log "C1 FAILED"; exit 1; }
NEX=$(ls "$POOL/examples" 2>/dev/null | wc -l); log "C1 done pool examples=$NEX"
space_guard 8
log "C2 codebook train (stratified flat ptn bpc2, d=64)"
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

# --- V: bundle sanity (shapes, NG=16, fp8 payload) — engine gates are P2
$PY - <<PYEOF >> "$LOG" 2>&1
import torch
b = torch.load("$CBFP8", map_location="cpu", weights_only=False)
F = b["forward"]; NG = len(b["bounds"]); K = b["codebooks"][(0,0)][0].shape[0]
G = b["bounds"][0][1] - b["bounds"][0][0]
print("BUNDLECHK layers", F.shape[0], "heads", F.shape[1], "d", F.shape[2],
      "NG", NG, "K", K, "G", G, "ptn", b.get("pertoken_norm"))
assert F.shape == (12, 8, 64, 64) and NG == 16 and G == 4 and K == 256
PYEOF
grep -q "BUNDLECHK" <(tail -4 "$LOG") || { log "V FAILED"; exit 1; }
log "=== BUILD_GPTOSS_P1_DONE codebook=$CBFP8"
echo BUILD_GPTOSS_P1_DONE >> "$LOG"
