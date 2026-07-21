#!/bin/bash
# Calibration-unified rotations: recompute OSCAR rotations for BOTH models
# from OUR canonical corpus — the same 198 GPQA-Diamond prompts that feed the
# gpqacc64k codebooks (v1 provenance: Llama = 50 prompts ours, Qwen = authors'
# released 83-prompt rotations). Sequential per model: each QKV dump is
# ~20 GB and /vault on lambda6 is tight, so dump -> rotations -> validate ->
# delete dump before the next model.
#
# Usage: recalibrate_rotations.sh <gpu>
set -u
ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
GPU="${1:?gpu}"
CSV="$ROOT/artifacts/prompt_rows/gpqa_diamond.csv"
PY="$ROOT/.venv/bin/python"
LOG="$ROOT/logs/recalibrate_rotations.log"; HB="$ROOT/logs/recalibrate_rotations.heartbeat"
mkdir -p logs
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; touch "$HB"; }
space_guard(){ local free=$(df --output=avail -BG /vault | tail -1 | tr -dc 0-9)
  [ "$free" -ge "${1:-25}" ] || { log "DISK GUARD: only ${free}G free (<${1}G) — abort"; exit 3; }; }

build_one(){ # model rot_out dump_dir
  local MODEL="$1" ROT="$2" DUMP="$3"
  local TAG; TAG=$(basename "$ROT")
  if [ -f "$ROT/k_rotation_qqt_r_h_pbr.pt" ]; then log "$TAG already built — skip"; return 0; fi
  space_guard 25
  log "$TAG: dump 198 prompts (gpu $GPU)"
  if [ ! -d "$DUMP/layer_0" ]; then
    CUDA_VISIBLE_DEVICES=$GPU $PY -u third_party/samuel_vq/capture_qkv_dump.py \
      --model "$MODEL" --csv "$CSV" --num-prompts 198 --gpu 0 --out "$DUMP" >> "$LOG" 2>&1
  fi
  [ -d "$DUMP/layer_0" ] || { log "$TAG dump FAILED"; return 1; }
  local NL; NL=$(ls -d "$DUMP"/layer_* | wc -l); log "$TAG dump done layers=$NL"
  space_guard 5
  log "$TAG: rotations (qqt_sst, r_h_pbr, head_dim 128)"
  mkdir -p "$ROT"
  OMP_NUM_THREADS=32 $PY vendor/OSCAR-vq/rotation/compute_kv_rotation.py \
    --dump-path "$DUMP" --output-dir "$ROT" --head-dim 128 \
    --method qqt_sst --composition r_h_pbr --chunk-id all >> "$LOG" 2>&1
  [ -f "$ROT/k_rotation_qqt_r_h_pbr.pt" ] || { log "$TAG rotations FAILED"; return 1; }
  $PY - <<PYEOF >> "$LOG" 2>&1
import torch
for f in ["k_rotation_qqt_r_h_pbr", "v_rotation_sst_r_h_pbr"]:
    d = torch.load("$ROT/%s.pt" % f, map_location="cpu", weights_only=False); L = d["layers"]
    R = L[sorted(L)[0]]["rotation"].double()
    e = (R @ R.T - torch.eye(R.shape[0])).abs().max().item()
    print("ROTVAL $TAG", f, "n_layers", len(L), "orth_err %.1e" % e)
    assert e < 1e-4
PYEOF
    grep -q "ROTVAL $TAG v_rotation" <(tail -6 "$LOG") || { log "$TAG validation FAILED"; return 1; }
  log "$TAG DONE; deleting dump"
  rm -rf "$DUMP"
  return 0
}

log "=== rotation recalibration start (gpu $GPU, corpus=$CSV 198 prompts)"
build_one "meta-llama/Llama-3.1-8B-Instruct" \
  "$ROOT/artifacts/oscar_llama31_8b/rotations_gpqa198" \
  "$ROOT/artifacts/oscar_llama31_8b/qkv_dump_gpqa198" || { echo ROTCAL_FAIL >> "$LOG"; exit 1; }
build_one "Qwen/Qwen3-8B" \
  "$ROOT/artifacts/oscar_e2e/rotzoo/Qwen3-8B/gpqa198_own" \
  "$ROOT/artifacts/oscar_e2e/qkv_dump_qwen_gpqa198" || { echo ROTCAL_FAIL >> "$LOG"; exit 1; }
log "=== ROTCAL_DONE"
echo ROTCAL_DONE >> "$LOG"
