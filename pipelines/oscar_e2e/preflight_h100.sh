#!/bin/bash
# Preflight for a fresh node: report exactly which required files are missing,
# then print the command that fetches only those.
#
# The handoff's payload list was assembled by hand and is easy to under-specify
# (it omitted artifacts/page_quant2, which the day-0 gate battery opens). This
# checks the concrete files the serve paths, gates and evals actually open, so
# a gap shows up as a named path instead of a confusing failure hours later.
#
#   bash pipelines/oscar_e2e/preflight_h100.sh            # everything
#   bash pipelines/oscar_e2e/preflight_h100.sh gptoss     # one model's needs
set -u
ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
WANT="${1:-all}"
MISS=(); OK=0

chk(){ # path  label
  if [ -e "$1" ]; then OK=$((OK+1)); else MISS+=("$1"); printf '  MISSING  %-62s %s\n' "$1" "$2"; fi
}

echo "=== repo root: $ROOT"

echo "=== environments"
chk .venv/bin/python                   "client/eval venv (uv venv + requirements.lock.txt)"
chk .venv-oscar/bin/python             "engine venv (uv pip install -e vendor/OSCAR/sglang-research/python)"
chk vendor/OSCAR/sglang-research/python "EDITABLE INSTALL ROOT for .venv-oscar - required, not a reference copy"
chk vendor/OSCAR-vq/sglang-research/python "engine clone that PYTHONPATH shadows at serve time"
chk requirements.lock.txt              "154 pins, tracked in git"

if [ "$WANT" = all ] || [ "$WANT" = gates ]; then
echo "=== day-0 gate battery"
chk artifacts/page_quant2/vqg_bundle__qwen3_8b_flat_ptn.pt      "verify_vq_engine.py G1-G3  <-- NOT in the original payload list"
chk third_party/samuel_vq/codebooks/vqv_G4_strided_gpqa_engine.pt "verify_vq_engine.py G4-G5 (V-side)"
chk pipelines/oscar_e2e/verify_gptoss_int2_decode.py            "gpt-oss hybrid gates (from git)"
fi

if [ "$WANT" = all ] || [ "$WANT" = gptoss ]; then
echo "=== gpt-oss-20B"
chk artifacts/oscar_gptoss20b/rotations_gpqa198/k_rotation_qqt_r_h_pbr.pt "K rotation"
chk artifacts/oscar_gptoss20b/rotations_gpqa198/v_rotation_sst_r_h_pbr.pt "V rotation"
chk artifacts/oscar_gptoss20b/rotations_gpqa198/layer_map.json            "dense<->global layer map (hybrid)"
chk artifacts/oscar_gptoss20b/vqa_gptoss20b_G4_strat_flat_ptn_gpqacc128k_fp8.pt "vq2 codebook (canonical 128k calib)"
for L in 8192 16384 32768 65536 131072; do
  chk "artifacts/prompt_rows/niah_${L}_gptoss.jsonl" "NIAH-${L} rows"
done
chk artifacts/prompt_rows/gpqa_diamond.csv "GPQA source (calibration prompts + gpqa cell)"
fi

if [ "$WANT" = all ] || [ "$WANT" = llama ]; then
echo "=== Llama-3.1-8B"
chk artifacts/oscar_llama31_8b/rotations_gpqa198/k_rotation_qqt_r_h_pbr.pt "K rotation"
chk artifacts/oscar_llama31_8b/rotations_gpqa198/v_rotation_sst_r_h_pbr.pt "V rotation"
chk artifacts/oscar_llama31_8b/vqa_llama31_8b_G4_strat_flat_ptn_gpqacc128k_fp8.pt "vq2 codebook (128k calib)"
chk artifacts/prompt_rows/gpqa_diamond_llama.jsonl "GPQA rows"
chk artifacts/prompt_rows/math500_llama.jsonl      "math500 rows"
chk artifacts/prompt_rows/aime25_llama.jsonl       "aime25 rows"
fi

if [ "$WANT" = all ] || [ "$WANT" = qwen ]; then
echo "=== Qwen3-8B (serve_oscar.sh default model)"
chk artifacts/oscar_e2e/rotzoo "rotation zoo (serve default ROT_DIR)"
fi

echo
if [ ${#MISS[@]} -eq 0 ]; then
  echo "PREFLIGHT OK - $OK checks passed, nothing missing."
  exit 0
fi
echo "PREFLIGHT INCOMPLETE - $OK present, ${#MISS[@]} missing."
echo
echo "Fetch ONLY the gaps. Run this ON lambda7, then move the tar the usual"
echo "two-hop way (lambda -> laptop -> switch VPN -> H100):"
echo
echo "  SRC=/vault/amir/efficient-llm/teamily-project; cd \"\$SRC\""
printf "  tar -czf /tmp/kvq_missing.tgz \\\\\n"
for m in "${MISS[@]}"; do
  case "$m" in
    .venv*|requirements.lock.txt|pipelines/*|vendor/OSCAR-vq/*) continue ;;   # built or cloned, not shipped
  esac
  printf "      %s \\\\\n" "$m"
done
echo "      --exclude='__pycache__'"
echo
echo "  # then on the H100, from the repo root:"
echo "  tar -xzf <staging>/kvq_missing.tgz -C \"\$ROOT\""
echo
echo "Anything under .venv*, requirements.lock.txt or pipelines/ is NOT a"
echo "transfer problem: those come from the venv build (steps 3-4) or the git"
echo "clone (step 1). Re-run those rather than copying files."
