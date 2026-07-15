#!/bin/bash
# page_quant bench: the decomposition chain on lcc / musique / 2wikimqa,
# v7-identical protocol (fraction=1.0, compact8 exclusions, layer0 fp16,
# Mode A, V = v_turboquant @ 2 bits). --model-tag llama31_8b (default) or
# qwen3_8b selects MODEL/CCA/VST/bundle/output roots (pgq5).
#
# Cells are "<kind>:<rate>" pairs:
#   ecu:1.0        ec_uniform control (qpca_unc uniform-step EC bundle @ that
#                  b_target; variable-rate stream, token-uniform)
#   pgq_pagerung:1.0  fixed pages, one rung per page (token-uniform)
#   pgq_plain:1.0     fixed pages, per-token RDO, omega=1
#   pgq_ea:1.0        fixed pages, per-token RDO, omega=ExpectedAttention(m)
#   pgq_fixed:1.0     fixed pages, per-token width, no rANS (serving variant)
#   pgq_fixed_ea:1.0  same + omega
#
# Labels embed the bundle file's sha8 so a refit mints new cell dirs (the
# worker's idempotent skip is mtime-blind — protocol-critic guard). Smoke
# runs (fraction < 1) additionally get their own output dirs.
#
#   bash pipelines/bench/launch_pgq_longbench.sh --cells ecu:1.5,ecu:1.0,ecu:0.75
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPUS="${GPUS:-0,1,2,3}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
MAX_RETRIES="${MAX_RETRIES:-10}"
FRACTION="${EVAL_FRACTION:-1.0}"
CELLS=""
TASKS_CSV="${TASKS_CSV:-lcc,musique,2wikimqa,qasper,hotpotqa}"
MODEL_TAG="${MODEL_TAG:-llama31_8b}"
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cells) CELLS="$2"; shift 2 ;;
        --tasks) TASKS_CSV="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --fraction) FRACTION="$2"; shift 2 ;;
        --jobs-per-gpu) JOBS_PER_GPU="$2"; shift 2 ;;
        --model-tag) MODEL_TAG="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done
[[ -n "$CELLS" ]] || { echo "ERROR: --cells required" >&2; exit 1; }

case "$MODEL_TAG" in
    llama31_8b)
        MODEL="meta-llama/Llama-3.1-8B-Instruct"
        CCA="${REPO_ROOT}/artifacts/bases/jointqk_llama31_8b_longbench_compact8_n400.pt"
        VST="${REPO_ROOT}/artifacts/v_bases/v_stats_llama31_8b_longbench_compact8_n400.pt"
        PGQ4_BUNDLE_DEFAULT="${REPO_ROOT}/artifacts/page_quant2/pgq4_bundle__3bases__compact8train40r400.pt"
        PGQ8_BUNDLE_DEFAULT="${REPO_ROOT}/artifacts/page_quant2/pgq8_bundle__llama31_8b.pt"
        ;;
    qwen3_8b)
        MODEL="Qwen/Qwen3-8B"
        CCA="${REPO_ROOT}/artifacts/bases/qpca_qwen3_8b_longbench_compact8_n400.pt"
        VST="${REPO_ROOT}/artifacts/v_bases/v_stats_longbench_compact8_n400.pt"
        PGQ4_BUNDLE_DEFAULT="${REPO_ROOT}/artifacts/page_quant2/pgq5_bundle__qpca_unc__qwen3_8b_compact8train12.pt"
        PGQ8_BUNDLE_DEFAULT="${REPO_ROOT}/artifacts/page_quant2/pgq8_bundle__qwen3_8b.pt"
        ;;
    *) echo "Unknown --model-tag: $MODEL_TAG" >&2; exit 1 ;;
esac
EXCLUDE_INDICES_FILE="${REPO_ROOT}/artifacts/calibration_splits/longbench_compact8_60_seed20260504_2k32k/exclude_train_indices_for_eval.json"
EC_DIR="${REPO_ROOT}/artifacts/ec/${MODEL_TAG}"
PGQ_BUNDLE="${REPO_ROOT}/artifacts/page_quant/pgq_bundle__qpca_unc__dz0.5__base1.5__compact8train18.pt"
PGQ2_BUNDLE="${PGQ2_BUNDLE:-${REPO_ROOT}/artifacts/page_quant2/pgq2_bundle__qpca_unc__compact8train18.pt}"
PGQ3_BUNDLE="${PGQ3_BUNDLE:-${REPO_ROOT}/artifacts/page_quant2/pgq3_bundle__qpca_unc__compact8train60r400.pt}"
PGQ4_BUNDLE="${PGQ4_BUNDLE:-$PGQ4_BUNDLE_DEFAULT}"
PGQ8_BUNDLE="${PGQ8_BUNDLE:-$PGQ8_BUNDLE_DEFAULT}"
VQG_BUNDLE="${VQG_BUNDLE:-${REPO_ROOT}/artifacts/page_quant2/vqg_bundle__${MODEL_TAG}_flat.pt}"
OUT_BASE="${REPO_ROOT}/artifacts/bench_pgq/${MODEL_TAG}"
LOG_DIR="${REPO_ROOT}/logs/bench_pgq_${MODEL_TAG}"

for f in "$CCA" "$VST" "$EXCLUDE_INDICES_FILE"; do
    [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done

IFS=',' read -ra TASKS <<< "$TASKS_CSV"
mkdir -p "$OUT_BASE" "$LOG_DIR"
STAMP=$(date '+%H%M%S')
CMDS="$LOG_DIR/commands_${STAMP}.jsonl"
: > "$CMDS"

sha8() { sha256sum "$1" | cut -c1-8; }

IFS=',' read -ra CELL_ARR <<< "$CELLS"
for cell in "${CELL_ARR[@]}"; do
    kind="${cell%%:*}"; rate="${cell##*:}"
    # Mode-B' decode cells: "<kind>+d<W>:<rate>" — decode tokens buffer fp16
    # and age out in chunks of 8 once more than W have accumulated
    DEC_W=""
    if [[ "$kind" == *"+d"* ]]; then
        DEC_W="${kind##*+d}"
        kind="${kind%%+*}"
    fi
    if [[ "$kind" == "ecu" ]]; then
        rate_g="${rate%.0}"   # fit_ec_bundle names use b{b:g}: 1.0 -> b1
        BUNDLE=$(ls "$EC_DIR"/ec_bundle__qpca_unc__b${rate_g}__dz0.5__compact8train*__uniform.pt 2>/dev/null | head -1 || true)
        [[ -n "$BUNDLE" ]] || { echo "ERROR: no ec_uniform bundle for b=$rate" >&2; exit 1; }
        K_METHOD="ec_qpca_unc"; K_BITS=2
    elif [[ "$kind" == pgq_nd_* || "$kind" == pgq_rvq_* ]]; then
        # pgq2 arms; for *_uni the value is the rung/stage index, else b/c
        BUNDLE="$PGQ2_BUNDLE"
        [[ -f "$BUNDLE" ]] || { echo "ERROR: missing $BUNDLE" >&2; exit 1; }
        K_METHOD="$kind"; K_BITS="$rate"
    elif [[ "$kind" == pgq_vqg* ]]; then
        # pgq10 arms: Samuel's group VQ (ported); rate baked into the codebook
        BUNDLE="$VQG_BUNDLE"
        [[ -f "$BUNDLE" ]] || { echo "ERROR: missing $BUNDLE" >&2; exit 1; }
        K_METHOD="$kind"; K_BITS="$rate"
    elif [[ "$kind" == pgq_dct* ]]; then
        # pgq8 arms: token-axis DCT — needs the dct_std-carrying bundle
        BUNDLE="$PGQ8_BUNDLE"
        [[ -f "$BUNDLE" ]] || { echo "ERROR: missing $BUNDLE" >&2; exit 1; }
        K_METHOD="$kind"; K_BITS="$rate"
    elif [[ "$kind" == pgq_fold* || "$kind" == pgq_prof* || "$kind" == pgq_mrg* ]]; then
        # pgq4 arms (folded-scalar / profile rungs, plan4) + pgq6 merge arms
        # (same bundle: clustering is runtime, no refit)
        BUNDLE="$PGQ4_BUNDLE"
        [[ -f "$BUNDLE" ]] || { echo "ERROR: missing $BUNDLE" >&2; exit 1; }
        K_METHOD="$kind"; K_BITS="$rate"
    elif [[ "$kind" == pgq_tcq_* || "$kind" == pgq_e8_* || "$kind" == pgq_oscar_* ]]; then
        # pgq3 arms (TCQ/E8/OSCAR); override PGQ3_BUNDLE for the scalar
        # compander control (__scalarctl bundle, same k_methods)
        BUNDLE="$PGQ3_BUNDLE"
        [[ -f "$BUNDLE" ]] || { echo "ERROR: missing $BUNDLE" >&2; exit 1; }
        K_METHOD="$kind"; K_BITS="$rate"
    else
        BUNDLE="$PGQ_BUNDLE"
        [[ -f "$BUNDLE" ]] || { echo "ERROR: missing $BUNDLE" >&2; exit 1; }
        K_METHOD="$kind"; K_BITS="$rate"
    fi
    SHA=$(sha8 "$BUNDLE")
    for task in "${TASKS[@]}"; do
        label="pgq__${kind}__b${rate}__${SHA}__${task}"
        [[ -z "$DEC_W" ]] || label="pgq__${kind}_d${DEC_W}__b${rate}__${SHA}__${task}"
        [[ "$FRACTION" == "1.0" ]] || label="${label}__f${FRACTION}"
        BUNDLE="$BUNDLE" K_METHOD="$K_METHOD" K_BITS="$K_BITS" task="$task" \
        label="$label" CCA="$CCA" VST="$VST" FRACTION="$FRACTION" \
        DEC_W="$DEC_W" \
        OUT_BASE="$OUT_BASE" EXCLUDE_INDICES_FILE="$EXCLUDE_INDICES_FILE" \
        .venv/bin/python - <<'PY' >> "$CMDS"
import json, os
e = os.environ
kw = {
    "cca_stats_path": e["CCA"],
    "v_stats_path": e["VST"],
    "k_method": e["K_METHOD"],
    "ec_bundle_path": e["BUNDLE"],
    "v_method": "v_turboquant",
    "k_bits": float(e["K_BITS"]),
    "v_bits": 2,
    "compress_decode": False,
    "layer0_full_precision": True,
    "quantize_k": True,
    "quantize_v": True,
}
if e.get("DEC_W"):
    kw.update({"compress_decode": True, "decode_chunk": 8,
               "decode_recent": int(e["DEC_W"])})
print(json.dumps({
    "_label": e["label"],
    "press_name": "jointqk",
    "press_kwargs": kw,
    "dataset": "longbench",
    "data_dir": e["task"],
    "fraction": float(e["FRACTION"]),
    "exclude_indices_file": e["EXCLUDE_INDICES_FILE"],
    "output_dir": f'{e["OUT_BASE"]}/{e["label"]}',
}))
PY
    done
done

n_jobs=$(wc -l < "$CMDS")
echo "[$(date '+%H:%M:%S')] queued $n_jobs pgq cells in $CMDS (cells: $CELLS, fraction: $FRACTION)"

if [[ "$DRY_RUN" -eq 1 ]]; then
    head -2 "$CMDS"
    echo "dry run — not executing"
    exit 0
fi

.venv/bin/python pipelines/bench/worker.py \
    --model "$MODEL" \
    --commands-file "$CMDS" \
    --log-dir "$LOG_DIR" \
    --gpus "$GPUS" \
    --jobs-per-gpu "$JOBS_PER_GPU" \
    --max-retries "$MAX_RETRIES"
