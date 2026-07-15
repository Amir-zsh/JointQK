#!/bin/bash
# pgq10 long-context battleground: RULER at ctx ∈ {8192,16384,32768,65536}.
# Same worker/protocol as launch_pgq_longbench.sh (Mode A, layer-0 fp16,
# V = v_turboquant @ 2 bits, fraction configurable); dataset=ruler with
# data_dir=<ctx>. Cells are "<kind>:<rate>" with the same kind grammar as the
# LongBench launcher plus:
#   fp16:0        uncompressed control through the same press path
#                 (quantize_k = quantize_v = False)
#
#   bash pipelines/bench/launch_pgq_ruler.sh --model-tag qwen3_8b \
#       --cells fp16:0,pgq_vqgb_flat:2.0 --ctxs 8192,16384,32768,65536
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPUS="${GPUS:-4,5,6}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
MAX_RETRIES="${MAX_RETRIES:-10}"
FRACTION="${EVAL_FRACTION:-1.0}"
CELLS=""
CTXS_CSV="${CTXS_CSV:-8192,16384,32768,65536}"
MODEL_TAG="${MODEL_TAG:-qwen3_8b}"
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cells) CELLS="$2"; shift 2 ;;
        --ctxs) CTXS_CSV="$2"; shift 2 ;;
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
PGQ4_BUNDLE="${PGQ4_BUNDLE:-$PGQ4_BUNDLE_DEFAULT}"
PGQ8_BUNDLE="${PGQ8_BUNDLE:-$PGQ8_BUNDLE_DEFAULT}"
PGQ3_BUNDLE="${PGQ3_BUNDLE:-${REPO_ROOT}/artifacts/page_quant2/pgq3_bundle__qpca_unc__compact8train60r400.pt}"
VQG_BUNDLE="${VQG_BUNDLE:-${REPO_ROOT}/artifacts/page_quant2/vqg_bundle__${MODEL_TAG}_flat.pt}"
OUT_BASE="${REPO_ROOT}/artifacts/bench_pgq_ruler/${MODEL_TAG}"
LOG_DIR="${REPO_ROOT}/logs/bench_pgq_ruler_${MODEL_TAG}"

for f in "$CCA" "$VST"; do
    [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done

IFS=',' read -ra CTXS <<< "$CTXS_CSV"
mkdir -p "$OUT_BASE" "$LOG_DIR"
STAMP=$(date '+%H%M%S')
CMDS="$LOG_DIR/commands_${STAMP}.jsonl"
: > "$CMDS"

sha8() { sha256sum "$1" | cut -c1-8; }

IFS=',' read -ra CELL_ARR <<< "$CELLS"
for cell in "${CELL_ARR[@]}"; do
    kind="${cell%%:*}"; rate="${cell##*:}"
    BUNDLE=""
    if [[ "$kind" == "fp16" ]]; then
        K_METHOD="fp16"; K_BITS=0; SHA="none"
    else
        if [[ "$kind" == pgq_vqg* ]]; then
            BUNDLE="$VQG_BUNDLE"
        elif [[ "$kind" == pgq_dct* ]]; then
            BUNDLE="$PGQ8_BUNDLE"
        elif [[ "$kind" == pgq_tcq_* || "$kind" == pgq_e8_* || "$kind" == pgq_oscar_* ]]; then
            BUNDLE="$PGQ3_BUNDLE"
        elif [[ "$kind" == pgq_fold* || "$kind" == pgq_prof* || "$kind" == pgq_mrg* ]]; then
            BUNDLE="$PGQ4_BUNDLE"
        else
            echo "ERROR: unknown kind $kind" >&2; exit 1
        fi
        [[ -f "$BUNDLE" ]] || { echo "ERROR: missing $BUNDLE" >&2; exit 1; }
        K_METHOD="$kind"; K_BITS="$rate"
        SHA=$(sha8 "$BUNDLE")
    fi
    for ctx in "${CTXS[@]}"; do
        label="pgqr__${kind}__b${rate}__${SHA}__ctx${ctx}"
        [[ "$FRACTION" == "1.0" ]] || label="${label}__f${FRACTION}"
        BUNDLE="$BUNDLE" K_METHOD="$K_METHOD" K_BITS="$K_BITS" ctx="$ctx" \
        label="$label" CCA="$CCA" VST="$VST" FRACTION="$FRACTION" \
        OUT_BASE="$OUT_BASE" \
        .venv/bin/python - <<'PY' >> "$CMDS"
import json, os
e = os.environ
fp16 = e["K_METHOD"] == "fp16"
kw = {
    "cca_stats_path": e["CCA"],
    "v_stats_path": e["VST"],
    "k_method": "r_sym_uniform" if fp16 else e["K_METHOD"],
    "v_method": "v_turboquant",
    "k_bits": 0.0 if fp16 else float(e["K_BITS"]),
    "v_bits": 2,
    "compress_decode": False,
    "layer0_full_precision": True,
    "quantize_k": not fp16,
    "quantize_v": not fp16,
}
if not fp16:
    kw["ec_bundle_path"] = e["BUNDLE"]
print(json.dumps({
    "_label": e["label"],
    "press_name": "jointqk",
    "press_kwargs": kw,
    "dataset": "ruler",
    "data_dir": e["ctx"],
    "fraction": float(e["FRACTION"]),
    "output_dir": f'{e["OUT_BASE"]}/{e["label"]}',
}))
PY
    done
done

n_jobs=$(wc -l < "$CMDS")
echo "[$(date '+%H:%M:%S')] queued $n_jobs RULER cells in $CMDS (cells: $CELLS, ctxs: $CTXS_CSV, fraction: $FRACTION)"

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
