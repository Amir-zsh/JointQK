#!/bin/bash
# Llama-3.1-8B experiment grid, one arm per invocation (Samuel's Qwen grid
# shape adapted to a non-thinking model):
#   NIAH 8/16/32/64K served greedy (800 rows each)
#   GPQA-198 / MATH500-full / AIME25 / HumanEval at K=5 seeds,
#   T=0.6 / top-p 0.9 / no top-k (Llama-3.1 generation_config defaults —
#   identical across arms, which is what the paired deltas need).
#
#   bash pipelines/oscar_e2e/run_llama_grid.sh --arm {bf16|int2|vq2} \
#       --gpu N --port P [--reuse-server]
#
# int2/vq2 use the Llama rotations (artifacts/oscar_llama31_8b/rotations)
# and the fresh ptn/64K codebook. Resumable: tasks with metrics.json skip;
# the client itself resumes partial cells.
set -u
ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"

ARM=""; GPU=0; PORT=30840; SAMPLES=5; REUSE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --arm) ARM="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --samples) SAMPLES="$2"; shift 2 ;;
        --reuse-server) REUSE=1; shift ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done
[[ -n "$ARM" ]] || { echo "--arm required" >&2; exit 1; }

MODEL="meta-llama/Llama-3.1-8B-Instruct"
export ROT_DIR="$ROOT/artifacts/oscar_llama31_8b/rotations"
LLAMA_CB="artifacts/oscar_llama31_8b/vqa_llama31_8b_G4_strat_flat_ptn_gpqacc64k_fp8.pt"
LOG="logs/llama_grid_${ARM}.log"; HB="logs/llama_grid_${ARM}.heartbeat"
OUTBASE="artifacts/oscar_llama31_8b/grid/${ARM}"
mkdir -p logs "$OUTBASE"
exec >>"$LOG" 2>&1
echo "=== $(date -Is) llama grid start arm=$ARM gpu=$GPU port=$PORT K=$SAMPLES"

SERVER_PID=""
if [[ "$REUSE" != "1" ]]; then
    SERVE_ARGS=(--model "$MODEL" --gpu "$GPU" --port "$PORT")
    case "$ARM" in
        bf16) SERVE_ARGS+=(--bf16) ;;
        int2) ;;
        vq2)  SERVE_ARGS+=(--vq2 --vq-codebook "$LLAMA_CB") ;;
        *) echo "unknown arm $ARM"; exit 1 ;;
    esac
    BOOTLOG="logs/llama_grid_${ARM}_server.log"
    nohup bash pipelines/oscar_e2e/serve_oscar.sh "${SERVE_ARGS[@]}" >"$BOOTLOG" 2>&1 &
    SERVER_PID=$!
    for i in $(seq 1 120); do
        grep -q "The server is fired up and ready to roll" "$BOOTLOG" && break
        grep -qE "Received sigquit|CUDA out of memory|Not enough memory" "$BOOTLOG" && { echo "BOOT FAILED"; tail -25 "$BOOTLOG"; exit 2; }
        sleep 5; touch "$HB"
    done
    grep -q "The server is fired up" "$BOOTLOG" || { echo "BOOT TIMEOUT"; exit 2; }
    echo "$(date -Is) server ready pid=$SERVER_PID"
fi

run_cell () { # name rows extra...
    local name="$1" rows="$2"; shift 2
    local out="$OUTBASE/$name"
    if [[ -f "$out/metrics.json" ]]; then
        echo "$(date -Is) SKIP $name"; return 0
    fi
    echo "=== $(date -Is) cell $name"
    touch "$HB"
    .venv/bin/python pipelines/oscar_e2e/run_prompts_client.py \
        --rows "$rows" --port "$PORT" --timeout 3600 "$@" --out "$out"
    local rc=$?
    echo "$(date -Is) cell $name rc=$rc"; touch "$HB"
    return $rc
}

FAIL=0
# NIAH sweep, greedy. 64K runs at threads 2 (pool fits ~2 concurrent 65K
# prompts); shorter contexts at 8.
for ctx in 8192 16384 32768; do
    run_cell "niah_${ctx}" "artifacts/prompt_rows/niah_${ctx}_llama.jsonl" --threads 8 || FAIL=1
done
run_cell "niah_65536" "artifacts/prompt_rows/niah_65536_llama.jsonl" --threads 2 || FAIL=1

# Reasoning/code at K=5 seeds, Llama generation defaults. Caps are
# non-thinking sized (2048/4096), so 6 threads is safely inside the pool.
SAMP=(--samples "$SAMPLES" --temperature 0.6 --top-p 0.9 --top-k -1 --threads 6)
run_cell gpqa      artifacts/prompt_rows/gpqa_diamond_llama.jsonl "${SAMP[@]}" || FAIL=1
run_cell math500   artifacts/prompt_rows/math500_llama.jsonl      "${SAMP[@]}" || FAIL=1
run_cell aime25    artifacts/prompt_rows/aime25_llama.jsonl       "${SAMP[@]}" || FAIL=1
run_cell humaneval artifacts/prompt_rows_code/humaneval_llama.jsonl "${SAMP[@]}" || FAIL=1

if [[ -n "$SERVER_PID" ]]; then
    echo "$(date -Is) killing server pid=$SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null; sleep 5
    PIDS=$(lsof -t -i :"$PORT" 2>/dev/null); [[ -n "$PIDS" ]] && kill $PIDS 2>/dev/null
fi
echo "=== $(date -Is) llama grid done arm=$ARM FAIL=$FAIL"
exit $FAIL
