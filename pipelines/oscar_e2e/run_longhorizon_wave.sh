#!/bin/bash
# plan11 C1: one long-horizon arm end-to-end on one GPU.
#   bash pipelines/oscar_e2e/run_longhorizon_wave.sh --arm vq2 --gpu 1 --port 30801 \
#       [--samples 4] [--reuse-server] [--vq-codebook <path>]
#
# Boots the arm's server (serve_oscar.sh, clone stack), waits for ready, runs
# gpqa-diamond(198)@8192 + math500(200)@4096 + aime25(30)@16384 with K
# samples each (OSCAR driver sampling: T=1.0/top_p=0.95/top_k=40), then kills
# the server by PID. Logs stream to logs/lh_<arm>.log with a heartbeat.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

ARM=""; GPU=5; PORT=30800; SAMPLES=4; REUSE=0; VQCB=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --arm) ARM="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --samples) SAMPLES="$2"; shift 2 ;;
        --vq-codebook) VQCB="$2"; shift 2 ;;
        --reuse-server) REUSE=1; shift ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done
[[ -n "$ARM" ]] || { echo "--arm required (bf16|int2|vq2)" >&2; exit 1; }

LOG="logs/lh_${ARM}.log"
HB="logs/lh_${ARM}.heartbeat"
OUTBASE="artifacts/oscar_e2e/lh/${ARM}"
mkdir -p logs "$OUTBASE"
exec >>"$LOG" 2>&1
echo "=== $(date -Is) wave start arm=$ARM gpu=$GPU port=$PORT K=$SAMPLES"

SERVER_PID=""
if [[ "$REUSE" != "1" ]]; then
    SERVE_ARGS=(--gpu "$GPU" --port "$PORT")
    case "$ARM" in
        bf16) SERVE_ARGS+=(--bf16) ;;
        int2) ;;
        vq2)  SERVE_ARGS+=(--vq2); [[ -n "$VQCB" ]] && SERVE_ARGS+=(--vq-codebook "$VQCB") ;;
        *) echo "unknown arm $ARM"; exit 1 ;;
    esac
    BOOTLOG="logs/lh_${ARM}_server.log"
    nohup bash pipelines/oscar_e2e/serve_oscar.sh "${SERVE_ARGS[@]}" >"$BOOTLOG" 2>&1 &
    SERVER_PID=$!
    echo "server pid=$SERVER_PID bootlog=$BOOTLOG"
    for i in $(seq 1 120); do
        grep -q "The server is fired up and ready to roll" "$BOOTLOG" && break
        grep -qE "Received sigquit|CUDA out of memory" "$BOOTLOG" && { echo "BOOT FAILED"; tail -30 "$BOOTLOG"; exit 2; }
        sleep 5; touch "$HB"
    done
    grep -q "The server is fired up" "$BOOTLOG" || { echo "BOOT TIMEOUT"; exit 2; }
    echo "$(date -Is) server ready"
fi

run_task () {
    local name="$1" rows="$2" extra="$3"
    local out="$OUTBASE/$name"
    if [[ -f "$out/metrics.json" ]]; then
        echo "$(date -Is) SKIP $name (metrics.json exists)"; return 0
    fi
    echo "=== $(date -Is) task $name rows=$rows"
    touch "$HB"
    # shellcheck disable=SC2086
    .venv/bin/python pipelines/oscar_e2e/run_prompts_client.py \
        --rows "$rows" --port "$PORT" --threads 6 --timeout 3600 \
        --samples "$SAMPLES" $extra --out "$out"
    local rc=$?
    echo "$(date -Is) task $name rc=$rc"
    touch "$HB"
    return $rc
}

# math500: first 200 rows (identical subset across arms).
M500="artifacts/prompt_rows/math500_think_qwen_n200.jsonl"
[[ -f "$M500" ]] || head -200 artifacts/prompt_rows/math500_think_qwen.jsonl > "$M500"

FAIL=0
run_task gpqa    artifacts/prompt_rows/gpqa_diamond_think_qwen.jsonl "" || FAIL=1
run_task math500 "$M500" ""                                            || FAIL=1
run_task aime25  artifacts/prompt_rows/aime25_think_qwen.jsonl ""      || FAIL=1

if [[ -n "$SERVER_PID" ]]; then
    echo "$(date -Is) killing server pid=$SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null
    sleep 5
    # serve_oscar.sh execs python; kill any child still holding the port.
    PIDS=$(lsof -t -i :"$PORT" 2>/dev/null)
    [[ -n "$PIDS" ]] && kill $PIDS 2>/dev/null
fi
echo "=== $(date -Is) wave done arm=$ARM FAIL=$FAIL"
exit $FAIL
