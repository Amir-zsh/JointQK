#!/bin/bash
# v7 sweep recovery loop — runs after the main 2-jobs/GPU launcher finishes.
# Counts completed cells via metrics.json on disk. If <192, reruns the launcher
# at progressively lower concurrency. worker.py's skip-if-exists guarantees
# only the failed cells re-execute.
#
# Order:
#   pass 1 (already running externally): 2 jobs/GPU, max-retries 10
#   pass 2: 1 job/GPU, max-retries 10  ← OOMs that survived 2/GPU usually fit at 1/GPU
#   pass 3: 1 job/GPU, max-retries 5   ← last-chance, more conservative
# Stops early when all 192 cells are present.

set -uo pipefail
REPO=/vault/amir/efficient-llm/teamily-project
cd "$REPO"

MODEL_TAG=qwen3_8b
GPUS=0,1,2,3,4,5
OUT_BASE="$REPO/artifacts/stage1/downstream_v7/$MODEL_TAG"
LOG_DIR="$REPO/experiments/stage1/logs/phase7_v7_$MODEL_TAG"
RECOVERY_LOG="$LOG_DIR/_recovery.log"
EXPECTED_TOTAL=192

mkdir -p "$LOG_DIR"
exec >> "$RECOVERY_LOG" 2>&1

ts() { date '+%Y-%m-%d %H:%M:%S'; }

count_done() {
    # Count output dirs that contain a non-empty metrics.json (terminal-success marker).
    find "$OUT_BASE" -type f -name 'metrics.json' -size +0c 2>/dev/null | wc -l
}

echo "[$(ts)] === v7 recovery loop started ==="
echo "[$(ts)] OUT_BASE=$OUT_BASE"
echo "[$(ts)] LOG_DIR=$LOG_DIR"
echo "[$(ts)] EXPECTED_TOTAL=$EXPECTED_TOTAL"

# Wait for the main launcher to exit before doing anything.
echo "[$(ts)] waiting for main launcher (worker.py) to finish..."
while pgrep -f 'launch_v7|worker.py.*phase7_v7_qwen3_8b' > /dev/null; do
    sleep 60
done
echo "[$(ts)] main launcher finished. Beginning recovery passes."

PASS=2
for cfg in "1 10" "1 5"; do
    set -- $cfg
    JPG=$1
    RETRIES=$2

    done_count=$(count_done)
    echo "[$(ts)] before pass $PASS: $done_count / $EXPECTED_TOTAL cells done."
    if (( done_count >= EXPECTED_TOTAL )); then
        echo "[$(ts)] all cells complete — recovery loop exiting."
        exit 0
    fi

    echo "[$(ts)] launching recovery pass $PASS: --jobs-per-gpu $JPG --max-retries $RETRIES"
    bash experiments/stage1/bench/launch_v7.sh \
        --model "$MODEL_TAG" \
        --gpus "$GPUS" \
        --jobs-per-gpu "$JPG" \
        --max-retries "$RETRIES" \
        || echo "[$(ts)] WARN: pass $PASS launcher exited non-zero"

    after=$(count_done)
    echo "[$(ts)] after pass $PASS: $after / $EXPECTED_TOTAL cells done (Δ=$((after - done_count)))"
    if (( after >= EXPECTED_TOTAL )); then
        echo "[$(ts)] all cells complete — recovery loop exiting after pass $PASS."
        exit 0
    fi

    PASS=$((PASS + 1))
done

final=$(count_done)
echo "[$(ts)] === recovery loop exhausted. Final: $final / $EXPECTED_TOTAL cells done. ==="
if (( final < EXPECTED_TOTAL )); then
    # Identify which cells are still missing for the human to investigate in the morning.
    MISSING=$(.venv/bin/python <<'PY'
import json
from pathlib import Path
out_base = Path("artifacts/stage1/downstream_v7/qwen3_8b")
cmds = Path("experiments/stage1/logs/phase7_v7_qwen3_8b/commands.jsonl").read_text().splitlines()
missing = []
for line in cmds:
    if not line.strip(): continue
    cfg = json.loads(line)
    out_dir = Path(cfg["output_dir"])
    sub = next((p for p in out_dir.iterdir() if p.is_dir()), None) if out_dir.exists() else None
    has_metrics = bool(sub and (sub / "metrics.json").exists())
    if not has_metrics:
        missing.append(cfg["_label"])
for m in missing:
    print(m)
PY
)
    echo "[$(ts)] MISSING cells:"
    echo "$MISSING" | sed 's/^/  /'
fi
echo "[$(ts)] recovery log saved to $RECOVERY_LOG"
