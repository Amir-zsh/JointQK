#!/bin/bash
# Overnight autonomous queue runner for the gptoss20b_mxfp4_v1 sweep
# (gpqa remainder, lcb, niah 8/16/32/64k, a clean vq2/math500 retry, and a
# gated niah_131072 ctx-extension smoke test).
#
# A SINGLE shared work queue (file-backed, flock-protected) is drained by two
# workers, one per GPU pair (0,1 and 2,3). Whichever GPU pair finishes its
# current cell first immediately grabs the next pending item -- no fixed
# per-lane split, so neither pair ever sits idle while work remains. A cell
# that fails (process exits without producing metrics.json) is re-pushed to
# the END of the shared queue, up to MAX_ATTEMPTS total, so one broken cell
# (e.g. the known vq2 inf/nan crash) never blocks the rest of the sweep.
set -u
cd /workspace/teamily-project
export OSCAR_PYTHON=/opt/venv-oscar/bin/python CLIENT_PYTHON=/opt/venv-client/bin/python HF_HOME=/workspace/hf
PROTO=pipelines/runpod/protocols/gptoss20b_mxfp4_v1.json
PROTO_CTX131K=pipelines/runpod/protocols/gptoss20b_mxfp4_v1_ctx131k.json
LOG=/workspace/overnight_queue.log
HB=/workspace/overnight_queue.heartbeat
CELLLOGDIR=/workspace/logs_overnight
QFILE=/workspace/overnight_queue.items
BFILE=/workspace/overnight_queue.busy
LOCKFILE=/workspace/overnight_queue.lock
MAX_ATTEMPTS=3
mkdir -p "$CELLLOGDIR"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; touch "$HB"; }

# ---- shared queue primitives (flock-protected, file-backed) ---------------
pop_and_mark_busy() {
    ( flock -x 200
      if [ -s "$QFILE" ]; then
          head -1 "$QFILE"
          tail -n +2 "$QFILE" > "${QFILE}.tmp" && mv "${QFILE}.tmp" "$QFILE"
          local b; b=$(cat "$BFILE"); echo $((b+1)) > "$BFILE"
      fi
    ) 200>"$LOCKFILE"
}
push_item() {
    ( flock -x 200; echo "$1" >> "$QFILE" ) 200>"$LOCKFILE"
}
mark_done() {
    ( flock -x 200
      local b; b=$(cat "$BFILE"); echo $((b-1)) > "$BFILE"
    ) 200>"$LOCKFILE"
}
get_busy() {
    ( flock -x 200; cat "$BFILE" ) 200>"$LOCKFILE"
}

cell_dir_for() {
    local proto="$1" arm="$2" task="$3"
    local pname
    pname=$("$CLIENT_PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['protocol'])" "$proto")
    echo "artifacts/runpod/$pname/$arm/$task"
}

wait_gpus_free() {
    local gpus="$1"
    local IFS=','
    local -a ids=($gpus)
    while true; do
        local busy=0
        for g in "${ids[@]}"; do
            local used
            used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
            [ -n "$used" ] && [ "$used" -gt 3000 ] && busy=1
        done
        [ "$busy" -eq 0 ] && return 0
        sleep 15
    done
}

run_cell_once() {
    local worker_name="$1" proto="$2" arm="$3" task="$4" gpus="$5" attempt="$6"
    local cell_dir cell_log
    cell_dir=$(cell_dir_for "$proto" "$arm" "$task")
    cell_log="$CELLLOGDIR/${worker_name}_${arm}_${task}_attempt${attempt}.log"
    log "[$worker_name] START $arm/$task (protocol=$(basename "$proto") attempt $attempt/$MAX_ATTEMPTS gpus=$gpus)"
    wait_gpus_free "$gpus"
    bash pipelines/runpod/run_cell.sh --protocol "$proto" --arm "$arm" --task "$task" --gpus "$gpus" \
        > "$cell_log" 2>&1
    if [ -f "$cell_dir/metrics.json" ]; then
        log "[$worker_name] DONE $arm/$task -> $cell_dir/metrics.json"
        return 0
    else
        log "[$worker_name] FAILED $arm/$task (attempt $attempt) -- see $cell_log"
        return 1
    fi
}

worker() {
    local worker_name="$1" gpus="$2"
    while true; do
        local item
        item=$(pop_and_mark_busy)
        if [ -z "$item" ]; then
            local b; b=$(get_busy)
            if [ "$b" -eq 0 ]; then
                log "[$worker_name] queue empty, nothing in-flight anywhere -- exiting"
                break
            fi
            sleep 15
            continue
        fi
        local n proto arm task
        IFS='|' read -r n proto arm task <<< "$item"
        local cell_dir
        cell_dir=$(cell_dir_for "$proto" "$arm" "$task")
        if [ -f "$cell_dir/metrics.json" ]; then
            log "[$worker_name] SKIP $arm/$task (already done)"
            mark_done
            continue
        fi
        if run_cell_once "$worker_name" "$proto" "$arm" "$task" "$gpus" "$n"; then
            if [ "$task" = "niah_131072_smoke" ]; then
                log "[$worker_name] niah_131072 smoke test PASSED -- queueing full 3-arm sweep at ctx131k"
                push_item "1|$PROTO_CTX131K|bf16|niah_131072"
                push_item "1|$PROTO_CTX131K|oscar_int2|niah_131072"
                push_item "1|$PROTO_CTX131K|vq2|niah_131072"
            fi
        else
            if [ "$task" = "niah_131072_smoke" ]; then
                log "[$worker_name] niah_131072 smoke test FAILED -- skipping the 131072 sweep entirely, not spending GPU-hours on an unvalidated context length"
            elif [ "$n" -lt "$MAX_ATTEMPTS" ]; then
                push_item "$((n+1))|$proto|$arm|$task"
                log "[$worker_name] requeued $arm/$task to end of shared queue (next attempt $((n+1)))"
            else
                log "[$worker_name] GIVING UP on $arm/$task after $MAX_ATTEMPTS attempts"
            fi
        fi
        mark_done
    done
}

# ---- seed the shared queue, or RESUME an existing one -----------------------
# If QFILE already has content (e.g. this is a relaunch after the pod died
# mid-sweep), items that were popped-but-in-flight when the process died are
# already gone from the file -- the caller is responsible for re-pushing any
# such items before invoking this script. We only fresh-seed when the queue
# file is empty/missing; we never blow away in-progress state.
: > "$LOCKFILE"
if [ -s "$QFILE" ]; then
    log "=== overnight queue RESUME (existing queue file has $(wc -l < "$QFILE") items) ==="
    echo 0 > "$BFILE"   # no worker is actually in-flight right after a fresh process start
else
    : > "$QFILE"
    echo 0 > "$BFILE"
    for item in \
        "1|$PROTO|vq2|math500" \
        "1|$PROTO|vq2|gpqa" \
        "1|$PROTO|bf16|lcb" "1|$PROTO|oscar_int2|lcb" "1|$PROTO|vq2|lcb" \
        "1|$PROTO|bf16|niah_8192"  "1|$PROTO|oscar_int2|niah_8192"  "1|$PROTO|vq2|niah_8192" \
        "1|$PROTO|bf16|niah_16384" "1|$PROTO|oscar_int2|niah_16384" "1|$PROTO|vq2|niah_16384" \
        "1|$PROTO|bf16|niah_32768" "1|$PROTO|oscar_int2|niah_32768" "1|$PROTO|vq2|niah_32768" \
        "1|$PROTO|bf16|niah_65536" "1|$PROTO|oscar_int2|niah_65536" "1|$PROTO|vq2|niah_65536" \
        "1|$PROTO_CTX131K|bf16|niah_131072_smoke" \
        ; do
        push_item "$item"
    done
    log "=== overnight queue start (shared queue, $(wc -l < "$QFILE") items) ==="
fi
worker workerA "0,1" &
PID_A=$!
worker workerB "2,3" &
PID_B=$!
wait "$PID_A" "$PID_B"
log "=== ALL WORK DONE — QUEUE EMPTY ==="
