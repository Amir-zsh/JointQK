#!/bin/bash
# Run ONE cell (or one shard of one cell): boot the server for an arm, verify
# the resolved config against the protocol, run the client with explicit
# sampling, write provenance, tear down. No scheduling — you decide where and
# when each cell runs; parallelism is invoking this script N times with
# disjoint --gpus.
#
#   bash pipelines/runpod/run_cell.sh \
#       --protocol pipelines/runpod/protocols/qwen3_8b_v1.json \
#       --arm vq2 --task niah_65536 --gpus 1 [--shard 0/4] [--port 30901] \
#       [--out-root artifacts/runpod]
#
# Sharding: --shard i/N slices the rows JSONL by line index modulo N (disjoint
# rids by construction) and writes to <cell>__s<i>, the layout
# pipelines/oscar_e2e/merge_shards.py merges and scores.
# Resume: a cell/shard whose metrics.json exists is skipped.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PROTOCOL="" ARM="" TASK="" GPUS="" SHARD="" PORT="" OUT_ROOT="artifacts/runpod"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --protocol) PROTOCOL="$2"; shift 2 ;;
        --arm)      ARM="$2"; shift 2 ;;
        --task)     TASK="$2"; shift 2 ;;
        --gpus)     GPUS="$2"; shift 2 ;;
        --shard)    SHARD="$2"; shift 2 ;;
        --port)     PORT="$2"; shift 2 ;;
        --out-root) OUT_ROOT="$2"; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done
[[ -n "$PROTOCOL" && -n "$ARM" && -n "$TASK" && -n "$GPUS" ]] \
    || { echo "required: --protocol --arm --task --gpus" >&2; exit 1; }

CLIENT_PY="${CLIENT_PYTHON:-$ROOT/.venv/bin/python}"

# Provenance SHAs resolved up front: a cell that cannot be attributed to two
# exact commits must not run (and must fail before GPU time, not after).
REPO_SHA="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)"
ENGINE_SHA="$(git -C "$ROOT/vendor/OSCAR-vq" rev-parse HEAD 2>/dev/null || true)"
[[ -n "$REPO_SHA" && -n "$ENGINE_SHA" ]] \
    || { echo "cannot resolve git SHAs — run bootstrap.sh first (safe.directory)" >&2; exit 1; }
REPO_DIRTY=$([[ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ]] && echo True || echo False)
ENGINE_DIRTY=$([[ -n "$(git -C "$ROOT/vendor/OSCAR-vq" status --porcelain 2>/dev/null)" ]] && echo True || echo False)

# ---- resolve every knob from the protocol (single source of truth) ---------
eval "$("$CLIENT_PY" - "$PROTOCOL" "$ARM" "$TASK" <<'PY'
import json, shlex, sys
p = json.load(open(sys.argv[1])); arm = p["arms"][sys.argv[2]]; task = p["tasks"][sys.argv[3]]
q = lambda k, v: print(f"{k}={shlex.quote(str(v))}")
q("P_NAME", p["protocol"]); q("P_MODEL", p["model"]); q("P_TP", p["tp"])
q("P_CTX", p["ctx"]); q("P_MEM_FRAC", p["mem_frac"]); q("P_POOL", p["max_tokens_pool"])
q("P_CHUNK", p["chunked_prefill_size"]); q("P_EXACT", int(p["exact_chunked_prefill"]))
q("P_GRAPH_BS", p["cuda_graph_max_bs"]); q("P_ROT_DIR", p["rot_dir"]); q("P_CB", p["vq_codebook"])
# Hybrid-SWA models (gpt-oss) need all three; dense models (Qwen) omit them and
# get serve_oscar.sh's own defaults (group=128, absorb=1, no MoE backend pin).
q("P_MOE_BACKEND", p.get("moe_runner_backend", ""))
q("P_QGS", p.get("quant_group_size", 128))
q("P_ABSORB_V", int(p.get("absorb_v_rotation", 1)))
# vq2-arm-only: the CUDA stage-1 decode kernel (opt-in, supports() falls back
# to Triton on any geometry/dtype mismatch it doesn't recognize).
q("P_VQ2_CUDA_GEOM", p.get("vq2_cuda_geom", ""))
q("A_MODE_FLAG", arm["mode_flag"]); q("A_MAX_REQS", arm["max_reqs"]); q("A_KV_SPLITS", arm["kv_splits"])
q("A_SCALE_DTYPE", arm.get("scale_dtype", ""))
# int2plain-arm-only (Naive/QuaRot/TurboQuant baselines): fixed Hadamard order
# (1=naive, 128=quarot) and no-clip K/V ratios (1.0). Empty = serve_oscar.sh's
# own defaults (unused by bf16/oscar_int2/vq2, which never hit the int2plain
# branch).
q("A_HADAMARD_ORDER", arm.get("hadamard_order", ""))
q("A_K_CLIP_RATIO", arm.get("k_clip_ratio", ""))
q("A_V_CLIP_RATIO", arm.get("v_clip_ratio", ""))
q("T_ROWS", task["rows"]); q("T_SAMPLES", task["samples"]); q("T_TEMP", task["temperature"])
q("T_TOP_P", task.get("top_p", "")); q("T_TOP_K", task.get("top_k", ""))
q("T_LIMIT", task.get("limit_rows", ""))
PY
)"

FIRST_GPU="${GPUS%%,*}"
PORT="${PORT:-$((30900 + FIRST_GPU))}"
CELL="$OUT_ROOT/$P_NAME/$ARM/$TASK"
[[ -n "$SHARD" ]] && CELL="${CELL}__s${SHARD%%/*}"
mkdir -p "$CELL" logs
SHARD_SUFFIX=""
[[ -n "$SHARD" ]] && SHARD_SUFFIX="_s${SHARD%%/*}"
LOG="$ROOT/logs/runpod_${ARM}_${TASK}${SHARD_SUFFIX}.log"
SERVE_LOG="${LOG%.log}_serve.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

if [[ -f "$CELL/metrics.json" ]]; then
    log "SKIP — $CELL/metrics.json exists (resume guard)"
    exit 0
fi

# ---- rows: shard slice / smoke limit (balanced per task when field exists) --
ROWS_FILE="$ROOT/$T_ROWS"
if [[ -n "$SHARD" || -n "$T_LIMIT" ]]; then
    ROWS_FILE="$CELL/rows.jsonl"
    "$CLIENT_PY" - "$ROOT/$T_ROWS" "$ROWS_FILE" "${SHARD:-}" "${T_LIMIT:-}" <<'PY'
import collections, json, sys
src, dst, shard, limit = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
lines = open(src).read().splitlines()
if shard:
    i, n = map(int, shard.split("/"))
    lines = lines[i::n]
if limit:
    per_task, seen, kept = collections.defaultdict(list), collections.Counter(), []
    tasks = {json.loads(l).get("task", "_") for l in lines}
    per = max(1, int(limit) // len(tasks))
    for l in lines:
        t = json.loads(l).get("task", "_")
        if seen[t] < per:
            seen[t] += 1; kept.append(l)
    lines = kept
open(dst, "w").write("\n".join(lines) + "\n")
print(f"[rows] {len(lines)} rows -> {dst}", file=sys.stderr)
PY
fi

# ---- boot the server -------------------------------------------------------
teardown() {
    [[ -n "${SERVER_PID:-}" ]] && kill -9 -- -"$SERVER_PID" 2>/dev/null || true
    sleep 3
}
trap teardown EXIT

SERVE_ARGS=(--gpu "$GPUS" --port "$PORT" --model "$P_MODEL"
            --ctx "$P_CTX" --mem-frac "$P_MEM_FRAC")
[[ -n "$A_MODE_FLAG" ]] && SERVE_ARGS+=("$A_MODE_FLAG")
[[ "$A_MODE_FLAG" == "--vq2" ]] && SERVE_ARGS+=(--vq-codebook "$P_CB")

CHUNK_EXTRA="--chunked-prefill-size $P_CHUNK"
[[ -n "$P_MOE_BACKEND" ]] && CHUNK_EXTRA="--moe-runner-backend $P_MOE_BACKEND $CHUNK_EXTRA"

SERVE_ENV=(TP="$P_TP" MAX_REQS="$A_MAX_REQS" KV_SPLITS="$A_KV_SPLITS"
           CUDA_GRAPH_BS="$P_GRAPH_BS" MAX_TOKENS="$P_POOL"
           ROT_DIR="$ROOT/$P_ROT_DIR"
           QUANT_GROUP_SIZE="$P_QGS" ABSORB_V_ROT="$P_ABSORB_V"
           SERVE_EXTRA="$CHUNK_EXTRA")
[[ -n "$A_SCALE_DTYPE" ]] && SERVE_ENV+=(SCALE_DTYPE="$A_SCALE_DTYPE")
[[ "$P_EXACT" == "1" ]] && SERVE_ENV+=(SGLANG_MIXED_KV_EXACT_CHUNKED_PREFILL=1)
[[ -n "$A_HADAMARD_ORDER" ]] && SERVE_ENV+=(HADAMARD_ORDER="$A_HADAMARD_ORDER")
[[ -n "$A_K_CLIP_RATIO" ]] && SERVE_ENV+=(SGLANG_OSCAR_K_CLIP_RATIO="$A_K_CLIP_RATIO")
[[ -n "$A_V_CLIP_RATIO" ]] && SERVE_ENV+=(SGLANG_OSCAR_V_CLIP_RATIO="$A_V_CLIP_RATIO")
if [[ "$A_MODE_FLAG" == "--vq2" && -n "$P_VQ2_CUDA_GEOM" ]]; then
    SERVE_ENV+=(SGLANG_VQ2_CUDA=1 "SGLANG_VQ2_CUDA_GEOM=$P_VQ2_CUDA_GEOM" SGLANG_VQ2_CUDA_FP32=1)
fi

log "cell=$CELL arm=$ARM task=$TASK gpus=$GPUS tp=$P_TP port=$PORT shard=${SHARD:-none}"
env "${SERVE_ENV[@]}" setsid bash pipelines/oscar_e2e/serve_oscar.sh \
    "${SERVE_ARGS[@]}" > "$SERVE_LOG" 2>&1 < /dev/null &
SERVER_PID=$!

for i in $(seq 1 180); do
    kill -0 "$SERVER_PID" 2>/dev/null \
        || { log "SERVER DIED — tail of $SERVE_LOG:"; tail -20 "$SERVE_LOG" | tee -a "$LOG"; exit 1; }
    [[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/get_model_info")" == "200" ]] && break
    sleep 10
    [[ "$i" == "180" ]] && { log "SERVER BOOT TIMEOUT"; exit 1; }
done
log "server up"

# ---- echo gate: the resolved config must match the protocol ----------------
EXPECT=$("$CLIENT_PY" -c "
import json
expect = {
    'context_length': $P_CTX,
    'chunked_prefill_size': $P_CHUNK,
    'mem_fraction_static': float('$P_MEM_FRAC'),
    'max_running_requests': $A_MAX_REQS,
    'triton_attention_num_kv_splits': $A_KV_SPLITS,
    'cuda_graph_max_bs': $P_GRAPH_BS,
    'sampling_backend': 'pytorch',
    'disable_radix_cache': True,
}
if '$A_MODE_FLAG' != '--bf16':
    # bf16 never sets --kv-cache-quant-group-size, so the server has no
    # opinion on it; only int2/vq2 arms resolve this field.
    expect['kv_cache_quant_group_size'] = None if '$P_QGS' == '0' else int('$P_QGS')
if '$P_MOE_BACKEND':
    expect['moe_runner_backend'] = '$P_MOE_BACKEND'
print(json.dumps(expect))")
"$CLIENT_PY" pipelines/runpod/echo_gate.py --port "$PORT" --expect "$EXPECT" \
    --out "$CELL/resolved_server_info.json" | tee -a "$LOG"

# ---- client (sampling always explicit — the client's own defaults are the
#      legacy trap and must never be reachable from here) --------------------
CLIENT_ARGS=(--rows "$ROWS_FILE" --port "$PORT" --out "$CELL"
             --samples "$T_SAMPLES" --temperature "$T_TEMP"
             --threads "$A_MAX_REQS" --timeout 3600)
if [[ -n "$T_TOP_P" ]]; then CLIENT_ARGS+=(--top-p "$T_TOP_P" --top-k "$T_TOP_K"); fi

log "client start ($T_SAMPLES samples, T=$T_TEMP)"
PYTHONPATH="$ROOT" "$CLIENT_PY" pipelines/oscar_e2e/run_prompts_client.py \
    "${CLIENT_ARGS[@]}" 2>&1 | tee -a "$LOG" | tail -5

# ---- provenance ------------------------------------------------------------
"$CLIENT_PY" - "$CELL" "$PROTOCOL" <<PY
import hashlib, json, subprocess, sys
from pathlib import Path
cell, proto = Path(sys.argv[1]), sys.argv[2]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
    capture_output=True, text=True).stdout.strip().splitlines()
prov = {
    "protocol": {"name": "$P_NAME", "file": proto, "sha256": sha(proto)},
    "arm": "$ARM", "task": "$TASK", "shard": "${SHARD:-}" or None,
    "repo": {"sha": "$REPO_SHA", "dirty": $REPO_DIRTY},
    "engine": {"sha": "$ENGINE_SHA", "dirty": $ENGINE_DIRTY},
    "rows": {"file": "$T_ROWS", "sha256": sha("$ROOT/$T_ROWS")},
    "sampling": {"samples": $T_SAMPLES, "temperature": $T_TEMP,
                 "top_p": "$T_TOP_P" or None, "top_k": "$T_TOP_K" or None},
    "serve": {"tp": $P_TP, "gpus": "$GPUS",
              "exact_chunked_prefill": bool($P_EXACT),
              "scale_dtype": "$A_SCALE_DTYPE" or None,
              "quant_group_size": "$P_QGS", "absorb_v_rotation": bool($P_ABSORB_V),
              "moe_runner_backend": "$P_MOE_BACKEND" or None,
              "vq2_cuda_geom": "$P_VQ2_CUDA_GEOM" or None},
    "gpu": gpu[0] if gpu else None,
    "image_tag": "${IMAGE_TAG:-}" or None,
}
if "$A_MODE_FLAG" != "--bf16":
    prov["rotations"] = {p.name: sha(p) for p in Path("$ROOT/$P_ROT_DIR").glob("*.pt")}
if "$A_MODE_FLAG" == "--vq2":
    prov["vq_codebook"] = {"file": "$P_CB", "sha256": sha("$ROOT/$P_CB")}
(cell / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
print(f"[provenance] {cell}/provenance.json")
PY

log "CELL DONE — $CELL"
