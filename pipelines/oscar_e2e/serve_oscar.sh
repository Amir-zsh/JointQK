#!/bin/bash
# Serve production OSCAR INT2-KV (authors' sglang-research stack, vendored) on
# one GPU. A100 note: the documented fa3-prefill path is Hopper-only; the
# triton/triton backend combo is the supported alternative accepted by
# _unified_mixed_kv_active (recorded caveat vs the paper's H100/fa3 numbers).
#
#   bash pipelines/oscar_e2e/serve_oscar.sh [--gpu 5] [--port 30800] [--bf16]
#
# --bf16 serves the SAME stack without INT2 (the serving-stack control arm).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU=5
PORT=30800
MODE=int2
CTX=73728          # RULER-64K + generation headroom; 131072 arenas OOM on A100-40GB
MEM_FRAC=0.78
# --model parameterizes BF16 for OSCAR's other model configs. int2/vq2 assume the
# Qwen3-8B rotzoo/codebook, so only override --model for --bf16 runs.
MODEL="${MODEL:-Qwen/Qwen3-8B}"
VQ_CODEBOOK="${VQ_CODEBOOK:-artifacts/page_quant2/vqg_bundle__qwen3_8b_flat_ptn.pt}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --bf16) MODE=bf16; shift ;;
        --vq2) MODE=vq2; shift ;;
        --int2plain) MODE=int2plain; shift ;;   # plain per-token int2, NO mixed-KV band; Hadamard via HADAMARD_ORDER (1=naive, 128=quarot)
        --model) MODEL="$2"; shift 2 ;;
        --vq-codebook) VQ_CODEBOOK="$2"; shift 2 ;;
        --ctx) CTX="$2"; shift 2 ;;
        --mem-frac) MEM_FRAC="$2"; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

# All modes run on the vq2-longhorizon clone (vendor/OSCAR-vq) so the three
# long-horizon arms share one engine stack; the vq2 additions are inert unless
# SGLANG_VQ_CODEBOOK_PATH is set. PYTHONPATH shadows the editable install of
# vendor/OSCAR in .venv-oscar.
export PYTHONPATH="$(cd "$(dirname "$0")/../.." && pwd)/vendor/OSCAR-vq/sglang-research/python:${PYTHONPATH:-}"

# ROT_DIR overridable so int2/vq2 can serve a non-8B model with its own OSCAR
# rotations (e.g. the Qwen3-4B build under JointQK/artifacts/oscar_4b/rotations).
ROT_DIR="${ROT_DIR:-$REPO_ROOT/artifacts/oscar_e2e/rotzoo/Qwen3-8B/seq20000_prompt83_group128}"
CUDA128="/vault/amir/.conda/envs/cuda128"

export CUDA_VISIBLE_DEVICES=$GPU
# Needed in BOTH modes: ctx 66560 exceeds Qwen3-8B's derived 40960 (RoPE
# extrapolation, matching the transformers-harness behaviour at 64K).
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export CUDA_HOME="$CUDA128"
export PATH="$CUDA128/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# System gcc is 9.4; their JIT kernels use C++20 <concepts> -> conda gcc-12.
GCC12="/vault/amir/.conda/envs/gcc12"
if [[ -x "$GCC12/bin/x86_64-conda-linux-gnu-g++" ]]; then
    export CC="$GCC12/bin/x86_64-conda-linux-gnu-gcc"
    export CXX="$GCC12/bin/x86_64-conda-linux-gnu-g++"
    export CUDAHOSTCXX="$CXX"
    export PATH="$GCC12/bin:$PATH"
    # nvcc ignores $CXX and host-compiles with /usr/bin/c++ (gcc 9.4, no
    # <concepts>) unless -ccbin is injected.
    export NVCC_PREPEND_FLAGS="-ccbin $CXX"
    # dlopen of the JIT .so needs gcc-12's libstdc++ (system one lacks
    # GLIBCXX_3.4.29) and tvm_ffi's own lib dir.
    export LD_LIBRARY_PATH="$GCC12/lib:$REPO_ROOT/.venv-oscar/lib/python3.12/site-packages/tvm_ffi/lib:$CUDA128/lib:${LD_LIBRARY_PATH:-}"
fi

ARGS=(
    --model-path "$MODEL"
    --tensor-parallel-size "${TP:-1}"
    --host 127.0.0.1 --port "$PORT"
    --trust-remote-code
    --prefill-attention-backend triton
    --decode-attention-backend triton
    # flashinfer's sampling kernel JIT-compiles on the first temperature>0
    # request and fails on this box (cuda128 conda env has no curand.h);
    # greedy runs never hit it. torch sampling is fine at bs<=8.
    --sampling-backend pytorch
    --context-length "$CTX"
    --mem-fraction-static "$MEM_FRAC"
    --cuda-graph-max-bs 8
    # A100-40GB: the unified INT2 pool's HP/scale arenas land on top of the
    # mem-fraction budget (sized for H100-80GB); pin the token pool directly
    # AND the request slots (the fp16 HP recent-ring arena scales with
    # max_running_requests * ring_size across all 36 layers).
    --max-total-tokens "${MAX_TOKENS:-140000}"
    --max-running-requests "${MAX_REQS:-8}"
    # Eval rows are unique prompts: the radix cache only RETAINS finished
    # requests' BF16 prefix windows until the HP-prefix pool exhausts
    # (observed after ~3200 rows); no reuse to gain, so disable it.
    --disable-radix-cache
    # F2: quant-tier decode split-K cap. vq2's gather kernel is decode-latency-bound
    # and wants high split-K (a matched-splits sweep found 48 best: 32K decode
    # 21.4->17.2 ms/tok, and it keeps vq2 competitive to 64K). int2/bf16's cheap
    # unpack is less split-sensitive at short ctx (8 is fine there) but also wants
    # more at long ctx. Default 48 for vq2, 8 otherwise; override with KV_SPLITS=<n>.
    --triton-attention-num-kv-splits "${KV_SPLITS:-$( [ "$MODE" = vq2 ] && echo 48 || echo 8 )}"
)

if [[ "$MODE" == "int2" || "$MODE" == "vq2" ]]; then
    # Serve-time parameters exactly as the authors' eval driver sets them
    # (64/256 band + clips live in the DRIVER env, not the code defaults).
    export SGLANG_ENABLE_MIXED_KV_WINDOWS=1
    export SGLANG_OSCAR_K_ROTATION_PATH="$ROT_DIR/k_rotation_qqt_r_h_pbr.pt"
    export SGLANG_OSCAR_V_ROTATION_PATH="$ROT_DIR/v_rotation_sst_r_h_pbr.pt"
    # Overridable so the OSCAR-Table-2 baselines can serve on the same int2 path:
    # Naive/QuaRot want no clip (1.0); TurboQuant wants Lloyd-Max levels (SGLANG_LLOYD_MAX=1).
    export SGLANG_OSCAR_K_CLIP_RATIO="${SGLANG_OSCAR_K_CLIP_RATIO:-0.96}"
    export SGLANG_OSCAR_V_CLIP_RATIO="${SGLANG_OSCAR_V_CLIP_RATIO:-0.92}"
    export SGLANG_LLOYD_MAX="${SGLANG_LLOYD_MAX:-0}"
    export SGLANG_OSCAR_ABSORB_V_ROTATION=1
    export SGLANG_MIXED_KV_PREFIX_TOKENS=64
    export SGLANG_MIXED_KV_RECENT_TOKENS="${RECENT_TOKENS:-256}"
    export SGLANG_MIXED_KV_HP_DTYPE=bfloat16
    export SGLANG_MIXED_KV_SCALE_DTYPE="${SCALE_DTYPE:-float32}"
    export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
    ARGS+=(--kv-cache-dtype int2 --kv-cache-quant-group-size 128)
fi
if [[ "$MODE" == "int2plain" ]]; then
    # OSCAR baseline reproduction: plain per-token INT2, NO mixed-KV band (no MP), NO OSCAR rotation.
    # The plain int2 pool always applies a segmented Hadamard of order HADAMARD_ORDER:
    #   HADAMARD_ORDER=1   -> identity  -> Naive-INT2 (no rotation)
    #   HADAMARD_ORDER=128 -> full Hadamard over head_dim -> QuaRot-INT2
    # (MIXED_KV_WINDOWS left unset -> pool_configurator falls back to the plain int2 pool.)
    export HADAMARD_ORDER="${HADAMARD_ORDER:-128}"
    export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
    ARGS+=(--kv-cache-dtype int2 --kv-cache-quant-group-size 128)
fi
if [[ "$MODE" == "vq2" ]]; then
    # K quant tier switches to group-VQ (Samuel's fused-gather kernel design);
    # V and the K clip env above stay on the int2 path (K clip is ignored by
    # the VQ encoder).
    case "$VQ_CODEBOOK" in
        /*) export SGLANG_VQ_CODEBOOK_PATH="$VQ_CODEBOOK" ;;
        *)  export SGLANG_VQ_CODEBOOK_PATH="$REPO_ROOT/$VQ_CODEBOOK" ;;
    esac
fi

# DISABLE_CUDA_GRAPH=1 forces eager decode so in-Python decode instrumentation
# (e.g. the VQ read-back probe) actually runs instead of being skipped by graph replay.
[[ -n "${DISABLE_CUDA_GRAPH:-}" ]] && ARGS+=(--disable-cuda-graph)

exec "$REPO_ROOT/.venv-oscar/bin/python" -m sglang.launch_server "${ARGS[@]}"
