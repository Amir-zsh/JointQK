#!/bin/bash
# Download HF model weights into $HF_HOME (the pod volume, so they survive
# restarts). Token comes from --token or the HF_TOKEN env var; it is needed
# for gated repos (Llama) and otherwise optional.
#
#   bash pipelines/runpod/download_models.sh [--token hf_xxx] [model ...]
#
# With no models listed, downloads the default experiment set.
set -euo pipefail

MODELS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --token) export HF_TOKEN="$2"; shift 2 ;;
        *) MODELS+=("$1"); shift ;;
    esac
done
[[ ${#MODELS[@]} -gt 0 ]] || MODELS=(Qwen/Qwen3-8B)

export HF_HOME="${HF_HOME:-/workspace/hf}"
mkdir -p "$HF_HOME"

# huggingface-hub 1.x ships the `hf` CLI inside the engine venv.
HF_CLI="$(dirname "${OSCAR_PYTHON:-/opt/venv-oscar/bin/python}")/hf"

for m in "${MODELS[@]}"; do
    echo "=== $m -> $HF_HOME"
    # --exclude takes ONE pattern per flag (repeatable) -- a second bare
    # string after it is parsed as an explicit filename to fetch, not a
    # second pattern, and 404s since "original/*" doesn't exist literally.
    "$HF_CLI" download "$m" --exclude "*.pth" --exclude "original/*"
done
echo "models ready under $HF_HOME"
