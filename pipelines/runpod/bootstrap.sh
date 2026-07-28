#!/bin/bash
# Idempotent pod setup check: environments -> engine import -> artifacts ->
# models. Prints READY when a cell can run, or the exact gap and the command
# that closes it. Safe to rerun any time; never destructive.
#
#   bash pipelines/runpod/bootstrap.sh [--group qwen3_8b] [--models "Qwen/Qwen3-8B"]
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
GROUP=qwen3_8b
MODELS="Qwen/Qwen3-8B"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --group) GROUP="$2"; shift 2 ;;
        --models) MODELS="$2"; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

# The repo may be volume-mounted with a different owner than the container
# user; without this git refuses ("dubious ownership") and provenance SHAs
# come back empty.
git config --global --get-all safe.directory 2>/dev/null | grep -qx "$ROOT" \
    || git config --global --add safe.directory "$ROOT"
git config --global --get-all safe.directory 2>/dev/null | grep -qx "$ROOT/vendor/OSCAR-vq" \
    || git config --global --add safe.directory "$ROOT/vendor/OSCAR-vq"

FAIL=0
step() { printf '\n=== %s\n' "$1"; }
ok()   { printf '  ok    %s\n' "$1"; }
bad()  { printf '  FAIL  %s\n' "$1"; FAIL=1; }

step "environments"
OSCAR_PY="${OSCAR_PYTHON:-$ROOT/.venv-oscar/bin/python}"
CLIENT_PY="${CLIENT_PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$OSCAR_PY" ]] && ok "engine python: $OSCAR_PY" || bad "engine python missing: $OSCAR_PY"
[[ -x "$CLIENT_PY" ]] && ok "client python: $CLIENT_PY" || bad "client python missing: $CLIENT_PY"

step "engine import (vendored fork on PYTHONPATH, exactly as serve_oscar.sh loads it)"
if PYTHONPATH="$ROOT/vendor/OSCAR-vq/sglang-research/python" \
        "$OSCAR_PY" -c "import sglang, torch; print(f'  ok    sglang from vendored tree, torch {torch.__version__}')"; then
    :
else
    bad "sglang import failed — engine deps or vendor/OSCAR-vq clone incomplete"
fi

step "client scorers"
if "$CLIENT_PY" -c "import pandas, sys; sys.path.insert(0, '$ROOT'); sys.path.insert(0, '$ROOT/vendor/kvpress'); from kvq.benchmarks.evaluate_registry import SCORER_REGISTRY; print(f'  ok    {len(SCORER_REGISTRY)} scorers registered')"; then
    :
else
    bad "scorer registry import failed"
fi

step "artifacts (group: $GROUP)"
if "$CLIENT_PY" pipelines/runpod/manifest.py verify --group "$GROUP"; then
    :
else
    bad "artifact payload incomplete — from the source host run:"
    echo "        bash pipelines/runpod/sync_results.sh push-artifacts <ssh-target> $GROUP"
fi

step "models"
export HF_HOME="${HF_HOME:-/workspace/hf}"
for m in $MODELS; do
    # cheap cache-presence probe: config resolves without network
    if HF_HUB_OFFLINE=1 "$OSCAR_PY" -c "
from huggingface_hub import snapshot_download
snapshot_download('$m', allow_patterns=['config.json'])" >/dev/null 2>&1; then
        ok "$m cached under $HF_HOME"
    else
        bad "$m not cached — run: bash pipelines/runpod/download_models.sh --token <HF_TOKEN> $m"
    fi
done

echo
if [[ "$FAIL" == "0" ]]; then
    echo "BOOTSTRAP READY — run_cell.sh can run against group '$GROUP'."
else
    echo "BOOTSTRAP INCOMPLETE — close the FAIL items above and rerun."
fi
exit "$FAIL"
