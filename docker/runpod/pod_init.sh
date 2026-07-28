#!/bin/bash
# Clone the two repos into /workspace and hand off to the repo's bootstrap.
# Baked into the image only because it must exist before the repo does;
# everything that changes week-to-week lives in the repo (pipelines/runpod/).
#
# Env:
#   GIT_TOKEN   GitHub PAT with read access to both private repos (https
#               clone). Alternative: ssh-agent forwarding, then set
#               MAIN_URL/ENGINE_URL to the git@ forms.
#   MAIN_URL    default https://github.com/Amir-zsh/JointQK.git
#   ENGINE_URL  default https://github.com/Amir-zsh/VQ-SGLang.git
#   MAIN_REF    branch/SHA to check out (default: repo default branch)
#   ENGINE_REF  branch/SHA for the engine (default: vq2-longhorizon)
#   KVPRESS_URL default https://github.com/NVIDIA/kvpress.git (public, no token)
set -euo pipefail

WS=/workspace
MAIN_URL="${MAIN_URL:-https://github.com/Amir-zsh/JointQK.git}"
ENGINE_URL="${ENGINE_URL:-https://github.com/Amir-zsh/VQ-SGLang.git}"
ENGINE_REF="${ENGINE_REF:-vq2-longhorizon}"
KVPRESS_URL="${KVPRESS_URL:-https://github.com/NVIDIA/kvpress.git}"
REPO="$WS/teamily-project"

auth_url() {  # embed the PAT for the clone only; the stored remote is scrubbed
    if [ -n "${GIT_TOKEN:-}" ]; then
        echo "${1/https:\/\//https://oauth2:${GIT_TOKEN}@}"
    else
        echo "$1"
    fi
}

clone_or_update() {  # url ref dest
    local url="$1" ref="$2" dest="$3"
    if [ -d "$dest/.git" ]; then
        git -C "$dest" fetch "$(auth_url "$url")" ${ref:+"$ref"} || true
    else
        git clone "$(auth_url "$url")" "$dest"
        git -C "$dest" remote set-url origin "$url"
    fi
    if [ -n "$ref" ]; then
        git -C "$dest" checkout "$ref" 2>/dev/null \
            || git -C "$dest" checkout -b "$ref" FETCH_HEAD
    fi
}

clone_or_update "$MAIN_URL" "${MAIN_REF:-}" "$REPO"
clone_or_update "$ENGINE_URL" "$ENGINE_REF" "$REPO/vendor/OSCAR-vq"
# Public repo (no GIT_TOKEN needed) — bootstrap.sh's scorer-registry check
# puts vendor/kvpress/evaluation/ on sys.path via kvq/benchmarks's __init__.
clone_or_update "$KVPRESS_URL" "" "$REPO/vendor/kvpress"

# Local patch to a kvpress scorer bug (comma-formatted numbers false-negative
# on exact substring match) — not upstream, can't be fetched by cloning; see
# fixes_to_apply.md. Idempotent (skipped if already applied); a genuine
# conflict (upstream changed the file) fails loudly via set -e rather than
# silently serving unpatched scoring.
KVPRESS_PATCH="$REPO/pipelines/runpod/patches/kvpress_ruler_comma_fix.patch"
KVPRESS_SCORER="$REPO/vendor/kvpress/evaluation/benchmarks/ruler/calculate_metrics.py"
if [ -f "$KVPRESS_PATCH" ] && ! grep -q "def polish" "$KVPRESS_SCORER" 2>/dev/null; then
    git -C "$REPO/vendor/kvpress" apply "$KVPRESS_PATCH"
fi

exec bash "$REPO/pipelines/runpod/bootstrap.sh"
