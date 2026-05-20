#!/usr/bin/env bash
# Sync teamily-project to a remote host via rsync over ssh.
#
# Executes by default. Pass --dry-run to preview without transferring.
#
# Profiles:
#   default  -- code, notes, vendor, .git, paper. Excludes artifacts/, .venv/,
#               logs/, notebooks/data/. ~165 MB.
#   --phase7 -- default + the small artifacts Phase 7 needs to run on the
#               remote: legacy v6 calibration files (cca_stats.pt, v_stats.pt,
#               k_floor.txt for qwen3_8b and llama31_8b), the current v_lock.txt
#               (which now selects v_turboquant per the v7 disconnect
#               investigation), AND the calibration_splits/ directory the v7
#               launcher reads (manifest.json + exclude_train_indices_for_eval
#               .json + train/test row jsonls). The Llama-specific v7 cca_stats
#               and v_stats are NOT included — those get built ON the remote
#               from the locally-captured prefill stats (see v7_llama_runbook).
#
# After --phase7 sync, on the remote:
#   1. cd /vault/amir/efficient-llm/teamily-project
#   2. uv venv && source .venv/bin/activate && uv sync   (or recreate .venv)
#   3. Read notes/stage1/v7_llama_runbook.md for the v7 pipeline.
#      For v6 reproduction:
#        bash experiments/stage1/scripts/launch_phase7_longbench.sh \
#            --model qwen3_8b --gpus 0,1,2,3
#
# Examples:
#   ./sync_to_remote.sh                         # default profile, live
#   ./sync_to_remote.sh --phase7                # default + phase7 artifacts
#   ./sync_to_remote.sh --dry-run --phase7      # preview only
#   ./sync_to_remote.sh --artifacts             # full artifacts/ (~119 GB)
#   ./sync_to_remote.sh --all                   # everything except .venv
#   ./sync_to_remote.sh --host other.host       # different host

set -euo pipefail

HOST="10.137.32.79"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/"
DEST="/vault/amir/efficient-llm/teamily-project/"
DRY=""
DELETE=""
INCLUDE_ARTIFACTS=0
INCLUDE_LOGS=0
INCLUDE_NOTEBOOK_DATA=0
INCLUDE_VENV=0
INCLUDE_PHASE7=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

  -n, --dry-run     preview only; nothing is transferred
  --host HOST       remote host (default: $HOST)
  --dest PATH       remote destination directory (default: $DEST)
  --phase7          add the calibration artifacts Phase 7 needs (~345 MB)
  --artifacts       include the entire artifacts/                  (~119 GB)
  --logs            include experiments/*/logs/                    (regenerable)
  --notebook-data   include experiments/*/notebooks/data/          (~7 GB)
  --venv            include .venv/                                 (~8 GB; usually rebuild remotely)
  --all             shorthand for --artifacts --logs --notebook-data
  --delete          delete files on remote that no longer exist locally
  -h, --help        show this help

Always-excluded: __pycache__/, *.pyc, .pytest_cache/, *.pre_* backups,
                 texput.log, .DS_Store
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--dry-run) DRY="--dry-run" ;;
    --host) HOST="$2"; shift ;;
    --dest) DEST="$2"; shift ;;
    --phase7) INCLUDE_PHASE7=1 ;;
    --artifacts) INCLUDE_ARTIFACTS=1 ;;
    --logs) INCLUDE_LOGS=1 ;;
    --notebook-data) INCLUDE_NOTEBOOK_DATA=1 ;;
    --venv) INCLUDE_VENV=1 ;;
    --all) INCLUDE_ARTIFACTS=1; INCLUDE_LOGS=1; INCLUDE_NOTEBOOK_DATA=1 ;;
    --delete) DELETE="--delete" ;;
    -h|--help) usage ;;
    *) echo "unknown arg: $1" >&2; usage ;;
  esac
  shift
done

EXCLUDES=(
  --exclude='__pycache__/'
  --exclude='.pytest_cache/'
  --exclude='*.pyc'
  --exclude='.DS_Store'
  --exclude='texput.log'
  --exclude='*.pre_*'
)

[[ $INCLUDE_VENV          -eq 0 ]] && EXCLUDES+=(--exclude='.venv/')
[[ $INCLUDE_LOGS          -eq 0 ]] && EXCLUDES+=(--exclude='logs/')
[[ $INCLUDE_NOTEBOOK_DATA -eq 0 ]] && EXCLUDES+=(--exclude='notebooks/data/')
[[ $INCLUDE_ARTIFACTS     -eq 0 ]] && EXCLUDES+=(--exclude='artifacts/')

echo "Source : $SRC"
echo "Target : $HOST:$DEST"
echo "Mode   : $([[ -n $DRY ]] && echo 'DRY-RUN (preview only)' || echo 'LIVE')"
[[ -n $DELETE ]] && echo "Delete : remote files missing locally will be removed"
echo "Includes:"
echo "  artifacts/        : $([[ $INCLUDE_ARTIFACTS     -eq 1 ]] && echo yes || echo no)"
echo "  phase7 artifacts  : $([[ $INCLUDE_PHASE7        -eq 1 ]] && echo yes || echo no)"
echo "  logs/             : $([[ $INCLUDE_LOGS          -eq 1 ]] && echo yes || echo no)"
echo "  notebooks/data/   : $([[ $INCLUDE_NOTEBOOK_DATA -eq 1 ]] && echo yes || echo no)"
echo "  .venv/            : $([[ $INCLUDE_VENV          -eq 1 ]] && echo yes || echo no)"
echo

ssh "$HOST" "mkdir -p '$DEST'"

echo "=== Pass 1: project tree ==="
rsync -avh --partial --human-readable --info=progress2 \
  $DRY $DELETE \
  "${EXCLUDES[@]}" \
  "$SRC" "$HOST:$DEST"

if [[ $INCLUDE_PHASE7 -eq 1 && $INCLUDE_ARTIFACTS -eq 0 ]]; then
  echo
  echo "=== Pass 2: phase7 / v7 calibration files ==="
  # Files Phase 7 (v6 + v7) launchers reference. Sync each explicitly so we
  # never push the multi-hundred-GB raw / stats trees.
  PHASE7_FILES=(
    # Legacy v6 calibration (kept for v6 reproducibility)
    "artifacts/stage1/cca_vs_waterfill_study/cca_stats.pt"
    "artifacts/stage1/cca_vs_waterfill_study/llama31_8b/cca_stats.pt"
    "artifacts/stage1/v_method_study/v_stats.pt"
    "artifacts/stage1/v_method_study/v_stats_llama31_8b.pt"
    "artifacts/stage1/v_method_study/k_floor.txt"
    # v7 lock (V_METHOD=v_turboquant after the disconnect investigation)
    "artifacts/stage1/v_method_study/v_lock.txt"
  )
  for rel in "${PHASE7_FILES[@]}"; do
    if [[ ! -e "$SRC$rel" ]]; then
      echo "WARN: missing locally: $rel" >&2
      continue
    fi
    ssh "$HOST" "mkdir -p '$DEST$(dirname "$rel")'"
    rsync -avh --partial --human-readable $DRY \
      "$SRC$rel" "$HOST:$DEST$rel"
  done

  # v7 calibration split directory: manifest.json (used by capture) +
  # exclude_train_indices_for_eval.json (used by sweep) + train/test row
  # jsonls. ~5 small files, total <5 MB.
  PHASE7_DIRS=(
    "artifacts/stage1/calibration_splits/longbench_compact8_60_seed20260504_2k32k/"
  )
  for rel in "${PHASE7_DIRS[@]}"; do
    if [[ ! -d "$SRC$rel" ]]; then
      echo "WARN: missing locally: $rel" >&2
      continue
    fi
    ssh "$HOST" "mkdir -p '$DEST$(dirname "$rel")'"
    rsync -avh --partial --human-readable $DRY \
      "$SRC$rel" "$HOST:$DEST$rel"
  done
fi

if [[ -n $DRY ]]; then
  echo
  echo "Dry-run complete. Re-run without --dry-run to actually transfer."
fi
