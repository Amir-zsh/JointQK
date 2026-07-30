#!/bin/bash
# rsync helpers between a source host and a pod. The pod's ssh endpoint comes
# from the RunPod connect tab, e.g.:
#   TARGET="root@213.x.x.x"  RSYNC_SSH="ssh -p 22171 -i ~/.ssh/id_ed25519"
#
#   bash pipelines/runpod/sync_results.sh push-artifacts <target> [group]
#   bash pipelines/runpod/sync_results.sh pull-results   <target> [protocol]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
CMD="${1:?push-artifacts|pull-results}" TARGET="${2:?ssh target}"
RSH="${RSYNC_SSH:-ssh}"
POD_REPO="${POD_REPO:-/workspace/teamily-project}"

case "$CMD" in
    push-artifacts)
        GROUP="${3:-qwen3_8b}"
        # file list straight from the tracked manifest — push exactly what
        # verify will check, nothing else.
        python3 - "$GROUP" <<'PY' > /tmp/runpod_payload.txt
import json, sys
from pathlib import Path
data = json.loads((Path("pipelines/runpod/artifact_manifest.json")).read_text())["groups"]
groups = list(data) if sys.argv[1] == "all" else [sys.argv[1]]
for g in groups:
    for row in data[g]:
        print(row["path"])
PY
        # --no-owner/--no-group: container root on the pod's mounted volume
        # commonly lacks chown capability even as uid 0 -- -a's ownership
        # preservation then fails every file with "Operation not permitted".
        rsync -av --no-owner --no-group --info=progress2 -e "$RSH" --files-from=/tmp/runpod_payload.txt \
            "$ROOT/" "$TARGET:$POD_REPO/"
        echo "pushed; on the pod run: bash pipelines/runpod/bootstrap.sh --group $GROUP"
        ;;
    pull-results)
        PROTO="${3:-}"
        rsync -av --no-owner --no-group --info=progress2 -e "$RSH" \
            "$TARGET:$POD_REPO/artifacts/runpod/${PROTO:+$PROTO/}" \
            "$ROOT/artifacts/runpod/${PROTO:+$PROTO/}"
        ;;
    *) echo "Unknown command: $CMD" >&2; exit 1 ;;
esac
