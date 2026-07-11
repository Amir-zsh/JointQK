#!/usr/bin/env python3
"""pgq5 Stage A: split the 16 surviving Qwen compact8 train raws into
12 fit + 4 selection roles (mirrors ec_compact8_train_26/roles.json format).

The pool has exactly 2 rows per task x 8 tasks. Selection takes ONE row from
each of 4 genre-spread tasks (repobench-p = code/OOD slice, musique =
multi-hop, qasper = scientific QA, qmsum = summarization); the deterministic
rule is "higher row_index of the task's pair". Fit keeps the rest (12 rows).

    python pipelines/scripts/make_qwen_fit_split.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from kvq.io import ensure_dir  # noqa: E402

RAW_ROOT = REPO / "artifacts/calibration/longbench_compact8_qkv_qwen3_8b/01_raw"
OUT = REPO / "artifacts/calibration_splits/pgq5_qwen_compact8_16/roles.json"
SELECTION_TASKS = ["repobench-p", "musique", "qasper", "qmsum"]
PAT = re.compile(r"longbench__(.+)__row(\d{5})__train\.pt$")


def main() -> None:
    rows: dict[str, list[int]] = {}
    for p in sorted(RAW_ROOT.glob("shard_*/*__train.pt")):
        m = PAT.search(p.name)
        rows.setdefault(m.group(1), []).append(int(m.group(2)))
    assert len(rows) == 8 and all(len(v) == 2 for v in rows.values()), rows

    fit, selection = [], []
    for task in sorted(rows):
        lo, hi = sorted(rows[task])
        if task in SELECTION_TASKS:
            fit.append([task, lo])
            selection.append([task, hi])
        else:
            fit += [[task, lo], [task, hi]]
    assert len(fit) == 12 and len(selection) == 4

    ensure_dir(OUT.parent)
    OUT.write_text(json.dumps({"fit": fit, "selection": selection}, indent=1))
    print(f"wrote {OUT}: fit={len(fit)} selection={selection}")


if __name__ == "__main__":
    main()
