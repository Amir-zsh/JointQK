#!/usr/bin/env python3
"""Capture manifest for the post-hoc honest-rate measurement: 4 EVAL-side rows
per target task (lcc, musique, 2wikimqa).

These rows are used ONLY to measure the frozen coder model's coded rate on the
actual bench tasks (measurement, no feedback into fitting). Sources:
  - musique: compact8 TEST rows (in the F1 eval set).
  - lcc: compact9 manifest rows (compact8 has no lcc; all lcc rows are eval).
  - 2wikimqa: no split manifest exists -> row indices 0..3 (no exclusions
    apply to 2wikimqa, so every row is an eval row). No content hashes are
    available for these; capture_raw skips hash validation when absent.

    python pipelines/ec/make_posthoc_rate_manifest.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
C8 = REPO / "artifacts/calibration_splits/longbench_compact8_60_seed20260504_2k32k/manifest.json"
C9 = REPO / "artifacts/calibration_splits/longbench_compact9_60_seed20260504_2k32k/manifest.json"
OUT_DIR = REPO / "artifacts/calibration_splits/ec_posthoc_rate_12"
N_PER_TASK = 4
LEN_PREF = (4_000, 12_000)


def pick(rows, n):
    rows = sorted(rows, key=lambda r: (r.get("split_index", 0), r["row_index"]))
    pref = [r for r in rows if LEN_PREF[0] <= r.get("prompt_length", 0) <= LEN_PREF[1]]
    return (pref + [r for r in rows if r not in pref])[:n]


def main() -> None:
    c8 = json.loads(C8.read_text())
    rows = []

    musique = [r for r in c8["rows"] if r["config"] == "musique" and r["split"] == "test"]
    rows += pick(musique, N_PER_TASK)

    if C9.exists():
        c9 = json.loads(C9.read_text())
        lcc = [r for r in c9["rows"] if r["config"] == "lcc" and r["split"] == "test"]
        if not lcc:  # fall back to any lcc rows in the manifest
            lcc = [r for r in c9["rows"] if r["config"] == "lcc"]
    else:
        lcc = []
    if lcc:
        rows += pick(lcc, N_PER_TASK)
    else:
        rows += [dict(dataset="longbench", config="lcc", task="lcc", split="test",
                      row_index=i, prompt_length=8000) for i in range(N_PER_TASK)]

    rows += [dict(dataset="longbench", config="2wikimqa", task="2wikimqa", split="test",
                  row_index=i, prompt_length=8000) for i in range(N_PER_TASK)]

    # normalize: mark all rows split=test (they're eval-side for rate purposes)
    out_rows = []
    for r in rows:
        r = dict(r)
        r["split"] = "test"
        r.pop("assigned_shard", None)
        out_rows.append(r)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": dict(c8["config"], tasks=["lcc", "musique", "2wikimqa"],
                       purpose="ec_posthoc_rate"),
        "num_rows": len(out_rows), "num_test_rows": len(out_rows), "num_train_rows": 0,
        "purpose": "EC post-hoc held-out rate measurement on bench tasks (eval-side rows)",
        "rows": out_rows,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
    for r in out_rows:
        print(f"  {r['config']:<10} row{r['row_index']:<6} len~{r.get('prompt_length')}")
    print(f"wrote {OUT_DIR}/manifest.json ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
