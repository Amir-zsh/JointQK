#!/usr/bin/env python3
"""Select compact8 TRAIN rows for the EC (entropy-coding) calibration capture.

Two disjoint role sets, both strictly train-split (already excluded from every
bench F1 eval via exclude_train_indices_for_eval.json, so no leakage):

  - fit:       2 rows/task x 8 tasks + 2 extra repobench-p = 18 rows.
               Used to fit the EC deadzone deltas + frozen coder models.
               repobench-p is over-weighted as the in-calibration code-task
               sentinel for lcc (which is OOD to compact8).
  - selection: 1 row/task = 8 rows. Held out from fitting; used to pick the
               winning (basis, dz) candidate and measure held-out coded rate.

Writes a capture split-manifest (same schema as the compact8 manifest, rows
subsetted) plus a roles.json mapping each (config, row_index) to its role.

    python pipelines/ec/make_ec_capture_manifest.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC_MANIFEST = REPO / "artifacts/calibration_splits/longbench_compact8_60_seed20260504_2k32k/manifest.json"
OUT_DIR = REPO / "artifacts/calibration_splits/ec_compact8_train_26"

FIT_PER_TASK = 2
EXTRA_FIT_REPOBENCH = 2
SELECTION_PER_TASK = 1
# Prompt-length windows (Qwen-tokenized manifest lengths; Llama is ~±30%).
FIT_LEN = (4_000, 12_000)
SEL_LEN = (4_000, 8_000)


def pick(rows: list[dict], n: int, lo: int, hi: int, taken: set[int]) -> list[dict]:
    """n rows preferring the [lo, hi] length window, deterministic order."""
    rows = sorted(rows, key=lambda r: (r["split_index"], r["row_index"]))
    in_window = [r for r in rows if lo <= r["prompt_length"] <= hi and r["row_index"] not in taken]
    fallback = [r for r in rows if r["row_index"] not in taken and r not in in_window]
    chosen = (in_window + fallback)[:n]
    if len(chosen) < n:
        raise RuntimeError(f"not enough rows: wanted {n}, found {len(chosen)}")
    taken.update(r["row_index"] for r in chosen)
    return chosen


def main() -> None:
    src = json.loads(SRC_MANIFEST.read_text())
    tasks = src["config"]["tasks"]
    by_task = {
        t: [r for r in src["rows"] if r["config"] == t and r["split"] == "train"]
        for t in tasks
    }

    fit_rows: list[dict] = []
    sel_rows: list[dict] = []
    for task in tasks:
        taken: set[int] = set()
        n_fit = FIT_PER_TASK + (EXTRA_FIT_REPOBENCH if task == "repobench-p" else 0)
        fit_rows.extend(pick(by_task[task], n_fit, *FIT_LEN, taken))
        sel_rows.extend(pick(by_task[task], SELECTION_PER_TASK, *SEL_LEN, taken))

    # Order rows longest-first so idx%num_shards round-robin balances token load.
    all_rows = sorted(fit_rows + sel_rows, key=lambda r: -r["prompt_length"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": dict(src["config"], purpose="ec_calibration_train_capture"),
        "num_rows": len(all_rows),
        "num_train_rows": len(all_rows),
        "num_test_rows": 0,
        "purpose": "EC coder-model calibration: compact8 TRAIN rows only (no eval leakage)",
        "source_manifest": str(SRC_MANIFEST.relative_to(REPO)),
        "rows": all_rows,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))

    roles = {
        "fit": [[r["config"], r["row_index"]] for r in fit_rows],
        "selection": [[r["config"], r["row_index"]] for r in sel_rows],
    }
    (OUT_DIR / "roles.json").write_text(json.dumps(roles, indent=1))

    tok = sum(r["prompt_length"] for r in all_rows)
    print(f"fit={len(fit_rows)} selection={len(sel_rows)} total_tokens~{tok:,} (Qwen-tokenized)")
    for r in all_rows:
        role = "fit" if [r["config"], r["row_index"]] in roles["fit"] else "sel"
        print(f"  {role:>3} {r['config']:<22} row{r['row_index']:<6} len={r['prompt_length']}")


if __name__ == "__main__":
    main()
