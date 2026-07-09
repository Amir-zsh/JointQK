#!/usr/bin/env python3
"""Aggregate the plan3 Thrust-A eviction bench cells into a summary table.

Same guards as aggregate_pgq.py: only full-fraction cells (metrics.json from
the canonical no-fraction results subdir; a missing one is a hard error),
smoke (`__f0*`) and quarantined (`BROKEN`) cell dirs skipped. Baselines
(full_precision) come from the EC study's bench_summary.json + v7 tree via
aggregate_pgq's conventions — read from the pgq summary, never re-run.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import _bootstrap  # noqa: E402,F401

import json  # noqa: E402

from pipelines.eval.aggregate_pgq import cell_f1  # noqa: E402
from kvq.io import save_json  # noqa: E402

BENCH = REPO / "artifacts/bench_evict/llama31_8b"
PGQ_SUMMARY = REPO / "artifacts/page_quant/bench_summary.json"
OUT = REPO / "artifacts/bench_evict/bench_summary.json"
TRIO = ["lcc", "musique", "2wikimqa"]
TASKS = ["lcc", "musique", "2wikimqa", "qasper", "hotpotqa"]


def main() -> None:
    rows: dict[str, dict] = {}
    for cell_dir in sorted(BENCH.glob("evict__*")):
        if "__f0" in cell_dir.name or "BROKEN" in cell_dir.name:
            continue
        # evict__<press-label>__r<ratio>__<task>
        parts = cell_dir.name.split("__")
        task = parts[-1]
        ratio = parts[-2].lstrip("r")
        press = "__".join(parts[1:-2])
        head = f"{press}@r{ratio}"
        rows.setdefault(head, {"f1": {}})
        rows[head]["f1"][task] = cell_f1(cell_dir)

    for head, row in rows.items():
        missing = [t for t in TASKS if t not in row["f1"]]
        if missing:
            print(f"WARNING: {head} missing tasks {missing}")
        got = [row["f1"][t] for t in TASKS if t in row["f1"]]
        row["mean_f1"] = sum(got) / len(got) if got else None
        trio = [row["f1"][t] for t in TRIO if t in row["f1"]]
        row["trio_mean_f1"] = sum(trio) / len(trio) if trio else None

    base = {}
    if PGQ_SUMMARY.exists():
        pg = json.loads(PGQ_SUMMARY.read_text())
        if "full_precision" in pg.get("baselines", {}):
            base["full_precision"] = pg["baselines"]["full_precision"]

    save_json(OUT, {"evict": rows, "baselines": base, "tasks": TASKS})

    hdr = " ".join(f"{t[:7]:>8s}" for t in TASKS)
    print(f"{'method':38s} {hdr} {'mean5':>7s} {'trio':>7s}")
    for name, row in list(base.items()) + sorted(rows.items()):
        f1 = row["f1"]
        cols = " ".join(f"{f1[t]:8.2f}" if t in f1 else f"{'-':>8s}"
                        for t in TASKS)
        m5, tr = row.get("mean_f1"), row.get("trio_mean_f1")
        print(f"{name:38s} {cols} "
              + (f"{m5:7.2f} " if m5 is not None else f"{'-':>7s} ")
              + (f"{tr:7.2f}" if tr is not None else f"{'-':>7s}"))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
