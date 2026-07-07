#!/usr/bin/env python3
"""Aggregate the page_quant bench cells into a summary table.

Guards (protocol-critic): only full-fraction cells are read — metrics.json is
taken from the canonical results subdir WITHOUT a fraction component, and a
cell whose full-fraction metrics is missing is a hard error, never a silent
fallback to a smoke run. Baselines (full_precision, turboquant_k2_v2,
ec_qpca_unc_dz0.5, ec_r_sym_dz0.375) are read from the EC study's
bench_summary.json (same eval set; never re-run).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import _bootstrap  # noqa: E402,F401

import json  # noqa: E402

from kvq.io import save_json  # noqa: E402

BENCH = REPO / "artifacts/bench_pgq/llama31_8b"
EC_SUMMARY = REPO / "artifacts/ec/llama31_8b/bench_summary.json"
HELDOUT = REPO / "artifacts/page_quant/pgq_heldout_report.json"
OUT = REPO / "artifacts/page_quant/bench_summary.json"
TASKS = ["lcc", "musique", "2wikimqa"]
BASELINES = ["full_precision", "turboquant_k2_v2", "jointqk_k2_v2",
             "ec_qpca_unc_dz0.5", "ec_r_sym_dz0.375"]


def cell_f1(cell_dir: Path) -> float:
    cands = []
    for mj in cell_dir.glob("*/metrics.json"):
        if "fraction" in mj.parent.name:
            continue  # smoke run — never aggregate
        cands.append(mj)
    if not cands:
        raise FileNotFoundError(
            f"no full-fraction metrics.json under {cell_dir}")
    if len(cands) > 1:
        cands.sort(key=lambda p: p.stat().st_mtime)
    blob = json.loads(cands[-1].read_text())
    if isinstance(blob, (int, float)):
        return float(blob)
    for v in blob.values():
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, dict):
            for vv in v.values():
                if isinstance(vv, (int, float)):
                    return float(vv)
    raise ValueError(f"no numeric metric in {cands[-1]}")


def main() -> None:
    rows: dict[str, dict] = {}
    for cell_dir in sorted(BENCH.glob("pgq__*")):
        parts = cell_dir.name.split("__")
        if parts[-1].startswith("f0") or "__f0" in cell_dir.name:
            continue  # smoke output dir
        if "BROKEN" in cell_dir.name:
            continue  # quarantined pre-scalefix pgq_fixed cells
        kind, rate, sha, task = parts[1], parts[2].lstrip("b"), parts[3], parts[4]
        head = f"{kind}@b{rate}"
        rows.setdefault(head, {"f1": {}, "bundle_sha8": sha})
        rows[head]["f1"][task] = cell_f1(cell_dir)

    for head, row in rows.items():
        missing = [t for t in TASKS if t not in row["f1"]]
        if missing:
            print(f"WARNING: {head} missing tasks {missing}")
        got = [row["f1"][t] for t in TASKS if t in row["f1"]]
        row["mean_f1"] = sum(got) / len(got) if got else None

    if HELDOUT.exists():
        hh = json.loads(HELDOUT.read_text())["report"]
        for head, row in rows.items():
            kind, rate = head.split("@b")
            norm = rate.rstrip("0").rstrip(".") if "." in rate else rate
            for cand in (f"{kind}@b{rate}", f"{kind}@b{norm}"):
                if cand in hh:
                    row["rate_heldout"] = hh[cand]["rate_heldout"]
                    row["overflow_frac"] = hh[cand]["overflow_frac"]
                    row["rung_hist"] = hh[cand]["rung_hist"]
                    break

    base = {}
    if EC_SUMMARY.exists():
        ec = json.loads(EC_SUMMARY.read_text())
        for m in BASELINES:
            if m in ec:
                base[m] = ec[m]

    out = {"pgq": rows, "baselines": base, "tasks": TASKS}
    save_json(OUT, out)

    fp = base.get("full_precision", {}).get("mean_f1")
    print(f"{'method':28s} {'lcc':>7s} {'musique':>8s} {'2wiki':>7s} "
          f"{'mean':>7s} {'rate':>7s} {'ovf%':>6s}")
    for m in BASELINES:
        if m in base:
            f1 = base[m]["f1"]
            print(f"{m:28s} " + " ".join(
                f"{f1.get(t, float('nan')):7.2f}" for t in TASKS)
                + f"  {base[m]['mean_f1']:6.2f} "
                + f"{str(base[m].get('rate', '-')):>7s}")
    for head in sorted(rows):
        row = rows[head]
        f1 = row["f1"]
        rate = row.get("rate_heldout")
        ovf = row.get("overflow_frac")
        print(f"{head:28s} " + " ".join(
            f"{f1.get(t, float('nan')):7.2f}" for t in TASKS)
            + f"  {row['mean_f1'] if row['mean_f1'] is not None else float('nan'):6.2f} "
            + (f"{rate:7.3f} " if rate is not None else f"{'-':>7s} ")
            + (f"{100 * ovf:5.2f}%" if ovf is not None else f"{'-':>6s}"))
    if fp is not None:
        print(f"\nfull_precision mean = {fp:.2f}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
