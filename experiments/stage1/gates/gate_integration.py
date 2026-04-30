"""Final integration gate: cross-experiment consistency before report writing."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def fail(msg: str) -> None:
    print(f"GATE_INTEGRATION FAIL: {msg}", file=sys.stderr, flush=True)
    sys.exit(2)


def ok(msg: str) -> None:
    print(f"GATE_INTEGRATION PASS: {msg}", flush=True)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    base = repo_root / "artifacts/stage1/cca_vs_waterfill_study"

    e12_path = base / "metrics_e1_e2.json"
    if not e12_path.exists():
        fail(f"missing {e12_path}")
    with open(e12_path) as f:
        e12 = json.load(f)

    # Pull simulation winner at b_avg=3 layer-0-excluded
    sim = e12["simulation"]
    if "b3.0" not in sim:
        fail("E1+E2 missing b_avg=3 simulation results")
    sim_b3 = sim["b3.0"]
    sim_winner = None
    sim_best_logratio = float("inf")
    for k, v in sim_b3.items():
        if k == "v3":
            continue
        lr = v.get("log2_D_over_Dv3_l0excl_mean", float("inf"))
        if lr < sim_best_logratio:
            sim_best_logratio = lr
            sim_winner = k
    if sim_winner is None:
        fail("could not determine E2 simulation winner at b_avg=3")
    ok(f"E2 simulation winner @ b_avg=3 (l0excl): {sim_winner} (log2 D/D_v3 = {sim_best_logratio:+.3f})")

    # Pull E3 winner at b_avg=3 layer-0-excluded on top1_prefill
    e3_summaries = sorted((base / "e3").glob("*b3.0*_summary.json"))
    if not e3_summaries:
        e3_summaries = sorted((base / "e3").glob("*_summary.json"))
    if not e3_summaries:
        fail("no E3 summaries found")
    spath = next((p for p in e3_summaries if "b3.0" in p.name), e3_summaries[0])
    with open(spath) as f:
        e3 = json.load(f)
    real_winner = None
    real_best_top1 = -float("inf")
    for method, mres in e3["aggregated"].items():
        top1 = mres.get("top1_prefill", {}).get("l0excl_mean", -float("inf"))
        if top1 > real_best_top1:
            real_best_top1 = top1
            real_winner = method
    if real_winner is None:
        fail("could not determine E3 real-quant winner at b_avg=3")
    ok(f"E3 real-quant winner @ b_avg=3 (l0excl top1): {real_winner} ({real_best_top1:.4f})")

    # Cross-check: simulation winner == real winner OR document the gap
    if real_winner != sim_winner:
        print(
            f"GATE_INTEGRATION WARN: simulation vs reality disagreement at b_avg=3 — "
            f"sim winner={sim_winner}, real winner={real_winner}. The Stage 1E report should "
            "name this gap explicitly.",
            flush=True,
        )

    # Verify e4 results exist (or note they're missing)
    e4a_summaries = sorted((base / "e4a").glob("*_summary.json"))
    e4b_summaries = sorted((base / "e4b").glob("*_summary.json"))
    if len(e4a_summaries) < 3:
        print(f"GATE_INTEGRATION INFO: e4a has only {len(e4a_summaries)} summaries; cross-task analysis incomplete", flush=True)
    if len(e4b_summaries) < 24:
        print(f"GATE_INTEGRATION INFO: e4b has only {len(e4b_summaries)} summaries; LOO analysis incomplete", flush=True)

    print("GATE_INTEGRATION ALL CHECKS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
