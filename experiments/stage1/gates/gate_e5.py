"""Gate for Stage 1E E5: validates decode-phase Q evaluation against compressed prefill cache.

Note: in this study E5 piggybacks on E3 with --query-phase both, so decode metrics live in the
same e3 summaries. This gate inspects those summaries' decode columns.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def fail(msg: str) -> None:
    print(f"GATE_E5 FAIL: {msg}", file=sys.stderr, flush=True)
    sys.exit(2)


def ok(msg: str) -> None:
    print(f"GATE_E5 PASS: {msg}", flush=True)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    e3_dir = repo_root / "artifacts/stage1/cca_vs_waterfill_study/e3"
    if not e3_dir.exists():
        fail(f"missing {e3_dir}")
    summaries = sorted(e3_dir.glob("e3_b*_r*_summary.json"))
    if not summaries:
        fail(f"no canonical E3 summaries under {e3_dir}")

    found_decode = False
    for spath in summaries:
        with open(spath) as f:
            sm = json.load(f)
        if sm.get("query_phase") not in {"decode", "both"}:
            print(f"GATE_E5 INFO: {spath.name} has query_phase={sm.get('query_phase')}; skipping decode checks", flush=True)
            continue
        agg = sm.get("aggregated", {})
        for method, mres in agg.items():
            top1_decode = mres.get("top1_decode")
            top1_prefill = mres.get("top1_prefill")
            if top1_decode is None or top1_prefill is None:
                continue
            decode_l0excl = top1_decode.get("l0excl_mean", float("nan"))
            prefill_l0excl = top1_prefill.get("l0excl_mean", float("nan"))
            if not math.isfinite(decode_l0excl) or not math.isfinite(prefill_l0excl):
                fail(f"{spath.name}/{method}: non-finite top1 (decode={decode_l0excl}, prefill={prefill_l0excl})")
            ok(
                f"{spath.name}/{method}: top1_prefill[l0excl]={prefill_l0excl:.4f} "
                f"top1_decode[l0excl]={decode_l0excl:.4f} "
                f"gap={decode_l0excl - prefill_l0excl:+.4f}"
            )
            found_decode = True

    if not found_decode:
        fail("no decode-phase metrics found in any e3 summary; rerun with --query-phase both")

    print("GATE_E5 ALL CHECKS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
