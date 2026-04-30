"""Gate for Stage 1E E4: validates cross-task and within-task LOO results."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def fail(msg: str) -> None:
    print(f"GATE_E4 FAIL: {msg}", file=sys.stderr, flush=True)
    sys.exit(2)


def ok(msg: str) -> None:
    print(f"GATE_E4 PASS: {msg}", flush=True)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    e4a_dir = repo_root / "artifacts/stage1/cca_vs_waterfill_study/e4a"
    e4b_dir = repo_root / "artifacts/stage1/cca_vs_waterfill_study/e4b"

    if not e4a_dir.exists():
        fail(f"missing {e4a_dir}; run launch_cca_study.sh --phase e4a")
    if not e4b_dir.exists():
        fail(f"missing {e4b_dir}; run launch_cca_study.sh --phase e4b")

    e4a_summaries = sorted(e4a_dir.glob("*_summary.json"))
    e4b_summaries = sorted(e4b_dir.glob("*_summary.json"))

    if len(e4a_summaries) < 3:
        fail(f"e4a: expected 3 calibration sources, found {len(e4a_summaries)} summaries")
    ok(f"e4a: {len(e4a_summaries)} calibration-source runs found")

    if len(e4b_summaries) < 24:
        fail(f"e4b: expected 24 LOO folds, found {len(e4b_summaries)} summaries")
    ok(f"e4b: {len(e4b_summaries)} LOO-fold runs found")

    for spath in e4a_summaries + e4b_summaries:
        with open(spath) as f:
            sm = json.load(f)
        agg = sm.get("aggregated", {})
        for method, mres in agg.items():
            for metric_key, mvals in mres.items():
                if metric_key == "bootstrap_ci":
                    continue
                pl = mvals.get("per_layer", [])
                non_finite = [i for i, v in enumerate(pl) if not math.isfinite(v)]
                if non_finite and metric_key not in {"logit_mse_decode", "top1_decode", "top5_decode", "logit_cosine_decode", "decode_query_count"}:
                    fail(f"{spath.name}/{method}/{metric_key}: non-finite at layers {non_finite[:5]}")

    print("GATE_E4 ALL CHECKS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
