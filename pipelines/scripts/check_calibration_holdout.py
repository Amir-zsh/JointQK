"""Tier 0.4 — Check that calibration examples are not in the Phase 7 eval set.

Calibration source: longbench-e configs {qasper_e, hotpotqa_e,
passage_retrieval_en_e}, 8 examples each.

Phase 7 eval source: longbench configs {qasper, hotpotqa, narrativeqa,
multifieldqa_en, 2wikimqa, ...} (full split, fraction-subsampled).

For each calibration task, load the corresponding eval task and check whether
the (input, context) pair from the calibration manifest appears in the eval
split. Report overlap counts. Pass: 0 overlaps. Non-zero overlap is not a
method bug, but it weakens any "held-out evaluation" framing — we'd either
filter eval set or document scope.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _bootstrap  # noqa: E402, F401

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


CALIB_TO_EVAL = {
    "qasper_e": "qasper",
    "hotpotqa_e": "hotpotqa",
    "passage_retrieval_en_e": "passage_retrieval_en",
}


def hash_row(question_str: str, context_str: str) -> str:
    h = hashlib.sha256()
    h.update(question_str.encode("utf-8"))
    h.update(b"\x00")
    h.update(context_str.encode("utf-8"))
    return h.hexdigest()


def load_calibration_manifest(manifest_path: Path):
    """Return list of (config, row_index, input, context) for each calibration example."""
    import torch

    manifest = json.loads(manifest_path.read_text())
    out = []
    for ex in manifest["examples"]:
        bundle_path = manifest_path.parent / ex["file"]
        b = torch.load(bundle_path, map_location="cpu", weights_only=False)
        # The bundle stores prompt as text via the metadata or the original example.
        # Re-load from HF using row_index from the LongBench-E split.
        # Actually we can just hash the prompt text if stored — fall back to
        # loading from HF using the row_index.
        out.append({
            "config": ex["config"],
            "row_index": ex["row_index"],
            "captured_length": ex["captured_length"],
            "metadata": ex["metadata"],
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=str(
        REPO / "artifacts/query_stats_longbench_under4k/manifest.json"))
    p.add_argument("--output", default=str(
        REPO / "logs/integrity/tier0/04_calibration_holdout.json"))
    args = p.parse_args()

    from datasets import load_dataset

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cal = load_calibration_manifest(Path(args.manifest))
    print(f"Loaded {len(cal)} calibration examples")

    # Group by config
    cal_by_config: dict[str, list] = {}
    for c in cal:
        cal_by_config.setdefault(c["config"], []).append(c)
    print(f"Calibration configs: { {k: len(v) for k, v in cal_by_config.items()} }")

    summary = {"calibration_total": len(cal), "checks": {}}

    for cal_cfg, eval_cfg in CALIB_TO_EVAL.items():
        if cal_cfg not in cal_by_config:
            continue
        print(f"\nChecking {cal_cfg} (calibration) vs {eval_cfg} (eval)...")

        # Load calibration HF split
        ds_cal = load_dataset("Xnhyacinth/LongBench", data_dir=cal_cfg, split="test")
        ds_eval = load_dataset("Xnhyacinth/LongBench", data_dir=eval_cfg, split="test")
        print(f"  calibration HF rows: {len(ds_cal)}")
        print(f"  eval HF rows:        {len(ds_eval)}")

        # Build hash sets for eval split (by content and by _id)
        eval_hashes = {hash_row(r["question"], r["context"]) for r in ds_eval}
        eval_ids = {r["_id"] for r in ds_eval}

        # Check each calibration example
        cal_examples = cal_by_config[cal_cfg]
        overlaps = []
        id_overlaps = []
        for c in cal_examples:
            row = ds_cal[c["row_index"]]
            h = hash_row(row["question"], row["context"])
            if h in eval_hashes:
                overlaps.append({
                    "calibration_row": c["row_index"],
                    "id": row.get("_id", ""),
                    "question_preview": row["question"][:80],
                })
            if row["_id"] in eval_ids:
                id_overlaps.append(row["_id"])
        print(f"  overlaps by content (question+context): {len(overlaps)} / {len(cal_examples)}")
        print(f"  overlaps by _id:                        {len(id_overlaps)} / {len(cal_examples)}")
        for o in overlaps[:3]:
            print(f"    row={o['calibration_row']:>4d}  id={o['id']!r}  q={o['question_preview']!r}")

        summary["checks"][cal_cfg] = {
            "eval_config": eval_cfg,
            "calibration_examples_checked": len(cal_examples),
            "eval_total_rows": len(ds_eval),
            "overlap_count_content": len(overlaps),
            "overlap_count_id": len(id_overlaps),
            "overlap_examples": overlaps[:10],
        }

    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to: {out_path}")

    total_overlap = sum(c["overlap_count_content"] for c in summary["checks"].values())
    if total_overlap == 0:
        print("Tier 0.4: PASS — no calibration/eval overlap")
    else:
        print(f"Tier 0.4: WARN — {total_overlap} calibration examples appear in Phase 7 eval split")
        print("  (does not invalidate Phase 7, but weakens 'held-out' framing — see plan)")


if __name__ == "__main__":
    main()
