"""Merge F11-only CCA-waterfill reruns into canonical Stage 1E artifacts.

The F11 fix changed only the real `cca_waterfill` allocator. The reruns are
stored in sibling `*_f11` directories with `methods=cca_waterfill`; this script
replaces only those rows in the canonical E3/E4 artifacts, then recomputes the
canonical summaries from the merged row files.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "artifacts/stage1/cca_vs_waterfill_study"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.stage1.run_cca_vs_waterfill_study import aggregate_results  # noqa: E402
from experiments.stage1.toolkit import save_json  # noqa: E402


def _backup_once(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".pre_f11")
    if not backup.exists():
        shutil.copy2(path, backup)


def _load_rows(path: Path) -> list[dict]:
    return torch.load(path, map_location="cpu", weights_only=False)["rows"]


def _merge_rows(canonical_rows: list[dict], f11_rows: list[dict]) -> list[dict]:
    kept = [row for row in canonical_rows if row.get("method") != "cca_waterfill"]
    if not f11_rows:
        raise ValueError("F11 rows are empty")
    methods = {row.get("method") for row in f11_rows}
    if methods != {"cca_waterfill"}:
        raise ValueError(f"expected only cca_waterfill rows, got {sorted(methods)}")
    return kept + f11_rows


def _write_merged_run(canonical_dir: Path, canonical_name: str, f11_dir: Path, f11_name: str) -> None:
    summary_path = canonical_dir / f"{canonical_name}_summary.json"
    rows_path = canonical_dir / f"{canonical_name}_rows.pt"
    f11_rows_path = f11_dir / f"{f11_name}_rows.pt"
    f11_summary_path = f11_dir / f"{f11_name}_summary.json"
    if not summary_path.exists() or not rows_path.exists():
        raise FileNotFoundError(f"missing canonical artifacts for {canonical_name}")
    if not f11_rows_path.exists() or not f11_summary_path.exists():
        raise FileNotFoundError(f"missing F11 artifacts for {f11_name}")

    with open(summary_path) as f:
        summary = json.load(f)
    canonical_rows = _load_rows(rows_path)
    f11_rows = _load_rows(f11_rows_path)
    merged_rows = _merge_rows(canonical_rows, f11_rows)

    methods = list(summary["methods"])
    n_layers = int(summary["n_layers_eff"])
    summary["aggregated"] = aggregate_results(merged_rows, methods, n_layers)
    summary.setdefault("artifact_notes", {})
    summary["artifact_notes"]["f11_merge"] = {
        "method": "cca_waterfill",
        "source_dir": str(f11_dir.relative_to(BASE)),
        "source_run_name": f11_name,
        "description": "Replaced only cca_waterfill rows with post-F11 trace-formula allocation rerun.",
    }

    _backup_once(summary_path)
    _backup_once(rows_path)
    torch.save({"rows": merged_rows}, rows_path)
    save_json(summary_path, summary)
    print(f"merged {canonical_name} <- {f11_name}: rows={len(merged_rows)}")


def _merge_e3_smoke(b_avg: int) -> None:
    canonical_path = BASE / "e3" / f"e3_b{b_avg}_r64_smoke_b16.json"
    f11_path = BASE / "e3_f11" / f"e3_f11_b{b_avg}_r64_smoke_b16.json"
    if not canonical_path.exists() or not f11_path.exists():
        return
    with open(canonical_path) as f:
        canonical = json.load(f)
    with open(f11_path) as f:
        f11 = json.load(f)
    if "cca_waterfill" not in f11:
        raise ValueError(f"{f11_path} lacks cca_waterfill smoke result")
    canonical["cca_waterfill"] = f11["cca_waterfill"]
    _backup_once(canonical_path)
    save_json(canonical_path, canonical)
    print(f"merged smoke e3_b{b_avg}_r64 <- e3_f11_b{b_avg}_r64")


def main() -> int:
    for b_avg in (2, 3, 4):
        _write_merged_run(
            BASE / "e3",
            f"e3_b{b_avg}_r64",
            BASE / "e3_f11",
            f"e3_f11_b{b_avg}_r64",
        )
        _merge_e3_smoke(b_avg)

    for cfg in ("qasper", "hotpotqa", "passage_retrieval_en"):
        _write_merged_run(
            BASE / "e4a",
            f"e4a_calib_{cfg}_b3_r64",
            BASE / "e4a_f11",
            f"e4a_f11_calib_{cfg}_b3_r64",
        )

    for idx in range(24):
        if idx < 8:
            canonical_short = f11_short = "qasper"
        elif idx < 16:
            canonical_short = "hotpot"
            f11_short = "hotpotqa"
        else:
            canonical_short = f11_short = "passage"
        _write_merged_run(
            BASE / "e4b",
            f"e4b_{canonical_short}_loo{idx}_b3_r64",
            BASE / "e4b_f11",
            f"e4b_f11_{f11_short}_loo{idx}_b3_r64",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
