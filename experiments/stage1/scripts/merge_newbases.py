"""Merge V_h-orthogonal-CCA and R_sym (`cca_orth_*`, `r_sym_*`) reruns into canonical Stage 1E artifacts.

These four methods do not appear in canonical, so the merge appends rows rather than
replacing them. Source artifacts live in sibling `*_newbases` subdirs with run-name suffix
`_newbases`. Canonical summary JSONs and row PTs get a one-time `.pre_newbases` backup before
overwrite. Re-aggregates the merged row set so the canonical summary lists all methods.
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

NEW_METHODS = {"cca_orth_uniform", "cca_orth_waterfill", "r_sym_uniform", "r_sym_waterfill"}
BACKUP_SUFFIX = ".pre_newbases"


def _backup_once(path: Path) -> None:
    backup = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(path, backup)


def _load_rows(path: Path) -> list[dict]:
    return torch.load(path, map_location="cpu", weights_only=False)["rows"]


def _merge_rows(canonical_rows: list[dict], new_rows: list[dict]) -> list[dict]:
    if not new_rows:
        raise ValueError("new-bases rows are empty")
    methods = {row.get("method") for row in new_rows}
    unexpected = methods - NEW_METHODS
    if unexpected:
        raise ValueError(f"unexpected methods in new-bases rows: {sorted(unexpected)}")
    # Drop any existing rows for new methods (safety; should be a no-op for first merge).
    kept = [row for row in canonical_rows if row.get("method") not in NEW_METHODS]
    return kept + new_rows


def _write_merged_run(canonical_dir: Path, canonical_name: str, new_dir: Path, new_name: str) -> None:
    summary_path = canonical_dir / f"{canonical_name}_summary.json"
    rows_path = canonical_dir / f"{canonical_name}_rows.pt"
    new_rows_path = new_dir / f"{new_name}_rows.pt"
    new_summary_path = new_dir / f"{new_name}_summary.json"
    if not summary_path.exists() or not rows_path.exists():
        raise FileNotFoundError(f"missing canonical artifacts for {canonical_name}")
    if not new_rows_path.exists() or not new_summary_path.exists():
        raise FileNotFoundError(f"missing new-bases artifacts for {new_name}")

    with open(summary_path) as f:
        summary = json.load(f)
    canonical_rows = _load_rows(rows_path)
    new_rows = _load_rows(new_rows_path)
    merged_rows = _merge_rows(canonical_rows, new_rows)

    canonical_methods = list(summary.get("methods", []))
    merged_methods = canonical_methods + [m for m in NEW_METHODS if m not in canonical_methods]
    n_layers = int(summary["n_layers_eff"])
    summary["aggregated"] = aggregate_results(merged_rows, merged_methods, n_layers)
    summary["methods"] = merged_methods
    summary.setdefault("artifact_notes", {})
    summary["artifact_notes"]["newbases_merge"] = {
        "methods": sorted(NEW_METHODS),
        "source_dir": str(new_dir.relative_to(BASE)),
        "source_run_name": new_name,
        "description": "Appended cca_orth_* and r_sym_* rows from new-bases rerun.",
    }

    _backup_once(summary_path)
    _backup_once(rows_path)
    torch.save({"rows": merged_rows}, rows_path)
    save_json(summary_path, summary)
    print(f"merged {canonical_name} <- {new_name}: rows={len(merged_rows)} methods={merged_methods}")


def _merge_e3_smoke(b_avg: int) -> None:
    canonical_path = BASE / "e3" / f"e3_b{b_avg}_r64_smoke_b16.json"
    new_path = BASE / "e3_newbases" / f"e3_b{b_avg}_r64_newbases_smoke_b16.json"
    if not canonical_path.exists() or not new_path.exists():
        return
    with open(canonical_path) as f:
        canonical = json.load(f)
    with open(new_path) as f:
        new = json.load(f)
    for method in NEW_METHODS:
        if method in new:
            canonical[method] = new[method]
    _backup_once(canonical_path)
    save_json(canonical_path, canonical)
    print(f"merged smoke e3_b{b_avg}_r64 <- e3_b{b_avg}_r64_newbases (methods added)")


def main() -> int:
    # E3: b_avg = 2, 3, 4
    for b_avg in (2, 3, 4):
        _write_merged_run(
            BASE / "e3",
            f"e3_b{b_avg}_r64",
            BASE / "e3_newbases",
            f"e3_b{b_avg}_r64_newbases",
        )
        _merge_e3_smoke(b_avg)

    # E4a: 3 calibration sources
    for cfg in ("qasper", "hotpotqa", "passage_retrieval_en"):
        _write_merged_run(
            BASE / "e4a",
            f"e4a_calib_{cfg}_b3_r64",
            BASE / "e4a_newbases",
            f"e4a_calib_{cfg}_b3_r64_newbases",
        )

    # E4b: 24 LOO folds. Both canonical and newbases use qasper/hotpot/passage shorthand.
    for idx in range(24):
        if idx < 8:
            short = "qasper"
        elif idx < 16:
            short = "hotpot"
        else:
            short = "passage"
        _write_merged_run(
            BASE / "e4b",
            f"e4b_{short}_loo{idx}_b3_r64",
            BASE / "e4b_newbases",
            f"e4b_{short}_loo{idx}_b3_r64_newbases",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
