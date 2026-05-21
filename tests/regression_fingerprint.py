"""Refactor regression fingerprint.

Walks the kept `artifacts/bench*` and
`artifacts/q_distribution_shift/` trees and emits a single JSON
fingerprint of every F1 metric and Q-drift summary value. Used to verify
that a refactor of the code organisation produces byte-identical numbers
(modulo a small float tolerance) compared to the pre-refactor baseline.

Usage:
    python -m tests.regression_fingerprint write \
        --out tests/baselines/fingerprint_pre.json

    python -m tests.regression_fingerprint check \
        --baseline tests/baselines/fingerprint_pre.json \
        [--current tests/baselines/fingerprint_post.json]

The `write` form is read-only on artifacts: it parses existing JSONs and
emits one new JSON. It never touches the source files. The `check` form
diffs two fingerprint JSONs numerically.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARTS = REPO / "artifacts"

BENCH_ROOTS = [
    ARTS / "bench",
    ARTS / "bench_llama_verify",
    ARTS / "bench_llama_compact9",
    ARTS / "bench_llama_lcconly",
]
Q_DRIFT_JSON = ARTS / "q_distribution_shift" / "per_task_drift.json"


def _read_f1(metrics_path: Path) -> float | None:
    raw = metrics_path.read_text().strip()
    try:
        return float(raw)
    except ValueError:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return None
        for key in ("f1", "score", "metric", "value"):
            if isinstance(obj, dict) and key in obj:
                return float(obj[key])
        return None


def _collect_f1(roots: list[Path]) -> dict[str, float]:
    out: dict[str, float] = {}
    for root in roots:
        if not root.exists():
            continue
        for metrics_path in sorted(root.rglob("metrics.json")):
            try:
                f1 = _read_f1(metrics_path)
            except Exception:
                continue
            if f1 is None:
                continue
            rel = metrics_path.relative_to(ARTS).as_posix()
            out[rel] = round(f1, 6)
    return out


def _collect_q_drift(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out: dict[str, dict[str, float]] = {}
    for task, td in (data.get("tasks") or {}).items():
        entry: dict[str, float] = {}
        pp = td.get("prefill_points") or []
        if pp:
            entry["prefill_mean_cos"] = round(
                sum(p["cos"] for p in pp) / len(pp), 6
            )
        dbins = td.get("decode_bins") or []
        if dbins:
            total = sum(b.get("n_samples", 0) for b in dbins)
            if total > 0:
                entry["decode_mean_cos"] = round(
                    sum(b["mean_cos"] * b.get("n_samples", 0) for b in dbins)
                    / total,
                    6,
                )
        if entry:
            out[task] = entry
    return out


def build_fingerprint() -> dict:
    return {
        "schema_version": 1,
        "f1": _collect_f1(BENCH_ROOTS),
        "q_drift": _collect_q_drift(Q_DRIFT_JSON),
    }


def cmd_write(args: argparse.Namespace) -> int:
    fp = build_fingerprint()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fp, indent=2, sort_keys=True) + "\n")
    n_f1 = len(fp["f1"])
    n_q = len(fp["q_drift"])
    print(f"wrote {out} ({n_f1} F1 cells, {n_q} q_drift tasks)")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    baseline = json.loads(Path(args.baseline).read_text())
    if args.current:
        current = json.loads(Path(args.current).read_text())
    else:
        current = build_fingerprint()

    tol = float(args.tol)
    failures: list[str] = []

    base_f1 = baseline.get("f1", {})
    cur_f1 = current.get("f1", {})
    shared_keys = set(base_f1) & set(cur_f1)
    only_base = set(base_f1) - set(cur_f1)
    only_cur = set(cur_f1) - set(base_f1)

    for k in sorted(shared_keys):
        b, c = base_f1[k], cur_f1[k]
        if not math.isclose(b, c, abs_tol=tol):
            failures.append(f"F1 mismatch [{k}]: baseline={b} current={c} (Δ={c-b:+.4f})")

    if only_base and not args.allow_dropped:
        for k in sorted(only_base):
            failures.append(f"F1 dropped (in baseline, missing now) [{k}]: {base_f1[k]}")

    base_q = baseline.get("q_drift", {})
    cur_q = current.get("q_drift", {})
    for task in sorted(set(base_q) & set(cur_q)):
        for field in sorted(set(base_q[task]) | set(cur_q[task])):
            b = base_q[task].get(field)
            c = cur_q[task].get(field)
            if b is None or c is None:
                failures.append(f"q_drift field missing [{task}.{field}]: baseline={b} current={c}")
                continue
            if not math.isclose(b, c, abs_tol=1e-4):
                failures.append(f"q_drift mismatch [{task}.{field}]: baseline={b} current={c}")

    if only_cur:
        print(f"note: {len(only_cur)} new F1 cells appeared (not in baseline) — informational only")
        for k in sorted(only_cur)[:5]:
            print(f"  + {k} = {cur_f1[k]}")

    if failures:
        print(f"FINGERPRINT CHECK FAILED — {len(failures)} discrepancies:")
        for f in failures[:30]:
            print(f"  {f}")
        if len(failures) > 30:
            print(f"  ... and {len(failures) - 30} more")
        return 1

    n = len(shared_keys) + sum(len(v) for v in base_q.values())
    print(f"FINGERPRINT CHECK PASSED — {n} values match within tolerance (F1 tol={tol}, q_drift tol=1e-4)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("write", help="emit a fingerprint JSON")
    w.add_argument("--out", required=True, help="output JSON path")
    w.set_defaults(func=cmd_write)

    c = sub.add_parser("check", help="compare current artifacts against a baseline fingerprint")
    c.add_argument("--baseline", required=True, help="baseline JSON to compare against")
    c.add_argument("--current", default=None, help="optional current JSON (default: build from artifacts now)")
    c.add_argument("--tol", default="0.05", help="absolute F1 tolerance (default 0.05)")
    c.add_argument("--allow-dropped", action="store_true",
                   help="treat keys present in baseline but missing now as informational, not failure")
    c.set_defaults(func=cmd_check)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
