#!/usr/bin/env python3
"""Snapshot raw per-sample generations into a durable, provenance-tracked archive.

Data hygiene for a long-running study: the scored numbers depend on raw generations
that (a) live partly in a read-only external tree we don't control and (b) are scattered
across incremental per-seed run dirs. This copies the raw artifacts (io_log.jsonl +
metrics.json + predictions.csv + server_info.json) for every run dir under the given
source roots into ARCHIVE/<tag>/<relpath>/, and records provenance + a re-derived
(n_seeds, mean, std) per run in MANIFEST.jsonl.

Idempotent: re-run any time to catch new/finished runs (skips byte-identical copies).
Once archived, per_k / io_log let us re-aggregate to ANY seed count (e.g. downcast a
6-seed HumanEval to 4) without ever re-generating.

Usage:
  python pipelines/eval/archive_raw_generations.py            # default roots
  python pipelines/eval/archive_raw_generations.py --roots <dir> [<dir> ...]
"""
import argparse
import hashlib
import json
import shutil
import statistics as st
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = REPO / "artifacts" / "raw_archive"
FILES = ["io_log.jsonl", "metrics.json", "predictions.csv", "server_info.json"]

# (tag, root) — tag namespaces the source so our runs and the external read-only tree
# never collide in the archive.
DEFAULT_ROOTS = [
    ("ours", REPO / "artifacts" / "oscar_e2e"),
    ("amir_lh", Path("/vault/amir/efficient-llm/teamily-project/artifacts/oscar_e2e/lh")),
    ("amir_top", Path("/vault/amir/efficient-llm/teamily-project/artifacts/oscar_e2e")),
]


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _seed_stats(run: Path):
    m = run / "metrics.json"
    if not m.exists():
        return None
    try:
        d = json.loads(m.read_text())
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict):
        return None
    pk = d.get("per_k")
    if pk:
        a = [x["accuracy"] for x in pk]
        return len(a), st.mean(a), (st.stdev(a) if len(a) > 1 else 0.0)
    if "accuracy" in d:
        return 1, d["accuracy"], 0.0
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=None,
                    help="source dirs (default: ours + amir trees)")
    args = ap.parse_args()
    roots = ([("cli", Path(r)) for r in args.roots] if args.roots else DEFAULT_ROOTS)

    ARCHIVE.mkdir(parents=True, exist_ok=True)
    manifest = ARCHIVE / "MANIFEST.jsonl"
    seen = set()  # (tag, relpath) already recorded this session
    n_runs = n_copied = 0
    lines = []
    for tag, root in roots:
        if not root.exists():
            print(f"  skip {tag}: {root} missing", flush=True)
            continue
        for io in sorted(root.rglob("io_log.jsonl")):
            run = io.parent
            rel = run.relative_to(root)
            key = (tag, str(rel))
            if key in seen:
                continue
            seen.add(key)
            dst = ARCHIVE / tag / rel
            dst.mkdir(parents=True, exist_ok=True)
            copied = []
            for fn in FILES:
                src = run / fn
                if not src.exists():
                    continue
                d = dst / fn
                if not d.exists() or _sha(src) != _sha(d):
                    shutil.copy2(src, d)
                    copied.append(fn)
            n_runs += 1
            n_copied += bool(copied)
            stats = _seed_stats(run)
            lines.append(json.dumps({
                "tag": tag, "run": str(rel), "source": str(run),
                "n_seeds": stats[0] if stats else None,
                "mean_acc": round(stats[1], 4) if stats else None,
                "std_acc": round(stats[2], 4) if stats else None,
                "copied": copied, "archived_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }))
    manifest.write_text("\n".join(lines) + "\n")
    print(f"archived {n_runs} run dirs ({n_copied} had new/changed files) -> {ARCHIVE}", flush=True)
    print(f"manifest: {manifest} ({len(lines)} entries)", flush=True)


if __name__ == "__main__":
    main()
