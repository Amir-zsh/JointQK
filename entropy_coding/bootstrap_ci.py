#!/usr/bin/env python3
"""Bootstrap 95% CIs for LongBench cells from the per-example scores already stored in
each predictions.csv -- no GPU, no re-running the model. Reuses the repo's EXACT
per-task metric (dataset2metric: code_sim for lcc, qa_f1 for musique/2wikimqa, ...).

Usage:
  python entropy_coding/bootstrap_ci.py artifacts/<bench_dir> [--order m1 m2 ...]
"""
import sys, glob, ast, argparse
from pathlib import Path
import numpy as np, pandas as pd

_LB = Path(__file__).resolve().parents[1] / "vendor/kvpress/evaluation/benchmarks/longbench"
sys.path.insert(0, str(_LB))
from calculate_metrics import dataset2metric  # noqa: E402

TASKS = ["lcc", "2wikimqa", "musique"]

# The worker's predictions.csv mangles multi-element gold `answers` (a numpy array) into a
# single space-joined string on write, so re-scoring the CSV `answers` column undercounts
# any row with >1 acceptable answer (~27% of musique, ~0% of lcc). We therefore fetch the
# true gold answers from the LongBench dataset by `_id`; this reproduces the worker's
# metrics.json exactly (predicted_answer itself round-trips fine). Cached per task.
_ANS_CACHE = {}


def _gold_answers(task):
    if task not in _ANS_CACHE:
        from datasets import load_dataset
        ds = load_dataset("Xnhyacinth/LongBench", data_dir=task, split="test").to_pandas()
        _ANS_CACHE[task] = {i: list(a) for i, a in zip(ds["_id"], ds["answers"])}
    return _ANS_CACHE[task]


def per_example(csv):
    df = pd.read_csv(csv)
    task = df["task"].tolist()[0]
    allc = df["all_classes"].tolist()[0]
    if isinstance(allc, str):
        try: allc = ast.literal_eval(allc)
        except Exception: allc = None
    metric = dataset2metric[task]
    try:
        gold = _gold_answers(task)                       # true (un-mangled) golds by _id
    except Exception:
        gold = None                                      # offline fallback -> CSV answers
    out = []
    for _id, pred, ans in zip(df["_id"].tolist(), df["predicted_answer"].tolist(), df["answers"].tolist()):
        golds = gold.get(_id) if gold is not None else None
        if golds is None:                                # fallback: parse the (lossy) CSV column
            golds = ast.literal_eval(ans) if isinstance(ans, str) else ans
            if not isinstance(golds, list): golds = [golds]
        s = 0.0
        for g in golds:
            s = max(s, metric(str(pred).lstrip(), str(g), all_classes=allc))
        out.append(100.0 * s)
    return np.array(out)


def boot(s, n=10000, seed=0):
    if len(s) == 0: return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    m = s[rng.integers(0, len(s), (n, len(s)))].mean(1)
    return float(s.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def find_cell(mdir, task):
    for p in Path(mdir).rglob("predictions.csv"):
        if f"__{task}__" in str(p): return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--order", nargs="*", default=None)
    ap.add_argument("--reps", type=int, default=10000)
    args = ap.parse_args()
    root = Path(args.root)
    methods = sorted(d.name for d in root.iterdir() if d.is_dir())
    if args.order:
        methods = [m for m in args.order if m in methods] + [m for m in methods if m not in args.order]
    w = max((len(m) for m in methods), default=10)
    hdr = f"{'method':<{w}} | " + " | ".join(f"{t:^16}" for t in TASKS) + " |     mean"
    print(hdr); print("-" * len(hdr))
    for m in methods:
        cells, means, ns = {}, [], []
        for t in TASKS:
            c = find_cell(root / m, t)
            if c is None: cells[t] = None; continue
            s = per_example(c); mean, lo, hi = boot(s, args.reps)
            cells[t] = (mean, lo, hi, len(s)); means.append(s)
        row = f"{m:<{w}} | "
        for t in TASKS:
            row += (f"{cells[t][0]:5.1f} ±{ (cells[t][2]-cells[t][1])/2:4.1f}     " if cells[t] else f"{'—':^16}") + "| "
        # mean over the 3 task means, with CI from bootstrapping the pooled per-task resample
        if all(cells[t] for t in TASKS):
            rng = np.random.default_rng(1)
            per_task_boot = np.stack([means[i][rng.integers(0, len(means[i]), (args.reps, len(means[i]))) ].mean(1) for i in range(3)])
            mm = per_task_boot.mean(0)
            row += f"{np.mean([cells[t][0] for t in TASKS]):5.1f} ±{(np.percentile(mm,97.5)-np.percentile(mm,2.5))/2:4.1f}"
        print(row)
    # sample sizes (once)
    ex = next((find_cell(root / methods[0], t) for t in TASKS if find_cell(root / methods[0], t)), None)
    print(f"\nsamples/task: " + ", ".join(
        f"{t}={len(per_example(find_cell(root/methods[0],t))) if find_cell(root/methods[0],t) else '?'}" for t in TASKS))


if __name__ == "__main__":
    main()
