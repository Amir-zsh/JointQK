#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.stage1.calibration.common import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_KEEP_RAW,
    DEFAULT_RUN_ID,
    DEFAULT_SPLIT,
    KEEP_RAW_CHOICES,
    RunPaths,
    parse_gpus,
)


REPO = Path(__file__).resolve().parents[3]
PY = REPO / ".venv/bin/python"
LAUNCHER = REPO / "experiments/stage1/scripts/parallel_launcher.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch final calibration stages across multiple GPUs.")
    parser.add_argument("--stage", choices=("capture", "stats", "analysis", "charts", "all"), required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--num-shards", default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--keep-raw",
        choices=KEEP_RAW_CHOICES,
        default=DEFAULT_KEEP_RAW,
        help="Forwarded to capture: which examples to keep raw fp16 q/k/v for. Default 'test' "
        "drops raw for train examples (used only for moments) and keeps raw for test (needed "
        "by empirical eval). Use 'none' for max IO savings, 'all' for full raw retention.",
    )
    parser.add_argument(
        "--empirical",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute empirical raw-tensor metrics during analysis (default ON). "
        "Pass --no-empirical to skip empirical eval and run analytic-only.",
    )
    parser.add_argument("--empirical-max-eval-examples", type=int, default=0)
    parser.add_argument("--empirical-max-layers", type=int, default=0)
    parser.add_argument("--empirical-max-heads", type=int, default=0)
    parser.add_argument("--empirical-max-tokens", type=int, default=0)
    parser.add_argument("--max-coord-bits", type=int, default=8)
    parser.add_argument(
        "--sample-sizes",
        default=None,
        help="Forwarded to analyze (e.g. '10,30,50'). Default = analyze's own default sweep.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=None,
        help="Forwarded to analyze. Default = analyze's own default (20).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def shell_quote(value: str | Path) -> str:
    s = str(value)
    return "'" + s.replace("'", "'\"'\"'") + "'"


def shard_count(args: argparse.Namespace) -> int:
    if args.num_shards == "auto":
        return len(parse_gpus(args.gpus)) * args.jobs_per_gpu
    n = int(args.num_shards)
    if n < 1:
        raise ValueError("--num-shards must be auto or >= 1")
    return n


def common_flags(args: argparse.Namespace, n_shards: int, shard_id: int | None = None) -> str:
    flags = [
        f"--split-manifest {shell_quote(args.split_manifest)}",
        f"--artifact-root {shell_quote(args.artifact_root)}",
        f"--run-id {shell_quote(args.run_id)}",
        f"--num-shards {n_shards}",
    ]
    if shard_id is not None:
        flags.append(f"--shard-id {shard_id}")
    if args.smoke:
        flags.append("--smoke")
    if args.resume:
        flags.append("--resume")
    return " ".join(flags)


def write_commands(args: argparse.Namespace, stage: str, n_shards: int) -> Path:
    paths = RunPaths.from_args(args.artifact_root, args.run_id)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    commands = paths.logs_dir / f"{stage}_commands.txt"
    script = {
        "capture": REPO / "experiments/stage1/calibration/capture_raw.py",
        "stats": REPO / "experiments/stage1/calibration/compute_stats.py",
        "analysis": REPO / "experiments/stage1/calibration/analyze_bases.py",
    }[stage]
    extra = ""
    if stage in {"stats", "analysis"}:
        extra += " --device cuda"
    if stage == "capture":
        extra += f" --keep-raw {args.keep_raw}"
    if stage == "analysis":
        extra += " --empirical" if args.empirical else " --no-empirical"
    if stage == "analysis" and args.empirical_max_eval_examples > 0:
        extra += f" --empirical-max-eval-examples {args.empirical_max_eval_examples}"
    if stage == "analysis" and args.empirical_max_layers > 0:
        extra += f" --empirical-max-layers {args.empirical_max_layers}"
    if stage == "analysis" and args.empirical_max_heads > 0:
        extra += f" --empirical-max-heads {args.empirical_max_heads}"
    if stage == "analysis" and args.empirical_max_tokens > 0:
        extra += f" --empirical-max-tokens {args.empirical_max_tokens}"
    if stage == "analysis":
        extra += f" --max-coord-bits {args.max_coord_bits}"
    if stage == "analysis" and args.sample_sizes is not None:
        extra += f" --sample-sizes {shell_quote(args.sample_sizes)}"
    if stage == "analysis" and args.repetitions is not None:
        extra += f" --repetitions {int(args.repetitions)}"
    with commands.open("w") as f:
        for shard_id in range(n_shards):
            flags = common_flags(args, n_shards, shard_id)
            label = f"{stage}_shard{shard_id:03d}"
            f.write(f"cd {shell_quote(REPO)} && {shell_quote(PY)} {shell_quote(script)} {flags}{extra} # label={label}\n")
    return commands


def run_parallel(args: argparse.Namespace, stage: str, n_shards: int) -> None:
    commands = write_commands(args, stage, n_shards)
    log_dir = RunPaths.from_args(args.artifact_root, args.run_id).logs_dir / stage
    cmd = [
        str(PY),
        str(LAUNCHER),
        "--commands-file",
        str(commands),
        "--log-dir",
        str(log_dir),
        "--gpus",
        args.gpus,
        "--jobs-per-gpu",
        str(args.jobs_per_gpu),
        "--label-from-comment",
    ]
    print(" ".join(map(str, cmd)))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


def run_cmd(cmd: list[str], dry_run: bool) -> None:
    print(" ".join(map(str, cmd)))
    if not dry_run:
        subprocess.run(cmd, check=True)


def run_stage(args: argparse.Namespace, stage: str, n_shards: int) -> None:
    flags = [
        "--split-manifest",
        str(args.split_manifest),
        "--artifact-root",
        str(args.artifact_root),
        "--run-id",
        args.run_id,
        "--num-shards",
        str(n_shards),
    ]
    if args.smoke:
        flags.append("--smoke")
    # capture writes per-example stats inline as of 2026-05-05; the stats stage is just merge + validate.
    # If the user explicitly wants the legacy per-example rebuild from raw, they can invoke
    # `compute_stats.py --rebuild-from-raw` directly; the launcher no longer parallelizes that path.
    if stage in {"capture", "analysis"}:
        run_parallel(args, stage, n_shards)
    if stage == "stats":
        run_cmd([str(PY), str(REPO / "experiments/stage1/calibration/compute_stats.py"), *flags], args.dry_run)
        run_cmd([str(PY), str(REPO / "experiments/stage1/calibration/validate_artifacts.py"), "--stage", "stats", *flags], args.dry_run)
    elif stage == "analysis":
        extra = ["--empirical" if args.empirical else "--no-empirical"]
        if args.empirical_max_eval_examples > 0:
            extra.extend(["--empirical-max-eval-examples", str(args.empirical_max_eval_examples)])
        if args.empirical_max_layers > 0:
            extra.extend(["--empirical-max-layers", str(args.empirical_max_layers)])
        if args.empirical_max_heads > 0:
            extra.extend(["--empirical-max-heads", str(args.empirical_max_heads)])
        if args.empirical_max_tokens > 0:
            extra.extend(["--empirical-max-tokens", str(args.empirical_max_tokens)])
        extra.extend(["--max-coord-bits", str(args.max_coord_bits)])
        if args.sample_sizes is not None:
            extra.extend(["--sample-sizes", str(args.sample_sizes)])
        if args.repetitions is not None:
            extra.extend(["--repetitions", str(args.repetitions)])
        run_cmd([str(PY), str(REPO / "experiments/stage1/calibration/analyze_bases.py"), *flags, "--merge-only", *extra], args.dry_run)
        run_cmd([str(PY), str(REPO / "experiments/stage1/calibration/validate_artifacts.py"), "--stage", "analysis", *flags], args.dry_run)
    elif stage == "capture":
        run_cmd([str(PY), str(REPO / "experiments/stage1/calibration/validate_artifacts.py"), "--stage", "raw", *flags], args.dry_run)
    elif stage == "charts":
        run_cmd([str(PY), str(REPO / "experiments/stage1/calibration/make_charts.py"), "--artifact-root", str(args.artifact_root), "--run-id", args.run_id], args.dry_run)
        run_cmd([str(PY), str(REPO / "experiments/stage1/calibration/validate_artifacts.py"), "--stage", "charts", *flags], args.dry_run)


def main() -> None:
    args = parse_args()
    n_shards = shard_count(args)
    stages = ["capture", "stats", "analysis", "charts"] if args.stage == "all" else [args.stage]
    base_flags = [
        "--split-manifest",
        str(args.split_manifest),
        "--artifact-root",
        str(args.artifact_root),
        "--run-id",
        args.run_id,
        "--num-shards",
        str(n_shards),
    ]
    if args.smoke:
        base_flags.append("--smoke")
    run_cmd([str(PY), str(REPO / "experiments/stage1/calibration/validate_artifacts.py"), "--stage", "split", *base_flags], args.dry_run)
    for stage in stages:
        run_stage(args, stage, n_shards)


if __name__ == "__main__":
    main()
