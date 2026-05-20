#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.calibration.common import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_KEEP_RAW,
    DEFAULT_RUN_ID,
    DEFAULT_SPLIT,
    KEEP_RAW_CHOICES,
    RAW_TENSORS,
    RunPaths,
    assign_shards,
    estimate_raw_bytes,
    log,
    read_json,
    read_jsonl,
    select_rows,
    should_keep_raw,
    stable_example_id,
    stats_example_path,
    validate_shard_args,
    validate_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate final calibration pipeline artifacts.")
    parser.add_argument("--stage", choices=("split", "raw", "stats", "analysis", "charts"), required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--keep-raw",
        choices=KEEP_RAW_CHOICES,
        default=DEFAULT_KEEP_RAW,
        help="Validation mode for raw stage: only require raw files for examples whose split "
        "matches the keep-raw policy. Defaults match the capture-time default.",
    )
    return parser.parse_args()


def expected_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    split = read_json(args.split_manifest)
    validate_split(split)
    rows = select_rows(split, args.smoke)
    return assign_shards(rows, args.num_shards)


def validate_split_stage(args: argparse.Namespace) -> None:
    rows = expected_rows(args)
    by_task_split = Counter((row["config"], row["split"]) for row in rows)
    ids = [stable_example_id(row) for row in rows]
    if len(ids) != len(set(ids)):
        dupes = [item for item, count in Counter(ids).items() if count > 1]
        raise RuntimeError(f"Duplicate stable ids: {dupes[:10]}")
    estimate = estimate_raw_bytes(rows, include_v=True)
    log(f"split ok: rows={len(rows)} estimated_raw_qkv={estimate / 1e9:.1f} GB")
    for key in sorted(by_task_split):
        log(f"  {key[0]} {key[1]}: {by_task_split[key]}")


def validate_raw_stage(args: argparse.Namespace) -> None:
    rows = expected_rows(args)
    paths = RunPaths.from_args(args.artifact_root, args.run_id)
    missing: list[str] = []
    expected = [r for r in rows if should_keep_raw(r, args.keep_raw)]
    for row in expected:
        raw_path = paths.raw_dir / f"shard_{int(row['assigned_shard']):03d}" / f"{stable_example_id(row)}.pt"
        if not raw_path.exists():
            missing.append(str(raw_path))
            continue
        payload = torch.load(raw_path, map_location="cpu", weights_only=False)
        for name in RAW_TENSORS:
            if name not in payload:
                raise RuntimeError(f"{raw_path} missing tensor {name}")
            tensor = payload[name]
            if tensor.dtype != torch.float16:
                raise RuntimeError(f"{raw_path} {name} dtype={tensor.dtype}, expected float16")
            if int(tensor.shape[2]) != int(payload["prompt_length"]):
                raise RuntimeError(f"{raw_path} {name} seq={tensor.shape[2]} prompt={payload['prompt_length']}")
        if int(payload["row_index"]) != int(row["row_index"]) or payload["config"] != row["config"]:
            raise RuntimeError(f"{raw_path} row metadata mismatch")
    if missing:
        raise RuntimeError(f"Missing {len(missing)} raw files; first={missing[0]}")
    skipped = len(rows) - len(expected)
    log(f"raw ok: {len(expected)} files validated (skipped {skipped} per keep-raw={args.keep_raw})")


def _check_symmetric(name: str, tensor: torch.Tensor, tol: float = 2e-3) -> None:
    diff = (tensor.float() - tensor.float().transpose(-1, -2)).abs().max().item()
    if diff > tol:
        raise RuntimeError(f"{name} not symmetric: max_abs_diff={diff:.4e}")


def validate_stats_stage(args: argparse.Namespace) -> None:
    rows = expected_rows(args)
    paths = RunPaths.from_args(args.artifact_root, args.run_id)
    missing = []
    for row in rows:
        path = stats_example_path(paths, row)
        if not path.exists():
            missing.append(str(path))
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for name in ("sq_sum", "sk_sum", "sv_sum"):
            if name not in payload:
                raise RuntimeError(f"{path} missing {name}")
            if not torch.isfinite(payload[name]).all():
                raise RuntimeError(f"{path} {name} has non-finite values")
            _check_symmetric(name, payload[name])
        for name in ("sum_q", "sum_k", "sum_v", "tokens"):
            if name not in payload:
                raise RuntimeError(f"{path} missing {name}")
            if name != "tokens" and not torch.isfinite(payload[name]).all():
                raise RuntimeError(f"{path} {name} has non-finite values")
        # The manifest's prompt_length is tokenizer-locked (computed when the split
        # was created, typically under Qwen3). Captures under a different model use a
        # different tokenizer → different token count for the same text. Only enforce
        # strict equality when capture used the same tokenizer as the manifest, which
        # we detect via the tokenizer-locked prompt_sha256. Otherwise just sanity-check
        # that tokens is a positive int.
        if int(payload["tokens"]) <= 0:
            raise RuntimeError(f"{path} non-positive tokens: {payload['tokens']}")
        same_tokenizer = (
            "prompt_sha256" in payload
            and "prompt_sha256" in row
            and payload["prompt_sha256"] == row["prompt_sha256"]
        )
        if same_tokenizer and int(payload["tokens"]) != int(row["prompt_length"]):
            raise RuntimeError(
                f"{path} token mismatch under same-tokenizer capture: "
                f"payload={payload['tokens']} vs manifest={row['prompt_length']}"
            )
    if missing:
        raise RuntimeError(f"Missing {len(missing)} stats files; first={missing[0]}")
    aggregate = paths.stats_dir / "aggregate.pt"
    if not aggregate.exists():
        raise RuntimeError(f"Missing aggregate stats: {aggregate}")
    agg = torch.load(aggregate, map_location="cpu", weights_only=False)
    for name in ("per_example", "groups", "metadata"):
        if name not in agg:
            raise RuntimeError(f"aggregate missing {name}")
    for group_name, group in agg["groups"].items():
        for name in ("mu_q", "mu_k", "mu_v", "cov_q", "cov_k", "cov_v"):
            if name not in group:
                raise RuntimeError(f"aggregate group {group_name} missing {name}")
            if not torch.isfinite(group[name]).all():
                raise RuntimeError(f"aggregate group {group_name} {name} has non-finite values")
    log(f"stats ok: {len(rows)} per-example files + aggregate validated")


def validate_analysis_stage(args: argparse.Namespace) -> None:
    paths = RunPaths.from_args(args.artifact_root, args.run_id)
    rows_path = paths.analysis_dir / "analysis_rows.jsonl"
    summary_path = paths.analysis_dir / "analysis_summary.json"
    if not rows_path.exists() or not summary_path.exists():
        raise RuntimeError(f"Missing analysis outputs under {paths.analysis_dir}")
    rows = read_jsonl(rows_path)
    if not rows:
        raise RuntimeError("analysis_rows.jsonl is empty")
    trial_counts = Counter(row["trial_idx"] for row in rows)
    # Each trial emits multiple eval rows, so completeness is checked via summary.
    summary = read_json(summary_path)
    expected = int(summary["metadata"]["expected_trials"])
    observed = sorted(set(int(row["trial_idx"]) for row in rows))
    if observed != list(range(expected)):
        missing = sorted(set(range(expected)) - set(observed))
        raise RuntimeError(f"Analysis trial ids incomplete: expected={expected} missing_first={missing[:10]}")
    methods = {row["method"] for row in rows}
    required = {"v3", "q_only", "k_only", "jointqk", "v_random", "v_eigen_uniform", "v_eigen_waterfill"}
    if not required.issubset(methods):
        raise RuntimeError(f"Missing methods: {sorted(required - methods)}")
    log(f"analysis ok: rows={len(rows)} trials={len(trial_counts)} methods={sorted(methods)}")


def validate_charts_stage(args: argparse.Namespace) -> None:
    paths = RunPaths.from_args(args.artifact_root, args.run_id)
    expected = [
        "k_pooled_logit_error_k3.png",
        "k_method_comparison_k3.png",
        "k_pooled_mse_k3.png",
        "k_mse_method_comparison_k3.png",
        "k_waterfill_vs_uniform_logit_k3.png",
        "k_waterfill_vs_uniform_mse_k3.png",
        "k_waterfill_active_coords_k3.png",
        "v_pooled_mse_v3.png",
        "layer0_sensitivity.png",
        "README.md",
    ]
    missing = []
    for name in expected:
        path = paths.reports_dir / name
        if not path.exists() or path.stat().st_size == 0:
            missing.append(str(path))
    if missing:
        raise RuntimeError(f"Missing/empty chart outputs: {missing}")
    log(f"charts ok: {len(expected)} report files validated")


def main() -> None:
    args = parse_args()
    validate_shard_args(args.num_shards, 0)
    if args.stage == "split":
        validate_split_stage(args)
    elif args.stage == "raw":
        validate_raw_stage(args)
    elif args.stage == "stats":
        validate_stats_stage(args)
    elif args.stage == "analysis":
        validate_analysis_stage(args)
    elif args.stage == "charts":
        validate_charts_stage(args)
    else:
        raise AssertionError(args.stage)


if __name__ == "__main__":
    main()
