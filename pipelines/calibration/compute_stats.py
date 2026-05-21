#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipelines.calibration.common import (
    RunPaths,
    add_common_args,
    add_shard_args,
    log,
    raw_example_path,
    read_json,
    rows_for_shard,
    select_rows,
    stable_example_id,
    stats_example_path,
    validate_shard_args,
    validate_split,
    write_json,
    write_stage_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per-example Q/K/V moments from raw calibration tensors.")
    add_common_args(parser)
    add_shard_args(parser)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument(
        "--rebuild-from-raw",
        action="store_true",
        help="Recompute per-example stats from on-disk raw tensors. Default OFF — capture_raw.py "
        "writes stats inline, so per-example computation here is only needed for legacy artifacts "
        "or after changing the moments accumulator.",
    )
    return parser.parse_args()


@torch.no_grad()
def compute_moments_from_tensors(
    q_post: torch.Tensor,
    k_post: torch.Tensor,
    v: torch.Tensor,
    prompt_length: int,
    device: torch.device,
    *,
    gpu_budget_bytes: int = 8 << 30,
) -> dict[str, torch.Tensor | int]:
    """Compute per-(layer, kv_head) second-moment sums from in-memory fp16 tensors.

    Pure helper — no I/O. Used by both `capture_raw.py` (inline, fused with model forward) and
    `compute_example_stats` (legacy: load raw `.pt`, call this).

    Inputs:
      q_post: (n_layers, n_q_heads, T_full, d), any device, any float dtype.
      k_post: (n_layers, n_kv_heads, T_full, d).
      v:      (n_layers, n_kv_heads, T_full, d).
      prompt_length: T to slice (tensors may be longer than T).
      device: where to do the einsum (CPU or CUDA).
      gpu_budget_bytes: target peak GPU memory for the fp32 promotion (per chunk). Tunes
          chunk_layers so peak per-chunk fp32 footprint stays under this budget.

    Returns CPU fp32 tensors with shape (n_layers, n_kv_heads, d, d) for sq/sk/cqk/sv and
    (n_layers, n_kv_heads, d) for sum_q/sum_k/sum_v, plus shape metadata.

    Implementation: moves the whole fp16 q/k/v slab to `device` once (cheap — typically a few
    GB), then promotes and processes layer chunks via batched einsums. Compared to the per-layer
    loop this cuts ~108 host→device transfers and ~180 small einsum kernel launches per example
    down to ~3 transfers and ~5 launches per chunk.
    """
    n_layers = int(k_post.shape[0])
    n_kv_heads = int(k_post.shape[1])
    n_q_heads = int(q_post.shape[1])
    head_dim = int(k_post.shape[-1])
    tokens = int(prompt_length)
    if n_q_heads % n_kv_heads != 0:
        raise ValueError(f"GQA mismatch: q_heads={n_q_heads}, kv_heads={n_kv_heads}")
    group = n_q_heads // n_kv_heads

    shape = (n_layers, n_kv_heads, head_dim, head_dim)
    sq_sum = torch.empty(shape, dtype=torch.float32)
    sk_sum = torch.empty(shape, dtype=torch.float32)
    cqk_sum = torch.empty(shape, dtype=torch.float32)
    sv_sum = torch.empty(shape, dtype=torch.float32)
    sum_q = torch.empty((n_layers, n_kv_heads, head_dim), dtype=torch.float32)
    sum_k = torch.empty((n_layers, n_kv_heads, head_dim), dtype=torch.float32)
    sum_v = torch.empty((n_layers, n_kv_heads, head_dim), dtype=torch.float32)

    # Move the whole fp16 slab to device once, then promote per chunk.
    q_dev = q_post[:, :, :tokens, :].to(device, non_blocking=True)
    k_dev = k_post[:, :, :tokens, :].to(device, non_blocking=True)
    v_dev = v[:, :, :tokens, :].to(device, non_blocking=True)

    # Per-layer fp32 footprint (q + k + v). We size chunks so a chunk's fp32 promotion
    # comfortably fits in `gpu_budget_bytes`.
    bytes_per_layer_fp32 = (n_q_heads + 2 * n_kv_heads) * tokens * head_dim * 4
    chunk_layers = max(1, min(n_layers, gpu_budget_bytes // max(bytes_per_layer_fp32, 1)))

    for start in range(0, n_layers, chunk_layers):
        stop = min(start + chunk_layers, n_layers)
        n = stop - start
        q_f = q_dev[start:stop].float()
        k_f = k_dev[start:stop].float()
        v_f = v_dev[start:stop].float()

        sq = torch.einsum("lhsd,lhse->lhde", q_f, q_f)
        q_group = q_f.view(n, n_kv_heads, group, tokens, head_dim)
        sq = sq.view(n, n_kv_heads, group, head_dim, head_dim).sum(dim=2)
        sk = torch.einsum("lhsd,lhse->lhde", k_f, k_f)
        q_grouped = q_group.mean(dim=2)
        cqk = torch.einsum("lhsd,lhse->lhde", q_grouped, k_f)
        sv = torch.einsum("lhsd,lhse->lhde", v_f, v_f)
        sum_q_chunk = q_group.sum(dim=(2, 3))
        sum_k_chunk = k_f.sum(dim=2)
        sum_v_chunk = v_f.sum(dim=2)

        sq_sum[start:stop] = sq.cpu()
        sk_sum[start:stop] = sk.cpu()
        cqk_sum[start:stop] = cqk.cpu()
        sv_sum[start:stop] = sv.cpu()
        sum_q[start:stop] = sum_q_chunk.cpu()
        sum_k[start:stop] = sum_k_chunk.cpu()
        sum_v[start:stop] = sum_v_chunk.cpu()
        del q_f, k_f, v_f, q_group, q_grouped, sq, sk, cqk, sv, sum_q_chunk, sum_k_chunk, sum_v_chunk

    del q_dev, k_dev, v_dev

    return {
        "tokens": tokens,
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "n_q_heads": n_q_heads,
        "group": group,
        "head_dim": head_dim,
        "sq_sum": sq_sum,
        "sk_sum": sk_sum,
        "cqk_sum": cqk_sum,
        "sv_sum": sv_sum,
        "sum_q": sum_q,
        "sum_k": sum_k,
        "sum_v": sum_v,
    }


def build_stats_artifact(moments: dict[str, Any], row: dict[str, Any], prompt_sha256: str) -> dict[str, Any]:
    """Combine moment tensors with per-example metadata into the artifact saved to 02_stats/."""
    return {
        "dataset": row["dataset"],
        "config": row["config"],
        "task": row["task"],
        "split": row["split"],
        "row_index": int(row["row_index"]),
        "prompt_sha256": prompt_sha256,
        **moments,
    }


@torch.no_grad()
def compute_example_stats(raw_path: Path, row: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Legacy path: load raw tensors from disk and compute moments. Used by --rebuild-from-raw."""
    payload = torch.load(raw_path, map_location="cpu", weights_only=False)
    moments = compute_moments_from_tensors(
        payload["q_post"], payload["k_post"], payload["v"],
        prompt_length=int(payload["prompt_length"]),
        device=device,
    )
    return build_stats_artifact(moments, row, prompt_sha256=row.get("prompt_sha256") or payload.get("prompt_sha256", ""))


def _zero_like(item: dict[str, Any], key: str) -> torch.Tensor:
    return torch.zeros_like(item[key])


def merge_stats(paths: RunPaths, rows: list[dict[str, Any]]) -> None:
    per_example: list[dict[str, Any]] = []
    loaded: list[dict[str, Any]] = []
    for compact_idx, row in enumerate(rows):
        path = stats_example_path(paths, row)
        if not path.exists():
            raise RuntimeError(f"Missing stats file for merge: {path}")
        item = torch.load(path, map_location="cpu", weights_only=False)
        rel = str(path.relative_to(paths.root))
        per_example.append(
            {
                "index": compact_idx,
                "file": rel,
                "raw_file": rel.replace("02_stats/", "01_raw/"),
                "dataset": item["dataset"],
                "config": item["config"],
                "task": item["task"],
                "split": item["split"],
                "row_index": int(item["row_index"]),
                "tokens": int(item["tokens"]),
            }
        )
        loaded.append(item)

    groups: dict[str, dict[str, Any]] = {}

    def add_group(name: str, indices: list[int]) -> None:
        if not indices:
            return
        first = loaded[indices[0]]
        sq = _zero_like(first, "sq_sum")
        sk = _zero_like(first, "sk_sum")
        cqk = _zero_like(first, "cqk_sum")
        sv = _zero_like(first, "sv_sum")
        sum_q = _zero_like(first, "sum_q")
        sum_k = _zero_like(first, "sum_k")
        sum_v = _zero_like(first, "sum_v")
        tokens = 0
        for idx in indices:
            item = loaded[idx]
            sq += item["sq_sum"]
            sk += item["sk_sum"]
            cqk += item["cqk_sum"]
            sv += item["sv_sum"]
            sum_q += item["sum_q"]
            sum_k += item["sum_k"]
            sum_v += item["sum_v"]
            tokens += int(item["tokens"])
        group = int(first["group"])
        sigma_q = sq / float(group * tokens)
        sigma_k = sk / float(tokens)
        cqk_mean = cqk / float(tokens)
        sigma_v = sv / float(tokens)
        mu_q = sum_q / float(group * tokens)
        mu_k = sum_k / float(tokens)
        mu_v = sum_v / float(tokens)
        cov_q = sigma_q - torch.einsum("lhd,lhe->lhde", mu_q, mu_q)
        cov_k = sigma_k - torch.einsum("lhd,lhe->lhde", mu_k, mu_k)
        cov_v = sigma_v - torch.einsum("lhd,lhe->lhde", mu_v, mu_v)
        groups[name] = {
            "indices": indices,
            "tokens": tokens,
            "sigma_q": sigma_q,
            "sigma_k": sigma_k,
            "cqk": cqk_mean,
            "sigma_v": sigma_v,
            "mu_q": mu_q,
            "mu_k": mu_k,
            "mu_v": mu_v,
            "cov_q": cov_q,
            "cov_k": cov_k,
            "cov_v": cov_v,
        }

    by_task_split: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_split: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(per_example):
        by_task_split[(row["config"], row["split"])].append(i)
        by_split[row["split"]].append(i)
    for (task, split), indices in sorted(by_task_split.items()):
        add_group(f"{task}|{split}", indices)
    for split, indices in sorted(by_split.items()):
        add_group(f"pooled|{split}", indices)

    metadata = {
        "num_examples": len(per_example),
        "n_layers": int(loaded[0]["n_layers"]) if loaded else 0,
        "n_kv_heads": int(loaded[0]["n_kv_heads"]) if loaded else 0,
        "n_q_heads": int(loaded[0]["n_q_heads"]) if loaded else 0,
        "group": int(loaded[0]["group"]) if loaded else 0,
        "head_dim": int(loaded[0]["head_dim"]) if loaded else 0,
    }
    torch.save({"per_example": per_example, "groups": groups, "metadata": metadata}, paths.stats_dir / "aggregate.pt")
    write_json(
        paths.stats_dir / "aggregate_summary.json",
        {
            "metadata": metadata,
            "groups": {name: {"n": len(g["indices"]), "tokens": int(g["tokens"])} for name, g in groups.items()},
        },
    )
    log(f"wrote aggregate stats: {paths.stats_dir / 'aggregate.pt'}")


def main() -> None:
    args = parse_args()
    validate_shard_args(args.num_shards, args.shard_id)
    paths = RunPaths.from_args(args.artifact_root, args.run_id)
    paths.ensure()
    split = read_json(paths.split_dir / "manifest.json" if (paths.split_dir / "manifest.json").exists() else args.split_manifest)
    validate_split(split)
    all_rows = select_rows(split, args.smoke)
    assigned_rows: list[dict[str, Any]] = []
    for shard_id in range(args.num_shards):
        assigned_rows.extend(rows_for_shard(all_rows, args.num_shards, shard_id))

    # Default behaviour: stats are written inline by capture_raw.py, so this script just merges.
    # --rebuild-from-raw recomputes per-example stats from the raw fp16 tensors (legacy / repair path).
    if not args.rebuild_from_raw:
        merge_stats(paths, assigned_rows)
        return

    if args.merge_only:
        # Backward compatibility: --merge-only is a synonym for "skip per-example, just merge".
        merge_stats(paths, assigned_rows)
        return

    rows = rows_for_shard(all_rows, args.num_shards, args.shard_id)
    shard_dir = paths.stats_dir / f"shard_{args.shard_id:03d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    completed: list[dict[str, Any]] = []
    t_all = time.time()
    log(f"rebuild-from-raw shard {args.shard_id}/{args.num_shards}: {len(rows)} examples")
    for local_idx, row in enumerate(rows):
        row = dict(row)
        row["assigned_shard"] = args.shard_id
        out_path = stats_example_path(paths, row)
        if args.resume and out_path.exists():
            log(f"skip existing {local_idx + 1}/{len(rows)} {out_path.name}")
            completed.append({**row, "file": str(out_path.relative_to(paths.root))})
            continue
        raw_path = raw_example_path(paths, row)
        if not raw_path.exists():
            raise RuntimeError(f"Missing raw artifact (cannot --rebuild-from-raw without raw): {raw_path}")
        t0 = time.time()
        log(f"stats {local_idx + 1}/{len(rows)} id={stable_example_id(row)}")
        stats = compute_example_stats(raw_path, row, device)
        tmp = out_path.with_suffix(out_path.suffix + f".tmp.{args.shard_id}.{local_idx}")
        torch.save(stats, tmp)
        tmp.replace(out_path)
        completed.append({**row, "file": str(out_path.relative_to(paths.root))})
        log(f"wrote {out_path.name} in {time.time() - t0:.1f}s")
        del stats
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_stage_manifest(
        shard_dir / "manifest.json",
        stage="compute_stats",
        args=args,
        rows=completed,
        extra={"elapsed_seconds": time.time() - t_all},
    )
    log(f"complete stats shard={args.shard_id}/{args.num_shards}: {len(completed)} files")


if __name__ == "__main__":
    main()
