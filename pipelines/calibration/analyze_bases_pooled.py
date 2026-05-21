#!/usr/bin/env python3
"""Multi-GPU K-basis sweep — pooled regime, parallelized fast path to analyze_bases.py.

Two modes:

  1. Launcher (default): spawns one shard subprocess per visible GPU. Each shard processes
     a slice of the eval set and writes per-shard accumulator JSON. Launcher merges them
     and prints the final V3-vs-basis comparison.

         CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
         ./.venv/bin/python pipelines/calibration/analyze_bases_pooled.py

  2. Shard mode (`--shard-id N --num-shards K`): single-process. Loads its slice of the
     eval set, computes per-bits per-layer accumulators for v3 / q_only / k_only / jointqk,
     writes one JSON to `--out-dir/shard_<id>.json`. Launcher invokes this internally.

The headline N=50 pooled trial uses 400 train (calibration) + 80 test (eval) examples; with
6 shards each GPU handles ~13-14 eval idx (~70 GB raw streamed), bounded resident.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _bootstrap  # noqa: E402, F401

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipelines.calibration.common import RunPaths, StatsCache
from pipelines.calibration.analyze_bases import (
    K_METHODS,
    RawLRU,
    _finalize_k_accums,
    _finalize_v3_accums,
    combine_stats,
    eigvec_desc,
    empirical_k_metrics,
    empirical_v3_metrics,
    jointqk_basis,
    k_rows_for_method,
    merge_accumulators,
    v3_rows_for_method,
)


REPO = Path(__file__).resolve().parents[2]
PY = REPO / ".venv/bin/python"


def ts() -> str:
    return time.strftime("%H:%M:%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-GPU pooled-regime K-basis sweep (v3 / q_only / k_only / jointqk).")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/calibration"))
    parser.add_argument("--run-id", default="longbench_compact8_qkv_qwen3_8b")
    parser.add_argument("--k-bits", default="2,3,4")
    parser.add_argument("--methods", default="v3,q_only,k_only,jointqk")
    parser.add_argument("--gpus", default=None,
                        help="Comma-separated GPU ids for the launcher to spawn shards on. "
                             "Default = visible GPUs from CUDA_VISIBLE_DEVICES, or '0' if unset.")
    parser.add_argument("--max-eval-examples", type=int, default=0,
                        help="Cap eval examples (0 = all 80). Smaller is faster but noisier.")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("artifacts/calibration/longbench_compact8_qkv_qwen3_8b/05_reports/pooled_n50"))
    parser.add_argument("--num-shards", type=int, default=0, help="Internal: shard mode if >0.")
    parser.add_argument("--shard-id", type=int, default=-1, help="Internal: 0-based shard index.")
    parser.add_argument("--seed", type=int, default=20260505)
    parser.add_argument("--max-coord-bits", type=int, default=8)
    return parser.parse_args()


def discover_gpus(arg_gpus: str | None) -> list[str]:
    if arg_gpus:
        return [g.strip() for g in arg_gpus.split(",") if g.strip()]
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if env:
        return [g.strip() for g in env.split(",") if g.strip()]
    return ["0"]


def slice_for_shard(items: list[int], num_shards: int, shard_id: int) -> list[int]:
    return [x for i, x in enumerate(items) if i % num_shards == shard_id]


def load_inputs(args: argparse.Namespace, device: torch.device, k_bits: list[int]) -> dict[str, Any]:
    paths = RunPaths.from_args(args.artifact_root, args.run_id)
    print(f"[{ts()}] loading aggregate (~2.4 GB)...", flush=True)
    agg = torch.load(paths.stats_dir / "aggregate.pt", map_location="cpu", weights_only=False)
    per_example = agg["per_example"]
    train_indices = [int(p["index"]) for p in per_example if p["split"] == "train"]
    test_indices = [int(p["index"]) for p in per_example if p["split"] == "test"]
    print(f"[{ts()}] pooled regime: {len(train_indices)} train + {len(test_indices)} test", flush=True)

    # Bounded LRU on stats so we don't hold all 480 in RAM. Most reads are during the
    # combine_stats sums; after that the cache can be evicted.
    stats_cache = StatsCache(max_entries=128)
    print(f"[{ts()}] summing train stats over {len(train_indices)} examples...", flush=True)
    train_stats = combine_stats(paths, per_example, train_indices, device, cache=stats_cache)
    print(f"[{ts()}] summing eval stats over {len(test_indices)} examples...", flush=True)
    eval_stats = combine_stats(paths, per_example, test_indices, device, cache=stats_cache)
    cs = stats_cache.stats()
    print(f"[{ts()}] stats cache: {cs}", flush=True)
    # Drop stats cache entries — we only need the summed train_stats / eval_stats now.
    del stats_cache, agg
    import gc; gc.collect()

    n_layers_total = int(train_stats["sigma_k"].shape[0])
    n_kv_heads_total = int(train_stats["sigma_k"].shape[1])
    head_dim = int(train_stats["sigma_k"].shape[-1])

    print(f"[{ts()}] building K bases...", flush=True)
    k_bases = {
        "q_only": eigvec_desc(train_stats["sigma_q"], 1e-4),
        "k_only": eigvec_desc(train_stats["sigma_k"], 1e-4),
        "jointqk": jointqk_basis(train_stats["sigma_q"], train_stats["sigma_k"], 1e-4),
    }
    ref_basis = jointqk_basis(eval_stats["sigma_q"], eval_stats["sigma_k"], 1e-4)
    return {
        "paths": paths,
        "per_example": per_example,
        "test_indices": test_indices,
        "train_stats": train_stats,
        "eval_stats": eval_stats,
        "k_bases": k_bases,
        "ref_basis": ref_basis,
        "head_dim": head_dim,
        "n_layers_total": n_layers_total,
        "n_kv_heads_total": n_kv_heads_total,
    }


def run_shard(args: argparse.Namespace) -> None:
    """Single-shard worker. Computes empirical accumulators on its eval slice and writes JSON."""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"shard_{args.shard_id:03d}.json"
    if out_path.exists():
        print(f"[{ts()}] shard {args.shard_id}: output already exists, skipping ({out_path})", flush=True)
        return

    k_bits = [int(b) for b in args.k_bits.split(",") if b.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{ts()}] shard {args.shard_id}/{args.num_shards} starting on {device} (visible GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')})", flush=True)

    inputs = load_inputs(args, device, k_bits)
    eval_subset_full = inputs["test_indices"]
    if args.max_eval_examples > 0:
        eval_subset_full = eval_subset_full[: args.max_eval_examples]
    eval_slice = slice_for_shard(eval_subset_full, args.num_shards, args.shard_id)
    print(f"[{ts()}] shard {args.shard_id}: eval slice = {len(eval_slice)} idx (of {len(eval_subset_full)})", flush=True)
    print(f"[{ts()}] shard {args.shard_id}: idx = {eval_slice}", flush=True)

    layer0_modes = [True, False]
    raw_lru = RawLRU(max_entries=1)  # one slot — cheap reuse across method calls within an idx is impossible here
                                      # (methods are serial), but the LRU API is uniform so we keep maxsize=1.

    shard_payload: dict[str, Any] = {
        "shard_id": int(args.shard_id),
        "num_shards": int(args.num_shards),
        "eval_slice": eval_slice,
        "k_bits": k_bits,
        "methods": methods,
        "max_coord_bits": int(args.max_coord_bits),
        "head_dim": int(inputs["head_dim"]),
        "n_layers_total": int(inputs["n_layers_total"]),
        "n_kv_heads_total": int(inputs["n_kv_heads_total"]),
        "accumulators": {},  # method -> bits -> layer -> {stat: number}
    }

    for method in methods:
        t0 = time.time()
        print(f"[{ts()}] shard {args.shard_id} === {method} ===", flush=True)
        if method == "v3":
            accums = empirical_v3_metrics(
                inputs["paths"], inputs["per_example"], eval_slice, k_bits, layer0_modes,
                device, seed=args.seed, head_dim=inputs["head_dim"],
                n_layers_total=inputs["n_layers_total"], n_kv_heads_total=inputs["n_kv_heads_total"],
                raw_lru=raw_lru, return_accumulators=True,
            )
        else:
            basis = inputs["k_bases"][method]
            accums = empirical_k_metrics(
                inputs["paths"], inputs["per_example"], eval_slice, basis, inputs["train_stats"],
                k_bits, layer0_modes, device, max_coord_bits=int(args.max_coord_bits),
                raw_lru=raw_lru, return_accumulators=True,
            )
        # Convert to JSON-friendly nested dicts (keys are int).
        shard_payload["accumulators"][method] = {
            str(int(b)): {str(int(l)): dict(stats) for l, stats in per_layer.items()}
            for b, per_layer in accums.items()
        }
        print(f"[{ts()}] shard {args.shard_id} {method} done in {time.time() - t0:.1f}s", flush=True)
        del accums
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(shard_payload) + "\n")
    tmp.replace(out_path)
    print(f"[{ts()}] shard {args.shard_id}: wrote {out_path}", flush=True)


def normalize_accumulators(per_method: dict[str, Any]) -> dict[str, dict[int, dict[int, dict[str, float]]]]:
    out = {}
    for method, by_bits in per_method.items():
        out[method] = {
            int(b): {int(l): dict(stats) for l, stats in per_layer.items()}
            for b, per_layer in by_bits.items()
        }
    return out


def run_launcher(args: argparse.Namespace) -> None:
    """Spawn one shard subprocess per visible GPU, then merge per-shard accumulator JSONs."""
    gpus = discover_gpus(args.gpus)
    num_shards = len(gpus)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{ts()}] launcher: {num_shards} shards on GPUs {gpus}", flush=True)
    print(f"[{ts()}] output dir: {args.out_dir}", flush=True)

    procs = []
    for shard_id, gpu in enumerate(gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [
            str(PY), str(Path(__file__).resolve()),
            "--artifact-root", str(args.artifact_root),
            "--run-id", args.run_id,
            "--k-bits", args.k_bits,
            "--methods", args.methods,
            "--max-eval-examples", str(args.max_eval_examples),
            "--out-dir", str(args.out_dir),
            "--num-shards", str(num_shards),
            "--shard-id", str(shard_id),
            "--seed", str(args.seed),
            "--max-coord-bits", str(args.max_coord_bits),
        ]
        log_path = args.out_dir / f"shard_{shard_id:03d}.log"
        log_fp = open(log_path, "w")
        p = subprocess.Popen(cmd, env=env, stdout=log_fp, stderr=subprocess.STDOUT)
        procs.append((shard_id, gpu, p, log_fp, log_path))
        print(f"[{ts()}] launcher: spawned shard {shard_id} on GPU {gpu} pid={p.pid} log={log_path}", flush=True)

    failures = []
    for shard_id, gpu, p, log_fp, log_path in procs:
        rc = p.wait()
        log_fp.close()
        status = "ok" if rc == 0 else f"FAILED rc={rc}"
        print(f"[{ts()}] launcher: shard {shard_id} (GPU {gpu}) {status}", flush=True)
        if rc != 0:
            failures.append((shard_id, gpu, log_path))

    if failures:
        print(f"[{ts()}] launcher: {len(failures)} shard failure(s); not merging.", flush=True)
        for shard_id, gpu, log_path in failures:
            print(f"  shard {shard_id} on GPU {gpu}: tail of {log_path}:", flush=True)
            try:
                tail = log_path.read_text().splitlines()[-30:]
                for line in tail:
                    print(f"    {line}")
            except Exception as e:
                print(f"    (could not read log: {e})")
        sys.exit(1)

    print(f"[{ts()}] launcher: all shards done; merging...", flush=True)
    k_bits = [int(b) for b in args.k_bits.split(",") if b.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    layer0_modes = [True, False]

    shard_files = sorted(args.out_dir.glob("shard_*.json"))
    if len(shard_files) != num_shards:
        print(f"[{ts()}] launcher: expected {num_shards} shard files, found {len(shard_files)}", flush=True)
        sys.exit(1)
    payloads = [json.loads(f.read_text()) for f in shard_files]
    n_layers_total = payloads[0]["n_layers_total"]
    layer_range_full = range(0, n_layers_total)

    # Merge accumulators per method, then finalize.
    per_method_merged: dict[str, dict[int, dict[int, dict[str, float]]]] = {}
    for method in methods:
        shards_for_method = [normalize_accumulators(p["accumulators"])[method] for p in payloads]
        per_method_merged[method] = merge_accumulators(shards_for_method)

    # Need analytic too — recompute on the launcher itself (cheap, stats only).
    # Use CPU device for the analytic recomputation to stay out of GPUs the shards owned.
    print(f"[{ts()}] launcher: recomputing analytic baselines (CPU)...", flush=True)
    cpu = torch.device("cpu")
    inputs = load_inputs(args, cpu, k_bits)

    final: dict[str, dict[str, dict[str, float]]] = {}
    for method in methods:
        if method == "v3":
            an = v3_rows_for_method(inputs["train_stats"], inputs["eval_stats"], k_bits, layer0_modes)
            emp = _finalize_v3_accums(per_method_merged[method], layer0_modes, k_bits, layer_range_full)
        else:
            basis = inputs["k_bases"][method]
            an = k_rows_for_method(
                method, basis, inputs["train_stats"], inputs["eval_stats"], inputs["ref_basis"],
                k_bits, [16, 32, 64], layer0_modes, args.max_coord_bits,
            )
            emp = _finalize_k_accums(per_method_merged[method], layer0_modes, k_bits, layer_range_full)
        final[method] = {f"layer0={m}": {**an[m], **emp[m]} for m in layer0_modes}

    out_path = args.out_dir / "merged.json"
    out_path.write_text(json.dumps(final, indent=2, sort_keys=True, default=str) + "\n")
    print(f"[{ts()}] launcher: wrote merged metrics to {out_path}", flush=True)

    print(f"\n[{ts()}] === FINAL COMPARISON (pooled, N=50, layer-0-excluded) ===", flush=True)
    header = f"{'method':<12} | {'bits':<4} | {'top1':>8} | {'top5':>8} | {'k_mse':>10} | {'logit_err':>10}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for method in methods:
        d = final[method]["layer0=False"]
        for b in k_bits:
            top1_key = f"empirical_top1_k{b}" if method == "v3" else f"empirical_top1_waterfill_k{b}"
            top5_key = f"empirical_top5_k{b}" if method == "v3" else f"empirical_top5_waterfill_k{b}"
            kmse_key = f"empirical_k_mse_k{b}" if method == "v3" else f"empirical_k_mse_waterfill_k{b}"
            logit_key = f"empirical_logit_error_k{b}" if method == "v3" else f"empirical_logit_error_waterfill_k{b}"
            print(
                f"{method:<12} | {b:<4} | {d[top1_key]:>8.4f} | {d[top5_key]:>8.4f} | "
                f"{d[kmse_key]:>10.3e} | {d[logit_key]:>10.3e}",
                flush=True,
            )


def main() -> None:
    args = parse_args()
    if args.num_shards > 0 and args.shard_id >= 0:
        run_shard(args)
    else:
        run_launcher(args)


if __name__ == "__main__":
    main()
