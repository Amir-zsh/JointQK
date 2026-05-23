#!/usr/bin/env python3
"""Per-task QPCA vs JointQK comparison on compact9 Qwen3.

Computes empirical k_mse / logit_err / top-1 / top-5 for both bases on each
LongBench task's 10 test rows (with the same pooled-train basis built from
450 train prompts across 9 tasks). Output is keyed by task so we can see
where qpca's advantage on logit_err / disadvantage on top-1 lands.

Mirrors `analysis/measure_qpca_compact9.py`'s shard architecture but:
- runs BOTH jointqk and qpca per row (one raw-tensor load amortised across
  both methods; the inner loop is structured to score each (L, h, bits)
  through both compressors before releasing the tensors).
- accumulators are keyed by (method, task, bits, layer) instead of just
  (bits, layer), so the merge produces per-task metrics.

Output dir (default): `artifacts/calibration/longbench_compact9_qkv_qwen3_8b/05_reports/per_task_qpca_vs_jointqk/`
- `shard_NNN.json` — per-shard accumulators
- `per_task_merged.json` — finalised {task: {method: {layer0_mode: {metric: float}}}}
- `shard_NNN.log`
- table comparison printed to launcher stdout
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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

from kvq.compression.per_coord import PerCoordCompressor
from pipelines.calibration.common import RunPaths, StatsCache
from pipelines.calibration.analyze_bases import (
    RawLRU,
    _release_idx,
    _sym,
    allocate_bits,
    combine_stats,
    jointqk_basis,
    regularize_batch,
)

# Reuse qpca_basis from the standalone driver (same module path).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_qpca_compact9 import qpca_basis


REPO = Path(__file__).resolve().parents[1]
PY = REPO / ".venv/bin/python"

METHODS = ("jointqk", "qpca")


def ts() -> str:
    return time.strftime("%H:%M:%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-task qpca vs jointqk on compact9 Qwen3.")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/calibration"))
    parser.add_argument("--run-id", default="longbench_compact9_qkv_qwen3_8b")
    parser.add_argument("--k-bits", default="2,3,4")
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--max-eval-examples", type=int, default=0)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("artifacts/calibration/longbench_compact9_qkv_qwen3_8b/05_reports/per_task_qpca_vs_jointqk"))
    parser.add_argument("--num-shards", type=int, default=0)
    parser.add_argument("--shard-id", type=int, default=-1)
    parser.add_argument("--eps", type=float, default=1e-4)
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


def jointqk_score_diag(basis: torch.Tensor, sigma_q: torch.Tensor, sigma_k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """For an orthogonal basis R, the water-fill score is diag(R^T Σ_Q R) * diag(R^T Σ_K R),
    and std-per-coord is sqrt(diag(R^T Σ_K R)). Matches analyze_bases.empirical_k_metrics.
    Returns (score, k_diag).
    """
    inv = basis.transpose(-1, -2)
    q_diag = (inv @ sigma_q @ basis).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    k_diag = (inv @ sigma_k @ basis).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    return (q_diag * k_diag), k_diag


def load_inputs(args: argparse.Namespace, device: torch.device, k_bits: list[int]) -> dict[str, Any]:
    paths = RunPaths.from_args(args.artifact_root, args.run_id)
    print(f"[{ts()}] loading aggregate (~2.5 GB)...", flush=True)
    agg = torch.load(paths.stats_dir / "aggregate.pt", map_location="cpu", weights_only=False)
    per_example = agg["per_example"]
    train_indices = [int(p["index"]) for p in per_example if p["split"] == "train"]
    test_indices = [int(p["index"]) for p in per_example if p["split"] == "test"]
    print(f"[{ts()}] pooled regime: {len(train_indices)} train + {len(test_indices)} test", flush=True)

    stats_cache = StatsCache(max_entries=128)
    print(f"[{ts()}] summing train stats over {len(train_indices)} examples...", flush=True)
    train_stats = combine_stats(paths, per_example, train_indices, device, cache=stats_cache)
    cs = stats_cache.stats()
    print(f"[{ts()}] stats cache: {cs}", flush=True)
    del stats_cache, agg
    import gc; gc.collect()

    n_layers_total = int(train_stats["sigma_k"].shape[0])
    n_kv_heads_total = int(train_stats["sigma_k"].shape[1])
    head_dim = int(train_stats["sigma_k"].shape[-1])

    print(f"[{ts()}] building jointqk basis (orthogonal eigvec)...", flush=True)
    jq = jointqk_basis(train_stats["sigma_q"], train_stats["sigma_k"], args.eps)
    jq_score, jq_k_diag = jointqk_score_diag(jq, train_stats["sigma_q"], train_stats["sigma_k"])

    print(f"[{ts()}] building qpca basis (fp64)...", flush=True)
    qp_fwd, qp_inv, qp_lam = qpca_basis(train_stats["sigma_q"], train_stats["sigma_k"], args.eps)

    return {
        "paths": paths,
        "per_example": per_example,
        "test_indices": test_indices,
        "jointqk": {
            "forward": jq,
            "inverse": jq.transpose(-1, -2).contiguous(),
            "score": jq_score,
            "std": jq_k_diag.sqrt(),
        },
        "qpca": {
            "forward": qp_fwd,
            "inverse": qp_inv,
            "score": qp_lam,
            "std": qp_lam.sqrt(),
        },
        "head_dim": head_dim,
        "n_layers_total": n_layers_total,
        "n_kv_heads_total": n_kv_heads_total,
    }


def run_shard(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"shard_{args.shard_id:03d}.json"
    if out_path.exists():
        print(f"[{ts()}] shard {args.shard_id}: output exists, skipping ({out_path})", flush=True)
        return

    k_bits = [int(b) for b in args.k_bits.split(",") if b.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{ts()}] shard {args.shard_id}/{args.num_shards} on {device} "
          f"(visible: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')})", flush=True)

    inputs = load_inputs(args, device, k_bits)
    head_dim = inputs["head_dim"]
    n_layers_total = inputs["n_layers_total"]
    n_kv_heads_total = inputs["n_kv_heads_total"]

    eval_subset_full = inputs["test_indices"]
    if args.max_eval_examples > 0:
        eval_subset_full = eval_subset_full[: args.max_eval_examples]
    eval_slice = slice_for_shard(eval_subset_full, args.num_shards, args.shard_id)
    print(f"[{ts()}] shard {args.shard_id}: eval slice = {len(eval_slice)} idx", flush=True)

    # Build per-(method, bits, L, H) compressors once. Lloyd-Max codebook solve is
    # bits-dependent, so we precompute over all (L, H) × bits × method.
    layer_range_full = range(0, n_layers_total)
    head_range = range(0, n_kv_heads_total)
    layer_head_pairs = [(l, h) for l in layer_range_full for h in head_range]

    print(f"[{ts()}] building compressors: {len(METHODS)} methods × {len(layer_head_pairs)} (L,H) × {len(k_bits)} bits...", flush=True)
    comps: dict[tuple[str, int], dict[tuple[int, int], PerCoordCompressor]] = {}
    for method in METHODS:
        spec = inputs[method]
        forward_cpu = spec["forward"].cpu()
        inverse_cpu = spec["inverse"].cpu()
        std_cpu = spec["std"].cpu()
        for bits in k_bits:
            alloc = allocate_bits(spec["score"], bits, max_coord_bits=int(args.max_coord_bits)).cpu()
            comps[(method, bits)] = {
                (l, h): PerCoordCompressor(
                    bits_per_coord=alloc[l, h],
                    std_per_coord=std_cpu[l, h],
                    forward_map=forward_cpu[l, h],
                    inverse_map=inverse_cpu[l, h],
                ).to(device)
                for (l, h) in layer_head_pairs
            }
    print(f"[{ts()}] compressors ready", flush=True)

    # Per-(method, task, bits, layer) accumulators.
    def fresh_acc() -> dict[str, float]:
        return {"mse_num": 0.0, "mse_den": 0, "logit_num": 0.0, "logit_den": 0,
                "top1_num": 0, "top1_den": 0, "top5_num": 0, "top5_den": 0}

    accums: dict[tuple[str, str, int, int], dict[str, float]] = {}

    raw_lru = RawLRU(max_entries=1)
    t_start = time.time()

    for n_idx, idx in enumerate(eval_slice):
        t_idx = time.time()
        meta = inputs["per_example"][idx]
        task = str(meta["task"])
        loader = lambda p=inputs["paths"].root / meta["raw_file"]: torch.load(p, map_location="cpu", weights_only=False)
        raw = raw_lru.get(idx, loader)
        k_all = raw["k_post"]
        q_all = raw["q_post"]
        tokens = int(raw["prompt_length"])
        d = int(k_all.shape[-1])
        group = q_all.shape[1] // n_kv_heads_total

        for layer, head in layer_head_pairs:
            k = k_all[layer, head, :tokens, :].to(device).float()
            q = q_all[layer, head * group : (head + 1) * group, :tokens, :].to(device).float().reshape(-1, d)
            qq = q.transpose(0, 1) @ q
            k_t = k.transpose(0, 1)

            for method in METHODS:
                for bits in k_bits:
                    key = (method, task, bits, layer)
                    if key not in accums:
                        accums[key] = fresh_acc()
                    acc = accums[key]
                    comp = comps[(method, bits)][(layer, head)]
                    recon = comp.roundtrip(k).float()
                    err = k - recon
                    acc["mse_num"] += float(err.square().sum().item())
                    acc["mse_den"] += int(err.numel())
                    ee = err.transpose(0, 1) @ err
                    acc["logit_num"] += float((qq * ee).sum().item())
                    acc["logit_den"] += int(group * tokens * tokens)
                    LOGIT_CHUNK_BYTES = 1 << 29
                    chunk_q = max(1, LOGIT_CHUNK_BYTES // (max(tokens, 1) * 4))
                    recon_t = recon.transpose(0, 1)
                    k_top = min(5, tokens)
                    for qstart in range(0, q.shape[0], chunk_q):
                        qchunk = q[qstart : qstart + chunk_q]
                        real_logits = qchunk @ k_t
                        approx_logits = qchunk @ recon_t
                        real_argmax = real_logits.argmax(dim=-1)
                        approx_argmax = approx_logits.argmax(dim=-1)
                        matches_top1 = (real_argmax == approx_argmax)
                        acc["top1_num"] += int(matches_top1.sum().item())
                        acc["top1_den"] += int(matches_top1.numel())
                        approx_top5_idx = approx_logits.topk(k_top, dim=-1).indices
                        matches_top5 = (approx_top5_idx == real_argmax.unsqueeze(-1)).any(dim=-1)
                        acc["top5_num"] += int(matches_top5.sum().item())
                        acc["top5_den"] += int(matches_top5.numel())
                        del real_logits, approx_logits, real_argmax, approx_argmax, approx_top5_idx
                    del recon, err, ee, recon_t
            del k, q, qq, k_t
        del raw, k_all, q_all
        _release_idx(raw_lru, idx)
        print(f"[{ts()}] shard {args.shard_id} idx {n_idx+1}/{len(eval_slice)} (#{idx}, task={task}) "
              f"done in {time.time()-t_idx:.1f}s", flush=True)

    # Serialize accumulators with composite string keys.
    serial: dict[str, dict[str, float]] = {}
    for (method, task, bits, layer), acc in accums.items():
        serial[f"{method}|{task}|{bits}|{layer}"] = dict(acc)

    payload: dict[str, Any] = {
        "shard_id": int(args.shard_id),
        "num_shards": int(args.num_shards),
        "eval_slice": eval_slice,
        "k_bits": k_bits,
        "methods": list(METHODS),
        "head_dim": int(head_dim),
        "n_layers_total": int(n_layers_total),
        "n_kv_heads_total": int(n_kv_heads_total),
        "accumulators": serial,
    }
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload) + "\n")
    tmp.replace(out_path)
    print(f"[{ts()}] shard {args.shard_id}: wrote {out_path} (elapsed {time.time()-t_start:.0f}s)", flush=True)


def merge_accumulators(shard_payloads: list[dict[str, Any]]) -> dict[tuple[str, str, int, int], dict[str, float]]:
    out: dict[tuple[str, str, int, int], dict[str, float]] = {}
    for p in shard_payloads:
        for compound_key, stats in p["accumulators"].items():
            method, task, bits_s, layer_s = compound_key.split("|")
            key = (method, task, int(bits_s), int(layer_s))
            if key not in out:
                out[key] = {k: 0 for k in stats}
            for k, v in stats.items():
                out[key][k] = out[key].get(k, 0) + v
    return out


def finalize_per_task(
    merged: dict[tuple[str, str, int, int], dict[str, float]],
    k_bits: list[int],
    n_layers_total: int,
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """{task: {method: {"layer0=False": {empirical_top1_waterfill_k2: ..., ...}, "layer0=True": {...}}}}"""
    layer_range_full = list(range(0, n_layers_total))
    modes = [True, False]
    tasks = sorted(set(t for (_, t, _, _) in merged.keys()))
    methods = sorted(set(m for (m, _, _, _) in merged.keys()))

    final: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for task in tasks:
        final[task] = {}
        for method in methods:
            per_mode: dict[str, dict[str, float]] = {}
            for m in modes:
                active_layers = [l for l in layer_range_full if (m or l > 0)]
                metrics: dict[str, float] = {}
                for bits in k_bits:
                    mse_n = sum(merged.get((method, task, bits, l), {}).get("mse_num", 0) for l in active_layers)
                    mse_d = sum(merged.get((method, task, bits, l), {}).get("mse_den", 0) for l in active_layers)
                    log_n = sum(merged.get((method, task, bits, l), {}).get("logit_num", 0) for l in active_layers)
                    log_d = sum(merged.get((method, task, bits, l), {}).get("logit_den", 0) for l in active_layers)
                    t1_n = sum(merged.get((method, task, bits, l), {}).get("top1_num", 0) for l in active_layers)
                    t1_d = sum(merged.get((method, task, bits, l), {}).get("top1_den", 0) for l in active_layers)
                    t5_n = sum(merged.get((method, task, bits, l), {}).get("top5_num", 0) for l in active_layers)
                    t5_d = sum(merged.get((method, task, bits, l), {}).get("top5_den", 0) for l in active_layers)
                    metrics[f"empirical_k_mse_waterfill_k{bits}"] = mse_n / max(1, mse_d)
                    metrics[f"empirical_logit_error_waterfill_k{bits}"] = log_n / max(1, log_d)
                    metrics[f"empirical_top1_waterfill_k{bits}"] = t1_n / max(1, t1_d)
                    metrics[f"empirical_top5_waterfill_k{bits}"] = t5_n / max(1, t5_d)
                per_mode[f"layer0={m}"] = metrics
            final[task][method] = per_mode
    return final


def print_comparison(final: dict[str, dict[str, dict[str, dict[str, float]]]], k_bits: list[int]) -> None:
    """Print per-task side-by-side comparison: jointqk vs qpca."""
    print(f"\n[{ts()}] === PER-TASK COMPARISON (layer-0 excluded, pooled-train basis on each task's 10 test rows) ===", flush=True)
    hdr = f"{'task':<22} | b | {'jq top1':>7} {'qp top1':>7} {'d':>7} | {'jq top5':>7} {'qp top5':>7} {'d':>7} | {'jq logit':>9} {'qp logit':>9}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for task in sorted(final.keys()):
        jq = final[task]["jointqk"]["layer0=False"]
        qp = final[task]["qpca"]["layer0=False"]
        for b in k_bits:
            t1_jq = jq[f"empirical_top1_waterfill_k{b}"]
            t1_qp = qp[f"empirical_top1_waterfill_k{b}"]
            t5_jq = jq[f"empirical_top5_waterfill_k{b}"]
            t5_qp = qp[f"empirical_top5_waterfill_k{b}"]
            le_jq = jq[f"empirical_logit_error_waterfill_k{b}"]
            le_qp = qp[f"empirical_logit_error_waterfill_k{b}"]
            d1 = t1_qp - t1_jq
            d5 = t5_qp - t5_jq
            sd1 = ("+%.4f" % d1) if d1 >= 0 else ("%.4f" % d1)
            sd5 = ("+%.4f" % d5) if d5 >= 0 else ("%.4f" % d5)
            print(f"{task:<22} | {b} | {t1_jq:>7.4f} {t1_qp:>7.4f} {sd1:>7} | "
                  f"{t5_jq:>7.4f} {t5_qp:>7.4f} {sd5:>7} | {le_jq:>9.3e} {le_qp:>9.3e}", flush=True)


def run_launcher(args: argparse.Namespace) -> None:
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
            "--max-eval-examples", str(args.max_eval_examples),
            "--out-dir", str(args.out_dir),
            "--num-shards", str(num_shards),
            "--shard-id", str(shard_id),
            "--eps", str(args.eps),
            "--max-coord-bits", str(args.max_coord_bits),
        ]
        log_path = args.out_dir / f"shard_{shard_id:03d}.log"
        log_fp = open(log_path, "w")
        p = subprocess.Popen(cmd, env=env, stdout=log_fp, stderr=subprocess.STDOUT)
        procs.append((shard_id, gpu, p, log_fp, log_path))
        print(f"[{ts()}] launcher: spawned shard {shard_id} on GPU {gpu} pid={p.pid}", flush=True)

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
            print(f"  tail of shard {shard_id} log:", flush=True)
            try:
                tail = log_path.read_text().splitlines()[-30:]
                for line in tail:
                    print(f"    {line}")
            except Exception as e:
                print(f"    (cannot read log: {e})")
        sys.exit(1)

    print(f"[{ts()}] launcher: all shards done; merging...", flush=True)
    k_bits = [int(b) for b in args.k_bits.split(",") if b.strip()]
    shard_files = sorted(args.out_dir.glob("shard_*.json"))
    if len(shard_files) != num_shards:
        print(f"[{ts()}] launcher: expected {num_shards} shards, found {len(shard_files)}", flush=True)
        sys.exit(1)
    payloads = [json.loads(f.read_text()) for f in shard_files]
    n_layers_total = payloads[0]["n_layers_total"]

    merged = merge_accumulators(payloads)
    final = finalize_per_task(merged, k_bits, n_layers_total)

    out_path = args.out_dir / "per_task_merged.json"
    out_path.write_text(json.dumps(final, indent=2, sort_keys=True, default=str) + "\n")
    print(f"[{ts()}] launcher: wrote {out_path}", flush=True)

    print_comparison(final, k_bits)


def main() -> None:
    args = parse_args()
    if args.num_shards > 0 and args.shard_id >= 0:
        run_shard(args)
    else:
        run_launcher(args)


if __name__ == "__main__":
    main()
