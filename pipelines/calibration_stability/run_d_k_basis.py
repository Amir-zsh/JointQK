#!/usr/bin/env python3
"""Phase 1D: K-basis calibration data requirement and cross-task stability.

This is an offline calibration experiment. It does not run model inference.

Question:
    How many calibration examples are needed before the JointQK K basis
    (`R_sym = eigvec((Sigma_Q Sigma_K + Sigma_K Sigma_Q) / 2)`) stabilizes,
    and how well does a basis calibrated on one task transfer to another?

Method:
    1. Load the existing 24-example LongBench-E Q/K bundle.
    2. Build per-example second-moment sums using the same Sigma_Q convention
       as Stage 1E / JointQK: treat each Q head in a GQA group as a sample.
    3. Fit R_sym from subsets of size n in each task, and from stratified pooled
       subsets with n examples per task.
    4. Compare each subset-fitted basis against full-task references and the
       full pooled reference using:
       - top-r subspace overlap at ranks 16/32/64/96,
       - water-fill integer bit-allocation L1 drift,
       - closed-form Q-weighted K-distortion regret under each eval task's
         full moments.

Outputs:
    <output-dir>/phase1d_k_basis_stability_rows.jsonl
    <output-dir>/phase1d_k_basis_stability_summary.json
    <output-dir>/phase1d_k_basis_stability_summary.md
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _bootstrap  # noqa: E402, F401

import argparse
import gc
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kvq.compression.metric_transform import water_fill
from kvq.compression.per_coord import round_bits_to_integer


DEFAULT_CONFIGS = ("qasper_e", "hotpotqa_e", "passage_retrieval_en_e")


def fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{sec:02d}s"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_csv_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def read_manifest(bundle_dir: Path) -> list[dict[str, Any]]:
    with open(bundle_dir / "manifest.json") as f:
        return json.load(f)["examples"]


def example_path(bundle_dir: Path, ex: dict[str, Any]) -> Path:
    return bundle_dir / ex["file"]


def infer_shape(bundle_dir: Path, examples: list[dict[str, Any]], limit_layers: int | None) -> tuple[int, int, int, int]:
    payload = torch.load(example_path(bundle_dir, examples[0]), map_location="cpu", weights_only=False)
    n_layers = int(payload["k_post"].shape[0])
    if limit_layers is not None:
        n_layers = min(n_layers, limit_layers)
    n_kv_heads = int(payload["k_post"].shape[1])
    n_q_heads = int(payload["q_post"].shape[1])
    head_dim = int(payload["k_post"].shape[-1])
    if n_q_heads % n_kv_heads != 0:
        raise ValueError(f"GQA mismatch: n_q_heads={n_q_heads}, n_kv_heads={n_kv_heads}")
    return n_layers, n_kv_heads, n_q_heads // n_kv_heads, head_dim


@torch.no_grad()
def example_moment_sums(
    bundle_dir: Path,
    ex: dict[str, Any],
    n_layers: int,
    n_kv_heads: int,
    group: int,
    head_dim: int,
    device: str,
) -> dict[str, torch.Tensor | int | str]:
    """Return unnormalized moment sums for one example."""
    payload = torch.load(example_path(bundle_dir, ex), map_location="cpu", weights_only=False)
    L = int(payload["prompt_length"])
    n_q_heads = int(payload["q_post"].shape[1])
    if n_q_heads != n_kv_heads * group:
        raise ValueError(f"GQA mismatch in {ex['file']}: q_heads={n_q_heads}")

    # Long prompts can make full-example Q tensors several GiB in fp32. Compute
    # layer-by-layer to keep peak GPU memory bounded.
    sq_sum = torch.empty((n_layers, n_kv_heads, head_dim, head_dim), dtype=torch.float32)
    sk_sum = torch.empty_like(sq_sum)
    cqk_sum = torch.empty_like(sq_sum)
    for layer in range(n_layers):
        q = payload["q_post"][layer : layer + 1, :, :L, :].to(device).float()
        k = payload["k_post"][layer : layer + 1, :, :L, :].to(device).float()
        sq_layer = torch.einsum("lhsd,lhse->lhde", q, q)
        sq_layer = sq_layer.view(1, n_kv_heads, group, head_dim, head_dim).sum(dim=2)
        sk_layer = torch.einsum("lhsd,lhse->lhde", k, k)
        q_grouped = q.view(1, n_kv_heads, group, L, head_dim).mean(dim=2)
        cqk_layer = torch.einsum("lhsd,lhse->lhde", q_grouped, k)
        sq_sum[layer] = sq_layer.squeeze(0).cpu()
        sk_sum[layer] = sk_layer.squeeze(0).cpu()
        cqk_sum[layer] = cqk_layer.squeeze(0).cpu()
        del q, k, sq_layer, sk_layer, q_grouped, cqk_layer
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    del payload
    gc.collect()
    return {
        "config": ex["config"],
        "file": ex["file"],
        "tokens": L,
        "sq_sum": sq_sum,
        "sk_sum": sk_sum,
        "cqk_sum": cqk_sum,
    }


def combine_moments(
    per_example: list[dict[str, Any]],
    indices: list[int],
    group: int,
    out_device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor | int]:
    if not indices:
        raise ValueError("Cannot combine an empty calibration subset")
    first = per_example[indices[0]]
    sq = torch.zeros_like(first["sq_sum"])
    sk = torch.zeros_like(first["sk_sum"])
    cqk = torch.zeros_like(first["cqk_sum"])
    total_tokens = 0
    for idx in indices:
        item = per_example[idx]
        sq += item["sq_sum"]
        sk += item["sk_sum"]
        cqk += item["cqk_sum"]
        total_tokens += int(item["tokens"])
    return {
        "sigma_q": (sq / (group * total_tokens)).to(out_device),
        "sigma_k": (sk / total_tokens).to(out_device),
        "cqk": (cqk / total_tokens).to(out_device),
        "total_tokens": total_tokens,
    }


def derive_rsym(sigma_q: torch.Tensor, sigma_k: torch.Tensor, eps: float) -> torch.Tensor:
    """Derive R_sym with the same convention as Stage 1E newbases."""
    n_layers, n_kv_heads, d, _ = sigma_q.shape
    flat_sq = sigma_q.reshape(-1, d, d).float()
    flat_sk = sigma_k.reshape(-1, d, d).float()
    sym_sq = 0.5 * (flat_sq + flat_sq.transpose(-1, -2))
    sym_sk = 0.5 * (flat_sk + flat_sk.transpose(-1, -2))
    trace_sq = sym_sq.diagonal(dim1=-2, dim2=-1).sum(-1)
    trace_sk = sym_sk.diagonal(dim1=-2, dim2=-1).sum(-1)
    eye = torch.eye(d, dtype=sym_sq.dtype, device=sym_sq.device)
    sq_reg = sym_sq + (eps * (trace_sq / d).clamp_min(1e-12)).view(-1, 1, 1) * eye
    sk_reg = sym_sk + (eps * (trace_sk / d).clamp_min(1e-12)).view(-1, 1, 1) * eye
    qk = sq_reg @ sk_reg
    sym = 0.5 * (qk + qk.transpose(-1, -2))
    eigvals, eigvecs = torch.linalg.eigh(sym)
    sort_idx = torch.argsort(eigvals, dim=-1, descending=True)
    rsym = torch.gather(eigvecs, -1, sort_idx.unsqueeze(-2).expand(-1, d, -1))
    return rsym.reshape(n_layers, n_kv_heads, d, d)


def subspace_overlap(candidate: torch.Tensor, reference: torch.Tensor, rank: int) -> torch.Tensor:
    """Per-head top-r subspace overlap, sign-invariant, in [0, 1]."""
    cand = candidate[..., :, :rank]
    ref = reference[..., :, :rank]
    cross = cand.transpose(-1, -2) @ ref
    return cross.square().sum(dim=(-2, -1)) / float(rank)


def rsym_scores(sigma_q: torch.Tensor, sigma_k: torch.Tensor, rsym: torch.Tensor) -> torch.Tensor:
    """Return per-coordinate Q-weighted variance scores under R_sym."""
    mq_diag = (rsym.transpose(-1, -2) @ sigma_q.float() @ rsym).diagonal(dim1=-2, dim2=-1)
    sk_diag = (rsym.transpose(-1, -2) @ sigma_k.float() @ rsym).diagonal(dim1=-2, dim2=-1)
    return (mq_diag.clamp_min(1e-30) * sk_diag.clamp_min(1e-30)).float()


def cap_bits_like_jointqk(bits_int: torch.Tensor, max_bits: int = 8) -> torch.Tensor:
    """Apply the same max-bit redistribution policy used by JointQK compressors."""
    flat = bits_int.reshape(-1, bits_int.shape[-1]).clone()
    for i in range(flat.shape[0]):
        excess = int((flat[i] - max_bits).clamp_min(0).sum().item())
        flat[i].clamp_(max=max_bits)
        while excess > 0:
            below = (flat[i] < max_bits).nonzero(as_tuple=True)[0]
            if below.numel() == 0:
                break
            min_val = flat[i, below].min()
            candidates = below[flat[i, below] == min_val]
            take = min(int(candidates.numel()), excess)
            flat[i, candidates[:take]] += 1
            excess -= take
    return flat.reshape_as(bits_int)


def allocate_bits(scores: torch.Tensor, k_bits: int) -> torch.Tensor:
    d = scores.shape[-1]
    flat_scores = scores.reshape(-1, d)
    bits_cont = water_fill(flat_scores, total_bits=float(k_bits * d))
    bits_int = round_bits_to_integer(bits_cont, total_bits=k_bits * d)
    return cap_bits_like_jointqk(bits_int).reshape(scores.shape)


def distortion_from_scores(scores: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
    return (scores * torch.pow(torch.tensor(2.0), -2.0 * bits.float())).sum(dim=-1)


def summarize(values: list[float]) -> dict[str, float]:
    t = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(t.mean()),
        "std": float(t.std(unbiased=False)) if t.numel() > 1 else 0.0,
        "min": float(t.min()),
        "p10": float(torch.quantile(t, 0.10)),
        "p50": float(torch.quantile(t, 0.50)),
        "p90": float(torch.quantile(t, 0.90)),
        "max": float(t.max()),
        "n": int(t.numel()),
    }


def count_trial_jobs(
    configs: tuple[str, ...],
    by_config: dict[str, list[int]],
    sample_sizes: list[int],
    repetitions: int,
    include_loo: bool,
) -> int:
    total = 0
    for cfg in configs:
        pool_size = len(by_config[cfg])
        for n in sample_sizes:
            if n > pool_size:
                continue
            total += 1 if n == pool_size else repetitions
    for n_per_task in sample_sizes:
        if any(n_per_task > len(by_config[cfg]) for cfg in configs):
            continue
        total += 1 if all(n_per_task == len(by_config[cfg]) for cfg in configs) else repetitions
    if include_loo and len(configs) > 1:
        for holdout in configs:
            calib_cfgs = [cfg for cfg in configs if cfg != holdout]
            for n_per_task in sample_sizes:
                if any(n_per_task > len(by_config[cfg]) for cfg in calib_cfgs):
                    continue
                total += 1 if all(n_per_task == len(by_config[cfg]) for cfg in calib_cfgs) else repetitions
    return total


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", default="artifacts/query_stats_longbench_under4k")
    p.add_argument("--output-dir", default="artifacts/v_bases/k_basis_stability")
    p.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    p.add_argument("--sample-sizes", default="1,2,4,8",
                   help="Per-task sample sizes; pooled runs use this many examples per task.")
    p.add_argument("--repetitions", type=int, default=20)
    p.add_argument("--ranks", default="16,32,64,96")
    p.add_argument("--k-bits", default="2,3,4")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit-layers", type=int, default=None,
                   help="Debug/smoke option; production run should leave unset.")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Split subset-trial jobs across this many independent workers.")
    p.add_argument("--shard-id", type=int, default=0,
                   help="Worker shard id in [0, num_shards).")
    p.add_argument("--rows-filename", default="phase1d_k_basis_stability_rows.jsonl")
    p.add_argument("--include-loo", action="store_true",
                   help="Also fit pooled bases on all-but-one task and evaluate the held-out task.")
    p.add_argument("--moments-cache", default=None,
                   help="Optional .pt cache of per-example moment sums. Created if missing.")
    p.add_argument("--prepare-moments-only", action="store_true",
                   help="Only build --moments-cache, then exit.")
    p.add_argument("--holdout-per-config", type=int, default=0,
                   help="Hold out this many examples per task for eval references; calibrate only on the rest.")
    p.add_argument("--holdout-seed", type=int, default=991,
                   help="Seed for selecting held-out eval examples when --holdout-per-config > 0.")
    args = p.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError(f"--shard-id must be in [0, {args.num_shards}), got {args.shard_id}")

    bundle_dir = Path(args.bundle)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = tuple(c.strip() for c in args.configs.split(",") if c.strip())
    sample_sizes = parse_csv_ints(args.sample_sizes)
    ranks = parse_csv_ints(args.ranks)
    k_bits_grid = parse_csv_ints(args.k_bits)
    rng = random.Random(args.seed)
    compute_device = torch.device(args.device)

    examples = read_manifest(bundle_dir)
    by_config: dict[str, list[int]] = {cfg: [] for cfg in configs}
    for i, ex in enumerate(examples):
        if ex["config"] in by_config:
            by_config[ex["config"]].append(i)
    missing = [cfg for cfg, idxs in by_config.items() if not idxs]
    if missing:
        raise SystemExit(f"No examples found for configs: {missing}")

    n_layers, n_kv_heads, group, head_dim = infer_shape(bundle_dir, examples, args.limit_layers)
    log(
        f"Phase 1D K-basis stability: layers={n_layers}, kv_heads={n_kv_heads}, "
        f"group={group}, d={head_dim}, configs={configs}, device={args.device}",
    )

    cache_path = Path(args.moments_cache) if args.moments_cache else None
    if cache_path is not None and cache_path.exists():
        t0 = time.time()
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        per_example = cache["per_example"]
        log(f"Loaded {len(per_example)} per-example moment sums from {cache_path} in {fmt_duration(time.time() - t0)}")
    else:
        per_example = []
        parts_dir = Path(str(cache_path) + ".parts") if cache_path is not None else None
        if parts_dir is not None:
            parts_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        compact_idx = 0
        for i, ex in enumerate(examples):
            if ex["config"] not in by_config:
                continue
            part_path = parts_dir / f"moment_{compact_idx:03d}.pt" if parts_dir is not None else None
            if part_path is not None and part_path.exists():
                log(f"moments {i+1:02d}/{len(examples)} {ex['file']} cfg={ex['config']} (cached)")
                item = torch.load(part_path, map_location="cpu", weights_only=False)
            else:
                log(f"moments {i+1:02d}/{len(examples)} {ex['file']} cfg={ex['config']}")
                item = example_moment_sums(bundle_dir, ex, n_layers, n_kv_heads, group, head_dim, args.device)
                if part_path is not None:
                    torch.save(item, part_path)
                    log(f"wrote moment part: {part_path}")
            per_example.append(item)
            compact_idx += 1
            gc.collect()
        log(f"Loaded {len(per_example)} per-example moment sums in {fmt_duration(time.time() - t0)}")
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "per_example": per_example,
                    "n_layers": n_layers,
                    "n_kv_heads": n_kv_heads,
                    "group": group,
                    "head_dim": head_dim,
                    "bundle": str(bundle_dir),
                },
                cache_path,
            )
            log(f"Wrote moment cache: {cache_path}")
    if args.prepare_moments_only:
        return

    # Remap manifest indices to compact per_example indices.
    compact_by_config: dict[str, list[int]] = defaultdict(list)
    for i, item in enumerate(per_example):
        compact_by_config[str(item["config"])].append(i)

    calib_by_config: dict[str, list[int]] = {}
    eval_by_config: dict[str, list[int]] = {}
    if args.holdout_per_config > 0:
        split_rng = random.Random(args.holdout_seed)
        for cfg in configs:
            pool = compact_by_config[cfg][:]
            if args.holdout_per_config >= len(pool):
                raise ValueError(
                    f"--holdout-per-config={args.holdout_per_config} leaves no calibration "
                    f"examples for {cfg} (available={len(pool)})"
                )
            split_rng.shuffle(pool)
            eval_by_config[cfg] = sorted(pool[:args.holdout_per_config])
            calib_by_config[cfg] = sorted(pool[args.holdout_per_config:])
            log(
                f"Split {cfg}: calibration={len(calib_by_config[cfg])} "
                f"heldout_eval={len(eval_by_config[cfg])}"
            )
    else:
        calib_by_config = {cfg: compact_by_config[cfg][:] for cfg in configs}
        eval_by_config = {cfg: compact_by_config[cfg][:] for cfg in configs}

    total_trials = count_trial_jobs(configs, calib_by_config, sample_sizes, args.repetitions, args.include_loo)
    shard_trials = sum(1 for trial_idx in range(total_trials) if trial_idx % args.num_shards == args.shard_id)
    log(
        f"Shard {args.shard_id}/{args.num_shards}: assigned {shard_trials}/{total_trials} "
        f"calibration trials"
    )

    # Full references: each task and pooled.
    references: dict[str, dict[str, Any]] = {}
    for cfg in configs:
        t_ref = time.time()
        indices = eval_by_config[cfg]
        stats = combine_moments(per_example, indices, group, out_device=compute_device)
        rsym = derive_rsym(stats["sigma_q"], stats["sigma_k"], args.eps)
        references[cfg] = {"indices": indices, "stats": stats, "rsym": rsym}
        log(
            f"Reference {cfg}: examples={len(indices)} tokens={stats['total_tokens']} "
            f"in {fmt_duration(time.time() - t_ref)}"
        )
    pooled_indices = [i for cfg in configs for i in eval_by_config[cfg]]
    t_ref = time.time()
    pooled_stats = combine_moments(per_example, pooled_indices, group, out_device=compute_device)
    references["pooled"] = {
        "indices": pooled_indices,
        "stats": pooled_stats,
        "rsym": derive_rsym(pooled_stats["sigma_q"], pooled_stats["sigma_k"], args.eps),
    }
    log(
        f"Reference pooled: examples={len(pooled_indices)} tokens={pooled_stats['total_tokens']} "
        f"in {fmt_duration(time.time() - t_ref)}"
    )

    for ref in references.values():
        t_ref = time.time()
        scores = rsym_scores(ref["stats"]["sigma_q"], ref["stats"]["sigma_k"], ref["rsym"])
        ref["scores"] = scores
        ref["bits"] = {kb: allocate_bits(scores, kb) for kb in k_bits_grid}
        ref["dist"] = {kb: distortion_from_scores(scores, ref["bits"][kb]) for kb in k_bits_grid}
        log(f"Reference scores/bits ready in {fmt_duration(time.time() - t_ref)}")

    rows: list[dict[str, Any]] = []

    trial_counter = 0
    completed_trials = 0
    shard_start = time.time()

    def maybe_evaluate_subset(source: str, indices: list[int], n_label: int, rep: int) -> None:
        nonlocal completed_trials, trial_counter
        trial_idx = trial_counter
        trial_counter += 1
        if trial_idx % args.num_shards != args.shard_id:
            return
        t_trial = time.time()
        stats = combine_moments(per_example, indices, group, out_device=compute_device)
        cand_rsym = derive_rsym(stats["sigma_q"], stats["sigma_k"], args.eps)
        cand_scores_for_alloc = rsym_scores(stats["sigma_q"], stats["sigma_k"], cand_rsym)
        cand_bits = {kb: allocate_bits(cand_scores_for_alloc, kb) for kb in k_bits_grid}

        for eval_name, ref in references.items():
            eval_stats = ref["stats"]
            ref_rsym = ref["rsym"]
            l0 = torch.arange(n_layers) > 0
            row: dict[str, Any] = {
                "source": source,
                "eval": eval_name,
                "n_examples": len(indices),
                "n_label": n_label,
                "rep": rep,
                "trial_idx": trial_idx,
                "shard_id": args.shard_id,
                "num_shards": args.num_shards,
                "total_tokens": int(stats["total_tokens"]),
            }
            for rank in ranks:
                ov = subspace_overlap(cand_rsym, ref_rsym, rank)[l0].mean().item()
                row[f"subspace_overlap_r{rank}"] = float(ov)

            for kb in k_bits_grid:
                eval_scores_under_candidate = rsym_scores(eval_stats["sigma_q"], eval_stats["sigma_k"], cand_rsym)
                cand_dist = distortion_from_scores(eval_scores_under_candidate, cand_bits[kb])
                ref_dist = ref["dist"][kb]
                log2_ratio = torch.log2((cand_dist / ref_dist).clamp_min(1e-30))[l0].mean().item()
                bit_l1 = (cand_bits[kb].float() - ref["bits"][kb].float()).abs()[l0].mean().item()
                row[f"log2_regret_k{kb}"] = float(log2_ratio)
                row[f"bit_l1_k{kb}"] = float(bit_l1)
            rows.append(row)
        completed_trials += 1
        if completed_trials <= 5 or completed_trials % 10 == 0 or completed_trials == shard_trials:
            elapsed = time.time() - shard_start
            avg = elapsed / max(1, completed_trials)
            eta = avg * max(0, shard_trials - completed_trials)
            log(
                f"Shard {args.shard_id}/{args.num_shards}: completed trial {completed_trials}/{shard_trials} "
                f"(global={trial_idx}, source={source}, n={n_label}, rep={rep}, "
                f"tokens={stats['total_tokens']}) in {fmt_duration(time.time() - t_trial)}; "
                f"elapsed={fmt_duration(elapsed)} eta={fmt_duration(eta)}"
            )

    # Task-specific subsets.
    for cfg in configs:
        pool = calib_by_config[cfg]
        for n in sample_sizes:
            if n > len(pool):
                continue
            reps = 1 if n == len(pool) else args.repetitions
            for rep in range(reps):
                indices = pool[:] if n == len(pool) else rng.sample(pool, n)
                maybe_evaluate_subset(source=cfg, indices=indices, n_label=n, rep=rep)

    # Stratified pooled subsets: n examples per task.
    for n_per_task in sample_sizes:
        if any(n_per_task > len(calib_by_config[cfg]) for cfg in configs):
            continue
        reps = 1 if all(n_per_task == len(calib_by_config[cfg]) for cfg in configs) else args.repetitions
        for rep in range(reps):
            indices: list[int] = []
            for cfg in configs:
                pool = calib_by_config[cfg]
                indices.extend(pool[:] if n_per_task == len(pool) else rng.sample(pool, n_per_task))
            maybe_evaluate_subset(source="pooled_stratified", indices=indices, n_label=n_per_task, rep=rep)

    # Leave-one-task-out pooled subsets: n examples per non-held-out task.
    if args.include_loo and len(configs) > 1:
        for holdout in configs:
            calib_cfgs = [cfg for cfg in configs if cfg != holdout]
            for n_per_task in sample_sizes:
                if any(n_per_task > len(calib_by_config[cfg]) for cfg in calib_cfgs):
                    continue
                reps = 1 if all(n_per_task == len(calib_by_config[cfg]) for cfg in calib_cfgs) else args.repetitions
                for rep in range(reps):
                    indices = []
                    for cfg in calib_cfgs:
                        pool = calib_by_config[cfg]
                        indices.extend(pool[:] if n_per_task == len(pool) else rng.sample(pool, n_per_task))
                    maybe_evaluate_subset(source=f"loo_excl_{holdout}", indices=indices, n_label=n_per_task, rep=rep)

    rows_path = output_dir / args.rows_filename
    with rows_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    # Aggregate by source/n/eval.
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["source"], int(row["n_label"]), row["eval"])].append(row)

    summary: dict[str, Any] = {
        "config": {
            "bundle": str(bundle_dir),
            "configs": configs,
            "sample_sizes": sample_sizes,
            "repetitions": args.repetitions,
            "ranks": ranks,
            "k_bits": k_bits_grid,
            "limit_layers": args.limit_layers,
            "eps": args.eps,
            "seed": args.seed,
            "include_loo": args.include_loo,
            "holdout_per_config": args.holdout_per_config,
            "holdout_seed": args.holdout_seed,
            "calibration_examples_per_config": {cfg: len(calib_by_config[cfg]) for cfg in configs},
            "eval_examples_per_config": {cfg: len(eval_by_config[cfg]) for cfg in configs},
        },
        "groups": {},
    }
    metric_names = [f"subspace_overlap_r{r}" for r in ranks]
    for kb in k_bits_grid:
        metric_names.extend([f"log2_regret_k{kb}", f"bit_l1_k{kb}"])
    for key, group_rows in sorted(grouped.items()):
        source, n_label, eval_name = key
        out_key = f"{source}|n={n_label}|eval={eval_name}"
        summary["groups"][out_key] = {
            metric: summarize([float(r[metric]) for r in group_rows])
            for metric in metric_names
        }
        summary["groups"][out_key]["n_trials"] = len(group_rows)

    summary_path = output_dir / "phase1d_k_basis_stability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    md_path = output_dir / "phase1d_k_basis_stability_summary.md"
    regret_cols = [f"log2_regret_k{kb}" for kb in k_bits_grid]
    # Pick the median rank (or the only one) for the headline overlap column.
    # Hard-coding r64 used to KeyError when --ranks omitted 64.
    overlap_rank = ranks[len(ranks) // 2]
    overlap_col = f"subspace_overlap_r{overlap_rank}"
    header = (
        f"| source | n label | eval | overlap r{overlap_rank} mean | "
        + " | ".join(f"regret k{kb} mean" for kb in k_bits_grid)
        + " | trials |"
    )
    align = (
        "|---|---:|---|---:|"
        + "".join("---:|" for _ in k_bits_grid)
        + "---:|"
    )
    lines = [
        "# Phase 1D K-Basis Stability Summary\n",
        "",
        "Lower `log2_regret_k*` is better. Zero means the subset-calibrated basis/allocation",
        "matches the full eval-reference basis/allocation under the closed-form K distortion metric.",
        "",
        header,
        align,
    ]
    for key in sorted(summary["groups"]):
        source, n_part, eval_part = key.split("|")
        n_label = n_part.split("=", 1)[1]
        eval_name = eval_part.split("=", 1)[1]
        g = summary["groups"][key]
        regret_values = " | ".join(f"{g[col]['mean']:.4f}" for col in regret_cols)
        lines.append(
            f"| {source} | {n_label} | {eval_name} | "
            f"{g[overlap_col]['mean']:.4f} | "
            f"{regret_values} | "
            f"{g['n_trials']} |"
        )
    md_path.write_text("\n".join(lines) + "\n")

    log(f"Wrote rows:    {rows_path}")
    log(f"Wrote summary: {summary_path}")
    log(f"Wrote table:   {md_path}")


if __name__ == "__main__":
    main()
