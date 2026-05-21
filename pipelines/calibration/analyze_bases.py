#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipelines.calibration.common import (
    DEFAULT_K_BITS,
    DEFAULT_RANKS,
    DEFAULT_SAMPLE_SIZES,
    DEFAULT_V_BITS,
    RunPaths,
    StatsCache,
    add_common_args,
    add_shard_args,
    log,
    parse_csv_ints,
    raw_example_path,
    read_json,
    read_jsonl,
    rows_for_shard,
    select_rows,
    stable_example_id,
    stats_example_path,
    summarize_values,
    validate_shard_args,
    validate_split,
    write_json,
    write_jsonl,
)
from kvq.toolkit.metric_transform import water_fill
from kvq.toolkit.per_coord_quantization import PerCoordCompressor, round_bits_to_integer
from kvq.toolkit.quantization import Stage1MSECompressor


K_METHODS = ("v3", "q_only", "k_only", "jointqk")
V_METHODS = ("v_random", "v_eigen_uniform", "v_eigen_waterfill")
PRODUCTION_MAX_COORD_BITS = 8  # mirrors `build_jointqk_compressor`'s MAX_BITS for waterfill methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze final calibration basis convergence and generalization.")
    add_common_args(parser)
    add_shard_args(parser)
    parser.add_argument("--sample-sizes", default=",".join(map(str, DEFAULT_SAMPLE_SIZES)))
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--k-bits", default=",".join(map(str, DEFAULT_K_BITS)))
    parser.add_argument("--v-bits", default=",".join(map(str, DEFAULT_V_BITS)))
    parser.add_argument("--ranks", default=",".join(map(str, DEFAULT_RANKS)))
    parser.add_argument("--seed", type=int, default=20260505)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument(
        "--empirical",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute empirical raw-tensor metrics (k_mse, logit_error, top1, top5) on held-out "
        "examples. Default ON because the analytic Bennett proxy cannot detect clipping bias. "
        "Pass --no-empirical for analytic-only runs (much cheaper). For unbounded full eval set, "
        "set --empirical-max-eval-examples 0 explicitly.",
    )
    parser.add_argument(
        "--empirical-max-eval-examples",
        type=int,
        default=0,
        help="Limit raw held-out examples per eval for empirical metrics; 0 means all.",
    )
    parser.add_argument("--empirical-max-layers", type=int, default=0, help="Limit layers for empirical metrics; 0 means all.")
    parser.add_argument("--empirical-max-heads", type=int, default=0, help="Limit KV heads for empirical metrics; 0 means all.")
    parser.add_argument("--empirical-max-tokens", type=int, default=0, help="Limit context tokens for empirical metrics; 0 means all.")
    parser.add_argument("--max-coord-bits", type=int, default=8, help="Maximum bits assigned to any one rotated coordinate.")
    parser.add_argument(
        "--stats-cache-entries",
        type=int,
        default=128,
        help="Per-shard LRU cap on cached per-example stats files (~75 MB each). 0 = unbounded "
        "(can OOM for large eval pools). Default 128 ≈ ~10 GB.",
    )
    parser.add_argument(
        "--raw-cache-entries",
        type=int,
        default=0,
        help="Per-trial cap on cached raw q/k/v payloads (~5 GB each). 0 = no cache (reload "
        "per (method, idx) but bounded). A positive value (e.g. 2-4) trades RAM for fewer "
        "disk reads. The previous unbounded `dict` cache crashed the server on pooled trials.",
    )
    return parser.parse_args()


def _sym(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x.float() + x.float().transpose(-1, -2))


def regularize_batch(cov: torch.Tensor, eps: float) -> torch.Tensor:
    d = cov.shape[-1]
    sym = _sym(cov)
    trace = sym.diagonal(dim1=-2, dim2=-1).sum(-1)
    eye = torch.eye(d, dtype=sym.dtype, device=sym.device)
    return sym + (eps * (trace / d).clamp_min(1e-12)).unsqueeze(-1).unsqueeze(-1) * eye


def eigvec_desc(cov: torch.Tensor, eps: float) -> torch.Tensor:
    reg = regularize_batch(cov, eps)
    vals, vecs = torch.linalg.eigh(reg)
    order = torch.argsort(vals, dim=-1, descending=True)
    return torch.gather(vecs, -1, order.unsqueeze(-2).expand(*vecs.shape[:-1], -1))


def jointqk_basis(sigma_q: torch.Tensor, sigma_k: torch.Tensor, eps: float) -> torch.Tensor:
    sq = regularize_batch(sigma_q, eps)
    sk = regularize_batch(sigma_k, eps)
    sym = _sym(sq @ sk)
    vals, vecs = torch.linalg.eigh(sym)
    order = torch.argsort(vals, dim=-1, descending=True)
    return torch.gather(vecs, -1, order.unsqueeze(-2).expand(*vecs.shape[:-1], -1))


def random_orthogonal(shape: tuple[int, int, int, int], seed: int, device: torch.device) -> torch.Tensor:
    n_layers, n_heads, d, _ = shape
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mats = torch.randn((n_layers, n_heads, d, d), generator=gen).to(device)
    q, r = torch.linalg.qr(mats)
    signs = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).clamp_min(0) * 2 - 1
    return q * signs.unsqueeze(-2)


def allocate_bits(scores: torch.Tensor, bits: int, max_coord_bits: int = PRODUCTION_MAX_COORD_BITS) -> torch.Tensor:
    """Per-coord integer bit allocation matching `build_jointqk_compressor`'s waterfill path.

    Mirrors `src/kvq/toolkit/per_coord_quantization.py:_waterfill` byte-for-byte
    (continuous water-fill with iterative saturation at 16, largest-remainder rounding to
    preserve sum, then cap at `max_coord_bits` with redistribution to lowest-allocation coords).
    Calling the same toolkit primitives is the point — calibration's allocation must match
    deployment's allocation, otherwise calibration's "winner" may not be the deployed winner.

    Args:
        scores: (..., d) per-coord scoring (e.g. `q_diag * k_diag` for K logit MSE).
        bits:   integer average bits per coord; total budget = bits * d.
        max_coord_bits: cap on per-coord bits (default 8, matching production's MAX_BITS).

    Returns: (..., d) long tensor with sum along last dim equal to `bits * d` (modulo
    cap-redistribute conservation when caps bite hard enough that not all excess can be reseated).
    """
    leading_shape = scores.shape[:-1]
    d = int(scores.shape[-1])
    device = scores.device
    total_bits = int(bits) * d

    # Continuous waterfill — water_fill itself iterates over saturation at internal max=16 bits.
    flat_scores = scores.detach().float().cpu().reshape(-1, d).clamp_min(1e-30)
    bits_continuous = water_fill(flat_scores, total_bits=float(total_bits))  # (N, d)

    # Largest-remainder integer rounding preserving the per-row sum.
    bits_int = round_bits_to_integer(bits_continuous, total_bits=total_bits)  # (N, d) long

    # Cap per-coord bits at max_coord_bits, redistribute the saved bits to coords with the
    # lowest current allocation — same loop as build_jointqk_compressor.
    out = bits_int.clone()
    for i in range(out.shape[0]):
        excess = int((out[i] - max_coord_bits).clamp_min(0).sum().item())
        out[i].clamp_(max=max_coord_bits)
        while excess > 0:
            below = (out[i] < max_coord_bits).nonzero(as_tuple=True)[0]
            if below.numel() == 0:
                break
            min_val = out[i, below].min()
            candidates = below[out[i, below] == min_val]
            take = min(int(candidates.numel()), excess)
            out[i, candidates[:take]] += 1
            excess -= take

    return out.reshape(leading_shape + (d,)).to(device)


def mse_dist(var_diag: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
    return (var_diag.clamp_min(1e-30) * torch.pow(torch.tensor(2.0), -2.0 * bits.float())).sum(dim=-1)


class RawLRU:
    """Bounded LRU for raw q/k/v `.pt` payloads (~5 GB each).

    Replaces the previous unbounded `dict` cache that crashed the server on pooled trials
    (80 eval idx × 5 GB = 400 GB resident). With max_entries=0 the cache is bypassed and
    each `get` reloads from disk — bounded but slower; pair with the per-idx free in the
    empirical funcs.
    """

    def __init__(self, max_entries: int = 0) -> None:
        self._d: "OrderedDict[int, dict[str, Any]]" = OrderedDict()
        self._max = int(max_entries)

    def get(self, idx: int, loader) -> dict[str, Any]:
        if self._max <= 0:
            return loader()
        cached = self._d.get(idx)
        if cached is not None:
            self._d.move_to_end(idx)
            return cached
        item = loader()
        self._d[idx] = item
        while len(self._d) > self._max:
            self._d.popitem(last=False)
        return item

    def evict(self, idx: int) -> None:
        if self._max <= 0:
            return
        self._d.pop(idx, None)

    def clear(self) -> None:
        self._d.clear()


def _release_idx(raw_lru: RawLRU | None, idx: int) -> None:
    """Drop `idx` from the LRU (no-op if disabled), then run gc + empty CUDA cache.

    Called at the bottom of each idx loop in empirical funcs so peak resident raw is
    bounded by `raw_cache_entries` (default 0 → ~5 GB peak per shard).
    """
    if raw_lru is not None:
        raw_lru.evict(idx)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def combine_stats(
    paths: RunPaths,
    per_example: list[dict[str, Any]],
    indices: list[int],
    device: torch.device,
    cache: StatsCache | None = None,
) -> dict[str, Any]:
    if not indices:
        raise ValueError("empty subset")

    def _load(idx: int) -> dict[str, Any]:
        path = paths.root / per_example[idx]["file"]
        if cache is not None:
            return cache.load(path)
        return torch.load(path, map_location="cpu", weights_only=False)

    first = _load(indices[0])
    sq = torch.zeros_like(first["sq_sum"])
    sk = torch.zeros_like(first["sk_sum"])
    sv = torch.zeros_like(first["sv_sum"])
    sum_q = torch.zeros_like(first["sum_q"])
    sum_k = torch.zeros_like(first["sum_k"])
    sum_v = torch.zeros_like(first["sum_v"])
    tokens = 0
    for idx in indices:
        item = _load(idx)
        sq += item["sq_sum"]
        sk += item["sk_sum"]
        sv += item["sv_sum"]
        sum_q += item["sum_q"]
        sum_k += item["sum_k"]
        sum_v += item["sum_v"]
        tokens += int(item["tokens"])
    group = int(first["group"])
    sigma_q = (sq / float(group * tokens)).to(device)
    sigma_k = (sk / float(tokens)).to(device)
    sigma_v = (sv / float(tokens)).to(device)
    mu_q = (sum_q / float(group * tokens)).to(device)
    mu_k = (sum_k / float(tokens)).to(device)
    mu_v = (sum_v / float(tokens)).to(device)
    cov_q = sigma_q - torch.einsum("lhd,lhe->lhde", mu_q, mu_q)
    cov_k = sigma_k - torch.einsum("lhd,lhe->lhde", mu_k, mu_k)
    cov_v = sigma_v - torch.einsum("lhd,lhe->lhde", mu_v, mu_v)
    return {
        "sigma_q": sigma_q,
        "sigma_k": sigma_k,
        "mu_q": mu_q,
        "mu_k": mu_k,
        "mu_v": mu_v,
        "cov_q": cov_q,
        "cov_k": cov_k,
        "cov_v": cov_v,
        "tokens": tokens,
        "indices": indices,
    }


def build_trials(per_example: list[dict[str, Any]], tasks: list[str], sample_sizes: list[int], repetitions: int, seed: int) -> list[dict[str, Any]]:
    by_task_train: dict[str, list[int]] = {task: [] for task in tasks}
    for item in per_example:
        if item["split"] == "train" and item["config"] in by_task_train:
            by_task_train[item["config"]].append(int(item["index"]))

    rng = random.Random(seed)
    trials: list[dict[str, Any]] = []

    def add(source: str, eval_name: str, n_label: int, rep: int, indices: list[int], regime: str) -> None:
        trials.append(
            {
                "trial_idx": len(trials),
                "source": source,
                "eval": eval_name,
                "n_label": n_label,
                "rep": rep,
                "indices": indices,
                "regime": regime,
            }
        )

    for task in tasks:
        pool = by_task_train[task]
        for n in sample_sizes:
            if n > len(pool):
                continue
            reps = 1 if n == len(pool) else repetitions
            for rep in range(reps):
                indices = pool[:] if n == len(pool) else rng.sample(pool, n)
                add(task, task, n, rep, indices, "same_task")

    for n in sample_sizes:
        if any(n > len(by_task_train[task]) for task in tasks):
            continue
        reps = 1 if all(n == len(by_task_train[task]) for task in tasks) else repetitions
        for rep in range(reps):
            indices: list[int] = []
            for task in tasks:
                pool = by_task_train[task]
                indices.extend(pool[:] if n == len(pool) else rng.sample(pool, n))
            add("pooled_stratified", "pooled", n, rep, indices, "pooled")

    for holdout in tasks:
        calib_tasks = [task for task in tasks if task != holdout]
        for n in sample_sizes:
            if any(n > len(by_task_train[task]) for task in calib_tasks):
                continue
            reps = 1 if all(n == len(by_task_train[task]) for task in calib_tasks) else repetitions
            for rep in range(reps):
                indices = []
                for task in calib_tasks:
                    pool = by_task_train[task]
                    indices.extend(pool[:] if n == len(pool) else rng.sample(pool, n))
                add(f"loo_excl_{holdout}", holdout, n, rep, indices, "loo")
    return trials


def eval_indices(per_example: list[dict[str, Any]], eval_name: str) -> list[int]:
    if eval_name == "pooled":
        return [int(item["index"]) for item in per_example if item["split"] == "test"]
    return [int(item["index"]) for item in per_example if item["split"] == "test" and item["config"] == eval_name]


def subspace_overlap(candidate: torch.Tensor, reference: torch.Tensor, rank: int) -> torch.Tensor:
    c = candidate[..., :, :rank]
    r = reference[..., :, :rank]
    return (c.transpose(-1, -2) @ r).square().sum(dim=(-2, -1)) / float(rank)


def _layer_mask(n_layers: int, include_layer0: bool, device: torch.device) -> torch.Tensor:
    mask = torch.ones(n_layers, dtype=torch.bool, device=device)
    if not include_layer0 and mask.numel() > 0:
        mask[0] = False
    return mask


def _ensure_modes(layer0_modes: list[bool] | None) -> list[bool]:
    if layer0_modes is None:
        return [True, False]
    return list(layer0_modes)


def v3_rows_for_method(
    train_stats: dict[str, Any],
    eval_stats: dict[str, Any],
    k_bits: list[int],
    layer0_modes: list[bool] | None = None,
) -> dict[bool, dict[str, float]]:
    """Analytic V3 / TurboQuant baseline (random Hadamard + unit-norm + uniform Lloyd-Max).

    V3's noise in the original key space is isotropic (random rotation makes it direction-blind),
    so the closed-form Bennett distortion has no dependence on per-coord variance structure —
    only the traces matter:

        E[||k − k̂||²]     = trace(Σ_K) · 2^{-2b}
        E[(qᵀ(k − k̂))²]   = (1/d) · trace(Σ_K) · trace(Σ_Q) · 2^{-2b}

    V3 doesn't depend on calibration (no basis, no allocation), so its analytic curves are flat
    across sample size — that's the point: a calibration-independent floor for the others to beat.

    Returns dict keyed by `include_layer0` mode so the caller can emit one row per mode from a
    single computation pass.
    """
    sq_eval = eval_stats["sigma_q"]
    sk_eval = eval_stats["sigma_k"]
    d = int(sk_eval.shape[-1])
    tr_q = sq_eval.diagonal(dim1=-2, dim2=-1).sum(-1)  # (n_layers, n_kv_heads)
    tr_k = sk_eval.diagonal(dim1=-2, dim2=-1).sum(-1)
    modes = _ensure_modes(layer0_modes)
    masks = {m: _layer_mask(tr_k.shape[0], m, tr_k.device) for m in modes}
    out: dict[bool, dict[str, float]] = {m: {} for m in modes}
    for bits in k_bits:
        scale = float(2.0 ** (-2 * bits))
        k_mse_per_lh = tr_k * scale
        logit_per_lh = (tr_q * tr_k * scale) / float(d)
        for m in modes:
            mask = masks[m]
            out[m][f"k_mse_k{bits}"] = float(k_mse_per_lh[mask].mean().item())
            out[m][f"logit_error_k{bits}"] = float(logit_per_lh[mask].mean().item())
    return out


def empirical_v3_metrics(
    paths: RunPaths,
    per_example: list[dict[str, Any]],
    eval_subset: list[int],
    k_bits: list[int],
    layer0_modes: list[bool] | None,
    device: torch.device,
    seed: int,
    head_dim: int,
    n_layers_total: int,
    n_kv_heads_total: int,
    max_layers: int = 0,
    max_heads: int = 0,
    max_tokens: int = 0,
    raw_lru: "RawLRU | None" = None,
    return_accumulators: bool = False,
) -> dict[bool, dict[str, float]] | dict[int, dict[int, dict[str, float]]]:
    """Empirical V3 / TurboQuant: K MSE, mean (qᵀ err)², top-1 / top-5 retention.

    Uses `Stage1MSECompressor(head_dim, bits, seed)` — the same compressor production V3 uses.

    Inverted loop order (`for idx: for bits`) so each raw .pt file is loaded at most once per
    call instead of once per (bits, idx) pair. Per-layer accumulators are scoped outside both
    loops; include_layer0 masks are applied at aggregation time so True/False outputs come
    from one raw walk.
    """
    modes = _ensure_modes(layer0_modes)
    last_layer = min(n_layers_total, max_layers) if max_layers > 0 else n_layers_total
    layer_range_full = range(0, last_layer)
    head_range = range(min(n_kv_heads_total, max_heads) if max_heads > 0 else n_kv_heads_total)
    layer_head_pairs = [(l, h) for l in layer_range_full for h in head_range]

    compressors = {
        bits: Stage1MSECompressor(head_dim=int(head_dim), bits=int(bits), seed=int(seed), device=device)
        for bits in k_bits
    }
    accums: dict[int, dict[int, dict[str, float]]] = {
        bits: {
            l: {"mse_num": 0.0, "mse_den": 0, "logit_num": 0.0, "logit_den": 0,
                "top1_num": 0, "top1_den": 0, "top5_num": 0, "top5_den": 0}
            for l in layer_range_full
        }
        for bits in k_bits
    }

    for idx in eval_subset:
        meta = per_example[idx]
        loader = lambda p=paths.root / meta["raw_file"]: torch.load(p, map_location="cpu", weights_only=False)
        raw = raw_lru.get(idx, loader) if raw_lru is not None else loader()
        k_all = raw["k_post"]
        q_all = raw["q_post"]
        tokens = int(raw["prompt_length"])
        if max_tokens > 0:
            tokens = min(tokens, max_tokens)
        d = int(k_all.shape[-1])
        group = q_all.shape[1] // n_kv_heads_total
        for layer, head in layer_head_pairs:
            k = k_all[layer, head, :tokens, :].to(device).float()
            q = q_all[layer, head * group : (head + 1) * group, :tokens, :].to(device).float().reshape(-1, d)
            qq = q.transpose(0, 1) @ q
            k_t = k.transpose(0, 1)
            for bits in k_bits:
                acc = accums[bits][layer]
                compressor = compressors[bits]
                recon = compressor.roundtrip(k.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).float()
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

    if return_accumulators:
        return accums
    return _finalize_v3_accums(accums, modes, k_bits, layer_range_full)


def _finalize_v3_accums(
    accums: dict[int, dict[int, dict[str, float]]],
    modes: list[bool],
    k_bits: list[int],
    layer_range_full,
    metric_prefix: str = "empirical",
) -> dict[bool, dict[str, float]]:
    out: dict[bool, dict[str, float]] = {m: {} for m in modes}
    for bits in k_bits:
        per_layer = accums[bits]
        for m in modes:
            active_layers = [l for l in layer_range_full if (m or l > 0)]
            mse_num = sum(per_layer[l]["mse_num"] for l in active_layers)
            mse_den = sum(per_layer[l]["mse_den"] for l in active_layers)
            logit_num = sum(per_layer[l]["logit_num"] for l in active_layers)
            logit_den = sum(per_layer[l]["logit_den"] for l in active_layers)
            top1_num = sum(per_layer[l]["top1_num"] for l in active_layers)
            top1_den = sum(per_layer[l]["top1_den"] for l in active_layers)
            top5_num = sum(per_layer[l]["top5_num"] for l in active_layers)
            top5_den = sum(per_layer[l]["top5_den"] for l in active_layers)
            d = out[m]
            d[f"{metric_prefix}_k_mse_k{bits}"] = mse_num / max(1, mse_den)
            d[f"{metric_prefix}_logit_error_k{bits}"] = logit_num / max(1, logit_den)
            d[f"{metric_prefix}_top1_k{bits}"] = top1_num / max(1, top1_den)
            d[f"{metric_prefix}_top5_k{bits}"] = top5_num / max(1, top5_den)
    return out


def merge_accumulators(shards: list[dict]) -> dict:
    """Sum a list of accumulator dicts (deep-numeric merge): {bits: {layer: {stat: number}}}."""
    if not shards:
        return {}
    out: dict[int, dict[int, dict[str, float]]] = {}
    for shard in shards:
        for bits, per_layer in shard.items():
            bits = int(bits)
            obits = out.setdefault(bits, {})
            for layer, stats in per_layer.items():
                layer = int(layer)
                olayer = obits.setdefault(layer, {})
                for k, v in stats.items():
                    olayer[k] = olayer.get(k, 0) + v
    return out


def k_rows_for_method(
    method: str,
    basis: torch.Tensor,
    train_stats: dict[str, Any],
    eval_stats: dict[str, Any],
    ref_basis: torch.Tensor,
    k_bits: list[int],
    ranks: list[int],
    layer0_modes: list[bool] | None,
    max_coord_bits: int,
) -> dict[bool, dict[str, float]]:
    sq_train = train_stats["sigma_q"]
    sk_train = train_stats["sigma_k"]
    sq_eval = eval_stats["sigma_q"]
    sk_eval = eval_stats["sigma_k"]
    train_q_diag = (basis.transpose(-1, -2) @ sq_train @ basis).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    train_k_diag = (basis.transpose(-1, -2) @ sk_train @ basis).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    eval_q_diag = (basis.transpose(-1, -2) @ sq_eval @ basis).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    eval_k_diag = (basis.transpose(-1, -2) @ sk_eval @ basis).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    modes = _ensure_modes(layer0_modes)
    masks = {m: _layer_mask(basis.shape[0], m, basis.device) for m in modes}
    out: dict[bool, dict[str, float]] = {m: {} for m in modes}

    # Compute per-(layer, head) values once, then take masked means per mode.
    for rank in ranks:
        overlap_per_lh = subspace_overlap(basis, ref_basis, rank)
        for m in modes:
            out[m][f"subspace_overlap_r{rank}"] = float(overlap_per_lh[masks[m]].mean().item())
    for bits in k_bits:
        uniform_alloc = torch.full_like(train_k_diag, bits, dtype=torch.long)
        waterfill_alloc = allocate_bits(train_q_diag * train_k_diag, bits, max_coord_bits=max_coord_bits)
        u_kmse_per_lh = mse_dist(eval_k_diag, uniform_alloc)
        u_logit_per_lh = mse_dist(eval_q_diag * eval_k_diag, uniform_alloc)
        w_kmse_per_lh = mse_dist(eval_k_diag, waterfill_alloc)
        w_logit_per_lh = mse_dist(eval_q_diag * eval_k_diag, waterfill_alloc)
        active_per_lh = (waterfill_alloc > 0).float().mean(dim=-1)  # fraction active per (layer, head)
        max_bits_per_lh = waterfill_alloc.float().amax(dim=-1)
        for m in modes:
            mask = masks[m]
            wkmse = float(w_kmse_per_lh[mask].mean().item())
            wlog = float(w_logit_per_lh[mask].mean().item())
            d = out[m]
            d[f"k_mse_uniform_k{bits}"] = float(u_kmse_per_lh[mask].mean().item())
            d[f"logit_error_uniform_k{bits}"] = float(u_logit_per_lh[mask].mean().item())
            d[f"k_mse_waterfill_k{bits}"] = wkmse
            d[f"logit_error_waterfill_k{bits}"] = wlog
            d[f"waterfill_active_coords_k{bits}"] = float(active_per_lh[mask].mean().item())
            d[f"waterfill_max_coord_bits_k{bits}"] = float(max_bits_per_lh[mask].mean().item())
            # Backward-compatible aliases: K headline metrics use water-fill allocation.
            d[f"k_mse_k{bits}"] = wkmse
            d[f"logit_error_k{bits}"] = wlog
    return out


def v_rows_for_method(
    method: str,
    basis: torch.Tensor,
    train_stats: dict[str, Any],
    eval_stats: dict[str, Any],
    v_bits: list[int],
    layer0_modes: list[bool] | None,
    max_coord_bits: int,
) -> dict[bool, dict[str, float]]:
    cov_train = train_stats["cov_v"]
    cov_eval = eval_stats["cov_v"]
    train_diag = (basis.transpose(-1, -2) @ cov_train @ basis).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    eval_diag = (basis.transpose(-1, -2) @ cov_eval @ basis).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    modes = _ensure_modes(layer0_modes)
    masks = {m: _layer_mask(basis.shape[0], m, basis.device) for m in modes}
    out: dict[bool, dict[str, float]] = {m: {} for m in modes}
    for bits in v_bits:
        if method.endswith("_waterfill"):
            bit_alloc = allocate_bits(train_diag, bits, max_coord_bits=max_coord_bits)
        else:
            bit_alloc = torch.full_like(train_diag, bits, dtype=torch.long)
        per_lh = mse_dist(eval_diag, bit_alloc)
        for m in modes:
            out[m][f"v_mse_v{bits}"] = float(per_lh[masks[m]].mean().item())
    return out


def empirical_k_metrics(
    paths: RunPaths,
    per_example: list[dict[str, Any]],
    eval_subset: list[int],
    basis: torch.Tensor,
    train_stats: dict[str, Any],
    k_bits: list[int],
    layer0_modes: list[bool] | None,
    device: torch.device,
    max_layers: int = 0,
    max_heads: int = 0,
    max_tokens: int = 0,
    max_coord_bits: int = 8,
    raw_lru: "RawLRU | None" = None,
    return_accumulators: bool = False,
) -> dict[bool, dict[str, float]] | dict[int, dict[int, dict[str, float]]]:
    sq_train = train_stats["sigma_q"]
    sk_train = train_stats["sigma_k"]
    train_q_diag = (basis.transpose(-1, -2) @ sq_train @ basis).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    train_k_diag = (basis.transpose(-1, -2) @ sk_train @ basis).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    bit_allocs = {bits: allocate_bits(train_q_diag * train_k_diag, bits, max_coord_bits=max_coord_bits).cpu() for bits in k_bits}
    basis_cpu = basis.detach().cpu()
    train_k_std_cpu = torch.sqrt(train_k_diag.detach().cpu())

    modes = _ensure_modes(layer0_modes)
    n_layers, n_kv_heads = sq_train.shape[:2]
    last_layer = min(n_layers, max_layers) if max_layers > 0 else n_layers
    layer_range_full = range(0, last_layer)
    head_range = range(min(n_kv_heads, max_heads) if max_heads > 0 else n_kv_heads)
    layer_head_pairs = [(l, h) for l in layer_range_full for h in head_range]

    # Build all (layer, head) compressors once per `bits`. Without this, the Lloyd-Max codebook
    # solve runs once per (idx, layer, head) — i.e. ~80 times per (bits, layer, head) on the
    # default pooled eval, even though the compressor only depends on (bits, layer, head).
    comps_by_bits: dict[int, dict[tuple[int, int], PerCoordCompressor]] = {}
    for bits in k_bits:
        alloc = bit_allocs[bits]
        comps_by_bits[bits] = {
            (l, h): PerCoordCompressor(
                bits_per_coord=alloc[l, h],
                std_per_coord=train_k_std_cpu[l, h],
                forward_map=basis_cpu[l, h],
                inverse_map=basis_cpu[l, h].transpose(-1, -2),
            ).to(device)
            for (l, h) in layer_head_pairs
        }

    accums: dict[int, dict[int, dict[str, float]]] = {
        bits: {
            l: {"mse_num": 0.0, "mse_den": 0, "logit_num": 0.0, "logit_den": 0,
                "top1_num": 0, "top1_den": 0, "top5_num": 0, "top5_den": 0}
            for l in layer_range_full
        }
        for bits in k_bits
    }

    for idx in eval_subset:
        meta = per_example[idx]
        loader = lambda p=paths.root / meta["raw_file"]: torch.load(p, map_location="cpu", weights_only=False)
        raw = raw_lru.get(idx, loader) if raw_lru is not None else loader()
        k_all = raw["k_post"]
        q_all = raw["q_post"]
        tokens = int(raw["prompt_length"])
        if max_tokens > 0:
            tokens = min(tokens, max_tokens)
        d = int(k_all.shape[-1])
        group = q_all.shape[1] // n_kv_heads
        for layer, head in layer_head_pairs:
            k = k_all[layer, head, :tokens, :].to(device).float()
            q = q_all[layer, head * group : (head + 1) * group, :tokens, :].to(device).float().reshape(-1, d)
            qq = q.transpose(0, 1) @ q
            k_t = k.transpose(0, 1)
            for bits in k_bits:
                acc = accums[bits][layer]
                comp = comps_by_bits[bits][(layer, head)]
                recon = comp.roundtrip(k).float()
                err = k - recon
                acc["mse_num"] += float(err.square().sum().item())
                acc["mse_den"] += int(err.numel())
                ee = err.transpose(0, 1) @ err
                acc["logit_num"] += float((qq * ee).sum().item())
                acc["logit_den"] += int(group * tokens * tokens)
                # Top-1 / top-5 attention retention: original top-1 key must match (top-1) or
                # be contained in approx top-5. Stage 1E showed this can disagree with Bennett
                # MSE under joint Q-K-aligned bases (V3-vs-CCA top-1 puzzle), so we measure it
                # directly. Chunk over queries to bound peak memory at ~512 MiB per logit matrix.
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

    if return_accumulators:
        return accums
    return _finalize_k_accums(accums, modes, k_bits, layer_range_full)


def _finalize_k_accums(
    accums: dict[int, dict[int, dict[str, float]]],
    modes: list[bool],
    k_bits: list[int],
    layer_range_full,
) -> dict[bool, dict[str, float]]:
    out: dict[bool, dict[str, float]] = {m: {} for m in modes}
    for bits in k_bits:
        per_layer = accums[bits]
        for m in modes:
            active_layers = [l for l in layer_range_full if (m or l > 0)]
            mse_num = sum(per_layer[l]["mse_num"] for l in active_layers)
            mse_den = sum(per_layer[l]["mse_den"] for l in active_layers)
            logit_num = sum(per_layer[l]["logit_num"] for l in active_layers)
            logit_den = sum(per_layer[l]["logit_den"] for l in active_layers)
            top1_num = sum(per_layer[l]["top1_num"] for l in active_layers)
            top1_den = sum(per_layer[l]["top1_den"] for l in active_layers)
            top5_num = sum(per_layer[l]["top5_num"] for l in active_layers)
            top5_den = sum(per_layer[l]["top5_den"] for l in active_layers)
            k_mse = mse_num / max(1, mse_den)
            logit_error = logit_num / max(1, logit_den)
            top1_ret = top1_num / max(1, top1_den)
            top5_ret = top5_num / max(1, top5_den)
            d = out[m]
            d[f"empirical_k_mse_waterfill_k{bits}"] = k_mse
            d[f"empirical_logit_error_waterfill_k{bits}"] = logit_error
            d[f"empirical_top1_waterfill_k{bits}"] = top1_ret
            d[f"empirical_top5_waterfill_k{bits}"] = top5_ret
            # Backward-compatible aliases: empirical K metrics use water-fill allocation.
            d[f"empirical_k_mse_k{bits}"] = k_mse
            d[f"empirical_logit_error_k{bits}"] = logit_error
            d[f"empirical_top1_k{bits}"] = top1_ret
            d[f"empirical_top5_k{bits}"] = top5_ret
    return out


def empirical_v_metrics(
    paths: RunPaths,
    per_example: list[dict[str, Any]],
    eval_subset: list[int],
    basis: torch.Tensor,
    train_stats: dict[str, Any],
    v_bits: list[int],
    method: str,
    layer0_modes: list[bool] | None,
    device: torch.device,
    max_layers: int = 0,
    max_heads: int = 0,
    max_tokens: int = 0,
    max_coord_bits: int = 8,
    raw_lru: "RawLRU | None" = None,
    return_accumulators: bool = False,
) -> dict[bool, dict[str, float]] | dict[int, dict[int, dict[str, float]]]:
    cov_train = train_stats["cov_v"]
    mu_v = train_stats["mu_v"].detach().cpu()
    train_diag = (basis.transpose(-1, -2) @ cov_train @ basis).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    if method.endswith("_waterfill"):
        bit_allocs = {bits: allocate_bits(train_diag, bits, max_coord_bits=max_coord_bits).cpu() for bits in v_bits}
    else:
        bit_allocs = {bits: torch.full_like(train_diag.cpu(), bits, dtype=torch.long) for bits in v_bits}
    basis_cpu = basis.detach().cpu()
    train_v_std_cpu = torch.sqrt(train_diag.detach().cpu())

    modes = _ensure_modes(layer0_modes)
    n_layers, n_kv_heads = cov_train.shape[:2]
    last_layer = min(n_layers, max_layers) if max_layers > 0 else n_layers
    layer_range_full = range(0, last_layer)
    head_range = range(min(n_kv_heads, max_heads) if max_heads > 0 else n_kv_heads)
    layer_head_pairs = [(l, h) for l in layer_range_full for h in head_range]

    # Build (layer, head) compressors once per `bits`. Same hoist pattern as empirical_k_metrics.
    comps_by_bits: dict[int, dict[tuple[int, int], PerCoordCompressor]] = {}
    for bits in v_bits:
        alloc = bit_allocs[bits]
        comps_by_bits[bits] = {
            (l, h): PerCoordCompressor(
                bits_per_coord=alloc[l, h],
                std_per_coord=train_v_std_cpu[l, h],
                forward_map=basis_cpu[l, h],
                inverse_map=basis_cpu[l, h].transpose(-1, -2),
            ).to(device)
            for (l, h) in layer_head_pairs
        }

    accums: dict[int, dict[int, dict[str, float]]] = {
        bits: {l: {"mse_num": 0.0, "mse_den": 0} for l in layer_range_full}
        for bits in v_bits
    }

    for idx in eval_subset:
        meta = per_example[idx]
        loader = lambda p=paths.root / meta["raw_file"]: torch.load(p, map_location="cpu", weights_only=False)
        raw = raw_lru.get(idx, loader) if raw_lru is not None else loader()
        v_all = raw["v"]
        tokens = int(raw["prompt_length"])
        if max_tokens > 0:
            tokens = min(tokens, max_tokens)
        for layer, head in layer_head_pairs:
            mean = mu_v[layer, head].to(device).float()
            v = v_all[layer, head, :tokens, :].to(device).float()
            v_centered = v - mean
            for bits in v_bits:
                acc = accums[bits][layer]
                comp = comps_by_bits[bits][(layer, head)]
                recon = comp.roundtrip(v_centered) + mean
                err = v - recon
                acc["mse_num"] += float(err.square().sum().item())
                acc["mse_den"] += int(err.numel())
                del recon, err
            del mean, v, v_centered
        del raw, v_all
        _release_idx(raw_lru, idx)

    if return_accumulators:
        return accums
    return _finalize_v_accums(accums, modes, v_bits, layer_range_full)


def _finalize_v_accums(
    accums: dict[int, dict[int, dict[str, float]]],
    modes: list[bool],
    v_bits: list[int],
    layer_range_full,
) -> dict[bool, dict[str, float]]:
    out: dict[bool, dict[str, float]] = {m: {} for m in modes}
    for bits in v_bits:
        per_layer = accums[bits]
        for m in modes:
            active_layers = [l for l in layer_range_full if (m or l > 0)]
            mse_num = sum(per_layer[l]["mse_num"] for l in active_layers)
            mse_den = sum(per_layer[l]["mse_den"] for l in active_layers)
            out[m][f"empirical_v_mse_v{bits}"] = mse_num / max(1, mse_den)
    return out


def merge_analysis(paths: RunPaths, expected_trials: int) -> None:
    rows: list[dict[str, Any]] = []
    for shard_file in sorted(paths.analysis_dir.glob("shard_*/rows.jsonl")):
        rows.extend(read_jsonl(shard_file))
    write_jsonl(paths.analysis_dir / "analysis_rows.jsonl", rows)
    groups: dict[tuple[str, str, int, str, str, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["eval"], int(row["n_label"]), row["regime"], row["metric_family"], bool(row["include_layer0"]))].append(row)
    summary: dict[str, Any] = {
        "metadata": {"expected_trials": expected_trials, "num_rows": len(rows)},
        "groups": {},
    }
    for key, vals in sorted(groups.items()):
        method, eval_name, n_label, regime, metric_family, include_layer0 = key
        gkey = f"{method}|eval={eval_name}|n={n_label}|regime={regime}|family={metric_family}|layer0={include_layer0}"
        metrics = sorted({
            k
            for row in vals
            for k in row
            if k.endswith("_mean")
            or k.startswith(("k_mse_", "logit_error_", "v_mse_", "subspace_overlap_", "waterfill_", "empirical_"))
        })
        summary["groups"][gkey] = {metric: summarize_values([float(row[metric]) for row in vals if metric in row]) for metric in metrics}
        summary["groups"][gkey]["n_rows"] = len(vals)
    write_json(paths.analysis_dir / "analysis_summary.json", summary)


def main() -> None:
    args = parse_args()
    validate_shard_args(args.num_shards, args.shard_id)
    paths = RunPaths.from_args(args.artifact_root, args.run_id)
    paths.ensure()
    split = read_json(paths.split_dir / "manifest.json" if (paths.split_dir / "manifest.json").exists() else args.split_manifest)
    validate_split(split)
    tasks = list(split["config"]["tasks"])
    all_rows = select_rows(split, args.smoke)
    assigned_rows = []
    for sid in range(args.num_shards):
        assigned_rows.extend(rows_for_shard(all_rows, args.num_shards, sid))

    aggregate = torch.load(paths.stats_dir / "aggregate.pt", map_location="cpu", weights_only=False)
    per_example = aggregate["per_example"]
    sample_sizes = parse_csv_ints(args.sample_sizes)
    if args.smoke:
        sample_sizes = [min(n for n in sample_sizes if n <= 2)]
        args.repetitions = 1
        if args.empirical and args.empirical_max_eval_examples <= 0:
            args.empirical_max_eval_examples = 1
        if args.empirical and args.empirical_max_layers <= 0:
            args.empirical_max_layers = 2
        if args.empirical and args.empirical_max_heads <= 0:
            args.empirical_max_heads = 1
        if args.empirical and args.empirical_max_tokens <= 0:
            args.empirical_max_tokens = 256
        if args.empirical:
            args.max_coord_bits = min(args.max_coord_bits, 3)
    trials = build_trials(per_example, tasks, sample_sizes, args.repetitions, args.seed)
    if args.merge_only:
        merge_analysis(paths, expected_trials=len(trials))
        return

    device = torch.device(args.device)
    k_bits = parse_csv_ints(args.k_bits)
    v_bits = parse_csv_ints(args.v_bits)
    ranks = parse_csv_ints(args.ranks)
    if args.smoke and args.empirical:
        k_bits = [3]
        v_bits = [3]
    elif args.smoke:
        k_bits = [bits for bits in k_bits if bits == 3] or [3]
        v_bits = [bits for bits in v_bits if bits == 3] or [3]
    rows_out: list[dict[str, Any]] = []
    shard_dir = paths.analysis_dir / f"shard_{args.shard_id:03d}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    eval_cache: dict[str, dict[str, Any]] = {}
    ref_cache: dict[str, torch.Tensor] = {}
    # Per-shard LRU cache for per-example stats `.pt` files. Each loaded payload is ~75 MB
    # (drops cqk_sum which we no longer use). With ~400 train + 80 test examples, the unbounded
    # cache fits in ~36 GB; on memory-constrained workers, set --stats-cache-entries to bound it.
    stats_cache = StatsCache(
        max_entries=args.stats_cache_entries if args.stats_cache_entries > 0 else None
    )

    def get_eval(name: str) -> dict[str, Any]:
        if name not in eval_cache:
            eval_cache[name] = combine_stats(
                paths, per_example, eval_indices(per_example, name), device, cache=stats_cache
            )
        return eval_cache[name]

    t_all = time.time()
    assigned_trials = [trial for trial in trials if int(trial["trial_idx"]) % args.num_shards == args.shard_id]
    trials_dir = shard_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    def trial_path(trial_idx: int) -> Path:
        return trials_dir / f"trial_{int(trial_idx):04d}.json"

    skipped = 0
    for i, trial in enumerate(assigned_trials):
        t0 = time.time()
        verbose_trial = bool(args.smoke)
        tpath = trial_path(int(trial["trial_idx"]))
        if args.resume and tpath.exists():
            try:
                cached = read_json(tpath)
                rows_out.extend(cached["rows"])
                skipped += 1
                if i < 3 or (i + 1) % 10 == 0 or i + 1 == len(assigned_trials):
                    print(
                        f"[{time.strftime('%H:%M:%S')}] analysis shard {args.shard_id}/{args.num_shards} "
                        f"resume-skip trial {i + 1}/{len(assigned_trials)} global={trial['trial_idx']} "
                        f"({len(cached['rows'])} rows from {tpath.name})",
                        flush=True,
                    )
                continue
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] could not read {tpath}: {e}; rerunning trial", flush=True)

        def log_trial_step(message: str) -> None:
            if verbose_trial:
                print(
                    f"[{time.strftime('%H:%M:%S')}] analysis shard {args.shard_id}/{args.num_shards} "
                    f"trial {i + 1}/{len(assigned_trials)} global={trial['trial_idx']} {message} "
                    f"elapsed={time.time() - t0:.1f}s",
                    flush=True,
                )

        trial_rows_start = len(rows_out)
        print(
            f"[{time.strftime('%H:%M:%S')}] analysis shard {args.shard_id}/{args.num_shards} "
            f"start trial {i + 1}/{len(assigned_trials)} global={trial['trial_idx']} "
            f"source={trial['source']} eval={trial['eval']} n={trial['n_label']}",
            flush=True,
        )
        train_stats = combine_stats(paths, per_example, trial["indices"], device, cache=stats_cache)
        eval_stats = get_eval(trial["eval"])
        eval_subset = eval_indices(per_example, trial["eval"])
        if args.empirical_max_eval_examples > 0:
            eval_subset = eval_subset[: args.empirical_max_eval_examples]
        compute_empirical = args.empirical and (not args.smoke or i == 0)
        # Bounded LRU for raw payloads (~5 GB each). 0 = bypass (each call reloads — bounded
        # memory, more disk I/O). The previous unbounded `dict` cache crashed the server on
        # pooled trials by accumulating all 80 eval idx (~400 GB resident).
        raw_lru = RawLRU(args.raw_cache_entries) if compute_empirical else None
        log_trial_step(f"prepared stats eval_examples={len(eval_subset)}")
        ref_key = trial["eval"]
        if ref_key not in ref_cache:
            ref_cache[ref_key] = jointqk_basis(eval_stats["sigma_q"], eval_stats["sigma_k"], args.eps)
        log_trial_step("built reference basis")
        k_bases = {
            "q_only": eigvec_desc(train_stats["sigma_q"], args.eps),
            "k_only": eigvec_desc(train_stats["sigma_k"], args.eps),
            "jointqk": jointqk_basis(train_stats["sigma_q"], train_stats["sigma_k"], args.eps),
        }
        log_trial_step("built k bases")
        v_shape = train_stats["cov_v"].shape
        v_bases = {
            "v_random": random_orthogonal(tuple(v_shape), seed=args.seed + int(trial["trial_idx"]), device=device),
            "v_eigen_uniform": eigvec_desc(train_stats["cov_v"], args.eps),
            "v_eigen_waterfill": eigvec_desc(train_stats["cov_v"], args.eps),
        }
        log_trial_step("built v bases")
        head_dim = int(train_stats["sigma_k"].shape[-1])
        n_layers_total = int(train_stats["sigma_k"].shape[0])
        n_kv_heads_total = int(train_stats["sigma_k"].shape[1])
        layer0_modes = [True, False]

        # Each *_rows_for_method / empirical_*_metrics call now returns dict[bool, dict[str, float]]:
        # one set of metric values per include_layer0 mode. The empirical loops process every layer
        # once and apply the layer-0 mask only at aggregation time, so we get both modes for the
        # cost of the True mode (~halves empirical wall time vs the previous two-pass loop).
        for method in K_METHODS:
            log_trial_step(f"k {method} analytic+empirical start")
            if method == "v3":
                analytic = v3_rows_for_method(train_stats, eval_stats, k_bits, layer0_modes)
                empirical = (
                    empirical_v3_metrics(
                        paths,
                        per_example,
                        eval_subset,
                        k_bits,
                        layer0_modes,
                        device,
                        seed=args.seed,
                        head_dim=head_dim,
                        n_layers_total=n_layers_total,
                        n_kv_heads_total=n_kv_heads_total,
                        max_layers=args.empirical_max_layers,
                        max_heads=args.empirical_max_heads,
                        max_tokens=args.empirical_max_tokens,
                        raw_lru=raw_lru,
                    )
                    if compute_empirical
                    else {m: {} for m in layer0_modes}
                )
            else:
                basis = k_bases[method]
                analytic = k_rows_for_method(
                    method,
                    basis,
                    train_stats,
                    eval_stats,
                    ref_cache[ref_key],
                    k_bits,
                    ranks,
                    layer0_modes,
                    args.max_coord_bits,
                )
                empirical = (
                    empirical_k_metrics(
                        paths,
                        per_example,
                        eval_subset,
                        basis,
                        train_stats,
                        k_bits,
                        layer0_modes,
                        device,
                        max_layers=args.empirical_max_layers,
                        max_heads=args.empirical_max_heads,
                        max_tokens=args.empirical_max_tokens,
                        max_coord_bits=args.max_coord_bits,
                        raw_lru=raw_lru,
                    )
                    if compute_empirical
                    else {m: {} for m in layer0_modes}
                )
            for mode in layer0_modes:
                metric_values = {**analytic[mode], **empirical[mode]}
                rows_out.append({**trial, "method": method, "metric_family": "k", "include_layer0": mode, **metric_values})

        for method, basis in v_bases.items():
            log_trial_step(f"v {method} analytic+empirical start")
            analytic = v_rows_for_method(method, basis, train_stats, eval_stats, v_bits, layer0_modes, args.max_coord_bits)
            empirical = (
                empirical_v_metrics(
                    paths,
                    per_example,
                    eval_subset,
                    basis,
                    train_stats,
                    v_bits,
                    method,
                    layer0_modes,
                    device,
                    max_layers=args.empirical_max_layers,
                    max_heads=args.empirical_max_heads,
                    max_tokens=args.empirical_max_tokens,
                    max_coord_bits=args.max_coord_bits,
                    raw_lru=raw_lru,
                )
                if compute_empirical
                else {m: {} for m in layer0_modes}
            )
            for mode in layer0_modes:
                metric_values = {**analytic[mode], **empirical[mode]}
                rows_out.append({**trial, "method": method, "metric_family": "v", "include_layer0": mode, **metric_values})
        # Atomic per-trial write so a mid-shard crash loses at most one trial. Resume picks
        # these up by checking trial_<idx>.json existence before recomputing.
        trial_rows = rows_out[trial_rows_start:]
        tmp = tpath.with_suffix(tpath.suffix + ".tmp")
        write_json(tmp, {"trial_idx": int(trial["trial_idx"]), "elapsed_seconds": time.time() - t0, "rows": trial_rows})
        tmp.replace(tpath)

        if i < 3 or (i + 1) % 10 == 0 or i + 1 == len(assigned_trials):
            print(
                f"[{time.strftime('%H:%M:%S')}] analysis shard {args.shard_id}/{args.num_shards} "
                f"trial {i + 1}/{len(assigned_trials)} global={trial['trial_idx']} "
                f"source={trial['source']} eval={trial['eval']} n={trial['n_label']} "
                f"in {time.time() - t0:.1f}s wrote {tpath.name}",
                flush=True,
            )
        del train_stats, k_bases, v_bases, raw_lru
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cs = stats_cache.stats()
    log(
        f"analysis shard {args.shard_id}/{args.num_shards} done in {time.time() - t_all:.1f}s "
        f"| trials={len(assigned_trials)} (resumed={skipped}) "
        f"| stats cache: entries={cs['entries']} hits={cs['hits']} misses={cs['misses']} "
        f"hit_rate={cs['hit_rate']:.2%}"
    )
    write_jsonl(shard_dir / "rows.jsonl", rows_out)
    write_json(
        shard_dir / "manifest.json",
        {
            "stage": "analysis",
            "num_rows": len(rows_out),
            "num_trials": len(assigned_trials),
            "expected_total_trials": len(trials),
            "elapsed_seconds": time.time() - t_all,
            "stats_cache": cs,
            "args": vars(args),
        },
    )


if __name__ == "__main__":
    main()
