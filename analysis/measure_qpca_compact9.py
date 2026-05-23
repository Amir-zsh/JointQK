#!/usr/bin/env python3
"""Standalone QPCA-only sweep on compact9 Qwen3 — comparison run against existing merged.json.

The QPCA basis (closed-form optimum to all-pairs inner-product reconstruction, see
background/derivation_qpca_for_queries.pdf):

  forward (k @ F)  = M_q^{1/2} V       where V eigvecs(M_q^{1/2} Σ_K M_q^{1/2}), descending
  inverse (r @ G)  = V^T M_q^{-1/2}
  code variance    = Λ (eigenvalues of A)
  bit allocation   = water-fill on Λ alone (Q-weight absorbed into the basis)

Mirrors analyze_bases_pooled.py's launcher + shard pattern but specialized to one method.
The point: we already have v3/q_only/k_only/jointqk numbers on compact9 in
artifacts/calibration/longbench_compact9_qkv_qwen3_8b/05_reports/pooled_n50/merged.json;
this script just computes the qpca column on the same coverage so we can diff.

Default output dir:
  artifacts/calibration/longbench_compact9_qkv_qwen3_8b/05_reports/pooled_n50_qpca/

Output files:
  shard_NNN.json — per-shard accumulators (sums of per-(bits, layer) numerators/denoms)
  shard_NNN.log  — per-shard stdout
  launcher.log   — orchestration + final comparison table
  qpca_merged.json — finalized per-(bits, layer0_mode) metrics, schema matches
                     merged.json["jointqk"] etc.
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
    _ensure_modes,
    _finalize_k_accums,
    _release_idx,
    _sym,
    allocate_bits,
    combine_stats,
    merge_accumulators,
    regularize_batch,
)


REPO = Path(__file__).resolve().parents[1]
PY = REPO / ".venv/bin/python"


def ts() -> str:
    return time.strftime("%H:%M:%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QPCA-only sweep on compact9 Qwen3 (or any compatible run).")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/calibration"))
    parser.add_argument("--run-id", default="longbench_compact9_qkv_qwen3_8b")
    parser.add_argument("--k-bits", default="2,3,4")
    parser.add_argument("--gpus", default=None,
                        help="Comma-separated GPU ids for the launcher to spawn shards on. "
                             "Default = visible GPUs from CUDA_VISIBLE_DEVICES, or '0' if unset.")
    parser.add_argument("--max-eval-examples", type=int, default=0,
                        help="Cap eval examples (0 = all 90). Smaller is faster but noisier.")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("artifacts/calibration/longbench_compact9_qkv_qwen3_8b/05_reports/pooled_n50_qpca"))
    parser.add_argument("--num-shards", type=int, default=0, help="Internal: shard mode if >0.")
    parser.add_argument("--shard-id", type=int, default=-1, help="Internal: 0-based shard index.")
    parser.add_argument("--eps", type=float, default=1e-4, help="Σ_Q regularization (trace * eps / d).")
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


def qpca_basis(
    sigma_q: torch.Tensor,
    sigma_k: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-(L, H) QPCA forward/inverse/eigvals.

    Eigendecomps in fp64 for numerical safety on rank-deficient Σ_Q (e.g. layer 0 sinks),
    cast back to fp32 at the end.

    Returns:
        forward: (L, H, d, d) — applied as `k @ forward` on row-vector keys
        inverse: (L, H, d, d) — applied as `r @ inverse` to reconstruct rows
        eigvals: (L, H, d)    — code-space per-coord variance Λ (descending)
    """
    # Inline symmetrize preserving dtype (analyze_bases._sym hard-casts to fp32).
    def _sym64(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (x + x.transpose(-1, -2))

    sq = regularize_batch(sigma_q, eps).to(torch.float64)
    sk = regularize_batch(sigma_k, eps).to(torch.float64)
    # M_q = sq; compute M_q^{1/2} and M_q^{-1/2} via eigendecomp.
    q_vals, q_vecs = torch.linalg.eigh(sq)
    q_vals = q_vals.clamp_min(float(eps))  # safety; regularize_batch already lifted these
    Mq_half = q_vecs @ torch.diag_embed(q_vals.sqrt()) @ q_vecs.transpose(-1, -2)
    Mq_neg_half = q_vecs @ torch.diag_embed(q_vals.rsqrt()) @ q_vecs.transpose(-1, -2)
    Mq_half = _sym64(Mq_half)
    Mq_neg_half = _sym64(Mq_neg_half)
    # A = M_q^{1/2} Σ_K M_q^{1/2}
    A = Mq_half @ sk @ Mq_half
    A = _sym64(A)
    lam, V = torch.linalg.eigh(A)
    # Sort descending
    order = torch.argsort(lam, dim=-1, descending=True)
    lam = torch.gather(lam, -1, order)
    V = torch.gather(V, -1, order.unsqueeze(-2).expand(*V.shape[:-1], -1))
    # Encoder for "k @ forward" convention: r^T = V^T M_q^{1/2} k^T  ⇒  r = k M_q^{1/2} V
    forward = Mq_half @ V
    # Decoder for "r @ inverse" convention: k^T = M_q^{-1/2} V r^T  ⇒  k = r V^T M_q^{-1/2}
    inverse = V.transpose(-1, -2) @ Mq_neg_half
    return forward.float(), inverse.float(), lam.float().clamp_min(1e-30)


def correctness_gates(
    forward: torch.Tensor,
    inverse: torch.Tensor,
    eigvals: torch.Tensor,
    paths: RunPaths,
    per_example: list[dict[str, Any]],
    test_indices: list[int],
    head_dim: int,
    device: torch.device,
) -> bool:
    """Run-time math sanity gates. Print results and return True if all pass."""
    print(f"[{ts()}] === correctness gates ===", flush=True)

    # Gate 1: forward @ inverse ≈ I, per (L, H)
    I_eye = torch.eye(head_dim, dtype=torch.float32)
    fi = forward @ inverse  # (L, H, d, d)
    err = (fi - I_eye.expand_as(fi)).reshape(-1, head_dim, head_dim)
    fro = err.norm(dim=(-2, -1))  # (L*H,)
    max_fro = float(fro.max().item())
    print(f"[{ts()}]   gate 1 (F@G - I, Frobenius max over (L,H)): {max_fro:.3e}",
          "OK" if max_fro < 1e-3 else "FAIL", flush=True)

    # Gate 2: on a small test row, code variance ≈ Λ (per (L, H)) on the first (L=1, H=0).
    first_idx = test_indices[0]
    meta = per_example[first_idx]
    raw_path = paths.root / meta["raw_file"]
    if not raw_path.exists():
        print(f"[{ts()}]   gate 2: SKIP — no raw for idx={first_idx} ({raw_path})", flush=True)
        gate2_pass = True
    else:
        raw = torch.load(raw_path, map_location="cpu", weights_only=False)
        k_all = raw["k_post"]  # (L, H, T, d)
        tokens = int(raw["prompt_length"])
        # Pick layer 1 (skip layer 0 sinks), head 0 for the variance check.
        L_test, H_test = 1, 0
        k = k_all[L_test, H_test, :tokens, :].to(device).float()  # (T, d)
        r = k @ forward[L_test, H_test].to(device)               # (T, d)
        emp_var = r.var(dim=0, unbiased=False).cpu()             # (d,)
        theo_var = eigvals[L_test, H_test]                        # (d,)
        # Per-token Λ ≈ (1/N) * Σ k^T A k, where r = M_q^{1/2} V k_row. The full-rank
        # population variance E[r r^T] = Λ ON THE TRAIN CORPUS, not on a single test row.
        # So we just sanity-check order-of-magnitude alignment and monotone decrease.
        emp_monotone_frac = float((emp_var[:-1] >= emp_var[1:]).float().mean().item())
        theo_monotone_frac = float((theo_var[:-1] >= theo_var[1:]).float().mean().item())
        ratio_log = (emp_var.clamp_min(1e-30).log() - theo_var.clamp_min(1e-30).log()).abs().median()
        print(f"[{ts()}]   gate 2 (code variance vs Λ on row {first_idx}, L={L_test} H={H_test}):",
              f"empirical sorted-monotone-frac={emp_monotone_frac:.2f}",
              f"theoretical={theo_monotone_frac:.2f}",
              f"|log-ratio| median={ratio_log:.2f}",
              flush=True)
        gate2_pass = ratio_log < 3.0  # within ~e^3 = 20x — generous for single-row sample variance vs train-pooled Λ
        del k_all, raw

    # Gate 3: b=8 roundtrip near-exact on the same row.
    if raw_path.exists():
        bits8 = torch.full((head_dim,), 8, dtype=torch.long)
        comp = PerCoordCompressor(
            bits_per_coord=bits8,
            std_per_coord=eigvals[L_test, H_test].sqrt(),
            forward_map=forward[L_test, H_test],
            inverse_map=inverse[L_test, H_test],
        ).to(device)
        recon = comp.roundtrip(k).float()
        rel_mse = float(((k - recon).square().sum() / k.square().sum().clamp_min(1e-30)).item())
        print(f"[{ts()}]   gate 3 (b=8 relative MSE on L={L_test} H={H_test}): {rel_mse:.3e}",
              "OK" if rel_mse < 1e-2 else "FAIL", flush=True)
        gate3_pass = rel_mse < 1e-2
        del k, r, comp, recon
    else:
        gate3_pass = True

    return (max_fro < 1e-3) and gate2_pass and gate3_pass


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

    print(f"[{ts()}] building QPCA basis (fp64 eigendecomps in M_q^±1/2 and A)...", flush=True)
    forward, inverse, eigvals = qpca_basis(train_stats["sigma_q"], train_stats["sigma_k"], args.eps)
    print(f"[{ts()}] QPCA shapes: forward={tuple(forward.shape)} inverse={tuple(inverse.shape)} "
          f"eigvals={tuple(eigvals.shape)}", flush=True)

    return {
        "paths": paths,
        "per_example": per_example,
        "test_indices": test_indices,
        "forward": forward,
        "inverse": inverse,
        "eigvals": eigvals,
        "head_dim": head_dim,
        "n_layers_total": n_layers_total,
        "n_kv_heads_total": n_kv_heads_total,
    }


def run_shard(args: argparse.Namespace) -> None:
    """Single-shard worker: build QPCA, compress, score on this shard's slice of test idx."""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"shard_{args.shard_id:03d}.json"
    if out_path.exists():
        print(f"[{ts()}] shard {args.shard_id}: output already exists, skipping ({out_path})", flush=True)
        return

    k_bits = [int(b) for b in args.k_bits.split(",") if b.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{ts()}] shard {args.shard_id}/{args.num_shards} on {device} "
          f"(visible: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')})", flush=True)

    inputs = load_inputs(args, device, k_bits)
    forward = inputs["forward"]
    inverse = inputs["inverse"]
    eigvals = inputs["eigvals"]
    head_dim = inputs["head_dim"]
    n_layers_total = inputs["n_layers_total"]
    n_kv_heads_total = inputs["n_kv_heads_total"]

    eval_subset_full = inputs["test_indices"]
    if args.max_eval_examples > 0:
        eval_subset_full = eval_subset_full[: args.max_eval_examples]
    eval_slice = slice_for_shard(eval_subset_full, args.num_shards, args.shard_id)
    print(f"[{ts()}] shard {args.shard_id}: eval slice = {len(eval_slice)} idx", flush=True)

    # Run the math gates ONCE on this shard.
    if args.shard_id == 0:
        gates_ok = correctness_gates(
            forward.cpu(), inverse.cpu(), eigvals.cpu(),
            inputs["paths"], inputs["per_example"], eval_subset_full, head_dim, device,
        )
        if not gates_ok:
            print(f"[{ts()}] shard 0: correctness gates FAILED — aborting", flush=True)
            sys.exit(1)

    # Bit allocations per (bits): water-fill on Λ. allocate_bits already mirrors production caps.
    bit_allocs: dict[int, torch.Tensor] = {}
    for bits in k_bits:
        bit_allocs[bits] = allocate_bits(eigvals, bits, max_coord_bits=int(args.max_coord_bits)).cpu()

    # Pre-build PerCoordCompressors per (bits, L, H) once. Lloyd-Max codebook solve is bits-dep,
    # not row-dep — building per row is wasteful.
    layer_range_full = range(0, n_layers_total)
    head_range = range(0, n_kv_heads_total)
    layer_head_pairs = [(l, h) for l in layer_range_full for h in head_range]

    forward_cpu = forward.cpu()
    inverse_cpu = inverse.cpu()
    std_cpu = eigvals.sqrt().cpu()

    print(f"[{ts()}] building compressors for {len(layer_head_pairs)} (L,H) × {len(k_bits)} bits...", flush=True)
    comps_by_bits: dict[int, dict[tuple[int, int], PerCoordCompressor]] = {}
    for bits in k_bits:
        alloc = bit_allocs[bits]
        comps_by_bits[bits] = {
            (l, h): PerCoordCompressor(
                bits_per_coord=alloc[l, h],
                std_per_coord=std_cpu[l, h],
                forward_map=forward_cpu[l, h],
                inverse_map=inverse_cpu[l, h],
            ).to(device)
            for (l, h) in layer_head_pairs
        }
    print(f"[{ts()}] compressors ready", flush=True)

    accums: dict[int, dict[int, dict[str, float]]] = {
        bits: {
            l: {"mse_num": 0.0, "mse_den": 0, "logit_num": 0.0, "logit_den": 0,
                "top1_num": 0, "top1_den": 0, "top5_num": 0, "top5_den": 0}
            for l in layer_range_full
        }
        for bits in k_bits
    }

    raw_lru = RawLRU(max_entries=1)
    t_start = time.time()
    for n_idx, idx in enumerate(eval_slice):
        t_idx = time.time()
        meta = inputs["per_example"][idx]
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
                # Top-1 / top-5 attention retention, chunked along queries to bound memory.
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
        print(f"[{ts()}] shard {args.shard_id} idx {n_idx+1}/{len(eval_slice)} (#{idx}) "
              f"done in {time.time()-t_idx:.1f}s", flush=True)

    payload: dict[str, Any] = {
        "shard_id": int(args.shard_id),
        "num_shards": int(args.num_shards),
        "eval_slice": eval_slice,
        "k_bits": k_bits,
        "head_dim": int(head_dim),
        "n_layers_total": int(n_layers_total),
        "n_kv_heads_total": int(n_kv_heads_total),
        "accumulators": {
            str(int(b)): {str(int(l)): dict(stats) for l, stats in per_layer.items()}
            for b, per_layer in accums.items()
        },
    }
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload) + "\n")
    tmp.replace(out_path)
    print(f"[{ts()}] shard {args.shard_id}: wrote {out_path} (elapsed {time.time()-t_start:.0f}s)", flush=True)


def normalize_accumulators(per_bits: dict[str, Any]) -> dict[int, dict[int, dict[str, float]]]:
    return {int(b): {int(l): dict(stats) for l, stats in per_layer.items()}
            for b, per_layer in per_bits.items()}


def run_launcher(args: argparse.Namespace) -> None:
    """Spawn one shard subprocess per visible GPU, then merge accumulator JSONs."""
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
    layer0_modes = _ensure_modes(None)  # [True, False]

    shard_files = sorted(args.out_dir.glob("shard_*.json"))
    if len(shard_files) != num_shards:
        print(f"[{ts()}] launcher: expected {num_shards} shard files, found {len(shard_files)}", flush=True)
        sys.exit(1)
    payloads = [json.loads(f.read_text()) for f in shard_files]
    n_layers_total = payloads[0]["n_layers_total"]
    layer_range_full = range(0, n_layers_total)

    shards_normalized = [normalize_accumulators(p["accumulators"]) for p in payloads]
    merged_accums = merge_accumulators(shards_normalized)
    finalized = _finalize_k_accums(merged_accums, layer0_modes, k_bits, layer_range_full)
    # Schema-shape to match merged.json["method"]: {"layer0=False": {...}, "layer0=True": {...}}
    final: dict[str, dict[str, float]] = {f"layer0={m}": v for m, v in finalized.items()}

    out_path = args.out_dir / "qpca_merged.json"
    out_path.write_text(json.dumps(final, indent=2, sort_keys=True, default=str) + "\n")
    print(f"[{ts()}] launcher: wrote {out_path}", flush=True)

    # Side-by-side comparison vs the existing pooled_n50/merged.json.
    ref_path = REPO / "artifacts/calibration/longbench_compact9_qkv_qwen3_8b/05_reports/pooled_n50/merged.json"
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())
        print(f"\n[{ts()}] === COMPARISON vs pooled_n50/merged.json (layer-0 excluded, pooled N=50) ===", flush=True)
        hdr = f"{'method':<10} | b | {'top1':>7} | {'top5':>7} | {'k_mse':>10} | {'logit_err':>10}"
        print(hdr, flush=True)
        print("-" * len(hdr), flush=True)
        rows = [
            ("v3", ref["v3"]["layer0=False"], "_k"),
            ("jointqk", ref["jointqk"]["layer0=False"], "_waterfill_k"),
            ("qpca", final["layer0=False"], "_waterfill_k"),
        ]
        for label, d, suf in rows:
            for b in k_bits:
                print(f"{label:<10} | {b} | "
                      f"{d.get(f'empirical_top1{suf}{b}', float('nan')):>7.4f} | "
                      f"{d.get(f'empirical_top5{suf}{b}', float('nan')):>7.4f} | "
                      f"{d.get(f'empirical_k_mse{suf}{b}', float('nan')):>10.3e} | "
                      f"{d.get(f'empirical_logit_error{suf}{b}', float('nan')):>10.3e}",
                      flush=True)


def main() -> None:
    args = parse_args()
    if args.num_shards > 0 and args.shard_id >= 0:
        run_shard(args)
    else:
        run_launcher(args)


if __name__ == "__main__":
    main()
