#!/usr/bin/env python3
"""Build jointqk.pt + v_stats.pt artifacts from the new pooled calibration stats.

Source: `combine_stats(train_indices)` over the train split of the calibration
run (typically 400 LongBench-compact8 train prompts).

For the K side we only need `sigma_q`, `sigma_k`, and `R_sym` (the joint-Q-K
eigenbasis) to drive the `r_sym_waterfill` k_method. The press constructor still
reads CCA-only fields (`P_K`, `P_K_inv`, `rho`, `cqk`, `P_Q`, `V_h`, `mq_*`) so
we provide identity / zero placeholders — they're unused by `r_sym_waterfill`.

For the V side we need `cov_v` and `mu_v` (centered second-moment + per-coord
mean), which are already in the train_stats payload.

Usage (Qwen3-8B, default):
    .venv/bin/python pipelines/scripts/build_calibration_artifacts_from_pool.py

Usage (Llama-3.1-8B):
    .venv/bin/python pipelines/scripts/build_calibration_artifacts_from_pool.py \
        --run-id longbench_compact8_qkv_llama31_8b \
        --output-suffix llama31_8b_n400
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _bootstrap  # noqa: E402, F401

import argparse
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pipelines.calibration.analyze_bases import (
    combine_stats,
    jointqk_basis,
)
from pipelines.calibration.common import RunPaths


def ts() -> str:
    return time.strftime("%H:%M:%S")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-root", type=Path, default=Path("artifacts/calibration"))
    p.add_argument("--run-id", default="longbench_compact8_qkv_qwen3_8b",
                   help="Run id of the calibration artifacts (the 02_stats/aggregate.pt source).")
    p.add_argument("--output-suffix", default="longbench_compact8_n400",
                   help="Suffix attached to the output cca_stats / v_stats filenames. "
                        "Default 'longbench_compact8_n400' produces "
                        "artifacts/bases/jointqk_longbench_compact8_n400.pt etc.")
    p.add_argument("--cca-out-dir", type=Path,
                   default=Path("artifacts/bases"))
    p.add_argument("--v-out-dir", type=Path,
                   default=Path("artifacts/v_bases"))
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing cca_stats / v_stats files at the target paths. "
                        "Default is to refuse — protects against accidentally clobbering a "
                        "different model's calibration when --output-suffix is wrong.")
    p.add_argument("--filter-config", default=None,
                   help="If set, restrict the pooled train indices to rows whose per_example config "
                        "matches this value (e.g. 'lcc' to build a single-task basis from a multi-task "
                        "calibration run). Default: include all train rows.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    artifact_root = (REPO / args.artifact_root) if not args.artifact_root.is_absolute() else args.artifact_root
    paths = RunPaths.from_args(artifact_root, args.run_id)

    print(f"[{ts()}] loading aggregate from {paths.stats_dir / 'aggregate.pt'}", flush=True)
    agg = torch.load(paths.stats_dir / "aggregate.pt", map_location="cpu", weights_only=False)
    per_example = agg["per_example"]
    train_indices = [int(p["index"]) for p in per_example if p["split"] == "train"]
    if args.filter_config:
        before = len(train_indices)
        train_indices = [int(p["index"]) for p in per_example
                         if p["split"] == "train" and p.get("config") == args.filter_config]
        print(f"[{ts()}] filter-config={args.filter_config!r}: {before} → {len(train_indices)} train examples", flush=True)
        if not train_indices:
            raise RuntimeError(f"No train rows matched filter-config={args.filter_config!r}")
    print(f"[{ts()}] pooling stats over {len(train_indices)} train examples", flush=True)

    train_stats = combine_stats(paths, per_example, train_indices, torch.device("cpu"))
    sigma_q = train_stats["sigma_q"].float()
    sigma_k = train_stats["sigma_k"].float()
    cov_v = train_stats["cov_v"].float()
    mu_v = train_stats["mu_v"].float()
    sigma_v = train_stats["sigma_v"] if "sigma_v" in train_stats else (cov_v + torch.einsum("lhd,lhe->lhde", mu_v, mu_v))
    sigma_v = sigma_v.float()
    n_layers, n_kv_heads, head_dim, _ = sigma_q.shape
    total_tokens = int(train_stats["tokens"])
    print(f"[{ts()}] shapes: sigma_q={tuple(sigma_q.shape)} cov_v={tuple(cov_v.shape)} tokens={total_tokens}", flush=True)

    print(f"[{ts()}] computing R_sym (joint Q-K eigenbasis) per (layer, kv_head)...", flush=True)
    r_sym = jointqk_basis(sigma_q, sigma_k, eps=float(args.eps)).float()

    # CCA-only placeholders. r_sym_waterfill never reads these, but the press
    # constructor accesses them as `cca["sigma_q"]` etc. on every (L, h) build.
    eye = torch.eye(head_dim, dtype=torch.float32).expand(n_layers, n_kv_heads, head_dim, head_dim).contiguous()
    zeros_diag = torch.zeros(n_layers, n_kv_heads, head_dim, dtype=torch.float32)

    cca_artifact = {
        "sigma_q": sigma_q,
        "sigma_k": sigma_k,
        "R_sym": r_sym,
        # Placeholders — required by press constructor signature, unused by r_sym_*
        "cqk": torch.zeros_like(sigma_q),
        "rho": zeros_diag,
        "P_K": eye,
        "P_K_inv": eye,
        "P_Q": eye,
        "mq_eigvals": zeros_diag,
        "mq_eigvecs": eye,
        "V_h": eye,
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "total_prefill_tokens": total_tokens,
        "calibration_source": f"{args.run_id} pooled ({len(train_indices)} train prompts)",
        "calibration_date": time.strftime("%Y-%m-%d"),
    }

    cca_out_dir = (REPO / args.cca_out_dir) if not args.cca_out_dir.is_absolute() else args.cca_out_dir
    v_out_dir = (REPO / args.v_out_dir) if not args.v_out_dir.is_absolute() else args.v_out_dir
    cca_out_dir.mkdir(parents=True, exist_ok=True)
    v_out_dir.mkdir(parents=True, exist_ok=True)
    cca_out = cca_out_dir / f"jointqk_{args.output_suffix}.pt"
    v_out = v_out_dir / f"v_stats_{args.output_suffix}.pt"
    if not args.force:
        for p in (cca_out, v_out):
            if p.exists():
                raise RuntimeError(
                    f"refusing to overwrite existing {p} (use --force to override). "
                    f"Pass a unique --output-suffix for each model to avoid this."
                )
    cca_tmp = cca_out.with_suffix(cca_out.suffix + ".tmp")
    torch.save(cca_artifact, cca_tmp)
    cca_tmp.replace(cca_out)
    size_mb = cca_out.stat().st_size / (1024 * 1024)
    print(f"[{ts()}] wrote {cca_out} ({size_mb:.0f} MB)", flush=True)

    v_artifact = {
        "cov_v": cov_v,
        "mu_v": mu_v,
        "sigma_v": sigma_v,  # legacy uncentered field for v1 readers
        "metadata": {
            "n_layers": n_layers,
            "n_kv_heads": n_kv_heads,
            "head_dim": head_dim,
            "n_examples": len(train_indices),
            "total_token_count": total_tokens,
            "prefill_only": True,
            "bundle": f"artifacts/calibration/{args.run_id} pooled train",
            "version": 2,
        },
    }
    v_tmp = v_out.with_suffix(v_out.suffix + ".tmp")
    torch.save(v_artifact, v_tmp)
    v_tmp.replace(v_out)
    v_size_mb = v_out.stat().st_size / (1024 * 1024)
    print(f"[{ts()}] wrote {v_out} ({v_size_mb:.0f} MB)", flush=True)

    print(f"\n[{ts()}] done. Use these artifacts in your v7 launcher by setting:")
    print(f"  CCA={cca_out}")
    print(f"  VST={v_out}")


if __name__ == "__main__":
    main()
