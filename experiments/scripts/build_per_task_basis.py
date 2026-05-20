#!/usr/bin/env python3
"""Build per-task cca_stats.pt artifacts from each task's 50 train examples.

Tests hypothesis D3: pooled-over-8-tasks calibration may be suboptimal vs
task-matched basis. Per-task calibration uses only the 50 train examples
from a single task, so the basis is fitted to that task's Q/K distribution.

Outputs:
  artifacts/bases/per_task/cca_stats_<task>.pt
  artifacts/v_bases/per_task/v_stats_<task>.pt
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.calibration.analyze_bases import combine_stats, jointqk_basis
from experiments.calibration.common import RunPaths


def ts() -> str:
    return time.strftime("%H:%M:%S")


def build_for_task(paths, per_example, task: str, out_cca: Path, out_v: Path) -> None:
    indices = [int(p["index"]) for p in per_example
               if p["split"] == "train" and p["config"] == task]
    if not indices:
        raise ValueError(f"no train examples for task={task}")
    print(f"[{ts()}] {task}: pooling over {len(indices)} train examples", flush=True)

    train_stats = combine_stats(paths, per_example, indices, torch.device("cpu"))
    sigma_q = train_stats["sigma_q"].float()
    sigma_k = train_stats["sigma_k"].float()
    cov_v = train_stats["cov_v"].float()
    mu_v = train_stats["mu_v"].float()
    n_layers, n_kv_heads, head_dim, _ = sigma_q.shape
    total_tokens = int(train_stats["tokens"])

    print(f"[{ts()}] {task}: tokens={total_tokens}; computing R_sym...", flush=True)
    r_sym = jointqk_basis(sigma_q, sigma_k, eps=1e-4).float()

    eye = torch.eye(head_dim, dtype=torch.float32).expand(n_layers, n_kv_heads, head_dim, head_dim).contiguous()
    zeros_diag = torch.zeros(n_layers, n_kv_heads, head_dim, dtype=torch.float32)
    cca_artifact = {
        "sigma_q": sigma_q, "sigma_k": sigma_k, "R_sym": r_sym,
        "cqk": torch.zeros_like(sigma_q), "rho": zeros_diag,
        "P_K": eye, "P_K_inv": eye, "P_Q": eye,
        "mq_eigvals": zeros_diag, "mq_eigvecs": eye, "V_h": eye,
        "n_layers": n_layers, "n_kv_heads": n_kv_heads, "head_dim": head_dim,
        "total_prefill_tokens": total_tokens,
        "calibration_source": f"longbench_compact8_qkv per-task ({task}, {len(indices)} train prompts)",
        "calibration_date": "2026-05-05",
    }
    sigma_v = cov_v + torch.einsum("lhd,lhe->lhde", mu_v, mu_v)

    v_artifact = {
        "cov_v": cov_v, "mu_v": mu_v, "sigma_v": sigma_v.float(),
        "metadata": {
            "n_layers": n_layers, "n_kv_heads": n_kv_heads, "head_dim": head_dim,
            "n_examples": len(indices), "total_token_count": total_tokens,
            "prefill_only": True,
            "bundle": f"longbench_compact8_qkv per-task ({task})",
            "version": 2,
        },
    }

    out_cca.parent.mkdir(parents=True, exist_ok=True)
    out_v.parent.mkdir(parents=True, exist_ok=True)
    tmp_c = out_cca.with_suffix(out_cca.suffix + ".tmp")
    torch.save(cca_artifact, tmp_c)
    tmp_c.replace(out_cca)
    tmp_v = out_v.with_suffix(out_v.suffix + ".tmp")
    torch.save(v_artifact, tmp_v)
    tmp_v.replace(out_v)
    print(f"[{ts()}] {task}: wrote {out_cca.name}, {out_v.name}", flush=True)


def main() -> None:
    artifact_root = REPO / "artifacts/calibration"
    paths = RunPaths.from_args(artifact_root, "longbench_compact8_qkv")
    agg = torch.load(paths.stats_dir / "aggregate.pt", map_location="cpu", weights_only=False)
    per_example = agg["per_example"]

    tasks = ["hotpotqa", "musique", "qasper", "qmsum"]
    for task in tasks:
        out_cca = REPO / f"artifacts/bases/per_task/cca_stats_{task}.pt"
        out_v = REPO / f"artifacts/v_bases/per_task/v_stats_{task}.pt"
        build_for_task(paths, per_example, task, out_cca, out_v)


if __name__ == "__main__":
    main()
