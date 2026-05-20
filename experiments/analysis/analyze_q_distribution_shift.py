#!/usr/bin/env python3
"""Compute per-task Σ_Q drift vs the compact8 calibration reference, for 8 tasks.

Two sources of windows:
  - Prefill: 200-token contiguous windows along the seq axis of each test prompt's
    captured q_post tensor.
  - Decode: bin decode-Q samples (captured via model.generate hook) by step range
    [1-50], [51-100], [101-200], [201-500]; aggregate across prompts within each bin.

Metric (primary): top-16 subspace cosine between window Σ_Q and reference Σ_Q,
averaged across (L≥1, h). Reference = pooled compact8 Σ_Q from the existing
`cca_stats_llama31_8b_longbench_compact8_n400.pt` artifact.

Outputs:
  artifacts/q_distribution_shift/per_task_drift.json
  notes/figs/q_drift/prefill_drift.png
  notes/figs/q_drift/decode_drift.png
  notes/figs/q_drift/combined_trajectory.png
  notes/figs/q_drift/summary_bars.png
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterator

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

TOP_K = 16
WINDOW_SIZE = 200
DECODE_BINS = [(1, 50), (51, 100), (101, 200), (201, 500)]
LAYER0_EXCLUDED = True

_FNAME_RE = re.compile(r"^longbench__(?P<task>[A-Za-z0-9_\-]+)__row(?P<row>\d+)__(?P<split>train|test)\.pt$")


def ts() -> str:
    return time.strftime("%H:%M:%S")


_REF_EIGVECS_CACHE: dict = {}


def get_ref_eigvecs(sigma_ref: torch.Tensor, k: int, device) -> torch.Tensor:
    """Compute top-k eigenvectors of reference Σ_Q once per (L, h), cache it."""
    cache_key = (id(sigma_ref), k, str(device))
    if cache_key in _REF_EIGVECS_CACHE:
        return _REF_EIGVECS_CACHE[cache_key]
    n_layers, n_kv_heads, d, _ = sigma_ref.shape
    if LAYER0_EXCLUDED:
        flat = sigma_ref[1:].reshape(-1, d, d).to(device)
    else:
        flat = sigma_ref.reshape(-1, d, d).to(device)
    _, vecs = torch.linalg.eigh(flat)  # (N, d, d), ascending
    v_r = vecs[..., -k:].contiguous()  # (N, d, k)
    _REF_EIGVECS_CACHE[cache_key] = v_r
    return v_r


def top_k_subspace_cosine(sigma_window: torch.Tensor, sigma_ref: torch.Tensor,
                          k: int = TOP_K, device=None) -> float:
    """Per-(L,h) top-k subspace overlap, batched over all (L≥1, h)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_layers, n_kv_heads, d, _ = sigma_window.shape
    if LAYER0_EXCLUDED:
        flat_w = sigma_window[1:].reshape(-1, d, d).to(device)
    else:
        flat_w = sigma_window.reshape(-1, d, d).to(device)
    _, vecs_w = torch.linalg.eigh(flat_w)
    v_w = vecs_w[..., -k:]                    # (N, d, k)
    v_r = get_ref_eigvecs(sigma_ref, k, device)  # (N, d, k)
    # principal-angle Frobenius: ||V_w^T V_r||_F / sqrt(k) per (L, h)
    inner = torch.bmm(v_w.transpose(-1, -2), v_r)  # (N, k, k)
    overlap_per_pair = inner.pow(2).sum(dim=(-1, -2)).sqrt() / (k ** 0.5)
    return float(overlap_per_pair.mean().item())


def compute_sigma_q_from_q(q: torch.Tensor, group: int) -> torch.Tensor:
    """q: (n_layers, n_q_heads, T, d). Returns Σ_Q: (n_layers, n_kv_heads, d, d), pooled over the GQA group."""
    n_layers, n_q_heads, T, d = q.shape
    n_kv_heads = n_q_heads // group
    qf = q.float()
    sq = torch.einsum("lhsd,lhse->lhde", qf, qf)
    sq = sq.view(n_layers, n_kv_heads, group, d, d).sum(dim=2) / (group * T)
    return sq


def iter_prefill_windows(q: torch.Tensor, window: int = WINDOW_SIZE) -> Iterator[tuple[int, torch.Tensor]]:
    """Yield (center_position, window_q) for contiguous 200-token windows."""
    T = q.shape[2]
    for start in range(0, T - window + 1, window):
        center = start + window // 2
        yield center, q[:, :, start:start + window, :]


def load_reference_sigma_q(cca_path: Path) -> torch.Tensor:
    """Load the pooled compact8 Σ_Q from the production basis artifact."""
    cca = torch.load(cca_path, map_location="cpu", weights_only=False)
    return cca["sigma_q"].float()


def find_test_prompts(raw_root: Path, task: str, n_prompts: int = 5) -> list[Path]:
    """Find up to n_prompts test raw files for `task` in raw_root."""
    matches = []
    for shard in sorted(raw_root.iterdir()):
        if not shard.is_dir():
            continue
        for f in sorted(shard.iterdir()):
            m = _FNAME_RE.match(f.name)
            if not m or m.group("split") != "test":
                continue
            if m.group("task") == task:
                matches.append(f)
            if len(matches) >= n_prompts:
                return matches
    return matches


def analyze_prefill_drift(task: str, raw_files: list[Path], sigma_ref: torch.Tensor,
                          n_kv_heads: int, group: int) -> list[dict]:
    """For each prompt: window the prefill, compute Σ_Q per window, cosine vs ref. Average over prompts per position bin."""
    print(f"[{ts()}] [{task}] prefill: {len(raw_files)} prompts", flush=True)
    # position_bins keyed by quantized window center (50-token resolution for plotting)
    points = []
    for fi, f in enumerate(raw_files):
        raw = torch.load(f, map_location="cpu", weights_only=False)
        q = raw["q_post"][:, :, :int(raw["prompt_length"]), :]
        for center, qwin in iter_prefill_windows(q, WINDOW_SIZE):
            sigma_w = compute_sigma_q_from_q(qwin, group)
            cos = top_k_subspace_cosine(sigma_w, sigma_ref, TOP_K)
            points.append({"prompt_idx": fi, "center": int(center), "cos": cos,
                           "T_prompt": int(q.shape[2])})
        del raw, q
    print(f"[{ts()}] [{task}] prefill: {len(points)} window evaluations", flush=True)
    return points


def aggregate_prefill_bins(points: list[dict], bin_size: int = 400) -> list[dict]:
    """Aggregate per-window points into position bins (default 400 tokens) with mean ± std cosine."""
    if not points:
        return []
    by_bin = {}
    for p in points:
        b = (p["center"] // bin_size) * bin_size + bin_size // 2
        by_bin.setdefault(b, []).append(p["cos"])
    out = []
    for b in sorted(by_bin):
        vals = by_bin[b]
        if not vals: continue
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        out.append({"position": b, "mean_cos": mean, "std_cos": std, "n": len(vals)})
    return out


def aggregate_decode_bins(decode_q_dir: Path, task: str, sigma_ref: torch.Tensor,
                          n_kv_heads: int, group: int) -> list[dict]:
    """Load decode-Q per-prompt tensors and compute Σ_Q per step-bin."""
    task_dir = decode_q_dir / task
    if not task_dir.exists():
        return []
    # decode-Q file format: dict with key "decode_q" shape (n_layers, n_q_heads, n_steps, d), plus "n_steps".
    bin_results = []
    for lo, hi in DECODE_BINS:
        bin_q_chunks = []
        for f in sorted(task_dir.glob("*.pt")):
            payload = torch.load(f, map_location="cpu", weights_only=False)
            dq = payload["decode_q"]  # (n_layers, n_q_heads, n_steps, d)
            n_steps = dq.shape[2]
            # Take steps in [lo-1, hi) i.e. step index lo-1..hi-1
            start = lo - 1
            stop = min(hi, n_steps)
            if start >= stop:
                continue
            bin_q_chunks.append(dq[:, :, start:stop, :])
        if not bin_q_chunks:
            continue
        # concat along step dim
        q_bin = torch.cat(bin_q_chunks, dim=2).float()
        if q_bin.shape[2] < 10:
            continue
        sigma_w = compute_sigma_q_from_q(q_bin, group)
        cos = top_k_subspace_cosine(sigma_w, sigma_ref, TOP_K)
        bin_results.append({"bin": f"[{lo}-{hi}]", "lo": lo, "hi": hi,
                            "n_samples": int(q_bin.shape[2]), "mean_cos": cos})
    return bin_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-cca",
        default=str(REPO / "artifacts/bases/cca_stats_llama31_8b_longbench_compact8_n400.pt"))
    parser.add_argument("--compact8-raw",
        default=str(REPO / "artifacts/calibration/longbench_compact8_qkv_llama31_8b/01_raw"))
    parser.add_argument("--compact9-raw",
        default=str(REPO / "artifacts/calibration/longbench_compact9_qkv_llama31_8b/01_raw"))
    parser.add_argument("--extra-raw-dirs", default="",
        help="Comma-separated list of additional raw dirs to search (e.g. 2wikimqa standalone capture).")
    parser.add_argument("--decode-q-dir",
        default=str(REPO / "artifacts/decode_q_captures_llama"))
    parser.add_argument("--tasks", default="hotpotqa,qasper,qmsum,repobench-p,musique,multi_news,lcc,2wikimqa")
    parser.add_argument("--n-prefill-prompts", type=int, default=5)
    parser.add_argument("--out", default=str(REPO / "artifacts/q_distribution_shift/per_task_drift.json"))
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_search_dirs = [Path(args.compact8_raw), Path(args.compact9_raw)]
    for extra in args.extra_raw_dirs.split(","):
        extra = extra.strip()
        if extra:
            raw_search_dirs.append(Path(extra))

    print(f"[{ts()}] loading reference Σ_Q from {args.reference_cca}", flush=True)
    sigma_ref = load_reference_sigma_q(Path(args.reference_cca))
    n_layers, n_kv_heads, d, _ = sigma_ref.shape
    # group is hardcoded for Llama-3.1-8B (32 q-heads / 8 kv-heads = 4)
    group = 32 // n_kv_heads
    print(f"[{ts()}] ref Σ_Q shape={tuple(sigma_ref.shape)}, group={group}", flush=True)

    results = {"tasks": {}, "reference_cca": args.reference_cca,
               "top_k": TOP_K, "window_size": WINDOW_SIZE,
               "decode_bins": DECODE_BINS, "n_layers": n_layers, "n_kv_heads": n_kv_heads}

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for task in tasks:
        # Find prefill captures across all known raw dirs
        prefill_files = []
        for d_root in raw_search_dirs:
            if not d_root.exists(): continue
            found = find_test_prompts(d_root, task, args.n_prefill_prompts)
            if found:
                prefill_files = found
                break
        if not prefill_files:
            print(f"[{ts()}] [{task}] WARN: no prefill captures found in any raw dir", flush=True)
            results["tasks"][task] = {"prefill_points": [], "prefill_binned": [],
                                       "decode_bins": []}
            continue

        prefill_points = analyze_prefill_drift(task, prefill_files, sigma_ref, n_kv_heads, group)
        prefill_binned = aggregate_prefill_bins(prefill_points, bin_size=400)
        decode_bins = aggregate_decode_bins(Path(args.decode_q_dir), task, sigma_ref, n_kv_heads, group)
        results["tasks"][task] = {
            "prefill_points": prefill_points,
            "prefill_binned": prefill_binned,
            "decode_bins": decode_bins,
        }
        print(f"[{ts()}] [{task}] prefill cosine summary: "
              f"mean={sum(p['cos'] for p in prefill_points)/max(1,len(prefill_points)):.4f}, "
              f"decode bins: {[(b['bin'], round(b['mean_cos'],3)) for b in decode_bins]}",
              flush=True)
        out_path.write_text(json.dumps(results, indent=2, default=str) + "\n")

    print(f"\n[{ts()}] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
