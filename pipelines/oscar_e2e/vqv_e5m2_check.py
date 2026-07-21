#!/usr/bin/env python3
"""Does the fp8-e5m2 centroid snap explain VQ-V's served deficit?

Served facts: VQ-V trails INT2-V by 1.0 (32K) / 2.8 (64K) NIAH points despite
Samuel's offline de-risk showing VQ-V at 0.49x the INT2-V reconstruction MSE.
His de-risk used fp16 centroids; the engine snaps them to e5m2 (the sm80
bitcast constraint). This reruns his comparison with the ENGINE's centroids:

  A  INT2-V: per-token affine-asymmetric 2-bit, percentile clip 0.92
     (torch replica of _launch_single_clip_int2 semantics)
  B  VQ-V, fp16 centroids  (his de-risk configuration)
  C  VQ-V, e5m2-snapped centroids, assignment against the snapped values
     (exactly what the engine encodes/decodes)

All on the same live-captured V (GPQA prompts through Qwen3-8B), rotated by
the same R_v the engine loads, strided-permuted like the engine. MSE in the
rotated basis == ambient (orthonormal). Layer 0 excluded from headline.

  PYTHONPATH=... .venv/bin/python pipelines/oscar_e2e/vqv_e5m2_check.py --gpu 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import _bootstrap  # noqa: E402,F401
from pipelines.calibration.capture_raw import run_prefill_qkv_capture  # noqa: E402
from kvq.capture.model import get_model_device, load_model_and_tokenizer  # noqa: E402

ROTZOO = REPO / ("artifacts/oscar_e2e/rotzoo/Qwen3-8B/"
                 "seq20000_prompt83_group128/v_rotation_sst_r_h_pbr.pt")
VQV = REPO / "third_party/samuel_vq/codebooks/vqv_G4_strided_gpqa_engine.pt"
ROWS = REPO / "artifacts/prompt_rows/gpqa_diamond_think_qwen.jsonl"


def int2v_affine(v: torch.Tensor, clip_ratio: float = 0.92) -> torch.Tensor:
    """Per-token affine-asymmetric INT2 with per-row percentile clip
    (replicates the engine's single-scale clip kernel semantics)."""
    D = v.shape[-1]
    idx = min(int(clip_ratio * D), D - 1)
    thr = v.abs().sort(dim=-1).values[..., idx : idx + 1]
    vc = v.clamp(-thr, thr)
    vmin = vc.amin(-1, keepdim=True)
    vmax = vc.amax(-1, keepdim=True)
    scale = (vmax - vmin).clamp_min(1e-8) / 3.0
    zero = -vmin / scale
    q = (vc / scale + zero + 0.5).floor().clamp(0, 3)
    return (q - zero) * scale


def vqv_recon(v_rot: torch.Tensor, cbs: list[torch.Tensor], perm, inv_perm,
              snap_e5m2: bool) -> torch.Tensor:
    """Strided-perm + per-token RMS norm + nearest-centroid + reconstruct."""
    r = v_rot[..., perm].float()
    scale = r.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
    rn = r / scale
    T, D = rn.shape
    G = cbs[0].shape[1]
    NG = D // G
    out = torch.empty_like(rn)
    for g in range(NG):
        cb = cbs[g].float()
        if snap_e5m2:
            cb = cb.to(torch.float8_e5m2).float()
        x = rn[:, g * G : (g + 1) * G]
        d = torch.cdist(x, cb)
        out[:, g * G : (g + 1) * G] = cb[d.argmin(1)]
    return (out * scale)[..., inv_perm]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--out", default="artifacts/oscar_e2e/lh/vqv_e5m2_check.json")
    args = ap.parse_args()

    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    model, tok = load_model_and_tokenizer(args.model, device_map="auto",
                                          dtype_name="float16")
    dev = get_model_device(model)

    rv = torch.load(ROTZOO, map_location="cpu", weights_only=False)["layers"]
    blob = torch.load(VQV, map_location="cpu", weights_only=False)
    bounds = blob["bounds"]
    G = bounds[0][1] - bounds[0][0]

    rows = [json.loads(l) for l in open(ROWS)][: args.n_prompts]
    per_layer = {}
    for row in rows:
        ids = tok(row["prompt"], return_tensors="pt",
                  truncation=True, max_length=3072).input_ids.to(dev)
        cap = run_prefill_qkv_capture(model, ids)
        for l, v in enumerate(cap["v"]):  # [H, T, D]
            per_layer.setdefault(l, []).append(v.float().cpu())
    n_layers = len(per_layer)

    results = {"int2v": [], "vqv_fp16": [], "vqv_e5m2": []}
    for l in range(n_layers):
        vs = torch.cat(per_layer[l], dim=1)  # [H, T_total, D]
        H, Ttot, D = vs.shape
        NG = D // G
        perm = torch.tensor([g + m * NG for g in range(NG) for m in range(G)])
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(D)
        R = rv[l]["rotation"].float()
        errs = {"int2v": [0.0, 0.0], "vqv_fp16": [0.0, 0.0], "vqv_e5m2": [0.0, 0.0]}
        for h in range(H):  # per-head codebooks, per-head rows
            v = vs[h]
            if v.shape[0] > 4000:
                v = v[torch.randperm(v.shape[0])[:4000]]
            v_rot = v @ R
            cbs = [c for c in blob["codebooks"][(l, h)]]
            base = v_rot.pow(2).sum().item()
            for name, rec in (
                ("int2v", int2v_affine(v_rot)),
                ("vqv_fp16", vqv_recon(v_rot, cbs, perm, inv_perm, False)),
                ("vqv_e5m2", vqv_recon(v_rot, cbs, perm, inv_perm, True)),
            ):
                errs[name][0] += (rec - v_rot).pow(2).sum().item()
                errs[name][1] += base
        for name in results:
            results[name].append(errs[name][0] / errs[name][1])
        print(f"layer {l:2d}: " + "  ".join(f"{k}={results[k][-1]:.4f}" for k in results), flush=True)

    def headline(xs):  # layer-0 excluded per repo convention
        return sum(xs[1:]) / len(xs[1:])

    summary = {k: headline(v) for k, v in results.items()}
    summary["ratio_fp16_vs_int2"] = summary["vqv_fp16"] / summary["int2v"]
    summary["ratio_e5m2_vs_int2"] = summary["vqv_e5m2"] / summary["int2v"]
    summary["ratio_e5m2_vs_fp16"] = summary["vqv_e5m2"] / summary["vqv_fp16"]
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_layer": results},
                              indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
