#!/usr/bin/env python3
"""TurboQuant-INT2 quantizer design check (offline, CPU by default).

The engine's TurboQuant baseline stores Lloyd-Max-bucketized indices but must
reconstruct through the readers' uniform `(q - zero) * scale` contract. This
script derives the MSE-optimal uniform spacing DELTA* for LM-thresholded
standard-Gaussian buckets and quantifies, on Gaussian data and (--real) on
captured post-Hadamard K/V:

  A  min-max uniform INT2 (the QuaRot/Naive write path)
  B  LM thresholds + LM_RATIO=1.16 uniform encoding (the mixed-pool approx)
  C  LM thresholds + DELTA* uniform encoding (the fix)
  D  exact Lloyd-Max (centroid reconstruction — the unattainable reference)

    .venv/bin/python pipelines/oscar_e2e/turbo_lm_check.py
    .venv/bin/python pipelines/oscar_e2e/turbo_lm_check.py --real --gpu 0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]

LM_T = 0.9810652732849121      # engine's LM threshold (|z| boundary)
LM_C_OUT = 1.5095585584640503  # engine's outer LM centroid
LM_C_IN = 2 * LM_T - LM_C_OUT  # inner centroid implied by threshold midpoint


def phi(x):
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def derive_delta():
    """Optimal uniform spacing for levels (q-1.5)*DELTA given LM thresholds.

    minimize E[(x - DELTA*l(x))^2] over DELTA, x ~ N(0,1),
    l(x) in {-1.5,-0.5,0.5,1.5} by LM-threshold bucket
    => DELTA* = E[x*l(x)] / E[l(x)^2].
    """
    p_out = 1 - Phi(LM_T)
    p_in = Phi(LM_T) - 0.5
    e_xl = 2 * (1.5 * phi(LM_T) + 0.5 * (phi(0) - phi(LM_T)))
    e_l2 = 2 * (2.25 * p_out + 0.25 * p_in)
    delta = e_xl / e_l2
    d_fix = 1 - e_xl**2 / e_l2
    d_lm = 1 - 2 * (phi(LM_T) ** 2 / p_out + (phi(0) - phi(LM_T)) ** 2 / p_in)
    r116 = (LM_C_OUT * 2 / 3) * 1.16
    d_116 = 1 - 2 * r116 * e_xl + r116**2 * e_l2
    return delta, {"D_exact_LM": d_lm, "D_delta_star": d_fix,
                   "D_ratio116": d_116,
                   "penalty_delta_star": d_fix / d_lm - 1,
                   "penalty_ratio116": d_116 / d_lm - 1}


def q_minmax(v):
    lo = v.amin(-1, keepdim=True)
    hi = v.amax(-1, keepdim=True)
    s = (hi - lo).clamp_min(1e-8) / 3.0
    z = -lo / s
    q = (v / s + z + 0.5).floor().clamp(0, 3)
    return (q - z) * s


def q_lm_uniform(v, delta):
    m = v.mean(-1, keepdim=True)
    sd = (v.var(-1, unbiased=False, keepdim=True) + 1e-8).sqrt()
    z = (v - m) / sd
    q = (z >= -LM_T).float() + (z >= 0).float() + (z >= LM_T).float()
    return (q - 1.5) * (delta * sd) + m


def q_lm_exact(v):
    m = v.mean(-1, keepdim=True)
    sd = (v.var(-1, unbiased=False, keepdim=True) + 1e-8).sqrt()
    z = (v - m) / sd
    q = ((z >= -LM_T).long() + (z >= 0).long() + (z >= LM_T).long())
    cent = torch.tensor([-LM_C_OUT, -LM_C_IN, LM_C_IN, LM_C_OUT], device=v.device)
    return cent[q] * sd + m


def rel_mse(rec, ref):
    return ((rec - ref) ** 2).sum().item() / (ref ** 2).sum().item()


def fwht(x):
    d = x.shape[-1]
    h = 1
    y = x.clone()
    while h < d:
        y = y.view(*y.shape[:-1], d // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2).reshape(*x.shape[:-1], d)
        h *= 2
    return y / math.sqrt(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--n-prompts", type=int, default=4)
    args = ap.parse_args()

    delta, gauss = derive_delta()
    print(f"DELTA* = {delta!r}")
    print(json.dumps({k: round(v, 6) for k, v in gauss.items()}, indent=2))

    # Monte-Carlo confirmation on 128-dim Gaussian rows (row-empirical
    # mean/std, like the kernel — the analytic numbers assume known moments).
    g = torch.Generator().manual_seed(7)
    x = torch.randn(200_000, 128, generator=g)
    mc = {"minmax": rel_mse(q_minmax(x), x),
          "lm_ratio116": rel_mse(q_lm_uniform(x, (LM_C_OUT * 2 / 3) * 1.16), x),
          "lm_delta_star": rel_mse(q_lm_uniform(x, delta), x),
          "lm_exact": rel_mse(q_lm_exact(x), x)}
    print("gaussian 128-dim rows:", json.dumps({k: round(v, 5) for k, v in mc.items()}))

    if args.real:
        import os
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
        sys.path.insert(0, str(REPO))
        import _bootstrap  # noqa
        from pipelines.calibration.capture_raw import run_prefill_qkv_capture
        from kvq.capture.model import load_model_and_tokenizer, get_model_device
        model, tok = load_model_and_tokenizer(args.model, device_map="auto",
                                              dtype_name="float16")
        dev = get_model_device(model)
        rows = [json.loads(l) for l in
                open(REPO / "artifacts/prompt_rows/gpqa_diamond_llama.jsonl")][: args.n_prompts]
        agg = {k: [0.0, 0.0] for k in ("minmax", "lm_ratio116", "lm_delta_star", "lm_exact")}
        for row in rows:
            ids = tok(row["prompt"], return_tensors="pt", truncation=True,
                      max_length=2048).input_ids.to(dev)
            cap = run_prefill_qkv_capture(model, ids)
            for name in ("k_post", "v"):
                t = cap[name] if name in cap else None
                if t is None:
                    continue
                L = t.shape[0]
                for l in range(1, L):  # layer-0 excluded
                    v = t[l].reshape(-1, t.shape[-1]).float().to(dev)
                    vr = fwht(v)  # order = head_dim (QuaRot/Turbo config)
                    base = (vr ** 2).sum().item()
                    for key, rec in (("minmax", q_minmax(vr)),
                                     ("lm_ratio116", q_lm_uniform(vr, (LM_C_OUT * 2 / 3) * 1.16)),
                                     ("lm_delta_star", q_lm_uniform(vr, delta)),
                                     ("lm_exact", q_lm_exact(vr))):
                        agg[key][0] += ((rec - vr) ** 2).sum().item()
                        agg[key][1] += base
            del cap
        real = {k: v[0] / v[1] for k, v in agg.items()}
        print("real post-FWHT K/V (Llama, layer-0 excl):",
              json.dumps({k: round(v, 5) for k, v in real.items()}))


if __name__ == "__main__":
    main()
