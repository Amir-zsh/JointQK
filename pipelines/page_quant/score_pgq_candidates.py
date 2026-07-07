#!/usr/bin/env python3
"""Phase-3 fidelity screening for the page_quant study (G-P3 gate).

Scores every candidate (pgq modes x rates + ec_uniform controls) on the 8
compact8-TRAIN selection rows with the EC study's exact metric definitions
(top1 / top5 / k_mse / logit_err, layers 1..31, GQA-flattened queries), plus
the pgq telemetry (achieved rate incl. sideband, overflow). Selection is made
BEFORE any F1 runs (selection-honesty rule); the G-P3 bar is >= TurboQuant on
every proxy at 2.0 b/c equivalent, and candidate ranking at lower rates.

    python pipelines/page_quant/score_pgq_candidates.py --device cuda:0 \
        --rates 1.5 1.0
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import _bootstrap  # noqa: E402,F401

sys.path.insert(0, str(REPO / "entropy_coding"))

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

from pipelines.ec.fit_ec_bundle import ROLES, RawPool  # noqa: E402
from kvq.compression.ec_roundtrip import load_ec_compressors_from_bundle  # noqa: E402
from kvq.compression.page_quant import load_pgq_compressors_from_bundle  # noqa: E402
from kvq.io import save_json  # noqa: E402

EC_DIR = REPO / "artifacts/ec/llama31_8b"
PGQ_BUNDLE = REPO / ("artifacts/page_quant/"
                     "pgq_bundle__qpca_unc__dz0.5__base1.5__compact8train18.pt")
OUT = REPO / "artifacts/page_quant/phaseA_pgq.json"
Q_CHUNK = 8192
PGQ_MODES = ["pgq_pagerung", "pgq_plain", "pgq_ea", "pgq_fixed",
             "pgq_fixed_ea"]


def zero_acc():
    return {"top1": 0, "top5": 0, "nq": 0, "se": 0.0, "ne": 0,
            "logit_num": 0.0, "logit_den": 0.0}


def score_head(comp, k, q, acc, dev):
    k_hat = comp.roundtrip(k.unsqueeze(0)).squeeze(0)
    err = (k - k_hat).float()
    acc["se"] += float(err.square().sum())
    acc["ne"] += err.numel()
    T = k.shape[0]
    k5 = min(5, T)
    for s in range(0, q.shape[0], Q_CHUNK):
        qc = q[s: s + Q_CHUNK]
        lg = qc @ k.T
        lh = qc @ k_hat.T
        ref = lg.argmax(1)
        acc["top1"] += int((ref == lh.argmax(1)).sum())
        topk = lh.topk(k5, dim=1).indices
        acc["top5"] += int((topk == ref.unsqueeze(1)).any(1).sum())
        acc["nq"] += qc.shape[0]
        e = err
        acc["logit_num"] += float(((qc @ e.T) ** 2).sum())
        acc["logit_den"] += qc.shape[0] * T * T
    return acc


def finish(acc):
    return {"top1": acc["top1"] / max(acc["nq"], 1),
            "top5": acc["top5"] / max(acc["nq"], 1),
            "k_mse": acc["se"] / max(acc["ne"], 1),
            "logit_err": acc["logit_num"] / max(acc["logit_den"], 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rates", type=float, nargs="+", default=[1.5, 1.0])
    ap.add_argument("--modes", nargs="+", default=PGQ_MODES)
    ap.add_argument("--q-cap", type=int, default=8192,
                    help="max queries per (l,h,row); full 4T queries x 18 "
                         "methods is a 4h pass — capped, seeded subsample")
    args = ap.parse_args()
    dev = torch.device(args.device)
    t0 = time.time()

    roles = json.loads(ROLES.read_text())
    pool = RawPool(roles["selection"])
    rng = torch.Generator().manual_seed(20260707)

    methods = {}
    for b in args.rates:
        for mode in args.modes:
            comps, _ = load_pgq_compressors_from_bundle(PGQ_BUNDLE, mode, b)
            methods[f"{mode}@b{b:g}"] = comps
        ecb = (EC_DIR / f"ec_bundle__qpca_unc__b{b:g}__dz0.5__"
                        f"compact8train18__uniform.pt")
        if ecb.exists():
            comps, _ = load_ec_compressors_from_bundle(ecb)
            methods[f"ecu@b{b:g}"] = comps

    L, Hkv = 32, 8
    accs = {name: zero_acc() for name in methods}
    for l in range(1, L):
        for h in range(Hkv):
            for key in pool.rows:
                art = pool.art(key)
                T = int(art["prompt_length"])
                k = art["k_post"][l, h, :T].to(dev).float()
                gs = art["q_post"].shape[1] // Hkv
                q = (art["q_post"][l, h * gs:(h + 1) * gs, :T]
                     .to(dev).float().reshape(-1, k.shape[-1]))
                if q.shape[0] > args.q_cap:
                    sel = torch.randperm(q.shape[0],
                                         generator=rng)[: args.q_cap].to(dev)
                    q = q[sel]
                for name, comps in methods.items():
                    c = comps[(l, h)]
                    c.to(dev) if hasattr(c, "to") else None
                    score_head(c, k, q, accs[name], dev)
        print(f"[score] layer {l}/{L - 1} done ({time.time() - t0:.0f}s)",
              flush=True)

    report = {}
    for name, acc in accs.items():
        report[name] = finish(acc)
        comps = methods[name]
        c0 = next(iter(comps.values()))
        if hasattr(c0, "bits_payload"):
            payload = sum(c.bits_payload for c in comps.values())
            side = sum(c.bits_side for c in comps.values())
            toks = sum(c.tokens_total for c in comps.values())
            over = sum(c.pages_overflow for c in comps.values())
            pages = sum(c.pages_total for c in comps.values())
            report[name]["rate_heldout"] = (payload + side) / max(toks, 1) / 128
            report[name]["overflow_frac"] = over / max(pages, 1)

    save_json(OUT, {"reference": {
        "tq_k2": {"top1": 0.4031, "top5": 0.9212, "k_mse": 0.5572,
                  "logit_err": 0.02329, "rate": 2.125,
                  "source": "phaseA_report.json (EC study, same rows)"},
        "jq_k2": {"top1": 0.5596, "top5": 0.9836, "k_mse": 0.2597,
                  "logit_err": 0.00207},
        "ec_qpca_unc_dz0.5@1.95": {"note": "see phaseA_report.json"},
    }, "candidates": report})

    print(f"\n{'method':22s} {'top1':>7s} {'top5':>7s} {'k_mse':>9s} "
          f"{'logit_err':>10s} {'rate':>7s} {'ovf%':>6s}")
    print(f"{'tq_k2 (ref @2.125)':22s} {0.4031:7.4f} {0.9212:7.4f} "
          f"{0.5572:9.4f} {0.02329:10.5f}")
    for name in sorted(report):
        r = report[name]
        rate = r.get("rate_heldout")
        ovf = r.get("overflow_frac")
        print(f"{name:22s} {r['top1']:7.4f} {r['top5']:7.4f} "
              f"{r['k_mse']:9.4f} {r['logit_err']:10.5f} "
              + (f"{rate:7.3f}" if rate is not None else f"{'-':>7s}")
              + (f" {100 * ovf:5.2f}%" if ovf is not None else ""))
    print(f"-> {OUT} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
