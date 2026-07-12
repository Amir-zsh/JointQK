#!/usr/bin/env python3
"""pgq6 P1 value-coupling check (plan6 sec.2, one-off verification).

The merge format's exactness argument: with replicated centroid KEYS, the
within-cluster attention weights are equal, so keeping per-token VALUES in
eval is output-identical to deployment storing v̄ = mean. The residual error
vs fp16 is therefore key-side only (weight approximation), which logit_err
already measures. This script verifies that claim DIRECTLY at the attention
output: ||softmax(qK̂ᵀ/√d)V − softmax(qKᵀ/√d)V|| / ||softmax(qKᵀ/√d)V||
for the mrg arm vs the proflm arm at the same rate — if mrg's output error is
inflated beyond what its logit_err predicts relative to proflm, the coupling
claim fails and plan6 stops-and-amends.

    python pipelines/page_quant/pgq6_output_check.py --device cuda:5
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import _bootstrap  # noqa: E402,F401

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

import pipelines.ec.fit_ec_bundle as ecmod  # noqa: E402
from pipelines.ec.fit_ec_bundle import RawPool  # noqa: E402
from kvq.compression.page_quant import (  # noqa: E402
    load_pgq_compressors_from_bundle,
)
from kvq.io import save_json  # noqa: E402

BUNDLE = REPO / "artifacts/page_quant2/pgq4_bundle__3bases__compact8train40r400.pt"
OUT = REPO / "artifacts/page_quant2/pgq6_output_check.json"
G2_LAYERS = [1, 8, 16, 24, 31]
HEADS = [0, 3, 7]
N_ROWS = 2
Q_CAP = 512


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rates", type=float, nargs="+", default=[1.5, 2.0])
    args = ap.parse_args()
    dev = torch.device(args.device)
    t0 = time.time()

    roles = json.loads(ecmod.ROLES.read_text())
    pool = RawPool(roles["selection"][:N_ROWS])
    report = {}
    for rate in args.rates:
        arms = {
            "mrg": load_pgq_compressors_from_bundle(
                str(BUNDLE), "pgq_mrglm_rdo", rate)[0],
            "prof": load_pgq_compressors_from_bundle(
                str(BUNDLE), "pgq_proflm_rdo", rate)[0],
        }
        errs = {a: [] for a in arms}
        for key in pool.rows:
            art = pool.art(key)
            T = int(art["prompt_length"])
            gs = art["q_post"].shape[1] // 8
            for l in G2_LAYERS:
                for h in HEADS:
                    k = art["k_post"][l, h, :T].to(dev).float()
                    v = art["v"][l, h, :T].to(dev).float()
                    q = (art["q_post"][l, h * gs:(h + 1) * gs,
                                       int(0.9 * T):T]
                         .to(dev).float().reshape(-1, k.shape[-1])[:Q_CAP])
                    d = k.shape[-1]
                    p0 = torch.softmax(q @ k.T / math.sqrt(d), 1)
                    o0 = p0 @ v
                    for a, comps in arms.items():
                        c = comps[(l, h)].to(dev)
                        khat = c.roundtrip(k.unsqueeze(0),
                                           start_pos=0).squeeze(0)
                        pf = torch.softmax(q @ khat.T / math.sqrt(d), 1)
                        of = pf @ v
                        errs[a].append(float(
                            (of - o0).norm()
                            / o0.norm().clamp_min(1e-9)))
        med = {a: float(torch.tensor(e).median()) for a, e in errs.items()}
        report[f"b{rate:g}"] = {
            "output_relerr_med": med,
            "mrg_over_prof": med["mrg"] / max(med["prof"], 1e-12),
        }
        print(f"[out-check] b={rate:g}: mrg={med['mrg']:.4f} "
              f"prof={med['prof']:.4f} "
              f"ratio={report[f'b{rate:g}']['mrg_over_prof']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    save_json(OUT, report)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
