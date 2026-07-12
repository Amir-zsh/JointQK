#!/usr/bin/env python3
"""pgq6 P1 observed-attention ORACLE (plan6 sec.2 — screening only).

Upper-bounds what phase-asymmetric signals could add: allocation driven by
each row's own realized attention (mean attention mass from the last-10%
prompt queries), used two ways — per-token distortion weights AND top-decile
merge protection. This signal does not exist at decode time (the user's
long-generation objection), so it is NEVER promoted to the codec; the gap
oracle-vs-plain quantifies the headroom the phase-symmetric format leaves.

Metric matches score_pgq_candidates (q-weighted logit_err on selection rows).

    python pipelines/page_quant/pgq6_oracle.py --device cuda:6
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
OUT = REPO / "artifacts/page_quant2/pgq6_oracle.json"
Q_CAP = 4096
PROTECT_Q = 0.90          # top decile protected from merging
WEIGHT_CLAMP = 16.0       # cap the attention-weight dynamic range


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rates", type=float, nargs="+", default=[1.25, 1.5])
    args = ap.parse_args()
    dev = torch.device(args.device)
    t0 = time.time()

    roles = json.loads(ecmod.ROLES.read_text())
    pool = RawPool(roles["selection"])
    rng = torch.Generator().manual_seed(20260710)

    blob = torch.load(str(BUNDLE), map_location="cpu", weights_only=False)
    L, Hkv = int(blob["n_layers"]), int(blob["n_kv_heads"])
    del blob

    report = {}
    for rate in args.rates:
        plain = load_pgq_compressors_from_bundle(
            str(BUNDLE), "pgq_mrglm_rdo", rate)[0]
        oracle = load_pgq_compressors_from_bundle(
            str(BUNDLE), "pgq_mrglm_rdo", rate)[0]
        acc = {a: [0.0, 0.0] for a in ("plain", "oracle")}   # num, den
        for l in range(1, L):
            for h in range(Hkv):
                for key in pool.rows:
                    art = pool.art(key)
                    T = int(art["prompt_length"])
                    d = art["k_post"].shape[-1]
                    k = art["k_post"][l, h, :T].to(dev).float()
                    gs = art["q_post"].shape[1] // Hkv
                    q = (art["q_post"][l, h * gs:(h + 1) * gs, :T]
                         .to(dev).float().reshape(-1, d))
                    if q.shape[0] > Q_CAP:
                        sel = torch.randperm(
                            q.shape[0], generator=rng)[:Q_CAP].to(dev)
                        q = q[sel]
                    # realized attention mass from the last-10% queries
                    qlate = (art["q_post"][l, h * gs:(h + 1) * gs,
                                           int(0.9 * T):T]
                             .to(dev).float().reshape(-1, d)[:512])
                    mass = torch.softmax(
                        qlate @ k.T / math.sqrt(d), 1).mean(0)
                    w = (mass / mass.mean().clamp_min(1e-12)) \
                        .clamp(1.0 / WEIGHT_CLAMP, WEIGHT_CLAMP)
                    prot = mass >= mass.quantile(PROTECT_Q)

                    for arm, comps, kw in (
                            ("plain", plain, {}),
                            ("oracle", oracle,
                             {"token_weights": w, "protect_mask": prot})):
                        c = comps[(l, h)].to(dev)
                        khat = c.roundtrip(k.unsqueeze(0), start_pos=0,
                                           **kw).squeeze(0)
                        e = (k - khat)
                        acc[arm][0] += float(((q @ e.T) ** 2).sum())
                        acc[arm][1] += q.shape[0] * T * T
            print(f"[oracle] b={rate:g} layer {l}/{L-1} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        le = {a: acc[a][0] / max(acc[a][1], 1) for a in acc}
        report[f"b{rate:g}"] = {
            "logit_err": le,
            "headroom": (le["plain"] - le["oracle"]) / max(le["plain"],
                                                           1e-12),
        }
        print(f"[oracle] b={rate:g}: plain={le['plain']:.5f} "
              f"oracle={le['oracle']:.5f} "
              f"headroom={report[f'b{rate:g}']['headroom']:.1%}", flush=True)

    save_json(OUT, report)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
