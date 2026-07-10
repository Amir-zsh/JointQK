"""pgq3 family (d): learned per-head linear pair (F, G) — first autograd
loop in the repo. Pre-registered trigger fired 2026-07-09: TCQ and E8 both
screened below the RVQ incumbent, so this is the closing control: can a
learned transform rescue scalar codes? (RVQ dominating under the SAME
frozen qpca_unc basis predicts no.)

Objective: Q-weighted raw-domain reconstruction. With qpca_unc F0 = G0^-1
and G0 Sigma_Q G0^T = I, Sigma_Q = F0 F0^T, so the Q-weighted SE of a raw
residual e is ||e @ F0||^2 — F0 is the FROZEN metric map; the trained pair
(F, G) is the codec. Quantizer in the loop: per-coord mid-tread scalar
with a learned per-coord log-scale, straight-through estimator (the
dominant w=2 rung of the 1.5 b/c ladder; norm/direction structure is
layered on top by the family at eval time, not trained here).

Overfit gate: selection/fit loss ratio > 1.3 per head -> fall back to
qpca_unc for that head (RVQ fallback pattern). Output transform override
is consumed by fit_pgq3_bundle.py --transform-override.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipelines"))
import _bootstrap  # noqa: E402,F401

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

from pipelines.ec.fit_ec_bundle import (  # noqa: E402
    CCA_STATS, ROLES, RawPool, build_basis,
)
from pipelines.page_quant.fit_pgq2_bundle import (  # noqa: E402
    BigPool, bigpool_rows,
)
from pipelines.page_quant.phase1_empirics import (  # noqa: E402
    load_mu_from_fit_stats,
)
from kvq.io import save_json, ensure_dir  # noqa: E402

OUT_DIR = REPO / "artifacts/page_quant2"
SEED = 20260709
STEPS = 1000
BATCH = 4096
LR = 3e-4
WIDTH_LEVELS = 1          # mid-tread clamp ±1 (w=2's symmetric core)
OVERFIT_MAX = 1.3
FIT_CAP = 100_000
SEL_CAP = 20_000


def ste_round(x):
    return x + (torch.round(x) - x).detach()


def raw_k(pool, l, h, dev, cap, seed):
    ks = torch.cat(list(pool.k_slices(l, h))).float()
    if cap and ks.shape[0] > cap:
        g = torch.Generator().manual_seed(seed)
        ks = ks[torch.randperm(ks.shape[0], generator=g)[:cap]]
    return ks.to(dev)


def train_head(kf, ks, F0, G0, mu, seed):
    """kf (Nf,d) fit rows, ks (Ns,d) selection rows -> F, G, ratio, gain."""
    g = torch.Generator().manual_seed(seed)
    F = torch.nn.Parameter(F0.clone())
    G = torch.nn.Parameter(G0.clone())
    logs0 = (kf - mu).matmul(F0).std(0).clamp_min(1e-4).log()
    logs = torch.nn.Parameter(logs0.clone())
    opt = torch.optim.Adam([F, G, logs], lr=LR)

    def codec_loss(kb, F_, G_, logs_):
        r = (kb - mu) @ F_
        s = logs_.exp()
        q = ste_round((r / s).clamp(-WIDTH_LEVELS, WIDTH_LEVELS)) * s
        k_hat = q @ G_ + mu
        return ((k_hat - kb) @ F0).square().sum(-1).mean()

    for _ in range(STEPS):
        idx = torch.randint(0, kf.shape[0], (BATCH,), generator=g).to(
            kf.device)
        opt.zero_grad(set_to_none=True)
        codec_loss(kf[idx], F, G, logs).backward()
        opt.step()

    with torch.no_grad():
        base_fit = codec_loss(kf[:SEL_CAP], F0, G0, logs0)
        fit_l = codec_loss(kf[:SEL_CAP], F, G, logs)
        sel_l = codec_loss(ks, F, G, logs)
        ratio = float(sel_l / fit_l.clamp_min(1e-12))
        gain = float(base_fit / fit_l.clamp_min(1e-12))
    return F.detach(), G.detach(), ratio, gain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pool-rows", type=int, default=60)
    args = ap.parse_args()
    dev = torch.device(args.device)
    ensure_dir(OUT_DIR)
    t0 = time.time()

    roles = json.loads(ROLES.read_text())
    _, mu_k = load_mu_from_fit_stats(roles["fit"])
    cca = torch.load(CCA_STATS, map_location="cpu", weights_only=False)
    F0s, G0s = build_basis("qpca_unc", cca, mu_k, None)
    L, Hkv, d, _ = F0s.shape
    gen = torch.Generator().manual_seed(SEED)
    fit_pool = BigPool(bigpool_rows(args.pool_rows, gen))
    sel_pool = RawPool(roles["selection"])

    Fout = F0s.clone()
    Gout = G0s.clone()
    report = {}
    n_fallback = 0
    for l in range(1, L):
        for h in range(Hkv):
            mu = mu_k[l, h].to(dev)
            kf = raw_k(fit_pool, l, h, dev, FIT_CAP, SEED + 8 * l + h)
            ks = raw_k(sel_pool, l, h, dev, SEL_CAP, SEED + 999)
            F, G, ratio, gain = train_head(kf, ks, F0s[l, h].to(dev),
                                           G0s[l, h].to(dev), mu,
                                           seed=1000 * l + h)
            ok = ratio <= OVERFIT_MAX
            if ok:
                Fout[l, h] = F.cpu()
                Gout[l, h] = G.cpu()
            else:
                n_fallback += 1
            report[f"l{l}h{h}"] = {"overfit_ratio": round(ratio, 4),
                                   "train_gain": round(gain, 4),
                                   "fallback": not ok}
        print(f"[lin] layer {l}/{L-1} fb={n_fallback} "
              f"({time.time()-t0:.0f}s)", flush=True)

    out = OUT_DIR / "linear_pair_llama31_8b.pt"
    torch.save({"forward": Fout, "inverse": Gout,
                "metric": "qweighted via frozen qpca_unc F0",
                "steps": STEPS, "width_levels": WIDTH_LEVELS,
                "overfit_max": OVERFIT_MAX, "seed": SEED}, out)
    save_json(OUT_DIR / "linear_pair_report.json",
              {"n_fallback": n_fallback, "n_heads": (L - 1) * Hkv,
               "heads": report})
    print(f"[done] fallback {n_fallback}/{(L-1)*Hkv} -> {out} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
