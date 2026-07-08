#!/usr/bin/env python3
"""Select and FREEZE the pgq2 omega hyperparameters (tau per rate, gate
theta) on the 8 compact8-TRAIN selection rows — before any F1 runs.

Scoring per (arm, rate, tau): realized-attention-mass-weighted logit error
(the omega-aligned metric; Q-MSE is tautologically anti-omega) + top-1
retention, on a fixed (layer, head) probe grid with each row's own
last-10% queries vs first-90% keys. theta (omega-confidence gate) is chosen
at the frozen tau by minimax regret across rows against the better of
{omega on, omega off} per row — repobench-p is the in-protocol OOD-code
sentinel (the lcc failure mode).

Writes artifacts/page_quant2/frozen_choices.json and PATCHES the bundle's
omega_tau_by_rate / gate_theta in place (ea/eag modes refuse to load until
this has run — selection honesty by construction).

    python pipelines/page_quant/select_pgq2_hparams.py --device cuda:0
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
import math  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

from pipelines.ec.fit_ec_bundle import ROLES, RawPool  # noqa: E402
from kvq.compression.norm_direction import NormDirectionPagedCompressor  # noqa: E402
from kvq.compression.rvq import ResidualVQPagedCompressor  # noqa: E402
from kvq.io import save_json  # noqa: E402

BUNDLE = REPO / "artifacts/page_quant2/pgq2_bundle__qpca_unc__compact8train18.pt"
OUT = REPO / "artifacts/page_quant2/frozen_choices.json"
TAUS = [0.0, 0.25, 0.5, 1.0]
THETAS = [0.25, 0.5, 0.75]
RATES = [1.5, 1.0, 0.75]
PROBE_LAYERS = [1, 8, 16, 24, 31]
PROBE_HEADS = [0, 3, 7]
NQ = 384


def build_comp(blob, fam, l, h, b, tau, theta):
    common = dict(
        forward_map=blob["forward"][l, h], inverse_map=blob["inverse"][l, h],
        mu=blob["mu"][l, h], mu_q=blob["mu_q"][l, h],
        b_page=b, ptok=int(blob["ptok"]), mode="rdo",
        omega_tau=tau, omega_clamp_bits=float(blob["omega_clamp_bits"]),
        gate_theta=theta, head_m_spread=float(blob["head_m_spread"][l, h]))
    if fam == "nd":
        return NormDirectionPagedCompressor(
            coord_std_dir=blob["coord_std_dir"][l, h],
            profiles=blob["nd_profiles"], uniform_rung=None, **common)
    return ResidualVQPagedCompressor(
        codebooks=[cb[l][h] for cb in blob["rvq_codebooks"]],
        perm=blob["rvq_perm"][l, h], uniform_stages=None, **common)


def probe_scores(blob, pool, fam, b, tau, theta, dev):
    """Returns per-row dict: row_key -> (wlogit, top1)."""
    d = int(blob["head_dim"])
    per_row = {}
    for l in PROBE_LAYERS:
        for h in PROBE_HEADS:
            comp = build_comp(blob, fam, l, h, b, tau, theta).to(dev)
            for key in pool.rows:
                art = pool.art(key)
                T = int(art["prompt_length"])
                t90 = max(64, int(0.9 * T) // 64 * 64)
                k = art["k_post"][l, h, :t90].to(dev).float()
                gs = art["q_post"].shape[1] // blob["n_kv_heads"]
                q = (art["q_post"][l, h * gs:(h + 1) * gs, t90:T]
                     .to(dev).float().reshape(-1, d)[:NQ])
                khat = comp.roundtrip(k.unsqueeze(0)).squeeze(0)
                lr = q @ k.T / math.sqrt(d)
                lh = q @ khat.T / math.sqrt(d)
                mass = torch.softmax(lr, 1).mean(0)
                w = (mass / mass.sum().clamp_min(1e-30)).unsqueeze(0)
                wl = float(((lr - lh).square() * w).sum(1).mean())
                t1 = float((lr.argmax(1) == lh.argmax(1)).float().mean())
                acc = per_row.setdefault(key[0], [0.0, 0.0, 0])
                acc[0] += wl
                acc[1] += t1
                acc[2] += 1
    return {k: (v[0] / v[2], v[1] / v[2]) for k, v in per_row.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)
    t0 = time.time()

    blob = torch.load(BUNDLE, map_location="cpu", weights_only=False)
    roles = json.loads(ROLES.read_text())
    pool = RawPool(roles["selection"])

    frozen = {"tau_by_rate": {}, "per_arm": {}, "theta": None}
    tau_scores = {}
    for b in RATES:
        for fam in ("nd", "rvq"):
            for tau in TAUS:
                sc = probe_scores(blob, pool, fam, b, tau, 0.0, dev)
                mean_wl = sum(v[0] for v in sc.values()) / len(sc)
                mean_t1 = sum(v[1] for v in sc.values()) / len(sc)
                tau_scores[(b, fam, tau)] = (mean_wl, mean_t1, sc)
                print(f"[tau] b={b} {fam} tau={tau}: wlogit={mean_wl:.4e} "
                      f"top1={mean_t1:.4f} ({time.time() - t0:.0f}s)",
                      flush=True)
        # pick tau per rate: min mean wlogit ACROSS ARMS (shared tau keeps
        # the ea contrast interpretable), tie-broken by top1
        best = min(TAUS, key=lambda t: (
            tau_scores[(b, "nd", t)][0] + tau_scores[(b, "rvq", t)][0]))
        frozen["tau_by_rate"][f"{b:g}"] = best
        print(f"[tau] FROZEN b={b}: tau={best}")

    # theta at the primary rate (1.0), winner-agnostic (scored on both arms):
    # minimax regret across rows vs the better of tau-on / tau-off per row
    b0 = 1.0
    tau0 = frozen["tau_by_rate"]["1"] if "1" in frozen["tau_by_rate"] \
        else frozen["tau_by_rate"]["1.0"]
    theta_best, theta_reg = None, float("inf")
    if tau0 > 0:
        for theta in THETAS:
            worst = 0.0
            for fam in ("nd", "rvq"):
                sc_on = tau_scores[(b0, fam, tau0)][2]
                sc_off = tau_scores[(b0, fam, 0.0)][2]
                sc_g = probe_scores(blob, pool, fam, b0, tau0, theta, dev)
                for row in sc_g:
                    ideal = min(sc_on[row][0], sc_off[row][0])
                    worst = max(worst, sc_g[row][0] - ideal)
            print(f"[theta] {theta}: minimax regret {worst:.4e} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if worst < theta_reg:
                theta_best, theta_reg = theta, worst
    frozen["theta"] = theta_best if theta_best is not None else 0.5

    git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True,
                             cwd=REPO).stdout.strip()
    frozen.update({
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_sha": git_sha,
        "bundle": str(BUNDLE),
        "selection_rows_only": True,
        "mandatory_cells": [
            "pgq_nd_uni:1", "pgq_nd_uni:2", "pgq_nd_rdo:1.0",
            "pgq_nd_rdo:1.5", "pgq_nd_ea:1.0", "pgq_rvq_uni:2",
            "pgq_rvq_rdo:1.5", "pgq_rvq_rdo:1.0", "pgq_rvq_rdo:0.75",
            "pgq_rvq_ea:1.0"],
    })
    save_json(OUT, frozen)

    blob["omega_tau_by_rate"] = {k: float(v)
                                 for k, v in frozen["tau_by_rate"].items()}
    # ea at the uni-rate keys too (ea only benched at rdo rates; harmless)
    blob["gate_theta"] = float(frozen["theta"])
    torch.save(blob, BUNDLE)
    print(f"[freeze] -> {OUT}; bundle patched "
          f"(tau={frozen['tau_by_rate']}, theta={frozen['theta']}, "
          f"{time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
