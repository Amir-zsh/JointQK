#!/usr/bin/env python3
"""Fit the paged per-token-precision (pgq) K bundle on compact8 TRAIN rows.

One bundle serves every rate: the rung ladder is fit once (base step solved at
--b-base bits/coord, uniform per head — qpca_unc's optimal config), and the
press's page budget (k_bits) alone sets the rate. This kills the fit-vs-fit
confound across rates and two fits' wall-clock (protocol-critic rec).

Emits:
  - qpca_unc forward/inverse, mu (=k_mean over fit rows), mu_q (GQA-pooled,
    combine_stats convention sum_q/(group*tokens), verified against the n400
    bundle's sigma_q trace)
  - delta_base (L,H) + m_grid, per-rung frozen models (support vals/lens/
    probs, per-rung Vmax padding); rungs whose pooled fit entropy < 0.25 b/c
    are dropped (silent k_mean-eviction guard)
  - code_std (L,H,d) + alphas (L,H,W) for the fixed-width family
  - held-out selection-row telemetry per (mode, rate): achieved rate incl.
    sideband, overflow fraction, rung histogram  (the G2/overflow gate)

    python pipelines/page_quant/fit_pgq_bundle.py --device cuda:0 \
        --omega-tau 1.0 --omega-clamp-bits 4.0
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

from pipelines.ec.fit_ec_bundle import (  # noqa: E402
    CCA_STATS, ROLES, RawPool, build_basis, entropy_256bins,
    entropy_per_coord_dev, solve_delta_uniform_dev,
)
from pipelines.page_quant.phase1_empirics import load_mu_from_fit_stats  # noqa: E402
from kvq.compression.ec_roundtrip import dz_round  # noqa: E402
from kvq.compression.page_quant import (  # noqa: E402
    PGQ_BUNDLE_VERSION, load_pgq_compressors_from_bundle,
)
from kvq.io import save_json, ensure_dir  # noqa: E402

OUT_DIR = REPO / "artifacts/page_quant"
M_GRID_FULL = [2.0 ** (j / 2.0) for j in range(-2, 6)]
WIDTHS = [0, 1, 2, 4]   # powers of 2 only: byte-aligned fields for the kernel
ALPHA_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
MIN_RUNG_ENTROPY = 0.25   # b/c; coarser rungs dropped (k_mean-eviction guard)
EVAL_RATES = [1.5, 1.0, 0.75]
EVAL_MODES = ["pgq_plain", "pgq_ea", "pgq_pagerung", "pgq_fixed",
              "pgq_fixed_ea"]


def freeze_models_flat(idx: torch.Tensor):
    """Per-coord alphabets + probs from integer codes idx (N, d), vectorized
    via the flat-offset bincount trick. Returns list of (vals, probs) CPU."""
    n, d = idx.shape
    dev = idx.device
    ii = idx.long()
    mins = ii.min(dim=0).values
    shifted = ii - mins.unsqueeze(0)
    widths = (shifted.max(dim=0).values + 1).clamp_min(1)
    offsets = torch.zeros(d, dtype=torch.long, device=dev)
    if d > 1:
        offsets[1:] = torch.cumsum(widths, 0)[:-1]
    total = int(offsets[-1].item() + widths[-1].item())
    counts = torch.bincount((shifted + offsets.unsqueeze(0)).reshape(-1),
                            minlength=total)
    out = []
    counts_cpu = counts.cpu()
    mins_cpu = mins.cpu()
    widths_cpu = widths.cpu()
    offs_cpu = offsets.cpu()
    for j in range(d):
        c = counts_cpu[offs_cpu[j]: offs_cpu[j] + widths_cpu[j]]
        present = torch.nonzero(c > 0).squeeze(1)
        vals = (present + mins_cpu[j]).float()
        p = c[present].double()
        p = (p / p.sum()).float()
        out.append((vals, p))
    return out


def fit_alphas(r: torch.Tensor, code_std: torch.Tensor) -> torch.Tensor:
    """Scale multipliers per width for saturating mid-rise grids on codes r
    (N, d). Width 0 has no scale (alpha=1 placeholder).

    v1 (MSE-fit scales at every width) clipped the heavy per-coord tails at
    EVERY width, and the clipped keys are exactly the attention sinks (~80%
    of realized mass) — the first pgq_fixed bench collapsed to F1 ~2 (see
    report). Fix: caller passes code_std = per-coord 99.9th-percentile |r|;
    w >= 2 use alpha_w = 2/nlev so (nlev/2)*s_j = q999_j exactly (tail-safe
    on every coord — outlier tokens always have an escape width); w = 1 is
    bulk-optimal instead (levels ~ +-E|r|; a 1-bit grid cannot cover tails
    AND bulk, and the RDO's exact per-width distortion routes outlier tokens
    away from w=1 on its own)."""
    best = torch.ones(len(WIDTHS))
    ratio = float(r.abs().mean() / code_std.mean().clamp_min(1e-12))
    for wi, w in enumerate(WIDTHS):
        if w == 0:
            continue
        best[wi] = 2.0 * ratio if w == 1 else 2.0 / (1 << w)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--b-base", type=float, default=1.5)
    ap.add_argument("--dz", type=float, default=0.5)
    ap.add_argument("--ptok", type=int, default=64)
    ap.add_argument("--omega-tau", type=float, required=True)
    ap.add_argument("--omega-clamp-bits", type=float, required=True)
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()
    dev = torch.device(args.device)
    ensure_dir(OUT_DIR)
    t0 = time.time()

    roles = json.loads(ROLES.read_text())
    cca = torch.load(CCA_STATS, map_location="cpu", weights_only=False)
    L, Hkv, d, _ = cca["sigma_q"].shape
    mu_q, mu_k = load_mu_from_fit_stats(roles["fit"])
    F, Finv = build_basis("qpca_unc", cca, mu_k, None)
    pool = RawPool(roles["fit"])
    R = len(M_GRID_FULL)

    delta_base = torch.zeros(L, Hkv)
    code_std = torch.zeros(L, Hkv, d)
    alphas = torch.zeros(L, Hkv, len(WIDTHS))
    models = [dict() for _ in range(R)]     # rung -> {(l,h): [(vals,p)]*d}
    rung_entropy = torch.zeros(R, dtype=torch.float64)
    heads_done = 0

    for l in range(1, L):
        for h in range(Hkv):
            r = pool.fetch_codes(l, h, F, mu_k, device=dev).float()
            hc = entropy_256bins(r.double())
            d0 = float(2.0 ** (hc.mean().item() - args.b_base))
            db = solve_delta_uniform_dev(r.double(), args.b_base, args.dz, d0)
            delta_base[l, h] = float(db[0])
            # per-coord 99.9% range, NOT std — see fit_alphas docstring
            code_std[l, h] = torch.quantile(r.abs().double(), 0.999,
                                            dim=0).float().cpu()
            alphas[l, h] = fit_alphas(r, code_std[l, h].to(dev))
            for ri, m in enumerate(M_GRID_FULL):
                delta = torch.as_tensor(delta_base[l, h] * m, device=dev)
                idx = dz_round(r, delta, args.dz)
                models[ri][(l, h)] = freeze_models_flat(idx)
                rung_entropy[ri] += float(
                    entropy_per_coord_dev(idx.long()).sum()) / d
            heads_done += 1
        print(f"[fit] layer {l}/{L - 1} done ({time.time() - t0:.0f}s)",
              flush=True)

    rung_entropy /= heads_done
    keep = [ri for ri in range(R)
            if rung_entropy[ri] >= MIN_RUNG_ENTROPY]
    print(f"[fit] pooled rung entropies (b/c): "
          f"{[f'{float(e):.3f}' for e in rung_entropy]}; keeping "
          f"{[f'{M_GRID_FULL[ri]:.3f}' for ri in keep]}", flush=True)

    m_grid = [M_GRID_FULL[ri] for ri in keep]
    support_vals, support_lens, support_probs = [], [], []
    for ri in keep:
        vmax = max(len(v) for mdl in models[ri].values() for v, _ in mdl)
        sv = torch.full((L, Hkv, d, vmax), float("inf"))
        sp = torch.zeros(L, Hkv, d, vmax)
        sl = torch.zeros(L, Hkv, d, dtype=torch.int32)
        for (l, h), mdl in models[ri].items():
            for j, (vals, p) in enumerate(mdl):
                nv = len(vals)
                sv[l, h, j, :nv] = vals
                sp[l, h, j, :nv] = p
                sl[l, h, j] = nv
        support_vals.append(sv)
        support_lens.append(sl)
        support_probs.append(sp)

    out_path = OUT_DIR / (f"pgq_bundle__qpca_unc__dz{args.dz:g}__"
                          f"base{args.b_base:g}__compact8train18.pt")
    blob = {
        "pgq_version": PGQ_BUNDLE_VERSION,
        "model_tag": "llama31_8b", "basis": "qpca_unc",
        "dz": args.dz, "b_base": args.b_base, "ptok": args.ptok,
        "omega_tau": args.omega_tau,
        "omega_clamp_bits": args.omega_clamp_bits,
        "omega_signal": "m = mu_q^T k / sqrt(d)  (phase1-selected)",
        "n_layers": L, "n_kv_heads": Hkv, "head_dim": d,
        "forward": F, "inverse": Finv, "mu": mu_k, "mu_q": mu_q,
        "delta_base": delta_base, "m_grid": m_grid,
        "rung_entropy_kept": [float(rung_entropy[ri]) for ri in keep],
        "support_vals": support_vals, "support_lens": support_lens,
        "support_probs": support_probs,
        "code_std": code_std, "widths": WIDTHS, "alphas": alphas,
        "fit_rows": roles["fit"], "selection_rows": roles["selection"],
        "cca_stats_path": str(CCA_STATS),
        "cca_stats_mtime": int(CCA_STATS.stat().st_mtime),
    }
    torch.save(blob, out_path)
    print(f"[fit] bundle -> {out_path} "
          f"({out_path.stat().st_size / 1e6:.0f} MB, "
          f"{time.time() - t0:.0f}s)", flush=True)

    if args.skip_eval:
        return

    # ---- held-out G2/overflow gate on the 8 selection rows ----------------
    sel_pool = RawPool(roles["selection"])
    report = {}
    for mode in EVAL_MODES:
        for b in EVAL_RATES:
            comps, _ = load_pgq_compressors_from_bundle(out_path, mode, b)
            payload = side = toks = over = total_pages = 0.0
            hist = None
            for l in range(1, L):
                for h in range(Hkv):
                    c = comps[(l, h)].to(dev)
                    for k_slice in sel_pool.k_slices(l, h):
                        c.roundtrip(k_slice.to(dev))
                    payload += c.bits_payload
                    side += c.bits_side
                    toks += c.tokens_total
                    over += c.pages_overflow
                    total_pages += c.pages_total
                    hist = ([x + y for x, y in zip(hist, c.rung_hist)]
                            if hist else list(c.rung_hist))
            rate = (payload + side) / (toks * d)
            report[f"{mode}@b{b:g}"] = {
                "rate_heldout": rate,
                "overflow_frac": over / max(total_pages, 1),
                "rung_hist": hist,
            }
            print(f"[eval] {mode}@b{b:g}: rate={rate:.4f} "
                  f"overflow={over / max(total_pages, 1):.4%} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    save_json(OUT_DIR / "pgq_heldout_report.json",
              {"bundle": str(out_path), "report": report})


if __name__ == "__main__":
    main()
