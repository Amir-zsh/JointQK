#!/usr/bin/env python3
"""Fit the pgq3 bundle: TCQ (families a/b/e), E8 lattice (c), OSCAR arm (f).

Two passes, selection-honesty by construction:

  1. HPARAM FREEZE (cheap, probe layers x selection rows): sequential-greedy
     choice of {warp p, trellis states, sparse-K, TCQ mixing} by held-out
     Q-weighted direction MSE at the width-2 rung ->
     artifacts/page_quant2/frozen_choices_pgq3.json (sha-stamped, BEFORE any
     F1; the E8 mixer is ON and the OSCAR params are fixed by amendment 2 —
     they are protocol constants, not searched).
  2. FULL FIT (all layers 1..L-1, fit pool): warped-LM tables per TCQ width,
     E8 beta (q999 subvector norm / inradius of the m=2 region) + water-
     filled profiles, then the held-out G1 (rate) / G2 (sink) / G3 (norm)
     sweep on the selection rows -> pgq3_heldout_report.json.

A second bundle with tcq_states=1 (same tables refit at 2^w levels) is
emitted as the compander control (family a) — same file tag + "__scalarctl".

omega_tau_by_rate carries the pgq2 frozen taus unchanged (no re-selection).

    python -u pipelines/page_quant/fit_pgq3_bundle.py --device cuda:0 \
        --pool train400
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import _bootstrap  # noqa: E402,F401

sys.path.insert(0, str(REPO / "entropy_coding"))

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
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
from kvq.compression.page_quant import (  # noqa: E402
    PGQ3_BUNDLE_VERSION, build_hadamard,
)
from kvq.compression.tcq import fit_warped_lm_tables, tcq_viterbi  # noqa: E402
from kvq.io import save_json, ensure_dir  # noqa: E402

OUT_DIR = REPO / "artifacts/page_quant2"
FROZEN2 = OUT_DIR / "frozen_choices.json"          # pgq2 taus, carried over
FROZEN3 = OUT_DIR / "frozen_choices_pgq3.json"
PTOK = 64
D_HEAD = 128
TCQ_WIDTHS = [1, 2, 3]
P_CANDS = [1.0, 1.5, 2.0]
STATE_CANDS = [8, 4]
K_CANDS = [8, 16]
MAG_BITS = 8
E8_M_TARGETS = [0, 4, 8, 12, 16, 24, 32]           # sum(m) rung ladder
OSCAR_WIDTHS = [2, 3]
OSCAR_GROUP = 128
OSCAR_CLIP_Q = 0.96
PROBE_LAYERS = [1, 8, 16, 24, 31]
G2_LAYERS = [1, 8, 16, 24, 31]
EVAL_MODES = [("pgq_tcq_rdo", 1.5), ("pgq_tcq_rdo", 1.0),
              ("pgq_tcq_uni", 2),
              ("pgq_e8_rdo", 1.5), ("pgq_e8_rdo", 1.0),
              ("pgq_e8_uni", 3),
              ("pgq_oscar_uni", 0), ("pgq_oscar_uni", 1)]
SEED = 20260708
FIT_SAMPLE_CAP = 200_000                           # per-head LM-fit tokens


def dirs_of(pool, l, h, F, mu, dev, cap=None):
    rs = [(kk.to(dev) - mu) @ F for kk in pool.k_slices(l, h)]
    r = torch.cat(rs)
    if cap is not None and r.shape[0] > cap:
        g = torch.Generator().manual_seed(SEED + l * 8 + h)
        r = r[torch.randperm(r.shape[0], generator=g)[:cap].to(dev)]
    n16 = r.norm(dim=1).to(torch.float16).float()
    return r, r / n16.clamp_min(1e-8).unsqueeze(1), n16


def water_fill_profiles(energy: torch.Tensor) -> torch.Tensor:
    """energy (16,) per-subvector mean-square -> (R, 16) m-profiles at the
    E8_M_TARGETS ladder. Greedy: each +1 m goes to the subvector with the
    largest remaining energy * 4^-m marginal (6 dB/bit-per-dim heuristic)."""
    n_sub = energy.shape[0]
    m = torch.zeros(n_sub, dtype=torch.long)
    out = []
    tgt_i = 0
    for total in range(0, max(E8_M_TARGETS) + 1):
        if total == E8_M_TARGETS[tgt_i]:
            out.append(m.clone())
            tgt_i += 1
            if tgt_i == len(E8_M_TARGETS):
                break
        gain = energy * 4.0 ** (-m.float())
        gain[m >= 2] = -torch.inf
        m[gain.argmax()] += 1
    return torch.stack(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pool", choices=["fit18", "train400"],
                    default="train400")
    ap.add_argument("--pool-rows", type=int, default=60)
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-freeze", action="store_true",
                    help="reuse an existing frozen_choices_pgq3.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    ensure_dir(OUT_DIR)
    t0 = time.time()
    gen = torch.Generator().manual_seed(SEED)

    roles = json.loads(ROLES.read_text())
    cca = torch.load(CCA_STATS, map_location="cpu", weights_only=False)
    L, Hkv, d, _ = cca["sigma_q"].shape
    assert d == D_HEAD
    mu_q, mu_k = load_mu_from_fit_stats(roles["fit"])
    F, Finv = build_basis("qpca_unc", cca, mu_k, None)
    hada = build_hadamard(d)
    if args.pool == "train400":
        fit_pool = BigPool(bigpool_rows(args.pool_rows, gen))
    else:
        fit_pool = RawPool(roles["fit"])
    sel_pool = RawPool(roles["selection"])

    # ---- pass 1: hparam freeze on probe layers ---------------------------
    if args.skip_freeze and FROZEN3.exists():
        frozen = json.loads(FROZEN3.read_text())
        print(f"[freeze] reusing {FROZEN3}")
    else:
        def sel_mse(p, states, mixing):
            """Held-out width-2 direction MSE over probe (l,h)."""
            tot, n = 0.0, 0
            hd = hada.to(dev)
            for l in PROBE_LAYERS:
                for h in range(Hkv):
                    Fh = F[l, h].to(dev)
                    muh = mu_k[l, h].to(dev)
                    _, u, _ = dirs_of(fit_pool, l, h, Fh, muh, dev,
                                      cap=FIT_SAMPLE_CAP // 4)
                    _, us, _ = dirs_of(sel_pool, l, h, Fh, muh, dev)
                    if mixing:
                        u, us = u @ hd, us @ hd
                    nlev = 8 if states > 1 else 4          # width-2 rung
                    tab = fit_warped_lm_tables(u.cpu(), nlev, p).to(dev)
                    uh = tcq_viterbi(us, tab, states)
                    tot += float((us - uh).square().sum())
                    n += us.numel()
            return tot / n

        choices, log = {}, {}
        p_best = min(P_CANDS, key=lambda p: log.setdefault(
            f"p={p}", sel_mse(p, STATE_CANDS[0], True)))
        choices["warp_p"] = p_best
        s_best = min(STATE_CANDS, key=lambda s: log.setdefault(
            f"states={s}", sel_mse(p_best, s, True)))
        choices["tcq_states"] = s_best
        mix_best = min([True, False], key=lambda mx: log.setdefault(
            f"mix={mx}", sel_mse(p_best, s_best, mx)))
        choices["tcq_mixing"] = mix_best

        def sel_sparse_mse(kk):
            from kvq.compression.tcq import sparse_k_quantize
            tot, n = 0.0, 0
            for l in PROBE_LAYERS:
                for h in range(Hkv):
                    Fh = F[l, h].to(dev)
                    muh = mu_k[l, h].to(dev)
                    _, us, _ = dirs_of(sel_pool, l, h, Fh, muh, dev)
                    uh = sparse_k_quantize(us, kk, MAG_BITS)
                    # per-bit efficiency: MSE * rate so K=8 vs 16 compare
                    tot += float((us - uh).square().sum()) \
                        * (kk * (7 + MAG_BITS))
                    n += us.numel()
            return tot / n

        k_best = min(K_CANDS, key=lambda kk: log.setdefault(
            f"K={kk}", sel_sparse_mse(kk)))
        choices["sparse_k"] = k_best
        frozen = {"choices": choices, "selection_log": log,
                  "rule": "sequential greedy on probe-layer selection-row "
                          "direction MSE (width-2 rung); sparse-K by "
                          "MSE*rate", "probe_layers": PROBE_LAYERS,
                  "tau_by_rate": (json.loads(FROZEN2.read_text())
                                  ["tau_by_rate"] if FROZEN2.exists()
                                  else {})}
        save_json(FROZEN3, frozen)
        print(f"[freeze] {choices} -> {FROZEN3} ({time.time()-t0:.0f}s)",
              flush=True)

    ch = frozen["choices"]
    warp_p, states = float(ch["warp_p"]), int(ch["tcq_states"])
    sparse_k, tcq_mix = int(ch["sparse_k"]), bool(ch["tcq_mixing"])

    # ---- pass 2: full fit -------------------------------------------------
    def fit_bundle(states_out: int, tag_suffix: str):
        tcq_tabs = [torch.zeros(L, Hkv, d,
                                2 ** (w + 1) if states_out > 1 else 2 ** w)
                    for w in TCQ_WIDTHS]
        e8_beta = torch.ones(L, Hkv)
        e8_prof = torch.zeros(L, Hkv, len(E8_M_TARGETS), d // 8,
                              dtype=torch.long)
        hd = hada.to(dev)
        for l in range(1, L):
            for h in range(Hkv):
                Fh = F[l, h].to(dev)
                muh = mu_k[l, h].to(dev)
                r, u, _ = dirs_of(fit_pool, l, h, Fh, muh, dev,
                                  cap=FIT_SAMPLE_CAP)
                ut = u @ hd if tcq_mix else u
                for wi, w in enumerate(TCQ_WIDTHS):
                    nlev = 2 ** (w + 1) if states_out > 1 else 2 ** w
                    tcq_tabs[wi][l, h] = fit_warped_lm_tables(
                        ut.cpu(), nlev, warp_p)
                rm = r @ hd
                sub = rm.reshape(-1, d // 8, 8)
                snorm = sub.norm(dim=2)
                # beta so the m=2 Voronoi inradius (2*sqrt(2)) covers q999
                e8_beta[l, h] = float(snorm.reshape(-1).quantile(0.999)
                                      / (2.0 * math.sqrt(2.0)))
                e8_prof[l, h] = water_fill_profiles(
                    sub.square().mean((0, 2)).cpu())
            print(f"[fit{tag_suffix}] layer {l}/{L-1} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        tag = ("compact8train18" if args.pool == "fit18"
               else f"compact8train{args.pool_rows}r400") + tag_suffix
        out_path = OUT_DIR / f"pgq3_bundle__qpca_unc__{tag}.pt"
        blob = {
            "pgq_version": PGQ3_BUNDLE_VERSION,
            "model_tag": "llama31_8b", "basis": "qpca_unc", "ptok": PTOK,
            "n_layers": L, "n_kv_heads": Hkv, "head_dim": d,
            "forward": F, "inverse": Finv, "mu": mu_k, "mu_q": mu_q,
            "tcq_tables": tcq_tabs, "tcq_widths": TCQ_WIDTHS,
            "tcq_states": states_out,
            "tcq_mixer": hada if tcq_mix else None,
            "sparse_ks": [sparse_k], "mag_bits": MAG_BITS,
            "warp_p": warp_p,
            "e8_profiles": e8_prof, "e8_beta": e8_beta, "e8_mixer": hada,
            "oscar_mixer": hada, "oscar_widths": OSCAR_WIDTHS,
            "oscar_group": OSCAR_GROUP, "oscar_clip_q": OSCAR_CLIP_Q,
            "omega_tau_by_rate": frozen.get("tau_by_rate", {}),
            "omega_clamp_bits": 4.0,
            "fit_rows": roles["fit"], "selection_rows": roles["selection"],
            "cca_stats_path": str(CCA_STATS),
            "cca_stats_mtime": int(CCA_STATS.stat().st_mtime),
        }
        torch.save(blob, out_path)
        sha8 = hashlib.sha256(out_path.read_bytes()).hexdigest()[:8]
        print(f"[fit{tag_suffix}] bundle -> {out_path} (sha8 {sha8}, "
              f"{out_path.stat().st_size / 1e6:.0f} MB)", flush=True)
        return out_path, sha8

    out_path, sha8 = fit_bundle(states, "")
    ctl_path, ctl_sha8 = fit_bundle(1, "__scalarctl")
    frozen.setdefault("bundles", {})
    frozen["bundles"] = {"main": {"path": str(out_path), "sha8": sha8},
                         "scalarctl": {"path": str(ctl_path),
                                       "sha8": ctl_sha8}}
    save_json(FROZEN3, frozen)

    if args.skip_eval:
        return

    # ---- held-out gates G1/G2/G3 on the selection rows -------------------
    # (fit_pgq2_bundle.py eval loop, retargeted at the pgq3 loader)
    from kvq.compression.page_quant import load_pgq_compressors_from_bundle
    report = {}
    for mode, kb in EVAL_MODES:
        comps, _ = load_pgq_compressors_from_bundle(out_path, mode, kb)
        payload = side = toks = over = pages = 0.0
        hist = None
        sink_dm, sink_ce, norm_ratio = [], [], []
        for l in range(1, L):
            for h in range(Hkv):
                c = comps[(l, h)].to(dev)
                for ridx, kk in enumerate(sel_pool.k_slices(l, h)):
                    kg = kk.to(dev)
                    khat = c.roundtrip(kg.unsqueeze(0)).squeeze(0)
                    if l in G2_LAYERS and h in (0, 3, 7) and ridx < 4:
                        art = sel_pool.art(sel_pool.rows[ridx])
                        T = int(art["prompt_length"])
                        gs = art["q_post"].shape[1] // Hkv
                        q = (art["q_post"][l, h * gs:(h + 1) * gs,
                                           int(0.9 * T):T]
                             .to(dev).float().reshape(-1, d)[:256])
                        p0 = torch.softmax(q @ kg.T / math.sqrt(d), 1)
                        pf = torch.softmax(q @ khat.T / math.sqrt(d), 1)
                        sink_dm.append(float((pf[:, :4].sum(1)
                                              - p0[:, :4].sum(1)).mean()))
                        Fh_ = F[l, h].to(dev)
                        muh_ = mu_k[l, h].to(dev)
                        rc0 = (kg[:4] - muh_) @ Fh_
                        rch = (khat[:4] - muh_) @ Fh_
                        sink_ce.append(float(
                            (rch - rc0).norm()
                            / rc0.norm().clamp_min(1e-9)))
                        norm_ratio.append(float(
                            khat.norm(dim=1).mean()
                            / kg.norm(dim=1).mean().clamp_min(1e-9)))
                payload += c.bits_payload
                side += c.bits_side
                toks += c.tokens_total
                over += c.pages_overflow
                pages += c.pages_total
                hist = ([a + b for a, b in zip(hist, c.rung_hist)]
                        if hist else list(c.rung_hist))
        key = f"{mode}@b{kb:g}"
        report[key] = {
            "rate_heldout": (payload + side) / max(toks, 1) / d,
            "overflow_frac": over / max(pages, 1),
            "rung_hist": hist,
            "sink_mass_shift_med": float(torch.tensor(sink_dm).median()),
            "sink_code_relerr_med": float(torch.tensor(sink_ce).median()),
            "norm_ratio_med": float(torch.tensor(norm_ratio).median()),
        }
        r = report[key]
        print(f"[eval] {key}: rate={r['rate_heldout']:.4f} "
              f"ovf={r['overflow_frac']:.4%} "
              f"sinkΔ={r['sink_mass_shift_med']:+.4f} "
              f"sinkCE={r['sink_code_relerr_med']:.3f} "
              f"normR={r['norm_ratio_med']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    save_json(OUT_DIR / "pgq3_heldout_report.json",
              {"bundle": str(out_path), "sha8": sha8, "report": report})


if __name__ == "__main__":
    main()
