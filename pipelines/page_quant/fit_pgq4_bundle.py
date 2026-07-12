#!/usr/bin/env python3
"""Fit the pgq4 bundle: folded-scalar families A/B/C over three bases.

Per plan4 (notes/page_quant/studies/plan4.md). Everything here is
calibration-static — that is the family's kernel story:

  bases     qpca_unc per-(layer,head) [default], OSCAR per-layer
            U_Q.Hadamard.Pbr (constructed WITH the vendored
            compute_kv_rotation functions — construction and parity check
            in one), r_sym (orthogonal control).
  stats     per basis x (layer,head): code_std (d,), uniform-grid alphas
            (one MSE-grid-fit scalar per width in {2,3,4}), sink_scale
            (absolute 8-bit escape grid covering the fit-pool sink rows).
  lm_cents  global unit Lloyd-Max levels for widths {2,3} (top width stays
            a covering uniform grid), near-zero level forced to exact 0.
  profiles  family B rungs: water-filled monotone 32-coord block-width
            profiles per head + per-layer pooled variant + the sharing
            penalty that freezes prof_share; prefix-truncation ladder
            (px_profiles) is global static.
  omega     tau carried from the pgq2 freeze (constant 0.5 at every rate it
            froze); extended to the pgq4 ladder rates unchanged — regime
            law says tau only matters where rate binds, and no re-selection
            is allowed post-registration.

--eval runs the held-out G1/G2/G3 sweep (selection rows, disjoint from fit)
for an explicit mode list, routing pgq_rvq_* refs to the pgq2 bundle, and
writes pgq4_heldout_report.json.

    python -u pipelines/page_quant/fit_pgq4_bundle.py --device cuda:0
    python -u pipelines/page_quant/fit_pgq4_bundle.py --device cuda:0 \
        --eval-modes pgq_fold_rdo:2.0 pgq_rvq_rdo:2.0
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
import importlib.util  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

import pipelines.ec.fit_ec_bundle as ecb  # noqa: E402
from pipelines.ec.fit_ec_bundle import (  # noqa: E402
    RawPool, build_basis,
)
from pipelines.page_quant.fit_pgq2_bundle import (  # noqa: E402
    BigPool, bigpool_rows,
)
import pipelines.page_quant.phase1_empirics as ph1  # noqa: E402
from pipelines.page_quant.phase1_empirics import (  # noqa: E402
    load_mu_from_fit_stats,
)
from kvq.compression.pgq4_folded import (  # noqa: E402
    PGQ4_BUNDLE_VERSION, SINK_LIM, uniform_quant,
)
from kvq.compression.per_coord import unit_gaussian_centroids  # noqa: E402
from kvq.io import save_json, ensure_dir  # noqa: E402

OUT_DIR = REPO / "artifacts/page_quant2"
FROZEN2 = OUT_DIR / "frozen_choices.json"            # pgq2 taus, carried
PGQ2_BUNDLE = OUT_DIR / "pgq2_bundle__qpca_unc__compact8train40r400.pt"
OSCAR_ROT = REPO / "vendor/OSCAR/rotation/compute_kv_rotation.py"
PTOK = 64
D_HEAD = 128
NBLK = 4                                             # 32-coord width blocks
# Amendment A1: family-B profiles draw from {0,2,3,4,6}; stats (alphas, LM
# cents) are fit for every positive width so both ladders load from one
# bundle. Family A / px stay on the registered {0,2,3,4}.
PROF_LADDER = (0, 2, 3, 4, 6)
WIDTHS_POS = [w for w in PROF_LADDER if w > 0]
ALPHA_GRID = torch.logspace(math.log10(0.10), math.log10(1.6), 28)
# rung ladder densified around the operating points (192 / 256 / 320 bits
# per token at d=128 for b/c in {1.5, 2.0, 2.5}) — the RDO mixes adjacent
# rungs on the convex hull, so hull density near the op points is what the
# allocator can actually use
WF_TARGETS = [0, 128, 192, 224, 256, 288, 320, 768]  # bits/token (d=128)
PX_PROFILES = torch.tensor([
    [0, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0],
    [3, 3, 0, 0], [4, 4, 0, 0], [4, 4, 4, 0], [4, 4, 4, 4]])
SINK_COVER = 1.05
FIT_SAMPLE_CAP = 120_000
G2_LAYERS = [1, 8, 16, 24, 31]
LADDER_RATES = ["1.5", "2", "2.5"]
SEED = 20260709


def load_vendor_rotation_module():
    spec = importlib.util.spec_from_file_location("oscar_rot", OSCAR_ROT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def oscar_layer_rotations(cca: dict) -> torch.Tensor:
    """Per-layer R = U_Q @ H @ Pbr from head-pooled Sigma_Q, built with the
    vendored OSCAR functions (r_h_pbr, their validated-best composition)."""
    vend = load_vendor_rotation_module()
    L, Hkv, d, _ = cca["sigma_q"].shape
    hada = vend.build_hadamard(d)
    out = torch.empty(L, d, d)
    for l in range(L):
        cov = cca["sigma_q"][l].double().mean(0)
        cov = 0.5 * (cov + cov.T)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        R = vend.compose_rotation(eigvecs, eigvals, hada, "r_h_pbr")
        ortho_err = (R @ R.T - torch.eye(d, dtype=R.dtype)).abs().max()
        assert float(ortho_err) < 1e-8, (l, float(ortho_err))
        out[l] = R.float()
    return out


def forced_zero_cents(bits: int) -> torch.Tensor:
    c = unit_gaussian_centroids(bits).sort().values
    c[c.abs().argmin()] = 0.0
    return c


def fit_alphas(r: torch.Tensor, code_std: torch.Tensor) -> torch.Tensor:
    """MSE grid-fit of the uniform-grid step multiplier per width. The
    distortion the RDO sees is snap-aware, so overload from a tight alpha is
    priced automatically — the empirical argmin is the right calibration."""
    out = torch.empty(len(WIDTHS_POS))
    grid = ALPHA_GRID.to(r.device)
    for wi, w in enumerate(WIDTHS_POS):
        best, best_err = 1.0, float("inf")
        for a in grid:
            dq = uniform_quant(r, a * code_std, w)
            err = float((r - dq).square().mean())
            if err < best_err:
                best, best_err = float(a), err
        out[wi] = best
    return out


def monotone_profiles(dist_blk: torch.Tensor) -> torch.Tensor:
    """dist_blk (n_ladder, NBLK): MEASURED summed-per-block distortion of
    each ladder width (row order = PROF_LADDER; row 0 = width 0 = E[r^2]).
    Returns the (R, NBLK) rung ladder: for each bits/token target, the
    monotone non-increasing block-width tuple minimizing the measured
    distortion among tuples fitting the target."""
    cands = [t for t in itertools.product(range(len(PROF_LADDER)),
                                          repeat=NBLK)
             if all(t[i] >= t[i + 1] for i in range(NBLK - 1))]
    blk = D_HEAD // NBLK
    rows = []
    for tgt in WF_TARGETS:
        feas = [t for t in cands
                if blk * sum(PROF_LADDER[i] for i in t) <= tgt]
        if not feas:
            continue
        proxy = [sum(float(dist_blk[wi, b]) for b, wi in enumerate(t))
                 for t in feas]
        rows.append(feas[min(range(len(feas)), key=proxy.__getitem__)])
    rows = sorted(set(rows),
                  key=lambda t: sum(PROF_LADDER[i] for i in t))
    top = tuple([len(PROF_LADDER) - 1] * NBLK)
    if top not in rows:
        rows.append(top)
    return torch.tensor([[PROF_LADDER[i] for i in t] for t in rows],
                        dtype=torch.long)


def pad_profiles(p: torch.Tensor, n: int) -> torch.Tensor:
    """Pad/trim a ladder to exactly n rows, top rung always last (keeps
    every head's ladder the same shape for the (L,H,R,NBLK) tensor)."""
    if p.shape[0] == n:
        return p
    if p.shape[0] > n:
        return torch.cat([p[:n - 1], p[-1:]], 0)
    reps = p[-1:].expand(n - p.shape[0], -1)
    return torch.cat([p, reps], 0)


def profile_distortion(prof: torch.Tensor, dist_blk: torch.Tensor) -> float:
    """Measured distortion of a block-width profile row under dist_blk
    (n_ladder, NBLK)."""
    widx = {w: i for i, w in enumerate(PROF_LADDER)}
    return sum(float(dist_blk[widx[int(w)], b]) for b, w in enumerate(prof))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pool-rows", type=int, default=40)
    ap.add_argument("--eval-modes", nargs="*", default=None,
                    help="mode:rate pairs for the held-out G1-G3 sweep, "
                         "e.g. pgq_fold_rdo:2.0 pgq_rvq_rdo:2.0")
    ap.add_argument("--skip-fit", action="store_true",
                    help="reuse the existing bundle; only run --eval-modes")
    ap.add_argument("--pgq2-bundle", default=str(PGQ2_BUNDLE))
    ap.add_argument("--model-tag", default="llama31_8b",
                    choices=list(ecb.MODEL_PATHS))
    ap.add_argument("--heldout-out", default=None,
                    help="override the held-out report filename (pgq6 gate "
                         "runs must not clobber the canonical pgq4 report)")
    ap.add_argument("--bundle", default=None,
                    help="evaluate --eval-modes against this bundle instead "
                         "of the canonical fit output (pgq8 gates load "
                         "pgq_dct* from the dct_std-carrying bundle)")
    args = ap.parse_args()
    dev = torch.device(args.device)
    ensure_dir(OUT_DIR)
    t0 = time.time()
    gen = torch.Generator().manual_seed(SEED)
    qwen = args.model_tag != "llama31_8b"
    if qwen:
        ph1.set_model_tag(args.model_tag)

    roles = json.loads(ecb.ROLES.read_text())
    cca = torch.load(ecb.CCA_STATS, map_location="cpu", weights_only=False)
    L, Hkv, d, _ = cca["sigma_q"].shape
    assert d == D_HEAD
    mu_q, mu_k = load_mu_from_fit_stats(roles["fit"])
    # Llama fits on the 40-row big pool; Qwen has exactly the 16 surviving
    # compact8 train raws, so its fit pool IS the 12 fit-role rows.
    fit_pool = (RawPool(roles["fit"]) if qwen
                else BigPool(bigpool_rows(args.pool_rows, gen)))
    sel_pool = RawPool(roles["selection"])

    if qwen:
        tag = f"{args.model_tag}_compact8train{len(roles['fit'])}"
        out_path = OUT_DIR / f"pgq5_bundle__qpca_unc__{tag}.pt"
    else:
        tag = f"compact8train{args.pool_rows}r400"
        out_path = OUT_DIR / f"pgq4_bundle__3bases__{tag}.pt"
    if args.bundle:
        assert args.skip_fit, "--bundle is an eval-only override"
        out_path = Path(args.bundle)

    if not args.skip_fit:
        Fq, Gq = build_basis("qpca_unc", cca, mu_k, None)
        if qwen:
            # 3-basis ablation closed by pgq4; Qwen refits qpca_unc only.
            bases = {"qpca_unc": {"forward": Fq, "inverse": Gq}}
        else:
            Fr, Gr = build_basis("r_sym", cca, mu_k, None)
            Ro = oscar_layer_rotations(cca)
            bases = {
                "qpca_unc": {"forward": Fq, "inverse": Gq},
                "r_sym": {"forward": Fr, "inverse": Gr},
                "oscar": {"forward": Ro,
                          "inverse": Ro.transpose(1, 2).contiguous()},
            }
        stats = {b: {"code_std": torch.ones(L, Hkv, d),
                     "alphas": torch.full((L, Hkv, len(WIDTHS_POS)), 0.5),
                     "sink_scale": torch.ones(L, Hkv, d)}
                 for b in bases}
        prof_head = torch.zeros(L, Hkv, len(WF_TARGETS), NBLK,
                                dtype=torch.long)
        prof_head[..., :] = PROF_LADDER[-1]
        # measured per-block distortion of every ladder width (LM grid for
        # bulk widths, covering uniform for the top — matching the proflm
        # arm), driving both profile selection and the sharing penalty
        dist_by_lh = torch.zeros(L, Hkv, len(PROF_LADDER), NBLK)
        cents_by_w = {w: forced_zero_cents(w) for w in WIDTHS_POS[:-1]}

        for l in range(1, L):
            for h in range(Hkv):
                ks, sinks, total = [], [], 0
                for kk in fit_pool.k_slices(l, h):
                    sinks.append(kk[:4])
                    if total < FIT_SAMPLE_CAP * 2:
                        ks.append(kk)
                        total += kk.shape[0]
                k_all = torch.cat(ks).to(dev)
                if k_all.shape[0] > FIT_SAMPLE_CAP:
                    sel = torch.randperm(k_all.shape[0], generator=gen)[
                        :FIT_SAMPLE_CAP].to(dev)
                    k_all = k_all[sel]
                k_sink = torch.cat(sinks).to(dev)
                muh = mu_k[l, h].to(dev)
                for bname, bset in bases.items():
                    Fh = (bset["forward"][l] if bset["forward"].dim() == 3
                          else bset["forward"][l, h]).to(dev)
                    r = (k_all - muh) @ Fh
                    cs = r.std(0).clamp_min(1e-6).cpu()
                    stats[bname]["code_std"][l, h] = cs
                    stats[bname]["alphas"][l, h] = fit_alphas(
                        r, cs.to(dev))
                    r_snk = (k_sink - muh) @ Fh
                    stats[bname]["sink_scale"][l, h] = (
                        r_snk.abs().amax(0).cpu() * SINK_COVER / SINK_LIM
                    ).clamp_min(1e-6)
                    if bname == "qpca_unc":
                        from kvq.compression.pgq4_folded import lm_quant
                        csd = cs.to(dev)
                        dist = torch.zeros(len(PROF_LADDER), NBLK)
                        blkw = d // NBLK
                        e0 = r.square().mean(0)
                        dist[0] = e0.reshape(NBLK, blkw).sum(1).cpu()
                        for wi, w in enumerate(WIDTHS_POS):
                            if w == WIDTHS_POS[-1]:
                                al = stats[bname]["alphas"][l, h, wi]
                                dq = uniform_quant(r, al * csd, w)
                            else:
                                dq = lm_quant(r, cents_by_w[w].to(dev), csd)
                            e2 = (r - dq).square().mean(0)
                            dist[wi + 1] = e2.reshape(NBLK, blkw).sum(1).cpu()
                        dist_by_lh[l, h] = dist
                        prof_head[l, h] = pad_profiles(
                            monotone_profiles(dist), len(WF_TARGETS))
            print(f"[fit] layer {l}/{L-1} ({time.time()-t0:.0f}s)",
                  flush=True)

        # per-layer pooled profiles + sharing penalty (measured distortion;
        # the freeze rule is <3% mean relative penalty)
        prof_layer = torch.zeros(L, len(WF_TARGETS), NBLK, dtype=torch.long)
        prof_layer[..., :] = PROF_LADDER[-1]
        pens = []
        for l in range(1, L):
            d_lay = dist_by_lh[l].mean(0)
            prof_layer[l] = pad_profiles(monotone_profiles(d_lay),
                                         len(WF_TARGETS))
            for h in range(Hkv):
                for ri in range(len(WF_TARGETS)):
                    dh = profile_distortion(prof_head[l, h, ri],
                                            dist_by_lh[l, h])
                    dl = profile_distortion(prof_layer[l, ri],
                                            dist_by_lh[l, h])
                    if dh > 0:
                        pens.append((dl - dh) / dh)
        share_penalty = float(torch.tensor(pens).mean())
        prof_share = "layer" if share_penalty < 0.03 else "head"
        print(f"[fit] prof sharing penalty {share_penalty:.4f} -> "
              f"prof_share={prof_share}", flush=True)

        taus = {}
        if FROZEN2.exists():
            taus = json.loads(FROZEN2.read_text()).get("tau_by_rate", {})
        base_tau = float(next(iter(taus.values()))) if taus else 0.5
        omega_tau_by_rate = {r: taus.get(r, base_tau) for r in LADDER_RATES}

        blob = {
            "pgq_version": PGQ4_BUNDLE_VERSION,
            "model_tag": args.model_tag, "ptok": PTOK,
            "n_layers": L, "n_kv_heads": Hkv, "head_dim": d,
            "mu": mu_k, "mu_q": mu_q,
            "bases": bases, "stats": stats,
            "lm_cents": [forced_zero_cents(w) for w in WIDTHS_POS[:-1]],
            "prof_width_ladder": list(PROF_LADDER),
            "prof_head": prof_head, "prof_layer": prof_layer,
            "prof_share": prof_share,
            "prof_share_penalty": share_penalty,
            "px_profiles": PX_PROFILES,
            "omega_tau_by_rate": omega_tau_by_rate,
            "omega_tau_carry_rule": (
                "pgq2 frozen constant carried to the pgq4 ladder rates; "
                "no re-selection (plan4 sec.1)"),
            "omega_clamp_bits": 4.0,
            "fit_rows": len(roles["fit"]) if qwen else args.pool_rows,
            "selection_rows": roles["selection"],
            "cca_stats_path": str(ecb.CCA_STATS),
            "cca_stats_mtime": int(ecb.CCA_STATS.stat().st_mtime),
        }
        torch.save(blob, out_path)
        sha8 = hashlib.sha256(out_path.read_bytes()).hexdigest()[:8]
        print(f"[fit] bundle -> {out_path} (sha8 {sha8}, "
              f"{out_path.stat().st_size / 1e6:.0f} MB)", flush=True)
    else:
        assert out_path.exists(), out_path
        sha8 = hashlib.sha256(out_path.read_bytes()).hexdigest()[:8]

    if not args.eval_modes:
        return

    # ---- held-out G1/G2/G3 on the selection rows (pgq3 harness) ----------
    from kvq.compression.page_quant import load_pgq_compressors_from_bundle
    Fq, _ = build_basis("qpca_unc", cca, mu_k, None)
    report = {}
    for spec in args.eval_modes:
        mode, kb_s = spec.rsplit(":", 1)
        kb = float(kb_s)
        bundle = (args.pgq2_bundle
                  if mode.startswith(("pgq_rvq_", "pgq_nd_"))
                  else str(out_path))
        comps, _ = load_pgq_compressors_from_bundle(bundle, mode, kb)
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
                        Fh_ = Fq[l, h].to(dev)
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
                        if hist and len(hist) == len(c.rung_hist)
                        else list(c.rung_hist))
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
    rep_name = args.heldout_out or ("pgq5_heldout_report.json" if qwen
                                    else "pgq4_heldout_report.json")
    save_json(OUT_DIR / rep_name,
              {"bundle": str(out_path), "sha8": sha8, "report": report})


if __name__ == "__main__":
    main()
