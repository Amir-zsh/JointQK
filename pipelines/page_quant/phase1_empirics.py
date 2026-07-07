#!/usr/bin/env python3
"""Phase 1 empirics for the page-unit quantization study (notes/page_quant/plan.md).

Three questions, all answerable from existing raw captures (no model forward):

  A. Predictiveness: does calibration-derived per-token importance (Expected
     Attention log z_t, quadratic k^T Sigma_Q k, ||k||^2 control) rank the keys
     that *realized* future queries actually attend?  Realized = softmax mass
     from the row's own last-10%-of-context queries (decode proxy) over the
     first-90% keys.  Measured on compact8-TRAIN selection rows AND
     (measurement-only, no parameter feedback) the eval-side posthoc rows to
     test OOD transfer (lcc / 2wikimqa are fully OOD to calibration).

  B. Within-page skew: how concentrated is realized attention inside 64-token
     pages, and what is the within-page dynamic range of the importance signal
     (drives the rung-ladder m-grid range).

  C. Oracle RD headroom: at matched empirical rate b in {2.0, 1.5, 1.0},
     Q-weighted distortion + realized top-1 retention for
       S0 token-uniform step (the ec_uniform control),
       S1 per-page rung (PageCodec status quo),
       S2 per-token RDO omega=1 (PageCodecRDO status quo),
       S3 per-token RDO omega=ExpectedAttention (the proposed method),
       S4 per-token RDO omega=realized attention (oracle upper bound).
     Basis: qpca_unc (code-domain SE == Q-weighted key MSE by G Sigma_Q G^T = I).

Selection rows drive all parameter choices; posthoc rows are measurement-only.
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
import time  # noqa: E402

import torch  # noqa: E402

from pipelines.ec.fit_ec_bundle import (  # noqa: E402
    CCA_STATS, ROLES, RawPool, build_basis, entropy_256bins,
    solve_delta_uniform_dev,
)
from kvq.compression.ec_roundtrip import dz_round, dz_dequant  # noqa: E402
from kvq.io import save_json, ensure_dir  # noqa: E402

POSTHOC_RAW = REPO / "artifacts/calibration/ec_posthoc_rate_llama31_8b/01_raw"
EC_RAW = REPO / "artifacts/calibration/ec_calib_compact8_train_llama31_8b/01_raw"
EC_STATS = REPO / "artifacts/calibration/ec_calib_compact8_train_llama31_8b/02_stats"
OUT_DIR = REPO / "artifacts/page_quant/phase1"

SEED = 20260707
D = 128
PTOK = 64
DZ = 0.5
M_GRID = [2.0 ** (j / 2.0) for j in range(-2, 6)]  # 0.5 .. ~5.66, R=8
RATES = [2.0, 1.5, 1.0]
N_QUERIES = 512
PARTB_LAYERS = [1, 4, 8, 12, 16, 20, 24, 28, 31]


def raw_path_any(config: str, row_index: int, split: str, root: Path) -> Path:
    name = f"longbench__{config}__row{int(row_index):05d}__{split}.pt"
    hits = sorted(root.glob(f"shard_*/{name}"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{name}: {len(hits)} matches under {root}")
    return hits[0]


def load_mu_from_fit_stats(fit_rows: list) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool mu_q, mu_k per (l, h) from the 18 fit rows' per-row 02_stats files.

    The sum_q convention (whether queries are counted T or group*T times per
    row) is resolved empirically: pick the denominator whose pooled sq_sum
    best reproduces the n400 bundle's sigma_q scale.
    """
    cca = torch.load(CCA_STATS, map_location="cpu", weights_only=False)
    sigma_q = cca["sigma_q"]
    sum_q = None
    sum_k = None
    sq_sum = None
    toks = 0
    for config, idx in fit_rows:
        name = f"longbench__{config}__row{int(idx):05d}__train.pt"
        hits = sorted(EC_STATS.glob(f"shard_*/{name}"))
        assert len(hits) == 1, f"stats missing for {name}"
        st = torch.load(hits[0], map_location="cpu", weights_only=False)
        g = int(st.get("group", 4))
        if sum_q is None:
            sum_q = st["sum_q"].double().clone()
            sum_k = st["sum_k"].double().clone()
            sq_sum = st["sq_sum"].double().clone()
        else:
            sum_q += st["sum_q"].double()
            sum_k += st["sum_k"].double()
            sq_sum += st["sq_sum"].double()
        toks += int(st["tokens"])
    # Denominator check: tokens vs tokens*group for the query-side stats.
    tr_bundle = torch.einsum("lhdd->lh", sigma_q.double())
    best = None
    for denom_name, denom in (("tokens", toks), ("tokens*group", toks * g)):
        tr_pool = torch.einsum("lhdd->lh", sq_sum / denom)
        ratio = (tr_pool / tr_bundle).median().item()
        err = abs(math.log(max(ratio, 1e-9)))
        if best is None or err < best[2]:
            best = (denom_name, denom, err, ratio)
    denom_name, denom_q, err, ratio = best
    print(f"[mu] query-side denominator resolved: {denom_name} "
          f"(median trace ratio {ratio:.4f}); k denominator = tokens")
    mu_q = (sum_q / denom_q).float()
    mu_k = (sum_k / toks).float()
    return mu_q, mu_k


def realized_attention(q: torch.Tensor, k: torch.Tensor):
    """q (n_q, d) queries, k (T90, d) keys -> (mass (T90,), top1 (n_q,) argmax)."""
    logits = (q @ k.T) / math.sqrt(D)
    probs = torch.softmax(logits, dim=1)
    return probs.mean(0), logits.argmax(1)


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    xr = x.argsort().argsort().float()
    yr = y.argsort().argsort().float()
    xr = xr - xr.mean()
    yr = yr - yr.mean()
    denom = (xr.norm() * yr.norm()).clamp_min(1e-12)
    return float((xr * yr).sum() / denom)


def top_capture(signal: torch.Tensor, realized: torch.Tensor,
                pred_frac=0.25, real_frac=0.10) -> float:
    t = signal.numel()
    n_pred = max(1, int(t * pred_frac))
    n_real = max(1, int(t * real_frac))
    pred = set(signal.topk(n_pred).indices.tolist())
    real = realized.topk(n_real).indices.tolist()
    hit = sum(1 for i in real if i in pred)
    return hit / len(real)


def part_a_row(art, sigma_q, mu_q, dev, rng) -> dict:
    """Predictiveness + within-page skew for one row. Returns per-(l,h) lists.

    Signals (all computable from the key + calibration only):
      m      mean logit         mu_q^T k / sqrt(d)
      logzc  Expected Attention m + k^T Sigma_c k / (2d), Sigma_c CENTERED
             (using uncentered Sigma_Q double-counts the mean: v1 bug)
      w2     uncentered energy  k^T Sigma_Q k  (v1 signal, kept for the record)
      kn     ||k||^2 control
    """
    T = int(art["prompt_length"])
    t90 = max(PTOK, int(0.9 * T) // PTOK * PTOK)  # page-aligned key horizon
    gs = art["q_post"].shape[1] // art["k_post"].shape[1]
    L = art["k_post"].shape[0]
    sigs = ("m", "logzc", "w2", "kn")
    out = {f"{p}_{s}": [] for s in sigs for p in ("rho", "cap")}
    out.update({s: [] for s in ("page_top8_share", "page_logzc_dr_bits",
                                "page_mean_energy_frac", "sink4_share")})
    for l in range(1, L):
        for h in range(art["k_post"].shape[1]):
            k = art["k_post"][l, h, :t90].to(dev).float()
            qg = art["q_post"][l, h * gs:(h + 1) * gs, t90:T].to(dev).float()
            q = qg.reshape(-1, D)
            if q.shape[0] > N_QUERIES:
                sel = torch.randperm(q.shape[0], generator=rng,
                                     device="cpu")[:N_QUERIES].to(dev)
                q = q[sel]
            mass, _ = realized_attention(q, k)

            sq = sigma_q[l, h].to(dev)
            mu = mu_q[l, h].to(dev)
            w2 = torch.einsum("td,de,te->t", k, sq, k)
            m = (k @ mu) / math.sqrt(D)
            kmu = k @ mu
            w2c = (w2 - kmu.square()).clamp_min(0.0)  # k^T (Sigma_Q - mu mu^T) k
            logzc = m + w2c / (2 * D)
            kn = (k * k).sum(1)

            for name, sig in (("m", m), ("logzc", logzc),
                              ("w2", w2), ("kn", kn)):
                out[f"rho_{name}"].append(spearman(sig, mass))
                out[f"cap_{name}"].append(top_capture(sig, mass))

            # within-page skew (realized) + importance dynamic range (predicted)
            npages = t90 // PTOK
            pm = mass[:npages * PTOK].reshape(npages, PTOK)
            top8 = pm.topk(8, dim=1).values.sum(1) / pm.sum(1).clamp_min(1e-12)
            out["page_top8_share"].append(float(top8.median()))
            out["sink4_share"].append(float(mass[:4].sum()))
            pz = logzc[:npages * PTOK].reshape(npages, PTOK) / math.log(2.0)
            dr = (pz.quantile(0.95, dim=1) - pz.quantile(0.05, dim=1))
            out["page_logzc_dr_bits"].append(float(dr.median()))

            # cross-token structure: fraction of (globally-centered) key energy
            # explained by per-page means
            kc = k[:npages * PTOK].reshape(npages, PTOK, D)
            kc = kc - kc.reshape(-1, D).mean(0)
            pmean = kc.mean(1, keepdim=True)
            frac = float((pmean.square().sum() * PTOK) /
                         kc.square().sum().clamp_min(1e-12))
            out["page_mean_energy_frac"].append(frac)
    return out


def rdo_assign(Dmat, Rmat, budget_bits, omega=None, iters=40):
    """Global-lambda per-token rung choice: argmin_r omega_t*D[t,r] + lam*R[t,r]
    with lam bisected so sum_t R[t, r_t] <= budget_bits."""
    Dw = Dmat if omega is None else Dmat * omega.unsqueeze(1)
    ar = torch.arange(Dmat.shape[0], device=Dmat.device)
    rmin = Rmat.argmin(1)
    if Rmat[ar, rmin].sum() > budget_bits:
        return rmin
    lo, hi = 0.0, max(1.0, float(Dw.max())) * 1e3
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        a = (Dw + mid * Rmat).argmin(1)
        if Rmat[ar, a].sum() > budget_bits:
            lo = mid
        else:
            hi = mid
    return (Dw + hi * Rmat).argmin(1)


def nlp_bits_per_token(idx: torch.Tensor) -> torch.Tensor:
    """Empirical -log2 p(idx[t, j]) summed over j -> (T,). Flat-offset trick."""
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
    flat = shifted + offsets.unsqueeze(0)
    counts = torch.bincount(flat.reshape(-1), minlength=total).float()
    nlp = -torch.log2((counts / n).clamp_min(1e-12))
    return nlp[flat].sum(1)


def rdo_assign_paged(Dmat, Rmat, page_budget_bits, ptok, omega=None, iters=40):
    """Per-page lambda bisection (what the fixed-byte codec actually does),
    vectorized over pages. Returns per-token rung assignment."""
    T = Dmat.shape[0]
    dev = Dmat.device
    Dw = (Dmat if omega is None else Dmat * omega.unsqueeze(1).float()).double()
    R64 = Rmat.double()
    npages = (T + ptok - 1) // ptok
    pad = npages * ptok - T
    if pad:
        zD = torch.zeros(pad, Dw.shape[1], device=dev, dtype=torch.float64)
        Dw = torch.cat([Dw, zD])
        R64 = torch.cat([R64, torch.zeros_like(zD)])
    Dp = Dw.reshape(npages, ptok, -1)
    Rp = R64.reshape(npages, ptok, -1)
    ntok = torch.full((npages,), float(ptok), device=dev, dtype=torch.float64)
    if pad:
        ntok[-1] = ptok - pad
    budget = page_budget_bits * ntok / ptok  # partial last page pro-rated
    lo = torch.zeros(npages, 1, 1, device=dev, dtype=torch.float64)
    hi = torch.full_like(lo, max(1.0, float(Dw.max())) * 1e3)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        a = (Dp + mid * Rp).argmin(2)
        rr = Rp.gather(2, a.unsqueeze(2)).squeeze(2).sum(1)
        over = (rr > budget).view(npages, 1, 1)
        lo = torch.where(over, mid, lo)
        hi = torch.where(over, hi, mid)
    a = (Dp + hi * Rp).argmin(2).reshape(-1)
    return a[:T]


def omega_from_m(m_sig, tau, clamp_bits):
    c = clamp_bits * math.log(2.0)
    return torch.exp((tau * (m_sig - m_sig.median())).clamp(-c, c))


def part_b_head(r, k, inv, mu_k_h, q, mass, b, dev, m_sig):
    """Oracle headroom for one (l, h) at target rate b. r: (T, d) codes.

    Arms: token-uniform / per-page rung / per-page RDO (flat + omega grid) /
    global RDO (flat, omega, realized-omega oracle). Sideband bits are NOT
    charged here (pure allocation headroom; sidebands enter in P2/P3).
    Omega arms are judged on wlogit_mse/top1 (attention-aligned), never on
    dist: under qpca_unc, flat RDO is the Q-MSE optimum by construction."""
    T = r.shape[0]
    hcoord = entropy_256bins(r)
    d0 = float(2.0 ** (hcoord.mean().item() - b))
    delta_base = solve_delta_uniform_dev(r, b, DZ, d0).to(dev).float()

    Dmat = torch.empty(T, len(M_GRID), device=dev)
    Rmat = torch.empty(T, len(M_GRID), device=dev)
    idx_by_rung = []
    for ri, m in enumerate(M_GRID):
        delta = delta_base * m
        idx = dz_round(r, delta.unsqueeze(0), DZ)
        dq = dz_dequant(idx, delta.unsqueeze(0), DZ)
        Dmat[:, ri] = (r - dq).square().sum(1)
        Rmat[:, ri] = nlp_bits_per_token(idx)
        idx_by_rung.append(idx)

    budget = b * D * T
    page_budget = b * D * PTOK
    om_real = (mass * mass.numel()).clamp(2.0 ** -6, 2.0 ** 6)
    schemes = {"S0_token_uniform": ("uniform", None),
               "S1_per_page_rung": ("pagerung", None),
               "S2p_rdo_paged_flat": ("paged", None),
               "S2_rdo_global_flat": ("global", None),
               "S4_rdo_global_oracle": ("global", om_real)}
    for tau in (0.5, 1.0, 2.0):
        for cb in (2.0, 4.0):
            om = omega_from_m(m_sig, tau, cb)
            schemes[f"S3p_paged_m_t{tau:g}_c{cb:g}"] = ("paged", om)
            if tau == 1.0:
                schemes[f"S3_global_m_t{tau:g}_c{cb:g}"] = ("global", om)

    base_ri = M_GRID.index(1.0)
    ar = torch.arange(T, device=dev)
    lg_ref = (q @ k.T) / math.sqrt(D)
    ref_arg = lg_ref.argmax(1)
    mass_w = (mass / mass.sum().clamp_min(1e-30)).unsqueeze(0)
    res = {}
    for name, (mode, om) in schemes.items():
        if mode == "uniform":
            assign = torch.full((T,), base_ri, dtype=torch.long, device=dev)
        elif mode == "pagerung":
            npages = (T + PTOK - 1) // PTOK
            assign = torch.empty(T, dtype=torch.long, device=dev)
            for pi in range(npages):
                sl = slice(pi * PTOK, min((pi + 1) * PTOK, T))
                pbud = b * D * (sl.stop - sl.start)
                chosen = len(M_GRID) - 1
                for ri in range(len(M_GRID)):
                    if Rmat[sl, ri].sum() <= pbud:
                        chosen = ri
                        break
                assign[sl] = chosen
        elif mode == "paged":
            assign = rdo_assign_paged(Dmat, Rmat, page_budget, PTOK, omega=om)
        else:
            assign = rdo_assign(Dmat, Rmat, budget, omega=om)
        rate = float(Rmat[ar, assign].sum() / (T * D))
        dist = float(Dmat[ar, assign].sum() / (T * D))
        r_hat = torch.empty_like(r)
        for ri in range(len(M_GRID)):
            msk = assign == ri
            if msk.any():
                delta = delta_base * M_GRID[ri]
                r_hat[msk] = dz_dequant(idx_by_rung[ri][msk],
                                        delta.unsqueeze(0), DZ)
        k_hat = r_hat @ inv + mu_k_h
        lg_hat = (q @ k_hat.T) / math.sqrt(D)
        err2 = (lg_ref - lg_hat).square()
        res[name] = {
            "rate": rate, "dist": dist,
            "top1": float((ref_arg == lg_hat.argmax(1)).float().mean()),
            "logit_mse": float(err2.mean()),
            "wlogit_mse": float((err2 * mass_w).sum(1).mean()),
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-part-b", action="store_true")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--clamp-bits", type=float, default=4.0)
    args = ap.parse_args()
    dev = torch.device(args.device)
    ensure_dir(OUT_DIR)
    rng = torch.Generator().manual_seed(SEED)
    t0 = time.time()

    roles = json.loads(ROLES.read_text())
    cca = torch.load(CCA_STATS, map_location="cpu", weights_only=False)
    sigma_q = cca["sigma_q"]
    mu_q, mu_k = load_mu_from_fit_stats(roles["fit"])

    # ---------- Part A + B on selection rows (TRAIN; parameter-eligible) ----
    rows_sel = [(c, int(i), "train", EC_RAW) for c, i in roles["selection"]]
    rows_post = []
    for p in sorted(POSTHOC_RAW.glob("shard_*/*.pt")):
        parts = p.name.split("__")
        rows_post.append((parts[1], int(parts[2][3:]), "test", POSTHOC_RAW))

    pred = {"selection": {}, "posthoc": {}}
    for group, rows in (("selection", rows_sel), ("posthoc", rows_post)):
        for config, idx, split, root in rows:
            art = torch.load(raw_path_any(config, idx, split, root),
                             map_location="cpu", mmap=True, weights_only=False)
            res = part_a_row(art, sigma_q, mu_q, dev, rng)
            pred[group].setdefault(config, []).append(
                {k: v for k, v in res.items()})
            del art
            print(f"[A] {group} {config} row{idx}  "
                  f"rho_m={torch.tensor(res['rho_m']).median():.3f} "
                  f"rho_logzc={torch.tensor(res['rho_logzc']).median():.3f} "
                  f"rho_w2={torch.tensor(res['rho_w2']).median():.3f} "
                  f"cap_m={torch.tensor(res['cap_m']).median():.3f} "
                  f"cap_logzc={torch.tensor(res['cap_logzc']).median():.3f} "
                  f"top8share={torch.tensor(res['page_top8_share']).median():.3f} "
                  f"sink4={torch.tensor(res['sink4_share']).median():.3f} "
                  f"dr_bits={torch.tensor(res['page_logzc_dr_bits']).median():.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    save_json(OUT_DIR / "predictiveness.json",
              {"meta": {"n_queries": N_QUERIES, "ptok": PTOK, "seed": SEED},
               "results": pred})

    if args.skip_part_b:
        return

    # ---------- Part C oracle headroom (selection rows only) ---------------
    F, Finv = build_basis("qpca_unc", cca, mu_k, None)
    pool = RawPool([[c, i] for c, i, _, _ in rows_sel])
    agg = {}
    for row_i, (config, idx, split, root) in enumerate(rows_sel):
        art = pool.art((config, idx))
        T = int(art["prompt_length"])
        t90 = max(PTOK, int(0.9 * T) // PTOK * PTOK)
        gs = art["q_post"].shape[1] // art["k_post"].shape[1]
        for l in PARTB_LAYERS:
            for h in range(8):
                k = art["k_post"][l, h, :t90].to(dev).float()
                qg = art["q_post"][l, h * gs:(h + 1) * gs, t90:T].to(dev)
                q = qg.reshape(-1, D).float()
                if q.shape[0] > N_QUERIES:
                    sel = torch.randperm(q.shape[0], generator=rng,
                                         device="cpu")[:N_QUERIES].to(dev)
                    q = q[sel]
                mass, _ = realized_attention(q, k)
                m_sig = (k @ mu_q[l, h].to(dev)) / math.sqrt(D)

                mu_k_h = mu_k[l, h].to(dev)
                r = (k - mu_k_h) @ F[l, h].to(dev)
                inv = Finv[l, h].to(dev)
                for b in RATES:
                    res = part_b_head(r, k, inv, mu_k_h, q, mass, b, dev,
                                      m_sig)
                    for name, m in res.items():
                        a = agg.setdefault((b, name), {"n": 0})
                        a["n"] += 1
                        for key, val in m.items():
                            a[key] = a.get(key, 0.0) + val
        print(f"[C] {config} row{idx} done ({time.time()-t0:.0f}s)", flush=True)

    headroom = {}
    for (b, name), a in agg.items():
        n = a.pop("n")
        headroom.setdefault(str(b), {})[name] = \
            {k: v / n for k, v in a.items()}
    save_json(OUT_DIR / "headroom.json",
              {"meta": {"m_grid": M_GRID, "layers": PARTB_LAYERS, "dz": DZ,
                        "tau": args.tau, "clamp_bits": args.clamp_bits},
               "headroom": headroom})
    print(json.dumps(headroom, indent=2))
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
