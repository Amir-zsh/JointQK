#!/usr/bin/env python3
"""Fit the pgq2 bundle: Arm A (norm + direction Lloyd-Max) and Arm B
(residual VQ), both on compact8 TRAIN fit rows, qpca_unc domain.

Emits ONE bundle serving all rates/modes (the press's k_bits selects), plus
held-out G1 (rate) / G2 (sink-mass) / G3 (norm-ratio) telemetry on the
selection rows -> artifacts/page_quant2/pgq2_heldout_report.json.

Field naming: coord_std_code / coord_std_dir are STDs (the v2 bundle's
`code_std` held a q999 range — the naming trap that broke sink scaling once).

omega_tau_by_rate / gate_theta are placeholders here; select_pgq2_hparams.py
patches the frozen values in AFTER the selection-row sweep (ea/eag modes
refuse to load until then — selection-honesty by construction).

    python pipelines/page_quant/fit_pgq2_bundle.py --device cuda:0
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
    CCA_STATS, ROLES, RawPool, build_basis,
)
from pipelines.page_quant.phase1_empirics import (  # noqa: E402
    load_mu_from_fit_stats,
)
from kvq.compression.page_quant import PGQ2_BUNDLE_VERSION  # noqa: E402
from kvq.io import save_json, ensure_dir  # noqa: E402

OUT_DIR = REPO / "artifacts/page_quant2"
BIGPOOL_RAW = REPO / "artifacts/calibration/pgq2_bigfit_llama31_8b/01_raw"
BIGPOOL_MANIFEST = REPO / ("artifacts/calibration_splits/pgq2_bigfit_40/"
                           "manifest.json")
EXCLUDE_FILE = REPO / ("artifacts/calibration_splits/"
                       "longbench_compact8_60_seed20260504_2k32k/"
                       "exclude_train_indices_for_eval.json")
FROZEN = REPO / "artifacts/page_quant2/frozen_choices.json"
PTOK = 64
ND_WIDTHS = [0, 1, 2, 3, 4, 8]   # 8 = absolute [-1,1] sink grid
RVQ_STAGES = 4
RVQ_NSUB = 8
RVQ_SD = 16
RVQ_K = 256
RVQ_OVERFIT_RATIO = 1.3
KMEANS_ITERS = 25
EVAL_MODES = [("pgq_nd_uni", 1), ("pgq_nd_uni", 2),
              ("pgq_nd_rdo", 1.5), ("pgq_nd_rdo", 1.0), ("pgq_nd_rdo", 0.75),
              ("pgq_rvq_uni", 2),
              ("pgq_rvq_rdo", 1.5), ("pgq_rvq_rdo", 1.0),
              ("pgq_rvq_rdo", 0.75)]
G2_LAYERS = [1, 8, 16, 24, 31]
SEED = 20260707


def kmeans_gpu(x: torch.Tensor, k: int, iters: int, gen: torch.Generator):
    """x (n_sub, N, sd) -> centroids (n_sub, k, sd). Random-sample init +
    Lloyd iterations, batched over subvectors; empty clusters re-seeded."""
    n_sub, N, sd = x.shape
    sel = torch.randperm(N, generator=gen)[:k].to(x.device)
    cent = x[:, sel, :].clone()
    for _ in range(iters):
        d2 = (x.square().sum(2, keepdim=True)
              - 2.0 * torch.bmm(x, cent.transpose(1, 2))
              + cent.square().sum(2).unsqueeze(1))       # (n_sub, N, k)
        idx = d2.argmin(2)
        cent_new = torch.zeros_like(cent)
        cnt = torch.zeros(n_sub, k, device=x.device)
        cent_new.scatter_add_(1, idx.unsqueeze(2).expand(-1, -1, sd), x)
        cnt.scatter_add_(1, idx, torch.ones_like(idx, dtype=x.dtype))
        empty = cnt == 0
        cent = torch.where(empty.unsqueeze(2),
                           cent, cent_new / cnt.clamp_min(1).unsqueeze(2))
        if empty.any():
            resel = torch.randperm(N, generator=gen)[: k].to(x.device)
            refill = x[:, resel, :]
            cent = torch.where(empty.unsqueeze(2), refill, cent)
    return cent


class BigPool(RawPool):
    """RawPool over the compact8 400-train-row capture (different root)."""

    def art(self, key):
        if key not in self._arts:
            c, i = key
            name = f"longbench__{c}__row{int(i):05d}__train.pt"
            hits = sorted(BIGPOOL_RAW.glob(f"shard_*/{name}"))
            assert len(hits) == 1, f"{name}: {len(hits)} hits"
            self._arts[key] = torch.load(hits[0], map_location="cpu",
                                         mmap=True, weights_only=False)
        return self._arts[key]


def bigpool_rows(n_rows: int, gen: torch.Generator):
    """Rows from the pgq2_bigfit capture manifest (40 compact8 TRAIN rows,
    disjoint from the EC fit/selection 26 — leakage-clean; captured fresh
    because the compact8 pool kept raw K for test rows only)."""
    man = json.loads(BIGPOOL_MANIFEST.read_text())
    return [[r["config"], int(r["row_index"])] for r in man["rows"]][:n_rows]


def quantize_rvq(x, cent):
    d2 = (x.square().sum(2, keepdim=True)
          - 2.0 * torch.bmm(x, cent.transpose(1, 2))
          + cent.square().sum(2).unsqueeze(1))
    idx = d2.argmin(2)
    return torch.gather(cent, 1,
                        idx.unsqueeze(2).expand(-1, -1, x.shape[2]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--pool", choices=["fit18", "train400"], default="fit18")
    ap.add_argument("--pool-rows", type=int, default=60,
                    help="rows sampled from the 400-train pool (train400)")
    args = ap.parse_args()
    dev = torch.device(args.device)
    ensure_dir(OUT_DIR)
    t0 = time.time()
    gen = torch.Generator().manual_seed(SEED)

    roles = json.loads(ROLES.read_text())
    cca = torch.load(CCA_STATS, map_location="cpu", weights_only=False)
    L, Hkv, d, _ = cca["sigma_q"].shape
    mu_q, mu_k = load_mu_from_fit_stats(roles["fit"])
    F, Finv = build_basis("qpca_unc", cca, mu_k, None)
    if args.pool == "train400":
        rows = bigpool_rows(args.pool_rows, gen)
        fit_pool = BigPool(rows)
        print(f"[fit] big pool: {len(rows)} compact8-TRAIN rows", flush=True)
    else:
        fit_pool = RawPool(roles["fit"])
    sel_pool = RawPool(roles["selection"])

    coord_std_code = torch.zeros(L, Hkv, d)
    coord_std_dir = torch.zeros(L, Hkv, d)
    head_m_spread = torch.ones(L, Hkv)
    rvq_perm = torch.zeros(L, Hkv, d, dtype=torch.long)
    rvq_cbs = [torch.zeros(L, Hkv, RVQ_NSUB, RVQ_K, RVQ_SD)
               for _ in range(RVQ_STAGES)]
    rvq_entries = torch.full((L, Hkv, RVQ_STAGES), RVQ_K,
                             dtype=torch.long)
    overfit_log = []

    for l in range(1, L):
        for h in range(Hkv):
            Fh = F[l, h].to(dev)
            muh = mu_k[l, h].to(dev)
            mqh = mu_q[l, h].to(dev)
            rs, ms = [], []
            for kk in fit_pool.k_slices(l, h):
                kg = kk.to(dev)
                rs.append((kg - muh) @ Fh)
                ms.append((kg @ mqh) / math.sqrt(d))
            r = torch.cat(rs)
            m = torch.cat(ms)
            head_m_spread[l, h] = float(m.quantile(0.9) - m.median())
            n16 = r.norm(dim=1).to(torch.float16).float()
            u = r / n16.clamp_min(1e-8).unsqueeze(1)
            coord_std_code[l, h] = r.std(0).cpu()
            coord_std_dir[l, h] = u.std(0).cpu()

            # variance-balancing permutation: coords sorted by dir-std desc,
            # dealt round-robin into RVQ_NSUB groups
            order = torch.argsort(u.std(0), descending=True).cpu()
            groups = [[] for _ in range(RVQ_NSUB)]
            for i, j in enumerate(order.tolist()):
                groups[i % RVQ_NSUB].append(j)
            perm = torch.tensor([j for grp in groups for j in grp])
            rvq_perm[l, h] = perm

            up = u[:, perm.to(dev)].reshape(-1, RVQ_NSUB, RVQ_SD) \
                .permute(1, 0, 2).contiguous()
            # held-out directions for the overfit gate
            rsel = torch.cat([(kk.to(dev) - muh) @ Fh
                              for kk in sel_pool.k_slices(l, h)])
            nsel = rsel.norm(dim=1).to(torch.float16).float()
            usel = rsel / nsel.clamp_min(1e-8).unsqueeze(1)
            upsel = usel[:, perm.to(dev)].reshape(-1, RVQ_NSUB, RVQ_SD) \
                .permute(1, 0, 2).contiguous()

            resid, resid_sel = up, upsel
            for s in range(RVQ_STAGES):
                k_entries = RVQ_K
                cent = kmeans_gpu(resid, k_entries, KMEANS_ITERS, gen)
                q_fit = quantize_rvq(resid, cent)
                q_sel = quantize_rvq(resid_sel, cent)
                d_fit = (resid - q_fit).square().mean()
                d_sel = (resid_sel - q_sel).square().mean()
                ratio = float(d_sel / d_fit.clamp_min(1e-12))
                if ratio > RVQ_OVERFIT_RATIO:
                    for k_try in (128, 64):
                        cent = kmeans_gpu(resid, k_try, KMEANS_ITERS, gen)
                        q_fit = quantize_rvq(resid, cent)
                        q_sel = quantize_rvq(resid_sel, cent)
                        ratio = float((resid_sel - q_sel).square().mean()
                                      / (resid - q_fit).square().mean()
                                      .clamp_min(1e-12))
                        k_entries = k_try
                        if ratio <= RVQ_OVERFIT_RATIO:
                            break
                    overfit_log.append([l, h, s, k_entries, ratio])
                rvq_entries[l, h, s] = k_entries
                rvq_cbs[s][l, h, :, :k_entries] = cent.cpu()
                resid = resid - q_fit
                resid_sel = resid_sel - q_sel
        print(f"[fit] layer {l}/{L - 1} done ({time.time() - t0:.0f}s)",
              flush=True)

    nd_profiles = torch.stack(
        [torch.full((d,), w, dtype=torch.long) for w in ND_WIDTHS])

    tag = "compact8train18" if args.pool == "fit18" \
        else f"compact8train{args.pool_rows}r400"
    out_path = OUT_DIR / f"pgq2_bundle__qpca_unc__{tag}.pt"
    blob = {
        "pgq_version": PGQ2_BUNDLE_VERSION,
        "model_tag": "llama31_8b", "basis": "qpca_unc", "ptok": PTOK,
        "n_layers": L, "n_kv_heads": Hkv, "head_dim": d,
        "forward": F, "inverse": Finv, "mu": mu_k, "mu_q": mu_q,
        "coord_std_code": coord_std_code, "coord_std_dir": coord_std_dir,
        "head_m_spread": head_m_spread,
        "nd_profiles": nd_profiles,
        "rvq_codebooks": rvq_cbs, "rvq_perm": rvq_perm,
        "rvq_entries": rvq_entries,
        "rvq_overfit_log": overfit_log,
        # placeholders; select_pgq2_hparams.py patches the frozen values.
        # For the pre-registered big-pool retrain the ALREADY-frozen tau/
        # theta carry over unchanged (no re-selection — the addendum only
        # changes codebook training data).
        "omega_tau_by_rate": (json.loads(FROZEN.read_text())["tau_by_rate"]
                              if args.pool == "train400" and FROZEN.exists()
                              else {}),
        "gate_theta": (json.loads(FROZEN.read_text())["theta"]
                       if args.pool == "train400" and FROZEN.exists()
                       else 0.0),
        "omega_clamp_bits": 4.0,
        "fit_rows": roles["fit"], "selection_rows": roles["selection"],
        "cca_stats_path": str(CCA_STATS),
        "cca_stats_mtime": int(CCA_STATS.stat().st_mtime),
        "kmeans": {"iters": KMEANS_ITERS, "seed": SEED,
                   "stages": RVQ_STAGES, "nsub": RVQ_NSUB, "sd": RVQ_SD},
    }
    torch.save(blob, out_path)
    print(f"[fit] bundle -> {out_path} "
          f"({out_path.stat().st_size / 1e6:.0f} MB, {time.time()-t0:.0f}s; "
          f"overfit refits: {len(overfit_log)})", flush=True)

    if args.skip_eval:
        return

    # ---- held-out gates G1/G2/G3 on the selection rows ------------------
    from kvq.compression.page_quant import load_pgq_compressors_from_bundle
    report = {}
    for mode, kb in EVAL_MODES:
        comps, _ = load_pgq_compressors_from_bundle(out_path, mode, kb)
        payload = side = toks = over = pages = 0.0
        hist = None
        sink_dm, sink_re, sink_ce, norm_ratio = [], [], [], []
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
                        sink_re.append(float(
                            (khat[:4] - kg[:4]).norm()
                            / kg[:4].norm().clamp_min(1e-9)))
                        # code-space (Q-weighted) sink relerr — the
                        # attention-relevant one; raw-k is amplified by
                        # Sigma_Q^{-1/2} directions under qpca_unc
                        Fh_ = comps[(l, h)].forward_map
                        rc0 = (kg[:4] - comps[(l, h)].mu) @ Fh_
                        rch = (khat[:4] - comps[(l, h)].mu) @ Fh_
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
            "sink_relerr_med": float(torch.tensor(sink_re).median()),
            "sink_code_relerr_med": float(torch.tensor(sink_ce).median()),
            "norm_ratio_med": float(torch.tensor(norm_ratio).median()),
        }
        r = report[key]
        print(f"[eval] {key}: rate={r['rate_heldout']:.4f} "
              f"ovf={r['overflow_frac']:.4%} "
              f"sinkΔ={r['sink_mass_shift_med']:+.4f} "
              f"sinkRE={r['sink_relerr_med']:.3f} "
              f"sinkCE={r['sink_code_relerr_med']:.3f} "
              f"normR={r['norm_ratio_med']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    save_json(OUT_DIR / "pgq2_heldout_report.json",
              {"bundle": str(out_path), "report": report})


if __name__ == "__main__":
    main()
