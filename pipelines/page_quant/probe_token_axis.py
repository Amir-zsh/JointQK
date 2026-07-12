"""pgq8 pre-study probe: token-axis correlation / transform-compaction study.

Question: would a page-level transform ALONG THE TOKEN AXIS (mix the 64
tokens of a page before quantizing — DCT/Haar/KLT rows instead of token rows)
buy anything over per-token coding? This probe measures, in the qpca_unc code
space (where SE == Q-weighted logit error):

  A. token-axis second-moment structure per (layer, head): pooled 64x64
     page covariance C -> energy spectra of identity / DCT / Haar /
     pooled-KLT coefficient rows, top-k energy fractions, AM/GM coding
     gains, high-rate bits-saved estimates 0.5*log2(G_T / G_id);
  B. per-page KLT oracle (sampled pages) — upper bound for ANY fixed T;
  C. token-lag correlations (lags 1..16) + per-coord lag-1 profile
     (leading vs tail qpca coords);
  D. stationarity: DCT-vs-pooled-KLT gain gap (DCT is the KLT of a
     stationary process) + Toeplitz-ness of C;
  E. RoPE arm (subset of rows): raw-key-space lag correlations and DCT
     compaction for post-RoPE keys vs analytically de-rotated (pre-RoPE)
     keys — how much token-axis correlation does RoPE scramble?
  F. empirical matched-rate quantization sim on sampled pages: identity vs
     DCT vs pooled-KLT rows under the SAME allocator (per-(row,block)
     lambda-bisection over widths {0,2,3,4,6}) and the SAME grids (bundle
     LM cents + covering uniform top), budgets b in {1.0, 1.5, 2.0} b/c.
     This is the direct predictor of a pgq8 screening outcome.

Fit-pool rows only (bundle selection rows excluded — selection stays unburned
for any later pgq8 screening). Sinks (first 4 tokens) dropped; layer 0
excluded per project convention.

Usage:
  python -u pipelines/page_quant/probe_token_axis.py --model-tag llama31_8b --gpu 0
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]

MODELS = {
    "llama31_8b": {
        "bundle": "artifacts/page_quant2/pgq4_bundle__3bases__compact8train40r400.pt",
        "raw_root": "artifacts/calibration/ec_calib_compact8_train_llama31_8b/01_raw",
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
    },
    "qwen3_8b": {
        "bundle": "artifacts/page_quant2/pgq5_bundle__qpca_unc__qwen3_8b_compact8train12.pt",
        "raw_root": "artifacts/calibration/longbench_compact8_qkv_qwen3_8b/01_raw",
        "hf_id": "Qwen/Qwen3-8B",
    },
}
PTOK = 64
NSINK = 4
BLOCK = 32
LAGS = (1, 2, 4, 8, 16)
WIDTHS = (0, 2, 3, 4, 6)
SIM_BUDGETS = (1.0, 1.5, 2.0)
SIM_PAGES_PER_CELL = 24
ROPE_ROWS = 4


def dct_matrix(n: int) -> torch.Tensor:
    t = torch.arange(n).float()
    k = t.unsqueeze(1)
    m = torch.cos(math.pi * (2 * t.unsqueeze(0) + 1) * k / (2 * n))
    m = m * math.sqrt(2.0 / n)
    m[0] *= 1.0 / math.sqrt(2.0)
    return m


def haar_matrix(n: int) -> torch.Tensor:
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        m = h.shape[0]
        top = torch.kron(h, torch.ones(1, 2))
        bot = torch.kron(torch.eye(m), torch.tensor([[1.0, -1.0]]))
        h = torch.cat([top, bot])
    return h / h.norm(dim=1, keepdim=True)


def rope_cos_sin(hf_id: str, T: int, d: int, device):
    from transformers import AutoConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    cfg = AutoConfig.from_pretrained(hf_id)
    rt = "default"
    if getattr(cfg, "rope_scaling", None):
        rt = cfg.rope_scaling.get("rope_type", cfg.rope_scaling.get("type", "default"))
    inv_freq, scaling = ROPE_INIT_FUNCTIONS[rt](cfg, device)
    pos = torch.arange(T, device=device, dtype=torch.float32)
    ang = pos.unsqueeze(1) * inv_freq.unsqueeze(0)          # (T, d/2)
    emb = torch.cat([ang, ang], dim=-1)                     # (T, d)
    return (emb.cos() * scaling), (emb.sin() * scaling)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def derotate(k, cos, sin):
    """Inverse RoPE: k_pre = k*cos - rotate_half(k)*sin (rotation by -theta)."""
    return k * cos - rotate_half(k) * sin


def amgm(e: torch.Tensor) -> float:
    e = e.clamp_min(1e-30)
    return float(e.mean() / e.log().mean().exp())


def topk_frac(e: torch.Tensor, ks=(1, 2, 4, 8, 16, 32)) -> dict:
    s = e.sort(descending=True).values
    tot = float(s.sum())
    return {str(k): float(s[:k].sum()) / tot for k in ks}


def grid_tables(bundle, device):
    """Unit grids per width + MC distortion factors g(w) (relative MSE on N(0,1))."""
    cents = {w: c.float().to(device) for w, c in zip((2, 3, 4), bundle["lm_cents"])}
    lim6 = (1 << 5) - 1
    step6 = 4.0 / lim6                                       # covering +-4 sigma
    g = {0: 1.0}
    x = torch.randn(2_000_000, device=device)
    for w in (2, 3, 4):
        c = cents[w]
        q = c[torch.bucketize(x, (c[1:] + c[:-1]) / 2)]
        g[w] = float((x - q).square().mean())
    q6 = torch.round(x / step6).clamp(-lim6, lim6) * step6
    g[6] = float((x - q6).square().mean())
    return cents, step6, lim6, g


def quantize_units(Y, sig, assign, cents, step6, lim6):
    """Y (B,64,d) coeff rows; sig (64,d) scale map; assign (B,64,nblk) width
    ids into WIDTHS. Returns real SE per page (B,)."""
    B, n, d = Y.shape
    u = Y / sig.clamp_min(1e-12)
    out = torch.zeros_like(Y)
    wmask = torch.tensor(WIDTHS, device=Y.device)[assign]      # (B,64,nblk)
    wfull = wmask.repeat_interleave(BLOCK, dim=2)              # (B,64,d)
    for w in (2, 3, 4):
        m = wfull == w
        if m.any():
            c = cents[w]
            q = c[torch.bucketize(u.clamp(c[0], c[-1]), (c[1:] + c[:-1]) / 2)]
            out[m] = q[m]
    m6 = wfull == 6
    if m6.any():
        out[m6] = (torch.round(u / step6).clamp(-lim6, lim6) * step6)[m6]
    yhat = out * sig
    return (Y - yhat).square().sum((1, 2))


def sim_alloc_and_quantize(pages, sig, g, cents, step6, lim6, budget_bits):
    """pages (B,64,d) in some row basis; per-page lambda-bisection over
    (64 x nblk) units, widths WIDTHS, rate 32*w bits; then real SE."""
    B, n, d = pages.shape
    nblk = d // BLOCK
    E = pages.reshape(B, n, nblk, BLOCK).square().sum(-1)      # (B,64,nblk)
    gvec = torch.tensor([g[w] for w in WIDTHS], device=pages.device)
    R = torch.tensor([32.0 * w for w in WIDTHS], device=pages.device)
    D = E.unsqueeze(-1) * gvec                                  # (B,64,nblk,W)
    lo = torch.zeros(B, 1, 1, 1, device=pages.device)
    hi = torch.full_like(lo, 1e6)
    for _ in range(40):
        mid = (lo + hi) / 2
        a = (D + mid * R).argmin(-1)
        rr = R[a].sum((1, 2))
        over = (rr > budget_bits).view(B, 1, 1, 1)
        lo = torch.where(over, mid, lo)
        hi = torch.where(over, hi, mid)
    a = (D + hi * R).argmin(-1)
    se = quantize_units(pages, sig, a, cents, step6, lim6)
    rate = R[a].sum((1, 2))
    return float(se.sum()), float(rate.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", required=True, choices=list(MODELS))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()
    spec = MODELS[args.model_tag]
    dev = torch.device(f"cuda:{args.gpu}")
    hb = REPO / "logs" / f"probe8_token_axis_{args.model_tag}.heartbeat"
    hb.parent.mkdir(exist_ok=True)

    blob = torch.load(REPO / spec["bundle"], map_location="cpu", weights_only=False)
    L, H, d = blob["n_layers"], blob["n_kv_heads"], blob["head_dim"]
    F = blob["bases"]["qpca_unc"]["forward"].to(dev).float()   # (L,H,d,d)
    MU = blob["mu"].to(dev).float()                            # (L,H,d)
    sel = {tuple(x) for x in map(tuple, blob["selection_rows"])}

    raws = sorted((REPO / spec["raw_root"]).glob("shard_*/*.pt"))
    rows = []
    for p in raws:
        m = re.match(r"longbench__(.+)__row(\d+)__", p.name)
        if m and (m.group(1), int(m.group(2))) not in sel:
            rows.append((m.group(1), int(m.group(2)), p))
    seen = set()
    rows = [r for r in rows if not (r[:2] in seen or seen.add(r[:2]))]
    if args.max_rows:
        rows = rows[: args.max_rows]
    print(f"[probe8] {args.model_tag}: {len(rows)} fit rows, {L}x{H} cells, gpu {args.gpu}",
          flush=True)

    Dm = dct_matrix(PTOK).to(dev)
    Hm = haar_matrix(PTOK).to(dev)
    ncell = (L - 1) * H
    cid = lambda l, h: (l - 1) * H + h

    C = torch.zeros(ncell, PTOK, PTOK, dtype=torch.float64, device=dev)
    Vid = torch.zeros(ncell, PTOK, d, device=dev)
    Vdct = torch.zeros(ncell, PTOK, d, device=dev)
    lag_num = torch.zeros(ncell, len(LAGS), dtype=torch.float64, device=dev)
    lag_den = torch.zeros(ncell, dtype=torch.float64, device=dev)
    c1_num = torch.zeros(ncell, d, dtype=torch.float64, device=dev)
    c1_den = torch.zeros(ncell, d, dtype=torch.float64, device=dev)
    npages = torch.zeros(ncell, dtype=torch.long, device=dev)
    task_C = {}
    task_lag = {}
    samples = [[] for _ in range(ncell)]
    rope_stats = {"post": {"C": torch.zeros(PTOK, PTOK, dtype=torch.float64, device=dev),
                           "lag": torch.zeros(len(LAGS), dtype=torch.float64, device=dev),
                           "den": torch.zeros((), dtype=torch.float64, device=dev)},
                  "pre": {"C": torch.zeros(PTOK, PTOK, dtype=torch.float64, device=dev),
                          "lag": torch.zeros(len(LAGS), dtype=torch.float64, device=dev),
                          "den": torch.zeros((), dtype=torch.float64, device=dev)}}

    t0 = time.time()
    for ri, (cfgname, ridx, path) in enumerate(rows):
        art = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        T = int(art["prompt_length"])
        k_all = art["k_post"][:, :, :T].to(dev).float()        # (L,H,T,d)
        if cfgname not in task_C:
            task_C[cfgname] = torch.zeros(PTOK, PTOK, dtype=torch.float64, device=dev)
            task_lag[cfgname] = torch.zeros(2, dtype=torch.float64, device=dev)
        do_rope = ri < ROPE_ROWS
        if do_rope:
            cos, sin = rope_cos_sin(spec["hf_id"], T, d, dev)
        for l in range(1, L):
            for h in range(H):
                i = cid(l, h)
                r = (k_all[l, h] - MU[l, h]) @ F[l, h]
                r = r[NSINK:]
                Tp = r.shape[0]
                P = Tp // PTOK
                if P == 0:
                    continue
                pg = r[: P * PTOK].reshape(P, PTOK, d)
                C[i] += torch.einsum("ptd,pud->tu", pg, pg).double()
                Vid[i] += pg.square().sum(0)
                yd = torch.einsum("st,ptd->psd", Dm, pg)
                Vdct[i] += yd.square().sum(0)
                npages[i] += P
                for j, lag in enumerate(LAGS):
                    lag_num[i, j] += float((r[:-lag] * r[lag:]).sum())
                lag_den[i] += float(r.square().sum())
                c1_num[i] += (r[:-1] * r[1:]).sum(0).double()
                c1_den[i] += r.square().sum(0).double()
                task_C[cfgname] += torch.einsum("ptd,pud->tu", pg, pg).double()
                task_lag[cfgname] += torch.tensor(
                    [float((r[:-1] * r[1:]).sum()), float(r.square().sum())],
                    dtype=torch.float64, device=dev)
                if len(samples[i]) < SIM_PAGES_PER_CELL:
                    take = min(SIM_PAGES_PER_CELL - len(samples[i]), P)
                    step = max(1, P // take)
                    samples[i].extend(pg[::step][:take].half().cpu())
                if do_rope:
                    kk = k_all[l, h, NSINK:]
                    kpre = derotate(kk, cos[NSINK:], sin[NSINK:])
                    for name, kv in (("post", kk), ("pre", kpre)):
                        Pv = kv[: P * PTOK].reshape(P, PTOK, d)
                        rope_stats[name]["C"] += torch.einsum(
                            "ptd,pud->tu", Pv, Pv).double()
                        for j, lag in enumerate(LAGS):
                            rope_stats[name]["lag"][j] += float(
                                (kv[:-lag] * kv[lag:]).sum())
                        rope_stats[name]["den"] += float(kv.square().sum())
        del k_all
        hb.touch()
        print(f"[probe8] row {ri+1}/{len(rows)} ({cfgname},{ridx}) T={T} "
              f"elapsed {time.time()-t0:.0f}s", flush=True)

    # ---- pass B: derived metrics -----------------------------------------
    print("[probe8] pass B: spectra + gains", flush=True)
    res_cells = {"gain_dct": [], "gain_haar": [], "gain_klt": [], "gain_id": [],
                 "gain_klt_page": [], "top_dct": [], "top_klt": [], "top_id": [],
                 "dc_frac": [], "lags": [], "c1_top16": [], "c1_tail": [],
                 "toeplitz_cv": []}
    Hd = Hm.double()
    Dd = Dm.double()
    for i in range(ncell):
        Ci = C[i] / max(1, int(npages[i]))
        e_id = Ci.diagonal().clone()
        e_dct = (Dd @ Ci @ Dd.T).diagonal()
        e_haar = (Hd @ Ci @ Hd.T).diagonal()
        e_klt = torch.linalg.eigvalsh(Ci).flip(0)
        res_cells["gain_id"].append(amgm(e_id))
        res_cells["gain_dct"].append(amgm(e_dct))
        res_cells["gain_haar"].append(amgm(e_haar))
        res_cells["gain_klt"].append(amgm(e_klt))
        res_cells["top_dct"].append(topk_frac(e_dct))
        res_cells["top_klt"].append(topk_frac(e_klt))
        res_cells["top_id"].append(topk_frac(e_id))
        res_cells["dc_frac"].append(float(e_dct[0] / e_dct.sum()))
        res_cells["lags"].append((lag_num[i] / lag_den[i].clamp_min(1e-30)).tolist())
        cc = (c1_num[i] / c1_den[i].clamp_min(1e-30))
        res_cells["c1_top16"].append(float(cc[:16].mean()))
        res_cells["c1_tail"].append(float(cc[16:].mean()))
        offd = [float(Ci.diagonal(o).mean()) for o in range(1, 8)]
        offsd = [float(Ci.diagonal(o).std()) for o in range(1, 8)]
        res_cells["toeplitz_cv"].append(
            sum(s / max(abs(m), 1e-30) for m, s in zip(offd, offsd)) / 7)
        if len(samples[i]):
            pgs = torch.stack(samples[i]).to(dev).float()
            sv = torch.linalg.svdvals(pgs)
            res_cells["gain_klt_page"].append(
                sum(amgm(s.square()) for s in sv) / len(sv))
        else:
            res_cells["gain_klt_page"].append(float("nan"))

    # ---- pass C: matched-rate quantization sim ---------------------------
    print("[probe8] pass C: matched-rate sim", flush=True)
    cents, step6, lim6, g = grid_tables(blob, dev)
    sim = {f"{b:g}": {"id": [0.0, 0.0], "dct": [0.0, 0.0], "klt": [0.0, 0.0]}
           for b in SIM_BUDGETS}
    for i in range(ncell):
        if not len(samples[i]):
            continue
        pgs = torch.stack(samples[i]).to(dev).float()          # (B,64,d)
        Ci = (C[i] / max(1, int(npages[i]))).float()
        klt_T = torch.linalg.eigh(Ci.double()).eigenvectors.flip(1).T.float()
        arms = {
            "id": (pgs, (Vid[i] / npages[i].clamp_min(1)).sqrt()),
            "dct": (torch.einsum("st,btd->bsd", Dm, pgs),
                    (Vdct[i] / npages[i].clamp_min(1)).sqrt()),
        }
        yk = torch.einsum("st,btd->bsd", klt_T, pgs)
        arms["klt"] = (yk, yk.square().mean(0).sqrt())
        for b in SIM_BUDGETS:
            budget = b * d * PTOK
            for name, (Y, sig) in arms.items():
                se, _ = sim_alloc_and_quantize(Y, sig, g, cents, step6, lim6,
                                               budget)
                sim[f"{b:g}"][name][0] += se
                sim[f"{b:g}"][name][1] += Y.shape[0]
        if (i + 1) % 40 == 0:
            hb.touch()
            print(f"[probe8]   sim cell {i+1}/{ncell}", flush=True)

    # ---- aggregate + write ------------------------------------------------
    def mean(xs):
        xs = [x for x in xs if x == x]
        return sum(xs) / max(1, len(xs))

    def bits_saved(gt, gi):
        return 0.5 * math.log2(max(gt, 1e-30) / max(gi, 1e-30))

    per_layer = {}
    for name in ("gain_dct", "gain_klt", "gain_id", "dc_frac"):
        per_layer[name] = [
            mean(res_cells[name][cid(l, 0):cid(l, 0) + H]) for l in range(1, L)]
    lag_mean = [mean([c[j] for c in res_cells["lags"]]) for j in range(len(LAGS))]

    rope_out = {}
    for name, st in rope_stats.items():
        Cn = st["C"] / st["C"].diagonal().sum().clamp_min(1e-30) * PTOK
        e_dct = (Dd @ Cn @ Dd.T).diagonal()
        rope_out[name] = {
            "lag_corr": (st["lag"] / st["den"].clamp_min(1e-30)).tolist(),
            "dct_gain": amgm(e_dct),
            "dct_top16": topk_frac(e_dct)["16"],
        }

    tasks_out = {}
    for t, Ct in task_C.items():
        Cn = Ct / Ct.diagonal().sum().clamp_min(1e-30) * PTOK
        e_dct = (Dd @ Cn @ Dd.T).diagonal()
        tasks_out[t] = {
            "lag1": float(task_lag[t][0] / task_lag[t][1].clamp_min(1e-30)),
            "dct_gain": amgm(e_dct),
            "dct_top16": topk_frac(e_dct)["16"],
            "bits_saved_vs_id": bits_saved(amgm(e_dct), amgm(Cn.diagonal())),
        }

    sim_out = {}
    for b, arms in sim.items():
        sid = arms["id"][0]
        sim_out[b] = {name: {"se_ratio_vs_id": v[0] / max(sid, 1e-30),
                             "pages": v[1]} for name, v in arms.items()}

    g_id, g_dct, g_klt = (mean(res_cells["gain_id"]), mean(res_cells["gain_dct"]),
                          mean(res_cells["gain_klt"]))
    g_pp = mean(res_cells["gain_klt_page"])
    out = {
        "model": args.model_tag, "rows_used": [(c, r) for c, r, _ in rows],
        "n_cells": ncell, "pages_total": int(npages.sum()),
        "ptok": PTOK, "nsink_dropped": NSINK, "layer0_excluded": True,
        "headline": {
            "lag_corr": dict(zip(map(str, LAGS), lag_mean)),
            "c1_top16_coords": mean(res_cells["c1_top16"]),
            "c1_tail_coords": mean(res_cells["c1_tail"]),
            "gain_id": g_id, "gain_dct": g_dct, "gain_haar":
                mean(res_cells["gain_haar"]), "gain_klt_pooled": g_klt,
            "gain_klt_perpage_oracle": g_pp,
            "bits_saved_dct_vs_id": bits_saved(g_dct, g_id),
            "bits_saved_klt_vs_id": bits_saved(g_klt, g_id),
            "bits_saved_perpage_oracle_vs_id": bits_saved(g_pp, g_id),
            "dct_dc_energy_frac": mean(res_cells["dc_frac"]),
            "dct_top16_frac": mean([t["16"] for t in res_cells["top_dct"]]),
            "klt_top16_frac": mean([t["16"] for t in res_cells["top_klt"]]),
            "id_top16_frac": mean([t["16"] for t in res_cells["top_id"]]),
            "stationarity_gap": (g_klt - g_dct) / max(g_klt - 1.0, 1e-30),
            "toeplitz_cv": mean(res_cells["toeplitz_cv"]),
        },
        "per_layer": per_layer,
        "rope_arm": {"rows": min(ROPE_ROWS, len(rows)), **rope_out},
        "per_task": tasks_out,
        "sim_matched_rate": sim_out,
        "cells": {k: res_cells[k] for k in
                  ("gain_dct", "gain_klt", "gain_klt_page", "dc_frac")},
    }
    dst = REPO / "artifacts/page_quant2" / f"token_axis_probe_{args.model_tag}.json"
    dst.write_text(json.dumps(out, indent=1))
    print(f"[probe8] wrote {dst}", flush=True)
    hl = out["headline"]
    print(f"[probe8] HEADLINE {args.model_tag}: lag1={hl['lag_corr']['1']:.4f} "
          f"G_dct={hl['gain_dct']:.3f} G_klt={hl['gain_klt_pooled']:.3f} "
          f"G_oracle={hl['gain_klt_perpage_oracle']:.3f} "
          f"bits_saved_dct={hl['bits_saved_dct_vs_id']:.4f} "
          f"sim@1.5 dct={sim_out['1.5']['dct']['se_ratio_vs_id']:.4f} "
          f"klt={sim_out['1.5']['klt']['se_ratio_vs_id']:.4f}", flush=True)


if __name__ == "__main__":
    main()
