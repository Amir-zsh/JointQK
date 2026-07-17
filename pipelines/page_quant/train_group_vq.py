#!/usr/bin/env python3
"""Train group-VQ codebooks (Samuel's method, ported in kvq/compression/group_vq.py)
from OUR calibration pool + OUR pgq bundle's qpca_unc basis.

Pipeline per (layer, kv_head): residual r = (k - mu) @ F_perm where F_perm is
the bundle's qpca_unc forward map with the stratified rank permutation folded
into its columns; per-group k-means (G coords/group, K_g = 2^{b_g}); fp8-e4m3
centroids. Allocation: flat (2.0 b/coord exact) or reverse-water-filling of
the code-space variance (code_std^2) over groups.

Fit/held-out split mirrors the source bundle (pgq8_fit_rows / selection_rows).
The gate compares held-out K-space relMSE against Samuel's snapshot codebook
(third_party/samuel_vq/codebooks/) — PASS if ours <= his * (1 + tol).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from kvq.compression.group_vq import (  # noqa: E402
    GroupVQCompressor, _kmeans, group_boundaries, group_bit_alloc,
    stratified_perm,
)

MODELS = {
    "qwen3_8b": {
        "bundle": "artifacts/page_quant2/pgq8_bundle__qwen3_8b.pt",
        "raw_root": "artifacts/calibration/longbench_compact8_qkv_qwen3_8b/01_raw",
        "ref_codebook": "third_party/samuel_vq/codebooks/vqa_G4_strat_flat_fair_fp8.pt",
    },
    "llama31_8b": {
        "bundle": "artifacts/page_quant2/pgq8_bundle__llama31_8b.pt",
        "raw_root": "artifacts/calibration/ec_calib_compact8_train_llama31_8b/01_raw",
        "ref_codebook": "third_party/samuel_vq/codebooks/vqa_G4_strat_flat_llama_fp8.pt",
    },
}


def sha8(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def load_rows(raw_root: Path, want_rows):
    """Map (task, row_index) -> shard path. want_rows entries may be list/tuple."""
    want = {tuple(r) for r in want_rows}
    hits = {}
    for p in sorted(raw_root.glob("shard_*/*.pt")):
        d = torch.load(p, map_location="cpu", weights_only=False)
        key = (d["task"], int(d["row_index"]))
        if key in want:
            hits[key] = (p, d["k_post"])
        del d
    missing = want - set(hits)
    if missing:
        raise FileNotFoundError(f"calibration rows not found: {sorted(missing)}")
    return [hits[tuple(r)] for r in want_rows]


def subsample(x: torch.Tensor, n: int, gen: torch.Generator) -> torch.Tensor:
    if x.shape[0] <= n:
        return x
    idx = torch.randperm(x.shape[0], generator=gen)[:n]
    return x[idx]


def train_dvq(args, spec, blob):
    """pgq10 D6: DVQ_LADDER codebooks on dct_std-normalized DCT coefficient
    rows (transform-eligible interior pages only), one codebook set per
    (l, h). Output: pgq8 bundle + dvq_codebooks (self-contained pgq10 bundle)."""
    from kvq.compression.pgq8_dct import DVQ_LADDER, dct_matrix

    L, H, d = blob["n_layers"], blob["n_kv_heads"], blob["head_dim"]
    ptok = int(blob["ptok"])
    F = blob["bases"]["qpca_unc"]["forward"]
    mu = blob["mu"]
    dct_std = blob["dct_std"].float().clamp_min(1e-12)
    dct_m = dct_matrix(ptok)
    dev = args.device
    raw_root = REPO / spec["raw_root"]
    fit = load_rows(raw_root, blob["pgq8_fit_rows"])
    t0 = time.time()
    print(f"[dvqtrain] {args.model_tag} L={L} H={H} ladder={DVQ_LADDER}",
          flush=True)
    dm = dct_m.to(dev)
    dvq = {}
    for l in range(L):
        Fl = F[l].to(dev).float()
        mul = mu[l].to(dev).float()
        for h in range(H):
            rows = []
            for _p, kp in fit:
                r = (kp[l, h].float().to(dev) - mul[h]) @ Fl[h]
                nfull = r.shape[0] // ptok
                if nfull < 6:
                    continue
                pages = r[: nfull * ptok].reshape(nfull, ptok, d)
                pages = pages[1: nfull - 4]          # skip sink + rw analog
                y = torch.einsum("st,ptd->psd", dm, pages)
                rows.append((y / dct_std[l, h].to(dev)).reshape(-1, d))
            yn = torch.cat(rows)
            sets = {}
            for bpc, g, K in DVQ_LADDER[1:]:
                ng = d // g
                cbs = [_kmeans(yn[:, gi * g:(gi + 1) * g], K,
                               iters=args.iters,
                               seed=args.seed + l * H + h + bpc * 100000)
                       for gi in range(ng)]
                sets[bpc] = torch.stack(cbs).half().cpu()
            dvq[(l, h)] = sets
        if l % 4 == 0 or l == L - 1:
            print(f"[dvqtrain] layer {l}/{L} ({time.time()-t0:.0f}s)",
                  flush=True)

    out_blob = dict(blob)
    out_blob["dvq_codebooks"] = dvq
    out_blob["dvq_provenance"] = dict(
        ladder=[list(x) for x in DVQ_LADDER], seed=args.seed,
        iters=args.iters, source_bundle=str(spec["bundle"]),
        trained_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_blob, out)
    print(f"[dvqtrain] SAVED {out} sha8={sha8(out)} "
          f"({time.time()-t0:.0f}s total)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", required=True, choices=list(MODELS))
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--bpc", type=int, default=2)
    ap.add_argument("--allocation", choices=["flat", "waterfill"], default="flat")
    ap.add_argument("--max-k-bits", type=int, default=13)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--ecvq-lambda", type=float, default=0.0)
    ap.add_argument("--pertoken-norm", action="store_true",
                    help="Samuel 3c65507: train on per-token RMS-normalized "
                         "residuals; codec renormalizes at decode (~+0.06 b/c "
                         "for the stored fp8 scale, not charged here)")
    ap.add_argument("--tokens-per-row", type=int, default=5500,
                    help="k-means samples subsampled per calibration row per (l,h)")
    ap.add_argument("--gate-tokens-per-row", type=int, default=2000)
    ap.add_argument("--gate-tol", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dvq", action="store_true",
                    help="train D6 DCT-row VQ codebooks instead (pgq10 bundle)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = MODELS[args.model_tag]
    bundle_path = REPO / spec["bundle"]
    blob = torch.load(bundle_path, map_location="cpu", weights_only=False)
    if args.dvq:
        train_dvq(args, spec, blob)
        return
    L, H, d = blob["n_layers"], blob["n_kv_heads"], blob["head_dim"]
    F = blob["bases"]["qpca_unc"]["forward"].clone()      # (L,H,d,d)
    inv = blob["bases"]["qpca_unc"]["inverse"].clone()
    mu = blob["mu"].clone()
    score = blob["stats"]["qpca_unc"]["code_std"].double() ** 2

    perm = stratified_perm(d, args.G)
    F = F[:, :, :, perm]
    inv = inv[:, :, perm, :]
    score = score[:, :, perm]

    bounds = group_boundaries(d, args.G, args.bpc)
    dev = args.device

    raw_root = REPO / spec["raw_root"]
    fit_rows = blob["pgq8_fit_rows"]
    holdout_rows = blob["selection_rows"]
    t0 = time.time()
    print(f"[vqtrain] model={args.model_tag} L={L} H={H} d={d} G={args.G} "
          f"alloc={args.allocation} | fit rows={len(fit_rows)} "
          f"holdout rows={len(holdout_rows)}", flush=True)
    fit = load_rows(raw_root, fit_rows)          # list of (path, k_post (L,H,T,d))
    hold = load_rows(raw_root, holdout_rows)
    print(f"[vqtrain] loaded {len(fit)}+{len(hold)} rows in {time.time()-t0:.0f}s",
          flush=True)

    gen = torch.Generator().manual_seed(args.seed)
    comps_cb, gbits_all = {}, {}
    bpc_acc = []
    for l in range(L):
        Fl = F[l].to(dev).float()
        mul = mu[l].to(dev).float()
        for h in range(H):
            r_parts = [subsample(kp[l, h].float(), args.tokens_per_row, gen)
                       for (_p, kp) in fit]
            k_lh = torch.cat(r_parts).to(dev)
            r = (k_lh - mul[h]) @ Fl[h]
            if args.pertoken_norm:
                r = r / r.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
            if args.allocation == "waterfill":
                gbits = group_bit_alloc(score[l, h], bounds,
                                        avg_bits=args.bpc,
                                        max_k_bits=args.max_k_bits)
            else:
                gbits = [b for (_s, _e, b) in bounds]
            cbs = []
            for (s, e, _b), gb in zip(bounds, gbits):
                K = 1 << gb
                if K <= 1:
                    cbs.append(r[:, s:e].mean(0, keepdim=True))
                else:
                    cbs.append(_kmeans(r[:, s:e], K, iters=args.iters,
                                       seed=args.seed + l * H + h,
                                       ecvq_lambda=args.ecvq_lambda))
            comps_cb[(l, h)] = [c.to(torch.float8_e4m3fn).cpu() for c in cbs]
            gbits_all[(l, h)] = gbits
            bpc_acc.append(sum(gbits) / d)
        if l % 4 == 0 or l == L - 1:
            print(f"[vqtrain] layer {l}/{L} done "
                  f"(avg b/coord {sum(bpc_acc)/len(bpc_acc):.4f}, "
                  f"{time.time()-t0:.0f}s)", flush=True)

    payload = dict(
        forward=F.cpu(), inverse=inv.cpu(), mean=mu.cpu(),
        codebooks=comps_cb, bounds=bounds, G=args.G,
        grouping="stratified", allocation="waterfill" if args.allocation == "waterfill" else "flat",
        whiten=False, ecvq_lambda=args.ecvq_lambda,
        pertoken_norm=bool(args.pertoken_norm),
        bits_per_coord=float(sum(bpc_acc) / len(bpc_acc)),
        group_bits={f"{l},{h}": g for (l, h), g in gbits_all.items()},
        provenance=dict(
            source_bundle=str(spec["bundle"]), source_bundle_sha8=sha8(bundle_path),
            raw_root=str(spec["raw_root"]),
            fit_rows=[list(r) for r in fit_rows],
            holdout_rows=[list(r) for r in holdout_rows],
            seed=args.seed, iters=args.iters,
            tokens_per_row=args.tokens_per_row,
            trained_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        ),
    )

    # ---- held-out gate vs Samuel's reference codebook -----------------------
    ref_path = REPO / spec["ref_codebook"]
    ref = torch.load(ref_path, map_location="cpu", weights_only=False)
    gen_g = torch.Generator().manual_seed(args.seed + 1)
    rel_ours, rel_ref = [], []
    for l in range(1, L):                        # layer-0 excluded (headline convention)
        for h in range(H):
            ours = GroupVQCompressor(F[l, h], inv[l, h], mu[l, h],
                                     [c for c in comps_cb[(l, h)]], bounds,
                                     pertoken_norm=bool(args.pertoken_norm)).to(dev)
            his = GroupVQCompressor(ref["forward"][l, h], ref["inverse"][l, h],
                                    ref["mean"][l, h],
                                    list(ref["codebooks"][(l, h)]),
                                    ref["bounds"],
                                    pertoken_norm=bool(ref.get("pertoken_norm", False))).to(dev)
            ks = torch.cat([subsample(kp[l, h].float(), args.gate_tokens_per_row, gen_g)
                            for (_p, kp) in hold]).to(dev)
            den = ks.pow(2).sum().clamp_min(1e-12)
            rel_ours.append(float((ours.roundtrip(ks) - ks).pow(2).sum() / den))
            rel_ref.append(float((his.roundtrip(ks) - ks).pow(2).sum() / den))
        if l % 8 == 0:
            print(f"[vqtrain] gate layer {l}: ours {sum(rel_ours)/len(rel_ours):.5f} "
                  f"ref {sum(rel_ref)/len(rel_ref):.5f}", flush=True)
    m_ours = sum(rel_ours) / len(rel_ours)
    m_ref = sum(rel_ref) / len(rel_ref)
    ratio = m_ours / m_ref
    gate_pass = bool(ratio <= 1.0 + args.gate_tol)
    payload["gate"] = dict(relmse_ours=m_ours, relmse_ref=m_ref, ratio=ratio,
                           tol=args.gate_tol, ref_codebook=str(spec["ref_codebook"]),
                           n_cells=len(rel_ours), gate_pass=gate_pass)
    print(f"[vqtrain] GATE held-out relMSE ours={m_ours:.5f} ref={m_ref:.5f} "
          f"ratio={ratio:.4f} (tol +{args.gate_tol:.0%}) -> "
          f"{'PASS' if gate_pass else 'FAIL'}", flush=True)

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"[vqtrain] SAVED {out} sha8={sha8(out)} "
          f"b/coord={payload['bits_per_coord']:.4f} "
          f"({time.time()-t0:.0f}s total)", flush=True)
    (out.parent / (out.stem + "_gate.json")).write_text(
        json.dumps(payload["gate"], indent=2))
    if not gate_pass:
        sys.exit(3)


if __name__ == "__main__":
    main()
