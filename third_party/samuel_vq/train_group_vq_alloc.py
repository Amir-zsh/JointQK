#!/usr/bin/env python3
"""Allocation-aware group-VQ trainer -- gives VQ the advantages the scalar baseline
(QPCA-Fixed) has, to test whether a *properly designed* VQ beats scalar INT2
(0.647 top-1 / 0.910 top-5) at a true 2.0 bits/coord. See
notes/entropy_coding_vq_accuracy_handoff.md.

Knobs (all fold into the SAME payload format vq_sweep_score.py reads -- no codec
or scorer change needed):
  --grouping consecutive|stratified
      stratified = rank-based coord permutation (rank r -> group r % NG) so every
      group spans the Lambda spectrum instead of a narrow band. Implemented as a
      column permutation of the QPCA basis (folds into forward/inverse), so groups
      stay contiguous in the working basis and decode-friendliness is preserved.
  --allocation flat|waterfill
      waterfill = per-coord reverse-water-filling of the QPCA score (Lambda) summed
      per group -> variable K per group (high-importance groups get more centroids),
      average held at 2.0 bits/coord. This is exactly scalar's advantage, lifted to
      groups. (Variable K per (head,group); the codec reads K from codebook shape.)
  --whiten
      divide each coord by its per-coord std before k-means (folds diag(1/std) into
      forward, diag(std) into inverse). Equalizes within-group variance.
  --G, --max-k-bits (cap so K <= 2^cap <= ~samples; avoids undertrained centroids).

G=1 + consecutive + waterfill should reproduce scalar (~0.647): it's per-coord
allocated quantization with k-means (data) centroids. If G>1 then beats it, VQ's
joint structure helps; if not, VQ is dominated on decorrelated QPCA coords.
"""
import argparse
import math
import torch

import run_pca_ec_deadzone as base
from group_vq_codec import _kmeans, group_boundaries


def stratified_perm(d, G):
    """rank r -> group (r % NG), laid out contiguous in the permuted basis.
    Returns a LongTensor perm of length d (perm[pos] = original coord/rank)."""
    NG = math.ceil(d / G)
    perm = []
    for g in range(NG):
        for j in range(G):
            r = g + j * NG
            if r < d:
                perm.append(r)
    assert sorted(perm) == list(range(d)), "not a permutation"
    return torch.tensor(perm, dtype=torch.long)


def balanced_perm(logv, G):
    """Volume-equalizing partition (flat-rate optimum of the grouping theorem): partition
    coords into groups of G that balance the group log-volumes Σ_{i∈group} log σ_i².
    Greedy least-loaded (LPT) seed + pairwise swap refinement -> hits the AM-GM bound.
    `logv`: 1-D np array of per-coord log-variances (only full groups; d must be divisible
    by G here — Qwen d=128, G=4). Returns a LongTensor perm (perm[pos] = original coord)."""
    import numpy as np
    logv = np.asarray(logv, dtype=np.float64)
    d = logv.shape[0]; L = d // G
    order = np.argsort(-logv)
    sums = np.zeros(L); members = [[] for _ in range(L)]
    for i in order:                                             # LPT seed
        l = min((l for l in range(L) if len(members[l]) < G), key=lambda l: sums[l])
        members[l].append(int(i)); sums[l] += logv[i]
    S = [logv[m].sum() for m in members]                        # swap refinement
    improved = True
    while improved:
        improved = False
        for a in range(L):
            for b in range(a + 1, L):
                base = abs(S[a] - S[b]); best = None
                for ia, ea in enumerate(members[a]):
                    for ib, eb in enumerate(members[b]):
                        nSa = S[a] - logv[ea] + logv[eb]; nSb = S[b] - logv[eb] + logv[ea]
                        if abs(nSa - nSb) < base - 1e-12:
                            base = abs(nSa - nSb); best = (ia, ib, nSa, nSb)
                if best:
                    ia, ib, nSa, nSb = best
                    members[a][ia], members[b][ib] = members[b][ib], members[a][ia]
                    S[a], S[b] = nSa, nSb; improved = True
    return torch.tensor([i for m in members for i in m], dtype=torch.long)


def waterfill_continuous(score, total):
    """reverse water-filling: bits_j = max(0, 0.5*log2(score_j/theta)), sum ~ total."""
    s = score.clamp_min(1e-30).double()
    lo = torch.tensor(1e-40, dtype=torch.float64); hi = s.max()
    for _ in range(80):
        th = (lo * hi).sqrt()
        b = (0.5 * torch.log2(s / th)).clamp_min(0)
        if b.sum() > total:
            lo = th
        else:
            hi = th
    th = (lo * hi).sqrt()
    return (0.5 * torch.log2(s / th)).clamp_min(0)


def group_bit_alloc(score_perm, bounds, avg_bits, max_k_bits):
    """Per-head integer bits per group from waterfilling the (permuted) score.
    Returns list[int] group_bits summing to avg_bits*d, each in [0, max_k_bits]."""
    d = bounds[-1][1]
    total = avg_bits * d
    bc = waterfill_continuous(score_perm, total)                 # (d,) continuous per-coord
    gb_cont = torch.tensor([float(bc[s:e].sum()) for (s, e, _) in bounds])
    # largest-remainder round to ints summing to total
    fl = torch.floor(gb_cont); rem = int(round(total - float(fl.sum())))
    frac = gb_cont - fl
    order = torch.argsort(frac, descending=True)
    gb = fl.clone()
    for i in range(max(0, rem)):
        gb[order[i % len(order)]] += 1
    gb = gb.long().clamp(0, max_k_bits)
    # redistribute any deficit/excess from the cap so total stays = avg*d
    def fix(gb):
        diff = int(total - int(gb.sum()))
        while diff > 0:                       # need to add bits: give to lowest below cap
            cand = (gb < max_k_bits).nonzero(as_tuple=True)[0]
            if len(cand) == 0: break
            j = cand[gb[cand].argmin()]; gb[j] += 1; diff -= 1
        while diff < 0:                       # remove bits: take from highest >0
            cand = (gb > 0).nonzero(as_tuple=True)[0]
            if len(cand) == 0: break
            j = cand[gb[cand].argmax()]; gb[j] -= 1; diff += 1
        return gb
    return fix(gb).tolist()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis-moments", default=None,
                    help="pooled 400-prompt basis-moment .pt for the QPCA basis+allocation. "
                         "If omitted, falls back to calib_moments on --code-idx (legacy 1-corpus mode).")
    ap.add_argument("--code-idx", type=int, nargs="+", default=list(range(24)),
                    help="EC-pool example indices for the k-means training samples.")
    ap.add_argument("--calib-idx", type=int, nargs="+", default=None,
                    help="legacy alias for --code-idx (1-corpus mode).")
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--grouping", choices=["consecutive", "stratified", "balanced"], default="consecutive")
    ap.add_argument("--allocation", choices=["flat", "waterfill"], default="flat")
    ap.add_argument("--bpc", type=int, default=2, help="target bits/coord (flat: uniform; waterfill: average)")
    ap.add_argument("--qtau", type=float, default=1.0, help="Sigma_Q whitening temperature (1=full QPCA, <1 tempers toward Sigma_K PCA)")
    ap.add_argument("--pertoken-norm", action="store_true", help="OSCAR/KIVI-style per-token RMS normalization before k-means (codebook sees unit-scale vectors)")
    ap.add_argument("--whiten", action="store_true")
    ap.add_argument("--max-k-bits", type=int, default=13)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--ecvq-lambda", type=float, default=0.0,
                    help="ECVQ rate penalty (Chou-Lookabaugh-Gray): assignment adds "
                         "lambda*(-log2 p_i) to ||x-c_i||^2. 0 = plain k-means.")
    ap.add_argument("--data-root", default=None,
                    help="override the per-example pool dir (query_stats format) for the "
                         "k-means training samples. Default: module FULL_DIR.")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    code_idx = args.calib_idx if args.calib_idx is not None else args.code_idx

    from pathlib import Path as _P
    root = _P(args.data_root) if args.data_root else base.data_root()
    man = base.load_manifest(root)
    if args.basis_moments:
        B = torch.load(args.basis_moments, map_location="cpu", weights_only=False)
        sq, sk, km, kc, meta = B["sigma_q"], B["sigma_k"], B["k_mean"], B["k_cov"], B["meta"]
        print(f"BASIS moments: {args.basis_moments} | prompts={B.get('n_prompts','?')} "
              f"ntok={B.get('ntok','?')}", flush=True)
    else:
        sq, sk, km, kc, meta = base.calib_moments(root, man, code_idx)
    L, Hkv, d = meta["n_layers"], meta["n_kv_heads"], meta["d_head"]
    qc = base.build_qpca_basis(sq, kc, tau=args.qtau); qc["sigma_k"] = sk
    F, inv, score = qc["forward"].clone(), qc["inverse"].clone(), qc["score"].clone()  # (L,Hkv,d,d)/(L,Hkv,d)
    std = qc["std"].clone()

    # ---- stratified: permute basis columns so rank-based groups are contiguous ----
    if args.grouping == "stratified":
        perm = stratified_perm(d, args.G)
        F = F[:, :, :, perm]                 # permute output coords
        inv = inv[:, :, perm, :]
        score = score[:, :, perm]
        std = std[:, :, perm]

    # ---- whiten: fold diag(1/std) into F, diag(std) into inv ----
    if args.whiten:
        F = F / std.unsqueeze(-2).clamp_min(1e-8)      # scale columns of F by 1/std
        inv = inv * std.unsqueeze(-1)                  # scale rows of inv by std
        score = torch.ones_like(score)                 # whitened -> flat score (all unit var)

    bounds = group_boundaries(d, args.G, args.bpc)
    fetch = base._codes_for_idx(root, man, code_idx, F, km, L, Hkv, d)

    print(f"cfg G={args.G} grouping={args.grouping} alloc={args.allocation} whiten={args.whiten} "
          f"| L={L} Hkv={Hkv} d={d} | out={args.out}", flush=True)

    comps_cb = {}
    bpc_acc = []
    for l in range(L):
        for h in range(Hkv):
            r = fetch(l, h).to(dev)                    # (N, d) permuted+whitened residual
            # balanced: per-head volume-equalizing permutation from this head's coefficient
            # variance spectrum, folded into F[l,h]/inv[l,h] so decode needs no extra step.
            if args.grouping == "balanced":
                perm = balanced_perm(r.var(0).log().cpu().numpy(), args.G)   # CPU LongTensor
                r = r[:, perm.to(r.device)]
                F[l, h] = F[l, h][:, perm]              # permute output (coefficient) coords
                inv[l, h] = inv[l, h][perm, :]          # permute coefficient rows of inverse
            if args.pertoken_norm:                     # OSCAR/KIVI-style per-token RMS scale
                r = r / r.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
            if args.allocation == "waterfill":
                gbits = group_bit_alloc(score[l, h], bounds, avg_bits=args.bpc, max_k_bits=args.max_k_bits)
            else:
                gbits = [b for (_, _, b) in bounds]     # flat 2*len(group)
            cbs = []
            for (s, e, _), gb in zip(bounds, gbits):
                K = 1 << gb
                if K <= 1:
                    cbs.append(r[:, s:e].mean(0, keepdim=True))     # dropped group -> single mean centroid
                else:
                    cbs.append(_kmeans(r[:, s:e], K, iters=args.iters, seed=l * Hkv + h,
                                       ecvq_lambda=args.ecvq_lambda))
            comps_cb[(l, h)] = [c.cpu() for c in cbs]
            bpc_acc.append(sum(gbits) / d)
        if l % 6 == 0:
            print(f"  layer {l}/{L} (avg bits/coord so far {sum(bpc_acc)/len(bpc_acc):.3f})", flush=True)

    payload = dict(forward=F.cpu(), inverse=inv.cpu(), mean=km.cpu(), codebooks=comps_cb,
                   bounds=bounds, G=args.G, grouping=args.grouping, allocation=args.allocation,
                   whiten=args.whiten, ecvq_lambda=args.ecvq_lambda,
                   pertoken_norm=args.pertoken_norm,
                   bits_per_coord=float(sum(bpc_acc) / len(bpc_acc)))
    torch.save(payload, args.out)
    print(f"SAVED {args.out} | real avg bits/coord = {payload['bits_per_coord']:.4f}", flush=True)
