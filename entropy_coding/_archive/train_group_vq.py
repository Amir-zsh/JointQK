#!/usr/bin/env python3
"""Train real group-VQ codebooks (see group_vq_codec.py) on the settled 8234-token
calibration corpus (calib_idx=[0,5,6], matched to OSCAR's ~8k-token default) and
cache the result for reuse by the accuracy harness (A2/A3/A4) and, later, the
timing harness (B2).

Uses the same "centered QPCA" basis (qc) as the Paged/rANS/Exp-Golomb methods in
test_codec_on_data.py, so VQ sits on the same basis as the entropy-coded methods
it's being compared against (per report's "same centered QPCA basis" convention).
"""
import argparse
import time
import torch

import run_pca_ec_deadzone as base
from group_vq_codec import train_group_vq_compressors

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-idx", type=int, nargs="+", default=[0, 5, 6])
    ap.add_argument("--G", type=int, default=6)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--out", type=str, default="group_vq_b2_calib056.pt")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    root = base.data_root()
    man = base.load_manifest(root)
    sq, sk, km, kc, meta = base.calib_moments(root, man, args.calib_idx)
    L, Hkv, d = meta["n_layers"], meta["n_kv_heads"], meta["d_head"]
    print(f"calib_idx={args.calib_idx} | L={L} Hkv={Hkv} d={d} | G={args.G} | out={args.out}",
          flush=True)

    qc = base.build_qpca_basis(sq, kc)
    qc["sigma_k"] = sk
    F, inv = qc["forward"], qc["inverse"]
    fetch_calib = base._codes_for_idx(root, man, args.calib_idx, F, km, L, Hkv, d)

    t0 = time.time()
    comps = train_group_vq_compressors(F, inv, km, fetch_calib, L, Hkv, d,
                                        G=args.G, iters=args.iters, device=dev)
    print(f"[train_group_vq] done in {time.time()-t0:.1f}s, "
          f"bits/coord={comps[(1, 0)].bits_per_coord:.3f}", flush=True)

    payload = {
        "calib_idx": args.calib_idx, "G": args.G, "iters": args.iters,
        "L": L, "Hkv": Hkv, "d": d,
        "codebooks": {lh: [c.cpu() for c in comp.codebooks] for lh, comp in comps.items()},
        "bounds": comps[(1, 0)].bounds,
        "forward": F.cpu(), "inverse": inv.cpu(), "mean": km.cpu(),
    }
    torch.save(payload, args.out)
    print(f"[train_group_vq] saved -> {args.out}", flush=True)
