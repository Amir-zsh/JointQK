#!/usr/bin/env python3
"""Build per-(layer, kv_head) K-compressor bundles for the non-VQ methods.

Follows the Llama EC study's calibration split (do-as-the-Llama-study): the BASIS
second moments (Sigma_Q, Sigma_K, k_mean, k_cov) come from a large pooled corpus
(`--basis-moments`, the 400-prompt compact8-train capture, ~calib_moments format),
while the entropy-coder-specific fit (deadzone step + frozen symbol model) reads
per-token codes from a small representative set of retained per-example captures
(`--code-idx` into the EC pool). scalar_int2 and oscar are purely Sigma-derived, so
they only ever see the big basis; only ec_deadzone touches the per-example codes.

  scalar_int2 : QPCADeadzoneFixed (INT2 / QPCA-Fixed, the report's scalar baseline)
  oscar       : OSCARCompressor (rotation + per-token dynamic INT2 + sink/recent)
  ec_deadzone : UniformECRoundtrip at the b=2 deadzone step -- the reconstruction the
                rANS / Exp-Golomb entropy coders decode (lossless coders don't change
                K_hat), so this one bundle stands in for both EC rows on F1.
"""
import argparse, torch
import run_pca_ec_deadzone as base
from oscar_codec import build_oscar_rotation, OSCARCompressor

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis-moments", default=None,
                    help="pooled 400-prompt basis-moment .pt (sigma_q/sigma_k/k_mean/k_cov). "
                         "If omitted, falls back to calib_moments on --code-idx (legacy 1-corpus mode).")
    ap.add_argument("--code-idx", type=int, nargs="+", default=list(range(24)),
                    help="EC-pool example indices for the entropy-coder code fit (deltas + model).")
    ap.add_argument("--calib-idx", type=int, nargs="+", default=None,
                    help="legacy alias for --code-idx (1-corpus mode).")
    ap.add_argument("--data-root", default=None,
                    help="override the EC per-example pool dir (query_stats format). "
                         "Default: the module FULL_DIR (Qwen pool). Set for the Llama gate.")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--suffix", default="", help="appended to mb_*.pt filenames (e.g. _n400, _llama)")
    ap.add_argument("--ec-basis", choices=["uncentered", "centered"], default="uncentered",
                    help="QPCA basis for EC: uncentered Sigma_K (study's ec_qpca_unc, better) "
                         "or centered k_cov (old default, worse).")
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
    print(f"basis=[{'pooled400' if args.basis_moments else code_idx}] code_idx={code_idx} "
          f"L={L} Hkv={Hkv} d={d}", flush=True)

    # ---- scalar INT2 / QPCA-Fixed (uncentered QPCA basis, per-coord waterfill + std) ----
    qpca_unc = base.build_qpca_basis(sq, sk)
    scalar = base.build_qpca_fixed_deadzone(qpca_unc, km, b=2, n_layers=L, n_kv_heads=Hkv,
                                            dz=0.375, load=3.0)
    torch.save({"comps": scalar, "method": "scalar_int2"}, f"{args.out_dir}/mb_scalar_int2{args.suffix}.pt")
    print(f"saved mb_scalar_int2{args.suffix}.pt", flush=True)

    # ---- OSCAR ----
    R = build_oscar_rotation(sq)                               # (L,Hkv,d,d)
    oscar = {(l, h): OSCARCompressor(R[l, h].float(), clip_ratio=0.96, sink=64, recent=256)
             for l in range(L) for h in range(Hkv)}
    torch.save({"comps": oscar, "method": "oscar"}, f"{args.out_dir}/mb_oscar{args.suffix}.pt")
    print(f"saved mb_oscar{args.suffix}.pt", flush=True)

    # ---- EC deadzone (rANS / Exp-Golomb reconstruction) ----
    # Basis: UNCENTERED Sigma_K (like scalar), matching the Llama study's ec_qpca_unc
    # (44.90) rather than the centered-cov ec_qpca (41.68) it beats. Using k_cov here was
    # the EC underperformance bug -- EC was on the study's worse basis variant.
    qu = base.build_qpca_basis(sq, sk); qu["sigma_k"], qu["sigma_q"] = sk, sq
    ec_cov = sk if args.ec_basis == "uncentered" else kc
    qc = base.build_qpca_basis(sq, ec_cov); qc["sigma_k"] = sk
    F, inv = qc["forward"], qc["inverse"]
    fc = base._codes_for_idx(root, man, code_idx, F, km, L, Hkv, d)
    _, delta0, _ = base.build_qpca_ec(qc, qu, km, 2, L, Hkv, fc, root, code_idx,
                                      dz=0.375, match_rate=False, uniform_step=True)
    ec = {(l, h): base.UniformECRoundtrip(F[l, h], inv[l, h], km[l, h], delta0[l, h], dz=0.375)
          for l in range(L) for h in range(Hkv)}
    torch.save({"comps": ec, "method": "ec_deadzone"}, f"{args.out_dir}/mb_ec_deadzone{args.suffix}.pt")
    print(f"saved mb_ec_deadzone{args.suffix}.pt", flush=True)
