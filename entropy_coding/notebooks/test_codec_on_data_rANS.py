#!/usr/bin/env python3
"""Validate the laddered PageCodec on the real `full` data AND compare it to the
TurboQuant / JointQK baselines from run_pca_split.py, with the original figure.

Everything frozen on calib (basis + rung ladder); nothing refit on eval. The
"Paged" method is the REAL codec: K_hat comes from decode(encode(K)) over actual
bytes, so its row also reports lossless round-trip, page-fit, and real stored rate.
Baselines (TurboQuant, JointQK) reconstruct via their .roundtrip as before.

    python test_codec_on_data.py --calib-idx 0 1 2 3 --eval-idx 4 5 \
        --bits 2 3 4 --ptok 16 --dz 0.375 \
        --m-grid 0.5 0.75 1.0 1.5 2.0 3.0 8.0
"""
import argparse
import numpy as np
import torch

try:
    import run_pca_ec_deadzone as base          # your repo's module name
except ModuleNotFoundError:
    import run_pca_split as base                 # fallback name
from kvq_codec import (build_codecs_from_ladder, _unpack, _CoordModel,
                       _dz_round, _dz_dequant, reset_timers, timer_report, save_config)
from kvq_codec_rans import build_codecs_from_ladder_rans

PAGED = "Paged"
METHODS = ["TurboQuant", "JointQK", PAGED]
SERIES = {"TurboQuant": ("#888", "s"), "JointQK": ("#328ac1", "o"), PAGED: ("#1c5d2c", "^")}


def build_ladder(qpca_cen, qpca_unc, k_mean, b, L, Hkv, fetch_calib, root, calib_idx, m_grid, dz):
    _, delta0, model0 = base.build_qpca_ec(qpca_cen, qpca_unc, k_mean, b, L, Hkv,
                                           fetch_calib, root, calib_idx, dz=dz, match_rate=False)
    d = delta0.shape[-1]
    from pathlib import Path
    cache_dir = getattr(base, "MOMENTS_CACHE", Path("moments_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    idxs = "_".join(str(i) for i in sorted(calib_idx))
    ladder = []
    for m in sorted(float(x) for x in m_grid):
        if abs(m - 1.0) < 1e-9:
            ladder.append((1.0, delta0, model0))
            continue
        dm = (delta0 * m).float()
        cp = cache_dir / f"{root.name}__rung__idx_{idxs}__b{b}__dz{dz:g}__m{m:g}.pt"
        if cp.exists():
            print(f"    [rung m={m:g}] cache hit", flush=True)
            model = torch.load(cp, map_location="cpu", weights_only=False)["model"]
        else:
            print(f"    [rung m={m:g}] freezing on calib (one-time)...", flush=True)
            model = base.freeze_coder_model(fetch_calib, dm, L, Hkv, d, dz)
            torch.save({"model": model, "m": m, "b": b, "dz": dz}, cp)
        ladder.append((m, dm, model))
    return ladder, delta0, model0


def _snapped_recon(codec, knp, buf, rungs_lh, P, dz, d):
    r = (knp - codec.mu) @ codec.fwd
    T = r.shape[0]; nb = (T + P - 1) // P
    rid, _ = _unpack(buf, nb, codec.id_bits, 16)
    rh = np.empty((T, d))
    for bi in range(nb):
        sl = slice(bi * P, min((bi + 1) * P, T)); ri = int(rid[bi])
        delta, model = rungs_lh[ri]
        q = _dz_round(r[sl], delta, dz).astype(np.int64); idx = np.zeros_like(q)
        for j in range(d):
            m = _CoordModel(*model[j])
            idx[:, j] = m.const_val if m.constant else m.vals[m.snap(q[:, j])]
        rh[sl] = _dz_dequant(idx, delta, dz)
    return ((rh @ codec.inv) + codec.mu).astype(np.float32)


def _zero():
    return dict(mse_num=0.0, mse_den=0, logit_num=0.0, logit_den=0,
                top1_num=0, top1_den=0, top5_num=0, top5_den=0)


def score_all(root, manifest, eval_idx, comps, codecs, rungs_np, k_bits, device,
              L, Hkv, d, P, dz, max_tokens=0):
    """One eval example at a time. Baselines via .roundtrip; PAGED via real
    encode/decode (also accumulates bytes, max_err, fit)."""
    acc = {m: {b: {l: _zero() for l in range(L)} for b in k_bits} for m in comps}
    cod = {b: dict(bytes=0, coords=0, maxerr=0.0) for b in k_bits}
    for ii, i in enumerate(eval_idx):
        print(f"  [eval {ii+1}/{len(eval_idx)}] ex {i}...", end=" ", flush=True)
        art = torch.load(root / manifest["examples"][i]["file"],
                         map_location="cpu", weights_only=False)
        q_all, k_all = art["q_post"], art["k_post"]
        T = int(art["prompt_length"]); gs = q_all.shape[1] // Hkv
        if max_tokens and T > max_tokens:
            T = max_tokens
        for l in range(L):
            for h in range(Hkv):
                k_dev = k_all[l, h, :T, :].to(device).float()
                q = q_all[l, h*gs:(h+1)*gs, :T, :].to(device).float().reshape(-1, d)
                qq = q.transpose(0, 1) @ q
                real_top = (q @ k_dev.transpose(0, 1)).argmax(-1)
                n_q = int(real_top.numel()); k5 = min(5, T)
                for m in comps:
                    for b in k_bits:
                        if m == PAGED:
                            c = codecs[b][(l, h)]
                            knp = k_all[l, h, :T, :].float().double().cpu().numpy()
                            buf = c.encode(k_all[l, h, :T, :].float())
                            rec = c.decode(buf)
                            if l != 0:                     # track real metrics on scored layers
                                ref = _snapped_recon(c, knp, buf, rungs_np[b][(l, h)], P, dz, d)
                                cod[b]["maxerr"] = max(cod[b]["maxerr"], float(np.abs(rec - ref).max()))
                                cod[b]["bytes"] += len(buf); cod[b]["coords"] += T * d
                            kh = torch.from_numpy(rec).to(device)
                        else:
                            kh = comps[m][b][(l, h)].roundtrip(k_dev).float()
                        err = k_dev - kh; a = acc[m][b][l]
                        a["mse_num"] += float(err.square().sum()); a["mse_den"] += int(err.numel())
                        ee = err.transpose(0, 1) @ err
                        a["logit_num"] += float((qq * ee).sum()); a["logit_den"] += int(q.shape[0]*T*T)
                        ap = q @ kh.transpose(0, 1)
                        a["top1_num"] += int((real_top == ap.argmax(-1)).sum()); a["top1_den"] += n_q
                        a5 = ap.topk(k5, dim=-1).indices
                        a["top5_num"] += int((a5 == real_top.unsqueeze(-1)).any(-1).sum()); a["top5_den"] += n_q
        print("done")
    return acc, cod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-idx", type=int, nargs="+", required=True)
    ap.add_argument("--eval-idx", type=int, nargs="+", required=True)
    ap.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--ptok", type=int, default=16)
    ap.add_argument("--dz", type=float, default=0.375)
    ap.add_argument("--m-grid", type=float, nargs="+", default=[1.0, 1.05, 1.1, 1.25, 1.5, 2.0, 8.0])
    ap.add_argument("--coder", choices=["constriction", "rans"], default="constriction",
                    help="per-page entropy coder for the Paged method. 'rans' uses the "
                         "interleaved-rANS GPU bitstream (rans_interleaved.py).")
    ap.add_argument("--lanes", type=int, default=1,
                    help="rANS intra-page lanes (N). 1 = one stream/page (parallel across "
                         "pages). N>1 shortens per-page chains but adds per-lane overhead.")
    ap.add_argument("--max-eval-tokens", type=int, default=0,
                    help="cap eval T per example (0=all). The numpy rANS coder is a "
                         "CORRECTNESS reference, not fast -- use e.g. 256 with --coder rans.")
    ap.add_argument("--save-config", type=str, default="")
    args = ap.parse_args()
    k_bits = sorted(args.bits); m_grid = sorted(set(args.m_grid) | {1.0})
    overlap = set(args.calib_idx) & set(args.eval_idx)
    if overlap:
        raise SystemExit(f"calib/eval overlap: {sorted(overlap)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = base.data_root(); manifest = base.load_manifest(root)
    sigma_q, sigma_k, k_mean, k_cov, meta = base.calib_moments(root, manifest, args.calib_idx)
    L, Hkv, d = meta["n_layers"], meta["n_kv_heads"], meta["d_head"]
    print(f"calib={args.calib_idx} eval={args.eval_idx} | L={L} Hkv={Hkv} d={d} | device {device}")
    print(f"Paged coder = {args.coder}" + (f" (N={args.lanes} lanes)" if args.coder == "rans" else ""))

    qpca_unc = base.build_qpca_basis(sigma_q, sigma_k)
    qpca_unc["sigma_k"], qpca_unc["sigma_q"] = sigma_k, sigma_q
    qpca_cen = base.build_qpca_basis(sigma_q, k_cov); qpca_cen["sigma_k"] = sigma_k
    F = qpca_cen["forward"]; inv = qpca_cen["inverse"]
    fetch_calib = base._codes_for_idx(root, manifest, args.calib_idx, F, k_mean, L, Hkv, d)
    fetch_eval = base._codes_for_idx(root, manifest, args.eval_idx, F, k_mean, L, Hkv, d)

    # baselines (as in the original script)
    turbo = base.build_turboquant(d, k_bits)
    jq = base.build_jointqk_basis(sigma_q, sigma_k)

    comps = {"TurboQuant": {}, "JointQK": {}, PAGED: {}}
    codecs = {}; rungs_np = {}; ec_rate = {}
    reset_timers()                                       # times codec build + eval
    for b in k_bits:
        comps["TurboQuant"][b] = {(l, h): turbo[b] for l in range(L) for h in range(Hkv)}
        comps["JointQK"][b] = base.build_compressors(jq, b, L, Hkv)
        ladder, delta0, model0 = build_ladder(qpca_cen, qpca_unc, k_mean, b, L, Hkv,
                                               fetch_calib, root, args.calib_idx, m_grid, args.dz)
        ec_rate[b] = base.coded_bits_eval(fetch_eval, delta0, model0, L, Hkv, d, dz=args.dz)
        page_bits = b * d * args.ptok
        if args.save_config:
            save_config(f"{args.save_config}.b{b}", F, inv, k_mean, ladder, meta)
        if args.coder == "rans":
            codecs[b] = build_codecs_from_ladder_rans(F, inv, k_mean, ladder, L, Hkv,
                                                      page_bits, args.ptok, args.dz, lanes=args.lanes)
        else:
            codecs[b] = build_codecs_from_ladder(F, inv, k_mean, ladder, L, Hkv,
                                                 page_bits, args.ptok, args.dz)
        rungs_np[b] = {(l, h): [(codecs[b][(l, h)].deltas[ri], ladder[ri][2][(l, h)])
                                for ri in range(codecs[b][(l, h)].R)]
                       for l in range(L) for h in range(Hkv)}
        comps[PAGED][b] = codecs[b]                      # placeholder; PAGED handled in score_all

    # move only the baselines to device (codec runs on cpu via constriction)
    base.move_to_device({k: v for k, v in comps.items() if k != PAGED}, device)

    if args.coder == "rans" and not args.max_eval_tokens:
        print("WARNING: --coder rans uses the pure-Python interleaved-rANS reference "
              "(a correctness oracle, not fast). The full eval will take hours. "
              "Re-run with e.g. --max-eval-tokens 256 for a quick lossless/rate check, "
              "or use --coder constriction for the full sweep.", flush=True)

    acc, cod = score_all(root, manifest, args.eval_idx, comps, codecs, rungs_np,
                         k_bits, device, L, Hkv, d, args.ptok, args.dz,
                         max_tokens=args.max_eval_tokens)
    final = base.finalize(acc, L, exclude_layer_0=True)

    def rate_of(m, b):
        if m == PAGED:
            return cod[b]["bytes"] * 8 / max(1, cod[b]["coords"])   # real stored rate
        return float(b)

    print(f"\n{'method':<11} | {'b':<2} | {'rate(b/c)':>9} | {'top-1':>7} | {'top-5':>7} | "
          f"{'k_mse':>11} | {'logit_err':>11}")
    print("-" * 78)
    for m in METHODS:
        for b in k_bits:
            dd = final[m][b]
            print(f"{m:<11} | {b:<2} | {rate_of(m, b):>9.3f} | {dd['top1']:>7.4f} | {dd['top5']:>7.4f} | "
                  f"{dd['k_mse']:>11.3e} | {dd['logit_err']:>11.3e}")
        print()
    print("rate(b/c): TurboQuant/JointQK = grid-bits; Paged = REAL stored bits/coord "
          "(constriction bytes incl. page headers). coded_bits_eval (unpaged ref): "
          + ", ".join(f"b{b}:{ec_rate[b]:.3f}" for b in k_bits))

    # codec validation + rung usage
    for b in k_bits:
        nb = sum(c.nblocks for (l, h), c in codecs[b].items() if l != 0)
        ovf = sum(c.overflow for (l, h), c in codecs[b].items() if l != 0)
        hist = np.zeros(len(m_grid), dtype=np.int64)
        for (l, h), c in codecs[b].items():
            if l != 0:
                hist += c.rung_hist
        usage = ", ".join(f"m{mm:g}:{h/max(1,hist.sum()):.2f}" for mm, h in zip(m_grid, hist) if h)
        print(f"[Paged b={b}] round-trip max_err={cod[b]['maxerr']:.2e} | "
              f"pages fit {nb-ovf}/{nb} (ovf {ovf}) | rung usage: {usage}")

    # ---- original figure (1x4 metrics vs bits/coord) ----
    plt = base.plt
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    titles = {"top1": ("top-1", "higher"), "top5": ("top-5", "higher"),
              "k_mse": ("k_mse", "lower"), "logit_err": ("logit_err", "lower")}
    for ax, met in zip(axes, ["top1", "top5", "k_mse", "logit_err"]):
        for m in METHODS:
            xs = [rate_of(m, b) for b in k_bits]; ys = [final[m][b][met] for b in k_bits]
            o = np.argsort(xs); color, mk = SERIES[m]
            ax.plot(np.array(xs)[o], np.array(ys)[o], marker=mk, color=color,
                    label=m, linewidth=1.8, markersize=7)
        t, dirn = titles[met]
        ax.set_title(f"{t}\n({dirn})", fontsize=11); ax.set_xlabel("bits / coord (stored)")
        if met in ("k_mse", "logit_err"):
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout()
    base.save_fig(fig, "04_paged_comparison.png")

    total_cs = sum(cod[b]["coords"] for b in k_bits)      # codec coord-symbols (scored layers)
    timer_report(total_cs // d, total_cs)
    print("\nDone.")


if __name__ == "__main__":
    main()