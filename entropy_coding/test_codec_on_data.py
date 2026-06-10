#!/usr/bin/env python3
"""Validate the laddered PageCodec on real data vs TurboQuant / JointQK baselines.

Paged codec is the REAL codec: K_hat = decode(encode(K)) over actual bytes.
Phase 1 (encode) runs on GPU via encode_gpu. --rdo switches the Paged method
from per-page rung selection (PageCodecRANSCUDA) to per-token Lagrangian RDO
(PageCodecRDO). RDO has no GPU encoder; it falls back to CPU encode.

    python test_codec_on_data.py --calib-idx 0 1 2 --eval-idx 4 \
        --bits 4 --ptok 64 --dz 0.375 --lanes 8 [--rdo] [--cpu-encode]
"""
import argparse
import numpy as np
import torch

import run_pca_ec_deadzone as base
from kvq_codec import (build_codecs_from_ladder, _unpack, _CoordModel,
                       _dz_round, _dz_dequant, reset_timers, timer_report, save_config,
                       build_codecs_from_ladder_rans_cuda, load_ext,
                       BatchRANSDecoder, BatchRANSEncoder)

PAGED = "Paged"
METHODS = ["TurboQuant", "JointQK", PAGED]
SERIES = {"TurboQuant": ("#888", "s"), "JointQK": ("#328ac1", "o"), PAGED: ("#1c5d2c", "^")}


def build_ladder(qpca_cen, qpca_unc, k_mean, b, L, Hkv, fetch_calib, root, calib_idx, m_grid, dz):
    _, delta0, model0 = base.build_qpca_ec(qpca_cen, qpca_unc, k_mean, b, L, Hkv,
                                           fetch_calib, root, calib_idx, dz=dz, match_rate=False)
    d = delta0.shape[-1]
    ladder = []
    for m in sorted(float(x) for x in m_grid):
        if abs(m - 1.0) < 1e-9:
            ladder.append((1.0, delta0, model0))
        else:
            dm = (delta0 * m).float()
            ladder.append((m, dm, base.freeze_coder_model(fetch_calib, dm, L, Hkv, d, dz)))
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
              L, Hkv, d, P, dz, rdo=False, cpu_encode=False):
    acc = {m: {b: {l: _zero() for l in range(L)} for b in k_bits} for m in comps}
    cod = {b: dict(bytes=0, coords=0, maxerr=0.0) for b in k_bits}

    batch_decoders = {} if rdo else {b: BatchRANSDecoder(codecs[b]) for b in k_bits}
    # GPU encode unless RDO (no GPU encoder) or explicitly forced to CPU
    use_gpu_encode = (not rdo) and (not cpu_encode)
    batch_encoders = ({b: BatchRANSEncoder(codecs[b]) for b in k_bits}
                      if use_gpu_encode else {})

    for ii, i in enumerate(eval_idx):
        print(f"  [eval {ii+1}/{len(eval_idx)}] ex {i}...", end=" ", flush=True)
        art = torch.load(root / manifest["examples"][i]["file"], map_location="cpu", weights_only=False)
        q_all, k_all = art["q_post"], art["k_post"]
        T = int(art["prompt_length"])
        gs = q_all.shape[1] // Hkv

        # --- PHASE 1: encode all heads ---
        paged_buffers = {b: {} for b in k_bits}
        for b in k_bits:
            if use_gpu_encode:
                k_grid = {(l, h): k_all[l, h, :T, :].float()
                          for l in range(L) for h in range(Hkv)}
                paged_buffers[b] = batch_encoders[b].encode_grid(k_grid)
            else:
                for l in range(L):
                    for h in range(Hkv):
                        paged_buffers[b][(l, h)] = codecs[b][(l, h)].encode(k_all[l, h, :T, :].float())

        # --- PHASE 2: decode on GPU ---
        paged_decoded = {}
        for b in k_bits:
            if rdo:
                paged_decoded[b] = {lh: codecs[b][lh].decode_to_gpu_rdo(paged_buffers[b][lh])
                                    for lh in codecs[b]}
            else:
                paged_decoded[b] = batch_decoders[b].decode_grid(paged_buffers[b])

        # --- TIMING: decode cost, Paged (GPU) vs TurboQuant, same heads ---
        if ii == 0:
            import time
            for b in k_bits:
                # Paged GPU decode over all heads (decode_grid already syncs once)
                if not rdo:
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    _ = batch_decoders[b].decode_grid(paged_buffers[b])
                    torch.cuda.synchronize()
                    paged_ms = (time.perf_counter() - t0) * 1e3
                else:
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    for lh in codecs[b]:
                        _ = codecs[b][lh].decode_to_gpu_rdo(paged_buffers[b][lh])
                    torch.cuda.synchronize()
                    paged_ms = (time.perf_counter() - t0) * 1e3

                # TurboQuant decode: its dequant over the same heads
                tq = comps["TurboQuant"][b]
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for (l, h) in paged_buffers[b]:
                    _ = tq[(l, h)].roundtrip(k_all[l, h, :T, :].to(device).float())
                torch.cuda.synchronize()
                tq_ms = (time.perf_counter() - t0) * 1e3

                # JointQK decode
                jqc = comps["JointQK"][b]
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for (l, h) in paged_buffers[b]:
                    _ = jqc[(l, h)].roundtrip(k_all[l, h, :T, :].to(device).float())
                torch.cuda.synchronize()
                jq_ms = (time.perf_counter() - t0) * 1e3

                print(f"[decode-time b={b}] Paged(GPU) {paged_ms:.1f} ms | "
                      f"JointQK {jq_ms:.1f} ms | TurboQuant {tq_ms:.1f} ms | {L*Hkv} heads "
                      f"(Paged/TQ {paged_ms/max(tq_ms,1e-9):.1f}x, JQK/TQ {jq_ms/max(tq_ms,1e-9):.1f}x)")

        # --- PHASE 3: metrics ---
        for l in range(L):
            for h in range(Hkv):
                k_dev = k_all[l, h, :T, :].to(device).float()
                q = q_all[l, h*gs:(h+1)*gs, :T, :].to(device).float().reshape(-1, d)
                qq = q.transpose(0, 1) @ q
                real_top = (q @ k_dev.transpose(0, 1)).argmax(-1)
                n_q = int(real_top.numel())
                k5 = min(5, T)

                for m in comps:
                    for b in k_bits:
                        if m == PAGED:
                            kh = paged_decoded[b][(l, h)]
                            if l != 0:
                                buf = paged_buffers[b][(l, h)]
                                cod[b]["bytes"] += len(buf)
                                cod[b]["coords"] += T * d
                                if not rdo:
                                    c = codecs[b][(l, h)]
                                    knp = k_all[l, h, :T, :].float().double().cpu().numpy()
                                    ref = _snapped_recon(c, knp, buf, rungs_np[b][(l, h)], P, dz, d)
                                    rec_np = kh.cpu().numpy()
                                    cod[b]["maxerr"] = max(cod[b]["maxerr"],
                                                           float(np.abs(rec_np - ref).max()))
                        else:
                            kh = comps[m][b][(l, h)].roundtrip(k_dev).float()

                        err = k_dev - kh
                        a = acc[m][b][l]
                        a["mse_num"] += float(err.square().sum())
                        a["mse_den"] += int(err.numel())
                        ee = err.transpose(0, 1) @ err
                        a["logit_num"] += float((qq * ee).sum())
                        a["logit_den"] += int(q.shape[0]*T*T)
                        ap = q @ kh.transpose(0, 1)
                        a["top1_num"] += int((real_top == ap.argmax(-1)).sum())
                        a["top1_den"] += n_q
                        a5 = ap.topk(k5, dim=-1).indices
                        a["top5_num"] += int((a5 == real_top.unsqueeze(-1)).any(-1).sum())
                        a["top5_den"] += n_q
        print("done")
    return acc, cod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-idx", type=int, nargs="+", required=True)
    ap.add_argument("--eval-idx", type=int, nargs="+", required=True)
    ap.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--ptok", type=int, default=16)
    ap.add_argument("--dz", type=float, default=0.375)
    ap.add_argument("--m-grid", type=float, nargs="+",
                    default=[1.0, 1.025, 1.05, 1.075, 1.1, 1.125, 1.15, 1.25, 1.5, 1.75])
    ap.add_argument("--lanes", type=int, default=8)
    ap.add_argument("--rdo", action="store_true",
                    help="per-token RDO codec (PageCodecRDO) instead of per-page selection")
    ap.add_argument("--cpu-encode", action="store_true",
                    help="force CPU encode (default is GPU encode for per-page)")
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
    mode = "RDO (per-token)" if args.rdo else "per-page selection"
    enc_mode = "CPU" if (args.rdo or args.cpu_encode) else "GPU"
    print(f"calib={args.calib_idx} eval={args.eval_idx} | L={L} Hkv={Hkv} d={d} | "
          f"device {device} | Paged={mode} | encode={enc_mode} | lanes={args.lanes}")

    qpca_unc = base.build_qpca_basis(sigma_q, sigma_k)
    qpca_unc["sigma_k"], qpca_unc["sigma_q"] = sigma_k, sigma_q
    qpca_cen = base.build_qpca_basis(sigma_q, k_cov); qpca_cen["sigma_k"] = sigma_k
    F = qpca_cen["forward"]; inv = qpca_cen["inverse"]
    fetch_calib = base._codes_for_idx(root, manifest, args.calib_idx, F, k_mean, L, Hkv, d)
    fetch_eval = base._codes_for_idx(root, manifest, args.eval_idx, F, k_mean, L, Hkv, d)

    turbo = base.build_turboquant(d, k_bits)
    jq = base.build_jointqk_basis(sigma_q, sigma_k)

    comps = {"TurboQuant": {}, "JointQK": {}, PAGED: {}}
    codecs = {}; rungs_np = {}; ec_rate = {}
    reset_timers()
    for b in k_bits:
        comps["TurboQuant"][b] = {(l, h): turbo[b] for l in range(L) for h in range(Hkv)}
        comps["JointQK"][b] = base.build_compressors(jq, b, L, Hkv)
        ladder, delta0, model0 = build_ladder(qpca_cen, qpca_unc, k_mean, b, L, Hkv,
                                               fetch_calib, root, args.calib_idx, m_grid, args.dz)
        ec_rate[b] = base.coded_bits_eval(fetch_eval, delta0, model0, L, Hkv, d, dz=args.dz)
        page_bits = b * d * args.ptok
        if args.save_config:
            save_config(f"{args.save_config}.b{b}", F, inv, k_mean, ladder, meta)

        if args.rdo:
            from kvq_codec_rdo import build_codecs_from_ladder_rdo_cuda
            codecs[b] = build_codecs_from_ladder_rdo_cuda(
                F, inv, k_mean, ladder, L, Hkv, page_bits, args.ptok, args.dz,
                lanes=args.lanes)
        else:
            ext = load_ext()
            codecs[b] = build_codecs_from_ladder_rans_cuda(
                F, inv, k_mean, ladder, L, Hkv, page_bits, args.ptok, args.dz,
                lanes=args.lanes, ext=ext, device="cuda")

        rungs_np[b] = {(l, h): [(codecs[b][(l, h)].deltas[ri], ladder[ri][2][(l, h)])
                                for ri in range(codecs[b][(l, h)].R)]
                       for l in range(L) for h in range(Hkv)}
        comps[PAGED][b] = codecs[b]

    base.move_to_device({k: v for k, v in comps.items() if k != PAGED}, device)

    acc, cod = score_all(root, manifest, args.eval_idx, comps, codecs, rungs_np,
                         k_bits, device, L, Hkv, d, args.ptok, args.dz,
                         rdo=args.rdo, cpu_encode=args.cpu_encode)
    final = base.finalize(acc, L, exclude_layer_0=True)

    def rate_of(m, b):
        if m == PAGED:
            return cod[b]["bytes"] * 8 / max(1, cod[b]["coords"])
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
    print("rate(b/c): TurboQuant/JointQK = grid-bits; Paged = REAL stored bits/coord. "
          "coded_bits_eval (unpaged ref): "
          + ", ".join(f"b{b}:{ec_rate[b]:.3f}" for b in k_bits))

    for b in k_bits:
        nb = sum(c.nblocks for (l, h), c in codecs[b].items() if l != 0)
        if args.rdo:
            ovf = sum(getattr(c, "rdo_overflow", 0) for (l, h), c in codecs[b].items() if l != 0)
        else:
            ovf = sum(c.overflow for (l, h), c in codecs[b].items() if l != 0)
        hist = np.zeros(len(m_grid), dtype=np.int64)
        for (l, h), c in codecs[b].items():
            if l != 0:
                hist += c.rung_hist
        usage = ", ".join(f"m{mm:g}:{h/max(1,hist.sum()):.2f}" for mm, h in zip(m_grid, hist) if h)
        merr = cod[b]["maxerr"]
        merr_s = f"{merr:.2e}" if not args.rdo else "n/a (rdo)"
        print(f"[Paged b={b}] round-trip max_err={merr_s} | "
              f"pages fit {nb-ovf}/{nb} (ovf {ovf}) | rung usage: {usage}")

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
    base.save_fig(fig, "04_paged_comparison_rdo.png" if args.rdo else "04_paged_comparison.png")

    total_cs = sum(cod[b]["coords"] for b in k_bits)
    timer_report(total_cs // d, total_cs)
    print("\nDone.")


if __name__ == "__main__":
    main()
