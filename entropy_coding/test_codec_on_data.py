#!/usr/bin/env python3
"""Validate the laddered PageCodec on real data vs TurboQuant / JointQK baselines.

Paged codecs are REAL codecs: K_hat = decode(encode(K)) over actual bytes. Two
paged arms:
  Paged       QPCA (centered, Sigma_Q x K-cov) basis, EC + uniform base step,
              m-ladder for per-page fit.
  JointQK-EC  JointQK (R_sym) basis through the SAME EC + uniform ladder. Direct
              basis comparison at the real stored rate.

Phase 1 (encode) runs on GPU via encode_gpu. --rdo switches the paged methods
from per-page rung selection (PageCodecRANSCUDA) to per-token Lagrangian RDO
(PageCodecRDO). RDO has no GPU encoder; it falls back to CPU encode.

    python test_codec_on_data.py --calib-idx 0 1 2 --eval-idx 4 \
        --bits 4 --ptok 64 --dz 0.375 --lanes 8 [--rdo] [--cpu-encode] [--no-jqec]
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
JQEC = "JointQK-EC"
METHODS = ["TurboQuant", "JointQK", JQEC, PAGED]
SERIES = {"TurboQuant": ("#888", "s"), "JointQK": ("#328ac1", "o"),
          JQEC: ("#9b1c9b", "P"), PAGED: ("#1c5d2c", "^")}


def build_ec_uniform(fwd, inv, k_mean, b, L, Hkv, d, fetch, root, calib_idx, dz, tag):
    """Single-uniform-step EC base fit on an arbitrary basis. One scalar delta/head
    bisected to pooled rate b, model frozen on calib, cached with a basis tag.
    Returns (delta0(L,Hkv,d), model0)."""
    base.MOMENTS_CACHE.mkdir(parents=True, exist_ok=True)
    key = (f"{root.name}__ecmodel_{tag}__idx_" + "_".join(str(i) for i in sorted(calib_idx))
           + f"__b{b}__dz{dz:g}__us1")
    cp = base.MOMENTS_CACHE / f"{key}.pt"
    if cp.exists():
        c = torch.load(cp, map_location="cpu", weights_only=False)
        print(f"  [EC-{tag}] b={b} dz={dz:g} us=1: model CACHE HIT {cp.name}")
        return c["delta"], c["model"]
    print(f"  [EC-{tag}] b={b} dz={dz:g} us=1: fitting on calib...")
    h = base._diff_entropy_from_fetch(fetch, L, Hkv, d)             # (L,Hkv,d)
    delta = torch.pow(2.0, (h - float(b))).float()                 # per-coord start guess
    for l in range(L):
        for hh in range(Hkv):
            d0 = 2.0 ** float(delta[l, hh].clamp_min(1e-30).log2().mean())
            delta[l, hh] = base._solve_delta_uniform(fetch(l, hh), b, dz, d0)
    delta = delta.float()
    model = base.freeze_coder_model(fetch, delta, L, Hkv, d, dz)
    torch.save({"delta": delta, "model": model, "calib_idx": sorted(calib_idx),
                "b": b, "dz": dz, "tag": tag}, cp)
    print(f"    cached -> {cp.name}")
    return delta, model


def build_ladder_qpca(qpca_cen, qpca_unc, k_mean, b, L, Hkv, fetch_calib, root,
                      calib_idx, m_grid, dz):
    """QPCA m-ladder with a UNIFORM base step (uniform_step=True). m=1 rung reuses
    build_qpca_ec's cached uniform delta/model; other rungs scale delta0 by m."""
    _, delta0, model0 = base.build_qpca_ec(qpca_cen, qpca_unc, k_mean, b, L, Hkv,
                                           fetch_calib, root, calib_idx, dz=dz,
                                           match_rate=False, uniform_step=True)
    d = delta0.shape[-1]
    ladder = []
    for m in sorted(float(x) for x in m_grid):
        if abs(m - 1.0) < 1e-9:
            ladder.append((1.0, delta0, model0))
        else:
            dm = (delta0 * m).float()
            ladder.append((m, dm, base.freeze_coder_model(fetch_calib, dm, L, Hkv, d, dz)))
    return ladder, delta0, model0


def build_ladder_uniform(fwd, inv, k_mean, b, L, Hkv, d, fetch, root, calib_idx,
                         m_grid, dz, tag):
    """Generic uniform-step m-ladder on an arbitrary basis (here JointQK R_sym).
    Base step from build_ec_uniform (cached, tagged); other rungs scale by m."""
    delta0, model0 = build_ec_uniform(fwd, inv, k_mean, b, L, Hkv, d, fetch,
                                       root, calib_idx, dz, tag)
    ladder = []
    for m in sorted(float(x) for x in m_grid):
        if abs(m - 1.0) < 1e-9:
            ladder.append((1.0, delta0, model0))
        else:
            dm = (delta0 * m).float()
            ladder.append((m, dm, base.freeze_coder_model(fetch, dm, L, Hkv, d, dz)))
    return ladder, delta0, model0


def build_ec_percoord(fwd, inv, sigma_q, k_mean, b, L, Hkv, d, fetch, root,
                      calib_idx, dz, tag, match_rate=False):
    """Per-coord ECSQ EC base fit on an arbitrary basis. delta varies per coordinate
    via ell = diag(F^T Sigma_Q F) weighting; for an ORTHONORMAL basis (R_sym) this
    is the allocation mechanism, since the basis itself carries none. match_rate
    re-solves delta per coord (deadzone-aware) to hit b. Cached with a basis tag.
    Returns (delta0(L,Hkv,d), model0)."""
    base.MOMENTS_CACHE.mkdir(parents=True, exist_ok=True)
    key = (f"{root.name}__ecmodel_{tag}__idx_" + "_".join(str(i) for i in sorted(calib_idx))
           + f"__b{b}__dz{dz:g}__mr{int(match_rate)}__pc1")
    cp = base.MOMENTS_CACHE / f"{key}.pt"
    if cp.exists():
        c = torch.load(cp, map_location="cpu", weights_only=False)
        print(f"  [EC-{tag}] b={b} dz={dz:g} mr={int(match_rate)} per-coord: CACHE HIT {cp.name}")
        return c["delta"], c["model"]
    print(f"  [EC-{tag}] b={b} dz={dz:g} mr={int(match_rate)} per-coord: fitting on calib...")
    ell = (fwd.transpose(-1, -2).double() @ base.regularize_batch(sigma_q, base.EPS).double()
           @ fwd.double()).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    h = base._diff_entropy_from_fetch(fetch, L, Hkv, d)            # (L,Hkv,d)
    target = h + 0.5 * ell.log2()
    Hj = (target - target.mean(-1, keepdim=True) + float(b)).clamp_min(0.0)
    Hj = Hj * (float(b) * d / Hj.sum(-1, keepdim=True).clamp_min(1e-9))
    delta = torch.pow(2.0, (h - Hj)).float()                      # closed-form per-coord step
    if match_rate:
        for l in range(L):
            for hh in range(Hkv):
                delta[l, hh] = base._solve_delta_matched(fetch(l, hh), Hj[l, hh], dz, delta[l, hh])
    delta = delta.float()
    model = base.freeze_coder_model(fetch, delta, L, Hkv, d, dz)
    torch.save({"delta": delta, "model": model, "calib_idx": sorted(calib_idx),
                "b": b, "dz": dz, "match_rate": match_rate, "tag": tag}, cp)
    print(f"    cached -> {cp.name}")
    return delta, model


def build_ladder_percoord(fwd, inv, sigma_q, k_mean, b, L, Hkv, d, fetch, root,
                          calib_idx, m_grid, dz, tag, match_rate=False):
    """Per-coord-step m-ladder on an arbitrary basis (JointQK R_sym). Base step from
    build_ec_percoord (cached, tagged); other rungs scale the per-coord delta by m."""
    delta0, model0 = build_ec_percoord(fwd, inv, sigma_q, k_mean, b, L, Hkv, d, fetch,
                                        root, calib_idx, dz, tag, match_rate=match_rate)
    ladder = []
    for m in sorted(float(x) for x in m_grid):
        if abs(m - 1.0) < 1e-9:
            ladder.append((1.0, delta0, model0))
        else:
            dm = (delta0 * m).float()
            ladder.append((m, dm, base.freeze_coder_model(fetch, dm, L, Hkv, d, dz)))
    return ladder, delta0, model0


def _build_codecs(fwd, inv, k_mean, ladder, L, Hkv, page_bits, P, dz, lanes, rdo):
    if rdo:
        from kvq_codec_rdo import build_codecs_from_ladder_rdo_cuda
        return build_codecs_from_ladder_rdo_cuda(
            fwd, inv, k_mean, ladder, L, Hkv, page_bits, P, dz, lanes=lanes)
    ext = load_ext()
    return build_codecs_from_ladder_rans_cuda(
        fwd, inv, k_mean, ladder, L, Hkv, page_bits, P, dz,
        lanes=lanes, ext=ext, device="cuda")


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


def score_all(root, manifest, eval_idx, comps, paged_codecs, rungs_np, k_bits, device,
              L, Hkv, d, P, dz, paged_methods, rdo=False, cpu_encode=False):
    acc = {m: {b: {l: _zero() for l in range(L)} for b in k_bits} for m in comps}
    cod = {pm: {b: dict(bytes=0, coords=0, maxerr=0.0) for b in k_bits}
           for pm in paged_methods}

    use_gpu_encode = (not rdo) and (not cpu_encode)
    batch_decoders = {pm: ({} if rdo else {b: BatchRANSDecoder(paged_codecs[pm][b]) for b in k_bits})
                      for pm in paged_methods}
    batch_encoders = {pm: ({b: BatchRANSEncoder(paged_codecs[pm][b]) for b in k_bits}
                           if use_gpu_encode else {})
                      for pm in paged_methods}

    for ii, i in enumerate(eval_idx):
        print(f"  [eval {ii+1}/{len(eval_idx)}] ex {i}...", end=" ", flush=True)
        art = torch.load(root / manifest["examples"][i]["file"], map_location="cpu", weights_only=False)
        q_all, k_all = art["q_post"], art["k_post"]
        T = int(art["prompt_length"])
        gs = q_all.shape[1] // Hkv

        # --- PHASE 1: encode all heads, per paged method ---
        paged_buffers = {pm: {b: {} for b in k_bits} for pm in paged_methods}
        for pm in paged_methods:
            for b in k_bits:
                if use_gpu_encode:
                    k_grid = {(l, h): k_all[l, h, :T, :].float()
                              for l in range(L) for h in range(Hkv)}
                    paged_buffers[pm][b] = batch_encoders[pm][b].encode_grid(k_grid)
                else:
                    for l in range(L):
                        for h in range(Hkv):
                            paged_buffers[pm][b][(l, h)] = \
                                paged_codecs[pm][b][(l, h)].encode(k_all[l, h, :T, :].float())

        # --- PHASE 2: decode on GPU ---
        paged_decoded = {pm: {} for pm in paged_methods}
        for pm in paged_methods:
            for b in k_bits:
                if rdo:
                    paged_decoded[pm][b] = {lh: paged_codecs[pm][b][lh].decode_to_gpu_rdo(
                        paged_buffers[pm][b][lh]) for lh in paged_codecs[pm][b]}
                else:
                    paged_decoded[pm][b] = batch_decoders[pm][b].decode_grid(paged_buffers[pm][b])

        # --- TIMING: decode cost, paged (GPU) vs TQ / JQ-grid, same heads ---
        if ii == 0:
            import time
            for b in k_bits:
                ms = {}
                for pm in paged_methods:
                    if not rdo:
                        torch.cuda.synchronize(); t0 = time.perf_counter()
                        _ = batch_decoders[pm][b].decode_grid(paged_buffers[pm][b])
                        torch.cuda.synchronize()
                    else:
                        torch.cuda.synchronize(); t0 = time.perf_counter()
                        for lh in paged_codecs[pm][b]:
                            _ = paged_codecs[pm][b][lh].decode_to_gpu_rdo(paged_buffers[pm][b][lh])
                        torch.cuda.synchronize()
                    ms[pm] = (time.perf_counter() - t0) * 1e3

                heads = list(paged_buffers[paged_methods[0]][b])
                tq = comps["TurboQuant"][b]
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for (l, h) in heads:
                    _ = tq[(l, h)].roundtrip(k_all[l, h, :T, :].to(device).float())
                torch.cuda.synchronize()
                tq_ms = (time.perf_counter() - t0) * 1e3

                jqc = comps["JointQK"][b]
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for (l, h) in heads:
                    _ = jqc[(l, h)].roundtrip(k_all[l, h, :T, :].to(device).float())
                torch.cuda.synchronize()
                jq_ms = (time.perf_counter() - t0) * 1e3

                paged_s = " | ".join(f"{pm}(GPU) {ms[pm]:.1f} ms" for pm in paged_methods)
                print(f"[decode-time b={b}] {paged_s} | JointQK {jq_ms:.1f} ms | "
                      f"TurboQuant {tq_ms:.1f} ms | {len(heads)} heads")

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
                        if m in paged_methods:
                            kh = paged_decoded[m][b][(l, h)]
                            if l != 0:
                                buf = paged_buffers[m][b][(l, h)]
                                cod[m][b]["bytes"] += len(buf)
                                cod[m][b]["coords"] += T * d
                                if not rdo:
                                    c = paged_codecs[m][b][(l, h)]
                                    knp = k_all[l, h, :T, :].float().double().cpu().numpy()
                                    ref = _snapped_recon(c, knp, buf, rungs_np[m][b][(l, h)], P, dz, d)
                                    rec_np = kh.cpu().numpy()
                                    cod[m][b]["maxerr"] = max(cod[m][b]["maxerr"],
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
    ap.add_argument("--no-jqec", action="store_true",
                    help="skip the JointQK-EC paged arm (R_sym basis, EC+per-coord ladder)")
    ap.add_argument("--jq-match-rate", action="store_true",
                    help="JointQK-EC: deadzone-aware per-coord delta re-solve to hit b "
                         "(default is closed-form per-coord step)")
    ap.add_argument("--save-config", type=str, default="")
    args = ap.parse_args()
    k_bits = sorted(args.bits); m_grid = sorted(set(args.m_grid) | {1.0})
    overlap = set(args.calib_idx) & set(args.eval_idx)
    if overlap:
        raise SystemExit(f"calib/eval overlap: {sorted(overlap)}")
    use_jqec = not args.no_jqec
    methods = [m for m in METHODS if m != JQEC or use_jqec]
    paged_methods = [PAGED] + ([JQEC] if use_jqec else [])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = base.data_root(); manifest = base.load_manifest(root)
    sigma_q, sigma_k, k_mean, k_cov, meta = base.calib_moments(root, manifest, args.calib_idx)
    L, Hkv, d = meta["n_layers"], meta["n_kv_heads"], meta["d_head"]
    mode = "RDO (per-token)" if args.rdo else "per-page selection"
    enc_mode = "CPU" if (args.rdo or args.cpu_encode) else "GPU"
    print(f"calib={args.calib_idx} eval={args.eval_idx} | L={L} Hkv={Hkv} d={d} | "
          f"device {device} | paged={mode} | encode={enc_mode} | lanes={args.lanes} | "
          f"paged arms: {paged_methods} (QPCA base step = uniform)")

    qpca_unc = base.build_qpca_basis(sigma_q, sigma_k)
    qpca_unc["sigma_k"], qpca_unc["sigma_q"] = sigma_k, sigma_q
    qpca_cen = base.build_qpca_basis(sigma_q, k_cov); qpca_cen["sigma_k"] = sigma_k
    F = qpca_cen["forward"]; inv = qpca_cen["inverse"]
    fetch_calib = base._codes_for_idx(root, manifest, args.calib_idx, F, k_mean, L, Hkv, d)
    fetch_eval = base._codes_for_idx(root, manifest, args.eval_idx, F, k_mean, L, Hkv, d)

    turbo = base.build_turboquant(d, k_bits)
    jq = base.build_jointqk_basis(sigma_q, sigma_k)
    F_jq, inv_jq = jq["forward"], jq["inverse"]
    # Centered per-coord std for the JointQK grid: variance of the CENTERED rotated
    # keys (diag(R^T k_cov R)), not the uncentered sigma_k that build_jointqk_basis
    # used. Matches the centering applied below so the quantizer grid is scaled to
    # the centered dynamic range.
    jq_std_cen = (inv_jq.double() @ base.regularize_batch(k_cov, base.EPS).double()
                  @ F_jq.double()).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30).sqrt().float()
    fetch_calib_jq = base._codes_for_idx(root, manifest, args.calib_idx, F_jq, k_mean, L, Hkv, d)
    fetch_eval_jq = base._codes_for_idx(root, manifest, args.eval_idx, F_jq, k_mean, L, Hkv, d)

    comps = {m: {} for m in methods}
    paged_codecs = {pm: {} for pm in paged_methods}
    rungs_np = {pm: {} for pm in paged_methods}
    ec_rate = {}        # QPCA unpaged held-out ref
    jq_ec_rate = {}     # JointQK unpaged held-out ref
    reset_timers()
    for b in k_bits:
        comps["TurboQuant"][b] = {(l, h): turbo[b] for l in range(L) for h in range(Hkv)}
        jq_grid = base.build_compressors(jq, b, L, Hkv, std_override=jq_std_cen)
        comps["JointQK"][b] = {(l, h): base.CenteredRoundtrip(jq_grid[(l, h)], k_mean[l, h])
                               for (l, h) in jq_grid}
        page_bits = b * d * args.ptok

        # QPCA paged (uniform base step)
        ladder, delta0, model0 = build_ladder_qpca(qpca_cen, qpca_unc, k_mean, b, L, Hkv,
                                                   fetch_calib, root, args.calib_idx, m_grid, args.dz)
        ec_rate[b] = base.coded_bits_eval(fetch_eval, delta0, model0, L, Hkv, d, dz=args.dz)
        if args.save_config:
            save_config(f"{args.save_config}.b{b}", F, inv, k_mean, ladder, meta)
        paged_codecs[PAGED][b] = _build_codecs(F, inv, k_mean, ladder, L, Hkv,
                                               page_bits, args.ptok, args.dz, args.lanes, args.rdo)
        rungs_np[PAGED][b] = {(l, h): [(paged_codecs[PAGED][b][(l, h)].deltas[ri], ladder[ri][2][(l, h)])
                                       for ri in range(paged_codecs[PAGED][b][(l, h)].R)]
                              for l in range(L) for h in range(Hkv)}
        comps[PAGED][b] = paged_codecs[PAGED][b]

        # JointQK-EC paged (R_sym basis, per-coord ECSQ ladder)
        if use_jqec:
            jq_ladder, jq_delta0, jq_model0 = build_ladder_percoord(
                F_jq, inv_jq, sigma_q, k_mean, b, L, Hkv, d, fetch_calib_jq,
                root, args.calib_idx, m_grid, args.dz, "rsym", match_rate=args.jq_match_rate)
            jq_ec_rate[b] = base.coded_bits_eval(fetch_eval_jq, jq_delta0, jq_model0,
                                                 L, Hkv, d, dz=args.dz)
            paged_codecs[JQEC][b] = _build_codecs(F_jq, inv_jq, k_mean, jq_ladder, L, Hkv,
                                                  page_bits, args.ptok, args.dz, args.lanes, args.rdo)
            rungs_np[JQEC][b] = {(l, h): [(paged_codecs[JQEC][b][(l, h)].deltas[ri], jq_ladder[ri][2][(l, h)])
                                          for ri in range(paged_codecs[JQEC][b][(l, h)].R)]
                                 for l in range(L) for h in range(Hkv)}
            comps[JQEC][b] = paged_codecs[JQEC][b]

    base.move_to_device({k: v for k, v in comps.items() if k not in paged_methods}, device)

    acc, cod = score_all(root, manifest, args.eval_idx, comps, paged_codecs, rungs_np,
                         k_bits, device, L, Hkv, d, args.ptok, args.dz,
                         paged_methods, rdo=args.rdo, cpu_encode=args.cpu_encode)
    final = base.finalize(acc, L, exclude_layer_0=True)

    def rate_of(m, b):
        if m in paged_methods:
            return cod[m][b]["bytes"] * 8 / max(1, cod[m][b]["coords"])
        return float(b)

    print(f"\n{'method':<11} | {'b':<2} | {'rate(b/c)':>9} | {'top-1':>7} | {'top-5':>7} | "
          f"{'k_mse':>11} | {'logit_err':>11}")
    print("-" * 78)
    for m in methods:
        for b in k_bits:
            dd = final[m][b]
            print(f"{m:<11} | {b:<2} | {rate_of(m, b):>9.3f} | {dd['top1']:>7.4f} | {dd['top5']:>7.4f} | "
                  f"{dd['k_mse']:>11.3e} | {dd['logit_err']:>11.3e}")
        print()
    print("rate(b/c): TurboQuant/JointQK = grid-bits; Paged & JointQK-EC = REAL stored "
          "bits/coord (paged). Unpaged held-out refs (coded_bits_eval): QPCA "
          + ", ".join(f"b{b}:{ec_rate[b]:.3f}" for b in k_bits)
          + (" | R_sym " + ", ".join(f"b{b}:{jq_ec_rate[b]:.3f}" for b in k_bits)
             if use_jqec else ""))

    for pm in paged_methods:
        for b in k_bits:
            cm = paged_codecs[pm][b]
            nb = sum(c.nblocks for (l, h), c in cm.items() if l != 0)
            if args.rdo:
                ovf = sum(getattr(c, "rdo_overflow", 0) for (l, h), c in cm.items() if l != 0)
            else:
                ovf = sum(c.overflow for (l, h), c in cm.items() if l != 0)
            hist = np.zeros(len(m_grid), dtype=np.int64)
            for (l, h), c in cm.items():
                if l != 0:
                    hist += c.rung_hist
            usage = ", ".join(f"m{mm:g}:{h/max(1,hist.sum()):.2f}" for mm, h in zip(m_grid, hist) if h)
            merr = cod[pm][b]["maxerr"]
            merr_s = f"{merr:.2e}" if not args.rdo else "n/a (rdo)"
            print(f"[{pm} b={b}] round-trip max_err={merr_s} | "
                  f"pages fit {nb-ovf}/{nb} (ovf {ovf}) | rung usage: {usage}")

    plt = base.plt
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    titles = {"top1": ("top-1", "higher"), "top5": ("top-5", "higher"),
              "k_mse": ("k_mse", "lower"), "logit_err": ("logit_err", "lower")}
    for ax, met in zip(axes, ["top1", "top5", "k_mse", "logit_err"]):
        for m in methods:
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

    total_cs = sum(cod[PAGED][b]["coords"] for b in k_bits)
    timer_report(total_cs // d, total_cs)
    print("\nDone.")


if __name__ == "__main__":
    main()