#!/usr/bin/env python3
"""Can Exp-Golomb hit rANS's achieved bits/coord by quantizing more aggressively
(coarser delta = fewer/smaller symbol values = shorter codes)? Sweeps a delta
multiplier m (SAME m-ladder convention as test_codec_on_data.py's rung search)
against Exp-Golomb's achieved bpc (per-coordinate k, no repacking needed -- just
the vectorized code-length sum) to find the m that rate-matches rANS's real
achieved rate on this exact calibration setup. Then reports, AT THAT MATCHED
RATE: the decode speed (does Exp-Golomb's speed edge survive?) and the
reconstruction accuracy cost (rANS gets to use the FINER delta0 and still hit
this rate via better entropy coding; Exp-Golomb needs the coarser delta -- this
is the actual "accuracy tax" of the cheaper codec at matched rate)."""
import numpy as np, torch, time
import run_pca_ec_deadzone as base
from kvq_codec import load_ext as load_rans_ext
from expgolomb_codec import (eg_encode_page_grid, choose_k_per_coord, estimate_bits_per_coord,
                             combine_encodings, eg_decode_gpu_batch)

dev = torch.device("cuda")
P = 64
N_LANES = 16
N_HEADS_SAMPLE = 280       # full grid (was 8 -- A6 tightening, 2026-07-07 faithful-report audit)
RANS_US_PER_PAGE = 12.394  # canonical, full-280-head-grid kernel-only rate (see eg_vs_rans_matched.py)

root = base.data_root(); man = base.load_manifest(root)
sq, sk, km, kc, meta = base.calib_moments(root, man, [0, 5, 6])
L, Hkv, d = meta["n_layers"], meta["n_kv_heads"], meta["d_head"]
qu = base.build_qpca_basis(sq, sk); qu["sigma_k"], qu["sigma_q"] = sk, sq
qc = base.build_qpca_basis(sq, kc); qc["sigma_k"] = sk
F, inv = qc["forward"], qc["inverse"]
fc_calib = base._codes_for_idx(root, man, [0, 5, 6], F, km, L, Hkv, d)
fc_eval = base._codes_for_idx(root, man, [4], F, km, L, Hkv, d)
_, delta0, model0 = base.build_qpca_ec(qc, qu, km, 2, L, Hkv, fc_calib, root, [0, 5, 6],
                                       dz=0.375, match_rate=False, uniform_step=True)
rans_rate = base.coded_bits_eval(fc_eval, delta0, model0, L, Hkv, d, dz=0.375)
print(f"rANS (arithmetic-coded) achieved rate on this exact calib/eval split: "
      f"{rans_rate:.3f} bits/coord (held-out, delta0 = the b=2 target step).\n")

eg_ext = load_rans_ext(source="expgolomb_decode.cu", name="expgolomb_decode")

art = torch.load(root / man["examples"][4]["file"], map_location="cpu", weights_only=False)
T0 = int(art["prompt_length"])
T_BENCH = 16384          # representative size for the rate-sweep + timing check


def tile_to(x, T):
    reps = (T + x.shape[0] - 1) // x.shape[0]
    return x.repeat(reps, 1)[:T].contiguous()


def timeit_ev(f, reps=20, warmup=5):
    for _ in range(warmup):
        f()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(reps):
        f()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


heads = [(l, h) for l in range(1, L) for h in range(Hkv)][:N_HEADS_SAMPLE]


def real_k_and_idx(l, h, m, T):
    kreal = tile_to(art["k_post"][l, h, :T0, :].float(), T)
    r = (kreal.double() - km[l, h].double()) @ F[l, h].double()
    delta_m = (delta0[l, h].double() * m).clamp_min(1e-12)
    idx = base._dz_round(r, delta_m, 0.375).long().numpy()
    return kreal.numpy(), idx, delta_m


print(f"m-sweep (T={T_BENCH}, {N_HEADS_SAMPLE} heads, per-coord-k Exp-Golomb, bpc estimate only "
      f"-- no repacking):")
print(f"{'m':>6} {'avg_bpc':>9} {'target_gap':>11}")
m_grid = [1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
bpcs_by_m = {}
for m in m_grid:
    bpcs = []
    for (l, h) in heads:
        _, idx, _ = real_k_and_idx(l, h, m, T_BENCH)
        k = choose_k_per_coord(idx)
        bpcs.append(estimate_bits_per_coord(idx, k))
    avg = float(np.mean(bpcs))
    bpcs_by_m[m] = avg
    print(f"{m:>6.2f} {avg:>9.3f} {avg - rans_rate:>+11.3f}")

# coarse pick, then a fine local linear search around the bracketing interval
# for a tighter rate match (A6 tightening, 2026-07-07 faithful-report audit --
# the coarse 9-point grid left ~1% gaps between test_codec_on_data.py's
# achieved bpc and rANS's real held-out rate).
ms = np.array(sorted(bpcs_by_m)); bs = np.array([bpcs_by_m[m] for m in ms])
coarse_m_star = float(ms[np.argmin(np.abs(bs - rans_rate))])
i0 = int(np.argmin(np.abs(bs - rans_rate)))
lo = ms[max(0, i0 - 1)]
hi = ms[min(len(ms) - 1, i0 + 1)]
fine_grid = np.linspace(lo, hi, 21)
fine_bpcs = {}
for m in fine_grid:
    m = float(m)
    bpcs = []
    for (l, h) in heads:
        _, idx, _ = real_k_and_idx(l, h, m, T_BENCH)
        k = choose_k_per_coord(idx)
        bpcs.append(estimate_bits_per_coord(idx, k))
    fine_bpcs[m] = float(np.mean(bpcs))
fms = np.array(sorted(fine_bpcs)); fbs = np.array([fine_bpcs[m] for m in fms])
m_star = float(fms[np.argmin(np.abs(fbs - rans_rate))])
print(f"\nCoarse pick: m={coarse_m_star:g} (bpc={bpcs_by_m[coarse_m_star]:.3f}); "
      f"fine search in [{lo:g}, {hi:g}] -> m={m_star:g} (bpc={fine_bpcs[m_star]:.3f}), "
      f"gap={fine_bpcs[m_star] - rans_rate:+.4f} bits/coord vs rANS's {rans_rate:.3f} target.\n")

print(f"At m={m_star:g} (matched rate): decode speed + reconstruction accuracy cost, "
      f"T={T_BENCH}, single combined kernel launch.")
encs = []
mse_num = 0.0
mse_den = 0
mse_num_baseline = 0.0
for (l, h) in heads:
    kreal, idx, delta_m = real_k_and_idx(l, h, m_star, T_BENCH)
    k = choose_k_per_coord(idx)
    enc = eg_encode_page_grid(idx, P, N_LANES, k)
    encs.append(enc)
    r_hat = base._dz_dequant(torch.from_numpy(idx).double(), delta_m, 0.375)
    k_hat = (r_hat @ inv[l, h].double() + km[l, h].double()).numpy()
    err = kreal - k_hat
    mse_num += float((err ** 2).sum()); mse_den += err.size

    # baseline: rANS's OWN operating point (delta0, m=1.0) reconstruction, for
    # the SAME data -- the accuracy rANS actually delivers at this rate.
    r0 = (torch.from_numpy(kreal).double() - km[l, h].double()) @ F[l, h].double()
    idx0 = base._dz_round(r0, delta0[l, h].double().clamp_min(1e-12), 0.375)
    r0_hat = base._dz_dequant(idx0, delta0[l, h].double(), 0.375)
    k0_hat = (r0_hat @ inv[l, h].double() + km[l, h].double()).numpy()
    mse_num_baseline += float(((kreal - k0_hat) ** 2).sum())

merged = combine_encodings(encs)


def decode_all():
    eg_decode_gpu_batch(merged, eg_ext, device="cuda")


t_ms = timeit_ev(decode_all)
n_pages = merged["n_pages"]
us_per_page = (t_ms / 1e3) / n_pages * 1e6
achieved_bpc = sum(len(e["blob"]) * 8 for e in encs) / sum(e["T"] * e["d"] for e in encs)

print(f"  achieved bpc (real bytes)     : {achieved_bpc:.3f}  (rANS target: {rans_rate:.3f})")
print(f"  decode                        : {us_per_page:.3f} us/page "
      f"({us_per_page / RANS_US_PER_PAGE:.2f}x rANS's {RANS_US_PER_PAGE}us/page)")
print(f"  k_mse @ m={m_star:g} (Exp-Golomb's operating point) : {mse_num / mse_den:.4e}")
print(f"  k_mse @ m=1.0    (rANS's operating point, delta0)   : {mse_num_baseline / mse_den:.4e}")
print(f"  accuracy tax (ratio)                                : "
      f"{(mse_num / mse_den) / (mse_num_baseline / mse_den):.2f}x higher MSE")
