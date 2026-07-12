#!/usr/bin/env python3
"""Kernel-only per-page Exp-Golomb-k decode cost, measured the SAME way as
kernel_only_micro.py / bw_regime.py's RANS_US_PER_PAGE constant: real b=2
QPCA-quantized K data, P=64 token pages, timed via CUDA events, us/page and
ns/symbol reported directly alongside the rANS reference number so the two are
apples-to-apples comparable.

Uses a handful of representative (layer, head) pairs (not the full 36x8 grid --
Exp-Golomb's CPU encoder is not the thing being timed, and is not optimized for
speed; the encode of N heads is startup cost paid once, not part of the reported
number). SUPERSEDED by eg_vs_rans_matched.py's full-280-head-grid rerun for the
canonical numbers (2026-07-07 faithful-report audit) -- an 8-head sample here
was found to underestimate both rANS's and Exp-Golomb's real per-page cost by
~1.5-4.6x; keep this script for quick spot-checks only, not headline numbers."""
import numpy as np, torch, time
import run_pca_ec_deadzone as base
from kvq_codec import load_ext as load_rans_ext
from expgolomb_codec import (eg_encode_page_grid, choose_k, choose_k_per_coord,
                             combine_encodings, eg_decode_gpu_batch)

dev = torch.device("cuda")
P = 64
N_LANES = 16                      # match rANS's default --lanes
N_HEADS_SAMPLE = 8                # representative subset of the 36x8 grid (encode is CPU/slow)
RANS_US_PER_PAGE = 12.394         # canonical, full-280-head-grid (see eg_vs_rans_matched.py)

root = base.data_root(); man = base.load_manifest(root)
sq, sk, km, kc, meta = base.calib_moments(root, man, [0, 5, 6])
L, Hkv, d = meta["n_layers"], meta["n_kv_heads"], meta["d_head"]
qu = base.build_qpca_basis(sq, sk); qu["sigma_k"], qu["sigma_q"] = sk, sq
qc = base.build_qpca_basis(sq, kc); qc["sigma_k"] = sk
F, inv = qc["forward"], qc["inverse"]
fc = base._codes_for_idx(root, man, [0, 5, 6], F, km, L, Hkv, d)
_, delta0, _ = base.build_qpca_ec(qc, qu, km, 2, L, Hkv, fc, root, [0, 5, 6],
                                  dz=0.375, match_rate=False, uniform_step=True)

eg_ext = load_rans_ext(source="expgolomb_decode.cu", name="expgolomb_decode")

art = torch.load(root / man["examples"][4]["file"], map_location="cpu", weights_only=False)
T0 = int(art["prompt_length"])


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
    return s.elapsed_time(e) / reps            # ms


heads = [(l, h) for l in range(1, L) for h in range(Hkv)][:N_HEADS_SAMPLE]

print(f"Exp-Golomb-k kernel-only decode cost, P={P}, lanes={N_LANES}, "
      f"{N_HEADS_SAMPLE} representative heads (of {L}x{Hkv} grid).")
print(f"rANS reference: {RANS_US_PER_PAGE} us/page (kernel-only, saturated, CDF-in-SRAM).\n")
print(f"{'T':>8} {'mode':>10} {'ks(range)':>10} {'avg_bpc':>8} {'n_pages':>9} "
      f"{'eg_us/page':>12} {'ns/symbol':>10} {'eg/rans':>8}")

for T in [4096, 8192, 16384, 32768]:
    idxs = []
    t_prep0 = time.time()
    for (l, h) in heads:
        r = (tile_to(art["k_post"][l, h, :T0, :].float(), T).double() - km[l, h].double()) @ F[l, h].double()
        idx = base._dz_round(r, delta0[l, h].double().clamp_min(1e-12), 0.375).long().numpy()
        idxs.append(idx)
    t_prep = time.time() - t_prep0

    for mode, kfn in [("scalar-k", choose_k), ("per-coord-k", choose_k_per_coord)]:
        encs = []
        ks = []
        bpcs = []
        t_encode0 = time.time()
        for idx in idxs:
            k = kfn(idx)
            enc = eg_encode_page_grid(idx, P, N_LANES, k)
            encs.append(enc); ks.append(np.atleast_1d(k))
            bpcs.append(len(enc["blob"]) * 8 / (T * d))
        t_encode = time.time() - t_encode0

        # ONE combined page set across all sampled heads -> ONE kernel launch, matching
        # rANS's BatchRANSDecoder.decode_grid convention. Per-launch overhead amortizes
        # over n_pages the SAME way in both codecs, so this is the steady-state number.
        merged = combine_encodings(encs)

        def decode_all():
            eg_decode_gpu_batch(merged, eg_ext, device="cuda")

        t_ms = timeit_ev(decode_all)
        n_pages = merged["n_pages"]
        us_per_page = (t_ms / 1e3) / n_pages * 1e6
        ns_per_symbol = us_per_page * 1e3 / (P * d)
        ratio = us_per_page / RANS_US_PER_PAGE
        all_ks = np.concatenate(ks)
        print(f"{T:>8} {mode:>10} {all_ks.min():>4}-{all_ks.max():<4} {np.mean(bpcs):>8.3f} "
              f"{n_pages:>9} {us_per_page:>10.3f}us {ns_per_symbol:>8.3f}ns {ratio:>7.2f}x  "
              f"(encode {t_encode:.1f}s, not timed)")

print("\neg/rans: >1 => Exp-Golomb decode kernel is SLOWER per page than rANS's; <1 => faster.")
print("avg_bpc: actual achieved bits/coord (Exp-Golomb has no rate-matching step -- whatever "
      "the chosen k gives on this data, unlike rANS's calibrated-to-target-b ladder).")
print("scalar-k: one Golomb param for the whole head. per-coord-k: one per QPCA coordinate "
      "(mirrors rANS's per-coord CDF) -- extra table lookup per symbol, better rate.")
