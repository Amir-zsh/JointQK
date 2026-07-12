#!/usr/bin/env python3
"""
Sub-page probe v2 — payload-bits x quality x decode-speed, one grid.

Corrected metric reading: the page is a FIXED 2-bit slot. bits/coeff = how much of
that fixed budget reached the PAYLOAD (higher = less overhead = more real K coded),
NOT how big the output got. So higher payload bits/coeff is BETTER.

Two forces pull opposite ways on page size:
  - bigger page  -> overhead amortizes -> MORE payload + better rel_err
  - smaller page / more lanes -> shorter serial chains -> FASTER decode
Your sub-block idea wants both: big page (packs the fixed budget) decoded in many
small independent sub-streams (fast). This table shows all three axes so you can
pick the corner instead of eyeballing two tables.

Per (P, N): payload bits/coeff (higher better), rel_err (lower better),
decode kernel ms/head (lower better). Then the Pareto frontier on (payload, speed).

Subset of heads; per-coeff + per-head metrics are representative.
"""
import numpy as np, torch, statistics
import run_pca_ec_deadzone as base
from kvq_codec import (build_codecs_from_ladder_rans_cuda, load_ext, load_enc_ext,
                       BatchRANSEncoder, BatchRANSDecoder)

P_SWEEP = [64, 128, 256, 512]
N_SWEEP = [1, 4, 16, 32]
L_USE   = 8
REPS    = 3
dev = torch.device("cuda")

root = base.data_root(); man = base.load_manifest(root)
sq, sk, km, kc, meta = base.calib_moments(root, man, [0, 1, 2])
L, Hkv, d = meta["n_layers"], meta["n_kv_heads"], meta["d_head"]
qu = base.build_qpca_basis(sq, sk); qu["sigma_k"], qu["sigma_q"] = sk, sq
qc = base.build_qpca_basis(sq, kc); qc["sigma_k"] = sk
F, inv = qc["forward"], qc["inverse"]
fc = base._codes_for_idx(root, man, [0, 1, 2], F, km, L, Hkv, d)
_, d0, m0 = base.build_qpca_ec(qc, qu, km, 2, L, Hkv, fc, root, [0, 1, 2],
                               dz=0.375, match_rate=False, uniform_step=True)
ladder = [(1.0, d0, m0)]
for m in [1.05, 1.1, 1.25, 1.5]:
    dm = (d0 * m).float()
    ladder.append((m, dm, base.freeze_coder_model(fc, dm, L, Hkv, d, 0.375)))
ext = load_ext(); enc_ext = load_enc_ext()
cod = build_codecs_from_ladder_rans_cuda(F, inv, km, ladder, L_USE, Hkv, 2*d*64, 64, 0.375,
                                         lanes=16, ext=ext, enc_ext=enc_ext, device="cuda")
art = torch.load(root / man["examples"][4]["file"], map_location="cpu", weights_only=False)
T = int(art["prompt_length"])
kg = {(l, h): art["k_post"][l, h, :T, :].float() for l in range(L_USE) for h in range(Hkv)}
nheads = len(kg); ncoeff = nheads * T * d
k_orig = {hh: kg[hh].to(dev) for hh in kg}
norm_sq = sum(float((k_orig[hh]**2).sum()) for hh in kg)

def recon_err(decoded):
    num = sum(float(((decoded[hh] - k_orig[hh])**2).sum()) for hh in decoded)
    return (num / norm_sq) ** 0.5

def set_config(P, N):
    pb = 2 * d * P
    for c in cod.values():
        c.P = P; c.page_bits = pb; c.payload_budget_bits = pb - c.id_bits; c.N = N

def measure(P, N):
    set_config(P, N)
    enc = BatchRANSEncoder(cod)
    bufs = enc.encode_grid(kg)
    payload = sum(len(b) for b in bufs.values()) * 8 / ncoeff
    dec = BatchRANSDecoder(cod)
    decoded = dec.decode_grid(bufs); err = recon_err(decoded)   # warmup + quality
    mss = []
    for _ in range(REPS):
        dec.decode_grid(bufs); mss.append(dec.last_kernel_ms)
    ms_head = statistics.median(mss) / nheads
    del bufs, decoded; torch.cuda.empty_cache()
    return payload, err, ms_head

pay, errt, mst = {}, {}, {}
for P in P_SWEEP:
    for N in N_SWEEP:
        if N > P: continue
        pay[(P, N)], errt[(P, N)], mst[(P, N)] = measure(P, N)

def table(title, tab, fmt):
    print(f"\n===== {title} =====")
    print("        " + "".join(f"N={N:<9}" for N in N_SWEEP))
    for P in P_SWEEP:
        row = f"P={P:<5} "
        for N in N_SWEEP:
            if (P, N) not in tab: row += f"{'--':<11}"; continue
            mark = "*" if N == P // 16 else " "
            row += f"{fmt(tab[(P,N)])}{mark}   "
        print(row)

table("payload bits/coeff  (HIGHER better -- less overhead)", pay, lambda v: f"{v:.3f}")
table("reconstruction rel_err  (lower better)", errt, lambda v: f"{v:.2e}")
table("decode kernel ms/head  (lower better)", mst, lambda v: f"{v:6.2f}")

# Pareto frontier on (payload high, ms/head low)
cfgs = list(pay.keys())
def dominated(c):
    for o in cfgs:
        if o == c: continue
        if pay[o] >= pay[c] and mst[o] <= mst[c] and (pay[o] > pay[c] or mst[o] < mst[c]):
            return True
    return False
front = sorted([c for c in cfgs if not dominated(c)], key=lambda c: mst[c])
base_p, base_e, base_m = pay[(64,16)], errt[(64,16)], mst[(64,16)]
print(f"\nbaseline P=64,N=16: payload {base_p:.3f}, rel_err {base_e:.2e}, {base_m:.2f} ms/head")
print("\n--- Pareto frontier (payload vs decode speed; nothing beats these on both) ---")
print(f"{'P':>5} {'N':>4} {'payload':>9} {'rel_err':>9} {'ms/head':>9}  vs baseline")
for (P, N) in front:
    dp = pay[(P,N)] - base_p; dm = mst[(P,N)] / base_m
    print(f"{P:>5} {N:>4} {pay[(P,N)]:>9.3f} {errt[(P,N)]:>9.2e} {mst[(P,N)]:>9.2f}  "
          f"payload {dp:+.3f}, {dm:.2f}x decode time")
print("\nPick the frontier point that fits your decode-time budget; that (P,N) is the")
print("config to lock in. If a single point beats baseline on BOTH payload and speed,")
print("it's a free win with zero new code -- just set P_tok and lanes.")