#!/usr/bin/env python3
"""
Decode microbench — tabled rANS vs your binary-search rANS, decode only.

  1. correctness: LUT decode must produce bit-identical s_pos to binsearch (per head).
  2. throughput: symbols/sec for each, over N_HEADS heads. LUT build time reported
     separately (it's a one-time per-(head,rung) calibration cost, ~0 amortized).

The ratio is the headline: how much per-symbol cost (and warp divergence) the table
removes. Win scales with alphabet size A per dim and with how divergent the binary
search was. If LUT lands within ~1 order of magnitude of an fp16 K-read, the
bandwidth/throughput claims reopen; if it's still far, the entropy stage is the wall
and the next lever is interleaved states / more lanes (encoder-coupled).

Run from entropy_coding/ alongside test_fused.py.
"""
import numpy as np, torch, struct, shutil, os, time
from torch.utils.cpp_extension import load
import run_pca_ec_deadzone as base
from kvq_codec import build_codecs_from_ladder_rans_cuda, load_ext, BatchRANSEncoder, _unpack

N_HEADS = 16
REPS    = 5
dev = torch.device("cuda")

shutil.rmtree("./build_decode", ignore_errors=True); os.makedirs("./build_decode", exist_ok=True)
db = load(name="decode_bench", sources=["decode_bench.cu"],
          build_directory="./build_decode", verbose=True)

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
ext = load_ext()
cod = build_codecs_from_ladder_rans_cuda(F, inv, km, ladder, L, Hkv, 2*d*64, 64, 0.375,
                                         lanes=16, ext=ext, device="cuda")
enc = BatchRANSEncoder(cod)
art = torch.load(root / man["examples"][4]["file"], map_location="cpu", weights_only=False)
T = int(art["prompt_length"])
kg = {(l, h): art["k_post"][l, h, :T, :].float() for l in range(L) for h in range(Hkv)}
bufs = enc.encode_grid(kg)

def _T(a, dt): return torch.as_tensor(a, dtype=dt, device=dev)

def build_decode_tables(l, h):
    c = cod[(l, h)]; N = c.N; P = c.P; R = c.R
    buf = bytes(bufs[(l, h)])
    magic, Tt, Phdr, nb = struct.unpack_from("<IIII", buf, 0)
    rung_ids, off0 = _unpack(buf, nb, c.id_bits, 16)
    all_blob = bytearray(); page_byte_off = np.zeros(nb, np.int64)
    lane_off = np.zeros((nb, N), np.int64); k0a = np.zeros((nb, N), np.int32)
    k1a = np.zeros((nb, N), np.int32); rung_of_page = np.zeros(nb, np.int32)
    nsym = 0
    off = off0
    for bi in range(nb):
        blob, off = c._read_blob(buf, off)
        ri = int(rung_ids[bi]); rung_of_page[bi] = ri
        n_toks = min((bi + 1) * Phdr, Tt) - bi * Phdr
        Ci = int(c._cuda_cdf_tensors[ri][0].shape[0]); S = n_toks * Ci; nsym += S
        lane_lens = [struct.unpack_from("<I", blob, 8 + 4*ln)[0] for ln in range(N)]
        starts = np.cumsum([0] + lane_lens); bounds = [round(k*S/N) for k in range(N+1)]
        page_byte_off[bi] = len(all_blob)
        hs = 8 + 4*N
        for ln in range(N):
            lane_off[bi, ln] = hs + int(starts[ln]); k0a[bi, ln] = bounds[ln]; k1a[bi, ln] = bounds[ln+1]
        all_blob += blob
    blob_arr = np.frombuffer(bytes(all_blob), dtype=np.uint8).copy()

    nonconst_parts = []; nonconst_base = np.zeros(R, np.int32); C_all = np.zeros(R, np.int32)
    cdf_parts = []; cdf_flat_base = np.zeros(R, np.int32); cdf_off_all = np.zeros((R, d + 1), np.int32)
    nb_cur = 0; cf_cur = 0
    for ri in range(R):
        nc_t, cf_t, co_t = c._cuda_cdf_tensors[ri]
        ncc = nc_t.cpu().numpy().astype(np.int32); C_all[ri] = len(ncc)
        nonconst_base[ri] = nb_cur; nonconst_parts.append(ncc); nb_cur += len(ncc)
        cf = cf_t.cpu().numpy().astype(np.uint32); cdf_flat_base[ri] = cf_cur
        cdf_parts.append(cf); cf_cur += len(cf)
        cdf_off_all[ri] = co_t.cpu().numpy().astype(np.int32)
    nonconst_cat = np.concatenate(nonconst_parts).astype(np.int32)
    cdf_flat_cat = np.concatenate(cdf_parts).astype(np.int32)

    t = dict(
        blobs=_T(blob_arr, torch.uint8), page_byte_off=_T(page_byte_off, torch.int64),
        lane_off=_T(lane_off.reshape(-1), torch.int64), k0a=_T(k0a.reshape(-1), torch.int32),
        k1a=_T(k1a.reshape(-1), torch.int32), rung_of_page=_T(rung_of_page, torch.int32),
        nonconst_cat=_T(nonconst_cat, torch.int32), nonconst_base=_T(nonconst_base, torch.int32),
        C_all=_T(C_all, torch.int32), cdf_flat=_T(cdf_flat_cat, torch.int32),
        cdf_flat_base=_T(cdf_flat_base, torch.int32), cdf_off_all=_T(cdf_off_all, torch.int32),
        N=N, P=P, d=d, nb=nb, R=R, nsym=nsym)
    return t

def run_decode(t, lut, use_lut):
    return db.decode_run(t["blobs"], t["page_byte_off"], t["lane_off"], t["k0a"], t["k1a"],
                         t["rung_of_page"], t["nonconst_cat"], t["nonconst_base"], t["C_all"],
                         t["cdf_flat"], t["cdf_flat_base"], t["cdf_off_all"], lut,
                         t["N"], t["P"], t["d"], t["nb"], use_lut)

heads = [(l, h) for l in range(L) for h in range(Hkv)]
step = max(1, len(heads) // N_HEADS); timed = heads[::step][:N_HEADS]
print(f"building tables + LUTs for {len(timed)} heads ...")
tabs, luts = [], []
lut_ms = 0.0
for (l, h) in timed:
    t = build_decode_tables(l, h); tabs.append(t)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    lut = db.build_lut(t["cdf_flat"], t["cdf_flat_base"], t["cdf_off_all"], t["R"], t["d"])
    torch.cuda.synchronize(); lut_ms += (time.perf_counter() - t0) * 1e3
    luts.append(lut)
total_sym = sum(t["nsym"] for t in tabs)

# correctness: LUT must equal binsearch, every head
bad = 0
for t, lut in zip(tabs, luts):
    a = run_decode(t, lut, 0); b = run_decode(t, lut, 1)
    if not torch.equal(a, b): bad += 1
print(f"correctness: {len(tabs)-bad}/{len(tabs)} heads bit-identical (LUT vs binsearch)"
      f"  {'PASS' if bad == 0 else 'FAIL'}")

def bench(use_lut):
    for t, lut in zip(tabs, luts): run_decode(t, lut, use_lut)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        for t, lut in zip(tabs, luts): run_decode(t, lut, use_lut)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / REPS * 1e3

tb = bench(0); tl = bench(1)
print(f"\n--- decode throughput ({len(tabs)} heads, {total_sym/1e6:.1f}M symbols) ---")
print(f"binsearch : {tb:8.2f} ms | {total_sym/(tb/1e3)/1e9:6.2f} G symbols/s | {tb/len(tabs):.3f} ms/head")
print(f"LUT       : {tl:8.2f} ms | {total_sym/(tl/1e3)/1e9:6.2f} G symbols/s | {tl/len(tabs):.3f} ms/head")
print(f"speedup (binsearch/LUT) = {tb/tl:.2f}x")
print(f"LUT build (one-time, amortized to ~0): {lut_ms/len(tabs):.2f} ms/head")
print("\nReference: an fp16 K-read is ~0.6 us/head/token. Decode ms/head is the wall "
      "to beat with later levers (more lanes, interleaved states).")