#!/usr/bin/env python3
"""
Fix A (corrected) — validate + sweep G for the query-tile-GROUP kernel (fused_attn_grp.cu).

  1. correctness: head (1,0) vs torch_causal  -> expect ~1e-3 / 6.4e-7
  2. timing: sweep G in {1,2,4,8,16}, full grid over N_HEADS heads, ms/head.
     G=1 should ~= your naive 130 ms/head (sanity: grouping is the only change).
     Larger G -> fewer re-decodes, but bigger blocks (register-pressure caps it).
     Expect the curve to flatten near the ~20 ms/head decode-serial floor; the G that
     hits it cheapest is the one to keep. Breaking below ~20 ms is split-K's job next.

Run from entropy_coding/ alongside test_fused.py.
"""
import numpy as np, torch, struct, shutil, os, time
from torch.utils.cpp_extension import load
import run_pca_ec_deadzone as base
from kvq_codec import build_codecs_from_ladder_rans_cuda, load_ext, BatchRANSEncoder, _unpack
from step2b_triton import torch_causal
import torch.nn.functional as Fnn
from torch.nn.attention import sdpa_kernel, SDPBackend

N_HEADS = 32
REPS    = 3
G_SWEEP = [1, 2, 4, 8, 16]
dev = torch.device("cuda")

shutil.rmtree("./build_fused_grp", ignore_errors=True); os.makedirs("./build_fused_grp", exist_ok=True)
fa = load(name="fused_attn_grp", sources=["fused_attn_grp.cu"],
          build_directory="./build_fused_grp", verbose=True)

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
T = int(art["prompt_length"]); gs = art["q_post"].shape[1] // Hkv
kg = {(l, h): art["k_post"][l, h, :T, :].float() for l in range(L) for h in range(Hkv)}
bufs = enc.encode_grid(kg)
sm = 1.0 / np.sqrt(d)

def _T(a, dt): return torch.as_tensor(a, dtype=dt, device=dev)

def build_head_args(l, h, G):
    c = cod[(l, h)]; N = c.N; P = c.P; R = c.R
    buf = bytes(bufs[(l, h)])
    magic, Tt, Phdr, nb = struct.unpack_from("<IIII", buf, 0)
    rung_ids, off0 = _unpack(buf, nb, c.id_bits, 16)
    all_blob = bytearray(); page_byte_off = np.zeros(nb, np.int64)
    lane_off = np.zeros((nb, N), np.int64); k0a = np.zeros((nb, N), np.int32)
    k1a = np.zeros((nb, N), np.int32); ns_arr = np.zeros(nb, np.int32)
    rung_of_page = np.zeros(nb, np.int32)
    off = off0
    for bi in range(nb):
        blob, off = c._read_blob(buf, off)
        ri = int(rung_ids[bi]); rung_of_page[bi] = ri
        n_toks = min((bi + 1) * Phdr, Tt) - bi * Phdr; ns_arr[bi] = n_toks
        Ci = int(c._cuda_cdf_tensors[ri][0].shape[0]); S = n_toks * Ci
        lane_lens = [struct.unpack_from("<I", blob, 8 + 4*ln)[0] for ln in range(N)]
        starts = np.cumsum([0] + lane_lens); bounds = [round(k*S/N) for k in range(N+1)]
        page_byte_off[bi] = len(all_blob)
        hs = 8 + 4*N
        for ln in range(N):
            lane_off[bi, ln] = hs + int(starts[ln]); k0a[bi, ln] = bounds[ln]; k1a[bi, ln] = bounds[ln+1]
        all_blob += blob
    blob_arr = np.frombuffer(bytes(all_blob), dtype=np.uint8).copy()

    A_max = max(int(c._vals_gpu[ri].shape[1]) for ri in range(R))
    nonconst_parts = []; nonconst_base = np.zeros(R, np.int32); C_all = np.zeros(R, np.int32)
    cdf_parts = []; cdf_flat_base = np.zeros(R, np.int32)
    cdf_off_all = np.zeros((R, d + 1), np.int32)
    vals_all = np.zeros((R, d, A_max), np.int64); cv_all = np.zeros((R, d), np.int32)
    nc_mask_all = np.zeros((R, d), np.uint8); delta_all = np.zeros(R, np.float32)
    nb_cur = 0; cf_cur = 0
    for ri in range(R):
        nc_t, cf_t, co_t = c._cuda_cdf_tensors[ri]
        ncc = nc_t.cpu().numpy().astype(np.int32); C_all[ri] = len(ncc)
        nonconst_base[ri] = nb_cur; nonconst_parts.append(ncc); nb_cur += len(ncc)
        cf = cf_t.cpu().numpy().astype(np.uint32); cdf_flat_base[ri] = cf_cur
        cdf_parts.append(cf); cf_cur += len(cf)
        cdf_off_all[ri] = co_t.cpu().numpy().astype(np.int32)
        vv = c._vals_gpu[ri].cpu().numpy(); vals_all[ri, :, :vv.shape[1]] = vv
        cv_all[ri] = c._cv_gpu[ri].cpu().numpy().astype(np.int32)
        nc_mask_all[ri] = c._nc_mask_gpu[ri].cpu().numpy().astype(np.uint8)
        delta_all[ri] = float(c._deltas_gpu[ri].reshape(-1)[0])
    nonconst_cat = np.concatenate(nonconst_parts).astype(np.int32)
    cdf_flat_cat = np.concatenate(cdf_parts).astype(np.int32)

    q = art["q_post"][l, h*gs:(h+1)*gs, :T, :].to(dev).float()
    invT = torch.as_tensor(c.inv, dtype=torch.float32, device=dev)
    q_proj = (q @ invT.T).contiguous()
    vv = art["v"][l, h, :T, :].to(dev).float().contiguous()

    args = (
        _T(blob_arr, torch.uint8), _T(page_byte_off, torch.int64), _T(lane_off.reshape(-1), torch.int64),
        _T(k0a.reshape(-1), torch.int32), _T(k1a.reshape(-1), torch.int32), _T(ns_arr, torch.int32),
        _T(rung_of_page, torch.int32),
        _T(nonconst_cat, torch.int32), _T(nonconst_base, torch.int32), _T(C_all, torch.int32),
        _T(cdf_flat_cat, torch.int32), _T(cdf_flat_base, torch.int32), _T(cdf_off_all, torch.int32),
        _T(vals_all, torch.int64), _T(cv_all, torch.int32), _T(nc_mask_all, torch.uint8),
        _T(delta_all, torch.float32),
        q_proj, vv, sm, 0.375, A_max, N, P, d, T, nb, gs, G)
    kh = c.decode_to_gpu(buf)[:T].float()
    return args, (q, kh, vv)

# --------------------------- correctness (G=8) ---------------------
args, (q, kh, v) = build_head_args(1, 0, G=8)
out = fa.fused_attn_grp(*args)
ref = torch_causal(q, kh, v, sm)
err = float((out - ref).abs().max())
print(f"\nGROUP kernel (G=8) vs torch causal-on-K  max|delta| = {err:.3e}   "
      f"({'PASS' if err < 5e-3 else 'FAIL'})")

# --------------------------- timing sweep --------------------------
heads = [(l, h) for l in range(L) for h in range(Hkv)]
step  = max(1, len(heads) // N_HEADS); timed = heads[::step][:N_HEADS]
print(f"building args for {len(timed)} heads x {len(G_SWEEP)} values of G ...")

def bench_run(run):
    run(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS): run()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/REPS*1e3

# fp16 baseline
fp16_packs = [build_head_args(l, h, G=1)[1] for (l, h) in timed]
def run_fp16():
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        for (q, kh, v) in fp16_packs:
            qh = q.half().unsqueeze(0)
            k16 = kh.half().unsqueeze(0).unsqueeze(0).expand(1, gs, T, d)
            v16 = v.half().unsqueeze(0).unsqueeze(0).expand(1, gs, T, d)
            Fnn.scaled_dot_product_attention(qh, k16, v16, is_causal=True)
tg = bench_run(run_fp16)

print(f"\n--- group-kernel G sweep ({len(timed)} heads, T={T}, gs={gs}) ---")
print(f"{'G':>3} {'block_threads':>14} {'ms/head':>10} {'vs fp16':>9}")
for G in G_SWEEP:
    packs = [build_head_args(l, h, G=G)[0] for (l, h) in timed]
    run = lambda packs=packs: [fa.fused_attn_grp(*a) for a in packs]
    t = bench_run(run) / len(timed)
    print(f"{G:>3} {G*32:>14} {t:>10.3f} {t/(tg/len(timed)):>8.1f}x")
    del packs; torch.cuda.empty_cache()
print(f"\nfp16 flash: {tg/len(timed):.3f} ms/head  (naive was ~130 ms/head, 544x)")
print("Pick the G at the knee. Sub-~20ms needs split-K (parallelize the per-group "
      "serial decode of the late groups) — that's the next build.")