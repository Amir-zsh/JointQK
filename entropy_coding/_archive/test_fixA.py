#!/usr/bin/env python3
"""Full-grid grp timing: all heads launched, occupancy-fair vs fp16 full-grid.
Decodes compressed K in-kernel; fp16 baseline reads materialized fp16 K."""
import numpy as np, torch, struct, os, shutil, time
from torch.utils.cpp_extension import load
import torch.nn.functional as Fnn
from torch.nn.attention import sdpa_kernel, SDPBackend
import run_pca_ec_deadzone as base
from kvq_codec import build_codecs_from_ladder_rans_cuda, load_ext, BatchRANSEncoder, _unpack
from step2b_triton import torch_causal

shutil.rmtree("./build_grp", ignore_errors=True); os.makedirs("./build_grp", exist_ok=True)
grp = load(name="fused_attn_grp", sources=["fused_attn_grp.cu"], build_directory="./build_grp", verbose=False)

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
dev = torch.device("cuda")
sm = 1.0 / np.sqrt(d)
G = 8

def _T(a, dt): return torch.as_tensor(a, dtype=dt, device=dev)

def build_head(l, h):
    c = cod[(l, h)]; N = c.N; P = c.P; R = c.R
    buf = bytes(bufs[(l, h)])
    magic, Tt, Phdr, nb = struct.unpack_from("<IIII", buf, 0)
    rung_ids, off0 = _unpack(buf, nb, c.id_bits, 16)
    all_blob = bytearray(); page_byte_off = np.zeros(nb, np.int64)
    lane_off = np.zeros((nb, N), np.int64); k0a = np.zeros((nb, N), np.int32)
    k1a = np.zeros((nb, N), np.int32); ns_arr = np.zeros(nb, np.int32); rop = np.zeros(nb, np.int32)
    off = off0
    for bi in range(nb):
        blob, off = c._read_blob(buf, off); ri = int(rung_ids[bi]); rop[bi] = ri
        n_toks = min((bi+1)*Phdr, Tt) - bi*Phdr; ns_arr[bi] = n_toks
        Ci = int(c._cuda_cdf_tensors[ri][0].shape[0]); S = n_toks*Ci
        lane_lens = [struct.unpack_from("<I", blob, 8+4*l_)[0] for l_ in range(N)]
        starts = np.cumsum([0]+lane_lens); bounds = [round(k*S/N) for k in range(N+1)]
        page_byte_off[bi] = len(all_blob); hs = 8+4*N
        for l_ in range(N):
            lane_off[bi,l_] = hs+int(starts[l_]); k0a[bi,l_] = bounds[l_]; k1a[bi,l_] = bounds[l_+1]
        all_blob += blob
    blob_arr = np.frombuffer(bytes(all_blob), dtype=np.uint8).copy()
    A_max = max(int(c._vals_gpu[ri].shape[1]) for ri in range(R))
    ncp = []; ncb = np.zeros(R, np.int32); C_all = np.zeros(R, np.int32)
    cfp = []; cfb = np.zeros(R, np.int32); coa = np.zeros((R, d+1), np.int32)
    va = np.zeros((R, d, A_max), np.int64); cva = np.zeros((R, d), np.int32)
    nma = np.zeros((R, d), np.uint8); da = np.zeros(R, np.float32)
    nbc = cfc = 0
    for ri in range(R):
        nc_t, cf_t, co_t = c._cuda_cdf_tensors[ri]
        ncc = nc_t.cpu().numpy().astype(np.int32); C_all[ri] = len(ncc)
        ncb[ri] = nbc; ncp.append(ncc); nbc += len(ncc)
        cf = cf_t.cpu().numpy().astype(np.uint32); cfb[ri] = cfc; cfp.append(cf); cfc += len(cf)
        coa[ri] = co_t.cpu().numpy().astype(np.int32)
        vv = c._vals_gpu[ri].cpu().numpy(); va[ri,:,:vv.shape[1]] = vv
        cva[ri] = c._cv_gpu[ri].cpu().numpy().astype(np.int32)
        nma[ri] = c._nc_mask_gpu[ri].cpu().numpy().astype(np.uint8)
        da[ri] = float(c._deltas_gpu[ri].reshape(-1)[0])
    q = art["q_post"][l, h*gs:(h+1)*gs, :T, :].to(dev).float()
    invT = torch.as_tensor(c.inv, dtype=torch.float32, device=dev)
    q_proj = (q @ invT.T).contiguous()
    v = art["v"][l, h, :T, :].to(dev).float().contiguous()
    return [_T(blob_arr, torch.uint8), _T(page_byte_off, torch.int64), _T(lane_off.reshape(-1), torch.int64),
            _T(k0a.reshape(-1), torch.int32), _T(k1a.reshape(-1), torch.int32), _T(ns_arr, torch.int32),
            _T(rop, torch.int32), _T(np.concatenate(ncp).astype(np.int32), torch.int32),
            _T(ncb, torch.int32), _T(C_all, torch.int32),
            _T(np.concatenate(cfp).astype(np.int32), torch.int32), _T(cfb, torch.int32), _T(coa, torch.int32),
            _T(va, torch.int64), _T(cva, torch.int32), _T(nma, torch.uint8), _T(da, torch.float32),
            q_proj, v, sm, 0.375, A_max, N, P, d, T, nb, gs]

print("building all-head metadata (one-time)...")
heads = [(l, h) for l in range(1, L) for h in range(Hkv)]
head_args = {(l, h): build_head(l, h) for (l, h) in heads}
print(f"{len(heads)} heads built")

# fp16 K resident for all heads
kh_all = {(l, h): cod[(l,h)].decode_to_gpu(bytes(bufs[(l,h)]))[:T].float() for (l,h) in heads}
v_all = {(l, h): art["v"][l, h, :T, :].to(dev).float().contiguous() for (l,h) in heads}
q_all = {(l, h): art["q_post"][l, h*gs:(h+1)*gs, :T, :].to(dev).float() for (l,h) in heads}

def run_grp_grid():
    for (l, h) in heads:
        grp.fused_attn_grp(*head_args[(l, h)], G)

def run_fp16_grid():
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        for (l, h) in heads:
            Fnn.scaled_dot_product_attention(
                q_all[(l,h)].half().unsqueeze(0),
                kh_all[(l,h)].half().unsqueeze(0).unsqueeze(0).expand(1,gs,T,d),
                v_all[(l,h)].half().unsqueeze(0).unsqueeze(0).expand(1,gs,T,d), is_causal=True)

run_grp_grid(); run_fp16_grid(); torch.cuda.synchronize()
def timeit(f, reps=3):
    t0=time.perf_counter()
    for _ in range(reps): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/reps*1e3
tg = timeit(run_grp_grid); tf = timeit(run_fp16_grid)
print(f"\n--- FULL GRID ({len(heads)} heads, T={T}) ---")
print(f"grp fused (G={G}) : {tg:8.1f} ms")
print(f"fp16 flash        : {tf:8.1f} ms")
print(f"ratio (fused/fp16) = {tg/tf:.1f}x")