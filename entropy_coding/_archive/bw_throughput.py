#!/usr/bin/env python3
"""Bandwidth-path decode throughput: full model, per-token, long context.
Compressed paged decode (fused kernel) vs BF16 flash-decoding. tok/s."""
import numpy as np, torch, time, struct, os, shutil
import torch.nn.functional as Fnn
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.cpp_extension import load
import run_pca_ec_deadzone as base
from kvq_codec import build_codecs_from_ladder_rans_cuda, load_ext, BatchRANSEncoder, _unpack

shutil.rmtree("./build_dec", ignore_errors=True); os.makedirs("./build_dec", exist_ok=True)
dec = load(name="fused_decode_attn", sources=["fused_decode_attn.cu"], build_directory="./build_dec", verbose=False)

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
ext = load_ext()
cod = build_codecs_from_ladder_rans_cuda(F, inv, km, ladder, L, Hkv, 2*d*64, 64, 0.375,
                                         lanes=16, ext=ext, device="cuda")
enc = BatchRANSEncoder(cod)
art = torch.load(root / man["examples"][4]["file"], map_location="cpu", weights_only=False)
T0 = int(art["prompt_length"]); gs = art["q_post"].shape[1] // Hkv
sm = 1.0 / np.sqrt(d)
heads = [(l, h) for l in range(1, L) for h in range(Hkv)]   # full model (skip layer 0 sink)

def tile_to(x, T):
    reps = (T + x.shape[0] - 1) // x.shape[0]
    return x.repeat(reps, 1)[:T].contiguous()

# precompute fused-kernel metadata for one head at a given tiled T, reused across the model
def build_meta(c, buf, T):
    N = c.N; P = c.P; R = c.R
    magic, Tt, Phdr, nb = struct.unpack_from("<IIII", buf, 0)
    rung_ids, off0 = _unpack(buf, nb, c.id_bits, 16)
    all_blob = bytearray(); pbo = np.zeros(nb, np.int64); lo = np.zeros((nb, N), np.int64)
    k0a = np.zeros((nb, N), np.int32); k1a = np.zeros((nb, N), np.int32)
    ns = np.zeros(nb, np.int32); rop = np.zeros(nb, np.int32)
    off = off0
    for bi in range(nb):
        blob, off = c._read_blob(buf, off); ri = int(rung_ids[bi]); rop[bi] = ri
        nt = min((bi+1)*Phdr, Tt) - bi*Phdr; ns[bi] = nt
        Ci = int(c._cuda_cdf_tensors[ri][0].shape[0]); S = nt*Ci
        ll = [struct.unpack_from("<I", blob, 8+4*x)[0] for x in range(N)]
        st = np.cumsum([0]+ll); bd = [round(k*S/N) for k in range(N+1)]
        pbo[bi] = len(all_blob); hs = 8+4*N
        for x in range(N):
            lo[bi,x] = hs+int(st[x]); k0a[bi,x] = bd[x]; k1a[bi,x] = bd[x+1]
        all_blob += blob
    blob_arr = np.frombuffer(bytes(all_blob), np.uint8).copy()
    A_max = max(int(c._vals_gpu[ri].shape[1]) for ri in range(R))
    ncp=[]; ncb=np.zeros(R,np.int32); C_all=np.zeros(R,np.int32); cfp=[]; cfb=np.zeros(R,np.int32)
    coa=np.zeros((R,d+1),np.int32); va=np.zeros((R,d,A_max),np.int64); cva=np.zeros((R,d),np.int32)
    nma=np.zeros((R,d),np.uint8); da=np.zeros(R,np.float32); nbc=cfc=0
    for ri in range(R):
        nc_t,cf_t,co_t = c._cuda_cdf_tensors[ri]
        ncc=nc_t.cpu().numpy().astype(np.int32); C_all[ri]=len(ncc); ncb[ri]=nbc; ncp.append(ncc); nbc+=len(ncc)
        cf=cf_t.cpu().numpy().astype(np.uint32); cfb[ri]=cfc; cfp.append(cf); cfc+=len(cf)
        coa[ri]=co_t.cpu().numpy().astype(np.int32); vv=c._vals_gpu[ri].cpu().numpy(); va[ri,:,:vv.shape[1]]=vv
        cva[ri]=c._cv_gpu[ri].cpu().numpy().astype(np.int32); nma[ri]=c._nc_mask_gpu[ri].cpu().numpy().astype(np.uint8)
        da[ri]=float(c._deltas_gpu[ri].reshape(-1)[0])
    def _T(a,dt): return torch.as_tensor(a,dtype=dt,device=dev)
    return [_T(blob_arr,torch.uint8),_T(pbo,torch.int64),_T(lo.reshape(-1),torch.int64),
            _T(k0a.reshape(-1),torch.int32),_T(k1a.reshape(-1),torch.int32),_T(ns,torch.int32),
            _T(rop,torch.int32),_T(np.concatenate(ncp).astype(np.int32),torch.int32),
            _T(ncb,torch.int32),_T(C_all,torch.int32),_T(np.concatenate(cfp).astype(np.int32),torch.int32),
            _T(cfb,torch.int32),_T(coa,torch.int32),_T(va,torch.int64),_T(cva,torch.int32),
            _T(nma,torch.uint8),_T(da,torch.float32)], nb

Ts = [4096, 8192, 16384, 32768]
B = 1   # decode-step query count per head (batch); raise if memory allows
G = 8

def timeit(f, reps=3, warmup=1):
    for _ in range(warmup): f()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(reps): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/reps

from step2b_triton import torch_causal

print(f"{'T':>8} {'bf16_tok/s':>12} {'comp_tok/s':>12} {'speedup':>9}")
for T in Ts:
    metas = {}; kT_all = {}; vT_all = {}
    for (l,h) in heads:
        c = cod[(l,h)]
        kT = tile_to(art["k_post"][l,h,:T0,:].float(), T)
        vT = tile_to(art["v"][l,h,:T0,:].float(), T).to(dev)
        buf = c.encode_gpu(kT.cpu().numpy())
        metas[(l,h)] = (build_meta(c, bytes(buf), T), c)
        vT_all[(l,h)] = vT
        kT_all[(l,h)] = kT.to(dev)
    q1 = {(l,h): torch.randn(gs, 1, d, device=dev) for (l,h) in heads}

    # ---- CORRECTNESS CHECK (one head, first T only) ----
    if T == Ts[0]:
        (l,h) = heads[0]
        (margs, nb), c = metas[(l,h)]
        invT = torch.as_tensor(c.inv, dtype=torch.float32, device=dev)
        qp = (q1[(l,h)] @ invT.T).contiguous()                 # (gs,1,d)
        out = dec.decode_attn(*margs, qp, vT_all[(l,h)], sm, 0.375,
                              margs[13].shape[2], c.N, c.P, d, T, nb, gs)  # (gs,1,d)
        # reference: decode K the trusted way, attend the single query (no causal mask:
        # 1 query attends ALL T keys, which is the decode-step semantics)
        kh = c.decode_to_gpu(bytes(c.encode_gpu(kT_all[(l,h)].cpu().numpy())))[:T].float()
        qf = q1[(l,h)].float()                                  # (gs,1,d) raw q (not projected)
        # ref attention: softmax(qf · kh^T / sqrt d) · vT   (full, no mask, decode step)
        sc = torch.einsum("gqd,td->gqt", qf, kh) * sm           # (gs,1,T)
        w = torch.softmax(sc, dim=-1)
        ref = torch.einsum("gqt,td->gqd", w, vT_all[(l,h)])     # (gs,1,d)
        err = float((out - ref).abs().max())
        print(f"[correctness T={T}] decode_attn vs torch full-attn  max|delta| = {err:.3e}")
        if err > 1e-2:
            print("  WARNING: large error — kernel output suspect, timing below may be meaningless")

    def step_bf16():
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            for (l,h) in heads:
                Fnn.scaled_dot_product_attention(
                    q1[(l,h)].half().unsqueeze(0),
                    kT_all[(l,h)].half().unsqueeze(0).unsqueeze(0).expand(1,gs,T,d),
                    vT_all[(l,h)].half().unsqueeze(0).unsqueeze(0).expand(1,gs,T,d))

    def step_comp():
        for (l,h) in heads:
            (margs, nb), c = metas[(l,h)]
            invT = torch.as_tensor(c.inv, dtype=torch.float32, device=dev)
            qp = (q1[(l,h)] @ invT.T).contiguous()
            dec.decode_attn(*margs, qp, vT_all[(l,h)], sm, 0.375,
                            margs[13].shape[2], c.N, c.P, d, T, nb, gs)

    tb = timeit(step_bf16); tc = timeit(step_comp)
    print(f"{T:>8} {1/tb:>12.1f} {1/tc:>12.1f} {tb/tc:>8.3f}x")

print("\ntok/s = 1 / per-token full-model time. speedup = bf16_time / comp_time.")