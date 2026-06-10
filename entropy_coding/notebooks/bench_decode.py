#!/usr/bin/env python3
import time, numpy as np, torch
import run_pca_ec_deadzone as base
from kvq_codec_cuda import build_codecs_from_ladder_rans_cuda, load_ext, BatchRANSDecoder
from test_codec_on_data import build_ladder

CALIB=[0,1,2]; EVAL=4; B=4; PTOK=64; DZ=0.375
M_GRID=sorted({1.0,1.025,1.05,1.1,1.15,1.25,1.5,1.75,2.0})

root=base.data_root(); manifest=base.load_manifest(root)
sig_q,sig_k,k_mean,k_cov,meta=base.calib_moments(root,manifest,CALIB)
L,Hkv,d=meta["n_layers"],meta["n_kv_heads"],meta["d_head"]
qpca_unc=base.build_qpca_basis(sig_q,sig_k); qpca_unc["sigma_k"],qpca_unc["sigma_q"]=sig_k,sig_q
qpca_cen=base.build_qpca_basis(sig_q,k_cov); qpca_cen["sigma_k"]=sig_k
F=qpca_cen["forward"]; inv=qpca_cen["inverse"]
fetch_calib=base._codes_for_idx(root,manifest,CALIB,F,k_mean,L,Hkv,d)
ladder,_,_=build_ladder(qpca_cen,qpca_unc,k_mean,B,L,Hkv,fetch_calib,root,CALIB,M_GRID,DZ)

ext=load_ext()
codecs=build_codecs_from_ladder_rans_cuda(F,inv,k_mean,ladder,L,Hkv,B*d*PTOK,PTOK,DZ,lanes=1,ext=ext,device="cuda")

art=torch.load(root/manifest["examples"][EVAL]["file"],map_location="cpu",weights_only=False)
k_all=art["k_post"]; T=int(art["prompt_length"])

# encode ONE layer's heads (8) -> bounded memory, real async batch
LYR=1
bufs={(LYR,h):codecs[(LYR,h)].encode(k_all[LYR,h,:T,:].float()) for h in range(Hkv)}
coords=Hkv*T*d

dec=BatchRANSDecoder(codecs)
for _ in range(3):                      # warmup (JIT, caches, clocks)
    dec.decode_grid(bufs)
torch.cuda.synchronize()

REPS=20
t0=time.perf_counter()
for _ in range(REPS):
    dec.decode_grid(bufs)
torch.cuda.synchronize()
wall=(time.perf_counter()-t0)/REPS

tot=dec.last_total_ms; ker=dec.last_kernel_ms
print(f"\n=== decode throughput (b={B}, layer {LYR}, {Hkv} heads, T={T}, coords={coords:,}) ===")
print(f"GPU total   : {tot:.2f} ms  -> {coords/(tot*1e-3):,.0f} sym/s")
print(f"entropy kern: {ker:.2f} ms  -> {coords/(ker*1e-3):,.0f} sym/s  ({100*ker/tot:.1f}% of GPU)")
print(f"wall/call   : {wall*1e3:.2f} ms  -> {coords/wall:,.0f} sym/s (incl. CPU host work)")