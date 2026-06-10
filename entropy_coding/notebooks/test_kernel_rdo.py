#!/usr/bin/env python3
import struct
import numpy as np
import torch
from torch.utils.cpp_extension import load

f = np.load("kernel_fixtures_rdo.npz")
d=int(f["d"]); N=int(f["N"]); R=int(f["R"]); P=64
nc0=f["nc0"].astype(np.int32); C=nc0.size
cdf=f["cdf"].astype(np.int64)                 # uint32 values fit in int64->int32 cast below
off2d=f["off2d"].astype(np.int64)
ns=f["ns"].astype(np.int32); n_pages=int(f["n_pages"])
blob_lens=f["blob_lens"].astype(np.int64)
blobs=f["blobs"].astype(np.uint8)
expected=f["expected_pos"].astype(np.int64)

page_byte_off=np.concatenate([[0],np.cumsum(blob_lens)])[:-1].astype(np.int64)
lane_off=np.zeros((n_pages,N),np.int64)
k0a=np.zeros((n_pages,N),np.int32); k1a=np.zeros((n_pages,N),np.int32)
rung_tok=np.zeros((n_pages,P),np.int32)
for p in range(n_pages):
    base=int(page_byte_off[p]); blob=blobs[base:base+int(blob_lens[p])].tobytes()
    Nh,S,nn=struct.unpack_from("<III",blob,0); o=12
    rt=np.frombuffer(blob,"<u2",nn,o).astype(np.int32); o+=2*nn
    rung_tok[p,:nn]=rt
    lane_len=[struct.unpack_from("<I",blob,o+4*l)[0] for l in range(N)]
    o += 4*N                                    # <-- add this line    
    starts=np.cumsum([0]+lane_len); bounds=[round(k*S/N) for k in range(N+1)]
    for l in range(N):
        lane_off[p,l]=o+int(starts[l]); k0a[p,l]=bounds[l]; k1a[p,l]=bounds[l+1]

dev="cuda"
ext=load(name="rans_decode_rdo", sources=["rans_decode_rdo.cu"], verbose=True)
def T(a,dt): return torch.as_tensor(a,dtype=dt,device=dev)
pos=ext.decode_pages_rdo(
    T(blobs,torch.uint8), T(page_byte_off,torch.int64), T(lane_off.reshape(-1),torch.int64),
    T(k0a.reshape(-1),torch.int32), T(k1a.reshape(-1),torch.int32),
    T(rung_tok.reshape(-1),torch.int32), T(nc0,torch.int32),
    T(cdf,torch.int32), T(off2d,torch.int64), N, C, P, d, R).cpu().numpy()

off=0; ok=True
for p in range(n_pages):
    n=int(ns[p]); exp=expected[off:off+n*d].reshape(n,d); off+=n*d
    got=pos[p,:n,:]
    # compare only coded coords (nc0); constants are 0 in both
    if not np.array_equal(got[:,nc0], exp[:,nc0]):
        ok=False; bad=np.where(got[:,nc0]!=exp[:,nc0])
        print(f"  page {p}: MISMATCH {len(bad[0])}, first {(int(bad[0][0]),int(bad[1][0]))}")
print("RDO KERNEL bit-exact vs golden fixtures:", ok)