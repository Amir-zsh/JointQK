import numpy as np, torch
import run_pca_ec_deadzone as base
import run_pca_ec_rdo as old
from test_codec_on_data import build_ladder
from kvq_codec_rdo import build_codecs_from_ladder_rdo
import math

CALIB=[0,1,2]; B=4; PTOK=16; DZ=0.375
M=sorted({0.5,0.625,0.75,0.875,1.0,1.25,1.5,2.0,3.0,8.0})
root=base.data_root(); man=base.load_manifest(root)
sq,sk,km,kc,meta=base.calib_moments(root,man,CALIB)
L,Hkv,d=meta['n_layers'],meta['n_kv_heads'],meta['d_head']
qu=base.build_qpca_basis(sq,sk); qu['sigma_k'],qu['sigma_q']=sk,sq
qc=base.build_qpca_basis(sq,kc); qc['sigma_k']=sk
F=qc['forward']; inv=qc['inverse']
fc=base._codes_for_idx(root,man,CALIB,F,km,L,Hkv,d)

# OLD ladder (run_paged_split path)
rungs_old,_,_=old.build_ladder(qc,qu,km,B,L,Hkv,fc,root,CALIB,M,DZ)
hdr=math.ceil(math.log2(max(2,len(M))))
old_rt=old.PagedRDORoundtrip(F[1,0],inv[1,0],km[1,0],
    [(m,dm[1,0],mod[(1,0)]) for (m,dm,mod) in rungs_old],
    B*d*PTOK, old.FLUSH_BITS, hdr, DZ, PTOK)

# NEW ladder (codec path)
lad,_,_=build_ladder(qc,qu,km,B,L,Hkv,fc,root,CALIB,M,DZ)
cod=build_codecs_from_ladder_rdo(F,inv,km,lad,L,Hkv,B*d*PTOK,PTOK,DZ,lanes=8)
new=cod[(1,0)]

# one real page
art=torch.load(root/man['examples'][4]['file'],map_location='cpu',weights_only=False)
k=art['k_post'][1,0,:PTOK,:].float()                 # first page only
r=((k.double().numpy()) - new.mu) @ new.fwd

# OLD per-token rungs on this page
old_rt.to('cpu')
import torch as T
kt=k.clone()
# replicate old roundtrip's choose for just this page:
rr=(kt - old_rt.mu)@old_rt.fwd
rn=rr.double().numpy()
D=np.zeros((old_rt.R,PTOK)); Rt=np.zeros((old_rt.R,PTOK))
for ri,(dl,dn,mdl) in enumerate(zip(old_rt.delta_dev,old_rt.delta_np,old_rt.models)):
    idx=base._dz_round(rr,dl,DZ); q=base._dz_dequant(idx,dl,DZ)
    D[ri]=((q-rr)**2).sum(1).numpy(); Rt[ri]=old._tok_bits_np(rn,dn,mdl,DZ,d)
budget=B*d*PTOK - old_rt.flush - PTOK*old_rt.hdr
lam_hi=max(1.0,float(D.max()))*1e3; lo=np.zeros(1); hi=np.full(1,lam_hi); ar=np.arange(PTOK)
for _ in range(40):
    mid=0.5*(lo+hi); ch=(D+np.repeat(mid,PTOK)[None,:]*Rt).argmin(0)
    br=Rt[ch,ar].sum()
    if br>budget: lo=mid
    else: hi=mid
old_ch=(D+np.repeat(hi,PTOK)[None,:]*Rt).argmin(0)

# NEW per-token rungs on this page
Dn,Rn=new._score_all_rungs(r)
side=8*(12+2*PTOK+4*new.N)
new_ch=new._alloc_page(Dn[:PTOK],Rn[:PTOK],new.page_bits-side)

print("old rungs:", old_ch)
print("new rungs:", new_ch)
print("agree:", int((old_ch==new_ch).sum()), "/", PTOK)
print("old budget:", budget, " new budget:", new.page_bits-side)
print("D max diff:", np.abs(D.T[:PTOK]-Dn[:PTOK]).max())
print("R max diff:", np.abs(Rt.T[:PTOK]-Rn[:PTOK]).max())