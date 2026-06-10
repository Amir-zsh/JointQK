import numpy as np, torch
import run_pca_ec_deadzone as base
from test_codec_on_data import build_ladder
from kvq_codec_cuda import build_codecs_from_ladder_rans_cuda, BatchRANSEncoder

CALIB=[0,1,2]; B=4; PTOK=64; DZ=0.375; N=8
M=sorted({1.0,1.025,1.05,1.1,1.15,1.25,1.5,1.75,2.0})
root=base.data_root(); man=base.load_manifest(root)
sq,sk,km,kc,meta=base.calib_moments(root,man,CALIB)
L,Hkv,d=meta['n_layers'],meta['n_kv_heads'],meta['d_head']
qu=base.build_qpca_basis(sq,sk); qu['sigma_k'],qu['sigma_q']=sk,sq
qc=base.build_qpca_basis(sq,kc); qc['sigma_k']=sk
fc=base._codes_for_idx(root,man,CALIB,qc['forward'],km,L,Hkv,d)
lad,_,_=build_ladder(qc,qu,km,B,L,Hkv,fc,root,CALIB,M,DZ)
codecs=build_codecs_from_ladder_rans_cuda(qc['forward'],qc['inverse'],km,lad,L,Hkv,B*d*PTOK,PTOK,DZ,lanes=N,device='cuda')

art=torch.load(root/man['examples'][4]['file'],map_location='cpu',weights_only=False)
k=art['k_post']; T=int(art['prompt_length'])
# test on a subset of heads for speed
test_heads=[(l,h) for l in range(4) for h in range(Hkv)]
k_grid={lh: k[lh[0],lh[1],:T,:].float() for lh in test_heads}

enc=BatchRANSEncoder({lh:codecs[lh] for lh in test_heads})
buf_batch=enc.encode_grid(k_grid)

mism=0
for lh in test_heads:
    bp=codecs[lh].encode_gpu(k_grid[lh])
    bb=buf_batch[lh]
    if bp!=bb:
        mism+=1
        print(f"head {lh}: per-head {len(bp)}B vs batch {len(bb)}B  identical={bp==bb}")
print(f"heads identical: {len(test_heads)-mism}/{len(test_heads)}")