import numpy as np, torch
import run_pca_ec_deadzone as base
from test_codec_on_data import build_ladder
from kvq_codec_cuda import build_codecs_from_ladder_rans_cuda

CALIB=[0,1,2]; B=4; PTOK=64; DZ=0.375; N=8
M=sorted({1.0,1.025,1.05,1.1,1.15,1.25,1.5,1.75,2.0})
root=base.data_root(); man=base.load_manifest(root)
sq,sk,km,kc,meta=base.calib_moments(root,man,CALIB)
L,Hkv,d=meta['n_layers'],meta['n_kv_heads'],meta['d_head']
qu=base.build_qpca_basis(sq,sk); qu['sigma_k'],qu['sigma_q']=sk,sq
qc=base.build_qpca_basis(sq,kc); qc['sigma_k']=sk
fc=base._codes_for_idx(root,man,CALIB,qc['forward'],km,L,Hkv,d)
lad,_,_=build_ladder(qc,qu,km,B,L,Hkv,fc,root,CALIB,M,DZ)

# factory compiles BOTH kernels (decode + encode) and shares them across heads
cu=build_codecs_from_ladder_rans_cuda(
    qc['forward'],qc['inverse'],km,lad,L,Hkv,
    B*d*PTOK,PTOK,DZ,lanes=N,device='cuda')[(1,0)]

art=torch.load(root/man['examples'][4]['file'],map_location='cpu',weights_only=False)
k=art['k_post']; T=int(art['prompt_length']); kd=k[1,0,:T,:].float()

buf_cpu=cu.encode(kd)         # CPU reference encoder
buf_gpu=cu.encode_gpu(kd)     # integrated GPU encoder (class method)

print('CPU bytes:',len(buf_cpu),' GPU bytes:',len(buf_gpu))

# decode both through the GPU decoder, compare k_hat
kh_cpu=cu.decode(buf_cpu)
kh_gpu=cu.decode(buf_gpu)
print('k_hat CPU-buf vs GPU-buf max diff:', float(np.abs(kh_cpu-kh_gpu).max()))
print('GPU-buf k_hat vs original k max diff:', float(np.abs(kh_gpu-kd.numpy()).max()))
print('CPU-buf k_hat vs original k max diff:', float(np.abs(kh_cpu-kd.numpy()).max()))
print('rate CPU:', len(buf_cpu)*8/(T*d), ' rate GPU:', len(buf_gpu)*8/(T*d))