import numpy as np, torch, struct
import run_pca_ec_deadzone as base
from test_codec_on_data import build_ladder
from kvq_codec_cuda import build_codecs_from_ladder_rans_cuda, load_ext
import rans_interleaved as ri

CALIB=[0,1,2]; B=4; PTOK=64; DZ=0.375; N=8
M=sorted({1.0,1.025,1.05,1.1,1.15,1.25,1.5,1.75,2.0})
root=base.data_root(); man=base.load_manifest(root)
sq,sk,km,kc,meta=base.calib_moments(root,man,CALIB)
L,Hkv,d=meta['n_layers'],meta['n_kv_heads'],meta['d_head']
qu=base.build_qpca_basis(sq,sk); qu['sigma_k'],qu['sigma_q']=sk,sq
qc=base.build_qpca_basis(sq,kc); qc['sigma_k']=sk
fc=base._codes_for_idx(root,man,CALIB,qc['forward'],km,L,Hkv,d)
lad,_,_=build_ladder(qc,qu,km,B,L,Hkv,fc,root,CALIB,M,DZ)
ext=load_ext()
cu=build_codecs_from_ladder_rans_cuda(qc['forward'],qc['inverse'],km,lad,L,Hkv,B*d*PTOK,PTOK,DZ,lanes=N,ext=ext,device='cuda')[(1,0)]
art=torch.load(root/man['examples'][4]['file'],map_location='cpu',weights_only=False)
k=art['k_post']; T=int(art['prompt_length']); kd=k[1,0,:T,:].float()

print('SCALE_BITS:', ri.SCALE_BITS, ' RANS_L:', ri.RANS_L)   # MUST be 14, 1<<23
assert ri.SCALE_BITS==14, "SCALE_BITS mismatch with kernel"

# run CPU encode, capture the ground-truth buffer (this is what GPU must reproduce)
buf_cpu = cu.encode(kd)

# also dump intermediate: chosen rungs, and for page 0 the per-symbol freq/start
from kvq_codec import _unpack
nb=(T+PTOK-1)//PTOK
rid,_=_unpack(buf_cpu,nb,cu.id_bits,16)
r=(kd.double().numpy()-cu.mu)@cu.fwd

# page 0 symbol-order freq/start at its chosen rung (input to the encode kernel)
ri0=int(rid[0])
n0=min(PTOK,T)
pos0=cu._snap_rung(r[:n0],ri0) if not isinstance(cu._snap_rung(r[:n0],ri0),tuple) else cu._snap_rung(r[:n0],ri0,want_bits=False)
pos0=np.asarray(pos0)
nc=[j for j in range(d) if not cu.models[ri0][j].constant]
C=len(nc)
order=[(t,j) for t in range(n0) for j in nc]   # token-major, matches encode_page
freq=np.empty(len(order),np.int64); start=np.empty(len(order),np.int64)
for idx,(t,j) in enumerate(order):
    ft=cu.rtab[ri0][j] if hasattr(cu,'rtab') else None
    # use the codec's freq table; rebuild via FreqTable to be safe
    m=cu.models[ri0][j]
    tab=ri.FreqTable(m.vals,m.p)
    p=int(pos0[t,j])
    freq[idx]=tab.freq[p]; start[idx]=tab.cdf[p]

# encode page 0 lanes via CPU reference -> ground truth lane bytes
b=ri._lane_bounds(len(order),N)
lane_blobs=[ri._encode_lane_nb(freq[b[l]:b[l+1]],start[b[l]:b[l+1]]) for l in range(N)]
print('page0 rung:',ri0,' C:',C,' n0:',n0,' S:',len(order))
print('lane lengths:',[len(x) for x in lane_blobs])

np.savez('encode_fixtures.npz',
    d=d,N=N,P=PTOK,SB=ri.SCALE_BITS,RANS_L=ri.RANS_L,
    rid=rid.astype(np.int64), ri0=ri0, C=C, n0=n0,
    freq_p0=freq, start_p0=start,
    lane_lens_p0=np.array([len(x) for x in lane_blobs],np.int64),
    lane_bytes_p0=np.frombuffer(b''.join(bytes(x) for x in lane_blobs),np.uint8),
    buf_cpu=np.frombuffer(buf_cpu,np.uint8))
print('wrote encode_fixtures.npz | buf_cpu bytes:',len(buf_cpu))