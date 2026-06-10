#!/usr/bin/env python3
"""How much rate would entropy-coding the Lloyd-Max bin indices recover?

For the current QPCA config (centered basis+input, Σ_K std, coord-0 widened),
quantize each coord and compute the ENTROPY H_j of its bin-index distribution on
calibration keys. Compare to the allocated grid-bits b_j:
  - sum(b_j)  = what fixed-width storage costs now
  - sum(H_j)  = what entropy coding would cost (same reconstruction, same top-1)
The gap is recoverable rate. Special attention to coord 0: if its wide 8-bit
codebook carries few bits of entropy, the widen is nearly free under entropy coding.
"""
import json, math, torch
from pathlib import Path
EPS = 1e-4

def _sym(x): return 0.5*(x+x.transpose(-1,-2))
def regularize_batch(cov, eps):
    d=cov.shape[-1]; sym=_sym(cov); tr=sym.diagonal(dim1=-2,dim2=-1).sum(-1)
    return sym+(eps*(tr/d).clamp_min(1e-12)).unsqueeze(-1).unsqueeze(-1)*torch.eye(d,dtype=sym.dtype)

def build_qpca_basis(sigma_q, sigma_k):
    sq=_sym(sigma_q.double()); sk=_sym(sigma_k.double())
    ev,U=torch.linalg.eigh(sq)
    sqrt_mq=U@torch.diag_embed(ev.clamp_min(1e-30).sqrt())@U.transpose(-1,-2)
    A=_sym(sqrt_mq@sk@sqrt_mq); lam,V=torch.linalg.eigh(A)
    o=torch.argsort(lam,dim=-1,descending=True)
    lam=torch.gather(lam,-1,o).clamp_min(1e-30)
    V=torch.gather(V,-1,o.unsqueeze(-2).expand(*V.shape[:-1],-1))
    fwd=sqrt_mq@V
    return {"forward":fwd,"score":(fwd.transpose(-1,-2)@sq@fwd).diagonal(dim1=-2,dim2=-1).clamp_min(1e-30)
            *( (fwd.transpose(-1,-2)@sk@fwd).diagonal(dim1=-2,dim2=-1).clamp_min(1e-30) )}

def water_fill_bits(score, b, d, max_bits=8):
    # simple integer water-fill: bits_j = clamp(0.5*log2(score_j)+c), tune c to hit sum=b*d
    ls=0.5*score.clamp_min(1e-30).log2()
    lo,hi=-50.0,50.0
    for _ in range(60):
        c=(lo+hi)/2
        bits=(ls+c).clamp(0,max_bits).round()
        if bits.sum()>b*d: hi=c
        else: lo=c
    return (ls+ (lo+hi)/2).clamp(0,max_bits).round().long()

def gaussian_centroids(bits):
    n=2**int(bits)
    if n<=1: return torch.zeros(1,dtype=torch.float64)
    x=torch.linspace(-12,12,40001,dtype=torch.float64); p=torch.exp(-0.5*x**2); p/=p.sum()
    c=torch.quantile(x,torch.linspace(0.5/n,1-0.5/n,n).double())
    for _ in range(60):
        e=(c[1:]+c[:-1])/2; idx=torch.bucketize(x,e)
        ws=torch.zeros(n,dtype=torch.double).scatter_add_(0,idx,p)
        xs=torch.zeros(n,dtype=torch.double).scatter_add_(0,idx,p*x)
        c=torch.where(ws>0,xs/ws.clamp_min(1e-30),c)
    v=(p*c[torch.bucketize(x,(c[1:]+c[:-1])/2)]**2).sum()
    return c/v.clamp_min(1e-30).sqrt()

def main(data_dir, dirname, b_avg=4, widen_coord=0, widen_mult=2.5):
    root=Path(data_dir)/dirname
    manifest=json.loads((root/"manifest.json").read_text())
    pooled=torch.load(root/"pooled_stats.pt",map_location="cpu",weights_only=False)
    q2,k2=pooled["q_post"][2],pooled["k_post"][2]
    k_mean,k_cov=pooled["k_post"][0],pooled["k_post"][1]
    L,Hq,d,_=q2.shape; _,Hkv,_,_=k2.shape; gs=Hq//Hkv
    sigma_q=q2.reshape(L,Hkv,gs,d,d).sum(2)
    qpca=build_qpca_basis(sigma_q,k_cov); F=qpca["forward"]
    su=(F.transpose(-1,-2)@_sym(k2.double())@F).diagonal(dim1=-2,dim2=-1).clamp_min(1e-30).sqrt()

    cb={}  # bits -> centroids
    sum_b=0.0; sum_H=0.0; nheads=0
    c0_b=0.0; c0_H=0.0
    for l in range(1,L):
        for h in range(Hkv):
            bits=water_fill_bits(qpca["score"][l,h], b_avg, d)
            std=su[l,h].clone()
            std[widen_coord]*=widen_mult
            # gather centered codes for this head across examples
            rcols=[]
            for e in manifest["examples"]:
                art=torch.load(root/e["file"],map_location="cpu",weights_only=False)
                T=int(art["prompt_length"])
                k=art["k_post"][l,h,:T].double(); mu=k_mean[l,h].double()
                rcols.append((k-mu)@F[l,h])
            r=torch.cat(rcols,0)   # (Ntot,d)
            for j in range(d):
                bj=int(bits[j])
                if bj<=0:
                    continue
                if bj not in cb: cb[bj]=gaussian_centroids(bj)
                c=cb[bj]*std[j]
                idx=(r[:,j:j+1]-c.unsqueeze(0)).abs().argmin(-1)   # bin per token
                counts=torch.bincount(idx,minlength=c.numel()).double()
                pmf=counts/counts.sum().clamp_min(1)
                Hj=float(-(pmf*pmf.clamp_min(1e-30).log2()).sum())
                sum_b+=bj; sum_H+=Hj
                if j==widen_coord: c0_b+=bj; c0_H+=Hj
            nheads+=1

    print(f"layers 1+, {nheads} heads, b_avg={b_avg}, coord {widen_coord} widened x{widen_mult}")
    print(f"  total grid-bits  sum(b_j)/head = {sum_b/nheads:.2f}")
    print(f"  total entropy    sum(H_j)/head = {sum_H/nheads:.2f}")
    print(f"  recoverable rate = {100*(1-sum_H/sum_b):.1f}%  (entropy coding vs fixed-width)")
    print(f"\n  coord {widen_coord} (widened):")
    print(f"    grid-bits/head = {c0_b/nheads:.2f}   entropy/head = {c0_H/nheads:.2f}")
    print(f"    -> the widen costs {c0_b/nheads:.1f} fixed bits but only "
          f"{c0_H/nheads:.1f} bits of entropy")

if __name__=="__main__":
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else "data",
         "query_stats_longbench_under4k_small", b_avg=4, widen_coord=0, widen_mult=2.5)