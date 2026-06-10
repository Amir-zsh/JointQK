import numpy as np, torch, struct
import rans_interleaved as ri

def encode_gpu(self, k, ext_enc):
    """Full GPU encode. ext_enc = compiled rans_encode extension.
    Returns bytes byte-compatible with the RANS decoder."""
    dev = self.device; P = self.P; R = self.R; N = self.N; d = self.d; dz = self.dz
    kd = torch.as_tensor(np.asarray(k), dtype=torch.float64, device=dev)
    T = kd.shape[0]; nb = (T + P - 1) // P; pad = nb * P - T
    mu = torch.as_tensor(self.mu, dtype=torch.float64, device=dev)
    fwd = torch.as_tensor(self.fwd, dtype=torch.float64, device=dev)
    r = (kd - mu) @ fwd
    budget = self.payload_budget_bits
    overhead_bits = 8 * (8 + 4*N + 4*N + N)

    # cache per-rung LUT tensors
    Ld = {}
    def Lt(ri_):
        if ri_ not in Ld:
            Ld[ri_] = dict(
                delta=torch.as_tensor(self.deltas[ri_], dtype=torch.float64, device=dev),
                lo=torch.as_tensor(self.lut_lo[ri_], dtype=torch.int64, device=dev),
                hi=torch.as_tensor(self.lut_hi[ri_], dtype=torch.int64, device=dev),
                off=torch.as_tensor(self.lut_off[ri_], dtype=torch.int64, device=dev),
                lut_pos=torch.as_tensor(self.lut_pos[ri_], dtype=torch.int64, device=dev),
                lut_nlp=torch.as_tensor(self.lut_nlp[ri_], dtype=torch.float64, device=dev),
                bc=torch.as_tensor(self._bitcost[ri_], dtype=torch.float64, device=dev),
                nc=self._cuda_cdf_tensors[ri_][0].to(torch.int64),
                cdf=self._cuda_cdf_tensors[ri_][1].to(torch.int64),
                coff=self._cuda_cdf_tensors[ri_][2].to(torch.int64),
            )
        return Ld[ri_]

    def pos_at(ri_):
        t = Lt(ri_)
        q = (torch.sign(r) * torch.floor(torch.abs(r)/t['delta'] + dz)).to(torch.int64)
        fi = (torch.clamp(q, t['lo'][None,:], t['hi'][None,:]) - t['lo'][None,:]) + t['off'][None,:]
        return t['lut_pos'][fi]                      # (T,d)

    # --- selection (verified) ---
    tref = Lt(self.ref)
    qr = (torch.sign(r)*torch.floor(torch.abs(r)/tref['delta']+dz)).to(torch.int64)
    fir = (torch.clamp(qr,tref['lo'][None,:],tref['hi'][None,:])-tref['lo'][None,:])+tref['off'][None,:]
    b_ref = tref['lut_nlp'][fir].sum(1)
    shift = torch.as_tensor(self.shift, dtype=torch.float64, device=dev)
    est = b_ref[None,:] + shift[:,None]
    if pad: est = torch.cat([est, torch.zeros((R,pad),dtype=torch.float64,device=dev)],1)
    page_est = est.reshape(R,nb,P).sum(2)
    fits = (page_est + overhead_bits) <= budget
    chosen0 = torch.where(fits.any(0), fits.float().argmax(0), torch.full((nb,),R-1,device=dev)).to(torch.int64)
    # proxy climb
    def proxy_pb(ri_):
        t = Lt(ri_); pos = pos_at(ri_); jidx = torch.arange(d,device=dev)
        tb = t['bc'][jidx[None,:], pos].sum(1)
        if pad: tb = torch.cat([tb, torch.zeros(pad,dtype=torch.float64,device=dev)])
        return tb.reshape(nb,P).sum(1)
    rung = chosen0.clone(); ppb = {}
    for _ in range(R):
        need = torch.zeros(nb,dtype=torch.bool,device=dev)
        for rr in torch.unique(rung).tolist():
            if rr not in ppb: ppb[rr] = proxy_pb(rr)
            need |= ((rung==rr) & ((ppb[rr]+overhead_bits)>budget) & (rung<R-1))
        if not bool(need.any()): break
        rung = torch.where(need, torch.clamp(rung+1,max=R-1), rung)

    # --- encode loop with real-byte backstop ---
    page_blobs = [None]*nb
    pending = list(range(nb))
    for _it in range(R+1):
        # group pending pages by rung, gather+encode each group
        rung_np = rung.cpu().numpy()
        by_rung = {}
        for bi in pending:
            by_rung.setdefault(int(rung_np[bi]), []).append(bi)
        redo = []
        for ri_, pages in by_rung.items():
            t = Lt(ri_); pos = pos_at(ri_)
            nc = t['nc']; C = int(nc.numel()); cdf = t['cdf']; coff = t['coff']
            # build freq/start for all pages in this group, concatenated, with per-page sym offset
            freq_parts=[]; start_parts=[]; sym_off=[]; metas=[]
            cur=0
            for bi in pages:
                s0=bi*P; n=min((bi+1)*P,T)-s0
                jj = nc.repeat(n); tt = (torch.arange(n,device=dev)+s0).repeat_interleave(C)
                psym = pos[tt,jj]; basej = coff[nc.repeat(n)]
                st = cdf[basej+psym]; fr = cdf[basej+psym+1]-st
                freq_parts.append(fr); start_parts.append(st)
                sym_off.append(cur); metas.append((bi,n,C)); cur += n*C
            freq_cat=torch.cat(freq_parts).to(torch.int64)
            start_cat=torch.cat(start_parts).to(torch.int64)
            npg=len(pages)
            # lane bounds per page
            k0a=np.zeros((npg,N),np.int32); k1a=np.zeros((npg,N),np.int32)
            psoff=np.zeros(npg,np.int64)
            for pi,(bi,n,C_) in enumerate(metas):
                S=n*C_; bnd=[round(x*S/N) for x in range(N+1)]
                k0a[pi]=bnd[:-1]; k1a[pi]=bnd[1:]; psoff[pi]=sym_off[pi]
            max_lane=int(2*(max(n for _,n,_ in metas)*C)//N + 64)
            out_bytes=torch.zeros(npg*N*max_lane,dtype=torch.uint8,device=dev)
            out_len=torch.zeros(npg*N,dtype=torch.int32,device=dev)
            def _T(a,dt): return torch.as_tensor(a,dtype=dt,device=dev)
            ext_enc.encode_pages(freq_cat,start_cat,
                _T(k0a.reshape(-1),torch.int32),_T(k1a.reshape(-1),torch.int32),
                _T(psoff,torch.int64),N,npg,max_lane,out_bytes,out_len)
            out_len=out_len.cpu().numpy().reshape(npg,N)
            ob=out_bytes.cpu().numpy()
            for pi,(bi,n,C_) in enumerate(metas):
                S=n*C_
                lane_byte=[ob[(pi*N+l)*max_lane:(pi*N+l)*max_lane+int(out_len[pi,l])] for l in range(N)]
                total_payload_bits = 8*sum(len(x) for x in lane_byte)
                if total_payload_bits > budget and int(rung_np[bi]) < R-1:
                    redo.append(bi); continue
                # assemble page blob: <N><S><lane_len[N]><lanes>
                hdr=struct.pack("<II",N,S)+b"".join(struct.pack("<I",len(x)) for x in lane_byte)
                page_blobs[bi]=hdr+b"".join(bytes(x) for x in lane_byte)
        if not redo: break
        for bi in redo: rung[bi]=min(int(rung[bi].item())+1,R-1)
        pending=redo

    # --- final stream assembly ---
    from kvq_codec import _pack
    rung_final=rung.cpu().numpy().astype(np.int64)
    out=bytearray()
    out+=struct.pack("<IIII", self.MAGIC, T, P, nb)
    out+=_pack(rung_final, self.id_bits)
    for bi in range(nb):
        blob=page_blobs[bi]
        out+=struct.pack("<I",len(blob))+blob
    return bytes(out)