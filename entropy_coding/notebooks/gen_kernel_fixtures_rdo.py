#!/usr/bin/env python3
"""Golden fixtures for the RDO (per-token rung) decode kernel.
Stacks all R rungs' CDF tables with a (rung,coord) offset table, emits real
RDO page blobs + expected positions, and a numpy port-target verified against
PageCodecRDO._decode_page_rdo."""
import struct
import numpy as np
import torch
import run_pca_ec_deadzone as base
from test_codec_on_data import build_ladder
from kvq_codec_rdo import build_codecs_from_ladder_rdo
import rans_interleaved as ri

SB = ri.SCALE_BITS; MASK = ri.MASK; RANS_L = ri.RANS_L

def build_stacked(codec):
    """Flatten codec's R rungs into: cdf_flat, off2d[(R)*(d+1)], vals_flat, voff2d,
    is_const[R,d], const_val[R,d]. Index coord j at rung r via off2d[r*(d+1)+j]."""
    R, d = codec.R, codec.d
    cdf_parts, vals_parts = [], []
    off2d = np.zeros(R * (d + 1), np.int64)
    voff2d = np.zeros(R * (d + 1), np.int64)
    is_const = np.zeros((R, d), np.uint8); const_val = np.zeros((R, d), np.int64)
    ccur = vcur = 0
    for r in range(R):
        co = codec._coff[r]            # (d+1,) flat coord offsets in codec's per-rung arrays
        for j in range(d):
            off2d[r * (d + 1) + j] = ccur
            voff2d[r * (d + 1) + j] = vcur
            a = int(co[j + 1] - co[j])
            t = codec.rtab[r][j]
            if t.constant:
                is_const[r, j] = 1; const_val[r, j] = t.const_val
                cdf_parts.append(np.array([0, ri.TOTAL], np.int64))  # 1-sym: start 0, sentinel
                vals_parts.append(np.array([t.const_val], np.int64))
                ccur += 2; vcur += 1
            else:
                cdf_j = np.concatenate([codec._cdf_f[r][co[j]:co[j]+a], [ri.TOTAL]]).astype(np.int64)
                cdf_parts.append(cdf_j)
                vals_parts.append(codec._vals_f[r][co[j]:co[j]+a].astype(np.int64))
                ccur += a + 1; vcur += a
        off2d[r * (d + 1) + d] = ccur
        voff2d[r * (d + 1) + d] = vcur
    return (np.concatenate(cdf_parts).astype(np.uint32), off2d,
            np.concatenate(vals_parts), voff2d, is_const, const_val)

def decode_page_flat_rdo(blob, n, d, N, nc0, cdf, off2d, R):
    """numpy port-target the kernel mirrors. Returns positions (n,d)."""
    Nh, S, nn = struct.unpack_from("<III", blob, 0)
    o = 12
    rung_tok = np.frombuffer(blob, "<u2", nn, o).astype(np.int64); o += 2 * nn
    lane_len = [struct.unpack_from("<I", blob, o + 4 * l)[0] for l in range(N)]; o += 4 * N
    order = [(t, j) for t in range(n) for j in nc0]
    bnd = [round(k * S / N) for k in range(N + 1)]
    starts = np.cumsum([0] + lane_len)
    pos = np.zeros((n, d), np.int64)
    for l in range(N):
        seg = order[bnd[l]:bnd[l + 1]]
        if not seg: continue
        buf = blob[o + int(starts[l]): o + int(starts[l + 1])]
        x = struct.unpack_from("<I", buf, 0)[0]; bp = 4
        for (t, j) in seg:
            r = int(rung_tok[t]); base = int(off2d[r * (d + 1) + j])
            hi = int(off2d[r * (d + 1) + j + 1]) - 1   # sentinel index
            slot = x & MASK
            a, b = base, hi - 1
            while a < b:
                m = (a + b + 1) >> 1
                if cdf[m] <= slot: a = m
                else: b = m - 1
            s = a - base
            start = int(cdf[base + s]); freq = int(cdf[base + s + 1]) - start
            x = freq * (x >> SB) + slot - start
            while x < RANS_L: x = ((x << 8) | buf[bp]) & 0xFFFFFFFF; bp += 1
            pos[t, j] = s
    return pos, rung_tok

if __name__ == "__main__":
    CALIB=[0,1,2]; B=4; PTOK=64; DZ=0.375; N=8
    M=sorted({1.0,1.025,1.05,1.1,1.15,1.25,1.5,1.75,2.0})
    root=base.data_root(); man=base.load_manifest(root)
    sq,sk,km,kc,meta=base.calib_moments(root,man,CALIB)
    L,Hkv,d=meta['n_layers'],meta['n_kv_heads'],meta['d_head']
    qu=base.build_qpca_basis(sq,sk); qu['sigma_k'],qu['sigma_q']=sk,sq
    qc=base.build_qpca_basis(sq,kc); qc['sigma_k']=sk
    fc=base._codes_for_idx(root,man,CALIB,qc['forward'],km,L,Hkv,d)
    lad,_,_=build_ladder(qc,qu,km,B,L,Hkv,fc,root,CALIB,M,DZ)
    cod=build_codecs_from_ladder_rdo(qc['forward'],qc['inverse'],km,lad,L,Hkv,B*d*PTOK,PTOK,DZ,lanes=N)
    c=cod[(1,0)]
    art=torch.load(root/man['examples'][4]['file'],map_location='cpu',weights_only=False)
    k=art['k_post']; T=int(art['prompt_length'])

    cdf, off2d, vals, voff2d, is_const, const_val = build_stacked(c)
    nc0 = np.array(c._nc0, np.int32)

    # take the first few real pages
    buf = c.encode(k[1,0,:T,:].float())
    magic, Tt, P, nb = struct.unpack_from("<IIII", buf, 0); o = 16
    pages = []
    for bi in range(min(nb, 6)):
        (blen,) = struct.unpack_from("<I", buf, o); o += 4
        blob = buf[o:o+blen]; o += blen
        n = min((bi+1)*P, Tt) - bi*P
        ref_idx, _ = c._decode_page_rdo(blob, n)            # CPU reference -> values
        flat_pos, rung_tok = decode_page_flat_rdo(blob, n, d, N, c._nc0, cdf, off2d, c.R)
        # map flat positions -> values via stacked vals, compare to ref
        got = np.zeros((n, d), np.int64)
        for t in range(n):
            r = int(rung_tok[t])
            for j in range(d):
                vb = int(voff2d[r*(d+1)+j])
                got[t, j] = vals[vb + flat_pos[t, j]] if not is_const[r, j] else const_val[r, j]
        assert np.array_equal(got, ref_idx), f"page {bi} mismatch"
        pages.append((n, blob, rung_tok.astype(np.int32), flat_pos.astype(np.int32)))
    print("flat RDO decode == CPU reference on", len(pages), "pages: True")

    np.savez("kernel_fixtures_rdo.npz",
        d=d, N=N, R=c.R, SB=SB, nc0=nc0,
        cdf=cdf, off2d=off2d.astype(np.int64),
        n_pages=len(pages),
        ns=np.array([p[0] for p in pages], np.int32),
        blob_lens=np.array([len(p[1]) for p in pages], np.int32),
        blobs=np.frombuffer(b"".join(p[1] for p in pages), np.uint8),
        expected_pos=np.concatenate([p[3].reshape(-1) for p in pages]))
    print("wrote kernel_fixtures_rdo.npz")