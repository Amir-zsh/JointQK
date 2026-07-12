#!/usr/bin/env python3
"""REAL fused decode+attention, ALL bandwidth-bound methods (BF16, INT2, VQ, OSCAR),
tensor-core (tl.dot), full-model scale, split-K, three depths, correctness-gated.
The honest kernel-faithful answer to "does K-compression beat BF16 in real fused
decode+attend, and how do the methods differ." Supersedes decode_timing_fullmodel.py
(fp64 un-fused, wrong) and bw_fullmodel.py (read-only, couldn't separate VQ from INT2).

Depths (compressed K read from HBM, reconstructed/scored in-register, no write-back):
  0 READ    : read the compressed K representation, checksum.           (bandwidth floor)
  1 READ+QK : + reconstruct/score -> scores (tensor-core qk; VQ gathers its codeword first).
  2 FULL    : + online softmax + tl.dot(p, V).                          (full decode)

Reconstruction per method (inverse rotation fused into the pre-rotated query, keys stay coded):
  BF16  : read fp16 K, tl.dot(q, K^T).
  INT2  : read 2-bit packed, PER-COORD affine dequant (q_idx*step+base), tl.dot.
  OSCAR : read 2-bit packed, PER-TOKEN affine dequant (dynamic scale/zero), tl.dot.
          (basically INT2's ops with per-token instead of per-coord scale -> expected ~= INT2.)
  VQ    : read NG int16 indices (GROUP-MAJOR, coalesced), reconstruct each key by gathering
          its codeword from cb (H,NG,K,G), score with a tensor-core tl.dot (same structure
          as INT2). The gather is RANDOM, unlike INT2's sequential unpack: at the accuracy
          config (K=4096 -> 1MB codebook/head, > L1) it runs at L2 speed -- ~7x INT2's read+qk,
          a hard architectural floor (a K=256 codebook that fits L1 is ~5x faster). Fusing
          softmax forces the full 1MB (all NG groups) as the per-key working set; the L1-
          resident fast regime needs grid-over-group, which can't fuse without materializing K.

V is left uncompressed (fp16) for all methods -- the V read + softmax are shared,
method-independent costs, so they dilute the K-compression win toward the "full" column;
compressing V too would push the full-decode win higher.
"""
import torch, triton, triton.language as tl

dev = "cuda"
d = 128
BLOCK_M = 16
BLOCK_T = 32          # BF16/INT2/OSCAR tile
VQ_BT = 32            # VQ tile (reconstruct+dot); read floor likes fewer warps, gather likes more
NG, G, K = 21, 6, 4096          # VQ: 21 groups x 6 coords, 4096-entry codebook (2 b/coord)
Dk = NG * G                      # coded key dim = 126 (last 2 of 128 coords dropped)


# ============================ kernels ============================
@triton.jit
def _bf16(q_ptr, k_ptr, v_ptr, m_ptr, l_ptr, acc_ptr, sink_ptr, T, CHUNK, ns, skh,
          DEPTH: tl.constexpr, BM: tl.constexpr, BT: tl.constexpr, D: tl.constexpr):
    h = tl.program_id(0).to(tl.int64); s = tl.program_id(1).to(tl.int64)
    qm = tl.arange(0, BM).to(tl.int64); dd = tl.arange(0, D).to(tl.int64)
    q = tl.load(q_ptr + h * BM * D + qm[:, None] * D + dd[None, :]).to(tl.float16)
    kb = k_ptr + h * skh; vb = v_ptr + h * skh
    lo = s * CHUNK; hi = tl.minimum(lo + CHUNK, T)
    mi = tl.zeros([BM], tl.float32) - float("inf"); li = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32); chk = 0.0
    for t0 in range(lo, hi, BT):
        o = t0 + tl.arange(0, BT).to(tl.int64); mk = o < hi
        k = tl.load(kb + o[:, None] * D + dd[None, :], mask=mk[:, None], other=0.0).to(tl.float16)
        if DEPTH == 0:
            chk += tl.sum(tl.where(mk[:, None], k.to(tl.float32), 0.0))
        else:
            qk = tl.where(mk[None, :], tl.dot(q, tl.trans(k)), -float("inf"))
            if DEPTH == 1:
                chk += tl.sum(tl.where(mk[None, :], qk, 0.0))
            else:
                mn = tl.maximum(mi, tl.max(qk, 1)); p = tl.exp(qk - mn[:, None]); a = tl.exp(mi - mn)
                li = li * a + tl.sum(p, 1)
                v = tl.load(vb + o[:, None] * D + dd[None, :], mask=mk[:, None], other=0.0).to(tl.float16)
                acc = acc * a[:, None] + tl.dot(p.to(tl.float16), v); mi = mn
    if DEPTH == 2:
        b = h * ns + s
        tl.store(m_ptr + b * BM + qm, mi); tl.store(l_ptr + b * BM + qm, li)
        tl.store(acc_ptr + b * BM * D + qm[:, None] * D + dd[None, :], acc)
    else:
        tl.atomic_add(sink_ptr, chk)


@triton.jit
def _int2(q_ptr, pk_ptr, step_ptr, base_ptr, v_ptr, m_ptr, l_ptr, acc_ptr, sink_ptr,
          T, CHUNK, ns, sph, svh, DEPTH: tl.constexpr, BM: tl.constexpr, BT: tl.constexpr, D: tl.constexpr):
    # per-coord affine dequant (qi*step+base) FOLDED INTO THE QUERY (once, amortized): the inner
    # loop just unpacks qi and does one tl.dot(q*step, qi) + a per-query constant -- no per-element
    # dequant. Same structure as OSCAR so the two land at the same runtime.
    h = tl.program_id(0).to(tl.int64); s = tl.program_id(1).to(tl.int64)
    qm = tl.arange(0, BM).to(tl.int64); dd = tl.arange(0, D).to(tl.int64)
    q = tl.load(q_ptr + h * BM * D + qm[:, None] * D + dd[None, :]).to(tl.float32)
    step = tl.load(step_ptr + h * D + dd).to(tl.float32); base = tl.load(base_ptr + h * D + dd).to(tl.float32)
    qs = (q * step[None, :]).to(tl.float16); qbase = tl.sum(q * base[None, :], 1)   # fold into query
    pb = pk_ptr + h * sph; vb = v_ptr + h * svh
    lo = s * CHUNK; hi = tl.minimum(lo + CHUNK, T)
    mi = tl.zeros([BM], tl.float32) - float("inf"); li = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32); chk = 0.0
    for t0 in range(lo, hi, BT):
        o = t0 + tl.arange(0, BT).to(tl.int64); mk = o < hi
        by = tl.load(pb + o[:, None] * (D // 4) + tl.arange(0, D // 4)[None, :].to(tl.int64),
                     mask=mk[:, None], other=0).to(tl.uint32)
        sh = (tl.arange(0, 4) * 2).to(tl.uint32)
        qi = tl.reshape((by[:, :, None] >> sh[None, None, :]) & 0x3, (BT, D)).to(tl.float16)
        if DEPTH == 0:
            chk += tl.sum(tl.where(mk[:, None], by.to(tl.float32), 0.0))
        else:
            qk = tl.where(mk[None, :], tl.dot(qs, tl.trans(qi)) + qbase[:, None], -float("inf"))
            if DEPTH == 1:
                chk += tl.sum(tl.where(mk[None, :], qk, 0.0))
            else:
                mn = tl.maximum(mi, tl.max(qk, 1)); p = tl.exp(qk - mn[:, None]); a = tl.exp(mi - mn)
                li = li * a + tl.sum(p, 1)
                v = tl.load(vb + o[:, None] * D + dd[None, :], mask=mk[:, None], other=0.0).to(tl.float16)
                acc = acc * a[:, None] + tl.dot(p.to(tl.float16), v); mi = mn
    if DEPTH == 2:
        b = h * ns + s
        tl.store(m_ptr + b * BM + qm, mi); tl.store(l_ptr + b * BM + qm, li)
        tl.store(acc_ptr + b * BM * D + qm[:, None] * D + dd[None, :], acc)
    else:
        tl.atomic_add(sink_ptr, chk)


@triton.jit
def _oscar(q_ptr, pk_ptr, sc_ptr, ze_ptr, v_ptr, m_ptr, l_ptr, acc_ptr, sink_ptr,
           T, CHUNK, ns, sph, svh, DEPTH: tl.constexpr, BM: tl.constexpr, BT: tl.constexpr, D: tl.constexpr):
    # OSCAR: PER-TOKEN scale/zero. The dynamic scale factors OUT of the dot -- score[m,t] =
    # sc[t]*(q.qi_raw - ze[t]*sum(q[m])) -- so the inner loop is unpack qi + one tl.dot(q, qi_raw),
    # identical to INT2, then a cheap (BM,BT) post-scale. Keeps OSCAR at INT2's runtime. sc/ze:(H,T).
    h = tl.program_id(0).to(tl.int64); s = tl.program_id(1).to(tl.int64)
    qm = tl.arange(0, BM).to(tl.int64); dd = tl.arange(0, D).to(tl.int64)
    q = tl.load(q_ptr + h * BM * D + qm[:, None] * D + dd[None, :]).to(tl.float16)
    sumq = tl.sum(q.to(tl.float32), 1)                                           # (BM,) for the zero term
    pb = pk_ptr + h * sph; vb = v_ptr + h * svh
    lo = s * CHUNK; hi = tl.minimum(lo + CHUNK, T)
    mi = tl.zeros([BM], tl.float32) - float("inf"); li = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32); chk = 0.0
    for t0 in range(lo, hi, BT):
        o = t0 + tl.arange(0, BT).to(tl.int64); mk = o < hi
        by = tl.load(pb + o[:, None] * (D // 4) + tl.arange(0, D // 4)[None, :].to(tl.int64),
                     mask=mk[:, None], other=0).to(tl.uint32)
        sc = tl.load(sc_ptr + h * T + o, mask=mk, other=1.0).to(tl.float32)      # per-token scale
        ze = tl.load(ze_ptr + h * T + o, mask=mk, other=0.0).to(tl.float32)      # per-token zero
        sh = (tl.arange(0, 4) * 2).to(tl.uint32)
        qi = tl.reshape((by[:, :, None] >> sh[None, None, :]) & 0x3, (BT, D)).to(tl.float16)
        if DEPTH == 0:
            chk += tl.sum(tl.where(mk[:, None], by.to(tl.float32), 0.0)) + tl.sum(tl.where(mk, sc, 0.0))
        else:
            qkr = tl.dot(q, tl.trans(qi))                                        # (BM,BT) raw q.qi
            qk = tl.where(mk[None, :], sc[None, :] * (qkr - ze[None, :] * sumq[:, None]), -float("inf"))
            if DEPTH == 1:
                chk += tl.sum(tl.where(mk[None, :], qk, 0.0))
            else:
                mn = tl.maximum(mi, tl.max(qk, 1)); p = tl.exp(qk - mn[:, None]); a = tl.exp(mi - mn)
                li = li * a + tl.sum(p, 1)
                v = tl.load(vb + o[:, None] * D + dd[None, :], mask=mk[:, None], other=0.0).to(tl.float16)
                acc = acc * a[:, None] + tl.dot(p.to(tl.float16), v); mi = mn
    if DEPTH == 2:
        b = h * ns + s
        tl.store(m_ptr + b * BM + qm, mi); tl.store(l_ptr + b * BM + qm, li)
        tl.store(acc_ptr + b * BM * D + qm[:, None] * D + dd[None, :], acc)
    else:
        tl.atomic_add(sink_ptr, chk)


@triton.jit
def _vq(q_ptr, cb_ptr, idx_ptr, v_ptr, m_ptr, l_ptr, acc_ptr, sink_ptr, T, CHUNK, ns, svh,
        DEPTH: tl.constexpr, NGc: tl.constexpr, NGpc: tl.constexpr, Kc: tl.constexpr,
        Gc: tl.constexpr, Dkc: tl.constexpr, VEC: tl.constexpr,
        BM: tl.constexpr, BT: tl.constexpr, D: tl.constexpr, VEC8: tl.constexpr = 0):
    # RECONSTRUCT-and-dot (realistic codebook path, not a per-query LUT): gather each key's
    #   codeword from cb and score with a tensor-core tl.dot -- same structure as INT2/OSCAR,
    #   just a random gather instead of a sequential unpack. idx: (n_heads, T, NG) int16,
    #   TOKEN-MAJOR so a token's NG indices are contiguous -> read+reconstruct are ONE coalesced
    #   (BT,*) tile (group-major (NG,T) aliased in L2 when NG,T both pow2 -- the G=4 read cliff).
    #   VEC (G=4 only): view cb as int64 (one 8-byte codeword = 1 wide load, 4x fewer LSU ops
    #   than 4 fp16 gathers -> gather 12.2->3.2ms), unpack 4 fp16 planes and reassemble in coord
    #   order via join/reshape for a single contraction-D dot. Non-VEC: per-coord fp16 gather.
    h = tl.program_id(0).to(tl.int64); s = tl.program_id(1).to(tl.int64)
    qm = tl.arange(0, BM).to(tl.int64); dd = tl.arange(0, D).to(tl.int64)
    gcol = dd // Gc; wcol = dd - gcol * Gc; colm = dd < Dkc; gng = tl.arange(0, NGpc).to(tl.int64)
    q = tl.load(q_ptr + h * BM * D + qm[:, None] * D + dd[None, :]).to(tl.float16)
    ib = idx_ptr + h * (T * NGc); vb = v_ptr + h * svh
    lo = s * CHUNK; hi = tl.minimum(lo + CHUNK, T)
    mi = tl.zeros([BM], tl.float32) - float("inf"); li = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32); chk = 0.0
    for t0 in range(lo, hi, BT):
        o = t0 + tl.arange(0, BT).to(tl.int64); mk = o < hi
        if DEPTH == 0:
            # read the index stream (compressed rep): one coalesced (BT,NG) tile, NG int16/key.
            ii = tl.load(ib + o[:, None] * NGc + gng[None, :],
                         mask=mk[:, None] & (gng < NGc)[None, :], other=0).to(tl.float32)
            chk += tl.sum(ii)
        else:
            if VEC8:
                # fp8 codebook: one int32 codeword = 4 fp8 bytes (HALF the int64 gather traffic).
                mg = mk[:, None] & (gng < NGc)[None, :]
                isel = tl.load(ib + o[:, None] * NGc + gng[None, :], mask=mg, other=0).to(tl.int32)
                cw = tl.load(cb_ptr + h * (NGc * Kc) + gng[None, :] * Kc + isel, mask=mg, other=0).to(tl.int32)
                p0 = ((cw) & 0xFF).to(tl.uint8).to(tl.float8e5, bitcast=True).to(tl.float16)
                p1 = ((cw >> 8) & 0xFF).to(tl.uint8).to(tl.float8e5, bitcast=True).to(tl.float16)
                p2 = ((cw >> 16) & 0xFF).to(tl.uint8).to(tl.float8e5, bitcast=True).to(tl.float16)
                p3 = ((cw >> 24) & 0xFF).to(tl.uint8).to(tl.float8e5, bitcast=True).to(tl.float16)
                kg = tl.reshape(tl.join(tl.join(p0, p2), tl.join(p1, p3)), (BT, NGpc * Gc))
            elif VEC:
                mg = mk[:, None] & (gng < NGc)[None, :]
                isel = tl.load(ib + o[:, None] * NGc + gng[None, :], mask=mg, other=0).to(tl.int64)
                cw = tl.load(cb_ptr + h * (NGc * Kc) + gng[None, :] * Kc + isel, mask=mg, other=0).to(tl.int64)
                p0 = (cw & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True)
                p1 = ((cw >> 16) & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True)
                p2 = ((cw >> 32) & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True)
                p3 = ((cw >> 48) & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True)
                kg = tl.reshape(tl.join(tl.join(p0, p2), tl.join(p1, p3)), (BT, NGpc * Gc))       # (BT,D) coord order
            else:
                m2 = mk[:, None] & colm[None, :]
                isel = tl.load(ib + o[:, None] * NGc + gcol[None, :], mask=m2, other=0).to(tl.int64)
                kg = tl.load(cb_ptr + h * (NGc * Kc * Gc) + gcol[None, :] * (Kc * Gc) + isel * Gc + wcol[None, :],
                             mask=m2, other=0.0).to(tl.float16)                                   # (BT,D)
            qk = tl.where(mk[None, :], tl.dot(q, tl.trans(kg)), -float("inf"))                    # (BM,BT)
            if DEPTH == 1:
                chk += tl.sum(tl.where(mk[None, :], qk, 0.0))
            else:
                mn = tl.maximum(mi, tl.max(qk, 1)); p = tl.exp(qk - mn[:, None]); a = tl.exp(mi - mn)
                li = li * a + tl.sum(p, 1)
                v = tl.load(vb + o[:, None] * D + dd[None, :], mask=mk[:, None], other=0.0).to(tl.float16)
                acc = acc * a[:, None] + tl.dot(p.to(tl.float16), v); mi = mn
    if DEPTH == 2:
        b = h * ns + s
        tl.store(m_ptr + b * BM + qm, mi); tl.store(l_ptr + b * BM + qm, li)
        tl.store(acc_ptr + b * BM * D + qm[:, None] * D + dd[None, :], acc)
    else:
        tl.atomic_add(sink_ptr, chk)


# ============================ host ============================
def combine(m, l, acc, H, ns):
    m = m.view(H, ns, BLOCK_M); l = l.view(H, ns, BLOCK_M); acc = acc.view(H, ns, BLOCK_M, d)
    mg = m.max(1, keepdim=True).values; sc = torch.exp(m - mg)
    return (acc * sc.unsqueeze(-1)).sum(1) / (l * sc).sum(1).unsqueeze(-1)


def timeit(f, reps=100, warmup=30):    # high reps: full-decode ms wobbles with machine contention
    for _ in range(warmup): f()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps): f()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


def alloc(H, ns):
    return (torch.zeros(H * ns * BLOCK_M, device=dev), torch.zeros(H * ns * BLOCK_M, device=dev),
            torch.zeros(H * ns * BLOCK_M * d, device=dev), torch.zeros(1, device=dev))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-heads", type=int, default=288)
    ap.add_argument("--Ts", type=int, nargs="+", default=[16384, 65536])
    ap.add_argument("--n-splits", type=int, default=32)
    ap.add_argument("--G", type=int, default=6, help="VQ group size; codebook/head = 256*2^(2G) bytes -> L1 cliff at G>=5")
    args = ap.parse_args()
    torch.manual_seed(0); H = args.n_heads; ns = args.n_splits
    G = args.G; NG = d // G; K = 1 << (2 * G); Dk = NG * G      # rate 2 bits/coord -> K=2^(2G)
    NGp = 1 << (NG - 1).bit_length()                            # next pow2 >= NG (for the depth-0 tile)
    VEC = (G == 4)                                              # G=4 codeword = 8B = int64-vectorizable (NG=32 pow2)
    VQ_BT = 128 if VEC else 32                                  # int64 gather amortizes far better at BT=128
    print(f"VQ config: G={G}, NG={NG}, K={K}, codebook/head={NG*K*G*2/1024:.0f}KB "
          f"({'L1-resident' if NG*K*G*2 <= 192*1024 else 'exceeds 192KB L1'})", flush=True)

    # ---------------- correctness (small) ----------------
    Hc, Tc = 8, 4096; CH = (Tc + ns - 1) // ns
    q = torch.randn(Hc, BLOCK_M, d, device=dev)
    kc = torch.randn(Hc, Tc, d, device=dev); vf = torch.randn(Hc, Tc, d, device=dev).half().contiguous()
    m, l, acc, sink = alloc(Hc, ns)
    ref_s = lambda kk: torch.einsum("hmt,htd->hmd", torch.softmax(torch.einsum("hmd,htd->hmt", q, kk), -1), vf.float())

    # INT2 per-coord
    std = kc.std(1); step = std.contiguous(); base = (-1.5 * std).contiguous()
    qidx = ((kc / std.unsqueeze(1) + 1.5).round().clamp(0, 3)).to(torch.uint8)
    qi = qidx.view(Hc, Tc, d // 4, 4)
    packed = (qi[..., 0] | (qi[..., 1] << 2) | (qi[..., 2] << 4) | (qi[..., 3] << 6)).to(torch.uint8).contiguous()
    k_i2 = qidx.float() * step.unsqueeze(1) + base.unsqueeze(1)
    _int2[(Hc, ns)](q, packed, step, base, vf, m, l, acc, sink, Tc, CH, ns, Tc*(d//4), Tc*d, 2, BLOCK_M, BLOCK_T, d)
    print(f"[correctness] INT2 err={(combine(m,l,acc,Hc,ns)-ref_s(k_i2)).abs().max().item():.2e}", flush=True)

    # OSCAR per-token
    tmin = kc.min(-1).values; tmax = kc.max(-1).values
    scale = ((tmax - tmin).clamp_min(1e-6) / 3.0).contiguous(); zero = (-tmin / scale).contiguous()
    oqi = ((kc / scale.unsqueeze(-1) + zero.unsqueeze(-1)).round().clamp(0, 3)).to(torch.uint8)
    oq = oqi.view(Hc, Tc, d // 4, 4)
    opacked = (oq[..., 0] | (oq[..., 1] << 2) | (oq[..., 2] << 4) | (oq[..., 3] << 6)).to(torch.uint8).contiguous()
    k_os = (oqi.float() - zero.unsqueeze(-1)) * scale.unsqueeze(-1)
    _oscar[(Hc, ns)](q, opacked, scale, zero, vf, m, l, acc, sink, Tc, CH, ns, Tc*(d//4), Tc*d, 2, BLOCK_M, BLOCK_T, d)
    print(f"[correctness] OSCAR err={(combine(m,l,acc,Hc,ns)-ref_s(k_os)).abs().max().item():.2e}", flush=True)

    # VQ (reconstruct+dot); coded key dim = NG*G = 126 (last 2 coords dropped)
    qk_vq = q[:, :, :Dk].contiguous()
    cb = torch.randn(Hc, NG, K, G, device=dev, dtype=torch.float16).contiguous() # (H,NG,K,G) fp16 (int64-viewable)
    vqidx = torch.randint(0, K, (Hc, Tc, NG), device=dev, dtype=torch.int16)
    krec = torch.gather(cb.unsqueeze(1).expand(Hc, Tc, NG, K, G), 3,
                        vqidx.long().unsqueeze(-1).unsqueeze(-1).expand(Hc, Tc, NG, 1, G)).squeeze(3).reshape(Hc, Tc, Dk)
    idx_tm = vqidx.contiguous()                                                 # (H,T,NG) token-major
    cb_k = cb.view(torch.int64).squeeze(-1).contiguous() if VEC else cb         # (H,NG,K) int64 codewords when VEC
    _vq[(Hc, ns)](q, cb_k, idx_tm, vf, m, l, acc, sink, Tc, CH, ns, Tc*d, 2, NG, NGp, K, G, Dk, VEC, BLOCK_M, VQ_BT, d)
    s_vq = torch.softmax(torch.einsum("hmd,htd->hmt", qk_vq, krec.float()), -1)
    o_vq_ref = torch.einsum("hmt,htd->hmd", s_vq, vf.float())
    print(f"[correctness] VQ   err={(combine(m,l,acc,Hc,ns)-o_vq_ref).abs().max().item():.2e}", flush=True)
    if VEC:   # VQ8: fp8 codebook -- reconstruct with fp8-dequantized centroids, validate the int32 unpack
        cb8 = cb.to(torch.float8_e5m2)
        cb_k8c = cb8.view(torch.int32).squeeze(-1).contiguous()
        cb_deq = cb8.float().half()
        krec8 = torch.gather(cb_deq.unsqueeze(1).expand(Hc, Tc, NG, K, G), 3,
                             vqidx.long().unsqueeze(-1).unsqueeze(-1).expand(Hc, Tc, NG, 1, G)).squeeze(3).reshape(Hc, Tc, Dk)
        _vq[(Hc, ns)](q, cb_k8c, idx_tm, vf, m, l, acc, sink, Tc, CH, ns, Tc*d, 2, NG, NGp, K, G, Dk, False, BLOCK_M, VQ_BT, d, VEC8=1)
        ref8 = torch.einsum("hmt,htd->hmd", torch.softmax(torch.einsum("hmd,htd->hmt", qk_vq, krec8.float()), -1), vf.float())
        print(f"[correctness] VQ8  err={(combine(m,l,acc,Hc,ns)-ref8).abs().max().item():.2e} (fp8 codebook)", flush=True)
    del kc, vf, packed, opacked, cb, krec, k_i2, k_os; torch.cuda.empty_cache()

    # ---------------- timing (full model) ----------------
    for T in args.Ts:
        CHUNK = (T + ns - 1) // ns
        q = torch.randn(H, BLOCK_M, d, device=dev)
        kf = torch.randn(H, T, d, dtype=torch.float16, device=dev)   # BF16 K (SEPARATE from V --
        vf = torch.randn(H, T, d, dtype=torch.float16, device=dev)   #  else BF16's V read is a free cache hit)
        m, l, acc, sink = alloc(H, ns)
        # build compressed reps (values don't matter for timing; use random of correct shape/size)
        packed = torch.randint(0, 256, (H, T, d // 4), device=dev, dtype=torch.uint8)
        step = torch.rand(H, d, device=dev) + 0.5; base = -1.5 * step
        scale = (torch.rand(H, T, device=dev) + 0.5).half(); zero = torch.rand(H, T, device=dev).half()  # fp16 per-token scale/zero (real OSCAR)
        idx_tm = torch.randint(0, K, (H, T, NG), device=dev, dtype=torch.int16)   # (H,T,NG) token-major
        cb = torch.randn(H, NG, K, G, device=dev, dtype=torch.float16)            # (H,NG,K,G) codebook
        cb_k = cb.view(torch.int64).squeeze(-1).contiguous() if VEC else cb       # int64 codewords when VEC (G=4)
        # fp8 codebook variant: 4 fp8/codeword -> int32 (HALF the gather bytes of int64)
        cb_k8 = cb.to(torch.float8_e5m2).view(torch.int32).squeeze(-1).contiguous() if VEC else None

        print(f"\n{'method':>7} {'read':>9} {'read+qk':>9} {'full':>9}  vs BF16(full)   ({H} heads x{BLOCK_M}q, T={T})", flush=True)
        rows = {}
        methods = ("BF16", "INT2", "OSCAR", "VQ") + (("VQ8",) if VEC else ())
        for name in methods:
            tt = []
            for dep in (0, 1, 2):
                if name == "BF16":
                    f = lambda: _bf16[(H, ns)](q, kf, vf, m, l, acc, sink, T, CHUNK, ns, T*d, dep, BLOCK_M, BLOCK_T, d)
                elif name == "INT2":
                    f = lambda: _int2[(H, ns)](q, packed, step, base, vf, m, l, acc, sink, T, CHUNK, ns, T*(d//4), T*d, dep, BLOCK_M, BLOCK_T, d)
                elif name == "OSCAR":
                    f = lambda: _oscar[(H, ns)](q, packed, scale, zero, vf, m, l, acc, sink, T, CHUNK, ns, T*(d//4), T*d, dep, BLOCK_M, BLOCK_T, d)
                elif name == "VQ8":
                    f = lambda: _vq[(H, ns)](q, cb_k8, idx_tm, vf, m, l, acc, sink, T, CHUNK, ns, T*d, dep, NG, NGp, K, G, Dk, False, BLOCK_M, VQ_BT, d, VEC8=1, num_warps=4)
                else:
                    nw = 4 if (VEC or dep == 0) else 8   # VEC/read like fewer warps; non-VEC random gather likes more
                    f = lambda nw=nw: _vq[(H, ns)](q, cb_k, idx_tm, vf, m, l, acc, sink, T, CHUNK, ns, T*d, dep, NG, NGp, K, G, Dk, VEC, BLOCK_M, VQ_BT, d, num_warps=nw)
                tt.append(timeit(f))
            rows[name] = tt
        bf_full = rows["BF16"][2]
        for name in methods:
            tt = rows[name]
            print(f"{name:>7} {tt[0]:>9.4f} {tt[1]:>9.4f} {tt[2]:>9.4f}   {bf_full/tt[2]:>6.2f}x", flush=True)

        # effective HBM bandwidth: known traffic (K-rep read + V read) / time, vs A100 peak.
        # Shows which cells are bandwidth-bound (near peak) vs compute/gather-bound (far below).
        # NOTE: VQ's codebook GATHER is extra L2/HBM traffic NOT in kbytes -> its %peak is a
        # lower bound; a VQ far below peak means it's gather-bound, not that it moves few bytes.
        PEAK = 1555.0  # A100-SXM4-40GB
        kbytes = {"BF16": H*T*d*2, "INT2": H*T*(d//4), "OSCAR": H*T*(d//4) + H*T*2*2, "VQ": H*NG*T*2}
        vbytes = H*T*d*2
        print(f"        {'read GB/s(%pk)':>16} {'full GB/s(%pk)':>16}   (known K-rep + V bytes / time)", flush=True)
        for name in ("BF16", "INT2", "OSCAR", "VQ"):
            r_gbs = kbytes[name]/1e9 / (rows[name][0]/1e3)
            f_gbs = (kbytes[name] + vbytes)/1e9 / (rows[name][2]/1e3)
            print(f"{name:>7} {r_gbs:>9.0f}({r_gbs/PEAK*100:>3.0f}%) {f_gbs:>10.0f}({f_gbs/PEAK*100:>3.0f}%)", flush=True)
        del q, kf, vf, packed, step, base, scale, zero, idx_tm, cb, cb_k, m, l, acc; torch.cuda.empty_cache()
