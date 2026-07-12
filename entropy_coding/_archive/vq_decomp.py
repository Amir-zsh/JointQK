#!/usr/bin/env python3
"""Decompose VQ read+qk cost into idx-read / codebook-gather / tensor-core-dot, to find
what actually dominates (and whether the "gather-throughput floor" claim holds or the dot /
redundant idx read is the real cost). Token-major idx (H,T,NG); cb (H,NG,K,G). T=65536."""
import argparse, torch, triton, triton.language as tl

dev = "cuda"; d = 128; BM = 16


@triton.jit
def _idx_only(idx_ptr, sink, T, CHUNK, ns, NGc: tl.constexpr, NGpc: tl.constexpr, BT: tl.constexpr):
    h = tl.program_id(0).to(tl.int64); s = tl.program_id(1).to(tl.int64)
    ib = idx_ptr + h * (T * NGc); lo = s * CHUNK; hi = tl.minimum(lo + CHUNK, T); chk = 0.0
    gg = tl.arange(0, NGpc).to(tl.int64); mg = gg < NGc
    for t0 in range(lo, hi, BT):
        o = t0 + tl.arange(0, BT).to(tl.int64); mk = o < hi
        ii = tl.load(ib + o[:, None] * NGc + gg[None, :], mask=mk[:, None] & mg[None, :], other=0).to(tl.float32)
        chk += tl.sum(ii)
    tl.atomic_add(sink, chk)


@triton.jit
def _gather_nodot(cb_ptr, idx_ptr, sink, T, CHUNK, ns, NGc: tl.constexpr, Kc: tl.constexpr,
                  Gc: tl.constexpr, Dkc: tl.constexpr, BT: tl.constexpr, D: tl.constexpr):
    h = tl.program_id(0).to(tl.int64); s = tl.program_id(1).to(tl.int64)
    dd = tl.arange(0, D).to(tl.int64); gcol = dd // Gc; wcol = dd - gcol * Gc; colm = dd < Dkc
    cbh = cb_ptr + h * (NGc * Kc * Gc); ib = idx_ptr + h * (T * NGc)
    lo = s * CHUNK; hi = tl.minimum(lo + CHUNK, T); chk = 0.0
    for t0 in range(lo, hi, BT):
        o = t0 + tl.arange(0, BT).to(tl.int64); mk = o < hi; m2 = mk[:, None] & colm[None, :]
        isel = tl.load(ib + o[:, None] * NGc + gcol[None, :], mask=m2, other=0).to(tl.int64)
        kg = tl.load(cbh + gcol[None, :] * (Kc * Gc) + isel * Gc + wcol[None, :], mask=m2, other=0.0).to(tl.float32)
        chk += tl.sum(kg)
    tl.atomic_add(sink, chk)


@triton.jit
def _gather_dot(q_ptr, cb_ptr, idx_ptr, sink, T, CHUNK, ns, NGc: tl.constexpr, Kc: tl.constexpr,
                Gc: tl.constexpr, Dkc: tl.constexpr, BM: tl.constexpr, BT: tl.constexpr, D: tl.constexpr):
    h = tl.program_id(0).to(tl.int64); s = tl.program_id(1).to(tl.int64)
    qm = tl.arange(0, BM).to(tl.int64); dd = tl.arange(0, D).to(tl.int64)
    gcol = dd // Gc; wcol = dd - gcol * Gc; colm = dd < Dkc
    q = tl.load(q_ptr + h * BM * D + qm[:, None] * D + dd[None, :]).to(tl.float16)
    cbh = cb_ptr + h * (NGc * Kc * Gc); ib = idx_ptr + h * (T * NGc)
    lo = s * CHUNK; hi = tl.minimum(lo + CHUNK, T); chk = 0.0
    for t0 in range(lo, hi, BT):
        o = t0 + tl.arange(0, BT).to(tl.int64); mk = o < hi; m2 = mk[:, None] & colm[None, :]
        isel = tl.load(ib + o[:, None] * NGc + gcol[None, :], mask=m2, other=0).to(tl.int64)
        kg = tl.load(cbh + gcol[None, :] * (Kc * Gc) + isel * Gc + wcol[None, :], mask=m2, other=0.0).to(tl.float16)
        qk = tl.where(mk[None, :], tl.dot(q, tl.trans(kg)), 0.0)
        chk += tl.sum(qk)
    tl.atomic_add(sink, chk)


@triton.jit
def _gather_dot_vec(q_ptr, cb64_ptr, idx_ptr, sink, T, CHUNK, ns, NGc: tl.constexpr, NGpc: tl.constexpr,
                    Kc: tl.constexpr, Gc: tl.constexpr, BM: tl.constexpr, BT: tl.constexpr, D: tl.constexpr):
    # VECTORIZED gather: codebook viewed as int64 (one G=4-fp16 codeword = 8 bytes = 1 int64) ->
    # ONE (BT,NG) int64 gather instead of (BT,128) fp16 loads (4x fewer LSU ops). Unpack to G
    # fp16 planes via shifts; plane j = coord j of every group = key coords j::G -> 1 tl.dot each.
    h = tl.program_id(0).to(tl.int64); s = tl.program_id(1).to(tl.int64)
    qm = tl.arange(0, BM).to(tl.int64); gng = tl.arange(0, NGpc).to(tl.int64); mg = gng < NGc
    cb64h = cb64_ptr + h * (NGc * Kc); ib = idx_ptr + h * (T * NGc)
    lo = s * CHUNK; hi = tl.minimum(lo + CHUNK, T); chk = 0.0
    for t0 in range(lo, hi, BT):
        o = t0 + tl.arange(0, BT).to(tl.int64); mk = o < hi; m2 = mk[:, None] & mg[None, :]
        isel = tl.load(ib + o[:, None] * NGc + gng[None, :], mask=m2, other=0).to(tl.int64)      # (BT,NG) coalesced
        cw = tl.load(cb64h + gng[None, :] * Kc + isel, mask=m2, other=0).to(tl.int64)             # (BT,NG) int64 gather
        score = tl.zeros([BM, BT], tl.float32)
        for j in tl.static_range(0, Gc):
            plane = ((cw >> (16 * j)) & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True)          # (BT,NG) coord-j
            qj = tl.load(q_ptr + h * BM * D + qm[:, None] * D + (j + Gc * gng)[None, :],
                         mask=mg[None, :], other=0.0).to(tl.float16)                               # (BM,NG)
            score += tl.dot(qj, tl.trans(plane))
        chk += tl.sum(tl.where(mk[None, :], score, 0.0))
    tl.atomic_add(sink, chk)


@triton.jit
def _gather_vec_nodot(cb64_ptr, idx_ptr, sink, T, CHUNK, ns, NGc: tl.constexpr, NGpc: tl.constexpr,
                      Kc: tl.constexpr, Gc: tl.constexpr, BT: tl.constexpr):
    h = tl.program_id(0).to(tl.int64); s = tl.program_id(1).to(tl.int64)
    gng = tl.arange(0, NGpc).to(tl.int64); mg = gng < NGc
    cb64h = cb64_ptr + h * (NGc * Kc); ib = idx_ptr + h * (T * NGc)
    lo = s * CHUNK; hi = tl.minimum(lo + CHUNK, T); chk = 0.0
    for t0 in range(lo, hi, BT):
        o = t0 + tl.arange(0, BT).to(tl.int64); mk = o < hi; m2 = mk[:, None] & mg[None, :]
        isel = tl.load(ib + o[:, None] * NGc + gng[None, :], mask=m2, other=0).to(tl.int64)
        cw = tl.load(cb64h + gng[None, :] * Kc + isel, mask=m2, other=0).to(tl.float32)
        chk += tl.sum(cw)
    tl.atomic_add(sink, chk)


@triton.jit
def _gather_dot_3d(q_ptr, cb_ptr, idx_ptr, sink, T, CHUNK, ns, NGc: tl.constexpr, NGpc: tl.constexpr,
                   Kc: tl.constexpr, Gc: tl.constexpr, Dkc: tl.constexpr, BM: tl.constexpr,
                   BT: tl.constexpr, D: tl.constexpr):
    # gather as (BT,NG,G) with the G dim CONTIGUOUS (last-dim stride 1 -> Triton vectorizes the
    # G-fp16 codeword into one wide load), reshape to (BT,NG*G) in natural coord order (g*G+w),
    # then ONE contraction-Dk tl.dot. Best of both: wide gather + single big dot.
    h = tl.program_id(0).to(tl.int64); s = tl.program_id(1).to(tl.int64)
    qm = tl.arange(0, BM).to(tl.int64); dd = tl.arange(0, D).to(tl.int64)
    gng = tl.arange(0, NGpc).to(tl.int64); ga = tl.arange(0, Gc).to(tl.int64)
    q = tl.load(q_ptr + h * BM * D + qm[:, None] * D + dd[None, :]).to(tl.float16)   # (BM,D), cols>=Dk unused
    cbh = cb_ptr + h * (NGc * Kc * Gc); ib = idx_ptr + h * (T * NGc)
    lo = s * CHUNK; hi = tl.minimum(lo + CHUNK, T); chk = 0.0
    for t0 in range(lo, hi, BT):
        o = t0 + tl.arange(0, BT).to(tl.int64); mk = o < hi
        isel = tl.load(ib + o[:, None] * NGc + gng[None, :], mask=mk[:, None] & (gng < NGc)[None, :], other=0).to(tl.int64)
        kg = tl.load(cbh + gng[None, :, None] * (Kc * Gc) + isel[:, :, None] * Gc + ga[None, None, :],
                     mask=mk[:, None, None] & (gng < NGc)[None, :, None], other=0.0)   # (BT,NG,G) G contiguous
        kg = tl.reshape(kg, (BT, NGpc * Gc)).to(tl.float16)                            # (BT,NG*G) coord order g*G+w
        qk = tl.where(mk[None, :], tl.dot(q, tl.trans(kg)), 0.0)
        chk += tl.sum(qk)
    tl.atomic_add(sink, chk)


@triton.jit
def _gather_dot_vec1(q_ptr, cb64_ptr, idx_ptr, sink, T, CHUNK, ns, NGc: tl.constexpr, NGpc: tl.constexpr,
                     Kc: tl.constexpr, Gc: tl.constexpr, BM: tl.constexpr, BT: tl.constexpr, D: tl.constexpr):
    # int64 wide gather (fast) + ONE contraction-D dot: reassemble the 4 fp16 planes with
    # join/reshape -- the int64 byte order already IS coord order (g*G+w), so q stays natural.
    h = tl.program_id(0).to(tl.int64); s = tl.program_id(1).to(tl.int64)
    qm = tl.arange(0, BM).to(tl.int64); dd = tl.arange(0, D).to(tl.int64)
    gng = tl.arange(0, NGpc).to(tl.int64)
    q = tl.load(q_ptr + h * BM * D + qm[:, None] * D + dd[None, :]).to(tl.float16)
    cb64h = cb64_ptr + h * (NGc * Kc); ib = idx_ptr + h * (T * NGc)
    lo = s * CHUNK; hi = tl.minimum(lo + CHUNK, T); chk = 0.0
    for t0 in range(lo, hi, BT):
        o = t0 + tl.arange(0, BT).to(tl.int64); mk = o < hi; m2 = mk[:, None] & (gng < NGc)[None, :]
        isel = tl.load(ib + o[:, None] * NGc + gng[None, :], mask=m2, other=0).to(tl.int64)
        cw = tl.load(cb64h + gng[None, :] * Kc + isel, mask=m2, other=0).to(tl.int64)          # (BT,NG)
        p0 = (cw & 0xFFFF).to(tl.int16, bitcast=False).to(tl.float16, bitcast=True)
        p1 = ((cw >> 16) & 0xFFFF).to(tl.int16, bitcast=False).to(tl.float16, bitcast=True)
        p2 = ((cw >> 32) & 0xFFFF).to(tl.int16, bitcast=False).to(tl.float16, bitcast=True)
        p3 = ((cw >> 48) & 0xFFFF).to(tl.int16, bitcast=False).to(tl.float16, bitcast=True)
        # reshape of join(join(a,b),join(c,d)) (BT,NG,2,2) flattens row-major -> col ng*4 + x*2 + y
        # (x = outer join, y = inner join); join(p0,p2)&join(p1,p3) then join -> [p0,p1,p2,p3].
        kg = tl.reshape(tl.join(tl.join(p0, p2), tl.join(p1, p3)), (BT, NGpc * Gc))            # (BT,NG*G) coord order
        qk = tl.where(mk[None, :], tl.dot(q, tl.trans(kg)), 0.0)
        chk += tl.sum(qk)
    tl.atomic_add(sink, chk)


def timeit(f, reps=40, warmup=15):
    for _ in range(warmup): f()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps): f()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / reps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=288); ap.add_argument("--T", type=int, default=65536)
    ap.add_argument("--ns", type=int, default=32); ap.add_argument("--BT", type=int, default=32)
    ap.add_argument("--G", type=int, nargs="+", default=[4, 6]); ap.add_argument("--warps", type=int, default=8)
    args = ap.parse_args()
    torch.manual_seed(0); H, T, ns, BT = args.H, args.T, args.ns, args.BT
    CHUNK = (T + ns - 1) // ns
    print(f"VQ read+qk DECOMPOSITION  H={H} T={T} ns={ns} BT={BT} warps={args.warps}")
    print(f"{'G':>3} {'NG':>4} {'cb/head':>9} {'idx_read':>9} {'+gather':>9} {'+dot':>9}   (ms)")
    for G in args.G:
        NG = d // G; K = 1 << (2 * G); Dk = NG * G; NGp = 1 << (NG - 1).bit_length()
        idx = torch.randint(0, K, (H, T, NG), device=dev, dtype=torch.int16)
        cb = torch.randn(H, NG, K, G, device=dev, dtype=torch.float16)
        q = torch.randn(H, BM, d, device=dev); sink = torch.zeros(1, device=dev)
        t_idx = timeit(lambda: _idx_only[(H, ns)](idx, sink, T, CHUNK, ns, NG, NGp, BT, num_warps=args.warps))
        t_gat = timeit(lambda: _gather_nodot[(H, ns)](cb, idx, sink, T, CHUNK, ns, NG, K, G, Dk, BT, d, num_warps=args.warps))
        t_dot = timeit(lambda: _gather_dot[(H, ns)](q, cb, idx, sink, T, CHUNK, ns, NG, K, G, Dk, BM, BT, d, num_warps=args.warps))
        vec = ""
        if G == 4:  # int64-vectorizable (8-byte codeword); NG=32 is pow2 (no mask waste)
            cb64 = cb.view(torch.int64).squeeze(-1).contiguous()   # (H,NG,K)
            s1 = torch.zeros(1, device=dev); s2 = torch.zeros(1, device=dev)
            _gather_dot[(H, ns)](q, cb, idx, s1, T, CHUNK, ns, NG, K, G, Dk, BM, BT, d, num_warps=args.warps)
            _gather_dot_vec[(H, ns)](q, cb64, idx, s2, T, CHUNK, ns, NG, NGp, K, G, BM, BT, d, num_warps=args.warps)
            rel = (s1 - s2).abs().item() / (s1.abs().item() + 1e-6)
            s3 = torch.zeros(1, device=dev)
            _gather_dot_3d[(H, ns)](q, cb, idx, s3, T, CHUNK, ns, NG, NGp, K, G, Dk, BM, BT, d, num_warps=args.warps)
            rel3 = (s1 - s3).abs().item() / (s1.abs().item() + 1e-6)
            t_vgat = timeit(lambda: _gather_vec_nodot[(H, ns)](cb64, idx, sink, T, CHUNK, ns, NG, NGp, K, G, BT, num_warps=args.warps))
            t_vec = timeit(lambda: _gather_dot_vec[(H, ns)](q, cb64, idx, sink, T, CHUNK, ns, NG, NGp, K, G, BM, BT, d, num_warps=args.warps))
            s4 = torch.zeros(1, device=dev)
            _gather_dot_vec1[(H, ns)](q, cb64, idx, s4, T, CHUNK, ns, NG, NGp, K, G, BM, BT, d, num_warps=args.warps)
            rel4 = (s1 - s4).abs().item() / (s1.abs().item() + 1e-6)
            t_v1 = timeit(lambda: _gather_dot_vec1[(H, ns)](q, cb64, idx, sink, T, CHUNK, ns, NG, NGp, K, G, BM, BT, d, num_warps=args.warps))
            vec = (f"  vec_gather={t_vgat:.3f} vec_dot4={t_vec:.3f}[{rel:.0e}] "
                   f"vec_dot1={t_v1:.3f}[{rel4:.0e}]"); del cb64
        print(f"{G:>3} {NG:>4} {NG*K*G*2/1024:>7.0f}KB {t_idx:>8.3f} {t_gat:>8.3f} {t_dot:>8.3f}{vec}", flush=True)
        del idx, cb, q; torch.cuda.empty_cache()
