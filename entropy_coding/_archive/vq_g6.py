#!/usr/bin/env python3
"""VQ G=6 wide-load gather: codeword = 6 fp16 = 12 B. Split into int64 (coords 0-3) + int32
(coords 4-5) -> 2 wide loads/group instead of 6 fp16 loads. Unpack to 6 planes, assemble two
sub-tiles (BT,NG*4) and (BT,NG*2), two tensor-core dots. Correctness-gated + BT sweep. T=65536."""
import argparse, torch, triton, triton.language as tl
dev = "cuda"; d = 128; BM = 16; G = 6; NG = d // G; K = 1 << (2 * G); Dk = NG * G  # NG=21,K=4096,Dk=126
NGp = 1 << (NG - 1).bit_length()   # 32


@triton.jit
def _g6_scalar(q_ptr, cb_ptr, idx_ptr, sink, T, CHUNK, ns, NGc: tl.constexpr, Kc: tl.constexpr,
               Gc: tl.constexpr, Dkc: tl.constexpr, BM: tl.constexpr, BT: tl.constexpr, D: tl.constexpr):
    # reference: per-coord fp16 gather + one dot (current shipped G!=4 path).
    h = tl.program_id(0).to(tl.int64); s = tl.program_id(1).to(tl.int64)
    qm = tl.arange(0, BM).to(tl.int64); dd = tl.arange(0, D).to(tl.int64)
    gcol = dd // Gc; wcol = dd - gcol * Gc; colm = dd < Dkc
    q = tl.load(q_ptr + h * BM * D + qm[:, None] * D + dd[None, :]).to(tl.float16)
    ib = idx_ptr + h * (T * NGc); lo = s * CHUNK; hi = tl.minimum(lo + CHUNK, T); chk = 0.0
    for t0 in range(lo, hi, BT):
        o = t0 + tl.arange(0, BT).to(tl.int64); mk = o < hi; m2 = mk[:, None] & colm[None, :]
        isel = tl.load(ib + o[:, None] * NGc + gcol[None, :], mask=m2, other=0).to(tl.int64)
        kg = tl.load(cb_ptr + h * (NGc * Kc * Gc) + gcol[None, :] * (Kc * Gc) + isel * Gc + wcol[None, :],
                     mask=m2, other=0.0).to(tl.float16)
        chk += tl.sum(tl.where(mk[None, :], tl.dot(q, tl.trans(kg)), 0.0))
    tl.atomic_add(sink, chk)


@triton.jit
def _g6_wide(q_ptr, cblo_ptr, cbhi_ptr, idx_ptr, sink, T, CHUNK, ns, NGc: tl.constexpr,
             NGpc: tl.constexpr, Kc: tl.constexpr, BM: tl.constexpr, BT: tl.constexpr, D: tl.constexpr):
    # cblo: (H,NG,K) int64 = coords 0-3;  cbhi: (H,NG,K) int32 = coords 4-5.
    h = tl.program_id(0).to(tl.int64); s = tl.program_id(1).to(tl.int64)
    qm = tl.arange(0, BM).to(tl.int64); gng = tl.arange(0, NGpc).to(tl.int64); mg = gng < NGc
    # q sub-tiles in the assembled coord order (col -> q coord)
    cA = tl.arange(0, NGpc * 4).to(tl.int64); gA = cA // 4; qA = tl.load(
        q_ptr + h * BM * D + qm[:, None] * D + (gA * 6 + (cA - gA * 4))[None, :],
        mask=(gA < NGc)[None, :], other=0.0).to(tl.float16)                       # (BM, NGp*4)
    cB = tl.arange(0, NGpc * 2).to(tl.int64); gB = cB // 2; qB = tl.load(
        q_ptr + h * BM * D + qm[:, None] * D + (gB * 6 + 4 + (cB - gB * 2))[None, :],
        mask=(gB < NGc)[None, :], other=0.0).to(tl.float16)                       # (BM, NGp*2)
    clo = cblo_ptr + h * (NGc * Kc); chi = cbhi_ptr + h * (NGc * Kc); ib = idx_ptr + h * (T * NGc)
    lo = s * CHUNK; hi = tl.minimum(lo + CHUNK, T); chk = 0.0
    for t0 in range(lo, hi, BT):
        o = t0 + tl.arange(0, BT).to(tl.int64); mk = o < hi; m2 = mk[:, None] & mg[None, :]
        isel = tl.load(ib + o[:, None] * NGc + gng[None, :], mask=m2, other=0).to(tl.int64)
        w0 = tl.load(clo + gng[None, :] * Kc + isel, mask=m2, other=0).to(tl.int64)   # (BT,NGp) coords0-3
        w1 = tl.load(chi + gng[None, :] * Kc + isel, mask=m2, other=0).to(tl.int32)   # (BT,NGp) coords4-5
        p0 = (w0 & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True)
        p1 = ((w0 >> 16) & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True)
        p2 = ((w0 >> 32) & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True)
        p3 = ((w0 >> 48) & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True)
        p4 = (w1 & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True)
        p5 = ((w1 >> 16) & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True)
        kgA = tl.reshape(tl.join(tl.join(p0, p2), tl.join(p1, p3)), (BT, NGpc * 4))   # [g:c0,c1,c2,c3]
        kgB = tl.reshape(tl.join(p4, p5), (BT, NGpc * 2))                             # [g:c4,c5]
        score = tl.dot(qA, tl.trans(kgA)) + tl.dot(qB, tl.trans(kgB))
        chk += tl.sum(tl.where(mk[None, :], score, 0.0))
    tl.atomic_add(sink, chk)


def timeit(f, reps=60, warmup=20):
    for _ in range(warmup): f()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps): f()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / reps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=288); ap.add_argument("--T", type=int, default=65536)
    ap.add_argument("--ns", type=int, default=32); ap.add_argument("--BT", type=int, nargs="+", default=[32, 64, 128])
    args = ap.parse_args()
    torch.manual_seed(0); H, T, ns = args.H, args.T, args.ns; CHUNK = (T + ns - 1) // ns
    idx = torch.randint(0, K, (H, T, NG), device=dev, dtype=torch.int16)
    cb = torch.randn(H, NG, K, G, device=dev, dtype=torch.float16)
    cblo = cb[..., :4].contiguous().view(torch.int64).squeeze(-1).contiguous()   # (H,NG,K)
    cbhi = cb[..., 4:6].contiguous().view(torch.int32).squeeze(-1).contiguous()  # (H,NG,K)
    q = torch.randn(H, BM, d, device=dev)
    print(f"VQ G=6 wide-load  H={H} T={T} NG={NG} K={K} cb/head={NG*K*G*2/1024:.0f}KB")
    s0 = torch.zeros(1, device=dev); s1 = torch.zeros(1, device=dev)
    _g6_scalar[(H, ns)](q, cb, idx, s0, T, CHUNK, ns, NG, K, G, Dk, BM, 32, d, num_warps=8)
    _g6_wide[(H, ns)](q, cblo, cbhi, idx, s1, T, CHUNK, ns, NG, NGp, K, BM, 32, d, num_warps=8)
    print(f"correctness rel_err = {(s0-s1).abs().item()/(s0.abs().item()+1e-6):.2e}")
    print(f"{'BT':>4} {'scalar':>8} {'wide':>8}   (read+qk ms)")
    for BT in args.BT:
        ts = timeit(lambda: _g6_scalar[(H, ns)](q, cb, idx, s0, T, CHUNK, ns, NG, K, G, Dk, BM, BT, d, num_warps=8))
        tw = timeit(lambda: _g6_wide[(H, ns)](q, cblo, cbhi, idx, s1, T, CHUNK, ns, NG, NGp, K, BM, BT, d, num_warps=8))
        print(f"{BT:>4} {ts:>8.3f} {tw:>8.3f}", flush=True)
