#!/usr/bin/env python3
"""Tune the int64-VEC VQ read+qk kernel (G=4) -- sweep BT/num_warps/num_stages, and isolate
gather vs unpack vs dot -- to push VQ full-decode below BF16. T=65536, 288 heads."""
import argparse, torch, triton, triton.language as tl
dev = "cuda"; d = 128; BM = 16; G = 4; NG = d // G; K = 1 << (2 * G); NGp = NG  # 32 (pow2)


@triton.jit
def _vec(q_ptr, cb64_ptr, idx_ptr, sink, T, CHUNK, ns, MODE: tl.constexpr, NGc: tl.constexpr,
         NGpc: tl.constexpr, Kc: tl.constexpr, Gc: tl.constexpr, BM: tl.constexpr, BT: tl.constexpr, D: tl.constexpr):
    # MODE 0=gather-only 1=gather+unpack(no dot) 2=gather+unpack+dot
    h = tl.program_id(0).to(tl.int64); s = tl.program_id(1).to(tl.int64)
    qm = tl.arange(0, BM).to(tl.int64); dd = tl.arange(0, D).to(tl.int64); gng = tl.arange(0, NGpc).to(tl.int64)
    q = tl.load(q_ptr + h * BM * D + qm[:, None] * D + dd[None, :]).to(tl.float16)
    cb64h = cb64_ptr + h * (NGc * Kc); ib = idx_ptr + h * (T * NGc)
    lo = s * CHUNK; hi = tl.minimum(lo + CHUNK, T); chk = 0.0
    for t0 in range(lo, hi, BT):
        o = t0 + tl.arange(0, BT).to(tl.int64); mk = o < hi; m2 = mk[:, None] & (gng < NGc)[None, :]
        isel = tl.load(ib + o[:, None] * NGc + gng[None, :], mask=m2, other=0).to(tl.int64)
        cw = tl.load(cb64h + gng[None, :] * Kc + isel, mask=m2, other=0).to(tl.int64)
        if MODE == 0:
            chk += tl.sum(cw.to(tl.float32))
        else:
            p0 = (cw & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True)
            p1 = ((cw >> 16) & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True)
            p2 = ((cw >> 32) & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True)
            p3 = ((cw >> 48) & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True)
            kg = tl.reshape(tl.join(tl.join(p0, p2), tl.join(p1, p3)), (BT, NGpc * Gc))
            if MODE == 1:
                chk += tl.sum(kg.to(tl.float32))
            else:
                chk += tl.sum(tl.where(mk[None, :], tl.dot(q, tl.trans(kg)), 0.0))
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
    ap.add_argument("--ns", type=int, default=32)
    ap.add_argument("--BT", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--warps", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--stages", type=int, nargs="+", default=[2, 3])
    args = ap.parse_args()
    torch.manual_seed(0); H, T, ns = args.H, args.T, args.ns; CHUNK = (T + ns - 1) // ns
    idx = torch.randint(0, K, (H, T, NG), device=dev, dtype=torch.int16)
    cb = torch.randn(H, NG, K, G, device=dev, dtype=torch.float16)
    cb64 = cb.view(torch.int64).squeeze(-1).contiguous()
    q = torch.randn(H, BM, d, device=dev); sink = torch.zeros(1, device=dev)
    print(f"VEC read+qk tune G=4 NG={NG} K={K}  H={H} T={T}")
    print(f"{'BT':>4} {'warps':>6} {'stages':>7} {'gather':>8} {'+unpack':>8} {'+dot':>8}   (ms)")
    for BT in args.BT:
        for w in args.warps:
            for st in args.stages:
                try:
                    t = []
                    for MODE in (0, 1, 2):
                        f = lambda MODE=MODE: _vec[(H, ns)](q, cb64, idx, sink, T, CHUNK, ns, MODE, NG, NGp, K, G, BM, BT, d, num_warps=w, num_stages=st)
                        t.append(timeit(f))
                    print(f"{BT:>4} {w:>6} {st:>7} {t[0]:>8.3f} {t[1]:>8.3f} {t[2]:>8.3f}", flush=True)
                except Exception as e:
                    print(f"{BT:>4} {w:>6} {st:>7}   FAIL {str(e)[:50]}", flush=True)
