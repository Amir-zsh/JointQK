#!/usr/bin/env python3
"""Fast tuning harness for the VQ LUT decode kernel (read+qk depth only, T=65536).
Sweeps num_warps / num_stages / BLOCK_T to find why the 21-group LUT gather is
latency-bound (fp16 LUT didn't help read+qk -> not bandwidth). Isolates the qk
gather so each config compiles fast."""
import argparse, torch, triton, triton.language as tl

dev = "cuda"
d = 128
BLOCK_M = 16
NG, G, K = 21, 6, 4096


@triton.jit
def _vq_qk(lut_ptr, idx_ptr, sink_ptr, T, CHUNK, ns,
           NGc: tl.constexpr, Kc: tl.constexpr, BM: tl.constexpr, BT: tl.constexpr):
    h = tl.program_id(0).to(tl.int64); s = tl.program_id(1).to(tl.int64)
    qm = tl.arange(0, BM).to(tl.int64)
    lh = lut_ptr + h * (NGc * Kc * BM); ib = idx_ptr + h * (NGc * T)
    lo = s * CHUNK; hi = tl.minimum(lo + CHUNK, T)
    chk = 0.0
    for t0 in range(lo, hi, BT):
        o = t0 + tl.arange(0, BT).to(tl.int64); mk = o < hi
        score = tl.zeros([BT, BM], tl.float32)
        for g in tl.static_range(0, NGc):
            ig = tl.load(ib + g * T + o, mask=mk, other=0).to(tl.int64)
            score += tl.load(lh + g * (Kc * BM) + ig[:, None] * BM + qm[None, :]).to(tl.float32)
        chk += tl.sum(tl.where(mk[:, None], score, 0.0))
    tl.atomic_add(sink_ptr, chk)


@triton.jit
def _vq_cb(q_ptr, cb_ptr, idx_ptr, sink_ptr, T, CHUNK, ns,
          NGc: tl.constexpr, Kc: tl.constexpr, Gc: tl.constexpr,
          BM: tl.constexpr, BT: tl.constexpr, D: tl.constexpr):
    # reconstruct-key path: gather codeword cb[g, idx[g,t]] (G contiguous fp16, coalesced
    # per group) and accumulate score via tensor-core partial dots. cb: (H,NG,K,G).
    h = tl.program_id(0).to(tl.int64); s = tl.program_id(1).to(tl.int64)
    qm = tl.arange(0, BM).to(tl.int64); gj = tl.arange(0, 16).to(tl.int64)
    qb = q_ptr + h * (BM * D); cbh = cb_ptr + h * (NGc * Kc * Gc); ib = idx_ptr + h * (NGc * T)
    lo = s * CHUNK; hi = tl.minimum(lo + CHUNK, T)
    chk = 0.0
    for t0 in range(lo, hi, BT):
        o = t0 + tl.arange(0, BT).to(tl.int64); mk = o < hi
        score = tl.zeros([BM, BT], tl.float32)
        for g in tl.static_range(0, NGc):
            ig = tl.load(ib + g * T + o, mask=mk, other=0).to(tl.int64)                       # (BT,)
            mkg = gj < Gc
            kg = tl.load(cbh + g * (Kc * Gc) + ig[:, None] * Gc + gj[None, :],
                         mask=(mk[:, None] & mkg[None, :]), other=0.0).to(tl.float16)          # (BT,16)
            qg = tl.load(qb + qm[:, None] * D + (g * Gc + gj)[None, :],
                         mask=mkg[None, :], other=0.0).to(tl.float16)                          # (BM,16)
            score += tl.dot(qg, tl.trans(kg))
        chk += tl.sum(tl.where(mk[None, :], score, 0.0))
    tl.atomic_add(sink_ptr, chk)


@triton.jit
def _vq_cb2(q_ptr, cb_ptr, idx_ptr, sink_ptr, T, CHUNK, ns,
           NGc: tl.constexpr, Kc: tl.constexpr, Gc: tl.constexpr, Dk: tl.constexpr,
           BM: tl.constexpr, BT: tl.constexpr, D: tl.constexpr):
    # ONE fused (BT,D) codeword gather + ONE tl.dot per tile (vs NG small gathers).
    # column c -> group c//G, within c%G; cb: (H,NG,K,G), idx: (H,NG,T).
    h = tl.program_id(0).to(tl.int64); s = tl.program_id(1).to(tl.int64)
    qm = tl.arange(0, BM).to(tl.int64); dd = tl.arange(0, D).to(tl.int64)
    gcol = dd // Gc; wcol = dd - gcol * Gc; colm = dd < Dk
    qb = q_ptr + h * (BM * D); cbh = cb_ptr + h * (NGc * Kc * Gc); ib = idx_ptr + h * (NGc * T)
    q = tl.load(qb + qm[:, None] * D + dd[None, :]).to(tl.float16)
    lo = s * CHUNK; hi = tl.minimum(lo + CHUNK, T)
    chk = 0.0
    for t0 in range(lo, hi, BT):
        o = t0 + tl.arange(0, BT).to(tl.int64); mk = o < hi
        m2 = mk[:, None] & colm[None, :]
        isel = tl.load(ib + gcol[None, :] * T + o[:, None], mask=m2, other=0).to(tl.int64)      # (BT,D)
        kg = tl.load(cbh + gcol[None, :] * (Kc * Gc) + isel * Gc + wcol[None, :],
                     mask=m2, other=0.0).to(tl.float16)                                          # (BT,D)
        score = tl.dot(q, tl.trans(kg))                                                          # (BM,BT)
        chk += tl.sum(tl.where(mk[None, :], score, 0.0))
    tl.atomic_add(sink_ptr, chk)


def timeit(f, reps=30, warmup=10):
    for _ in range(warmup): f()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps): f()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=288)
    ap.add_argument("--T", type=int, default=65536)
    ap.add_argument("--ns", type=int, default=32)
    ap.add_argument("--BT", type=int, nargs="+", default=[32, 64])
    ap.add_argument("--warps", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--K", type=int, default=K)
    ap.add_argument("--kernels", type=str, nargs="+", default=["LUT", "CB", "CB2"])
    args = ap.parse_args()
    torch.manual_seed(0)
    K = args.K
    H, T, ns = args.H, args.T, args.ns
    CHUNK = (T + ns - 1) // ns
    idx_gm = torch.randint(0, K, (H, NG, T), device=dev, dtype=torch.int16)
    LUT_t = torch.randn(H, NG, K, BLOCK_M, device=dev, dtype=torch.float16)
    q = torch.randn(H, BLOCK_M, d, device=dev)
    cb = torch.randn(H, NG, K, G, device=dev, dtype=torch.float16)
    sink = torch.zeros(1, device=dev)
    print(f"VQ read+qk tune, H={H} T={T} ns={ns}. LUT gather={H*NG*T*BLOCK_M*2/1e9:.1f}GB  "
          f"CB gather={H*NG*T*G*2/1e9:.1f}GB")
    print(f"{'kernel':>7} {'BT':>4} {'warps':>6} {'stages':>7} {'ms':>9}")
    for BT in args.BT:
        for w in args.warps:
            for st in args.stages:
                for kn in args.kernels:
                    try:
                        if kn == "LUT":
                            f = lambda: _vq_qk[(H, ns)](LUT_t, idx_gm, sink, T, CHUNK, ns,
                                                        NG, K, BLOCK_M, BT, num_warps=w, num_stages=st)
                        elif kn == "CB":
                            f = lambda: _vq_cb[(H, ns)](q, cb, idx_gm, sink, T, CHUNK, ns,
                                                        NG, K, G, BLOCK_M, BT, d, num_warps=w, num_stages=st)
                        else:
                            f = lambda: _vq_cb2[(H, ns)](q, cb, idx_gm, sink, T, CHUNK, ns,
                                                         NG, K, G, NG * G, BLOCK_M, BT, d, num_warps=w, num_stages=st)
                        t = timeit(f)
                        print(f"{kn:>7} {BT:>4} {w:>6} {st:>7} {t:>8.3f}", flush=True)
                    except Exception as e:
                        print(f"{kn:>7} {BT:>4} {w:>6} {st:>7}   FAIL {str(e)[:60]}", flush=True)
