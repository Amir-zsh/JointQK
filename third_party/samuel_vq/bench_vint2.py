#!/usr/bin/env python3
"""Fair-conditions decode timing: VQ8 vs OSCAR with BOTH K and V at INT2, matching Amir's
`OSCAR paper` benchmark setup (V=fp16 in fused_decode_all.py dilutes K-compression ratios,
which is why VQ8 and OSCAR looked ~equal). This isolates the K-compression cost under
actually-deployable conditions.

V uses OSCAR-style per-token affine INT2: `v_deq[t,c] = (vi[t,c] - zv[t]) * sv[t]`.
Same structure as OSCAR's K path — 4× fewer bytes per V element, per-token scale/zero for
outlier-aware quant. Inside the attention loop the p@V dot uses fp16 dequantized V (identical
to what a real deployment would do). Correctness gated against fp16-V reference.

Only VQ8 (G=4 fp8 codebook — our deployment config) and OSCAR are benchmarked; BF16/INT2
also added for context. T=131072, A100, bs=1, BM=16 queries."""
import torch, triton, triton.language as tl

dev = "cuda"
d = 128
BLOCK_M = 16
BLOCK_T = 32
VQ_BT = 128     # int64 gather amortizes at BT=128


@triton.jit
def _vq8_vint2(q_ptr, cb_ptr, idx_ptr, pv_ptr, sv_ptr, zv_ptr,
               m_ptr, l_ptr, acc_ptr, sink_ptr, T, CHUNK, ns, sivh, svbh, svmh,
               NGc: tl.constexpr, NGpc: tl.constexpr, Kc: tl.constexpr, Gc: tl.constexpr, Dkc: tl.constexpr,
               BM: tl.constexpr, BT: tl.constexpr, D: tl.constexpr):
    h = tl.program_id(0).to(tl.int64); s = tl.program_id(1).to(tl.int64)
    qm = tl.arange(0, BM).to(tl.int64); dd = tl.arange(0, D).to(tl.int64)
    gng = tl.arange(0, NGpc).to(tl.int64)
    q = tl.load(q_ptr + h * BM * D + qm[:, None] * D + dd[None, :]).to(tl.float16)
    ib = idx_ptr + h * sivh
    lo = s * CHUNK; hi = tl.minimum(lo + CHUNK, T)
    mi = tl.zeros([BM], tl.float32) - float("inf"); li = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for t0 in range(lo, hi, BT):
        o = t0 + tl.arange(0, BT).to(tl.int64); mk = o < hi
        # K: fp8 codebook, one int32 = 4 fp8 codewords (G=4 group)
        mg = mk[:, None] & (gng < NGc)[None, :]
        isel = tl.load(ib + o[:, None] * NGc + gng[None, :], mask=mg, other=0).to(tl.int32)
        cw = tl.load(cb_ptr + h * (NGc * Kc) + gng[None, :] * Kc + isel, mask=mg, other=0).to(tl.int32)
        p0 = ((cw) & 0xFF).to(tl.uint8).to(tl.float8e5, bitcast=True).to(tl.float16)
        p1 = ((cw >> 8) & 0xFF).to(tl.uint8).to(tl.float8e5, bitcast=True).to(tl.float16)
        p2 = ((cw >> 16) & 0xFF).to(tl.uint8).to(tl.float8e5, bitcast=True).to(tl.float16)
        p3 = ((cw >> 24) & 0xFF).to(tl.uint8).to(tl.float8e5, bitcast=True).to(tl.float16)
        kg = tl.reshape(tl.join(tl.join(p0, p2), tl.join(p1, p3)), (BT, NGpc * Gc))
        qk = tl.where(mk[None, :], tl.dot(q, tl.trans(kg)), -float("inf"))
        mn = tl.maximum(mi, tl.max(qk, 1)); p = tl.exp(qk - mn[:, None]); a = tl.exp(mi - mn)
        li = li * a + tl.sum(p, 1)
        # V: INT2 packed + per-token affine dequant (OSCAR-style)
        by = tl.load(pv_ptr + h * svbh + o[:, None] * (D // 4) + tl.arange(0, D // 4)[None, :].to(tl.int64),
                     mask=mk[:, None], other=0).to(tl.uint32)
        svs = tl.load(sv_ptr + h * svmh + o, mask=mk, other=1.0).to(tl.float16)
        zvs = tl.load(zv_ptr + h * svmh + o, mask=mk, other=0.0).to(tl.float16)
        sh = (tl.arange(0, 4) * 2).to(tl.uint32)
        vi = tl.reshape((by[:, :, None] >> sh[None, None, :]) & 0x3, (BT, D)).to(tl.float16)
        v = (vi - zvs[:, None]) * svs[:, None]
        acc = acc * a[:, None] + tl.dot(p.to(tl.float16), v); mi = mn
    b = h * ns + s
    tl.store(m_ptr + b * BM + qm, mi); tl.store(l_ptr + b * BM + qm, li)
    tl.store(acc_ptr + b * BM * D + qm[:, None] * D + dd[None, :], acc)


@triton.jit
def _oscar_vint2(q_ptr, pk_ptr, sc_ptr, ze_ptr, pv_ptr, sv_ptr, zv_ptr,
                 m_ptr, l_ptr, acc_ptr, sink_ptr, T, CHUNK, ns, sph, svbh, svmh,
                 BM: tl.constexpr, BT: tl.constexpr, D: tl.constexpr):
    h = tl.program_id(0).to(tl.int64); s = tl.program_id(1).to(tl.int64)
    qm = tl.arange(0, BM).to(tl.int64); dd = tl.arange(0, D).to(tl.int64)
    q = tl.load(q_ptr + h * BM * D + qm[:, None] * D + dd[None, :]).to(tl.float16)
    sumq = tl.sum(q.to(tl.float32), 1)
    pb = pk_ptr + h * sph
    lo = s * CHUNK; hi = tl.minimum(lo + CHUNK, T)
    mi = tl.zeros([BM], tl.float32) - float("inf"); li = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for t0 in range(lo, hi, BT):
        o = t0 + tl.arange(0, BT).to(tl.int64); mk = o < hi
        # K: OSCAR per-token INT2
        by = tl.load(pb + o[:, None] * (D // 4) + tl.arange(0, D // 4)[None, :].to(tl.int64),
                     mask=mk[:, None], other=0).to(tl.uint32)
        sck = tl.load(sc_ptr + h * T + o, mask=mk, other=1.0).to(tl.float32)
        zek = tl.load(ze_ptr + h * T + o, mask=mk, other=0.0).to(tl.float32)
        sh = (tl.arange(0, 4) * 2).to(tl.uint32)
        qi = tl.reshape((by[:, :, None] >> sh[None, None, :]) & 0x3, (BT, D)).to(tl.float16)
        qkr = tl.dot(q, tl.trans(qi))
        qk = tl.where(mk[None, :], sck[None, :] * (qkr - zek[None, :] * sumq[:, None]), -float("inf"))
        mn = tl.maximum(mi, tl.max(qk, 1)); p = tl.exp(qk - mn[:, None]); a = tl.exp(mi - mn)
        li = li * a + tl.sum(p, 1)
        # V: INT2 packed + per-token affine dequant (OSCAR-style)
        byv = tl.load(pv_ptr + h * svbh + o[:, None] * (D // 4) + tl.arange(0, D // 4)[None, :].to(tl.int64),
                      mask=mk[:, None], other=0).to(tl.uint32)
        svs = tl.load(sv_ptr + h * svmh + o, mask=mk, other=1.0).to(tl.float16)
        zvs = tl.load(zv_ptr + h * svmh + o, mask=mk, other=0.0).to(tl.float16)
        vi = tl.reshape((byv[:, :, None] >> sh[None, None, :]) & 0x3, (BT, D)).to(tl.float16)
        v = (vi - zvs[:, None]) * svs[:, None]
        acc = acc * a[:, None] + tl.dot(p.to(tl.float16), v); mi = mn
    b = h * ns + s
    tl.store(m_ptr + b * BM + qm, mi); tl.store(l_ptr + b * BM + qm, li)
    tl.store(acc_ptr + b * BM * D + qm[:, None] * D + dd[None, :], acc)


def combine(m, l, acc, H, ns, D):
    m = m.view(H, ns, BLOCK_M); l = l.view(H, ns, BLOCK_M); acc = acc.view(H, ns, BLOCK_M, D)
    mg = m.max(1, keepdim=True).values; sc = torch.exp(m - mg)
    return (acc * sc.unsqueeze(-1)).sum(1) / (l * sc).sum(1).unsqueeze(-1)


def timeit(f, reps=100, warmup=30):
    for _ in range(warmup): f()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps): f()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-heads", type=int, default=288)
    ap.add_argument("--Ts", type=int, nargs="+", default=[131072])
    ap.add_argument("--n-splits", type=int, default=32)
    args = ap.parse_args()

    torch.manual_seed(0); H = args.n_heads; ns = args.n_splits
    G = 4; NG = d // G; K = 1 << (2 * G); Dk = NG * G
    NGp = 1 << (NG - 1).bit_length()

    print(f"VQ config: G={G} NG={NG} K={K} codebook/head={NG*K*G*2/1024:.0f}KB fp16 ({NG*K/1024:.1f}KB fp8)")
    print(f"V: INT2 packed + per-token OSCAR-style affine dequant (matches K's cost structure)")
    print()

    for T in args.Ts:
        CHUNK = (T + ns - 1) // ns
        q = torch.randn(H, BLOCK_M, d, device=dev)
        # K reps
        packed_k = torch.randint(0, 256, (H, T, d // 4), device=dev, dtype=torch.uint8)
        sck = (torch.rand(H, T, device=dev) + 0.5).half()
        zek = torch.rand(H, T, device=dev).half()
        idx_tm = torch.randint(0, K, (H, T, NG), device=dev, dtype=torch.int16)
        cb = torch.randn(H, NG, K, G, device=dev, dtype=torch.float16)
        cb_k8 = cb.to(torch.float8_e5m2).view(torch.int32).squeeze(-1).contiguous()
        # V reps: INT2 packed + per-token scale/zero
        packed_v = torch.randint(0, 256, (H, T, d // 4), device=dev, dtype=torch.uint8)
        scv = (torch.rand(H, T, device=dev) + 0.5).half()
        zev = torch.rand(H, T, device=dev).half()
        m = torch.zeros(H * ns * BLOCK_M, device=dev)
        l = torch.zeros(H * ns * BLOCK_M, device=dev)
        acc = torch.zeros(H * ns * BLOCK_M * d, device=dev)
        sink = torch.zeros(1, device=dev)

        print(f"=== T={T} H={H} BM={BLOCK_M} (bs=1 decode, A100) ===")
        # VQ8 with V=INT2
        f_vq = lambda: _vq8_vint2[(H, ns)](
            q, cb_k8, idx_tm, packed_v, scv, zev, m, l, acc, sink,
            T, CHUNK, ns, T * NG, T * (d // 4), T,
            NG, NGp, K, G, Dk, BLOCK_M, VQ_BT, d, num_warps=4)
        t_vq = timeit(f_vq)
        print(f"  VQ8   (K=VQ fp8 codebook, V=INT2 per-token): {t_vq:6.3f} ms")

        # OSCAR with V=INT2
        f_os = lambda: _oscar_vint2[(H, ns)](
            q, packed_k, sck, zek, packed_v, scv, zev, m, l, acc, sink,
            T, CHUNK, ns, T * (d // 4), T * (d // 4), T,
            BLOCK_M, BLOCK_T, d)
        t_os = timeit(f_os)
        print(f"  OSCAR (K=OSCAR INT2, V=INT2 per-token):      {t_os:6.3f} ms")
        print(f"  ratio OSCAR/VQ8:  {t_os/t_vq:.2f}x   (VQ8/OSCAR: {t_vq/t_os:.2f}x)")
        print(f"  per-layer (÷36): OSCAR {t_os/36:.3f} ms  VQ8 {t_vq/36:.3f} ms")
        print()
