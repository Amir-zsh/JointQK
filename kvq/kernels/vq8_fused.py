# Ported from Samuel's JointQK entropy_coding/bench_vint2.py (commit 3c65507),
# with his permission — see third_party/samuel_vq/PROVENANCE.md. His fused
# single-kernel decode design: dequant + online softmax + V in ONE kernel,
# whole model per launch (H = layers*kv_heads programs x splits), fp8e5m2
# codebook with one int32 load per 4-coord codeword, BT=128 token tiles.
# This architecture is what closes VQ to within ~16% of the authors' OSCAR
# kernel at bs=1 (report10 Amendment A10-2); our two-phase harness charged
# VQ ~3x for its u-buffer round-trip.
"""Fused group-VQ (fp8 codebook) and OSCAR-style INT2 decode+attention
kernels, both with INT2 per-token-affine V. Returns (H, BM, D) outputs;
per-layer time = whole-model time / n_layers."""
import torch
import triton
import triton.language as tl


@triton.jit
def _vq8_vint2(q_ptr, cb_ptr, idx_ptr, pv_ptr, sv_ptr, zv_ptr,
               m_ptr, l_ptr, acc_ptr, T, CHUNK, ns, sivh, svbh, svmh,
               NGc: tl.constexpr, NGpc: tl.constexpr, Kc: tl.constexpr,
               Gc: tl.constexpr,
               BM: tl.constexpr, BT: tl.constexpr, D: tl.constexpr):
    h = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1).to(tl.int64)
    qm = tl.arange(0, BM).to(tl.int64)
    dd = tl.arange(0, D).to(tl.int64)
    gng = tl.arange(0, NGpc).to(tl.int64)
    q = tl.load(q_ptr + h * BM * D + qm[:, None] * D + dd[None, :]).to(tl.float16)
    ib = idx_ptr + h * sivh
    lo = s * CHUNK
    hi = tl.minimum(lo + CHUNK, T)
    mi = tl.zeros([BM], tl.float32) - float("inf")
    li = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for t0 in range(lo, hi, BT):
        o = t0 + tl.arange(0, BT).to(tl.int64)
        mk = o < hi
        mg = mk[:, None] & (gng < NGc)[None, :]
        isel = tl.load(ib + o[:, None] * NGc + gng[None, :], mask=mg, other=0).to(tl.int32)
        cw = tl.load(cb_ptr + h * (NGc * Kc) + gng[None, :] * Kc + isel,
                     mask=mg, other=0).to(tl.int32)
        p0 = (cw & 0xFF).to(tl.uint8).to(tl.float8e5, bitcast=True).to(tl.float16)
        p1 = ((cw >> 8) & 0xFF).to(tl.uint8).to(tl.float8e5, bitcast=True).to(tl.float16)
        p2 = ((cw >> 16) & 0xFF).to(tl.uint8).to(tl.float8e5, bitcast=True).to(tl.float16)
        p3 = ((cw >> 24) & 0xFF).to(tl.uint8).to(tl.float8e5, bitcast=True).to(tl.float16)
        kg = tl.reshape(tl.join(tl.join(p0, p2), tl.join(p1, p3)), (BT, NGpc * Gc))
        qk = tl.where(mk[None, :], tl.dot(q, tl.trans(kg)), -float("inf"))
        mn = tl.maximum(mi, tl.max(qk, 1))
        p = tl.exp(qk - mn[:, None])
        a = tl.exp(mi - mn)
        li = li * a + tl.sum(p, 1)
        by = tl.load(pv_ptr + h * svbh + o[:, None] * (D // 4)
                     + tl.arange(0, D // 4)[None, :].to(tl.int64),
                     mask=mk[:, None], other=0).to(tl.uint32)
        svs = tl.load(sv_ptr + h * svmh + o, mask=mk, other=1.0).to(tl.float16)
        zvs = tl.load(zv_ptr + h * svmh + o, mask=mk, other=0.0).to(tl.float16)
        sh = (tl.arange(0, 4) * 2).to(tl.uint32)
        vi = tl.reshape((by[:, :, None] >> sh[None, None, :]) & 0x3, (BT, D)).to(tl.float16)
        v = (vi - zvs[:, None]) * svs[:, None]
        acc = acc * a[:, None] + tl.dot(p.to(tl.float16), v)
        mi = mn
    b = h * ns + s
    tl.store(m_ptr + b * BM + qm, mi)
    tl.store(l_ptr + b * BM + qm, li)
    tl.store(acc_ptr + b * BM * D + qm[:, None] * D + dd[None, :], acc)


def vq8_case(T, dev, n_heads=288, splits=32, BM=16, D=128, G=4, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    NG = D // G
    K = 1 << (2 * G)
    q = torch.randn(n_heads, BM, D, device=dev, generator=g)
    cb = torch.randn(n_heads, NG, K, G, device=dev, generator=g,
                     dtype=torch.float32).half()
    cb8 = cb.to(torch.float8_e5m2).view(torch.int32).squeeze(-1).contiguous()
    idx = torch.randint(0, K, (n_heads, T, NG), device=dev,
                        dtype=torch.int16, generator=g)
    pv = torch.randint(0, 256, (n_heads, T, D // 4), device=dev,
                       dtype=torch.uint8, generator=g)
    sv = (torch.rand(n_heads, T, device=dev, generator=g) + 0.5).half()
    zv = torch.rand(n_heads, T, device=dev, generator=g).half()
    ns = splits
    m = torch.zeros(n_heads * ns * BM, device=dev)
    l = torch.zeros(n_heads * ns * BM, device=dev)
    acc = torch.zeros(n_heads * ns * BM * D, device=dev)
    CHUNK = (T + ns - 1) // ns
    NGp = 1 << (NG - 1).bit_length()
    return dict(q=q, cb8=cb8, idx=idx, pv=pv, sv=sv, zv=zv, m=m, l=l, acc=acc,
                T=T, CHUNK=CHUNK, ns=ns, NG=NG, NGp=NGp, K=K, G=G, BM=BM, D=D,
                H=n_heads)


def run_vq8(c, BT=128, num_warps=4):
    _vq8_vint2[(c["H"], c["ns"])](
        c["q"], c["cb8"], c["idx"], c["pv"], c["sv"], c["zv"],
        c["m"], c["l"], c["acc"],
        c["T"], c["CHUNK"], c["ns"], c["T"] * c["NG"],
        c["T"] * (c["D"] // 4), c["T"],
        c["NG"], c["NGp"], c["K"], c["G"], c["BM"], BT, c["D"],
        num_warps=num_warps)


def combine(c):
    H, ns, BM, D = c["H"], c["ns"], c["BM"], c["D"]
    m = c["m"].view(H, ns, BM)
    l = c["l"].view(H, ns, BM)
    acc = c["acc"].view(H, ns, BM, D)
    mg = m.max(1, keepdim=True).values
    sc = torch.exp(m - mg)
    return (acc * sc.unsqueeze(-1)).sum(1) / (l * sc).sum(1).unsqueeze(-1)
