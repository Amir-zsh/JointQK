#!/usr/bin/env python3
"""REAL fused decode+attention, per method, three depths, FULL-MODEL scale, with
TENSOR-CORE qk/pv (tl.dot) so the attention math is hidden under the K read the way
a production kernel keeps it -- decode attention is memory-bound (arithmetic
intensity ~0.5 madd/byte), so a faithful kernel must not let a hand-rolled GEMV
mask the bandwidth win. (An earlier version used elementwise sum for qk and
wrongly showed the win washing out at depth>=1; that was the kernel, not the method.)

Three depths (DEPTH constexpr): compressed K read from HBM, reconstructed in
registers, NEVER written back full-precision:
  0 READ    : read compressed K bytes, checksum.                       (bandwidth floor)
  1 READ+QK : + reconstruct + scores = tl.dot(q_rot, k_code^T).        (tensor core)
  2 FULL    : + online softmax + acc = tl.dot(p, V).                   (full decode)

Inverse rotation fused into the query (q_rot precomputed once), keys stay coded:
scores = q_rot . k_code == q . k. GQA: BLOCK_M queries share each kv-head's K read.

Parallelism: grid=(n_heads, n_splits) split-K, saturating the SMs. Depth 2 writes
per-(head,split) softmax partials, combined in torch and checked vs a torch SDPA
reference (small-T correctness pass, separate from the large-T timing pass so the
fp32 references don't OOM).

Methods: BF16, INT2 (VQ/OSCAR added once these pass). rANS/Exp-Golomb are
compute-bound serial decode, reported separately.
"""
import torch, triton, triton.language as tl

dev = torch.device("cuda")
d = 128
BLOCK_M = 16
BLOCK_T = 64


@triton.jit
def _bf16_kernel(q_ptr, k_ptr, v_ptr, m_ptr, l_ptr, acc_ptr, sink_ptr,
                 T, CHUNK, n_splits, stride_kh, DEPTH: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_T: tl.constexpr, D: tl.constexpr):
    h = tl.program_id(0).to(tl.int64); s = tl.program_id(1).to(tl.int64)
    qm = tl.arange(0, BLOCK_M).to(tl.int64); dd = tl.arange(0, D).to(tl.int64)
    q = tl.load(q_ptr + h * BLOCK_M * D + qm[:, None] * D + dd[None, :]).to(tl.float16)  # (M,D)
    kbase = k_ptr + h * stride_kh; vbase = v_ptr + h * stride_kh
    t_lo = s * CHUNK; t_hi = tl.minimum(t_lo + CHUNK, T)
    m_i = tl.zeros([BLOCK_M], tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], tl.float32); acc = tl.zeros([BLOCK_M, D], tl.float32); chk = 0.0
    for t0 in range(t_lo, t_hi, BLOCK_T):
        offs = t0 + tl.arange(0, BLOCK_T).to(tl.int64); mask = offs < t_hi
        k = tl.load(kbase + offs[:, None] * D + dd[None, :], mask=mask[:, None], other=0.0).to(tl.float16)  # (T,D)
        if DEPTH == 0:
            chk += tl.sum(tl.where(mask[:, None], k.to(tl.float32), 0.0))
        else:
            qk = tl.dot(q, tl.trans(k))                              # (M,T) fp32
            qk = tl.where(mask[None, :], qk, -float("inf"))
            if DEPTH == 1:
                chk += tl.sum(tl.where(mask[None, :], qk, 0.0))
            else:
                m_new = tl.maximum(m_i, tl.max(qk, axis=1))
                p = tl.exp(qk - m_new[:, None]); a = tl.exp(m_i - m_new)
                l_i = l_i * a + tl.sum(p, axis=1)
                v = tl.load(vbase + offs[:, None] * D + dd[None, :], mask=mask[:, None], other=0.0).to(tl.float16)
                acc = acc * a[:, None] + tl.dot(p.to(tl.float16), v)
                m_i = m_new
    if DEPTH == 2:
        b = h * n_splits + s
        tl.store(m_ptr + b * BLOCK_M + qm, m_i); tl.store(l_ptr + b * BLOCK_M + qm, l_i)
        tl.store(acc_ptr + b * BLOCK_M * D + qm[:, None] * D + dd[None, :], acc)
    else:
        tl.atomic_add(sink_ptr, chk)


@triton.jit
def _int2_kernel(q_ptr, packed_ptr, step_ptr, base_ptr, v_ptr, m_ptr, l_ptr, acc_ptr, sink_ptr,
                 T, CHUNK, n_splits, stride_ph, stride_vh, DEPTH: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_T: tl.constexpr, D: tl.constexpr):
    # packed: (n_heads, T, D//4) uint8 contiguous-4. Affine dequant coded value =
    # q_idx*step[coord] + base[coord]. step/base: (n_heads, D).
    h = tl.program_id(0).to(tl.int64); s = tl.program_id(1).to(tl.int64)
    qm = tl.arange(0, BLOCK_M).to(tl.int64); dd = tl.arange(0, D).to(tl.int64)
    q = tl.load(q_ptr + h * BLOCK_M * D + qm[:, None] * D + dd[None, :]).to(tl.float16)
    step = tl.load(step_ptr + h * D + dd).to(tl.float32)
    base = tl.load(base_ptr + h * D + dd).to(tl.float32)
    pbase = packed_ptr + h * stride_ph; vbase = v_ptr + h * stride_vh
    t_lo = s * CHUNK; t_hi = tl.minimum(t_lo + CHUNK, T)
    m_i = tl.zeros([BLOCK_M], tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], tl.float32); acc = tl.zeros([BLOCK_M, D], tl.float32); chk = 0.0
    for t0 in range(t_lo, t_hi, BLOCK_T):
        offs = t0 + tl.arange(0, BLOCK_T).to(tl.int64); mask = offs < t_hi
        bytes_ = tl.load(pbase + offs[:, None] * (D // 4) + tl.arange(0, D // 4)[None, :].to(tl.int64),
                         mask=mask[:, None], other=0).to(tl.uint32)          # (T, D//4) coalesced
        shifts = (tl.arange(0, 4) * 2).to(tl.uint32)
        b4 = (bytes_[:, :, None] >> shifts[None, None, :]) & 0x3             # (T, D//4, 4)
        q_idx = tl.reshape(b4, (BLOCK_T, D)).to(tl.float32)
        k = (q_idx * step[None, :] + base[None, :]).to(tl.float16)          # (T,D) affine dequant
        if DEPTH == 0:
            chk += tl.sum(tl.where(mask[:, None], bytes_.to(tl.float32), 0.0))
        else:
            qk = tl.dot(q, tl.trans(k))
            qk = tl.where(mask[None, :], qk, -float("inf"))
            if DEPTH == 1:
                chk += tl.sum(tl.where(mask[None, :], qk, 0.0))
            else:
                m_new = tl.maximum(m_i, tl.max(qk, axis=1))
                p = tl.exp(qk - m_new[:, None]); a = tl.exp(m_i - m_new)
                l_i = l_i * a + tl.sum(p, axis=1)
                v = tl.load(vbase + offs[:, None] * D + dd[None, :], mask=mask[:, None], other=0.0).to(tl.float16)
                acc = acc * a[:, None] + tl.dot(p.to(tl.float16), v)
                m_i = m_new
    if DEPTH == 2:
        b = h * n_splits + s
        tl.store(m_ptr + b * BLOCK_M + qm, m_i); tl.store(l_ptr + b * BLOCK_M + qm, l_i)
        tl.store(acc_ptr + b * BLOCK_M * D + qm[:, None] * D + dd[None, :], acc)
    else:
        tl.atomic_add(sink_ptr, chk)


def combine(m, l, acc, H, ns):
    m = m.view(H, ns, BLOCK_M); l = l.view(H, ns, BLOCK_M); acc = acc.view(H, ns, BLOCK_M, d)
    m_g = m.max(1, keepdim=True).values
    sc = torch.exp(m - m_g)
    l_g = (l * sc).sum(1); acc_g = (acc * sc.unsqueeze(-1)).sum(1)
    return acc_g / l_g.unsqueeze(-1)                              # (H, BLOCK_M, d)


def timeit(f, reps=30, warmup=10):
    for _ in range(warmup): f()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(reps): f()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


def make_int2(k_code, H, T):
    std = k_code.std(1)                                          # (H,d)
    step = std.contiguous(); base = (-1.5 * std).contiguous()
    qidx = ((k_code / std.unsqueeze(1) + 1.5).round().clamp(0, 3)).to(torch.uint8)
    qi = qidx.view(H, T, d // 4, 4)
    packed = (qi[..., 0] | (qi[..., 1] << 2) | (qi[..., 2] << 4) | (qi[..., 3] << 6)).to(torch.uint8).contiguous()
    return step, base, packed, qidx


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-heads", type=int, default=288)
    ap.add_argument("--Ts", type=int, nargs="+", default=[16384, 65536, 100000])
    ap.add_argument("--n-splits", type=int, default=32)
    args = ap.parse_args()
    torch.manual_seed(0); H = args.n_heads; ns = args.n_splits

    # ---------- correctness pass: small T, small H, full fp32 references ----------
    Hc, Tc = 8, 4096; CH = (Tc + ns - 1) // ns
    q = torch.randn(Hc, BLOCK_M, d, device=dev)
    k_code = torch.randn(Hc, Tc, d, device=dev); v = torch.randn(Hc, Tc, d, device=dev)
    step, base, packed, qidx = make_int2(k_code, Hc, Tc)
    k_deq = qidx.float() * step.unsqueeze(1) + base.unsqueeze(1)
    m = torch.zeros(Hc * ns * BLOCK_M, device=dev); l = torch.zeros(Hc * ns * BLOCK_M, device=dev)
    acc = torch.zeros(Hc * ns * BLOCK_M * d, device=dev); sink = torch.zeros(1, device=dev)

    _bf16_kernel[(Hc, ns)](q, k_code.half().contiguous(), v.half().contiguous(), m, l, acc, sink,
                           Tc, CH, ns, Tc * d, 2, BLOCK_M, BLOCK_T, d)
    o = combine(m, l, acc, Hc, ns)
    sref = torch.softmax(torch.einsum("hmd,htd->hmt", q, k_code), -1)
    oref = torch.einsum("hmt,htd->hmd", sref, v)
    print(f"[correctness] BF16 err={ (o-oref).abs().max().item():.2e}", flush=True)

    _int2_kernel[(Hc, ns)](q, packed, step, base, v.half().contiguous(), m, l, acc, sink,
                           Tc, CH, ns, Tc * (d // 4), Tc * d, 2, BLOCK_M, BLOCK_T, d)
    o2 = combine(m, l, acc, Hc, ns)
    s2 = torch.softmax(torch.einsum("hmd,htd->hmt", q, k_deq), -1)
    o2ref = torch.einsum("hmt,htd->hmd", s2, v)
    print(f"[correctness] INT2 err={ (o2-o2ref).abs().max().item():.2e}", flush=True)
    del k_code, v, k_deq, packed, qidx, step, base, m, l, acc; torch.cuda.empty_cache()

    # ---------- timing pass: full H, large T, buffers only (no fp32 refs) ----------
    for T in args.Ts:
        CHUNK = (T + ns - 1) // ns
        q = torch.randn(H, BLOCK_M, d, device=dev)
        k_code = torch.randn(H, T, d, device=dev)
        vh = torch.randn(H, T, d, device=dev).half().contiguous()
        step, base, packed, _ = make_int2(k_code, H, T)
        kf16 = k_code.half().contiguous(); del k_code; torch.cuda.empty_cache()
        m = torch.zeros(H * ns * BLOCK_M, device=dev); l = torch.zeros(H * ns * BLOCK_M, device=dev)
        acc = torch.zeros(H * ns * BLOCK_M * d, device=dev); sink = torch.zeros(1, device=dev)

        print(f"\n{'method':>8} {'read':>9} {'read+qk':>9} {'full':>9}   (ms, {H} heads x {BLOCK_M} q, T={T})", flush=True)
        for name in ("BF16", "INT2"):
            times = []
            for dep in (0, 1, 2):
                if name == "BF16":
                    f = lambda: _bf16_kernel[(H, ns)](q, kf16, vh, m, l, acc, sink, T, CHUNK, ns, T*d, dep, BLOCK_M, BLOCK_T, d)
                else:
                    f = lambda: _int2_kernel[(H, ns)](q, packed, step, base, vh, m, l, acc, sink, T, CHUNK, ns, T*(d//4), T*d, dep, BLOCK_M, BLOCK_T, d)
                times.append(timeit(f))
            print(f"{name:>8} {times[0]:>9.4f} {times[1]:>9.4f} {times[2]:>9.4f}", flush=True)
        del q, kf16, vh, packed, step, base, m, l, acc; torch.cuda.empty_cache()
