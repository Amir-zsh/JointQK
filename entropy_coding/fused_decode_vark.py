#!/usr/bin/env python3
"""VARIABLE-K G=4 fused decode+attention kernel — the modification the report flagged
as the gap for the accuracy-winning VQ (G4 stratified+waterfill, 0.747/0.964), which
uses a *different K per group* (waterfill of the Lambda budget) that the fixed-K int64
VEC kernel in fused_decode_all.py can't run.

The change vs the fixed-K VEC path is one line of addressing: fixed-K gathered a
codeword at `h*(NG*K) + g*K + idx`; here each (head, group) codebook lives in a FLAT
concatenated int64 buffer at a precomputed base `off[h, g]`, so the gather is
`off[h, g] + idx`. Everything else — the int64 wide-load (8-byte G=4 codeword = 1
load), fp16 unpack, tensor-core tl.dot, online softmax — is byte-for-byte the fixed-K
kernel. So variable-K decodes at the same speed, gated only by the codebook's L1
residency (strat_wf ~125 KB/head avg → L1-resident → fast).

Loads a real train_group_vq_alloc.py waterfill payload, builds the flat codebook +
offsets, correctness-gates the kernel against a torch reconstruct+attend reference,
and times full decode vs BF16.
"""
import argparse
import torch, triton, triton.language as tl

dev = "cuda"
d = 128
BLOCK_M = 16
BT = 128            # int64 gather amortizes best at BT=128 (as in the fixed-K VEC path)


@triton.jit
def _bf16(q_ptr, k_ptr, v_ptr, m_ptr, l_ptr, acc_ptr, sink_ptr, T, CHUNK, ns, skh,
          DEPTH: tl.constexpr, BM: tl.constexpr, BTc: tl.constexpr, D: tl.constexpr):
    h = tl.program_id(0).to(tl.int64); s = tl.program_id(1).to(tl.int64)
    qm = tl.arange(0, BM).to(tl.int64); dd = tl.arange(0, D).to(tl.int64)
    q = tl.load(q_ptr + h * BM * D + qm[:, None] * D + dd[None, :]).to(tl.float16)
    kb = k_ptr + h * skh; vb = v_ptr + h * skh
    lo = s * CHUNK; hi = tl.minimum(lo + CHUNK, T)
    mi = tl.zeros([BM], tl.float32) - float("inf"); li = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for t0 in range(lo, hi, BTc):
        o = t0 + tl.arange(0, BTc).to(tl.int64); mk = o < hi
        k = tl.load(kb + o[:, None] * D + dd[None, :], mask=mk[:, None], other=0.0).to(tl.float16)
        qk = tl.where(mk[None, :], tl.dot(q, tl.trans(k)), -float("inf"))
        mn = tl.maximum(mi, tl.max(qk, 1)); p = tl.exp(qk - mn[:, None]); a = tl.exp(mi - mn)
        li = li * a + tl.sum(p, 1)
        v = tl.load(vb + o[:, None] * D + dd[None, :], mask=mk[:, None], other=0.0).to(tl.float16)
        acc = acc * a[:, None] + tl.dot(p.to(tl.float16), v); mi = mn
    b = h * ns + s
    tl.store(m_ptr + b * BM + qm, mi); tl.store(l_ptr + b * BM + qm, li)
    tl.store(acc_ptr + b * BM * D + qm[:, None] * D + dd[None, :], acc)


@triton.jit
def _vq_vark(q_ptr, cbflat_ptr, off_ptr, idx_ptr, v_ptr, m_ptr, l_ptr, acc_ptr, sink_ptr,
             T, CHUNK, ns, svh, DEPTH: tl.constexpr, NGc: tl.constexpr, NGpc: tl.constexpr,
             Gc: tl.constexpr, Dkc: tl.constexpr, BM: tl.constexpr, BTc: tl.constexpr, D: tl.constexpr):
    # VARIABLE-K G=4 VEC gather: codeword address = off[h,g] + idx[t,g] into a flat int64
    # codebook buffer (vs fixed-K's h*NG*K + g*K + idx). Identical arithmetic otherwise.
    h = tl.program_id(0).to(tl.int64); s = tl.program_id(1).to(tl.int64)
    qm = tl.arange(0, BM).to(tl.int64); dd = tl.arange(0, D).to(tl.int64)
    gng = tl.arange(0, NGpc).to(tl.int64)
    q = tl.load(q_ptr + h * BM * D + qm[:, None] * D + dd[None, :]).to(tl.float16)
    off = tl.load(off_ptr + h * NGc + gng, mask=gng < NGc, other=0).to(tl.int64)   # (NGp,) per-group base
    ib = idx_ptr + h * (T * NGc); vb = v_ptr + h * svh
    lo = s * CHUNK; hi = tl.minimum(lo + CHUNK, T)
    mi = tl.zeros([BM], tl.float32) - float("inf"); li = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for t0 in range(lo, hi, BTc):
        o = t0 + tl.arange(0, BTc).to(tl.int64); mk = o < hi
        mg = mk[:, None] & (gng < NGc)[None, :]
        isel = tl.load(ib + o[:, None] * NGc + gng[None, :], mask=mg, other=0).to(tl.int64)
        cw = tl.load(cbflat_ptr + off[None, :] + isel, mask=mg, other=0).to(tl.int64)   # variable-K gather
        p0 = (cw & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True)
        p1 = ((cw >> 16) & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True)
        p2 = ((cw >> 32) & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True)
        p3 = ((cw >> 48) & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True)
        kg = tl.reshape(tl.join(tl.join(p0, p2), tl.join(p1, p3)), (BTc, NGpc * Gc))    # (BT,D)
        qk = tl.where(mk[None, :], tl.dot(q, tl.trans(kg)), -float("inf"))
        mn = tl.maximum(mi, tl.max(qk, 1)); p = tl.exp(qk - mn[:, None]); a = tl.exp(mi - mn)
        li = li * a + tl.sum(p, 1)
        v = tl.load(vb + o[:, None] * D + dd[None, :], mask=mk[:, None], other=0.0).to(tl.float16)
        acc = acc * a[:, None] + tl.dot(p.to(tl.float16), v); mi = mn
    b = h * ns + s
    tl.store(m_ptr + b * BM + qm, mi); tl.store(l_ptr + b * BM + qm, li)
    tl.store(acc_ptr + b * BM * D + qm[:, None] * D + dd[None, :], acc)


def combine(m, l, acc, H, ns):
    m = m.view(H, ns, BLOCK_M); l = l.view(H, ns, BLOCK_M); acc = acc.view(H, ns, BLOCK_M, d)
    mg = m.max(1, keepdim=True).values; sc = torch.exp(m - mg)
    return (acc * sc.unsqueeze(-1)).sum(1) / (l * sc).sum(1).unsqueeze(-1)


def timeit(f, reps=60, warmup=20):
    for _ in range(warmup): f()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps): f()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


def build_flat_codebook(payload):
    """Concatenate every (head, group) variable-K codebook into one flat int64 buffer;
    return (flat int64 (Ntot,), off int64 (H, NG), Kpg int64 (H, NG), NG, Dk)."""
    cbs = payload["codebooks"]; bounds = payload["bounds"]
    NG = len(bounds); G = bounds[0][1] - bounds[0][0]
    assert all((e - s) == 4 for s, e, _ in bounds), "variable-K VEC kernel is G=4 only"
    keys = sorted(cbs.keys())                       # (l, h) in order -> flat head index
    H = len(keys); Dk = NG * G
    flat, off, Kpg, cur = [], torch.zeros(H, NG, dtype=torch.int64), torch.zeros(H, NG, dtype=torch.int64), 0
    for hh, k in enumerate(keys):
        for g, c in enumerate(cbs[k]):
            cw = c.to(torch.float16).contiguous().view(torch.int64).reshape(-1)   # (K_g,)
            off[hh, g] = cur; Kpg[hh, g] = cw.numel(); cur += cw.numel()
            flat.append(cw)
    return torch.cat(flat).to(dev), off.to(dev), Kpg.to(dev), NG, Dk, H


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--codebook", default="vqa_G4_strat_wf.pt")
    ap.add_argument("--Ts", type=int, nargs="+", default=[16384, 65536])
    ap.add_argument("--n-splits", type=int, default=32)
    args = ap.parse_args()
    torch.manual_seed(0); ns = args.n_splits

    payload = torch.load(args.codebook, map_location="cpu", weights_only=False)
    flat, off, Kpg, NG, Dk, H = build_flat_codebook(payload)
    NGp = 1 << (NG - 1).bit_length()
    fp_kb = (Kpg.float().sum(1) * 4 * 2 / 1024)     # per-head codebook KB (K*G*2 bytes)
    print(f"{args.codebook}: H={H} NG={NG} Dk={Dk} | per-head codebook avg {fp_kb.mean():.0f}KB "
          f"max {fp_kb.max():.0f}KB (L1=192KB -> {'resident' if fp_kb.max()<=192 else 'avg-resident'})",
          flush=True)

    # ---- correctness (small): real codebook, random per-group indices, kernel vs torch ----
    Hc, Tc = 8, 4096; CH = (Tc + ns - 1) // ns
    q = torch.randn(Hc, BLOCK_M, d, device=dev); vf = torch.randn(Hc, Tc, d, device=dev).half().contiguous()
    idx = torch.zeros(Hc, Tc, NG, dtype=torch.int16, device=dev)
    for hh in range(Hc):
        for g in range(NG):
            idx[hh, :, g] = torch.randint(0, int(Kpg[hh, g]), (Tc,), device=dev, dtype=torch.int16)
    m = torch.zeros(Hc * ns * BLOCK_M, device=dev); l = torch.zeros(Hc * ns * BLOCK_M, device=dev)
    acc = torch.zeros(Hc * ns * BLOCK_M * d, device=dev); sink = torch.zeros(1, device=dev)
    _vq_vark[(Hc, ns)](q, flat, off, idx, vf, m, l, acc, sink, Tc, CH, ns, Tc * d, 2, NG, NGp, 4, Dk,
                       BLOCK_M, BT, d)
    o_k = combine(m, l, acc, Hc, ns)
    # torch reference: reconstruct kg from flat codebook, attend on first Dk coords
    flat_f16 = flat.view(torch.float16).view(-1, 4)                                  # (Ntot,4)
    krec = torch.empty(Hc, Tc, Dk, device=dev, dtype=torch.float32)
    for hh in range(Hc):
        for g in range(NG):
            cw = flat_f16[off[hh, g]: off[hh, g] + Kpg[hh, g]]                        # (K_g,4)
            krec[hh, :, g * 4:(g + 1) * 4] = cw[idx[hh, :, g].long()].float()
    qk = q[:, :, :Dk]
    s_ref = torch.softmax(torch.einsum("hmd,htd->hmt", qk, krec), -1)
    o_ref = torch.einsum("hmt,htd->hmd", s_ref, vf.float())
    err = (o_k - o_ref).abs().max().item()
    print(f"[correctness] variable-K VQ decode+attend vs torch: max_err={err:.2e} "
          f"{'PASS' if err < 5e-2 else 'FAIL'}", flush=True)

    # ---- timing (full model) vs BF16 ----
    for T in args.Ts:
        CHUNK = (T + ns - 1) // ns
        q = torch.randn(H, BLOCK_M, d, device=dev)
        kf = torch.randn(H, T, d, dtype=torch.float16, device=dev)
        vf = torch.randn(H, T, d, dtype=torch.float16, device=dev)
        # random valid per-group indices (values don't affect timing; shape/footprint do)
        idx = torch.randint(0, 32, (H, T, NG), device=dev, dtype=torch.int16)
        idx = torch.minimum(idx.long(), (Kpg[:, None, :] - 1).clamp_min(0)).to(torch.int16).contiguous()
        m = torch.zeros(H * ns * BLOCK_M, device=dev); l = torch.zeros(H * ns * BLOCK_M, device=dev)
        acc = torch.zeros(H * ns * BLOCK_M * d, device=dev); sink = torch.zeros(1, device=dev)
        tb = timeit(lambda: _bf16[(H, ns)](q, kf, vf, m, l, acc, sink, T, CHUNK, ns, T * d, 2, BLOCK_M, BT, d))
        tv = timeit(lambda: _vq_vark[(H, ns)](q, flat, off, idx, vf, m, l, acc, sink, T, CHUNK, ns, T * d, 2,
                                              NG, NGp, 4, Dk, BLOCK_M, BT, d))
        print(f"T={T}: BF16 full {tb:.3f}ms | VQ variable-K full {tv:.3f}ms | vs BF16 {tb/tv:.2f}x", flush=True)
