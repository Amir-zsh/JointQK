#!/usr/bin/env python3
"""Fused paged-dequant decode attention prototype (page_quant P5).

Demonstrates the serving path for the pgq_fixed codec: K cache stored as
fixed-byte pages (width ids @2 bits/token + power-of-2-width token rows),
dequantized INSIDE the attention kernel in the rotated code domain.

Key design decisions (systems-critic reviewed):
  - Rotated-domain scoring: q' = q @ inv^T (+ 1/sqrt(d)) once per (layer,
    q-head) per decode step; logits = r_hat . q'. The 128x128 inverse GEMM
    never runs in the kernel (it would cost more than the whole fp16
    attention read at 64k ctx). The q^T mu_k term is constant per head and
    cancels in softmax when every key in the head is compressed (mixed
    caches must add it — documented, not needed here).
  - Widths {0,1,2,4}: power-of-2 bit fields never cross u32 boundaries.
  - Page layout (fixed page_bytes = b_page*ptok*d/8):
      [16B width ids (64 x 2b)] [16B pad] [token rows, 16*w bytes each,
      token order] — every row 16B-aligned (d=128).
  - Split-K flash-decoding: grid (kv_heads, nsplits); partial (m, l, acc)
    reduced on the host. GQA group (4 q rows) padded to M=16 for tl.dot.

Microbench (A100, bf16 SDPA enable_gqa baseline — native KV heads, never
.expand()ed):
    python pipelines/page_quant/bench_fused_kernel.py --ctx 8192 32768 65536

Reports the three honest traffic numbers: both-fp16 baseline, K-compressed +
V-fp16 (the end-to-end ceiling ~1.8x), and K-only bytes (8-16x at b<=2).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import _bootstrap  # noqa: E402,F401

import argparse  # noqa: E402
import math  # noqa: E402

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

D = 128
PTOK = 64
WIDTHS = (0, 1, 2, 4)
ID_BITS = 2
HEADER_BYTES = 32              # 16B width ids + 16B pad
GQA = 4                        # q heads per kv head
M_PAD = 16                     # tl.dot needs M >= 16


# --------------------------------------------------------------------------
# Page packing (host side, torch) — the format the kernel decodes.
# --------------------------------------------------------------------------
def pack_pages(r_idx: torch.Tensor, widths_tok: torch.Tensor,
               page_bytes: int) -> torch.Tensor:
    """r_idx (T, D) int32 biased indices (idx + 2^(w-1)), widths_tok (T,)
    from WIDTHS. Returns (npages, page_bytes) uint8. Payload rows are
    16*w bytes; pages are zero-padded to page_bytes.

    Bit convention (must match the kernel): coord j occupies bits
    [j*w, (j+1)*w) of the row, little-endian within each byte."""
    T = r_idx.shape[0]
    npages = (T + PTOK - 1) // PTOK
    out = torch.zeros(npages, page_bytes, dtype=torch.uint8)

    # per-token packed rows (vectorized per width class)
    row_bytes_all = torch.zeros(T, 16 * 4, dtype=torch.uint8)
    for w in (1, 2, 4):
        msk = widths_tok == w
        if not msk.any():
            continue
        rows = (r_idx[msk].to(torch.int64) & ((1 << w) - 1))
        c = 8 // w                                    # coords per byte
        v = rows.view(-1, 16 * w, c)
        byte = torch.zeros(v.shape[0], 16 * w, dtype=torch.int64)
        for i in range(c):
            byte |= v[:, :, i] << (w * i)
        row_bytes_all[msk, : 16 * w] = byte.to(torch.uint8)

    wmap = {w: i for i, w in enumerate(WIDTHS)}
    wid_ids = torch.tensor([wmap[int(w)] for w in widths_tok.tolist()],
                           dtype=torch.int64)
    nb_tok = (16 * widths_tok).to(torch.int64)
    for p in range(npages):
        sl = slice(p * PTOK, min((p + 1) * PTOK, T))
        n = sl.stop - sl.start
        ids = torch.zeros(PTOK, dtype=torch.int64)
        ids[:n] = wid_ids[sl]
        idbytes = (ids.view(16, 4)
                   * (1 << (2 * torch.arange(4)))).sum(1)
        out[p, :16] = idbytes.to(torch.uint8)
        offs = HEADER_BYTES + torch.cumsum(nb_tok[sl], 0) - nb_tok[sl]
        for t in range(n):
            nb = int(nb_tok[sl.start + t])
            if nb:
                o = int(offs[t])
                out[p, o: o + nb] = row_bytes_all[sl.start + t, :nb]
        assert int(offs[-1] + nb_tok[sl.stop - 1]) <= page_bytes
    return out


def make_paged_k(T: int, b_page: float, device, seed: int = 0):
    """Synthetic head: gaussian codes quantized with a width mix that fills
    the page budget. Returns (pages u8, r_hat_ref (T, D) fp32, scales (W, D),
    page_bytes)."""
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(T, D, generator=g)
    std = r.std(0)
    alphas = torch.tensor([1.0, 1.6, 2.5, 4.0])
    scales = alphas.unsqueeze(1) * std.unsqueeze(0)       # (W, D)

    page_bits = int(b_page * D * PTOK)
    page_bytes = page_bits // 8
    payload_bits = page_bits - HEADER_BYTES * 8
    # deterministic width mix per page that fits the payload:
    # greedy from a repeating pattern scaled to the budget
    per_tok = payload_bits / PTOK / D                     # avg bits/coord
    base = [w for w in (2, 1) if w <= per_tok] or [1]
    widths_tok = torch.full((T,), base[0], dtype=torch.int64)
    # upgrade every 8th token to 4 bits while budget allows
    wt = widths_tok.view(-1)
    for p in range((T + PTOK - 1) // PTOK):
        sl = slice(p * PTOK, min((p + 1) * PTOK, T))
        used = int(wt[sl].sum()) * D
        i = sl.start
        while i < sl.stop:
            up = 4 - int(wt[i])
            if used + up * D <= payload_bits:
                used += up * D
                wt[i] = 4
            i += 8
        while used > payload_bits:
            # demote from the end
            for i in range(sl.stop - 1, sl.start - 1, -1):
                if wt[i] > 0:
                    used -= int(wt[i]) * D
                    wt[i] = 0 if wt[i] == 1 else wt[i] // 2
                    used += int(wt[i]) * D
                    if used <= payload_bits:
                        break

    r_idx = torch.zeros(T, D, dtype=torch.int32)
    r_hat = torch.zeros(T, D)
    for wi, w in enumerate(WIDTHS):
        msk = wt == w
        if w == 0 or not msk.any():
            continue
        s = scales[wi].unsqueeze(0)
        nlev = 1 << w
        idx = torch.floor(r[msk] / s).clamp(-(nlev // 2), nlev // 2 - 1)
        r_hat[msk] = ((idx + 0.5) * s).float()
        r_idx[msk] = (idx + nlev // 2).to(torch.int32)
    pages = pack_pages(r_idx, wt, page_bytes)
    return (pages.to(device), r_hat.to(device), scales.to(device),
            page_bytes, wt.to(device))


# --------------------------------------------------------------------------
# Triton kernel
# --------------------------------------------------------------------------
@triton.jit
def _fused_paged_decode(
    qr_ptr,            # (H, M_PAD, D) fp32   rotated+scaled queries
    pages_ptr,         # (H, NP, PB) uint8
    v_ptr,             # (H, T, D) fp16
    scales_ptr,        # (W, D) fp32
    pm_ptr, pl_ptr, pacc_ptr,   # partials: (H, S, M) , (H, S, M), (H, S, M, D)
    T, NP, PB, NSPLIT,
    H: tl.constexpr, M: tl.constexpr, DH: tl.constexpr,
    PT: tl.constexpr,
):
    h = tl.program_id(0)
    s = tl.program_id(1)
    pages_per_split = (NP + NSPLIT - 1) // NSPLIT
    p0 = s * pages_per_split
    p1 = tl.minimum(p0 + pages_per_split, NP)

    mrow = tl.arange(0, M)
    dcol = tl.arange(0, DH)
    trow = tl.arange(0, PT)

    q = tl.load(qr_ptr + h * M * DH + mrow[:, None] * DH + dcol[None, :])

    m_i = tl.full((M,), float("-inf"), tl.float32)
    l_i = tl.zeros((M,), tl.float32)
    acc = tl.zeros((M, DH), tl.float32)

    qh = q.to(tl.float16)
    strow = tl.arange(0, 16)
    for p in range(p0, p1):
        pbase = pages_ptr + h * NP * PB + p * PB
        # Sub-tile the page 16 tokens at a time: keeps the dequant block at
        # (16, DH) fp16 (register-budget fix — the (64, DH) fp32 version
        # spilled to local memory and ran 4x slower than SDPA).
        base_off = tl.zeros((), tl.int32) + 32           # skip header
        for st in tl.static_range(4):
            trs = st * 16 + strow
            idb = tl.load(pbase + trs // 4).to(tl.int32)
            wid = (idb >> (2 * (trs % 4))) & 3
            wbits = tl.where(wid == 0, 0,
                     tl.where(wid == 1, 1, tl.where(wid == 2, 2, 4)))
            row_bytes = 16 * wbits
            off = base_off + tl.cumsum(row_bytes, 0) - row_bytes
            r_hat = tl.zeros((16, DH), tl.float16)
            for wi_c in tl.static_range(1, 4):
                w = 1 << (wi_c - 1)      # widths 1, 2, 4 for ids 1, 2, 3
                is_w = wid == wi_c
                bitpos = dcol[None, :] * w
                addr = pbase + off[:, None] + bitpos // 8
                byte = tl.load(addr, mask=is_w[:, None], other=0).to(tl.int32)
                val = (byte >> (bitpos % 8)) & ((1 << w) - 1)
                sc = tl.load(scales_ptr + wi_c * DH + dcol)
                dq = (val.to(tl.float32) - (1 << (w - 1)) + 0.5) * sc[None, :]
                r_hat = tl.where(is_w[:, None], dq.to(tl.float16), r_hat)
            logits = tl.dot(qh, tl.trans(r_hat)).to(tl.float32)  # (M, 16)
            tok = p * PT + trs
            logits = tl.where((tok < T)[None, :], logits, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(logits, 1))
            alpha = tl.exp(m_i - m_new)
            pexp = tl.exp(logits - m_new[:, None])
            l_i = l_i * alpha + tl.sum(pexp, 1)
            vblk = tl.load(v_ptr + h * T * DH + tok[:, None] * DH
                           + dcol[None, :],
                           mask=(tok < T)[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(pexp.to(tl.float16),
                                                vblk.to(tl.float16))
            m_i = m_new
            base_off += tl.sum(row_bytes, 0)

    base2 = h * NSPLIT * M + s * M
    tl.store(pm_ptr + base2 + mrow, m_i)
    tl.store(pl_ptr + base2 + mrow, l_i)
    tl.store(pacc_ptr + (base2 + mrow)[:, None] * DH + dcol[None, :], acc)


def fused_paged_attention(q_rot, pages, v, scales, T, nsplit=None):
    """q_rot (H, M_PAD, D) fp32 (already inv-rotated + 1/sqrt(d) scaled);
    pages (H, NP, PB) u8; v (H, T, D) fp16. Returns (H, M_PAD, D) fp32."""
    H, NP, PB = pages.shape
    if nsplit is None:
        nsplit = max(4, min(64, NP // 8))
    dev = q_rot.device
    pm = torch.empty(H, nsplit, M_PAD, device=dev, dtype=torch.float32)
    pl = torch.empty_like(pm)
    pacc = torch.empty(H, nsplit, M_PAD, D, device=dev, dtype=torch.float32)
    _fused_paged_decode[(H, nsplit)](
        q_rot, pages, v, scales, pm, pl, pacc,
        T, NP, PB, nsplit, H=H, M=M_PAD, DH=D, PT=PTOK,
        num_warps=8, num_stages=2)
    m = pm.max(1, keepdim=True).values                     # (H,1,M)
    w = torch.exp(pm - m) * pl
    denom = w.sum(1)                                       # (H,M)
    out = (pacc * torch.exp(pm - m).unsqueeze(-1)).sum(1) / \
        denom.clamp_min(1e-30).unsqueeze(-1)
    return out


# --------------------------------------------------------------------------
# Bench
# --------------------------------------------------------------------------
def bench_one(ctx, b_page, device, iters=50):
    H = 8
    g = torch.Generator(device="cpu").manual_seed(1)
    q = torch.randn(1, H * GQA, 1, D, generator=g).to(device)

    pages_h, rhat_h, v_h = [], [], []
    scales = None
    for h in range(H):
        pages, r_hat, scales, PB, _ = make_paged_k(ctx, b_page, device,
                                                   seed=h)
        pages_h.append(pages)
        rhat_h.append(r_hat)
    pages = torch.stack(pages_h)                           # (H, NP, PB)
    r_hat = torch.stack(rhat_h)                            # (H, T, D)
    v = (torch.randn(H, ctx, D, generator=g) * 0.05).to(device).half()

    # rotated-domain queries: identity rotation for the synthetic bench
    # (real path: q' = q @ inv.T per (layer, q-head)); scale folded in.
    qg = q.view(H, GQA, D).float() / math.sqrt(D)
    q_rot = torch.zeros(H, M_PAD, D, device=device)
    q_rot[:, :GQA] = qg

    out = fused_paged_attention(q_rot, pages, v, scales, ctx)
    # reference: same r_hat through fp32 attention
    lg = torch.einsum("hmd,htd->hmt", q_rot[:, :GQA], r_hat)
    p = torch.softmax(lg, dim=-1)
    ref = torch.einsum("hmt,htd->hmd", p, v.float())
    err = (out[:, :GQA] - ref).abs().max() / ref.abs().max()

    # timing: fused kernel
    torch.cuda.synchronize()
    t_fused = triton.testing.do_bench(
        lambda: fused_paged_attention(q_rot, pages, v, scales, ctx),
        warmup=10, rep=iters)

    # baseline: bf16 SDPA enable_gqa, native KV heads
    qs = q.to(torch.bfloat16)
    k_sd = r_hat.unsqueeze(0).to(torch.bfloat16)           # (1,H,T,D)
    v_sd = v.unsqueeze(0).to(torch.bfloat16)
    def sdpa():
        return torch.nn.functional.scaled_dot_product_attention(
            qs.view(1, H * GQA, 1, D), k_sd, v_sd, enable_gqa=True)
    t_sdpa = triton.testing.do_bench(sdpa, warmup=10, rep=iters)

    # dequant-then-SDPA two-step (what a non-fused compressed path costs)
    def two_step():
        kk = r_hat.to(torch.bfloat16).unsqueeze(0)
        return torch.nn.functional.scaled_dot_product_attention(
            qs.view(1, H * GQA, 1, D), kk, v_sd, enable_gqa=True)
    t_two = triton.testing.do_bench(two_step, warmup=10, rep=iters)

    bytes_fp16_kv = 2 * H * ctx * D * 2
    bytes_k_comp = int(pages.numel())
    bytes_v_fp16 = H * ctx * D * 2
    return {
        "ctx": ctx, "b_page": b_page, "rel_err": float(err),
        "t_fused_ms": t_fused, "t_sdpa_bf16_ms": t_sdpa,
        "t_dequant_sdpa_ms": t_two,
        "bytes_fp16_kv_MB": bytes_fp16_kv / 1e6,
        "bytes_Kcomp_plus_Vfp16_MB": (bytes_k_comp + bytes_v_fp16) / 1e6,
        "bytes_Konly_ratio": (H * ctx * D * 2) / bytes_k_comp,
        "traffic_ratio_end2end":
            bytes_fp16_kv / (bytes_k_comp + bytes_v_fp16),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, nargs="+",
                    default=[8192, 32768, 65536])
    ap.add_argument("--b-page", type=float, default=2.0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.cuda.set_device(args.device)
    rows = []
    for ctx in args.ctx:
        r = bench_one(ctx, args.b_page, args.device)
        rows.append(r)
        print(f"ctx={ctx:6d} b={args.b_page:g}  rel_err={r['rel_err']:.2e}  "
              f"fused={r['t_fused_ms']:.3f}ms  sdpa_bf16={r['t_sdpa_bf16_ms']:.3f}ms  "
              f"speedup={r['t_sdpa_bf16_ms'] / r['t_fused_ms']:.2f}x  "
              f"traffic: e2e {r['traffic_ratio_end2end']:.2f}x, "
              f"K-only {r['bytes_Konly_ratio']:.1f}x", flush=True)
    import json
    outp = REPO / "artifacts/page_quant/kernel_bench.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(rows, indent=2))
    print(f"-> {outp}")


if __name__ == "__main__":
    main()
