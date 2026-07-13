"""pgq7 K2 stage-1 (v0, correctness-first): fused unpack + dequant + logits.

One Triton program per (page, q-head group). Per row of the page it
extracts the packed codes (little-endian, width from the row's rung and the
block-width table), dequantizes as LUT[code] * sigma[row_in_page, coord],
dots against the folded queries q~ = q G^T, and — for DCT pages — applies
the page-local inverse transform on the logit tile (z = u @ D). Page 0
(sinks), partial tails, and fp16-tier pages are served by the torch path.

v0 is generic over rungs (runtime width tables, gather loads). The
perf-shaped variant (rung-segmented launches with constexpr widths, the
OSCAR-style split-K online softmax and V accumulation) builds on this once
parity is pinned; this file's contract is exactness vs kvq.kernels.golden.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _page_logits_kernel(
    payload_ptr, row_ptr_ptr, row_rung_ptr,
    bw_ptr, lut_ptr, lut_off_ptr,
    sigma_id_ptr, sigma_dct_ptr, dct_ptr,
    pages_ptr, kind_ptr, qt_ptr, out_ptr,
    T, HG: tl.constexpr, D: tl.constexpr, PTOK: tl.constexpr,
    NBLK: tl.constexpr, CBLK: tl.constexpr,
):
    pid = tl.program_id(0)
    page = tl.load(pages_ptr + pid)
    kind = tl.load(kind_ptr + pid)
    t0 = page * PTOK

    hg = tl.arange(0, HG)
    ci = tl.arange(0, CBLK)                                    # 32 coords
    u = tl.zeros((HG, PTOK), dtype=tl.float32)

    for s in range(PTOK):
        t = t0 + s
        rung = tl.load(row_rung_ptr + t).to(tl.int32)
        base = tl.load(row_ptr_ptr + t)
        # dequantized row assembled block by block
        acc = tl.zeros((HG,), dtype=tl.float32)
        boff = 0
        for b in tl.static_range(NBLK):
            w = tl.load(bw_ptr + rung * NBLK + b)
            # bit positions of the 32 codes of this block
            bitpos = ci * w
            byte0 = tl.load(payload_ptr + base + boff + (bitpos >> 3),
                            mask=w > 0, other=0).to(tl.int32)
            byte1 = tl.load(payload_ptr + base + boff + (bitpos >> 3) + 1,
                            mask=w > 0, other=0).to(tl.int32)
            code = ((byte0 | (byte1 << 8)) >> (bitpos & 7)) & ((1 << w) - 1)
            lo = tl.load(lut_off_ptr + w, mask=w > 0, other=0)
            lev = tl.load(lut_ptr + lo + code, mask=w > 0, other=0.0)
            coord = b * CBLK + ci
            sid = tl.load(sigma_id_ptr + s * D + coord)
            sdc = tl.load(sigma_dct_ptr + s * D + coord)
            sig = tl.where(kind == 2, sdc, sid)
            deq = tl.where(w > 0, lev * sig, 0.0)              # (CBLK,)
            qslice = tl.load(qt_ptr + hg[:, None] * D + coord[None, :])
            acc += tl.sum(qslice * deq[None, :], axis=1)
            boff += (w * CBLK) >> 3
        u = tl.where(tl.arange(0, PTOK)[None, :] == s, acc[:, None], u)

    if kind == 2:
        # z = u @ D  (page-local inverse transform on the logit tile)
        srow = tl.arange(0, PTOK)
        z = tl.zeros((HG, PTOK), dtype=tl.float32)
        for s in range(PTOK):
            dcol = tl.load(dct_ptr + s * PTOK + srow)          # D[s, :]
            us = tl.sum(tl.where(srow[None, :] == s, u, 0.0), axis=1)
            z += us[:, None] * dcol[None, :]
        u = z

    tt = tl.arange(0, PTOK)
    tl.store(out_ptr + hg[:, None] * T + t0 + tt[None, :], u)


@triton.jit
def _page_attn_kernel(
    payload_ptr, row_ptr_ptr, row_rung_ptr,
    bw_ptr, lut_ptr, lut_off_ptr,
    sigma_id_ptr, sigma_dct_ptr, dct_ptr,
    pages_ptr, kind_ptr, qt_ptr, v_ptr,
    m_out_ptr, l_out_ptr, acc_out_ptr,
    NPAGES, PPS, sm_scale,
    T, HG: tl.constexpr, D: tl.constexpr, PTOK: tl.constexpr,
    NBLK: tl.constexpr, CBLK: tl.constexpr,
):
    """Split-K stage 1: each program owns PPS consecutive entries of the
    page list and emits online-softmax partials (m, l, acc)."""
    split = tl.program_id(0)
    hg = tl.arange(0, HG)
    dj = tl.arange(0, D)
    ci = tl.arange(0, CBLK)
    tt = tl.arange(0, PTOK)

    m = tl.full((HG,), float("-inf"), dtype=tl.float32)
    l = tl.zeros((HG,), dtype=tl.float32)
    acc = tl.zeros((HG, D), dtype=tl.float32)

    for pi in range(split * PPS, tl.minimum((split + 1) * PPS, NPAGES)):
        page = tl.load(pages_ptr + pi)
        kind = tl.load(kind_ptr + pi)
        t0 = page * PTOK
        u = tl.zeros((HG, PTOK), dtype=tl.float32)
        for s in range(PTOK):
            t = t0 + s
            rung = tl.load(row_rung_ptr + t).to(tl.int32)
            base = tl.load(row_ptr_ptr + t)
            arow = tl.zeros((HG,), dtype=tl.float32)
            boff = 0
            for b in tl.static_range(NBLK):
                w = tl.load(bw_ptr + rung * NBLK + b)
                bitpos = ci * w
                byte0 = tl.load(payload_ptr + base + boff + (bitpos >> 3),
                                mask=w > 0, other=0).to(tl.int32)
                byte1 = tl.load(payload_ptr + base + boff + (bitpos >> 3) + 1,
                                mask=w > 0, other=0).to(tl.int32)
                code = ((byte0 | (byte1 << 8)) >> (bitpos & 7)) \
                    & ((1 << w) - 1)
                lo = tl.load(lut_off_ptr + w, mask=w > 0, other=0)
                lev = tl.load(lut_ptr + lo + code, mask=w > 0, other=0.0)
                coord = b * CBLK + ci
                sid = tl.load(sigma_id_ptr + s * D + coord)
                sdc = tl.load(sigma_dct_ptr + s * D + coord)
                sig = tl.where(kind == 2, sdc, sid)
                deq = tl.where(w > 0, lev * sig, 0.0)
                qslice = tl.load(qt_ptr + hg[:, None] * D + coord[None, :])
                arow += tl.sum(qslice * deq[None, :], axis=1)
                boff += (w * CBLK) >> 3
            u = tl.where(tt[None, :] == s, arow[:, None], u)
        if kind == 2:
            srow = tl.arange(0, PTOK)
            z = tl.zeros((HG, PTOK), dtype=tl.float32)
            for s in range(PTOK):
                dcol = tl.load(dct_ptr + s * PTOK + srow)
                us = tl.sum(tl.where(srow[None, :] == s, u, 0.0), axis=1)
                z += us[:, None] * dcol[None, :]
            u = z
        z = u * sm_scale
        m_new = tl.maximum(m, tl.max(z, axis=1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(z - m_new[:, None])                       # (HG, PTOK)
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        for s in range(PTOK):
            vrow = tl.load(v_ptr + (t0 + s) * D + dj).to(tl.float32)
            ps = tl.sum(tl.where(tt[None, :] == s, p, 0.0), axis=1)
            acc += ps[:, None] * vrow[None, :]
        m = m_new

    tl.store(m_out_ptr + split * HG + hg, m)
    tl.store(l_out_ptr + split * HG + hg, l)
    tl.store(acc_out_ptr + split * HG * D + hg[:, None] * D + dj[None, :],
             acc)


@triton.jit
def _page_attn_kernel_v1(
    payload_ptr, pay_off_ptr, row_ptr_ptr, row_rung_ptr,
    bw_ptr, lut_ptr, lut_off_ptr, LUTN: tl.constexpr,
    sigma_id_ptr, sigma_dct_ptr, dct_ptr,
    pages_ptr, kind_ptr, qt_ptr, v_ptr,
    m_out_ptr, l_out_ptr, acc_out_ptr,
    NPAGES, PPS, sm_scale,
    T, HGP: tl.constexpr, D: tl.constexpr, PTOK: tl.constexpr,
    NBLK: tl.constexpr, CBLK: tl.constexpr,
):
    """K2c perf shape: rows-as-lanes tiles + tensor-core dots, one program
    per (page-range split, kv head)."""
    split = tl.program_id(0)
    h = tl.program_id(1)
    rows = tl.arange(0, PTOK)
    ci = tl.arange(0, CBLK)
    hg = tl.arange(0, HGP)
    dj = tl.arange(0, D)
    pbase = tl.load(pay_off_ptr + h)

    m = tl.full((HGP,), float("-inf"), dtype=tl.float32)
    l = tl.zeros((HGP,), dtype=tl.float32)
    acc = tl.zeros((HGP, D), dtype=tl.float32)

    for pi in range(split * PPS, tl.minimum((split + 1) * PPS, NPAGES)):
        page = tl.load(pages_ptr + pi)
        kind = tl.load(kind_ptr + pi)
        t0 = page * PTOK
        base = tl.load(row_ptr_ptr + h * T + t0 + rows)        # (PTOK,)
        rung = tl.load(row_rung_ptr + h * T + t0 + rows).to(tl.int32)
        u = tl.zeros((HGP, PTOK), dtype=tl.float32)
        boff = tl.zeros((PTOK,), dtype=tl.int32)
        for b in tl.static_range(NBLK):
            w = tl.load(bw_ptr + rung * NBLK + b)              # (PTOK,)
            bitpos = w[:, None] * ci[None, :]                  # (PTOK, CBLK)
            bidx = pbase + base[:, None] + boff[:, None] + (bitpos >> 3)
            live = (w > 0)[:, None]
            b0 = tl.load(payload_ptr + bidx, mask=live, other=0).to(tl.int32)
            b1 = tl.load(payload_ptr + bidx + 1, mask=live, other=0).to(tl.int32)
            code = ((b0 | (b1 << 8)) >> (bitpos & 7)) \
                & ((1 << w[:, None]) - 1)
            lo = tl.load(lut_off_ptr + w)                      # (PTOK,)
            lev = tl.load(lut_ptr + h * LUTN + lo[:, None] + code,
                          mask=live, other=0.0)
            coord = b * CBLK + ci
            sid = tl.load(sigma_id_ptr + h * PTOK * D
                          + rows[:, None] * D + coord[None, :])
            sdc = tl.load(sigma_dct_ptr + h * PTOK * D
                          + rows[:, None] * D + coord[None, :])
            sig = tl.where(kind == 2, sdc, sid)
            deq = tl.where(live, lev * sig, 0.0).to(tl.float16)
            qb = tl.load(qt_ptr + h * HGP * D + hg[:, None] * D
                         + coord[None, :]).to(tl.float16)      # (HGP, CBLK)
            u += tl.dot(qb, tl.trans(deq))                     # (HGP, PTOK)
            boff += (w * CBLK) >> 3
        if kind == 2:
            dt = tl.load(dct_ptr + rows[:, None] * PTOK
                         + rows[None, :]).to(tl.float16)       # (PTOK, PTOK)
            u = tl.dot(u.to(tl.float16), dt)
        z = u * sm_scale
        m_new = tl.maximum(m, tl.max(z, axis=1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(z - m_new[:, None])
        l = l * alpha + tl.sum(p, axis=1)
        vt = tl.load(v_ptr + h * T * D + (t0 + rows)[:, None] * D
                     + dj[None, :])                            # (PTOK, D) f16
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), vt)
        m = m_new

    S = tl.num_programs(0)
    tl.store(m_out_ptr + (h * S + split) * HGP + hg, m)
    tl.store(l_out_ptr + (h * S + split) * HGP + hg, l)
    tl.store(acc_out_ptr + ((h * S + split) * HGP + hg[:, None]) * D
             + dj[None, :], acc)


@triton.jit
def _seg_u_kernel(
    pay_ptr, tok_ptr, pay_off_ptr, tok_off_ptr, n_ptr,
    stride_ptr, w_ptr, lo_ptr,
    lut_ptr, LUTN: tl.constexpr,
    sig_ptr, kind_ptr, qt_ptr, u_ptr,
    T, HG: tl.constexpr, HGP: tl.constexpr, D: tl.constexpr,
    PTOK: tl.constexpr, CBLK: tl.constexpr,
):
    """Phase A for one rung: coalesced (64-row) payload tiles -> dequant ->
    q-dot -> scatter u[h, hg, t]. Widths / stride / LUT bases are loaded
    per (rung, head) as program-uniform scalars (Qwen shares profiles per
    HEAD, so heads carry different width rows for the same rung id);
    coalescing is unaffected because head is a grid axis."""
    tile = tl.program_id(0)
    h = tl.program_id(1)
    n = tl.load(n_ptr + h)
    rows = tile * PTOK + tl.arange(0, PTOK)
    live_r = rows < n
    stride = tl.load(stride_ptr + h).to(tl.int64)
    pbase = tl.load(pay_off_ptr + h)
    tbase = tl.load(tok_off_ptr + h)
    tok = tl.load(tok_ptr + tbase + rows, mask=live_r, other=0)
    s_in_page = tok % PTOK
    kind = tl.load(kind_ptr + tok // PTOK, mask=live_r, other=1)
    # plan9 S2: one fp16 scale map, selected per row by page kind
    # (sig_cat layout: [H, 2, PTOK, D]; map 0 = identity, 1 = DCT)
    mrow = (kind == 2).to(tl.int32)
    sig_row = ((h * 2 + mrow) * PTOK + s_in_page) * D
    ci = tl.arange(0, CBLK)
    hg = tl.arange(0, HGP)
    u = tl.zeros((HGP, PTOK), dtype=tl.float32)
    boff = 0
    for b in tl.static_range(4):
        w = tl.load(w_ptr + h * 4 + b)
        lo = tl.load(lo_ptr + h * 4 + b)
        if w > 0:            # program-uniform: dominant rungs have 1-2
            bitpos = ci * w  # zero-width blocks — skip their whole body
            addr = (pbase + rows[:, None].to(tl.int64) * stride + boff
                    + (bitpos >> 3)[None, :])
            live = live_r[:, None]
            b0 = tl.load(pay_ptr + addr, mask=live, other=0).to(tl.int32)
            b1 = tl.load(pay_ptr + addr + 1, mask=live, other=0).to(tl.int32)
            code = ((b0 | (b1 << 8)) >> (bitpos & 7)[None, :])                 & ((1 << w) - 1)
            lev = tl.load(lut_ptr + h * LUTN + lo + code, mask=live,
                          other=0.0)                          # fp16 LUT
            coord = b * CBLK + ci
            sig = tl.load(sig_ptr + sig_row[:, None] + coord[None, :],
                          mask=live, other=0.0)               # fp16 map
            deq = (lev * sig).to(tl.float16)
            qb = tl.load(qt_ptr + h * HGP * D + hg[:, None] * D
                         + coord[None, :]).to(tl.float16)
            u += tl.dot(qb, tl.trans(deq))
            boff += (w * CBLK) >> 3
    uptr = u_ptr + (h * HG + hg[:, None]) * T + tok[None, :]
    tl.store(uptr, u, mask=(hg[:, None] < HG) & live_r[None, :])


@triton.jit
def _seg_u_kernel_pl(
    pay32_ptr, tok_ptr, pl_off_ptr, tok_off_ptr, n_ptr,
    stride32_ptr, w_ptr, lo_ptr,
    lut_ptr, LUTN: tl.constexpr,
    sig_ptr, kind_ptr, qt_ptr, u_ptr,
    T, HG: tl.constexpr, HGP: tl.constexpr, D: tl.constexpr,
    PTOK: tl.constexpr, CBLK: tl.constexpr,
):
    """plan9 fusion step 1: phase A over bit-PLANE payload. Per block of 32
    codes at width w, the payload holds w uint32 planes (plane j = bit j of
    every code), so unpack is w coalesced word loads + register shifts —
    the 256 byte-gathers/row of the byte layout (phase A's measured
    instruction wall) disappear."""
    tile = tl.program_id(0)
    h = tl.program_id(1)
    n = tl.load(n_ptr + h)
    rows = tile * PTOK + tl.arange(0, PTOK)
    live_r = rows < n
    stride32 = tl.load(stride32_ptr + h).to(tl.int64)
    pbase = tl.load(pl_off_ptr + h)
    tbase = tl.load(tok_off_ptr + h)
    tok = tl.load(tok_ptr + tbase + rows, mask=live_r, other=0)
    s_in_page = tok % PTOK
    kind = tl.load(kind_ptr + tok // PTOK, mask=live_r, other=1)
    mrow = (kind == 2).to(tl.int32)
    sig_row = ((h * 2 + mrow) * PTOK + s_in_page) * D
    ci = tl.arange(0, CBLK)
    hg = tl.arange(0, HGP)
    u = tl.zeros((HGP, PTOK), dtype=tl.float32)
    row_base = pbase + rows.to(tl.int64) * stride32
    boff = 0
    for b in tl.static_range(4):
        w = tl.load(w_ptr + h * 4 + b)
        lo = tl.load(lo_ptr + h * 4 + b)
        code = tl.zeros((PTOK, CBLK), dtype=tl.int32)
        for j in range(0, w):
            plane = tl.load(pay32_ptr + row_base + boff + j,
                            mask=live_r, other=0)
            code |= ((plane[:, None] >> ci[None, :]) & 1) << j
        live = live_r[:, None] & (w > 0)
        lev = tl.load(lut_ptr + h * LUTN + lo + code, mask=live, other=0.0)
        coord = b * CBLK + ci
        sig = tl.load(sig_ptr + sig_row[:, None] + coord[None, :],
                      mask=live_r[:, None], other=0.0)
        deq = tl.where(live, lev * sig, 0.0).to(tl.float16)
        qb = tl.load(qt_ptr + h * HGP * D + hg[:, None] * D
                     + coord[None, :]).to(tl.float16)
        u += tl.dot(qb, tl.trans(deq))
        boff += w
    uptr = u_ptr + (h * HG + hg[:, None]) * T + tok[None, :]
    tl.store(uptr, u, mask=(hg[:, None] < HG) & live_r[None, :])


@triton.jit
def _seg_u_kernel_cw(
    pay_ptr, tok_ptr, PBASE, TBASE, N, H_IDX,
    lut_ptr, LUTN: tl.constexpr,
    sig_ptr, kind_ptr, qt_ptr, u_ptr,
    T, HG: tl.constexpr, HGP: tl.constexpr, D: tl.constexpr,
    PTOK: tl.constexpr, CBLK: tl.constexpr, STRIDE: tl.constexpr,
    W0: tl.constexpr, W1: tl.constexpr, W2: tl.constexpr, W3: tl.constexpr,
    L0: tl.constexpr, L1: tl.constexpr, L2: tl.constexpr, L3: tl.constexpr,
):
    """plan9 S3: constexpr-width phase A, one launch per (rung, head)
    (per-HEAD profiles make widths head-dependent; specializing per pair
    restores compile-time shifts/masks/addresses — launch count is
    amortized by CUDA graph replay)."""
    tile = tl.program_id(0)
    rows = tile * PTOK + tl.arange(0, PTOK)
    live_r = rows < N
    tok = tl.load(tok_ptr + TBASE + rows, mask=live_r, other=0)
    s_in_page = tok % PTOK
    kind = tl.load(kind_ptr + tok // PTOK, mask=live_r, other=1)
    mrow = (kind == 2).to(tl.int32)
    sig_row = ((H_IDX * 2 + mrow) * PTOK + s_in_page) * D
    ci = tl.arange(0, CBLK)
    hg = tl.arange(0, HGP)
    u = tl.zeros((HGP, PTOK), dtype=tl.float32)
    row64 = rows.to(tl.int64) * STRIDE
    if W0 > 0:
        bp = ci * W0
        addr = PBASE + row64[:, None] + (bp >> 3)[None, :]
        b0 = tl.load(pay_ptr + addr, mask=live_r[:, None], other=0).to(tl.int32)
        b1 = tl.load(pay_ptr + addr + 1, mask=live_r[:, None], other=0).to(tl.int32)
        code = ((b0 | (b1 << 8)) >> (bp & 7)[None, :]) & ((1 << W0) - 1)
        lev = tl.load(lut_ptr + H_IDX * LUTN + L0 + code,
                      mask=live_r[:, None], other=0.0)
        sig = tl.load(sig_ptr + sig_row[:, None] + ci[None, :],
                      mask=live_r[:, None], other=0.0)
        deq = (lev * sig).to(tl.float16)
        qb = tl.load(qt_ptr + H_IDX * HGP * D + hg[:, None] * D
                     + ci[None, :]).to(tl.float16)
        u += tl.dot(qb, tl.trans(deq))
    if W1 > 0:
        bp = ci * W1
        addr = PBASE + row64[:, None] + ((W0 * CBLK) >> 3) + (bp >> 3)[None, :]
        b0 = tl.load(pay_ptr + addr, mask=live_r[:, None], other=0).to(tl.int32)
        b1 = tl.load(pay_ptr + addr + 1, mask=live_r[:, None], other=0).to(tl.int32)
        code = ((b0 | (b1 << 8)) >> (bp & 7)[None, :]) & ((1 << W1) - 1)
        lev = tl.load(lut_ptr + H_IDX * LUTN + L1 + code,
                      mask=live_r[:, None], other=0.0)
        co = CBLK + ci
        sig = tl.load(sig_ptr + sig_row[:, None] + co[None, :],
                      mask=live_r[:, None], other=0.0)
        deq = (lev * sig).to(tl.float16)
        qb = tl.load(qt_ptr + H_IDX * HGP * D + hg[:, None] * D
                     + co[None, :]).to(tl.float16)
        u += tl.dot(qb, tl.trans(deq))
    if W2 > 0:
        bp = ci * W2
        addr = PBASE + row64[:, None] + (((W0 + W1) * CBLK) >> 3) \
            + (bp >> 3)[None, :]
        b0 = tl.load(pay_ptr + addr, mask=live_r[:, None], other=0).to(tl.int32)
        b1 = tl.load(pay_ptr + addr + 1, mask=live_r[:, None], other=0).to(tl.int32)
        code = ((b0 | (b1 << 8)) >> (bp & 7)[None, :]) & ((1 << W2) - 1)
        lev = tl.load(lut_ptr + H_IDX * LUTN + L2 + code,
                      mask=live_r[:, None], other=0.0)
        co = 2 * CBLK + ci
        sig = tl.load(sig_ptr + sig_row[:, None] + co[None, :],
                      mask=live_r[:, None], other=0.0)
        deq = (lev * sig).to(tl.float16)
        qb = tl.load(qt_ptr + H_IDX * HGP * D + hg[:, None] * D
                     + co[None, :]).to(tl.float16)
        u += tl.dot(qb, tl.trans(deq))
    if W3 > 0:
        bp = ci * W3
        addr = PBASE + row64[:, None] + (((W0 + W1 + W2) * CBLK) >> 3) \
            + (bp >> 3)[None, :]
        b0 = tl.load(pay_ptr + addr, mask=live_r[:, None], other=0).to(tl.int32)
        b1 = tl.load(pay_ptr + addr + 1, mask=live_r[:, None], other=0).to(tl.int32)
        code = ((b0 | (b1 << 8)) >> (bp & 7)[None, :]) & ((1 << W3) - 1)
        lev = tl.load(lut_ptr + H_IDX * LUTN + L3 + code,
                      mask=live_r[:, None], other=0.0)
        co = 3 * CBLK + ci
        sig = tl.load(sig_ptr + sig_row[:, None] + co[None, :],
                      mask=live_r[:, None], other=0.0)
        deq = (lev * sig).to(tl.float16)
        qb = tl.load(qt_ptr + H_IDX * HGP * D + hg[:, None] * D
                     + co[None, :]).to(tl.float16)
        u += tl.dot(qb, tl.trans(deq))
    uptr = u_ptr + (H_IDX * HG + hg[:, None]) * T + tok[None, :]
    tl.store(uptr, u, mask=(hg[:, None] < HG) & live_r[None, :])


@triton.jit
def _page_attn_from_u_kernel(
    u_ptr, dct_ptr, pages_ptr, kind_ptr, v_ptr,
    m_out_ptr, l_out_ptr, acc_out_ptr,
    NPAGES, PPS, sm_scale,
    T, HG: tl.constexpr, HGP: tl.constexpr, D: tl.constexpr,
    PTOK: tl.constexpr,
):
    """Phase B: per (split, head) — load u page tiles, apply the page DCT
    where flagged, online softmax + V, emit split partials."""
    split = tl.program_id(0)
    h = tl.program_id(1)
    hg = tl.arange(0, HGP)
    tt = tl.arange(0, PTOK)
    dj = tl.arange(0, D)
    m = tl.full((HGP,), float("-inf"), dtype=tl.float32)
    l = tl.zeros((HGP,), dtype=tl.float32)
    acc = tl.zeros((HGP, D), dtype=tl.float32)
    for pi in range(split * PPS, tl.minimum((split + 1) * PPS, NPAGES)):
        page = tl.load(pages_ptr + pi)
        kind = tl.load(kind_ptr + pi)
        t0 = page * PTOK
        u = tl.load(u_ptr + (h * HG + hg[:, None]) * T + t0 + tt[None, :],
                    mask=(hg < HG)[:, None], other=0.0)
        if kind == 2:
            dt = tl.load(dct_ptr + tt[:, None] * PTOK
                         + tt[None, :]).to(tl.float16)
            u = tl.dot(u.to(tl.float16), dt)
        z = u * sm_scale
        z = tl.where((hg < HG)[:, None], z, float("-inf"))
        m_new = tl.maximum(m, tl.max(z, axis=1))
        alpha = tl.where(m == float("-inf"), 0.0, tl.exp(m - m_new))
        p = tl.exp(z - m_new[:, None])
        l = l * alpha + tl.sum(p, axis=1)
        vt = tl.load(v_ptr + h * T * D + (t0 + tt)[:, None] * D
                     + dj[None, :])
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), vt)
        m = m_new
    S = tl.num_programs(0)
    tl.store(m_out_ptr + (h * S + split) * HGP + hg, m)
    tl.store(l_out_ptr + (h * S + split) * HGP + hg, l)
    tl.store(acc_out_ptr + ((h * S + split) * HGP + hg[:, None]) * D
             + dj[None, :], acc)


@triton.jit
def _page_attn_from_u2_kernel(
    u_ptr, dct_ptr, pages_ptr, kind_ptr, v_ptr,
    m_out_ptr, l_out_ptr, acc_out_ptr,
    NPAGES, PPS, sm_scale,
    T, HG: tl.constexpr, HGP: tl.constexpr, D: tl.constexpr,
    PTOK: tl.constexpr,
):
    """plan9 fusion step 2: phase B over 128-token tiles (2 pages per
    online-softmax iteration) — halves the exp/max/rescale/dot iteration
    count that dominates phase B at 64-token granularity."""
    split = tl.program_id(0)
    h = tl.program_id(1)
    hg = tl.arange(0, HGP)
    tt = tl.arange(0, PTOK)
    dj = tl.arange(0, D)
    m = tl.full((HGP,), float("-inf"), dtype=tl.float32)
    l = tl.zeros((HGP,), dtype=tl.float32)
    acc = tl.zeros((HGP, D), dtype=tl.float32)
    lo_i = split * PPS
    hi_i = tl.minimum((split + 1) * PPS, NPAGES)
    for pi in range(lo_i, hi_i, 2):
        page_a = tl.load(pages_ptr + pi)
        kind_a = tl.load(kind_ptr + pi)
        t0a = page_a * PTOK
        ua = tl.load(u_ptr + (h * HG + hg[:, None]) * T + t0a + tt[None, :],
                     mask=(hg < HG)[:, None], other=0.0)
        if kind_a == 2:
            dta = tl.load(dct_ptr + tt[:, None] * PTOK
                          + tt[None, :]).to(tl.float16)
            ua = tl.dot(ua.to(tl.float16), dta)
        have_b = (pi + 1) < hi_i
        pb_idx = tl.where(have_b, pi + 1, pi)
        page_b = tl.load(pages_ptr + pb_idx)
        kind_b = tl.load(kind_ptr + pb_idx)
        t0b = page_b * PTOK
        ub = tl.load(u_ptr + (h * HG + hg[:, None]) * T + t0b + tt[None, :],
                     mask=(hg < HG)[:, None] & have_b, other=0.0)
        if kind_b == 2:
            dtb = tl.load(dct_ptr + tt[:, None] * PTOK
                          + tt[None, :]).to(tl.float16)
            ub = tl.dot(ub.to(tl.float16), dtb)
        pad = (hg < HG)[:, None]
        za = tl.where(pad, ua * sm_scale, float("-inf"))
        zb = tl.where(pad & have_b, ub * sm_scale, float("-inf"))
        m_new = tl.maximum(m, tl.maximum(tl.max(za, axis=1),
                                         tl.max(zb, axis=1)))
        alpha = tl.where(m == float("-inf"), 0.0, tl.exp(m - m_new))
        pa = tl.exp(za - m_new[:, None]).to(tl.float16)
        pb = tl.exp(zb - m_new[:, None]).to(tl.float16)
        l = l * alpha + tl.sum(pa.to(tl.float32), axis=1)             + tl.sum(pb.to(tl.float32), axis=1)
        vta = tl.load(v_ptr + h * T * D + (t0a + tt)[:, None] * D
                      + dj[None, :])
        vtb = tl.load(v_ptr + h * T * D + (t0b + tt)[:, None] * D
                      + dj[None, :], mask=have_b & (tt[:, None] >= 0),
                      other=0.0)
        acc = acc * alpha[:, None] + tl.dot(pa, vta) + tl.dot(pb, vtb)
        m = m_new
    S = tl.num_programs(0)
    tl.store(m_out_ptr + (h * S + split) * HGP + hg, m)
    tl.store(l_out_ptr + (h * S + split) * HGP + hg, l)
    tl.store(acc_out_ptr + ((h * S + split) * HGP + hg[:, None]) * D
             + dj[None, :], acc)


@triton.jit
def _page_attn_from_u2_vq_kernel(
    u_ptr, dct_ptr, pages_ptr, kind_ptr,
    vq_ptr, vs_ptr, vz_ptr,
    m_out_ptr, l_out_ptr, acc_out_ptr,
    NPAGES, PPS, sm_scale,
    T, HG: tl.constexpr, HGP: tl.constexpr, D: tl.constexpr,
    PTOK: tl.constexpr,
):
    """plan9: 128-token tiles + INT2 V (OSCAR dequant pattern). V bytes are
    the step's dominant HBM term; at 2 pages/iter the cut finally pays."""
    split = tl.program_id(0)
    h = tl.program_id(1)
    Q: tl.constexpr = D // 4
    hg = tl.arange(0, HGP)
    tt = tl.arange(0, PTOK)
    qj = tl.arange(0, Q)
    m = tl.full((HGP,), float("-inf"), dtype=tl.float32)
    l = tl.zeros((HGP,), dtype=tl.float32)
    a0 = tl.zeros((HGP, Q), dtype=tl.float32)
    a1 = tl.zeros((HGP, Q), dtype=tl.float32)
    a2 = tl.zeros((HGP, Q), dtype=tl.float32)
    a3 = tl.zeros((HGP, Q), dtype=tl.float32)
    lo_i = split * PPS
    hi_i = tl.minimum((split + 1) * PPS, NPAGES)
    for pi in range(lo_i, hi_i, 2):
        page_a = tl.load(pages_ptr + pi)
        kind_a = tl.load(kind_ptr + pi)
        t0a = page_a * PTOK
        ua = tl.load(u_ptr + (h * HG + hg[:, None]) * T + t0a + tt[None, :],
                     mask=(hg < HG)[:, None], other=0.0)
        if kind_a == 2:
            dta = tl.load(dct_ptr + tt[:, None] * PTOK
                          + tt[None, :]).to(tl.float16)
            ua = tl.dot(ua.to(tl.float16), dta)
        have_b = (pi + 1) < hi_i
        pb_idx = tl.where(have_b, pi + 1, pi)
        page_b = tl.load(pages_ptr + pb_idx)
        kind_b = tl.load(kind_ptr + pb_idx)
        t0b = page_b * PTOK
        ub = tl.load(u_ptr + (h * HG + hg[:, None]) * T + t0b + tt[None, :],
                     mask=(hg < HG)[:, None] & have_b, other=0.0)
        if kind_b == 2:
            dtb = tl.load(dct_ptr + tt[:, None] * PTOK
                          + tt[None, :]).to(tl.float16)
            ub = tl.dot(ub.to(tl.float16), dtb)
        pad = (hg < HG)[:, None]
        za = tl.where(pad, ua * sm_scale, float("-inf"))
        zb = tl.where(pad & have_b, ub * sm_scale, float("-inf"))
        m_new = tl.maximum(m, tl.maximum(tl.max(za, axis=1),
                                         tl.max(zb, axis=1)))
        alpha = tl.where(m == float("-inf"), 0.0, tl.exp(m - m_new))
        pa = tl.exp(za - m_new[:, None]).to(tl.float16)
        pb = tl.exp(zb - m_new[:, None]).to(tl.float16)
        l = l * alpha + tl.sum(pa.to(tl.float32), axis=1)             + tl.sum(pb.to(tl.float32), axis=1)
        vqa = tl.load(vq_ptr + h * T * Q + (t0a + tt)[:, None] * Q
                      + qj[None, :])
        vsa = tl.load(vs_ptr + h * T + t0a + tt).to(tl.float16)
        vza = tl.load(vz_ptr + h * T + t0a + tt).to(tl.float16)
        vqb2 = tl.load(vq_ptr + h * T * Q + (t0b + tt)[:, None] * Q
                       + qj[None, :], mask=have_b & (tt[:, None] >= 0),
                       other=0)
        vsb = tl.load(vs_ptr + h * T + t0b + tt, mask=have_b, other=0.0
                      ).to(tl.float16)
        vzb = tl.load(vz_ptr + h * T + t0b + tt, mask=have_b, other=0.0
                      ).to(tl.float16)
        a0 = a0 * alpha[:, None]             + tl.dot(pa, (vqa & 3).to(tl.float16) * vsa[:, None]
                     + vza[:, None])             + tl.dot(pb, (vqb2 & 3).to(tl.float16) * vsb[:, None]
                     + vzb[:, None])
        a1 = a1 * alpha[:, None]             + tl.dot(pa, ((vqa >> 2) & 3).to(tl.float16) * vsa[:, None]
                     + vza[:, None])             + tl.dot(pb, ((vqb2 >> 2) & 3).to(tl.float16) * vsb[:, None]
                     + vzb[:, None])
        a2 = a2 * alpha[:, None]             + tl.dot(pa, ((vqa >> 4) & 3).to(tl.float16) * vsa[:, None]
                     + vza[:, None])             + tl.dot(pb, ((vqb2 >> 4) & 3).to(tl.float16) * vsb[:, None]
                     + vzb[:, None])
        a3 = a3 * alpha[:, None]             + tl.dot(pa, ((vqa >> 6) & 3).to(tl.float16) * vsa[:, None]
                     + vza[:, None])             + tl.dot(pb, ((vqb2 >> 6) & 3).to(tl.float16) * vsb[:, None]
                     + vzb[:, None])
        m = m_new
    S = tl.num_programs(0)
    tl.store(m_out_ptr + (h * S + split) * HGP + hg, m)
    tl.store(l_out_ptr + (h * S + split) * HGP + hg, l)
    base = ((h * S + split) * HGP + hg[:, None]) * D
    tl.store(acc_out_ptr + base + 0 * Q + qj[None, :], a0)
    tl.store(acc_out_ptr + base + 1 * Q + qj[None, :], a1)
    tl.store(acc_out_ptr + base + 2 * Q + qj[None, :], a2)
    tl.store(acc_out_ptr + base + 3 * Q + qj[None, :], a3)


@triton.jit
def _page_attn_from_u_vq_kernel(
    u_ptr, dct_ptr, pages_ptr, kind_ptr,
    vq_ptr, vs_ptr, vz_ptr,
    m_out_ptr, l_out_ptr, acc_out_ptr,
    NPAGES, PPS, sm_scale,
    T, HG: tl.constexpr, HGP: tl.constexpr, D: tl.constexpr,
    PTOK: tl.constexpr,
):
    """Phase B with INT2 values, OSCAR's measured-good pattern (plan9 S1;
    vendored decode_attention.py:1715-1896): quarters stay separate, the
    per-token zero folds into dequantized V BEFORE the dot, everything in
    fp16, four tl.dot into quarter accumulators."""
    split = tl.program_id(0)
    h = tl.program_id(1)
    Q: tl.constexpr = D // 4
    hg = tl.arange(0, HGP)
    tt = tl.arange(0, PTOK)
    qj = tl.arange(0, Q)
    m = tl.full((HGP,), float("-inf"), dtype=tl.float32)
    l = tl.zeros((HGP,), dtype=tl.float32)
    a0 = tl.zeros((HGP, Q), dtype=tl.float32)
    a1 = tl.zeros((HGP, Q), dtype=tl.float32)
    a2 = tl.zeros((HGP, Q), dtype=tl.float32)
    a3 = tl.zeros((HGP, Q), dtype=tl.float32)
    for pi in range(split * PPS, tl.minimum((split + 1) * PPS, NPAGES)):
        page = tl.load(pages_ptr + pi)
        kind = tl.load(kind_ptr + pi)
        t0 = page * PTOK
        u = tl.load(u_ptr + (h * HG + hg[:, None]) * T + t0 + tt[None, :],
                    mask=(hg < HG)[:, None], other=0.0)
        if kind == 2:
            dt = tl.load(dct_ptr + tt[:, None] * PTOK
                         + tt[None, :]).to(tl.float16)
            u = tl.dot(u.to(tl.float16), dt)
        z = u * sm_scale
        z = tl.where((hg < HG)[:, None], z, float("-inf"))
        m_new = tl.maximum(m, tl.max(z, axis=1))
        alpha = tl.where(m == float("-inf"), 0.0, tl.exp(m - m_new))
        p16 = tl.exp(z - m_new[:, None]).to(tl.float16)
        l = l * alpha + tl.sum(p16.to(tl.float32), axis=1)
        vqb = tl.load(vq_ptr + h * T * Q + (t0 + tt)[:, None] * Q
                      + qj[None, :])                           # (PTOK, Q) u8
        vs = tl.load(vs_ptr + h * T + t0 + tt).to(tl.float16)  # (PTOK,)
        vz = tl.load(vz_ptr + h * T + t0 + tt).to(tl.float16)
        # our quantizer stores v = c*s + z (z = vmin in VALUE units)
        v0 = (vqb & 0x03).to(tl.float16) * vs[:, None] + vz[:, None]
        v1 = ((vqb >> 2) & 0x03).to(tl.float16) * vs[:, None] + vz[:, None]
        v2 = ((vqb >> 4) & 0x03).to(tl.float16) * vs[:, None] + vz[:, None]
        v3 = ((vqb >> 6) & 0x03).to(tl.float16) * vs[:, None] + vz[:, None]
        a0 = a0 * alpha[:, None] + tl.dot(p16, v0)
        a1 = a1 * alpha[:, None] + tl.dot(p16, v1)
        a2 = a2 * alpha[:, None] + tl.dot(p16, v2)
        a3 = a3 * alpha[:, None] + tl.dot(p16, v3)
        m = m_new
    S = tl.num_programs(0)
    tl.store(m_out_ptr + (h * S + split) * HGP + hg, m)
    tl.store(l_out_ptr + (h * S + split) * HGP + hg, l)
    base = ((h * S + split) * HGP + hg[:, None]) * D
    tl.store(acc_out_ptr + base + 0 * Q + qj[None, :], a0)
    tl.store(acc_out_ptr + base + 1 * Q + qj[None, :], a1)
    tl.store(acc_out_ptr + base + 2 * Q + qj[None, :], a2)
    tl.store(acc_out_ptr + base + 3 * Q + qj[None, :], a3)


def page_attention_v2(gm: dict, pages_per_split: int = 16,
                      num_warps: int = 4, v_int2: bool = False,
                      return_partials: bool = False,
                      phase_a: str = "rt", phase_b: str = "kernel"):
    """Coalesced two-phase path (K2c'). Needs golden.segment_layout keys;
    v_int2=True additionally needs golden.add_v_int2 buffers (vq/vqs/vqz)
    and serves values from INT2 codes (the torch tier still reads gm['v'],
    which must then be the dequantized v_hat)."""
    H, T, d, ptok = gm["H"], gm["T"], gm["d"], gm["ptok"]
    hg, hgp = gm["hg"], gm["hgp"]
    dev = gm["qt"].device
    u = torch.zeros(H * hg, T, device=dev, dtype=torch.float32)
    if phase_a == "pl":
        if "_pl_plan" not in gm:
            seg_n = gm["seg_n"].cpu()
            s32 = gm["seg_w"].sum(-1)
            gm["_stride32"] = s32.to(torch.int32)
            s32c = s32.cpu()
            plan = []
            for r in range(seg_n.shape[0]):
                nmax = int(seg_n[r].max())
                if nmax == 0 or int(s32c[r].max()) == 0:
                    continue
                plan.append((r, nmax))
            gm["_pl_plan"] = plan
        for r, nmax in gm["_pl_plan"]:
            grid = ((nmax + ptok - 1) // ptok, H)
            _seg_u_kernel_pl[grid](
                gm["seg_pay_pl"], gm["seg_tok"],
                gm["seg_pl_off"][r], gm["seg_tok_off"][r], gm["seg_n"][r],
                gm["_stride32"][r], gm["seg_w"][r], gm["seg_lo"][r],
                gm["lut16"], gm["lut16"].shape[1],
                gm["sig_cat"], gm["kind_dense"], gm["qt"], u,
                T, HG=hg, HGP=hgp, D=d, PTOK=ptok,
                CBLK=d // gm["seg_w"].shape[2],
                num_warps=num_warps, num_stages=3,
            )
    elif phase_a == "cw":
        if "_cw_plan" not in gm:
            seg_n = gm["seg_n"].cpu()
            strides = gm["seg_stride"].cpu()
            w_host = gm["seg_w"].cpu()
            lo_host = gm["seg_lo"].cpu()
            po = gm["seg_pay_off"].cpu()
            to = gm["seg_tok_off"].cpu()
            plan = []
            for r in range(seg_n.shape[0]):
                for hh in range(H):
                    n = int(seg_n[r, hh])
                    stride = int(strides[r, hh])
                    if n == 0 or stride == 0:
                        continue
                    plan.append((int(po[r, hh]), int(to[r, hh]), n, hh,
                                 stride,
                                 [int(x) for x in w_host[r, hh]],
                                 [int(x) for x in lo_host[r, hh]]))
            gm["_cw_plan"] = plan
        for pbase, tbase, n, hh, stride, ws, los in gm["_cw_plan"]:
            _seg_u_kernel_cw[((n + ptok - 1) // ptok,)](
                gm["seg_pay"], gm["seg_tok"], pbase, tbase, n, hh,
                gm["lut16"], gm["lut16"].shape[1],
                gm["sig_cat"], gm["kind_dense"], gm["qt"], u,
                T, HG=hg, HGP=hgp, D=d, PTOK=ptok,
                CBLK=d // gm["seg_w"].shape[2], STRIDE=stride,
                W0=ws[0], W1=ws[1], W2=ws[2], W3=ws[3],
                L0=los[0], L1=los[1], L2=los[2], L3=los[3],
                num_warps=num_warps, num_stages=3,
            )
    elif "_launch_plan" not in gm:
        seg_n = gm["seg_n"].cpu()
        strides = gm["seg_stride"].cpu()
        plan = []
        for r in range(seg_n.shape[0]):
            nmax = int(seg_n[r].max())
            if nmax == 0 or int(strides[r].max()) == 0:
                continue
            plan.append((r, nmax))
        gm["_launch_plan"] = plan
    for r, nmax in (gm.get("_launch_plan", [])
                    if phase_a not in ("cw", "pl") else []):
        grid = ((nmax + ptok - 1) // ptok, H)
        _seg_u_kernel[grid](
            gm["seg_pay"], gm["seg_tok"],
            gm["seg_pay_off"][r], gm["seg_tok_off"][r], gm["seg_n"][r],
            gm["seg_stride"][r], gm["seg_w"][r], gm["seg_lo"][r],
            gm["lut16"], gm["lut16"].shape[1],
            gm["sig_cat"], gm["kind_dense"],
            gm["qt"], u,
            T, HG=hg, HGP=hgp, D=d, PTOK=ptok,
            CBLK=d // gm["seg_w"].shape[2],
            num_warps=num_warps, num_stages=3,
        )
    if phase_b == "torch":
        # plan9 S3b: phase B as dense batched torch under the same CUDA
        # graph — one cuBLAS GEMM applies the page DCT to every DCT page at
        # once, then an EXACT full softmax (no split partials, no stage 2)
        # and one bmm for p @ V. Correct because u already holds every
        # served row's coefficient logit.
        if "_tb" not in gm:
            # index prep cached ONCE (host syncs here are fine — this runs
            # during warmup, never inside CUDA graph capture)
            pages = gm["pages"].long()
            kinds = gm["page_kind"]
            ar = torch.arange(ptok, device=dev)
            dctp = pages[kinds == 2]
            idp = pages[kinds != 2]
            gm["_tb"] = {
                "di": (dctp[:, None] * ptok + ar[None, :]) if dctp.numel()
                      else None,
                "ii": (idp[:, None] * ptok + ar[None, :]) if idp.numel()
                      else None,
                "ti": gm["tier_idx"] if gm["tier_idx"].numel() else None,
                "vf": gm["v"].float(),
            }
        tb = gm["_tb"]
        uv = u.view(H, hg, T)
        z = torch.full_like(uv, float("-inf"))
        if tb["di"] is not None:
            z[:, :, tb["di"]] = uv[:, :, tb["di"]] @ gm["dct_t"]
        if tb["ii"] is not None:
            z[:, :, tb["ii"]] = uv[:, :, tb["ii"]]
        if tb["ti"] is not None:
            z[:, :, tb["ti"]] = gm["z_tier"].float()
        att = torch.softmax(z * gm["sm_scale"], dim=2)
        return torch.bmm(att.reshape(H, hg, T), tb["vf"])
    npages = int(gm["pages"].numel())
    nsplit = max(1, (npages + pages_per_split - 1) // pages_per_split)
    m_p = torch.empty(H, nsplit, hgp, device=dev)
    l_p = torch.empty(H, nsplit, hgp, device=dev)
    acc_p = torch.empty(H, nsplit, hgp, d, device=dev)
    if phase_b == "kernel2" and v_int2:
        _page_attn_from_u2_vq_kernel[(nsplit, H)](
            u.reshape(H, hg, T), gm["dct_t"], gm["pages"], gm["page_kind"],
            gm["vq"], gm["vqs"], gm["vqz"], m_p, l_p, acc_p,
            npages, pages_per_split, gm["sm_scale"],
            T, HG=hg, HGP=hgp, D=d, PTOK=ptok, num_warps=num_warps,
            num_stages=3,
        )
    elif phase_b == "kernel2":
        _page_attn_from_u2_kernel[(nsplit, H)](
            u.reshape(H, hg, T), gm["dct_t"], gm["pages"], gm["page_kind"],
            gm["v"], m_p, l_p, acc_p,
            npages, pages_per_split, gm["sm_scale"],
            T, HG=hg, HGP=hgp, D=d, PTOK=ptok, num_warps=num_warps,
            num_stages=3,
        )
    elif v_int2:
        _page_attn_from_u_vq_kernel[(nsplit, H)](
            u.reshape(H, hg, T), gm["dct_t"], gm["pages"], gm["page_kind"],
            gm["vq"], gm["vqs"], gm["vqz"], m_p, l_p, acc_p,
            npages, pages_per_split, gm["sm_scale"],
            T, HG=hg, HGP=hgp, D=d, PTOK=ptok, num_warps=num_warps,
            num_stages=3,
        )
    else:
        _page_attn_from_u_kernel[(nsplit, H)](
            u.reshape(H, hg, T), gm["dct_t"], gm["pages"], gm["page_kind"],
            gm["v"], m_p, l_p, acc_p,
            npages, pages_per_split, gm["sm_scale"],
            T, HG=hg, HGP=hgp, D=d, PTOK=ptok, num_warps=num_warps,
        )
    if return_partials:
        return m_p, l_p, acc_p
    ti = gm["tier_idx"]
    if ti.numel():
        zt = gm["z_tier"] * gm["sm_scale"]
        mt = torch.full((H, 1, hgp), float("-inf"), device=dev)
        lt = torch.zeros(H, 1, hgp, device=dev)
        at = torch.zeros(H, 1, hgp, d, device=dev)
        mt[:, 0, :hg] = zt.max(dim=2).values
        pt = torch.exp(zt - mt[:, 0, :hg, None])
        lt[:, 0, :hg] = pt.sum(2)
        at[:, 0, :hg] = pt @ gm["v"][:, ti].float()
        m_p = torch.cat([m_p, mt], 1)
        l_p = torch.cat([l_p, lt], 1)
        acc_p = torch.cat([acc_p, at], 1)
    m_star = m_p.max(1).values
    w = torch.exp(m_p - m_star[:, None, :]).nan_to_num(0.0)
    l_star = (l_p * w).sum(1).clamp_min(1e-30)
    o = (acc_p * w[..., None]).sum(1) / l_star[..., None]
    return o[:, :hg]


def page_attention_v1(gm: dict, pages_per_split: int = 8,
                      num_warps: int = 4) -> torch.Tensor:
    """Multi-head v1 path over a stack_heads dict; returns (H, Hg, d)."""
    H, T, d, ptok = gm["H"], gm["T"], gm["d"], gm["ptok"]
    hg, hgp = gm["hg"], gm["hgp"]
    dev = gm["qt"].device
    npages = int(gm["pages"].numel())
    nsplit = max(1, (npages + pages_per_split - 1) // pages_per_split)
    m_p = torch.empty(H, nsplit, hgp, device=dev)
    l_p = torch.empty(H, nsplit, hgp, device=dev)
    acc_p = torch.empty(H, nsplit, hgp, d, device=dev)
    _page_attn_kernel_v1[(nsplit, H)](
        gm["payload"], gm["pay_off"], gm["row_ptr"], gm["row_rung"],
        gm["bw"], gm["lut"], gm["lut_off"], gm["lut"].shape[1],
        gm["sigma_id"], gm["sigma_dct"], gm["dct_t"],
        gm["pages"], gm["page_kind"], gm["qt"], gm["v"],
        m_p, l_p, acc_p,
        npages, pages_per_split, gm["sm_scale"],
        T, HGP=hgp, D=d, PTOK=ptok,
        NBLK=gm["bw"].shape[1], CBLK=d // gm["bw"].shape[1],
        num_warps=num_warps,
    )
    # fold the torch tier in as one more split, reduce (stage 2)
    ti = gm["tier_idx"]
    if ti.numel():
        zt = gm["z_tier"] * gm["sm_scale"]                     # (H, Hg, nt)
        mt = torch.full((H, 1, hgp), float("-inf"), device=dev)
        lt = torch.zeros(H, 1, hgp, device=dev)
        at = torch.zeros(H, 1, hgp, d, device=dev)
        mt[:, 0, :hg] = zt.max(dim=2).values
        pt = torch.exp(zt - mt[:, 0, :hg, None])
        lt[:, 0, :hg] = pt.sum(2)
        at[:, 0, :hg] = pt @ gm["v"][:, ti].float()
        m_p = torch.cat([m_p, mt], 1)
        l_p = torch.cat([l_p, lt], 1)
        acc_p = torch.cat([acc_p, at], 1)
    m_star = m_p.max(1).values
    w = torch.exp(m_p - m_star[:, None, :])
    l_star = (l_p * w).sum(1).clamp_min(1e-30)
    o = (acc_p * w[..., None]).sum(1) / l_star[..., None]
    return o[:, :hg]


def page_attention(g: dict, pages_per_split: int = 8) -> torch.Tensor:
    """Full decode attention over kernel pages + torch tier. Returns
    (Hg, d) fp32 output; reference is g['o_ref'] (build_golden with v)."""
    Hg, d = g["qt"].shape
    T, ptok = g["T"], g["ptok"]
    dev = g["qt"].device
    npages = int(g["pages"].numel())
    nsplit = max(1, (npages + pages_per_split - 1) // pages_per_split)
    m_p = torch.empty(nsplit, Hg, device=dev, dtype=torch.float32)
    l_p = torch.empty(nsplit, Hg, device=dev, dtype=torch.float32)
    acc_p = torch.empty(nsplit, Hg, d, device=dev, dtype=torch.float32)
    if npages:
        _page_attn_kernel[(nsplit,)](
            g["payload"], g["row_ptr"], g["row_rung"],
            g["bw"], g["lut"], g["lut_off"],
            g["sigma_id"], g["sigma_dct"], g["dct_t"],
            g["pages"], g["page_kind"], g["qt"], g["v"],
            m_p, l_p, acc_p,
            npages, pages_per_split, g["sm_scale"],
            T, HG=Hg, D=d, PTOK=ptok,
            NBLK=g["bw"].shape[1], CBLK=d // g["bw"].shape[1],
            num_warps=4,
        )
    else:
        m_p.fill_(float("-inf"))
        l_p.zero_()
        acc_p.zero_()

    # torch tier (sink page, partial tail, fp16 ring) as one extra split
    ti = g["tier_idx"]
    if ti.numel():
        zt = g["z_tier"] * g["sm_scale"]                     # (Hg, n_tier)
        mt = zt.max(dim=1).values
        pt = torch.exp(zt - mt[:, None])
        lt = pt.sum(1)
        at = pt @ g["v"][ti].float()
        m_p = torch.cat([m_p, mt[None]], 0)
        l_p = torch.cat([l_p, lt[None]], 0)
        acc_p = torch.cat([acc_p, at[None]], 0)

    # stage-2 reduce across splits (torch in v0; Triton port is the K2c
    # perf pass — the math is OSCAR's _fwd_kernel_stage2_unified)
    m_star = m_p.max(0).values                               # (Hg,)
    w = torch.exp(m_p - m_star[None, :])                     # (S, Hg)
    l_star = (l_p * w).sum(0)
    o = (acc_p * w[:, :, None]).sum(0) / l_star[:, None]
    return o


def page_logits(g: dict) -> torch.Tensor:
    """Run the v0 kernel over golden dict g; returns (Hg, T) fp32 logits
    with NaN outside kernel-served pages (matching g['z_ref'])."""
    Hg, d = g["qt"].shape
    T, ptok = g["T"], g["ptok"]
    out = torch.full((Hg, T), float("nan"),
                     device=g["qt"].device, dtype=torch.float32)
    npages = int(g["pages"].numel())
    if npages == 0:
        return out
    _page_logits_kernel[(npages,)](
        g["payload"], g["row_ptr"], g["row_rung"],
        g["bw"], g["lut"], g["lut_off"],
        g["sigma_id"], g["sigma_dct"], g["dct_t"],
        g["pages"], g["page_kind"], g["qt"], out,
        T, HG=Hg, D=d, PTOK=ptok,
        NBLK=g["bw"].shape[1], CBLK=d // g["bw"].shape[1],
        num_warps=4,
    )
    return out
