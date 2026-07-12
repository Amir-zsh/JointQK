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
