"""pgq7 K2 golden vectors: kernel-ready device arrays + fp32 references.

Converts one (layer, head)'s compressor + emit dict (pgq4 token rows or
pgq8 DCT coefficient rows) into the flat buffers the Triton decode kernel
consumes, plus the exact torch reference the kernel is gated against.

Kernel-side view of the format (v0, correctness-first):
  payload   uint8 [total+1]   per-row packed codes, original row order
  row_ptr   int32 [T]         byte offset of each row's payload
  row_rung  uint8 [T]         3-bit rung id (uint8 here)
  bw        int32 [R, NBLK]   per-rung block widths
  lut       fp32  [91]        unit levels: w2 (4) | w3 (8) | w4 (16) |
                              w6 (63, alpha_top folded in)
  lut_off   int32 [7]         width -> offset into lut (0 for unused)
  sigma_id  fp32 [ptok, d]    identity-page scales (code_std broadcast)
  sigma_dct fp32 [ptok, d]    DCT-page per-(coefficient-row, coord) scales
  dct_t     fp32 [ptok, ptok] transform D (z_page = u @ D for DCT pages)
  pages     int32 [Np]        page indices the kernel processes
  page_kind int8  [Np]        1 = identity-quant page, 2 = DCT page
  qt        fp32 [Hg, d]      folded queries q @ G^T (group of q heads)

Page 0 (sink escape) and any fp16-tier pages are EXCLUDED from `pages` —
they go through the BF16/torch path like OSCAR's HP tier.

References: logits z_ref [Hg, T] computed from emit['y_hat'] (so kernel
parity checks quantization + unpack + scales + transform, not the RDO), and
attention output o_ref for the full-attention gate later in K2.
"""
from __future__ import annotations

import torch

from kvq.kernels.pgq_pack import BLOCK, block_widths

WIDTH_SET = (2, 3, 4, 6)


def build_lut(comp) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit dequant levels per width; top-width alpha folded in."""
    assert comp.grid == "lm", "golden v0 targets the shipping LM grids"
    lut = torch.zeros(4 + 8 + 16 + 63)
    off = torch.zeros(7, dtype=torch.int32)
    o = 0
    top = comp.width_ladder[-1]
    for wi, w in enumerate(comp.widths_pos):
        if w < top:
            vals = comp.lm_cents[wi]
        else:
            lim = (1 << (w - 1)) - 1
            vals = (torch.arange(2 * lim + 1) - lim).float() * comp.alphas[wi]
        lut[o:o + vals.numel()] = vals
        off[w] = o
        o += vals.numel()
    return lut, off


def build_golden(comp, emit: dict, packed: dict, q: torch.Tensor,
                 device="cuda", v: torch.Tensor | None = None) -> dict:
    """comp: FoldedScalarPagedCompressor or PageDCTCompressor (d=128,
    ptok=64 geometry). emit: comp.roundtrip(..., emit={}) output. packed:
    pgq_pack.pack_sequence of that emit. q: (Hg, d) raw queries. Pass
    v (T, d) values to additionally build the full-attention reference
    (o_ref) and the torch-tier logits (tier_idx/z_tier) for rows the
    kernel doesn't serve — all logits mu-dropped, so tiers compose."""
    T, d = emit["codes"].shape
    ptok = comp.ptok
    P = (T + ptok - 1) // ptok
    dev = torch.device(device)

    # flat per-row payload in original row order
    strides = packed["strides"].long()
    row_rung = emit["assign"].to(torch.uint8)
    row_len = strides[emit["assign"]]
    if emit["nsink"]:
        row_len = row_len.clone()
        row_len[: emit["nsink"]] = 0                 # sink rows: no payload
    row_ptr = torch.zeros(T, dtype=torch.int32)
    row_ptr[1:] = row_len.cumsum(0)[:-1].to(torch.int32)
    payload = torch.zeros(int(row_len.sum()) + 1, dtype=torch.uint8)
    for ri, toks in enumerate(packed["rung_tokens"]):
        buf = packed["payload"][ri]
        L = int(strides[ri])
        for j, t in enumerate(toks.tolist()):
            payload[int(row_ptr[t]): int(row_ptr[t]) + L] = buf[j]

    tmask = emit.get("tmask")
    nfull = emit.get("nfull", P - (1 if T % ptok else 0))
    is_dct = tmask is not None and bool(tmask.any())
    pages, kinds = [], []
    for p in range(P):
        if p == 0 and emit["nsink"]:
            continue                                  # sink page: torch tier
        if (p + 1) * ptok > T:
            continue                                  # partial page: torch tier
        if is_dct and p < len(tmask) and bool(tmask[p]):
            pages.append(p)
            kinds.append(2)
        else:
            pages.append(p)
            kinds.append(1)

    sigma_id = comp.code_std.unsqueeze(0).expand(ptok, d).contiguous()
    sigma_dct = (comp.dct_std if hasattr(comp, "dct_std")
                 else sigma_id).contiguous()
    dct_t = (comp.dct_m if hasattr(comp, "dct_m")
             else torch.eye(ptok)).contiguous()
    lut, lut_off = build_lut(comp)
    qt = q.float() @ comp.inverse_map.t()                   # q G^T

    # fp32 reference logits from the emitted quantized rows (pgq8 emits the
    # coefficient-domain y_hat; the pgq4 parent's rows ARE its r_hat)
    y_hat = emit.get("y_hat", emit.get("r_hat"))
    z_ref = torch.full((q.shape[0], T), float("nan"))
    for p, kind in zip(pages, kinds):
        sl = slice(p * ptok, (p + 1) * ptok)
        u = qt @ y_hat[sl].t()                        # (Hg, ptok)
        if kind == 2:
            u = u @ dct_t                             # z = u D
        z_ref[:, sl] = u

    extra = {}
    if v is not None:
        served = torch.zeros(T, dtype=torch.bool)
        for p in pages:
            served[p * ptok: (p + 1) * ptok] = True
        tier_idx = torch.nonzero(~served).squeeze(1)
        z_full = z_ref.clone()
        z_full[:, tier_idx] = qt @ y_hat[tier_idx].t()   # identity rows
        import math as _math
        sm = 1.0 / _math.sqrt(d)
        att = torch.softmax(z_full * sm, dim=1)
        extra = {"v": v.half(), "o_ref": att @ v.float(),
                 "tier_idx": tier_idx.to(torch.int64),
                 "z_tier": (qt @ y_hat[tier_idx].t()), "sm_scale": sm}

    out = {
        **extra,
        "payload": payload, "row_ptr": row_ptr, "row_rung": row_rung,
        "bw": block_widths(comp.profiles, BLOCK).to(torch.int32),
        "strides": strides.to(torch.int32),
        "lut": lut, "lut_off": lut_off,
        "sigma_id": sigma_id, "sigma_dct": sigma_dct, "dct_t": dct_t,
        "pages": torch.tensor(pages, dtype=torch.int32),
        "page_kind": torch.tensor(kinds, dtype=torch.int8),
        "qt": qt.contiguous(), "z_ref": z_ref,
        "T": T, "d": d, "ptok": ptok,
    }
    return {k: (v.to(dev) if torch.is_tensor(v) else v)
            for k, v in out.items()}
