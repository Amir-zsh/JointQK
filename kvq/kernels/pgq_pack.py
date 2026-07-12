"""pgq7 K1: packed byte format for the folded-scalar paged codec (plan7 §3).

Kernel-facing serialization of FoldedScalarPagedCompressor output, per
(layer, kv-head) sequence:

  header      3-bit rung id per token, original order, little-endian packed.
              Sink rows (positions 0-3 when start_pos == 0) keep their
              nominal id but are decoded from the sink segment by POSITION —
              zero sideband, exactly the compressor's escape rule.
  payload     per rung: (n_tokens, stride) uint8. A token's payload is its
              profile's width-blocks in coordinate order; a block of BLOCK
              coords at width w packs BLOCK*w/8 bytes little-endian (bit j of
              code i lands at bit i*w + j) — byte-aligned for the ladder
              {2,3,4,6} at BLOCK=32 (8/12/16/24 B). Width-0 blocks pack
              nothing. Stride is constant GIVEN the rung: one constexpr
              stage-1 launch per rung segment (plan7 §3).
  rung_tokens per rung: original-order token indices, built per PAGE so
              same-rung tokens stay contiguous within a page (plan7 R2) —
              these are the K2 stage-1 index lists.
  sink        (nsink, d) uint8 absolute 8-bit codes, code = idx + SINK_LIM.

`dequant_codes` reproduces the compressor's reconstruction BIT-IDENTICALLY —
same operand order per coord (LM: cents[c] * code_std_j; uniform:
(c - lim) * (alpha_w * code_std_j); sink: (c - SINK_LIM) * sink_scale_j) —
so pack -> unpack -> dequant equals roundtrip's emitted r_hat exactly.
tests/test_pgq_pack.py pins this on synthetic data and real Qwen selection
rows. The K2 kernel folds code_std / sink_scale into the query instead and
dots LUT unit levels against integer codes; that path is last-ulp different
and is gated at 1e-2 rel, not bit identity.
"""
from __future__ import annotations

import math

import torch

from kvq.compression.pgq4_folded import SINK_LIM

PACK_VERSION = 1
BLOCK = 32


def _pack_bits(codes: torch.Tensor, w: int) -> torch.Tensor:
    """(n, m) uint8 codes < 2^w -> (n, ceil(m*w/8)) uint8 little-endian."""
    n, m = codes.shape
    dev = codes.device
    j = torch.arange(w, device=dev, dtype=torch.uint8)
    bits = ((codes.unsqueeze(2) >> j) & 1).reshape(n, m * w)
    nbytes = (m * w + 7) // 8
    if nbytes * 8 != m * w:
        bits = torch.cat([bits, torch.zeros(n, nbytes * 8 - m * w,
                                            dtype=torch.uint8, device=dev)], 1)
    weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128],
                           dtype=torch.int16, device=dev)
    return ((bits.reshape(n, nbytes, 8).to(torch.int16) * weights)
            .sum(2).to(torch.uint8))


def _unpack_bits(data: torch.Tensor, w: int, m: int) -> torch.Tensor:
    """(n, nbytes) uint8 -> (n, m) uint8 codes; inverse of _pack_bits."""
    n = data.shape[0]
    dev = data.device
    j = torch.arange(8, device=dev, dtype=torch.uint8)
    bits = ((data.unsqueeze(2) >> j) & 1).reshape(n, -1)[:, :m * w]
    weights = (torch.ones(1, dtype=torch.int16, device=dev)
               << torch.arange(w, device=dev, dtype=torch.int16))
    return ((bits.reshape(n, m, w).to(torch.int16) * weights)
            .sum(2).to(torch.uint8))


def block_widths(profiles: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """(R, d) per-coord widths -> (R, d/block) per-block widths; raises if a
    profile changes width inside a block (the format's alignment invariant)."""
    R, d = profiles.shape
    pw = profiles.reshape(R, d // block, block)
    if not bool((pw == pw[:, :, :1]).all()):
        raise ValueError("profiles must be block-constant")
    return pw[:, :, 0].long()


def pack_sequence(codes: torch.Tensor, assign: torch.Tensor,
                  profiles: torch.Tensor, ptok: int, nsink: int = 0,
                  sink_codes: torch.Tensor | None = None,
                  block: int = BLOCK) -> dict:
    """Pack one (layer, head) sequence from roundtrip's emit dict."""
    T, d = codes.shape
    R = profiles.shape[0]
    dev = codes.device
    bw = block_widths(profiles, block)
    strides = (bw * block).sum(1) // 8                       # bytes/token

    npages = (T + ptok - 1) // ptok
    rung_tokens = [[] for _ in range(R)]
    for p in range(npages):
        lo = p * ptok + (nsink if p == 0 else 0)
        hi = min((p + 1) * ptok, T)
        idx = torch.arange(lo, hi, device=dev)
        a = assign[lo:hi]
        for ri in range(R):
            sel = idx[a == ri]
            if sel.numel():
                rung_tokens[ri].append(sel)
    rung_tokens = [torch.cat(x) if x
                   else torch.empty(0, dtype=torch.long, device=dev)
                   for x in rung_tokens]

    payload = []
    for ri in range(R):
        c = codes[rung_tokens[ri]]
        segs = [_pack_bits(c[:, b * block:(b + 1) * block], int(w))
                for b, w in enumerate(bw[ri]) if w > 0]
        payload.append(torch.cat(segs, 1) if segs
                       else torch.empty(len(c), 0, dtype=torch.uint8,
                                        device=dev))

    id_bits = max(1, math.ceil(math.log2(max(2, R))))
    return {
        "version": PACK_VERSION, "T": T, "d": d, "block": block,
        "ptok": int(ptok), "nsink": int(nsink), "id_bits": id_bits,
        "block_widths": bw, "strides": strides,
        "rung_ids": _pack_bits(assign.to(torch.uint8).unsqueeze(0),
                               id_bits).squeeze(0),
        "rung_tokens": rung_tokens, "payload": payload,
        "sink_payload": (sink_codes if nsink
                         else torch.empty(0, d, dtype=torch.uint8,
                                          device=dev)),
    }


def unpack_sequence(packed: dict):
    """-> (codes (T, d) uint8 with width-0 coords zero, assign (T,) long,
    sink_codes). Exact inverse of pack_sequence."""
    T, d, block = packed["T"], packed["d"], packed["block"]
    bw = packed["block_widths"]
    dev = packed["rung_ids"].device
    assign = _unpack_bits(packed["rung_ids"].unsqueeze(0),
                          packed["id_bits"], T).squeeze(0).long()
    codes = torch.zeros(T, d, dtype=torch.uint8, device=dev)
    for ri, toks in enumerate(packed["rung_tokens"]):
        if not toks.numel():
            continue
        buf = packed["payload"][ri]
        out = torch.zeros(toks.numel(), d, dtype=torch.uint8, device=dev)
        off = 0
        for b, w in enumerate(bw[ri]):
            w = int(w)
            if w == 0:
                continue
            nb = block * w // 8
            out[:, b * block:(b + 1) * block] = _unpack_bits(
                buf[:, off:off + nb], w, block)
            off += nb
        codes[toks] = out
    return codes, assign, packed["sink_payload"]


def dequant_codes(codes: torch.Tensor, assign: torch.Tensor, comp,
                  nsink: int = 0,
                  sink_codes: torch.Tensor | None = None) -> torch.Tensor:
    """Codes -> r_hat, bit-identical to the compressor's reconstruction
    (comp is the FoldedScalarPagedCompressor the codes came from)."""
    T, d = codes.shape
    r_hat = torch.zeros(T, d, device=codes.device)
    top = comp.width_ladder[-1]
    for ri in range(comp.n_rungs):
        msk = assign == ri
        if not msk.any():
            continue
        row = comp.profiles[ri]
        block = torch.zeros(int(msk.sum()), d, device=codes.device)
        for wi, w in enumerate(comp.widths_pos):
            cols = row == w
            if not cols.any():
                continue
            c = codes[msk][:, cols].long()
            if comp.grid == "lm" and w < top:
                block[:, cols] = (comp.lm_cents[wi][c]
                                  * comp.code_std[cols].unsqueeze(0))
            else:
                lim = (1 << (w - 1)) - 1
                s = (comp.alphas[wi] * comp.code_std)[cols]
                block[:, cols] = (c.float() - lim) * s.unsqueeze(0)
        r_hat[msk] = block
    if nsink:
        r_hat[:nsink] = ((sink_codes.float() - SINK_LIM)
                         * comp.sink_scale.unsqueeze(0))
    return r_hat


def payload_bytes(packed: dict) -> int:
    """Total payload bytes (rung segments + sink); header excluded."""
    return (sum(int(p.numel()) for p in packed["payload"])
            + int(packed["sink_payload"].numel()))
