#!/usr/bin/env python3
"""Interleaved rANS reference (numpy) for GPU KV-cache entropy coding.

Defines the bitstream format a CUDA kernel must reproduce. Range/arithmetic
coding is serial (each symbol depends on the prior renorm state), so GPU speed
comes from many INDEPENDENT streams, not a faster single stream. This reference
fixes those streams exactly and is bit-exact round-trip, so "fast on GPU" reduces
to porting a spec that already works.

Why this fits your codec:
  - models are STATIC (calib-frozen per coord) -> no adaptive context to serialize;
  - pages are INDEPENDENT (random access) -> one warp/block per page;
  - one rung per page -> the whole page shares one rung's d tables (table id per
    symbol is its coord position, fully static -- no branching to pick a model).

PARALLELISM (two levels):
  across pages : ~10^5 pages -> block per page (the bulk of the win)
  within page  : split a page's S symbols into N independent lanes (e.g. N=32 = a
                 warp) -> per-thread serial chain shrinks to ~S/N symbols.

BITSTREAM SPEC (this is the contract for the kernel)
----------------------------------------------------
rANS params: 32-bit state, byte (8-bit) renorm, SCALE_BITS=12, RANS_L=1<<23.
Per page of n tokens x d coords:
  symbol order : token-major (t outer, coord j inner), CONSTANT coords omitted.
  lane split   : S symbols into N contiguous lanes, bounds[k]=round(k*S/N).
                 lane l owns symbols [bounds[l], bounds[l+1]).
  per lane     : independent rANS. Encode processes the lane's symbols in REVERSE,
                 emits renorm bytes, then flushes the 4-byte final state; the lane
                 byte buffer is read FORWARD on decode (state first). LE byte order.
  freq table   : per (rung, coord): integer freqs summing to 2^SCALE_BITS (>=1 for
                 every present symbol), exclusive-prefix cdf. slot->symbol by cdf
                 search (use alias/binary search on GPU; the 2^SCALE_BITS LUT is too
                 big at d=128 distinct tables -- alphabets are tiny so cdf search is
                 ~4 steps).
  page blob    : <u32 N><u32 S><u32 lane_len[0..N-1]><lane bytes 0..N-1 concatenated>

The constant coords carry 0 bits (decoder fills the single value). The decoder
reconstructs the symbol order from (n, d, constant-mask), so no per-symbol coord
ids are stored. Lanes are independently decodable from their byte slice (offsets
are the prefix sums of lane_len) -- that is the GPU thread decomposition.

NOTE: a coalesced kernel (DietGPU-style) may interleave lane WORDS in memory so a
warp's reads coalesce; that is a memory-layout optimization on top of the SAME
per-lane stream semantics defined here. This reference uses the simple concatenated
layout; the kernel must preserve per-lane content, not necessarily byte placement.
"""

import struct
import numpy as np

RANS_L = 1 << 23
SCALE_BITS = 14
TOTAL = 1 << SCALE_BITS
MASK = TOTAL - 1

from numba import njit

@njit(cache=True)
def _encode_lane_nb(freqs, starts):
    # freqs, starts: int64 arrays in FORWARD symbol order. Returns uint8 array:
    # 4-byte LE state first, then renorm bytes in decode order. Byte-identical to encode_lane.
    n = freqs.shape[0]
    x = RANS_L
    buf = np.empty(n * 2 + 8, np.uint8)   # worst case <=1 renorm byte/sym + 4 state; pad
    top = buf.shape[0]
    w = top                               # write renorm bytes downward from the end
    for i in range(n - 1, -1, -1):
        f = freqs[i]; s = starts[i]
        x_max = ((RANS_L >> SCALE_BITS) << 8) * f
        while x >= x_max:
            w -= 1; buf[w] = x & 0xFF; x >>= 8
        x = ((x // f) << SCALE_BITS) + (x % f) + s
    # state (LE) then the renorm bytes in forward (decode) order
    out = np.empty(4 + (top - w), np.uint8)
    out[0] = x & 0xFF
    out[1] = (x >> 8) & 0xFF
    out[2] = (x >> 16) & 0xFF
    out[3] = (x >> 24) & 0xFF
    out[4:] = buf[w:top]
    return out

def normalize_freqs(p):
    p = np.asarray(p, np.float64); p = p / p.sum()
    f = np.maximum(1, np.round(p * TOTAL).astype(np.int64))
    diff = TOTAL - int(f.sum())
    if diff:
        order = np.argsort(-f); i = 0; step = 1 if diff > 0 else -1
        while diff:
            j = order[i % len(order)]
            if f[j] + step >= 1:
                f[j] += step; diff -= step
            i += 1
    assert f.sum() == TOTAL and f.min() >= 1
    return f


class FreqTable:
    """One (rung, coord). Static frequencies for interleaved rANS."""
    def __init__(self, vals, p):
        self.vals = np.asarray(vals, np.int64)
        self.constant = self.vals.size <= 1
        if self.constant:
            self.const_val = int(self.vals[0]) if self.vals.size else 0
            return
        self.freq = normalize_freqs(p)
        self.cdf = np.concatenate([[0], np.cumsum(self.freq)]).astype(np.int64)  # len A+1

    def slot2sym(self, slot):
        return int(np.searchsorted(self.cdf, slot, side="right") - 1)


# --- single lane (serial rANS); the per-thread kernel body ------------------
def encode_lane(freq_start):
    """freq_start: list of (freq, start) in FORWARD symbol order. Returns bytes
    (state first, then renorm bytes in decode order)."""
    x = RANS_L
    out = bytearray()
    for freq, start in reversed(freq_start):
        x_max = ((RANS_L >> SCALE_BITS) << 8) * freq
        while x >= x_max:
            out.append(x & 0xFF); x >>= 8
        x = ((x // freq) << SCALE_BITS) + (x % freq) + start
    state = struct.pack("<I", x)              # 4-byte final state
    out.reverse()
    return state + bytes(out)


def decode_lane(buf, tables):
    """buf: lane bytes. tables: list of FreqTable in FORWARD order (one per symbol).
    Returns list of symbol positions."""
    x = struct.unpack_from("<I", buf, 0)[0]
    pos = 4
    syms = []
    for t in tables:
        slot = x & MASK
        s = t.slot2sym(slot)
        freq = int(t.freq[s]); start = int(t.cdf[s])
        x = freq * (x >> SCALE_BITS) + slot - start
        while x < RANS_L:
            x = ((x << 8) | buf[pos]) & 0xFFFFFFFF; pos += 1
        syms.append(s)
    return syms


# --- page = N interleaved lanes ---------------------------------------------
def _page_order(n, d, models_r):
    """Deterministic (t,j) symbol order: token-major, skip constant coords."""
    return [(t, j) for t in range(n) for j in range(d) if not models_r[j].constant]


def _lane_bounds(S, N):
    return [round(k * S / N) for k in range(N + 1)]


def encode_page(positions, models_r, n, d, N):
    """positions:(n,d) int alphabet positions. Returns the page blob (bytes)."""
    order = _page_order(n, d, models_r)
    S = len(order)
    b = _lane_bounds(S, N)
    # gather (freq,start) for every symbol in order, vectorized per coord
    ts = np.fromiter((t for (t, _) in order), np.int64, S)
    js = np.fromiter((j for (_, j) in order), np.int64, S)
    freqs = np.empty(S, np.int64); starts = np.empty(S, np.int64)
    for j in range(d):
        m = models_r[j]
        if m.constant:
            continue
        sel = js == j
        pos_j = positions[ts[sel], j]
        freqs[sel] = m.freq[pos_j]
        starts[sel] = m.cdf[pos_j]
    blobs = [_encode_lane_nb(freqs[b[l]:b[l+1]], starts[b[l]:b[l+1]]) for l in range(N)]
    head = struct.pack("<II", N, S) + b"".join(struct.pack("<I", len(x)) for x in blobs)
    return head + b"".join(bytes(x) for x in blobs)

def decode_page(blob, models_r, n, d):
    """Returns positions:(n,d). Lanes are decoded independently (parallelizable)."""
    N, S = struct.unpack_from("<II", blob, 0)
    off = 8
    lane_len = [struct.unpack_from("<I", blob, off + 4 * l)[0] for l in range(N)]
    off += 4 * N
    order = _page_order(n, d, models_r)
    b = _lane_bounds(S, N)
    positions = np.zeros((n, d), np.int64)            # constant coords -> pos 0
    starts = np.cumsum([0] + lane_len)
    for l in range(N):                                # INDEPENDENT: any order / parallel
        seg = order[b[l]:b[l+1]]
        tabs = [models_r[j] for (_, j) in seg]
        lane_buf = blob[off + starts[l]: off + starts[l+1]]
        syms = decode_lane(lane_buf, tabs)
        for (t, j), s in zip(seg, syms):
            positions[t, j] = s
    return positions


def positions_to_values(positions, models_r, n, d):
    """Map alphabet positions back to quantization index values (decode tail)."""
    idx = np.zeros((n, d), np.int64)
    for j in range(d):
        t = models_r[j]
        idx[:, j] = t.const_val if t.constant else t.vals[positions[:, j]]
    return idx