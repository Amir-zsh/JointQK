#!/usr/bin/env python3
"""Exp-Golomb-k entropy codec, paged + lane-interleaved to mirror the rANS codec's
GPU decode layout (kvq_codec.py / rans_decode.cu) so the two are directly timeable
against each other: same P (page size), same N (lane count), same n_pages, same
kernel-only measurement methodology.

Unlike rANS, Exp-Golomb needs no probability model / CDF / calibration ladder --
just a Golomb-Rice-style integer parameter k (bigger k for larger-magnitude data).
k can be a single scalar (one value for the whole head) or a per-coordinate array
(d,) -- the latter mirrors rANS's per-coordinate CDF and gets much closer to its
achieved bits/coord, at the cost of one extra table lookup per symbol on decode.

Encode is CPU (Python), matching the rest of this repo's convention that encode
cost is not the thing being measured. Decode is a CUDA kernel (expgolomb_decode.cu)
built the same way as rans_decode.cu via kvq_codec.load_ext(source=..., name=...).
"""
import numpy as np
import torch


def zigzag_encode(s: np.ndarray) -> np.ndarray:
    s = s.astype(np.int64)
    return np.where(s >= 0, 2 * s, -2 * s - 1).astype(np.int64)


def zigzag_decode(u: np.ndarray) -> np.ndarray:
    u = u.astype(np.int64)
    return np.where(u % 2 == 0, u // 2, -(u + 1) // 2)


def _k_from_mean(mean_u: float) -> int:
    if mean_u <= 0:
        return 0
    return max(0, int(round(np.log2(mean_u + 1.0))))


def choose_k(idx: np.ndarray) -> int:
    """Golomb-Rice-style parameter: k ~ log2(mean(zigzag(idx))), clamped >= 0.
    Optimal Golomb parameter for a geometric source has M ~ mean value; Exp-Golomb-k
    approximates a Golomb code of divisor 2^k, so k = round(log2(mean)). One value
    for the whole (T, d) array."""
    u = zigzag_encode(idx.reshape(-1))
    return _k_from_mean(float(u.mean()) if u.size else 0.0)


def choose_k_per_coord(idx: np.ndarray) -> np.ndarray:
    """Same heuristic, one k per coordinate (column) -- mirrors rANS's per-coord
    CDF adaptivity. idx: (T, d). Returns int array (d,)."""
    u = zigzag_encode(idx)
    means = u.mean(axis=0)
    return np.array([_k_from_mean(float(m)) for m in means], dtype=np.int64)


def estimate_bits_per_coord(idx: np.ndarray, k) -> float:
    """Achieved bits/coord for Exp-Golomb-k on idx (T, d), WITHOUT byte-packing --
    just sums the vectorized code-length array. For rate sweeps over many
    candidate quantization steps, where only the rate (not the actual bytes) is
    needed; skips the O(n) Python packing loop."""
    u = zigzag_encode(idx)
    k_arr = np.broadcast_to(np.asarray(k, dtype=np.int64), (idx.shape[1],))
    _, lens = _code_len_and_parts(u, k_arr[None, :])
    return float(lens.sum()) / idx.size


def _code_len_and_parts(u: np.ndarray, k):
    """Vectorized (val, length) for Exp-Golomb-k of unsigned ints u (>=0).
    k: scalar int, or array broadcastable to u.shape (per-symbol k)."""
    k = np.asarray(k, dtype=np.int64)
    high = (u >> k).astype(np.int64)
    low = np.where(k > 0, u & ((np.int64(1) << k) - 1), 0).astype(np.int64)
    m = high + 1
    # floor(log2(m)) via frexp, exact for integers representable in float64 (m << 2**52)
    _, exp = np.frexp(m.astype(np.float64))
    Q = (exp - 1).astype(np.int64)
    length = 2 * Q + 1 + k
    val = (m << k) | low
    return val, length


def _pack_lane(vals: np.ndarray, lens: np.ndarray) -> bytes:
    """Bit-pack a sequence of (value, length) codewords MSB-first into bytes,
    zero-padded at the end to a byte boundary. O(n) rolling 64-bit accumulator."""
    out = bytearray()
    acc = 0
    nbits = 0
    for val, ln in zip(vals.tolist(), lens.tolist()):
        acc = (acc << ln) | val
        nbits += ln
        while nbits >= 8:
            shift = nbits - 8
            out.append((acc >> shift) & 0xFF)
            acc &= (1 << shift) - 1
            nbits -= 8
    if nbits > 0:
        out.append((acc << (8 - nbits)) & 0xFF)
    return bytes(out)


def eg_encode_page_grid(idx: np.ndarray, P: int, N: int, k):
    """idx: (T, d) int array of (already-quantized) signed symbol indices.
    Pads T up to a multiple of P. Within each page, symbols are flattened
    row-major (t*d+j) and interleaved round-robin across N lanes (lane =
    flat_pos % N), matching the rANS codec's lane-interleaving convention.
    P*d must be divisible by N.

    k: int (one value for the whole head) or (d,) array (per-coordinate).

    Returns dict(blob: bytes, page_byte_off: int64[n_pages], lane_off: int64[n_pages,N],
    lane_len: int64[n_pages,N], n_pages, P, N, d, k_percoord: int64[d], T, syms_per_lane).
    """
    T, d = idx.shape
    if (P * d) % N != 0:
        raise ValueError(f"P*d={P*d} must be divisible by N={N}")
    k_percoord = np.broadcast_to(np.asarray(k, dtype=np.int64), (d,)).copy()
    syms_per_lane = P * d // N
    n_pages = (T + P - 1) // P
    pad = n_pages * P - T
    if pad:
        idx_p = np.concatenate([idx, np.zeros((pad, d), dtype=idx.dtype)], axis=0)
    else:
        idx_p = idx
    k_page_flat = np.tile(k_percoord, P)   # (P*d,), aligned to the row-major t*d+j flatten

    blob_parts = []
    page_byte_off = np.zeros(n_pages, dtype=np.int64)
    lane_off = np.zeros((n_pages, N), dtype=np.int64)
    lane_len = np.zeros((n_pages, N), dtype=np.int64)
    cursor = 0
    for p in range(n_pages):
        page = idx_p[p * P:(p + 1) * P].reshape(-1)          # (P*d,) flat, row-major t*d+j
        u = zigzag_encode(page)
        vals, lens = _code_len_and_parts(u, k_page_flat)
        page_byte_off[p] = cursor
        off = 0
        for lane in range(N):
            lane_vals = vals[lane::N]
            lane_lens = lens[lane::N]
            packed = _pack_lane(lane_vals, lane_lens)
            lane_off[p, lane] = off
            lane_len[p, lane] = len(packed)
            blob_parts.append(packed)
            off += len(packed)
        cursor += off
    blob = b"".join(blob_parts)
    return dict(blob=blob, page_byte_off=page_byte_off, lane_off=lane_off,
                lane_len=lane_len, n_pages=n_pages, P=P, N=N, d=d,
                k_percoord=k_percoord, T=T, syms_per_lane=syms_per_lane)


def eg_decode_cpu_reference(enc: dict) -> np.ndarray:
    """Pure-Python/numpy reference decoder (bit-exact match for the CUDA kernel's
    algorithm) -- slow, for correctness validation only."""
    P, N, d, n_pages = enc["P"], enc["N"], enc["d"], enc["n_pages"]
    k_percoord = enc["k_percoord"]
    blob = enc["blob"]
    out = np.zeros((n_pages * P, d), dtype=np.int64)
    for p in range(n_pages):
        page_flat = np.zeros(P * d, dtype=np.int64)
        base = enc["page_byte_off"][p]
        for lane in range(N):
            start = base + enc["lane_off"][p, lane]
            length = enc["lane_len"][p, lane]
            data = blob[start:start + length]
            bits = "".join(f"{byte:08b}" for byte in data)
            pos = 0
            syms = []
            for i in range(enc["syms_per_lane"]):
                flat_pos = i * N + lane
                k = int(k_percoord[flat_pos % d])
                j = pos
                while bits[j] == "0":
                    j += 1
                Q = j - pos
                m = int(bits[j:j + Q + 1], 2)
                j += Q + 1
                low = int(bits[j:j + k], 2) if k else 0
                j += k
                u = ((m - 1) << k) | low
                syms.append(u)
                pos = j
            page_flat[lane::N] = zigzag_decode(np.array(syms, dtype=np.int64))
        out[p * P:(p + 1) * P] = page_flat.reshape(P, d)
    return out[:enc["T"]]


def _decode_pages_raw(blob, page_byte_off, lane_off, lane_len, ks_pagecoord, N, P, d, ext, device):
    """ks_pagecoord: int32[n_pages, d] flattened to [n_pages*d]."""
    blob_t = torch.frombuffer(bytearray(blob), dtype=torch.uint8).to(device)
    page_byte_off_t = torch.as_tensor(page_byte_off, dtype=torch.int64, device=device)
    lane_off_t = torch.as_tensor(lane_off.reshape(-1), dtype=torch.int64, device=device)
    lane_len_t = torch.as_tensor(lane_len.reshape(-1), dtype=torch.int64, device=device)
    ks_t = torch.as_tensor(ks_pagecoord.reshape(-1), dtype=torch.int32, device=device)
    return ext.decode_pages(blob_t, page_byte_off_t, lane_off_t, lane_len_t, ks_t, N, P, d)


def eg_decode_gpu(enc: dict, ext, device="cuda") -> torch.Tensor:
    """Run the compiled expgolomb_decode CUDA extension for a single head's
    encoding. Returns (T, d) int32 tensor on `device` (padded rows dropped)."""
    ks_pagecoord = np.tile(enc["k_percoord"], (enc["n_pages"], 1))
    pos_out = _decode_pages_raw(enc["blob"], enc["page_byte_off"], enc["lane_off"],
                                enc["lane_len"], ks_pagecoord, enc["N"], enc["P"], enc["d"],
                                ext, device)
    return pos_out.reshape(enc["n_pages"] * enc["P"], enc["d"])[:enc["T"]]


def combine_encodings(encs: list):
    """Concatenate several single-head eg_encode_page_grid(...) results (same P, N, d)
    into one merged page set for a SINGLE kernel launch -- matches rANS's
    BatchRANSDecoder.decode_grid convention (one launch over the whole grid, so
    per-page timing reflects steady-state throughput, not per-launch overhead).
    k_percoord may differ per source head."""
    P, N, d = encs[0]["P"], encs[0]["N"], encs[0]["d"]
    assert all(e["P"] == P and e["N"] == N and e["d"] == d for e in encs)
    blob = bytearray()
    page_byte_off = []
    lane_off = []
    lane_len = []
    ks = []
    cursor = 0
    for e in encs:
        page_byte_off.append(e["page_byte_off"] + cursor)
        lane_off.append(e["lane_off"])
        lane_len.append(e["lane_len"])
        ks.append(np.tile(e["k_percoord"], (e["n_pages"], 1)))
        blob += e["blob"]
        cursor += len(e["blob"])
    return dict(blob=bytes(blob), page_byte_off=np.concatenate(page_byte_off),
               lane_off=np.concatenate(lane_off, axis=0), lane_len=np.concatenate(lane_len, axis=0),
               ks=np.concatenate(ks, axis=0), N=N, P=P, d=d, n_pages=sum(e["n_pages"] for e in encs))


def eg_decode_gpu_batch(merged: dict, ext, device="cuda") -> torch.Tensor:
    """Decode a combine_encodings(...) merged page set in ONE kernel launch.
    Returns (n_pages, P, d) int32 tensor (caller slices per-head as needed)."""
    return _decode_pages_raw(merged["blob"], merged["page_byte_off"], merged["lane_off"],
                             merged["lane_len"], merged["ks"], merged["N"], merged["P"],
                             merged["d"], ext, device)
