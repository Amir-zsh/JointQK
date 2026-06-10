#!/usr/bin/env python3
"""Flat, GPU-ready table format + golden fixtures for the rANS decode kernel."""
import struct
import numpy as np
import rans_interleaved as ri

SB = ri.SCALE_BITS
TOTAL = ri.TOTAL
MASK = ri.MASK
RANS_L = ri.RANS_L


def flatten_rung(models_r):
    d = len(models_r)
    is_const = np.zeros(d, np.uint8); const_val = np.zeros(d, np.int32)
    cdf_off = np.zeros(d + 1, np.int32); vals_off = np.zeros(d + 1, np.int32)
    cdf_parts, vals_parts = [], []; nonconst = []
    for j, m in enumerate(models_r):
        if m.constant:
            is_const[j] = 1; const_val[j] = m.const_val
            cdf_off[j + 1] = cdf_off[j]; vals_off[j + 1] = vals_off[j]
            continue
        nonconst.append(j)
        cdf = m.cdf.astype(np.uint32)
        cdf_parts.append(cdf); cdf_off[j + 1] = cdf_off[j] + cdf.size
        vals_parts.append(m.vals.astype(np.int32)); vals_off[j + 1] = vals_off[j] + m.vals.size
    return dict(d=d, is_const=is_const, const_val=const_val,
                cdf_off=cdf_off, cdf_flat=(np.concatenate(cdf_parts) if cdf_parts else np.zeros(0, np.uint32)),
                vals_off=vals_off, vals_flat=(np.concatenate(vals_parts) if vals_parts else np.zeros(0, np.int32)),
                nonconst=np.array(nonconst, np.int32))


def _cdf_search(cdf, lo, hi, slot):
    a, b = lo, hi - 1
    while a < b:
        mid = (a + b + 1) // 2
        if cdf[mid] <= slot:
            a = mid
        else:
            b = mid - 1
    return a - lo


def decode_page_flat(blob, tab, n, N):
    d = tab["d"]; nonconst = tab["nonconst"]; C = nonconst.size
    cdf = tab["cdf_flat"]; cdf_off = tab["cdf_off"]
    Nh, S = struct.unpack_from("<II", blob, 0)
    assert Nh == N and S == n * C, (Nh, N, S, n * C)
    off = 8
    lane_len = [struct.unpack_from("<I", blob, off + 4 * l)[0] for l in range(N)]
    off += 4 * N
    starts = np.cumsum([0] + lane_len)
    bounds = [round(k * S / N) for k in range(N + 1)]
    pos = np.zeros((n, d), np.int64)
    for j in range(d):
        if tab["is_const"][j]:
            pos[:, j] = 0
    for l in range(N):
        buf = blob[off + starts[l]: off + starts[l + 1]]
        if bounds[l] == bounds[l + 1]:
            continue
        x = struct.unpack_from("<I", buf, 0)[0]; bp = 4
        for k in range(bounds[l], bounds[l + 1]):
            t = k // C; j = int(nonconst[k % C])
            slot = x & MASK
            s = _cdf_search(cdf, int(cdf_off[j]), int(cdf_off[j + 1]), slot)
            base = int(cdf_off[j])
            start = int(cdf[base + s]); freq = int(cdf[base + s + 1]) - start
            x = freq * (x >> SB) + slot - start
            while x < RANS_L:
                x = ((x << 8) | buf[bp]) & 0xFFFFFFFF; bp += 1
            pos[t, j] = s
    return pos


def values_from_pos(pos, tab, n):
    d = tab["d"]; idx = np.zeros((n, d), np.int64)
    for j in range(d):
        if tab["is_const"][j]:
            idx[:, j] = tab["const_val"][j]
        else:
            v = tab["vals_flat"][tab["vals_off"][j]:tab["vals_off"][j + 1]]
            idx[:, j] = v[pos[:, j]]
    return idx


if __name__ == "__main__":
    rng = np.random.default_rng(0); d = 128; P = 64; N = 32
    models = []
    for j in range(d):
        A = int(rng.integers(2, 40)); p = rng.random(A) ** 2; p /= p.sum()
        models.append(ri.FreqTable(np.arange(A) - A // 2, p))
    models[5] = ri.FreqTable(np.array([0]), np.array([1.0]))
    tab = flatten_rung(models)

    def snap(qcol, t):
        if t.constant: return np.zeros(len(qcol), int)
        pp = np.clip(np.searchsorted(t.vals, qcol), 0, t.vals.size - 1)
        ll = np.clip(pp - 1, 0, t.vals.size - 1)
        return np.where(np.abs(t.vals[ll] - qcol) <= np.abs(t.vals[pp] - qcol), ll, pp)

    fixtures = []
    ok = True
    for n in (64, 64, 37, 1):
        posq = np.stack([rng.integers(0, (1 if m.constant else m.vals.size), n) for m in models], 1)
        blob = ri.encode_page(posq, models, n, d, N)
        ref = ri.decode_page(blob, models, n, d)
        got = decode_page_flat(blob, tab, n, N)
        nonc = tab["nonconst"]
        if not np.array_equal(got[:, nonc], ref[:, nonc]) or not np.array_equal(got[:, nonc], posq[:, nonc]):
            ok = False
        fixtures.append((n, blob, posq))
    print("decode_page_flat == decode_page == encoded positions, all pages:", ok)

    np.savez("kernel_fixtures.npz",
             d=d, N=N, SB=SB, TOTAL=TOTAL, RANS_L=RANS_L,
             is_const=tab["is_const"], const_val=tab["const_val"],
             cdf_off=tab["cdf_off"], cdf_flat=tab["cdf_flat"],
             vals_off=tab["vals_off"], vals_flat=tab["vals_flat"], nonconst=tab["nonconst"],
             n_pages=len(fixtures),
             ns=np.array([f[0] for f in fixtures], np.int32),
             blob_lens=np.array([len(f[1]) for f in fixtures], np.int32),
             blobs=np.frombuffer(b"".join(f[1] for f in fixtures), np.uint8),
             expected_pos=np.concatenate([f[2].astype(np.int32).reshape(-1) for f in fixtures]))
    print("wrote kernel_fixtures.npz |", len(fixtures), "pages | C nonconst coords =", tab["nonconst"].size)