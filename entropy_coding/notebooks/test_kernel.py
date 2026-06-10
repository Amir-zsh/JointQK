#!/usr/bin/env python3
import struct
import numpy as np
import torch
from torch.utils.cpp_extension import load

f = np.load("kernel_fixtures.npz")
d = int(f["d"]); N = int(f["N"]); P = 64
cdf_flat = f["cdf_flat"].astype(np.int64)
cdf_off = f["cdf_off"].astype(np.int32)
nonconst = f["nonconst"].astype(np.int32); C = nonconst.size
ns = f["ns"].astype(np.int32); n_pages = int(f["n_pages"])
blob_lens = f["blob_lens"].astype(np.int64)
blobs = f["blobs"].astype(np.uint8)
expected = f["expected_pos"].astype(np.int64)
SB = int(f["SB"]) if "SB" in f else 14

page_byte_off = np.concatenate([[0], np.cumsum(blob_lens)])[:-1].astype(np.int64)

lane_off = np.zeros((n_pages, N), np.int64)
k0a = np.zeros((n_pages, N), np.int32); k1a = np.zeros((n_pages, N), np.int32)
for p in range(n_pages):
    base = int(page_byte_off[p]); blob = blobs[base:base + int(blob_lens[p])].tobytes()
    Nh, S = struct.unpack_from("<II", blob, 0)
    assert Nh == N and S == int(ns[p]) * C
    lane_len = [struct.unpack_from("<I", blob, 8 + 4 * l)[0] for l in range(N)]
    starts = np.cumsum([0] + lane_len)
    bounds = [round(k * S / N) for k in range(N + 1)]
    for l in range(N):
        lane_off[p, l] = 8 + 4 * N + int(starts[l])
        k0a[p, l] = bounds[l]; k1a[p, l] = bounds[l + 1]

# ---- build two-level LUT from the fixture cdf (must match the kernel) ----
Lbits = 10
G = 1 << Lbits
lut_shift = SB - Lbits
lut_parts = []
lut_off = np.zeros(d + 1, dtype=np.int32)
for j in range(d):
    lo = int(cdf_off[j]); hi = int(cdf_off[j + 1])
    if hi <= lo:                                   # constant coord: no cdf slice
        lut_off[j + 1] = lut_off[j]; continue
    cdf_j = cdf_flat[lo:hi].astype(np.int64)       # len A+1, last == TOTAL
    A = len(cdf_j) - 1
    starts_g = (np.arange(G, dtype=np.int64) << lut_shift)
    lut = np.clip(np.searchsorted(cdf_j, starts_g, side='right') - 1, 0, A - 1).astype(np.int32)
    lut_parts.append(lut)
    lut_off[j + 1] = lut_off[j] + G
lut_flat = (np.concatenate(lut_parts) if lut_parts else np.zeros(0, np.int32)).astype(np.int32)

dev = "cuda"
ext = load(name="rans_decode", sources=["rans_decode.cu"], verbose=True)

def T(a, dt): return torch.as_tensor(a, dtype=dt, device=dev)
pos = ext.decode_pages(
    T(blobs, torch.uint8), T(page_byte_off, torch.int64), T(lane_off.reshape(-1), torch.int64),
    T(k0a.reshape(-1), torch.int32), T(k1a.reshape(-1), torch.int32), T(ns, torch.int32),
    T(nonconst, torch.int32), T(cdf_flat, torch.int32), T(cdf_off, torch.int32),
    T(lut_flat, torch.int32), T(lut_off, torch.int32),
    N, C, P, d, lut_shift)
pos = pos.cpu().numpy()

# ---- raw kernel output inspection, page 0 ----
print("=== RAW pos_out page0 ===")
print("nonzero entries page0:", int((pos[0] != 0).sum()), "of", pos[0].size)
e0 = expected[:int(ns[0])*d].reshape(int(ns[0]), d)
print("tok0 coords 0..9  kernel:", pos[0,0,:10].tolist())
print("tok0 coords 0..9  expect:", e0[0,:10].tolist())
print("tok1 coords 0..9  kernel:", pos[0,1,:10].tolist())
print("tok1 coords 0..9  expect:", e0[1,:10].tolist())
print("tok63 coords 0..9 kernel:", pos[0,63,:10].tolist())
print("tok63 coords 0..9 expect:", e0[63,:10].tolist())
rows_written = np.where((pos[0] != 0).any(axis=1))[0]
print("tokens with any nonzero:", rows_written.tolist()[:20], "..." if len(rows_written)>20 else "")

off = 0; ok = True
for p in range(n_pages):
    n = int(ns[p])
    exp = expected[off:off + n * d].reshape(n, d); off += n * d
    got = pos[p, :n, :]
    if not np.array_equal(got[:, nonconst], exp[:, nonconst]):
        ok = False
        bad = np.where(got[:, nonconst] != exp[:, nonconst])
        print(f"  page {p}: MISMATCH at {len(bad[0])} positions, first {(bad[0][0], bad[1][0])}")
print("KERNEL bit-exact vs golden fixtures:", ok)