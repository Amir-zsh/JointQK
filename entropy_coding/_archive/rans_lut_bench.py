#!/usr/bin/env python3
"""Head-to-head: existing rANS decode (binary search over CDF staged in shared
memory, rans_decode.cu) vs a direct per-coordinate slot LUT (rans_decode_lut.cu),
on the IDENTICAL real encoded bytes and CDFs -- same page/lane layout, same data,
same b=2 QPCA quantization, same measurement methodology as kernel_only_micro.py
(per-head kernel launch, CUDA-event kernel-only timing, summed over heads).

Single-rung ladder (m_grid=[1.0]) to sidestep the production rung-selection
machinery, which is orthogonal to the question here (decode ALGORITHM, not rung
choice) -- every page uses the same one CDF per (layer, head, coord), so a LUT
per head is exactly d*16384 entries, no per-rung multiplication.

rans_interleaved.py's own spec comment already flags "the 2^14 LUT is too big at
d=128 distinct tables" as the reason binary search was chosen originally; this
measures whether that tradeoff is actually worth it in wall-clock terms."""
import struct
import numpy as np, torch, time
import run_pca_ec_deadzone as base
from kvq_codec import build_codecs_from_ladder_rans_cuda, load_ext, BatchRANSEncoder

dev = torch.device("cuda")
P = 64
N_LANES = 16
N_HEADS_SAMPLE = 8
SCALE_BITS = 14
TOTAL = 1 << SCALE_BITS

root = base.data_root(); man = base.load_manifest(root)
sq, sk, km, kc, meta = base.calib_moments(root, man, [0, 1, 2])
L, Hkv, d = meta["n_layers"], meta["n_kv_heads"], meta["d_head"]
qu = base.build_qpca_basis(sq, sk); qu["sigma_k"], qu["sigma_q"] = sk, sq
qc = base.build_qpca_basis(sq, kc); qc["sigma_k"] = sk
F, inv = qc["forward"], qc["inverse"]
fc = base._codes_for_idx(root, man, [0, 1, 2], F, km, L, Hkv, d)
_, d0, m0 = base.build_qpca_ec(qc, qu, km, 2, L, Hkv, fc, root, [0, 1, 2],
                               dz=0.375, match_rate=False, uniform_step=True)
ladder = [(1.0, d0, m0)]         # single rung -- isolates the decode algorithm question

bs_ext = load_ext()                                              # existing binary-search kernel
lut_ext = load_ext(source="rans_decode_lut.cu", name="rans_decode_lut")  # new LUT kernel

cod = build_codecs_from_ladder_rans_cuda(F, inv, km, ladder, L, Hkv, 2 * d * 64, P,
                                         0.375, lanes=N_LANES, ext=bs_ext, device="cuda")
enc = BatchRANSEncoder(cod)

art = torch.load(root / man["examples"][4]["file"], map_location="cpu", weights_only=False)
T0 = int(art["prompt_length"])


def tile_to(x, T):
    reps = (T + x.shape[0] - 1) // x.shape[0]
    return x.repeat(reps, 1)[:T].contiguous()


def build_lut(cdf_flat_t: torch.Tensor, cdf_off_t: torch.Tensor, d: int, nonconst_t: torch.Tensor):
    cdf_flat = cdf_flat_t.cpu().numpy(); cdf_off = cdf_off_t.cpu().numpy()
    lut_meta = np.zeros(d * TOTAL, dtype=np.uint32)
    lut_sym = np.zeros(d * TOTAL, dtype=np.int16)
    for j in nonconst_t.cpu().numpy().tolist():
        lo, hi = int(cdf_off[j]), int(cdf_off[j + 1])
        cdf = cdf_flat[lo:hi]
        n_sym = len(cdf) - 1
        base_i = j * TOTAL
        for s in range(n_sym):
            start, end = int(cdf[s]), int(cdf[s + 1])
            freq = end - start
            if freq <= 0:
                continue
            lut_meta[base_i + start: base_i + end] = (start << SCALE_BITS) | freq
            lut_sym[base_i + start: base_i + end] = s
    return (torch.as_tensor(lut_meta, dtype=torch.int32, device=dev),
            torch.as_tensor(lut_sym, dtype=torch.int16, device=dev))


def page_arrays_from_buf(buf: bytes, codec, N):
    """Reproduce PageCodecRANSCUDA.decode_to_gpu's internal per-page k0a/k1a/lane_off
    extraction (single rung here, so exactly one rung group covering all pages)."""
    magic, T, Phdr, nb = struct.unpack_from("<IIII", buf, 0)
    from kvq_codec import _unpack
    rung_ids, off0 = _unpack(buf, nb, codec.id_bits, 16)
    off = off0
    pages = []
    for bi in range(nb):
        blob, off = codec._read_blob(buf, off)
        n = min((bi + 1) * Phdr, T) - bi * Phdr
        pages.append((blob, n))
    ri = int(rung_ids[0])
    nc_t, cf_t, co_t = codec._cuda_cdf_tensors[ri]
    C = int(nc_t.shape[0])
    n_pg = len(pages)

    all_bytes = b"".join(blob for blob, _ in pages)
    blob_arr = np.frombuffer(all_bytes, dtype=np.uint8).copy()
    page_byte_off = np.zeros(n_pg, dtype=np.int64)
    lane_off_arr = np.zeros((n_pg, N), dtype=np.int64)
    k0a = np.zeros((n_pg, N), dtype=np.int32)
    k1a = np.zeros((n_pg, N), dtype=np.int32)
    ns_arr = np.zeros(n_pg, dtype=np.int32)
    cumoff = 0
    for pi, (blob, n_toks) in enumerate(pages):
        S = n_toks * C
        lane_lens = [struct.unpack_from("<I", blob, 8 + 4 * l)[0] for l in range(N)]
        starts = np.cumsum([0] + lane_lens)
        bounds = [round(k * S / N) for k in range(N + 1)]
        page_byte_off[pi] = cumoff
        ns_arr[pi] = n_toks
        header_size = 8 + 4 * N
        for l in range(N):
            lane_off_arr[pi, l] = header_size + int(starts[l])
            k0a[pi, l] = bounds[l]
            k1a[pi, l] = bounds[l + 1]
        cumoff += len(blob)
    return dict(blob_arr=blob_arr, page_byte_off=page_byte_off, lane_off=lane_off_arr,
               k0a=k0a, k1a=k1a, ns=ns_arr, nc_t=nc_t, cf_t=cf_t, co_t=co_t, C=C, n_pg=n_pg)


def timeit_ev(f, reps=20, warmup=5):
    for _ in range(warmup):
        f()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(reps):
        f()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps  # ms


heads = [(l, h) for l in range(1, L) for h in range(Hkv)][:N_HEADS_SAMPLE]


def run_sweep():
    print(f"rANS decode: binary-search (existing) vs direct slot LUT (new), "
          f"P={P}, lanes={N_LANES}, {N_HEADS_SAMPLE} heads, single rung (b=2).\n")
    print(f"{'T':>8} {'n_pages':>9} {'bs_us/page':>12} {'lut_us/page':>12} {'lut/bs':>8} {'lut_mem/head':>13}")
    for T in [4096, 8192, 16384, 32768]:
        _run_one_T(T)


def _run_one_T(T):
    kg = {(l, h): tile_to(art["k_post"][l, h, :T0, :].float(), T) for (l, h) in heads}
    bufs = enc.encode_grid(kg)

    per_head = []
    for (l, h) in heads:
        pa = page_arrays_from_buf(bufs[(l, h)], cod[(l, h)], N_LANES)
        lut_meta, lut_sym = build_lut(pa["cf_t"], pa["co_t"], d, pa["nc_t"])
        per_head.append((pa, lut_meta, lut_sym))

    def _T(a, dt): return torch.as_tensor(a, dtype=dt, device=dev)

    def decode_bs():
        for pa, _, _ in per_head:
            bs_ext.decode_pages(
                _T(pa["blob_arr"], torch.uint8), _T(pa["page_byte_off"], torch.int64),
                _T(pa["lane_off"].reshape(-1), torch.int64), _T(pa["k0a"].reshape(-1), torch.int32),
                _T(pa["k1a"].reshape(-1), torch.int32), _T(pa["ns"], torch.int32),
                pa["nc_t"], pa["cf_t"], pa["co_t"], N_LANES, pa["C"], P, d)

    def decode_lut():
        for pa, lut_meta, lut_sym in per_head:
            lut_ext.decode_pages_lut(
                _T(pa["blob_arr"], torch.uint8), _T(pa["page_byte_off"], torch.int64),
                _T(pa["lane_off"].reshape(-1), torch.int64), _T(pa["k0a"].reshape(-1), torch.int32),
                _T(pa["k1a"].reshape(-1), torch.int32), pa["nc_t"], lut_meta, lut_sym,
                N_LANES, pa["C"], P, d)

    t_bs = timeit_ev(decode_bs)
    t_lut = timeit_ev(decode_lut)
    n_pages = sum(pa["n_pg"] for pa, _, _ in per_head)
    us_bs = (t_bs / 1e3) / n_pages * 1e6
    us_lut = (t_lut / 1e3) / n_pages * 1e6
    lut_mem_mb = d * TOTAL * (4 + 2) / 1e6
    print(f"{T:>8} {n_pages:>9} {us_bs:>10.3f}us {us_lut:>10.3f}us {us_lut/us_bs:>7.2f}x "
          f"{lut_mem_mb:>10.1f}MB")


if __name__ == "__main__":
    run_sweep()
    print("\nlut/bs: >1 => LUT is SLOWER than binary search; <1 => LUT wins.")
    print("lut_mem/head: per-head LUT footprint (d * 2^14 slots * 6 bytes/entry) -- "
          "confirms rans_interleaved.py's memory-cost flag on this parameter regime.")
