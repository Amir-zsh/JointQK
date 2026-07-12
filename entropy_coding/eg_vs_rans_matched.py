#!/usr/bin/env python3
"""Apples-to-apples re-check of the rANS vs Exp-Golomb throughput numbers used in
notes/entropy_coding_throughput_report.md. Two things were NOT controlled for in
the earlier expgolomb_bench.py / expgolomb_rate_match.py runs, both fixed here:

  1. Scale: rANS's reference (3.88 us/page) was measured across the ~full 280-head
     grid (kernel_only_micro.py); Exp-Golomb's numbers used an 8-head sample. If
     GPU memory-bandwidth contention or L2 cache locality changes with dataset
     size, an 8-head sample could be a biased (likely optimistic) estimate of the
     full-grid steady-state cost.
  2. Launch strategy: rANS's reference sums many per-head kernel launches (each
     individually CUDA-event-bracketed, matching PageCodecRANSCUDA.decode_to_gpu);
     Exp-Golomb's numbers used ONE combined launch across all sampled heads.
     Both are legitimate "kernel-only" measurements if bracketed correctly, but
     using the SAME strategy removes any doubt.

This script uses the FULL ~280-head grid (layers 1..L-1, matching this repo's
layer-0-excluded convention) and per-head CUDA-event-bracketed launches for BOTH
codecs, so every number is measured the identical way."""
import struct
import numpy as np, torch, time
import run_pca_ec_deadzone as base
from kvq_codec import build_codecs_from_ladder_rans_cuda, load_ext, BatchRANSEncoder
from expgolomb_codec import eg_encode_page_grid, choose_k_per_coord

dev = torch.device("cuda")
P = 64
N_LANES = 16
T_BENCH = 16384          # already established saturating size in prior runs

root = base.data_root(); man = base.load_manifest(root)
sq, sk, km, kc, meta = base.calib_moments(root, man, [0, 5, 6])  # 8234-tok single-task (qasper) calib, matched to OSCAR's ~8k default
L, Hkv, d = meta["n_layers"], meta["n_kv_heads"], meta["d_head"]
qu = base.build_qpca_basis(sq, sk); qu["sigma_k"], qu["sigma_q"] = sk, sq
qc = base.build_qpca_basis(sq, kc); qc["sigma_k"] = sk
F, inv = qc["forward"], qc["inverse"]
fc_calib = base._codes_for_idx(root, man, [0, 1, 2], F, km, L, Hkv, d)
_, delta0, model0 = base.build_qpca_ec(qc, qu, km, 2, L, Hkv, fc_calib, root, [0, 1, 2],
                                       dz=0.375, match_rate=False, uniform_step=True)
ladder = [(1.0, delta0, model0)]   # single rung -- matches rans_lut_bench.py's convention

bs_ext = load_ext()
eg_ext = load_ext(source="expgolomb_decode.cu", name="expgolomb_decode")

cod = build_codecs_from_ladder_rans_cuda(F, inv, km, ladder, L, Hkv, 2 * d * 64, P,
                                         0.375, lanes=N_LANES, ext=bs_ext, device="cuda")
enc = BatchRANSEncoder(cod)

art = torch.load(root / man["examples"][4]["file"], map_location="cpu", weights_only=False)
T0 = int(art["prompt_length"])
heads = [(l, h) for l in range(1, L) for h in range(Hkv)]   # FULL grid, layer 0 excluded
print(f"Full grid: {len(heads)} heads (L={L}, Hkv={Hkv}, layer 0 excluded), T={T_BENCH}, P={P}\n")


def tile_to(x, T):
    reps = (T + x.shape[0] - 1) // x.shape[0]
    return x.repeat(reps, 1)[:T].contiguous()


def page_arrays_from_buf(buf: bytes, codec, N):
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


def _T(a, dt): return torch.as_tensor(a, dtype=dt, device=dev)


# ---------------------------------------------------------------------------
# rANS: full grid, per-head launches, CUDA-event kernel-only timing (SAME
# methodology PageCodecRANSCUDA.decode_to_gpu uses for the 3.88us/page reference).
# ---------------------------------------------------------------------------
print("Encoding + timing rANS (full grid, per-head launches)...")
t0 = time.time()
kg = {(l, h): tile_to(art["k_post"][l, h, :T0, :].float(), T_BENCH) for (l, h) in heads}
bufs = enc.encode_grid(kg)
print(f"  rANS GPU encode: {time.time()-t0:.1f}s")

pas = {}
for (l, h) in heads:
    pas[(l, h)] = page_arrays_from_buf(bufs[(l, h)], cod[(l, h)], N_LANES)

for _ in range(3):   # warmup
    for (l, h) in heads[:4]:
        pa = pas[(l, h)]
        bs_ext.decode_pages(_T(pa["blob_arr"], torch.uint8), _T(pa["page_byte_off"], torch.int64),
                            _T(pa["lane_off"].reshape(-1), torch.int64), _T(pa["k0a"].reshape(-1), torch.int32),
                            _T(pa["k1a"].reshape(-1), torch.int32), _T(pa["ns"], torch.int32),
                            pa["nc_t"], pa["cf_t"], pa["co_t"], N_LANES, pa["C"], P, d)
torch.cuda.synchronize()

kernel_events = []
for (l, h) in heads:
    pa = pas[(l, h)]
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    bs_ext.decode_pages(_T(pa["blob_arr"], torch.uint8), _T(pa["page_byte_off"], torch.int64),
                        _T(pa["lane_off"].reshape(-1), torch.int64), _T(pa["k0a"].reshape(-1), torch.int32),
                        _T(pa["k1a"].reshape(-1), torch.int32), _T(pa["ns"], torch.int32),
                        pa["nc_t"], pa["cf_t"], pa["co_t"], N_LANES, pa["C"], P, d)
    e.record()
    kernel_events.append((s, e))
torch.cuda.synchronize()
rans_kernel_ms = sum(s.elapsed_time(e) for s, e in kernel_events)
rans_n_pages = sum(pa["n_pg"] for pa in pas.values())
rans_us_per_page = (rans_kernel_ms / 1e3) / rans_n_pages * 1e6
print(f"  rANS: {rans_n_pages} pages, {rans_kernel_ms:.2f}ms total kernel time, "
      f"{rans_us_per_page:.3f} us/page  (prior reference: 3.88 us/page)\n")


# ---------------------------------------------------------------------------
# Exp-Golomb: SAME full grid, SAME per-head-launch methodology, both at delta0
# (m=1.0, full precision) and delta0*1.75 (rate-matched to rANS).
# ---------------------------------------------------------------------------
def eg_run(m, label):
    print(f"Encoding + timing Exp-Golomb @ m={m:g} ({label}), full grid, per-head launches...")
    t0 = time.time()
    encs = {}
    total_bits = 0
    total_coords = 0
    for (l, h) in heads:
        kreal = tile_to(art["k_post"][l, h, :T0, :].float(), T_BENCH)
        r = (kreal.double() - km[l, h].double()) @ F[l, h].double()
        delta_m = (delta0[l, h].double() * m).clamp_min(1e-12)
        idx = base._dz_round(r, delta_m, 0.375).long().numpy()
        k = choose_k_per_coord(idx)
        enc_ = eg_encode_page_grid(idx, P, N_LANES, k)
        encs[(l, h)] = enc_
        total_bits += len(enc_["blob"]) * 8
        total_coords += T_BENCH * d
    t_encode = time.time() - t0
    print(f"  CPU encode: {t_encode:.1f}s, achieved bpc: {total_bits/total_coords:.3f}")

    def _decode_pages_raw(enc_):
        ks_pagecoord = np.tile(enc_["k_percoord"], (enc_["n_pages"], 1))
        blob_t = torch.frombuffer(bytearray(enc_["blob"]), dtype=torch.uint8).to(dev)
        page_byte_off_t = torch.as_tensor(enc_["page_byte_off"], dtype=torch.int64, device=dev)
        lane_off_t = torch.as_tensor(enc_["lane_off"].reshape(-1), dtype=torch.int64, device=dev)
        lane_len_t = torch.as_tensor(enc_["lane_len"].reshape(-1), dtype=torch.int64, device=dev)
        ks_t = torch.as_tensor(ks_pagecoord.reshape(-1), dtype=torch.int32, device=dev)
        return blob_t, page_byte_off_t, lane_off_t, lane_len_t, ks_t

    gpu_bufs = {hh: _decode_pages_raw(encs[hh]) for hh in heads}

    for _ in range(3):
        for hh in heads[:4]:
            blob_t, pbo, lo, ll, ks_t = gpu_bufs[hh]
            eg_ext.decode_pages(blob_t, pbo, lo, ll, ks_t, N_LANES, P, d)
    torch.cuda.synchronize()

    kevents = []
    for hh in heads:
        blob_t, pbo, lo, ll, ks_t = gpu_bufs[hh]
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        eg_ext.decode_pages(blob_t, pbo, lo, ll, ks_t, N_LANES, P, d)
        e.record()
        kevents.append((s, e))
    torch.cuda.synchronize()
    eg_kernel_ms = sum(s.elapsed_time(e) for s, e in kevents)
    eg_n_pages = sum(encs[hh]["n_pages"] for hh in heads)
    eg_us_per_page = (eg_kernel_ms / 1e3) / eg_n_pages * 1e6
    print(f"  Exp-Golomb @ m={m:g}: {eg_n_pages} pages, {eg_kernel_ms:.2f}ms total kernel time, "
          f"{eg_us_per_page:.3f} us/page\n")
    return eg_us_per_page


eg_full_precision_us = eg_run(1.0, "full precision, ~2.77 bpc")
eg_matched_us = eg_run(1.75, "rate-matched to rANS, ~2.1 bpc")

print("=" * 78)
print("SUMMARY (full ~280-head grid, matched per-head-launch methodology both codecs)")
print(f"  rANS                        : {rans_us_per_page:.3f} us/page  (8-head-sample/prior ref was 3.88)")
print(f"  Exp-Golomb (full precision) : {eg_full_precision_us:.3f} us/page  (8-head-sample was ~0.91)")
print(f"  Exp-Golomb (rate-matched)   : {eg_matched_us:.3f} us/page  (8-head-sample was ~0.67)")
print(f"  Exp-Golomb matched speedup vs rANS: {rans_us_per_page/eg_matched_us:.2f}x "
      f"(8-head-sample estimate was 5.8x)")
