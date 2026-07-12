#!/usr/bin/env python3
"""Stage 1b: in-kernel vals-map + dequant -> r̂, vs decode_to_idx dequant (host)."""
import numpy as np, torch, struct, os, shutil
from collections import defaultdict
from torch.utils.cpp_extension import load
import run_pca_ec_deadzone as base
from kvq_codec import build_codecs_from_ladder_rans_cuda, load_ext, BatchRANSEncoder, _unpack

shutil.rmtree("./build_s1b", ignore_errors=True); os.makedirs("./build_s1b", exist_ok=True)
s1b = load(name="fused_s1b", sources=["fused_s1b.cu"], build_directory="./build_s1b", verbose=True)

root = base.data_root(); man = base.load_manifest(root)
sq, sk, km, kc, meta = base.calib_moments(root, man, [0, 1, 2])
L, Hkv, d = meta["n_layers"], meta["n_kv_heads"], meta["d_head"]
qu = base.build_qpca_basis(sq, sk); qu["sigma_k"], qu["sigma_q"] = sk, sq
qc = base.build_qpca_basis(sq, kc); qc["sigma_k"] = sk
F, inv = qc["forward"], qc["inverse"]
fc = base._codes_for_idx(root, man, [0, 1, 2], F, km, L, Hkv, d)
_, d0, m0 = base.build_qpca_ec(qc, qu, km, 2, L, Hkv, fc, root, [0, 1, 2],
                               dz=0.375, match_rate=False, uniform_step=True)
ladder = [(1.0, d0, m0)]
for m in [1.05, 1.1, 1.25, 1.5]:
    dm = (d0 * m).float()
    ladder.append((m, dm, base.freeze_coder_model(fc, dm, L, Hkv, d, 0.375)))
ext = load_ext()
cod = build_codecs_from_ladder_rans_cuda(F, inv, km, ladder, L, Hkv, 2*d*64, 64, 0.375,
                                         lanes=16, ext=ext, device="cuda")
enc = BatchRANSEncoder(cod)
art = torch.load(root / man["examples"][4]["file"], map_location="cpu", weights_only=False)
T = int(art["prompt_length"])
kg = {(l, h): art["k_post"][l, h, :T, :].float() for l in range(L) for h in range(Hkv)}
bufs = enc.encode_grid(kg)

c = cod[(1, 0)]; dev = c.device
buf = bytes(bufs[(1, 0)])
magic, Tt, Phdr, nb = struct.unpack_from("<IIII", buf, 0)
rung_ids, off0 = _unpack(buf, nb, c.id_bits, 16)
blobs_meta = []; off = off0
for bi in range(nb):
    blob, off = c._read_blob(buf, off)
    n = min((bi + 1) * Phdr, Tt) - bi * Phdr
    blobs_meta.append((blob, int(rung_ids[bi]), n))
rung_groups = defaultdict(list)
for bi, (blob, ri, n) in enumerate(blobs_meta):
    rung_groups[ri].append((bi, blob, n))

# host reference r̂ for the whole head (trusted path)
r_ref_full = c.decode_to_rhat(buf)[:T].float().cpu().numpy()

max_err = 0.0
for ri, pages in rung_groups.items():
    nc_t, cf_t, co_t = c._cuda_cdf_tensors[ri]
    C = int(nc_t.shape[0]); n_pg = len(pages)
    all_bytes = b"".join(blob for _, blob, _ in pages)
    blob_arr = np.frombuffer(all_bytes, dtype=np.uint8).copy()
    page_byte_off = np.zeros(n_pg, np.int64); lane_off_arr = np.zeros((n_pg, c.N), np.int64)
    k0a = np.zeros((n_pg, c.N), np.int32); k1a = np.zeros((n_pg, c.N), np.int32)
    ns_arr = np.zeros(n_pg, np.int32); page_bi = []
    cumoff = 0
    for pi, (bi, blob, n_toks) in enumerate(pages):
        S = n_toks * C
        lane_lens = [struct.unpack_from("<I", blob, 8 + 4*l)[0] for l in range(c.N)]
        starts = np.cumsum([0] + lane_lens); bounds = [round(k*S/c.N) for k in range(c.N+1)]
        page_byte_off[pi] = cumoff; ns_arr[pi] = n_toks; page_bi.append(bi)
        hs = 8 + 4*c.N
        for l in range(c.N):
            lane_off_arr[pi, l] = hs + int(starts[l]); k0a[pi, l] = bounds[l]; k1a[pi, l] = bounds[l+1]
        cumoff += len(blob)
    vals_t = c._vals_gpu[ri].to(torch.int64).contiguous()   # (d, A_max)
    A_max = vals_t.shape[1]
    cv_t = c._cv_gpu[ri].to(torch.int32).contiguous()
    ncm = c._nc_mask_gpu[ri].to(torch.uint8).contiguous()
    delta = float(c._deltas_gpu[ri].reshape(-1)[0])
    def _T(a, dt): return torch.as_tensor(a, dtype=dt, device=dev)
    rhat_pg = s1b.s1b_decode(
        _T(blob_arr, torch.uint8), _T(page_byte_off, torch.int64),
        _T(lane_off_arr.reshape(-1), torch.int64), _T(k0a.reshape(-1), torch.int32),
        _T(k1a.reshape(-1), torch.int32), _T(ns_arr, torch.int32),
        nc_t, cf_t, co_t, vals_t, cv_t, ncm,
        delta, 0.375, A_max, c.N, C, c.P, c.d)        # (n_pg, P, d)
    for pi, bi in enumerate(page_bi):
        s0 = bi * c.P; n = ns_arr[pi]
        got = rhat_pg[pi, :n].cpu().numpy()
        ref = r_ref_full[s0:s0+n]
        max_err = max(max_err, float(np.abs(got - ref).max()))

print(f"\nstage1b in-kernel r̂ vs host decode_to_rhat  max|delta| = {max_err:.3e}")
print("PASS if ~1e-4 — vals-map + dequant in SRAM correct; ready for stage 2 (attention in-block).")