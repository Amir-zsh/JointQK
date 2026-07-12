#!/usr/bin/env python3
"""Per-page decode cost vs per-page fp16 read cost. The single ratio that decides
whether decode-on-read can beat fp16-read. No attention, no harness — just the two
primitive costs. Times all nb pages of a real head in one launch, divides by nb."""
import numpy as np, torch, time, struct, os, shutil
import run_pca_ec_deadzone as base
from kvq_codec import build_codecs_from_ladder_rans_cuda, load_ext, BatchRANSDecoder

dev = torch.device("cuda")
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
decod = BatchRANSDecoder(cod)
art = torch.load(root / man["examples"][4]["file"], map_location="cpu", weights_only=False)
T0 = int(art["prompt_length"]); P = 64

def tile_to(x, T):
    reps = (T + x.shape[0] - 1) // x.shape[0]
    return x.repeat(reps, 1)[:T].contiguous()

def timeit(f, reps=20, warmup=5):
    for _ in range(warmup): f()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(reps): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / reps

print(f"{'T':>8} {'nb':>5} {'decode_ms':>11} {'read_ms':>10} {'dec/read':>9} "
      f"{'per_page_dec_us':>16} {'per_page_read_us':>17} {'C_breakeven':>12}")

for T in [4096, 8192, 16384, 32768]:
    # one representative head, real K tiled to T
    l, h = 1, 0
    c = cod[(l, h)]
    kT = tile_to(art["k_post"][l, h, :T0, :].float(), T)
    buf = c.encode_gpu(kT.cpu().numpy())
    nb = (T + P - 1) // P

    # --- DECODE: full head decode (all nb pages) via decode_grid, one launch ---
    # decode_grid batches; to isolate ONE head's decode, call its single-head decode.
    def run_decode():
        c.decode_to_rhat(bytes(buf))     # rANS decode all nb pages of this head -> r̂

    # --- READ: read the equivalent fp16 K (T x d) from HBM, one pass ---
    # allocate the fp16 cache once (resident), time a bandwidth-bound read of it.
    kfp16 = kT.to(dev).half().contiguous()      # (T, d) fp16, the "decompressed cache"
    sink = torch.zeros(d, device=dev, dtype=torch.float32)
    def run_read():
        # force a full read of all T*d fp16 elements (reduction touches every byte)
        sink.copy_((kfp16.float().sum(dim=0)))   # sum over T -> reads all T*d

    td = timeit(run_decode)
    tr = timeit(run_read)
    per_page_dec = td / nb * 1e6      # us per page
    per_page_read = tr / nb * 1e6     # us per page
    ratio = td / tr if tr > 0 else float('nan')
    # C_breakeven: queries-per-chunk so decode amortizes below read: per_page_dec/C = per_page_read
    C_break = per_page_dec / per_page_read if per_page_read > 0 else float('nan')
    print(f"{T:>8} {nb:>5} {td*1e3:>9.3f}ms {tr*1e3:>8.3f}ms {ratio:>8.1f}x "
          f"{per_page_dec:>14.2f}us {per_page_read:>15.3f}us {C_break:>11.0f}")

print("\ndec/read: how many x slower decode is than fp16 read, per page.")
print("C_breakeven: queries-per-chunk needed so decode amortizes to fp16-read cost.")
print("If C_breakeven <= 64 (page size), chunked decode of a full page wins. If >> 64, wall.")