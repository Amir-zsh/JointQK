#!/usr/bin/env python3
"""Kernel-only per-page rANS decode cost (entropy kernel ONLY, no parse/gather/dequant/
inverse), vs per-page fp16 read. The honest bandwidth-relevant number: the fused attn
kernel does dequant in SRAM, so the entropy stream only costs the decode kernel itself.
Uses decode_grid's internal 'entropy kernel' timing / total pages."""
import numpy as np, torch, time, io, contextlib, re
import run_pca_ec_deadzone as base
from kvq_codec import build_codecs_from_ladder_rans_cuda, load_ext, BatchRANSEncoder, BatchRANSDecoder

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
enc = BatchRANSEncoder(cod); decod = BatchRANSDecoder(cod)
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

# capture decode_grid's printed "entropy kernel X ms" line
def grid_entropy_ms(bufs):
    s = io.StringIO()
    with contextlib.redirect_stdout(s):
        decod.decode_grid(bufs)
    torch.cuda.synchronize()
    out = s.getvalue()
    m = re.search(r"entropy kernel\s+([\d.]+)\s*ms", out)
    return float(m.group(1)) if m else None

print(f"{'T':>8} {'nb/head':>8} {'tot_pages':>10} {'ent_kernel_ms':>14} "
      f"{'per_page_dec_us':>16} {'per_page_read_us':>17} {'dec/read':>9} {'C_break':>8}")

for T in [4096, 8192, 16384, 32768]:
    # build a full-grid encode at this T (tile every head's real K)
    kg = {}
    for l in range(1, L):
        for h in range(Hkv):
            kg[(l, h)] = tile_to(art["k_post"][l, h, :T0, :].float(), T)
    bufs = enc.encode_grid(kg)
    n_heads = len(kg)
    nb_head = (T + P - 1) // P
    tot_pages = n_heads * nb_head

    # warm + median of a few entropy-kernel readings
    vals = []
    for _ in range(5):
        e = grid_entropy_ms(bufs)
        if e is not None: vals.append(e)
    ent_ms = float(np.median(vals)) if vals else float('nan')
    per_page_dec = (ent_ms / 1e3) / tot_pages * 1e6      # us/page, kernel only

    # fp16 read per page: read T*d fp16 for ONE head, /nb_head
    kfp16 = tile_to(art["k_post"][1, 0, :T0, :].float(), T).to(dev).half().contiguous()
    sink = torch.zeros(d, device=dev)
    def run_read(): sink.copy_(kfp16.float().sum(dim=0))
    tr = timeit(run_read)
    per_page_read = tr / nb_head * 1e6                    # us/page

    ratio = per_page_dec / per_page_read if per_page_read > 0 else float('nan')
    C_break = ratio
    print(f"{T:>8} {nb_head:>8} {tot_pages:>10} {ent_ms:>12.2f}ms "
          f"{per_page_dec:>14.3f}us {per_page_read:>15.3f}us {ratio:>8.1f}x {C_break:>7.0f}")

print("\nper_page_dec: ENTROPY KERNEL ONLY (no parse/gather/dequant/inverse).")
print("C_break: queries/chunk for decode to amortize to fp16-read. <=64 => page-chunk wins.")