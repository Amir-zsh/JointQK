#!/usr/bin/env python3
"""Separate fixed overhead from marginal per-page decode cost in the entropy kernel.
Hold T constant (identical per-page work + tables), vary #heads -> vary #pages only.
Fit entropy_kernel_ms = a + b*pages. a = fixed launch/setup; b = marginal per-page decode."""
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

def grid_entropy_ms(bufs):
    s = io.StringIO()
    with contextlib.redirect_stdout(s):
        decod.decode_grid(bufs)
    torch.cuda.synchronize()
    m = re.search(r"entropy kernel\s+([\d.]+)\s*ms", s.getvalue())
    return float(m.group(1)) if m else None

T = 32768                       # fix T -> identical per-page work
nb_head = (T + P - 1) // P       # pages per head
all_heads = [(l, h) for l in range(1, L) for h in range(Hkv)]

# vary number of heads -> vary total pages, same per-page work
head_counts = [10, 40, 80, 140, 200, 280]
xs, ys = [], []
print(f"fixed T={T}, nb/head={nb_head}")
print(f"{'#heads':>7} {'pages':>9} {'ent_ms(med)':>12}")
for nh in head_counts:
    heads = all_heads[:nh]
    kg = {(l, h): tile_to(art["k_post"][l, h, :T0, :].float(), T) for (l, h) in heads}
    bufs = enc.encode_grid(kg)
    vals = []
    for _ in range(5):
        e = grid_entropy_ms(bufs)
        if e is not None: vals.append(e)
    ent = float(np.median(vals))
    pages = nh * nb_head
    xs.append(pages); ys.append(ent)
    print(f"{nh:>7} {pages:>9} {ent:>10.2f}ms")

xs = np.array(xs, float); ys = np.array(ys, float)
# linear fit ent_ms = a + b*pages
b, a = np.polyfit(xs, ys, 1)   # b = slope (ms/page), a = intercept (ms)
b_us = b * 1e3                  # us/page marginal
print(f"\nfit: ent_ms = {a:.2f}ms (fixed) + {b_us:.4f}us/page (marginal) * pages")
print(f"  fixed overhead a   = {a:.1f} ms")
print(f"  marginal per page b = {b_us:.4f} us/page")

# compare marginal to fp16 read budget at this T
kfp16 = tile_to(art["k_post"][1,0,:T0,:].float(), T).to(dev).half().contiguous()
sink = torch.zeros(d, device=dev)
def run_read(): sink.copy_(kfp16.float().sum(dim=0))
for _ in range(5): run_read()
torch.cuda.synchronize(); t0=time.perf_counter()
for _ in range(50): run_read()
torch.cuda.synchronize()
read_us_per_page = (time.perf_counter()-t0)/50 / nb_head * 1e6
budget = 0.875 * read_us_per_page
print(f"\n  fp16 read       = {read_us_per_page:.4f} us/page")
print(f"  bandwidth budget= {budget:.4f} us/page  (0.875 x read, at 8x compression)")
print(f"  marginal decode = {b_us:.4f} us/page")
print(f"  marginal/budget = {b_us/budget:.1f}x  (>1 = over budget even at the margin)")
print(f"  marginal/read   = {b_us/read_us_per_page:.1f}x")
print(f"\nReading: if marginal b is small and fixed a is large, the ~500ms is launch/setup")
print("overhead, not decode -> a persistent/fused kernel paying 'a' once could approach b.")
print("If marginal b alone is still >> budget, the entropy decode itself is the wall.")