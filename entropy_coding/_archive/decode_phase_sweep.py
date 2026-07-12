#!/usr/bin/env python3
"""Decode-phase long-context sweep: 1 query vs length-T cache, per-token latency.
Compressed path (rANS decode + attend) vs BF16 flash-decoding. Real K tiled to
long T (keeps real data distribution for honest rANS rate/decode cost).
Analog of OSCAR's batch-1 decode speedup curve (their 1.98/2.52/3.08x at 30/60/100k)."""
import numpy as np, torch, time
import torch.nn.functional as Fnn
from torch.nn.attention import sdpa_kernel, SDPBackend
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
T0 = int(art["prompt_length"]); gs = art["q_post"].shape[1] // Hkv
sm = 1.0 / np.sqrt(d)

# single representative head (l=1,h=0), tile its real K to long T
l, h = 1, 0
c = cod[(l, h)]
k_real = art["k_post"][l, h, :T0, :].float()          # (T0, d) real K
v_real = art["v"][l, h, :T0, :].float()

Ts = [4096, 8192, 16384, 32768, 65536, 100000]

def tile_to(x, T):
    reps = (T + x.shape[0] - 1) // x.shape[0]
    return x.repeat(reps, 1)[:T].contiguous()

def timeit(f, reps=10, warmup=3):
    for _ in range(warmup): f()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3

print(f"{'T':>8} {'bf16_decode':>12} {'comp_attend':>12} {'comp_decode':>12} "
      f"{'comp_total':>12} {'speedup':>9}")

for T in Ts:
    kT = tile_to(k_real, T).to(dev)           # (T,d) fp16-ish real K, tiled
    vT = tile_to(v_real, T).to(dev)
    # one decode-step query (1 token), gs grouped query heads
    q1 = torch.randn(gs, 1, d, device=dev, dtype=torch.float32)

    # --- BF16 flash-decoding baseline: 1 query vs T cache ---
    def run_bf16():
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            Fnn.scaled_dot_product_attention(
                q1.half().unsqueeze(0),
                kT.half().unsqueeze(0).unsqueeze(0).expand(1, gs, T, d),
                vT.half().unsqueeze(0).unsqueeze(0).expand(1, gs, T, d))
    # --- compressed path: encode once (offline-ish), then per-step decode + attend ---
    # encode the tiled cache (this is the "stored compressed cache", done once)
    # we time: decode the resident compressed cache (bandwidth win) + flash attend
    # NOTE: encode is one-time (cache build), NOT in the per-step timed loop.
    try:
        # encode tiled K via codec (one head) — this is cache construction, untimed
        buf = c.encode_gpu(kT.cpu().numpy()) if hasattr(c, "encode_gpu") else None
    except Exception:
        buf = None

    if buf is None:
        print(f"{T:>8}  (encode unavailable for tiled length; skipping comp path)")
        tb = timeit(run_bf16)
        print(f"{T:>8} {tb:>10.3f}ms {'--':>12} {'--':>12} {'--':>12} {'--':>9}")
        continue

    def run_comp_decode():
        return c.decode_to_rhat(buf)          # rANS decode the resident cache -> r̂

    def run_comp_attend():
        rhat = c.decode_to_rhat(buf)[:T]
        invT = torch.as_tensor(c.inv, dtype=torch.float32, device=dev)
        q1p = (q1 @ invT.T)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            Fnn.scaled_dot_product_attention(
                q1p.half().unsqueeze(0),
                rhat.half().unsqueeze(0).unsqueeze(0).expand(1, gs, T, d),
                vT.half().unsqueeze(0).unsqueeze(0).expand(1, gs, T, d))

    tb = timeit(run_bf16)
    tcd = timeit(run_comp_decode)
    tca = timeit(run_comp_attend)
    speedup = tb / tca if tca > 0 else float('nan')
    print(f"{T:>8} {tb:>10.3f}ms {tca-tcd:>10.3f}ms {tcd:>10.3f}ms {tca:>10.3f}ms {speedup:>8.2f}x")

print("\nspeedup = bf16_decode / comp_total. >1 = compressed decode-phase wins.")
print("NOTE: comp_total INCLUDES re-decoding the full cache each step (worst case).")
print("Real serving decodes incrementally; this is the conservative bound.")