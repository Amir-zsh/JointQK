"""Validate SGLANG_VQ_OPT_KMAP against the einsum path, then time it.

Covers both call sites' shapes: the decode aging flush (T = bs*flush_interval,
tens to hundreds) and the prefill VQ write (T = chunk, thousands), and a
non-contiguous k, which the prefill path can hand us.
"""
import os, sys, torch

os.environ["SGLANG_VQ_OPT_KMAP"] = "0"
from sglang.srt.environ import envs
from sglang.srt.mem_cache import vq_codebook as V

H, D = 8, 128
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(0)
fails = 0
print(f"{torch.cuda.get_device_name(0)}\n")
print(f"{'dtype':>9} {'T':>6} {'k':>12} {'max abs err':>12} {'rel':>10}  {'ok':>4}")
for dt in (torch.bfloat16, torch.float16, torch.float32):
    fw = torch.randn(H, D, D, generator=g, device=dev, dtype=dt) * (D ** -0.5)
    mean = torch.randn(H, D, generator=g, device=dev, dtype=dt)
    for T in (1, 8, 17, 64, 512, 2048):
        for lay in ("contig", "strided"):
            if lay == "contig":
                k = torch.randn(T, H, D, generator=g, device=dev, dtype=dt)
            else:                       # slice a wider tensor -> non-contiguous
                k = torch.randn(T, 2 * H, D, generator=g, device=dev, dtype=dt)[:, :H]
            envs.SGLANG_VQ_OPT_KMAP.set("0")
            ref = V.vq_map_k(k, fw, mean)
            envs.SGLANG_VQ_OPT_KMAP.set("1")
            got = V.vq_map_k(k, fw, mean)
            assert got.shape == ref.shape and got.dtype == ref.dtype and got.is_contiguous()
            err = (got.float() - ref.float()).abs().max().item()
            den = ref.float().abs().max().item()
            ok = err / max(den, 1e-9) < (1e-6 if dt is torch.float32 else 6e-3)
            fails += not ok
            print(f"{str(dt).split('.')[-1]:>9} {T:>6} {lay:>12} {err:>12.3e} "
                  f"{err / max(den, 1e-9):>10.2e}  {'ok' if ok else 'FAIL':>4}")

print(f"\n{'correctness: PASS' if not fails else f'correctness: {fails} FAILURES'}\n")


def bench(fn, reps=200):
    """GPU time under graph replay -- these run inside sglang's captured graph."""
    fn(); torch.cuda.synchronize()
    gr, s0 = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    s0.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s0):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s0)
    with torch.cuda.graph(gr):
        fn()
    gr.replay(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps):
        gr.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps * 1e3


dt = torch.bfloat16
fw = torch.randn(H, D, D, generator=g, device=dev, dtype=dt) * (D ** -0.5)
mean = torch.randn(H, D, generator=g, device=dev, dtype=dt)
print(f"{'T':>6} {'einsum':>10} {'fused':>10} {'speedup':>9}")
for T in (8, 32, 64, 128, 512, 2048, 8192):
    k = torch.randn(T, H, D, generator=g, device=dev, dtype=dt)
    envs.SGLANG_VQ_OPT_KMAP.set("0")
    t0 = bench(lambda: V.vq_map_k(k, fw, mean))
    envs.SGLANG_VQ_OPT_KMAP.set("1")
    t1 = bench(lambda: V.vq_map_k(k, fw, mean))
    print(f"{T:>6} {t0:>9.1f}u {t1:>9.1f}u {t0 / t1:>8.2f}x")
sys.exit(1 if fails else 0)
