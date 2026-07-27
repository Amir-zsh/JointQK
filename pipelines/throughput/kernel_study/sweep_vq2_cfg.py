"""Retune the Triton vq2 stage-1 launch config on H100.

The kernel compiles to 255 registers WITH SPILLS at its current defaults
(BLOCK_N=128, BLOCK_H=8, warps=4), which cap it at 2 CTAs/SM; the int2 kernel
sits at 8. Those defaults came from an A100 sweep (see the launcher docstring),
never re-tuned for sm90. Pure env knobs, no code change.
"""
import itertools, os, sys
sys.path.insert(0, "logs")
import torch

def run(bn, bh, nw, ns):
    for k, v in (("SGL_VQ2_BLOCK_N", bn), ("SGL_VQ2_BLOCK_H", bh),
                 ("SGL_VQ2_NUM_WARPS", nw), ("SGL_VQ2_NUM_STAGES", ns)):
        os.environ[k] = str(v)
    for m in [m for m in list(sys.modules) if m.startswith("sglang") or m == "test_tl_vq2"]:
        del sys.modules[m]
    from test_tl_vq2 import build, run_triton, bench
    from sglang.srt.layers.attention.triton_ops import decode_attention as D
    d = build()
    run_triton(d)
    ms = bench(lambda: run_triton(d))
    K = D._fwd_grouped_kernel_stage1_quant_vq2
    regs = spills = shared = 0
    dc = getattr(K, "device_caches", None) or {}
    for _, tup in dc.items():
        kc = tup[0] if isinstance(tup, (tuple, list)) else tup
        for _, ck in list(kc.items())[:1]:
            regs, spills, shared = ck.n_regs, ck.n_spills, ck.metadata.shared
    thr = nw * 32
    ctas = min(65536 // max(regs * thr, 1), (228 * 1024) // max(shared, 1))
    return ms, regs, spills, shared, ctas

print(f"{'BN':>4} {'BH':>3} {'W':>2} {'S':>2} {'us':>9} {'regs':>5} {'spill':>6} "
      f"{'smem':>7} {'CTA/SM':>7}")
best = None
for bn, bh, nw, ns in itertools.product((32, 64, 128), (8, 16), (1, 2, 4), (2, 3)):
    try:
        ms, regs, sp, sh, ctas = run(bn, bh, nw, ns)
    except Exception as e:
        continue
    star = ""
    if best is None or ms < best[0]:
        best, star = (ms, bn, bh, nw, ns), "  <-- best"
    print(f"{bn:>4} {bh:>3} {nw:>2} {ns:>2} {ms*1e3:>9.1f} {regs:>5} {sp:>6} "
          f"{sh:>7} {ctas:>7}{star}")
print(f"\nbest: BLOCK_N={best[1]} BLOCK_H={best[2]} warps={best[3]} stages={best[4]}"
      f"  {best[0]*1e3:.1f} us")
