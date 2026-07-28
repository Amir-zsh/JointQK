"""Microbenchmark the vq2 per-head maps, and test Samuel's column-split idea.

Two maps run on the vq2 decode path and neither is stage-1 attention:

  * `vq_map_q`  -- q @ q_map per KV head, EVERY layer EVERY step.
    Our tree already has a fused Triton version (SGLANG_VQ_OPT_QMAP), but its
    grid is (cdiv(T*GRP, 16), H): at decode T*GRP is 4-256, so it launches as
    few as 8 blocks on 132 SMs.
  * `vq_map_k`  -- (k - mean) @ forward per head, on the flush path
    (unified_kv_pool.py:808). Still `torch.einsum(...).contiguous()`; we never
    fused it.

Samuel's fix is to split the OUTPUT COLUMNS as well as the heads, so the grid
becomes (rows, H, D/BLOCK_E) instead of (rows, H) -- the work is tiny and
utterly launch-bound, so more blocks is the whole game. He reports 3.56x on the
K map and 4.06x on the q map, bit-identical.

This measures, at real decode shapes, on our tree:
  torch  vs  our fused  vs  column-split fused,
for both maps, and checks bit-exactness against torch.

  docker exec -e CUDA_VISIBLE_DEVICES=1 oscar-ab bash -lc 'cd <repo> && \
    PYTHONPATH=vendor/OSCAR-vq/sglang-research/python \
    /opt/venv-oscar/bin/python pipelines/throughput/kernel_study/bench_vq_maps.py'
"""
import torch
import triton
import triton.language as tl

H, D = 8, 128
GRP = 4                      # Qwen3: 32 q heads / 8 kv heads


@triton.jit
def _map_split_kernel(
    X, MAP, MEAN, OUT,
    n_rows, XH,
    D: tl.constexpr, GRP: tl.constexpr, BLOCK_T: tl.constexpr,
    BLOCK_E: tl.constexpr, HAS_MEAN: tl.constexpr,
):
    """out[t, h*GRP+g, e0:e0+BLOCK_E] = (x[...] - mean[h]) @ map[h][:, e0:]

    Same maths as _vq_qmap_kernel, plus (a) a third grid axis over output
    columns and (b) optional mean subtraction so ONE kernel serves both the
    K map (GRP=1, mean) and the q map (GRP=group, no mean).
    """
    pid = tl.program_id(0)
    h = tl.program_id(1)
    pe = tl.program_id(2)
    offs_i = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_i = offs_i < n_rows
    t = offs_i // GRP
    g = offs_i % GRP
    xh = h * GRP + g
    offs_d = tl.arange(0, D)
    offs_e = pe * BLOCK_E + tl.arange(0, BLOCK_E)

    base_in = t[:, None] * (XH * D) + xh[:, None] * D + offs_d[None, :]
    xv = tl.load(X + base_in, mask=mask_i[:, None], other=0.0)
    if HAS_MEAN:
        xv = xv - tl.load(MEAN + h * D + offs_d)[None, :]
    m = tl.load(MAP + h * D * D + offs_d[:, None] * D + offs_e[None, :])
    o = tl.dot(xv, m)
    base_out = t[:, None] * (XH * D) + xh[:, None] * D + offs_e[None, :]
    tl.store(OUT + base_out, o.to(xv.dtype), mask=mask_i[:, None])


def map_split(x, mp, mean, grp, block_e):
    T, XH, Dx = x.shape
    h = mp.shape[0]
    xc = x.to(mp.dtype).contiguous()
    out = torch.empty_like(xc)
    n_rows = T * grp
    bt = 16
    grid = (triton.cdiv(n_rows, bt), h, Dx // block_e)
    _map_split_kernel[grid](
        xc, mp, mean if mean is not None else xc, out, n_rows, XH,
        D=Dx, GRP=grp, BLOCK_T=bt, BLOCK_E=block_e,
        HAS_MEAN=mean is not None, num_warps=4, num_stages=2)
    return out


def bench(fn, reps=200):
    """GPU time under CUDA-graph replay.

    A plain Python loop measures CPU launch overhead, not the kernel: torch's
    einsum path is 3 ops and the fused path is 1, so a launch-bound loop reports
    a ~3x "speedup" that is pure dispatch cost. In the server these run INSIDE a
    captured graph, where that overhead is gone -- so capture a graph and replay
    it, which is the condition that actually applies.
    """
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s0 = torch.cuda.Stream()
    s0.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s0):          # warm allocator/autotune before capture
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s0)
    with torch.cuda.graph(g):
        fn()
    g.replay(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps):
        g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps * 1e3          # us


def main() -> int:
    from sglang.srt.mem_cache.vq_codebook import (
        vq_map_k, vq_map_q, vq_map_q_fused)
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    print(f"{torch.cuda.get_device_name(0)}  H={H} D={D} GRP={GRP}\n")

    print("=== q map: q[T,32,128] @ q_map[8,128,128], per layer per STEP ===")
    print(f"{'T (=bs)':>8} {'torch bmm':>11} {'fused (ours)':>13} "
          f"{'split E=32':>11} {'split E=64':>11} {'best speedup':>13}")
    qm = torch.randn(H, D, D, generator=g, device=dev, dtype=torch.bfloat16)
    for T in (1, 4, 16, 64):
        q = torch.randn(T, H * GRP, D, generator=g, device=dev, dtype=torch.bfloat16)
        ref = vq_map_q(q, qm)
        r = [bench(lambda: vq_map_q(q, qm)), bench(lambda: vq_map_q_fused(q, qm))]
        for be in (32, 64):
            got = map_split(q, qm, None, GRP, be)
            assert torch.equal(got, ref) or (got - ref).abs().max() < 1e-2, \
                f"q map mismatch at T={T} BLOCK_E={be}"
            r.append(bench(lambda: map_split(q, qm, None, GRP, be)))
        print(f"{T:>8} {r[0]:>10.1f}u {r[1]:>12.1f}u {r[2]:>10.1f}u {r[3]:>10.1f}u "
              f"{r[0] / min(r[1:]):>12.2f}x")

    print("\n=== K map: (k[n,8,128]-mean) @ forward[8,128,128], on FLUSH ===")
    print("     n = bs * flush_interval, so it grows with batch")
    print(f"{'n':>8} {'torch einsum':>13} {'split E=32':>11} {'split E=64':>11} "
          f"{'split E=128':>12} {'speedup':>9}")
    fw = torch.randn(H, D, D, generator=g, device=dev, dtype=torch.bfloat16)
    mean = torch.randn(H, D, generator=g, device=dev, dtype=torch.bfloat16)
    for n in (8, 32, 64, 128, 512):
        k = torch.randn(n, H, D, generator=g, device=dev, dtype=torch.bfloat16)
        ref = vq_map_k(k, fw, mean)
        r = [bench(lambda: vq_map_k(k, fw, mean))]
        for be in (32, 64, 128):
            got = map_split(k, fw, mean, 1, be)
            err = (got.float() - ref.float()).abs().max().item()
            assert err < 5e-2, f"K map mismatch at n={n} BLOCK_E={be}: {err}"
            r.append(bench(lambda: map_split(k, fw, mean, 1, be)))
        print(f"{n:>8} {r[0]:>12.1f}u {r[1]:>10.1f}u {r[2]:>10.1f}u {r[3]:>11.1f}u "
              f"{r[0] / min(r[1:]):>8.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
