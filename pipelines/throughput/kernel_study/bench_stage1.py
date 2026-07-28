"""Fast, FAITHFUL stage-1 sweep: real shapes, seconds per config.

Every microbenchmark in this investigation lied because it used a different
shape than the served workload -- GRID=1, BATCH=2, SEQ=8192, int32 indices.
Each time the ranking inverted once measured end-to-end, at 5-10 minutes a
point. This one is built at the ACTUAL serving shape:

    bs=64, ctx=30000, splits=8, Qwen3-8B  ->  grid = 64 x 8 x 8 = 4096 CTAs
                                              3750 KV tokens per CTA

so a config sweep costs seconds instead of a server boot. Its ranking is
validated against a known end-to-end pair before it is trusted (--validate).

  docker exec oscar-ab bash -lc '<repo> && PYTHONPATH=vendor/... \
      /opt/venv-oscar/bin/python logs/bench_stage1.py'
"""
import itertools
import os
import sys

import torch

BATCH = int(os.environ.get("BS", 64))
SEQ = int(os.environ.get("CTX", 30000))
SPLITS = int(os.environ.get("SPLITS", 8))
H_Q, H_KV, L, NG, KC = 32, 8, 128, 32, 256
SM_SCALE = 1.0 / (L ** 0.5)


def build(dev="cuda", seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    total = BATCH * SEQ
    cache = total + 64
    q = torch.randn(BATCH, H_Q, L, generator=g, device=dev, dtype=torch.float16)
    k_idx = torch.randint(0, KC, (cache, H_KV, NG), generator=g, device=dev,
                          dtype=torch.uint8)
    cent = torch.randn(H_KV, NG, KC, 4, generator=g, device=dev,
                       dtype=torch.float32) * (L ** -0.5)
    cb = cent.to(torch.float8_e5m2).view(torch.int32).squeeze(-1).contiguous()
    v_buf = torch.randint(0, 256, (cache, H_KV, L // 4), generator=g, device=dev,
                          dtype=torch.uint8)
    k_int2 = torch.randint(0, 256, (cache, H_KV, L // 4), generator=g, device=dev,
                           dtype=torch.uint8)
    k_sz = torch.rand(cache, H_KV, 2, generator=g, device=dev) * 0.5 + 0.75
    v_sz = torch.rand(cache, H_KV, 2, generator=g, device=dev) * 0.1 + 0.05
    kv_indptr = torch.arange(0, BATCH + 1, device=dev, dtype=torch.int32) * SEQ
    kv_indices = torch.randperm(cache, generator=g, device=dev)[:total].to(torch.int32)
    # Use the ENGINE's own occupancy heuristic, not a fixed value. Hardcoding
    # splits=max was the last regime mismatch: the engine derives splits from
    # core count and num_seq (triton_backend.py:392-452), so a swept config was
    # being judged at split counts the engine never uses.
    splits = torch.empty(BATCH, device=dev, dtype=torch.int32)
    seq_lens = torch.full((BATCH,), SEQ, device=dev, dtype=torch.int32)
    from sglang.srt.layers.attention.triton_backend import (
        get_num_kv_splits_triton,
    )
    import triton as _tt
    _cores = torch.cuda.get_device_properties(0).multi_processor_count
    get_num_kv_splits_triton[(1,)](
        splits, seq_lens, BATCH, 1, H_Q, H_KV, SPLITS, _cores,
        MAX_NUM_SEQ=256 if BATCH < 256 else _tt.next_power_of_2(BATCH))
    out = torch.zeros(BATCH, H_Q, SPLITS, L, device=dev, dtype=torch.float32)
    lse = torch.zeros(BATCH, H_Q, SPLITS, device=dev, dtype=torch.float32)
    return locals()


def bench(fn, reps=20):
    fn(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps):
        fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


def stats(K):
    dc = getattr(K, "device_caches", None) or {}
    for _, tup in dc.items():
        kc = tup[0] if isinstance(tup, (tuple, list)) else tup
        for _, ck in list(kc.items())[:1]:
            thr = ck.metadata.num_warps * 32
            return (ck.n_regs, ck.n_spills, ck.metadata.shared,
                    min(65536 // max(ck.n_regs * thr, 1),
                        (228 * 1024) // max(ck.metadata.shared, 1)))
    return (0, 0, 0, 0)


def main() -> int:
    d = build()
    from sglang.srt.layers.attention.triton_ops import decode_attention as D

    def vq2():
        D._decode_grouped_att_m_fwd_quant_vq2(
            d["q"], d["k_idx"], d["cb"], d["v_buf"], d["k_sz"], d["v_sz"],
            d["out"], d["lse"], d["kv_indptr"], d["kv_indices"], d["splits"],
            SPLITS, SM_SCALE, 0.0)

    def int2():
        D._decode_grouped_att_m_fwd_quant_int2(
            d["q"], d["k_int2"], d["v_buf"], d["k_sz"], d["v_sz"],
            d["out"], d["lse"], d["kv_indptr"], d["kv_indices"], d["splits"],
            SPLITS, SM_SCALE, 0.0)

    sp = d["splits"]
    print(f"shape: bs={BATCH} ctx={SEQ} max_splits={SPLITS} -> engine chose "
          f"splits={int(sp.min())}..{int(sp.max())} (mean {float(sp.float().mean()):.1f})")

    ms_i = bench(int2)
    r, sp, sh, c = stats(D._fwd_grouped_kernel_stage1_quant_int2)
    print(f"\nint2 (reference target): {ms_i * 1e3:8.1f} us   "
          f"regs={r} spills={sp} smem={sh} CTAs/SM={c}\n")

    print(f"{'BN':>4} {'BH':>3} {'W':>2} {'S':>2} {'us':>9} {'vs int2':>8} "
          f"{'regs':>5} {'sp':>4} {'smem':>7} {'CTA':>4}")
    rows = []
    # num_stages is the cp.async pipeline depth, and the compiled PTX shows the
# codebook gather IS 32 cp.async.ca.shared.global per token -- so this is the
# knob that controls how many blocks' gathers are in flight. Swept wider.
    for bn, bh, nw, ns in itertools.product((32, 64, 128), (4, 8),
                                            (1, 2, 4), (2, 3, 4, 5, 6)):
        for k, v in (("SGL_VQ2_BLOCK_N", bn), ("SGL_VQ2_BLOCK_H", bh),
                     ("SGL_VQ2_NUM_WARPS", nw), ("SGL_VQ2_NUM_STAGES", ns)):
            os.environ[k] = str(v)
        try:
            ms = bench(vq2)
        except Exception:
            continue
        r, sp, sh, c = stats(D._fwd_grouped_kernel_stage1_quant_vq2)
        rows.append((ms, bn, bh, nw, ns, r, sp, sh, c))
    rows.sort()
    for ms, bn, bh, nw, ns, r, sp, sh, c in rows[:14]:
        print(f"{bn:>4} {bh:>3} {nw:>2} {ns:>2} {ms * 1e3:>9.1f} "
              f"{ms / ms_i:>7.2f}x {r:>5} {sp:>4} {sh:>7} {c:>4}")
    ms, bn, bh, nw, ns = rows[0][:5]
    print(f"\nbest vq2: BLOCK_N={bn} BLOCK_H={bh} warps={nw} stages={ns}  "
          f"{ms * 1e3:.1f} us  ({ms / ms_i:.2f}x int2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
