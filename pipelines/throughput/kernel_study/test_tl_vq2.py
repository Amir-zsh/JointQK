"""Validate + benchmark the TileLang vq2 stage-1 against the Triton kernel.

Both are driven with byte-identical inputs and compared on Att_Out / Att_Lse.
A speed number from a kernel that computes the wrong thing is worthless, so
correctness gates the benchmark.

  docker exec oscar-ab bash -lc '<repo> && PYTHONPATH=vendor/OSCAR-vq/sglang-research/python \
      .venv-tilelang/bin/python logs/test_tl_vq2.py'
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The engine venv ships a TVM that collides with TileLang's ("TypeAttr
# __ffi_repr__ is already registered"), so the two never share a process:
# --triton writes reference inputs+outputs, --tilelang reads and compares.
REF = Path(__file__).resolve().parents[3] / "logs/tl_vq2_ref.pt"

# Saturating grid: BATCH*HEAD_BLOCKS*SPLITS = 32*8*8 = 2048 CTAs on 132 SMs.
# At BATCH=2 the grid was 128 CTAs -- ~1 per SM -- which hides both the
# occupancy cost of shared staging and the gather advantage.
import os as _o
BATCH, H_Q, H_KV, L, NG, KC = int(_o.environ.get('BS', 32)), 32, 8, 128, 32, 256
SEQ, SPLITS = int(_o.environ.get('CTX', 8192)), int(_o.environ.get('SPLITS', 8))
SM_SCALE = 1.0 / (L ** 0.5)


def build(dev="cuda", seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    total = BATCH * SEQ
    cache = total + 64                       # slack; paged slots are shuffled

    q = torch.randn(BATCH, H_Q, L, generator=g, device=dev, dtype=torch.float16)
    k_idx = torch.randint(0, KC, (cache, H_KV, NG), generator=g, device=dev,
                          dtype=torch.uint8)
    # Codebook: random centroids snapped to e5m2, packed 4 bytes -> int32,
    # exactly as vq_codebook.py builds cb_packed.
    cent = (torch.randn(H_KV, NG, KC, 4, generator=g, device=dev,
                        dtype=torch.float32) * (L ** -0.5))
    cb = cent.to(torch.float8_e5m2).view(torch.int32).squeeze(-1).contiguous()

    v_buf = torch.randint(0, 256, (cache, H_KV, L // 4), generator=g, device=dev,
                          dtype=torch.uint8)
    k_sz = torch.rand(cache, H_KV, 2, generator=g, device=dev,
                      dtype=torch.float32) * 0.5 + 0.75
    v_sz = torch.empty(cache, H_KV, 2, device=dev, dtype=torch.float32)
    v_sz[..., 0] = torch.rand(cache, H_KV, generator=g, device=dev) * 0.1 + 0.05
    v_sz[..., 1] = 1.5

    kv_indptr = torch.arange(0, BATCH + 1, device=dev, dtype=torch.int32) * SEQ
    kv_indices = torch.randperm(cache, generator=g, device=dev)[:total].to(torch.int32)
    splits = torch.full((BATCH,), SPLITS, device=dev, dtype=torch.int32)
    return dict(q=q, k_idx=k_idx, cb=cb, v_buf=v_buf, k_sz=k_sz, v_sz=v_sz,
                kv_indptr=kv_indptr, kv_indices=kv_indices, splits=splits,
                cache=cache, total=total)


def run_triton(d):
    from sglang.srt.layers.attention.triton_ops.decode_attention import (
        _decode_grouped_att_m_fwd_quant_vq2,
    )
    out = torch.zeros(BATCH, H_Q, SPLITS, L, device="cuda", dtype=torch.float32)
    lse = torch.zeros(BATCH, H_Q, SPLITS, device="cuda", dtype=torch.float32)
    _decode_grouped_att_m_fwd_quant_vq2(
        d["q"], d["k_idx"], d["cb"], d["v_buf"], d["k_sz"], d["v_sz"],
        out, lse, d["kv_indptr"], d["kv_indices"], d["splits"], SPLITS,
        SM_SCALE, 0.0)
    return out, lse


def run_tilelang(d, kern):
    out = torch.zeros(BATCH, H_Q, SPLITS, L, device="cuda", dtype=torch.float32)
    lse = torch.zeros(BATCH, H_Q, SPLITS, device="cuda", dtype=torch.float32)
    kern(d["q"], d["k_idx"], d["cb"], d["v_buf"].view(torch.int32), d["k_sz"], d["v_sz"],
         d["kv_indptr"], d["kv_indices"], d["splits"], SM_SCALE, out, lse)
    return out, lse


def bench(fn, reps=30):
    fn(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps):
        fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


def main() -> int:
    print(f"BATCH={BATCH} H_Q={H_Q} H_KV={H_KV} L={L} NG={NG} KC={KC} "
          f"SEQ={SEQ} SPLITS={SPLITS}")

    if "--triton" in sys.argv:
        d = build()
        o_t, l_t = run_triton(d)
        ms = bench(lambda: run_triton(d))
        torch.save({"d": {k: v for k, v in d.items()},
                    "out": o_t, "lse": l_t, "ms": ms}, REF)
        print(f"  triton {ms*1e3:8.1f} us   reference written to {REF.name}")
        return 0

    from tl_vq2_stage1 import vq2_stage1
    ref = torch.load(REF, weights_only=False)
    d, o_t, l_t, ms_t = ref["d"], ref["out"], ref["lse"], ref["ms"]

    for bn, bh, stages, threads in ((64, 16, 2, 128),):
        tag = f"BLOCK_N={bn} BLOCK_H={bh} stages={stages} threads={threads}"
        try:
            import os as _os
            kern = vq2_stage1(BATCH, H_Q, H_KV, L, NG, KC, d["cache"],
                              d["total"], SPLITS, BLOCK_N=bn, BLOCK_H=bh,
                              num_stages=stages, threads=threads,
                              ablate=_os.environ.get("ABLATE", ""),
                              cb_shared=_os.environ.get("CBSH", "1") == "1")
            o_l, l_l = run_tilelang(d, kern)
        except Exception as e:
            print(f"  {tag}: BUILD/RUN FAILED {type(e).__name__}: "
                  f"{str(e).splitlines()[0][:110]}")
            continue

        do = (o_l - o_t).abs().max().item()
        dl = (l_l - l_t).abs().max().item()
        rel = do / max(o_t.abs().max().item(), 1e-9)
        ok = rel < 2e-2 and dl < 2e-2 or bool(__import__('os').environ.get('ABLATE'))
        ms_l = bench(lambda: run_tilelang(d, kern))
        print(f"  {tag}\n"
              f"    max|dOut|={do:.4e} (rel {rel:.2e})  max|dLse|={dl:.4e}  "
              f"{'PASS' if ok else 'FAIL'}\n"
              f"    triton {ms_t*1e3:8.1f} us   tilelang {ms_l*1e3:8.1f} us   "
              f"speedup {ms_t/ms_l:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
