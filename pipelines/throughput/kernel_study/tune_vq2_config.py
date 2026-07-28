"""Offline autotuner for the Triton vq2 decode stage-1 launch config.

Why offline rather than `@triton.autotune`
------------------------------------------
Runtime autotuning is the wrong tool inside sglang:

  * it benchmarks every config the first time it sees a new key value, which is
    seconds of stall in the serving loop;
  * it CANNOT run during CUDA-graph capture -- it launches kernels and
    synchronises -- and capture is exactly where sglang first calls the decode
    kernel. That is the same class of bug as launching on the wrong stream;
  * batch changes every decode step, so a batch-keyed autotune would re-tune
    constantly.

Offline is tractable because the search space is small and, measured, stable:

  * sglang buckets batch to its CUDA-graph sizes, so there are ~8 points;
  * the optimum does NOT move with context length -- BLOCK_N=64, BLOCK_H=4,
    warps=2, stages=4 wins at ctx=8k, 30k and 60k alike at bs=64 -- so the table
    needs no ctx axis;
  * it does move with batch and with geometry, which is exactly what the table
    is keyed on.

What it emits
-------------
JSON keyed by geometry and batch bucket, consumed by
`_decode_grouped_att_m_fwd_quant_vq2` when SGL_VQ2_CONFIG_JSON points at it.
Without the file the launcher falls back to its built-in ladder, so this is
strictly optional.

  python pipelines/throughput/kernel_study/tune_vq2_config.py \
      --model Qwen/Qwen3-8B --out artifacts/throughput/vq2_tuned_qwen3_8b.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import torch
import triton

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "pipelines/throughput"))

# Same ladder sglang captures graphs at; tuning off-bucket is wasted effort
# because decode always replays one of these.
DEFAULT_BATCHES = (1, 2, 4, 8, 16, 32, 64)
# stages is the cp.async pipeline depth and matters most -- the codebook gather
# lowers to 32 cp.async.ca.shared.global per token. An earlier sweep capped it
# at 3 and sat on the wrong side of an interaction with BLOCK_H.
GRID = dict(block_n=(32, 64, 128), block_h=(4, 8), warps=(1, 2, 4),
            stages=(2, 3, 4, 5, 6))


def geometry(model: str) -> dict:
    import capacity
    g = capacity.model_geometry(model)
    return {"h_kv": g["kv_heads"], "head_dim": g["head_dim"],
            "h_q": g["kv_heads"] * 4 if g["kv_heads"] else 0, "layers": g["layers"]}


def build(bs, ctx, h_q, h_kv, L, ng, kc, splits, dev="cuda", seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    total, cache = bs * ctx, bs * ctx + 64
    q = torch.randn(bs, h_q, L, generator=g, device=dev, dtype=torch.float16)
    k_idx = torch.randint(0, kc, (cache, h_kv, ng), generator=g, device=dev, dtype=torch.uint8)
    cent = torch.randn(h_kv, ng, kc, 4, generator=g, device=dev, dtype=torch.float32) * (L ** -0.5)
    cb = cent.to(torch.float8_e5m2).view(torch.int32).squeeze(-1).contiguous()
    v_buf = torch.randint(0, 256, (cache, h_kv, L // 4), generator=g, device=dev, dtype=torch.uint8)
    k_sz = torch.rand(cache, h_kv, 2, generator=g, device=dev) * 0.5 + 0.75
    v_sz = torch.rand(cache, h_kv, 2, generator=g, device=dev) * 0.1 + 0.05
    kv_indptr = torch.arange(0, bs + 1, device=dev, dtype=torch.int32) * ctx
    kv_indices = torch.randperm(cache, generator=g, device=dev)[:total].to(torch.int32)
    nsp = torch.empty(bs, device=dev, dtype=torch.int32)
    seq_lens = torch.full((bs,), ctx, device=dev, dtype=torch.int32)
    from sglang.srt.layers.attention.triton_backend import get_num_kv_splits_triton
    cores = torch.cuda.get_device_properties(0).multi_processor_count
    get_num_kv_splits_triton[(1,)](
        nsp, seq_lens, bs, 1, h_q, h_kv, splits, cores,
        MAX_NUM_SEQ=256 if bs < 256 else triton.next_power_of_2(bs))
    out = torch.zeros(bs, h_q, splits, L, device=dev, dtype=torch.float32)
    lse = torch.zeros(bs, h_q, splits, device=dev, dtype=torch.float32)
    return locals()


def bench(fn, reps=20):
    fn(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps):
        fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--ctx", type=int, default=30000,
                    help="tuning context; the optimum is measured to be "
                         "ctx-independent, so this only sets the runtime")
    ap.add_argument("--batches", type=lambda s: [int(x) for x in s.split(",")],
                    default=list(DEFAULT_BATCHES))
    ap.add_argument("--ng", type=int, default=32)
    ap.add_argument("--kc", type=int, default=256)
    ap.add_argument("--splits", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    geo = geometry(a.model)
    h_kv, L = geo["h_kv"], geo["head_dim"]
    h_q = geo["h_q"]
    from sglang.srt.layers.attention.triton_ops import decode_attention as D
    gpu = torch.cuda.get_device_name(0)
    print(f"{a.model}: H_Q={h_q} H_KV={h_kv} head_dim={L} NG={a.ng} KC={a.kc} on {gpu}")

    table = {}
    for bs in a.batches:
        d = build(bs, a.ctx, h_q, h_kv, L, a.ng, a.kc, a.splits)

        def run():
            D._decode_grouped_att_m_fwd_quant_vq2(
                d["q"], d["k_idx"], d["cb"], d["v_buf"], d["k_sz"], d["v_sz"],
                d["out"], d["lse"], d["kv_indptr"], d["kv_indices"], d["nsp"],
                a.splits, 1.0 / (L ** 0.5), 0.0)

        best = None
        for bn, bh, nw, ns in itertools.product(
                GRID["block_n"], GRID["block_h"], GRID["warps"], GRID["stages"]):
            os.environ.update(SGL_VQ2_BLOCK_N=str(bn), SGL_VQ2_BLOCK_H=str(bh),
                              SGL_VQ2_NUM_WARPS=str(nw), SGL_VQ2_NUM_STAGES=str(ns))
            try:
                ms = bench(run)
            except Exception:
                continue          # config does not compile at this shape
            if best is None or ms < best[0]:
                best = (ms, [bn, bh, nw, ns])
        for k in ("SGL_VQ2_BLOCK_N", "SGL_VQ2_BLOCK_H", "SGL_VQ2_NUM_WARPS",
                  "SGL_VQ2_NUM_STAGES"):
            os.environ.pop(k, None)
        ms, cfg = best
        table[str(bs)] = cfg
        print(f"  bs={bs:<4} -> BLOCK_N={cfg[0]:<4} BLOCK_H={cfg[1]} warps={cfg[2]} "
              f"stages={cfg[3]}   {ms * 1e3:8.1f} us")
        del d
        torch.cuda.empty_cache()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "gpu": gpu, "model": a.model, "ctx_tuned_at": a.ctx,
        "geometry": {"h_q": h_q, "h_kv": h_kv, "head_dim": L,
                     "ng": a.ng, "kc": a.kc},
        "configs": table,
    }, indent=2) + "\n")
    print(f"\nwrote {a.out}\nexport SGL_VQ2_CONFIG_JSON={a.out} to use it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
