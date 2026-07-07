#!/usr/bin/env python3
"""Measure the existing CUDA rANS decoder's throughput (page_quant probe).

Answers "can entropy decode sit in the per-token decode path?" with a datum:
synthetic N(0,1) codes at ~2 b/c through the real PageCodecRANSCUDA
(entropy_coding/kvq_codec.py + rans_decode.cu), 8 heads x 64k tokens,
kernel-only ms from the codec's own CUDA events. Scales the result to a full
Llama-3.1-8B decode step (31 layers x 8 kv-heads).

Run from repo root:  .venv/bin/python pipelines/page_quant/probe_rans_throughput.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import _bootstrap  # noqa: E402,F401

os.chdir(REPO / "entropy_coding")          # load_ext uses relative .cu paths
sys.path.insert(0, str(REPO / "entropy_coding"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from kvq_codec import (  # noqa: E402
    BatchRANSDecoder, BatchRANSEncoder, build_codecs_from_ladder_rans_cuda,
    load_ext, load_enc_ext,
)

T = 65536
D = 128
H = 8
PTOK = 64
LANES = 16
DELTA = 0.35   # ~2.1 b/c deadzone entropy for N(0,1)


def main() -> None:
    rng = np.random.default_rng(0)
    ks = {(0, h): rng.standard_normal((T, D)).astype(np.float32)
          for h in range(H)}

    # frozen model from the data itself (throughput probe, not a rate claim)
    model = {}
    for h in range(H):
        k = ks[(0, h)]
        idx = np.sign(k) * np.floor(np.abs(k) / DELTA + 0.5)
        mdl = []
        for j in range(D):
            v, c = np.unique(idx[:, j].astype(np.int64), return_counts=True)
            mdl.append((v, (c / c.sum()).astype(np.float64)))
        model[(0, h)] = mdl

    F = torch.eye(D).expand(1, H, D, D).contiguous()
    k_mean = torch.zeros(1, H, D)
    delta = torch.full((1, H, D), DELTA)
    ladder = [(1.0, delta, model)]

    ext = load_ext()
    enc_ext = load_enc_ext()
    codecs = build_codecs_from_ladder_rans_cuda(
        F, F.transpose(-1, -2).contiguous(), k_mean, ladder, 1, H,
        page_bits=1 << 24, P_tok=PTOK, dz=0.5, lanes=LANES,
        ext=ext, enc_ext=enc_ext, device="cuda")

    enc = BatchRANSEncoder(codecs)
    bufs = enc.encode_grid({k: torch.as_tensor(v) for k, v in ks.items()})
    nbytes = sum(len(b) for b in bufs.values())
    syms = H * T * D
    print(f"[probe] encoded {syms / 1e6:.1f}M symbols -> {nbytes / 1e6:.1f} MB "
          f"({8 * nbytes / (H * T * D):.3f} b/c)")

    dec = BatchRANSDecoder(codecs)
    for it in range(3):
        out = dec.decode_grid(bufs)
        total_ms = dec.last_total_ms
        kern_ms = dec.last_kernel_ms
        gsps = syms / (kern_ms * 1e6) if kern_ms else float("nan")
        print(f"[probe] iter{it}: total {total_ms:.1f} ms | kernel "
              f"{kern_ms:.2f} ms | {gsps:.2f} Gsym/s (kernel-only)")

    # scale to one full decode step: 31 layers x 8 heads at this T
    full_syms = 31 * 8 * T * D
    per_step_kernel_ms = full_syms / (gsps * 1e9) * 1e3
    print(f"[probe] full-model whole-context decode at T={T}: "
          f"{full_syms / 1e9:.2f}G symbols -> ~{per_step_kernel_ms:.1f} ms "
          f"per generated token (kernel-only, host parsing excluded); "
          f"fp16 KV attention read budget at this ctx ~= 4-6 ms.")
    err = (out[(0, 0)].cpu().numpy()
           - np.sign(ks[(0, 0)]) * (np.floor(np.abs(ks[(0, 0)]) / DELTA + 0.5))
           * DELTA)
    print(f"[probe] decode sanity max|err| vs direct dequant: "
          f"{np.abs(err).max():.2e}")


if __name__ == "__main__":
    main()
