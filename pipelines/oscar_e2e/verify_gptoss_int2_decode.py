"""Offline correctness gate for the unified HP+int2 decode path at gpt-oss shapes.

gpt-oss-20B differs from every previously-served model on three axes that all
meet inside ``decode_attention_fwd_int2_unified``: head_dim 64 (not 128),
kv_group_num 8, and learned attention sinks. The served int2 smoke returned
coherent text but zero NIAH retrieval — the signature of a quant-tier read
that is wrong while the bf16 HP band (sinks + recent window) still carries
local context. This gate reproduces that read with no model weights.

Run:
  PYTHONPATH=vendor/OSCAR-vq/sglang-research/python \
      .venv-oscar/bin/python pipelines/oscar_e2e/verify_gptoss_int2_decode.py --gpu 1

Cells: (head_dim, q_heads, kv_heads) x (sinks on/off). head_dim=128 is the
llama-shaped control — it exercises the same code with the one axis that is
known to serve correctly, so a D=64-only failure localises the bug to the
head_dim assumptions rather than to the sink or unified-stage2 additions.
"""

import argparse
import os

import torch


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(torch.float64)
    b = b.to(torch.float64)
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def quantize_int2(values: torch.Tensor, num_groups: int):
    """Mirror ``_groupwise_affine_quantize`` + the engine's crumb packing.

    Returns (packed uint8 [T,H,D//4], scales_zeros f32 [T,H,2*num_groups]).
    """
    T, H, D = values.shape
    gs = D // num_groups
    grouped = values.to(torch.float32).reshape(T, H, num_groups, gs)
    vmin = grouped.amin(dim=-1)
    vmax = grouped.amax(dim=-1)
    scale = (vmax - vmin).clamp_min(1e-8) / 3.0
    zero = -vmin / scale
    q = torch.clamp(
        grouped / scale.unsqueeze(-1) + zero.unsqueeze(-1) + 0.5, 0, 3
    ).to(torch.uint8).reshape(T, H, D)

    qd = D // 4
    packed = (
        q[..., :qd]
        | (q[..., qd : 2 * qd] << 2)
        | (q[..., 2 * qd : 3 * qd] << 4)
        | (q[..., 3 * qd :] << 6)
    ).contiguous()
    sz = torch.stack((scale, zero), dim=-1).reshape(T, H, 2 * num_groups)
    return packed, sz.contiguous()


def dequantize_int2(packed: torch.Tensor, sz: torch.Tensor, D: int) -> torch.Tensor:
    """Reference dequant (``_groupwise_dequantize_int2_torch``, fp32 out)."""
    qd = D // 4
    ng = sz.shape[-1] // 2
    gs = D // ng
    scale = sz[..., 0::2].to(torch.float32)
    zero = sz[..., 1::2].to(torch.float32)
    crumbs = [((packed >> (2 * i)) & 0x03).to(torch.float32) for i in range(4)]
    out = torch.empty(packed.shape[0], packed.shape[1], D, device=packed.device)
    for i, c in enumerate(crumbs):
        g = (torch.arange(qd, device=packed.device) + i * qd) // gs
        out[..., i * qd : (i + 1) * qd] = (c - zero[..., g]) * scale[..., g]
    return out


def run_cell(D, QH, H, use_sink, device, bs=2, n_hp=320, n_quant=2000, tol=2e-2,
             sink_bias=0.0, return_out=False):
    from sglang.srt.layers.attention.triton_ops.decode_attention import (
        decode_attention_fwd_int2_unified,
    )

    torch.manual_seed(0)
    hp_cache = bs * n_hp + 8
    q_cache = bs * n_quant + 8

    q = torch.randn(bs, QH, D, device=device, dtype=torch.bfloat16)
    hp_k = torch.randn(hp_cache, H, D, device=device, dtype=torch.bfloat16)
    hp_v = torch.randn(hp_cache, H, D, device=device, dtype=torch.bfloat16)
    k_f = torch.randn(q_cache, H, D, device=device)
    v_f = torch.randn(q_cache, H, D, device=device)
    k_buf, k_sz = quantize_int2(k_f, num_groups=1)
    v_buf, v_sz = quantize_int2(v_f, num_groups=1)

    hp_idx = torch.randperm(hp_cache, device=device)[: bs * n_hp].to(torch.int64)
    q_idx = torch.randperm(q_cache, device=device)[: bs * n_quant].to(torch.int64)
    hp_indptr = torch.tensor([0, n_hp, 2 * n_hp], device=device, dtype=torch.int64)
    q_indptr = torch.tensor([0, n_quant, 2 * n_quant], device=device, dtype=torch.int64)

    hp_splits, q_splits = 8, 16
    total = hp_splits + q_splits
    logits = torch.zeros(bs, QH, total, D, device=device, dtype=torch.float32)
    lse = torch.full((bs, QH, total), float("-inf"), device=device, dtype=torch.float32)
    hp_nks = torch.full((bs,), hp_splits, device=device, dtype=torch.int32)
    q_nks = torch.full((bs,), q_splits, device=device, dtype=torch.int32)

    # A sink drawn near 0 is swamped by ~2300 exp terms in the denominator, so
    # it cannot distinguish "sink applied" from "sink ignored". sink_bias
    # pushes it above the logit max, where it dominates the softmax mass.
    sinks = (
        torch.randn(QH, device=device, dtype=torch.float32) * 2.0 + sink_bias
        if use_sink
        else None
    )
    o = torch.zeros(bs, QH, D, device=device, dtype=torch.bfloat16)
    sm_scale = 1.0 / (D**0.5)

    decode_attention_fwd_int2_unified(
        q, hp_k, hp_v, k_buf, v_buf, k_sz, v_sz, o,
        hp_indptr, hp_idx, q_indptr, q_idx,
        logits, lse, hp_nks, q_nks, hp_splits, q_splits,
        sm_scale, logit_cap=0.0, sinks=sinks,
    )

    # fp32 reference over the same slots: HP rows exact, quant rows dequantized.
    k_hat = dequantize_int2(k_buf, k_sz, D)
    v_hat = dequantize_int2(v_buf, v_sz, D)
    worst = 0.0
    worst_qh = -1
    for b in range(bs):
        hs = hp_idx[b * n_hp : (b + 1) * n_hp]
        qs = q_idx[b * n_quant : (b + 1) * n_quant]
        for qh in range(0, QH, max(1, QH // 8)):
            h = qh // (QH // H)
            kk = torch.cat([hp_k[hs][:, h].float(), k_hat[qs][:, h]])
            vv = torch.cat([hp_v[hs][:, h].float(), v_hat[qs][:, h]])
            s = (q[b, qh].float() @ kk.T) * sm_scale
            if sinks is not None:
                s = torch.cat([s, sinks[qh].view(1)])
                p = torch.softmax(s, -1)[:-1]
            else:
                p = torch.softmax(s, -1)
            e = rel_err(o[b, qh], p @ vv)
            if e > worst:
                worst, worst_qh = e, qh
    ok = worst <= tol
    label = f"sink={'+%.0f' % sink_bias if use_sink else 'off'}"
    print(
        f"  D={D:3d} QH={QH:2d} H={H} group={QH // H} {label:>9}: "
        f"worst rel {worst:.3e} (qh {worst_qh})  {'PASS' if ok else 'FAIL'}"
    )
    return (ok, o.clone()) if return_out else ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    device = torch.device("cuda:0")

    print("== unified HP+int2 decode vs fp32 reference")
    cells = [
        (64, 64, 8),    # gpt-oss-20B
        (128, 32, 8),   # llama-3.1-8B control (known-good in serving)
        (128, 64, 8),   # isolates head_dim from kv_group_num
        (64, 32, 8),    # isolates kv_group_num from head_dim
    ]
    results = []
    for D, QH, H in cells:
        for use_sink in (False, True):
            results.append(run_cell(D, QH, H, use_sink, device))

    # Dominant-sink cell: with sink >> max logit the sink carries most of the
    # softmax mass, so an ignored `sinks` argument shows up as a large error
    # and as output identity with the sink-off run.
    print("== dominant-sink cell (sink must change the output)")
    ok_off, o_off = run_cell(64, 64, 8, False, device, return_out=True)
    ok_big, o_big = run_cell(64, 64, 8, True, device, sink_bias=12.0, return_out=True)
    delta = (o_off.float() - o_big.float()).abs().max().item()
    applied = delta > 1e-3
    print(f"  max |o(sink=off) - o(sink=+12)| = {delta:.3e}  "
          f"{'sink APPLIED' if applied else 'sink IGNORED — BUG'}")
    results += [ok_off, ok_big, applied]
    n_fail = sum(1 for r in results if not r)
    print(f"\n{'ALL PASS' if n_fail == 0 else f'{n_fail}/{len(results)} CELLS FAIL'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
