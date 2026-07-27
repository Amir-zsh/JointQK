#!/usr/bin/env python3
"""Check an eager live VQ decode dump against an independent torch reference."""

from __future__ import annotations

import argparse

import torch


def rel_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        (actual.double() - expected.double()).norm()
        / expected.double().norm().clamp_min(1e-12)
    )


def unpack_int2(
    packed: torch.Tensor, scale_zero: torch.Tensor, dim: int
) -> torch.Tensor:
    crumbs = torch.stack(
        [((packed.to(torch.int32) >> (2 * i)) & 0x3) for i in range(4)],
        dim=-1,
    ).float()
    values = crumbs.permute(0, 1, 3, 2).reshape(packed.shape[0], packed.shape[1], dim)
    return (values - scale_zero[..., 1:2].float()) * scale_zero[..., 0:1].float()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump")
    parser.add_argument("--tolerance", type=float, default=2e-2)
    args = parser.parse_args()

    data = torch.load(args.dump, map_location="cpu", weights_only=False)
    if data.get("format") != "sglang-vq-live-decode-v1":
        raise ValueError(f"unsupported dump format: {data.get('format')}")

    q_original = data["q_original"].float()
    q_mapped = data["q_mapped"].float()
    q_map = data["q_map"].float()
    mean = data["mean"].float()
    heads = mean.shape[0]
    q_heads, dim = q_mapped.shape
    q_per_kv = q_heads // heads

    q_grouped = (
        q_original.view(heads, q_per_kv, dim)
        .permute(0, 1, 2)
        .reshape(heads, q_per_kv, dim)
    )
    q_map_ref = torch.einsum("hgd,hde->hge", q_grouped, q_map).reshape(q_heads, dim)
    q_map_error = rel_error(q_mapped, q_map_ref)

    cb = data["cb16"].float()
    quant_idx = data["quant_k_idx"].long()
    tokens, _, groups = quant_idx.shape
    h_ids = torch.arange(heads).view(1, heads, 1)
    g_ids = torch.arange(groups).view(1, 1, groups)
    quant_k = cb[h_ids, g_ids, quant_idx].reshape(tokens, heads, dim)
    quant_k *= data["quant_k_scale"].float().unsqueeze(-1)
    quant_v = unpack_int2(data["quant_v_packed"], data["quant_v_scale_zero"], dim)
    hp_k = data["hp_k"].float()
    hp_v = data["hp_v"].float()
    all_k = torch.cat([hp_k, quant_k], dim=0)
    all_v = torch.cat([hp_v, quant_v], dim=0)

    expected = torch.empty_like(data["output_stored"].float())
    sink_errors = []
    for q_head in range(q_heads):
        kv_head = q_head // q_per_kv
        logits = (q_mapped[q_head] @ all_k[:, kv_head].T) * data["sm_scale"]
        if data["sinks"] is not None:
            shift = (q_original[q_head] @ mean[kv_head]) * data["sm_scale"]
            sink = data["sinks"][q_head].float() - shift
            weights = torch.softmax(torch.cat([logits, sink.view(1)]), dim=0)
            sink_errors.append(float(weights[-1]))
            weights = weights[:-1]
        else:
            weights = torch.softmax(logits, dim=0)
        expected[q_head] = weights @ all_v[:, kv_head]

    output_error = rel_error(data["output_stored"].float(), expected)
    print(f"q-map relative error: {q_map_error:.6e}")
    print(f"fused output relative error: {output_error:.6e}")
    if sink_errors:
        print(
            "sink probability min/mean/max: "
            f"{min(sink_errors):.6f}/"
            f"{sum(sink_errors) / len(sink_errors):.6f}/"
            f"{max(sink_errors):.6f}"
        )
    if output_error > args.tolerance:
        print(
            f"FAIL: fused output error {output_error:.6e} exceeds "
            f"{args.tolerance:.6e}"
        )
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
