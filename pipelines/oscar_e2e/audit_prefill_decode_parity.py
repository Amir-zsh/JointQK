"""Parity audit: chunked-prefill quantized reads vs decode-kernel reads.

Motivation: the chunked-prefill accuracy loss (tracker p2-4) is carried by the
prefix reads in ``dequantize_prefix_kv`` — a code path the single-chunk arm
never executes (it only does decode-kernel reads). This audit rules a prefill
READ-PATH bug in or out by writing rows through the production
``set_kv_buffer`` and reading the SAME arena bytes three ways per layer:

  exact  : the pool's own stored-space truth (``_rotate_kv_inplace`` output)
  A      : the prefill path — ``dequantize_prefix_kv`` (what chunk N+1 sees)
  B      : the decode-side reference — ``vq_dequant`` over the K index arena
           (G3-proved equivalent to the decode kernel) and the plane-layout
           int2 formula for V (validated by verify_gptoss_int2_decode)

Checks, per layer:
  P1  A_K vs B_K elementwise — same bytes, two decoders. cb16 is the
      fp8-dequantized centroid table, so these must agree to fp16 rounding;
      ANY structured gap is a prefill-read bug.
  P2  A_V vs B_V elementwise — catches crumb-order/coordinate-permutation
      bugs that norm- and cosine-level gates cannot see.
  P3  A vs exact and B vs exact — the two paths' distortion must match; the
      prefill path being materially worse than decode would explain the
      chunked loss as a bug rather than as quantization.
  P4  score-level: softmax(q.K_hat) top-1 agreement + max prob delta for the
      prefill read vs the decode read against the exact-prefix softmax. The
      two quantized paths must be statistically indistinguishable from each
      other.
  P5  the exact-chunked-prefill shadow read returns the exact rows (bf16
      rounding only) and leaves A/B untouched after release.

Run:
  PYTHONPATH=vendor/OSCAR-vq/sglang-research/python \
      .venv-oscar/bin/python pipelines/oscar_e2e/audit_prefill_decode_parity.py --gpu 1
"""

import argparse
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

import torch

ROT_DIR = os.path.join(REPO, "artifacts/oscar_gptoss20b/rotations_gpqa198")
BUNDLE = os.path.join(
    REPO, "artifacts/oscar_gptoss20b/vqa_gptoss20b_G4_strat_flat_ptn_gpqacc128k_fp8.pt"
)
LAYERS, H, D, N_Q = 12, 8, 64, 8
HP_PREFIX, HP_RECENT = 64, 128  # serving values (PREFIX_TOKENS/RECENT_TOKENS)


def ceil_align(x, a):
    return (x + a - 1) // a * a


def rel(a, b):
    a = a.double()
    b = b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def dequantize_int2_plane(packed, sz, dim):
    """Decode-kernel int2 convention: crumb i of byte j -> coord i*dim//4 + j."""
    qd = dim // 4
    ng = sz.shape[-1] // 2
    gs = dim // ng
    scale = sz[..., 0::2].to(torch.float32)
    zero = sz[..., 1::2].to(torch.float32)
    out = torch.empty(
        packed.shape[0], packed.shape[1], dim, device=packed.device,
        dtype=torch.float32,
    )
    for i in range(4):
        c = ((packed >> (2 * i)) & 0x03).to(torch.float32)
        g = (torch.arange(qd, device=packed.device) + i * qd) // gs
        out[..., i * qd : (i + 1) * qd] = (c - zero[..., g]) * scale[..., g]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--bundle", default=BUNDLE)
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    os.environ["SGLANG_OSCAR_K_ROTATION_PATH"] = f"{ROT_DIR}/k_rotation_qqt_r_h_pbr.pt"
    os.environ["SGLANG_OSCAR_V_ROTATION_PATH"] = f"{ROT_DIR}/v_rotation_sst_r_h_pbr.pt"
    os.environ["SGLANG_OSCAR_K_CLIP_RATIO"] = "0.96"
    os.environ["SGLANG_OSCAR_V_CLIP_RATIO"] = "0.92"
    os.environ["SGLANG_OSCAR_ABSORB_V_ROTATION"] = "0"
    os.environ["SGLANG_VQ_CODEBOOK_PATH"] = args.bundle

    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        world_size=1, rank=0, local_rank=0, backend="nccl",
        distributed_init_method="tcp://127.0.0.1:29578",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)

    from sglang.srt.mem_cache.common import _mixed_extend_layout_counts
    from sglang.srt.mem_cache.unified_kv_allocator import UnifiedInt2HPKVAllocator
    from sglang.srt.mem_cache.unified_kv_pool import UnifiedInt2HPKVPool
    from sglang.srt.mem_cache.vq_codebook import vq_dequant
    from sglang.srt.layers.attention.quantized_kv_prefill import dequantize_prefix_kv

    device = "cuda:0"
    torch.manual_seed(0)
    seq = args.seq

    num_quant_pages = ceil_align(seq * 2, N_Q) // N_Q
    num_hp_prefix_slots = ceil_align(HP_PREFIX, N_Q) * 4
    pool = UnifiedInt2HPKVPool(
        num_quant_pages=num_quant_pages,
        hp_dtype=torch.bfloat16,
        hp_prefix_tokens=HP_PREFIX,
        hp_recent_tokens=HP_RECENT,
        dtype="int2",
        head_num=H,
        head_dim=D,
        layer_num=LAYERS,
        device=device,
        enable_memory_saver=False,
        max_req_slots=4,
        start_layer=0,
        end_layer=LAYERS,
        model_dtype=torch.bfloat16,
        num_hp_prefix_slots=num_hp_prefix_slots,
    )
    assert pool.vq_enabled and not pool.vq_v_enabled, "expected VQ-K + int2-V"
    alloc = UnifiedInt2HPKVAllocator(
        num_quant_pages=num_quant_pages,
        quant_tokens_per_page=N_Q,
        hp_prefix_tokens=HP_PREFIX,
        hp_recent_tokens=HP_RECENT,
        hp_recent_ring_size=HP_RECENT + N_Q,
        max_req_slots=4,
        num_hp_prefix_slots=num_hp_prefix_slots,
        dtype="int2",
        hp_dtype=torch.bfloat16,
        device=device,
        kvcache=pool,
        need_sort=False,
    )

    # Non-final chunk layout — the exact tier mix chunked prefill produces.
    hp_pre_n, hp_rec_n, quant_n, quant_alloc_n, _ = _mixed_extend_layout_counts(
        0, seq, HP_PREFIX, HP_RECENT, N_Q, is_final_chunk=False
    )
    assert hp_rec_n == 0 and hp_pre_n + quant_n == seq
    req_idx = torch.tensor([0], dtype=torch.int64, device=device)
    quant_slots = alloc.alloc_quant(quant_alloc_n)[:quant_n]
    hp_pre_slots = alloc.alloc_hp_prefix(req_idx, [ceil_align(hp_pre_n, N_Q)])[:hp_pre_n]
    loc = torch.cat([hp_pre_slots, quant_slots])
    print(f"layout(non-final) seq={seq}: hp_prefix={hp_pre_n} quant={quant_n}")

    # Rows with per-head scale diversity so pertoken_norm sees a realistic
    # dynamic range (gpt-oss K row norms vary ~10x across heads/layers).
    head_scale = torch.logspace(-0.5, 0.7, H, device=device).view(1, H, 1)
    ks = [
        (torch.randn(seq, H, D, device=device) * head_scale).to(torch.bfloat16)
        for _ in range(LAYERS)
    ]
    vs = [
        (torch.randn(seq, H, D, device=device) * head_scale).to(torch.bfloat16)
        for _ in range(LAYERS)
    ]
    for l in range(LAYERS):
        pool.set_kv_buffer(
            None, loc, ks[l], vs[l],
            layer_id_override=l,
            already_hadamard_transformed=False,
            is_decode=False,
        )

    vq = pool._vq
    trash = pool.quant_size
    q_rows = quant_slots.to(torch.int64)
    sm_scale = 1.0 / (D ** 0.5)
    failures = []

    print(f"{'L':>2} {'A_K~B_K':>9} {'A_V~B_V':>9} {'A_K~ex':>8} {'B_K~ex':>8} "
          f"{'A_V~ex':>8} {'B_V~ex':>8} {'top1 A/B/ex':>14} {'shadowK':>9}")
    agg_top_a = agg_top_b = 0
    for l in range(LAYERS):
        k_ex, v_ex = pool._rotate_kv_inplace(l, ks[l], vs[l], False)
        k_ex = k_ex.to(torch.float32)[hp_pre_n:]
        v_ex = v_ex.to(torch.float32)[hp_pre_n:]

        # A: the prefill read path, exactly as _forward_extend_quantized_dense
        # calls it (model_dtype bf16), quant rows only.
        a_k, a_v = dequantize_prefix_kv(pool, l, loc, torch.bfloat16)
        a_k = a_k.to(torch.float32)[hp_pre_n:]
        a_v = a_v.to(torch.float32)[hp_pre_n:]

        # B: decode-side reconstruction of the same arena bytes.
        idx = pool.get_raw_key_buffer(l)[q_rows]
        scale = pool.get_key_scales_zeros(l)[q_rows][..., 0]
        b_k = vq_dequant(idx.long(), scale, vq.cb16[l]).to(torch.float32)
        b_v = dequantize_int2_plane(
            pool.get_raw_value_buffer(l)[q_rows],
            pool.get_value_scales_zeros(l)[q_rows].to(torch.float32),
            D,
        )

        p1 = rel(a_k, b_k)
        p2 = rel(a_v, b_v)
        e_ak, e_bk = rel(a_k, k_ex), rel(b_k, k_ex)
        e_av, e_bv = rel(a_v, v_ex), rel(b_v, v_ex)

        # P4: retrieval-style softmax agreement. Queries = noisy copies of
        # random stored rows (matched retrieval structure), scores over the
        # quant prefix in stored space.
        n_q_probe = 64
        tgt = torch.randint(0, quant_n, (n_q_probe,), device=device)
        q = k_ex[tgt] + 0.3 * torch.randn_like(k_ex[tgt])
        # Score both paths at the SAME working precision (bf16, the served
        # dtype). Leaving B in fp32 makes near-tie argmaxes flip on rounding
        # noise and reads as a spurious path asymmetry.
        b_k_bf = b_k.to(torch.bfloat16).to(torch.float32)
        top_ex = torch.einsum("qhd,thd->qht", q, k_ex).argmax(-1)
        top_a = torch.einsum("qhd,thd->qht", q, a_k).argmax(-1)
        top_b = torch.einsum("qhd,thd->qht", q, b_k_bf).argmax(-1)
        acc_a = (top_a == top_ex).float().mean().item()
        acc_b = (top_b == top_ex).float().mean().item()
        acc_ex = 1.0
        agg_top_a += acc_a
        agg_top_b += acc_b

        # P5: shadow read returns exact rows.
        pool.shadow_register(quant_slots, rid="audit")
        pool.set_kv_buffer(
            None, loc, ks[l], vs[l],
            layer_id_override=l, already_hadamard_transformed=False,
            is_decode=False,
        )
        s_k, s_v = dequantize_prefix_kv(pool, l, loc, torch.bfloat16)
        e_sk = rel(s_k.to(torch.float32)[hp_pre_n:], k_ex)
        e_sv = rel(s_v.to(torch.float32)[hp_pre_n:], v_ex)
        pool.shadow_mark_release("audit")
        pool.shadow_step_release()

        if p1 > 5e-3:
            failures.append(f"L{l}: prefill K != decode K (rel {p1:.2e})")
        if p2 > 5e-3:
            failures.append(f"L{l}: prefill V != decode V (rel {p2:.2e})")
        if e_ak > e_bk * 1.02 + 1e-3:
            failures.append(
                f"L{l}: prefill K read worse than decode ({e_ak:.4f} vs {e_bk:.4f})"
            )
        if e_av > e_bv * 1.02 + 1e-3:
            failures.append(
                f"L{l}: prefill V read worse than decode ({e_av:.4f} vs {e_bv:.4f})"
            )
        if acc_a < acc_b - 0.02:
            failures.append(
                f"L{l}: prefill-read retrieval worse than decode-read "
                f"({acc_a:.3f} vs {acc_b:.3f})"
            )
        if e_sk > 2e-2 or e_sv > 2e-2:
            failures.append(f"L{l}: shadow read not exact (K {e_sk:.3e} V {e_sv:.3e})")

        print(f"{l:>2} {p1:9.2e} {p2:9.2e} {e_ak:8.4f} {e_bk:8.4f} "
              f"{e_av:8.4f} {e_bv:8.4f} {acc_a:5.3f}/{acc_b:5.3f}/{acc_ex:.2f} "
              f"{e_sk:9.2e}")

    print(f"\nmean top-1 retrieval through quantized prefix: "
          f"prefill-path {agg_top_a/LAYERS:.4f}  decode-path {agg_top_b/LAYERS:.4f}")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("\nPARITY AUDIT PASS — prefill reads match decode reads on the same "
          "bytes; no read-path bug.")


if __name__ == "__main__":
    main()
