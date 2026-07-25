"""End-to-end offline reproduction of the gpt-oss int2 decode, pool included.

Two existing gates each cover half of the path and both pass:
  verify_gptoss_int2_decode.py  kernel correctness on synthetic buffers
  verify_gptoss_mixed_pool.py   pool write/flush correctness, content only

Neither joins them: nothing checks the attention output the decode kernel
produces *from buffers the pool actually wrote*, using the hp/quant index
split the backend actually builds. A bug in that seam — index classification,
split accounting, or the two-tier stage-2 merge — is invisible to both.

vq2 shares the pool, the accessors and the index construction and serves
correctly, so if this reproduces the collapse the fault is in the int2 tier's
own read; if it passes, the fault is upstream in the live serving path
(prefill/extend, chunking, or per-request state) rather than in the decode.

Production parameters are used deliberately: bf16 scale dtype and
--triton-attention-num-kv-splits 8 (int2's default; vq2 runs 48).

  PYTHONPATH=vendor/OSCAR-vq/sglang-research/python:. \
      .venv-oscar/bin/python pipelines/oscar_e2e/verify_gptoss_int2_e2e.py --gpu 1
"""

import argparse
import os

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ROT_DIR = os.path.join(REPO, "artifacts/oscar_gptoss20b/rotations_gpqa198")
LAYERS, H, D, N_Q = 12, 8, 64, 8
HP_PREFIX, HP_RECENT = 64, 256
QH = 64  # gpt-oss attention heads (kv_group_num = 8)


def ceil_align(x, a):
    return ((int(x) + a - 1) // a) * a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--hp-splits", type=int, default=8)
    ap.add_argument("--quant-splits", type=int, default=8)
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    os.environ["SGLANG_OSCAR_K_ROTATION_PATH"] = f"{ROT_DIR}/k_rotation_qqt_r_h_pbr.pt"
    os.environ["SGLANG_OSCAR_V_ROTATION_PATH"] = f"{ROT_DIR}/v_rotation_sst_r_h_pbr.pt"
    os.environ["SGLANG_OSCAR_K_CLIP_RATIO"] = "0.96"
    os.environ["SGLANG_OSCAR_V_CLIP_RATIO"] = "0.92"
    os.environ["SGLANG_OSCAR_ABSORB_V_ROTATION"] = "0"

    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        world_size=1, rank=0, local_rank=0, backend="nccl",
        distributed_init_method="tcp://127.0.0.1:29581",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)

    from sglang.srt.layers.attention.triton_ops.decode_attention import (
        decode_attention_fwd_int2_unified,
    )
    from sglang.srt.mem_cache.common import _mixed_extend_layout_counts
    from sglang.srt.mem_cache.unified_kv_allocator import UnifiedInt2HPKVAllocator
    from sglang.srt.mem_cache.unified_kv_pool import UnifiedInt2HPKVPool

    from pipelines.oscar_e2e.verify_gptoss_int2_decode import dequantize_int2

    device = "cuda:0"
    torch.manual_seed(0)
    seq, L = args.seq, args.layer

    num_quant_pages = max(8, ceil_align(seq * 2, N_Q) // N_Q)
    num_hp_prefix_slots = ceil_align(HP_PREFIX, N_Q) * 4
    pool = UnifiedInt2HPKVPool(
        num_quant_pages=num_quant_pages, hp_dtype=torch.bfloat16,
        hp_prefix_tokens=HP_PREFIX, hp_recent_tokens=HP_RECENT, dtype="int2",
        head_num=H, head_dim=D, layer_num=LAYERS, device=device,
        enable_memory_saver=False, max_req_slots=4, start_layer=0,
        end_layer=LAYERS, model_dtype=torch.bfloat16,
        num_hp_prefix_slots=num_hp_prefix_slots,
    )
    alloc = UnifiedInt2HPKVAllocator(
        num_quant_pages=num_quant_pages, quant_tokens_per_page=N_Q,
        hp_prefix_tokens=HP_PREFIX, hp_recent_tokens=HP_RECENT,
        hp_recent_ring_size=HP_RECENT + N_Q, max_req_slots=4,
        num_hp_prefix_slots=num_hp_prefix_slots, dtype="int2",
        hp_dtype=torch.bfloat16, device=device, kvcache=pool, need_sort=False,
    )

    hp_pre_n, hp_rec_n, quant_n, quant_alloc_n, _ = _mixed_extend_layout_counts(
        0, seq, HP_PREFIX, HP_RECENT, N_Q, is_final_chunk=True
    )
    req = torch.tensor([0], dtype=torch.int64, device=device)
    qs = alloc.alloc_quant(quant_alloc_n)
    hp_p = alloc.alloc_hp_prefix(req, [ceil_align(hp_pre_n, N_Q)])
    hp_r = alloc.alloc_hp_recent(req, [hp_rec_n])
    loc = torch.cat([hp_p[:hp_pre_n], qs[:quant_n], hp_r[:hp_rec_n]])
    print(f"seq={seq}: hp_prefix={hp_pre_n} quant={quant_n} hp_recent={hp_rec_n}")

    k = torch.randn(seq, H, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(seq, H, D, device=device, dtype=torch.bfloat16)
    pool.set_kv_buffer(None, loc, k, v, layer_id_override=L,
                       already_hadamard_transformed=False, is_decode=False)

    # Index split exactly as _build_mixed_kv_indices does it: classify each
    # position's slot id against hp_global_offset, preserving position order
    # within each tier.
    hp_off = pool.hp_global_offset
    is_hp = loc >= hp_off
    # The two tiers use different index spaces: _scatter_mixed_kv_indices_kernel
    # stores `slot - HP_OFFSET` for HP (get_hp_key_buffer is a local view) and
    # the raw `slot` for quant (k_buffer is indexed globally). Mixing them up
    # silently reads the wrong rows.
    hp_idx = (loc[is_hp] - hp_off).contiguous()
    q_idx = loc[~is_hp].contiguous()
    hp_indptr = torch.tensor([0, hp_idx.numel()], dtype=torch.int64, device=device)
    q_indptr = torch.tensor([0, q_idx.numel()], dtype=torch.int64, device=device)

    total = args.hp_splits + args.quant_splits
    logits = torch.zeros(1, QH, total, D, device=device, dtype=torch.float32)
    lse = torch.full((1, QH, total), float("-inf"), device=device, dtype=torch.float32)
    q_dec = torch.randn(1, QH, D, device=device, dtype=torch.bfloat16)
    sinks = torch.randn(QH, device=device, dtype=torch.float32)
    o = torch.zeros(1, QH, D, device=device, dtype=torch.bfloat16)
    sm_scale = 1.0 / (D ** 0.5)

    decode_attention_fwd_int2_unified(
        q_dec,
        pool.get_hp_key_buffer(L), pool.get_hp_value_buffer(L),
        pool.get_raw_key_buffer(L), pool.get_raw_value_buffer(L),
        pool.get_key_scales_zeros(L), pool.get_value_scales_zeros(L),
        o, hp_indptr, hp_idx, q_indptr, q_idx, logits, lse,
        torch.full((1,), args.hp_splits, device=device, dtype=torch.int32),
        torch.full((1,), args.quant_splits, device=device, dtype=torch.int32),
        args.hp_splits, args.quant_splits, sm_scale, logit_cap=0.0, sinks=sinks,
    )

    # Reference over the same slots, in the same tier order the kernel consumes.
    hp_k = pool.get_hp_key_buffer(L)[hp_idx].float()
    hp_v = pool.get_hp_value_buffer(L)[hp_idx].float()
    qk = dequantize_int2(pool.get_raw_key_buffer(L)[q_idx],
                         pool.get_key_scales_zeros(L)[q_idx].float(), D)
    qv = dequantize_int2(pool.get_raw_value_buffer(L)[q_idx],
                         pool.get_value_scales_zeros(L)[q_idx].float(), D)
    worst = 0.0
    for qh in range(0, QH, 4):
        h = qh // (QH // H)
        kk = torch.cat([hp_k[:, h], qk[:, h]])
        vv = torch.cat([hp_v[:, h], qv[:, h]])
        s = (q_dec[0, qh].float() @ kk.T) * sm_scale
        s = torch.cat([s, sinks[qh].view(1)])
        p = torch.softmax(s, -1)[:-1]
        ref = p @ vv
        e = ((o[0, qh].float() - ref).norm() / ref.norm().clamp_min(1e-9)).item()
        worst = max(worst, e)
    ok = worst <= 2e-2
    print(f"unified decode over pool-written buffers: worst rel {worst:.3e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
