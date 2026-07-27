"""Flush-plan accounting when only SOME requests flush — a bs>1-only case.

Empirically the served failure needs concurrency: 48 sequential requests are
clean, 12 concurrent ones produce non-finite logits. The index scatter is
verified correct at bs>1 (verify_mixed_indices_bs.py), so the remaining bs>1
surface is the decode-time flush.

`_alloc_for_decode_mixed` over-provisions `bs * N_Q` quant slots every step
because it cannot know which requests will flush, then returns the unused ones
through `plan.returned_slot_ids` -> `allocator.free(...)`. At bs=1 a request
either flushes or it does not; the MIXED case -- some requests flushing while
others do not, in the same step, sharing one over-provisioned block -- exists
only at bs>1 and no gate covers it.

The invariant under test is the one whose violation produces exactly the
observed symptom:

    a slot returned to the free pool must NOT still be referenced by
    req_to_token

Break it and the slot is handed to another request while the first still reads
it: two requests share KV, one of them reads values written for the other, and
the resulting logits are garbage. It is also the shape of the ~2 slot/req leak.

  PYTHONPATH=vendor/OSCAR-vq/sglang-research/python:. \
      .venv-oscar/bin/python pipelines/oscar_e2e/verify_flush_plan_bs.py --gpu 3
"""

import argparse
import os

import torch

LAYERS, H, D, N_Q = 12, 8, 64, 8
HP_PREFIX, HP_RECENT = 64, 256


def ceil_align(x, a):
    return ((int(x) + a - 1) // a) * a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--seq", type=int, default=1024)
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ROT = os.path.join(REPO, "artifacts/oscar_gptoss20b/rotations_gpqa198")
    os.environ["SGLANG_OSCAR_K_ROTATION_PATH"] = f"{ROT}/k_rotation_qqt_r_h_pbr.pt"
    os.environ["SGLANG_OSCAR_V_ROTATION_PATH"] = f"{ROT}/v_rotation_sst_r_h_pbr.pt"
    os.environ["SGLANG_OSCAR_K_CLIP_RATIO"] = "0.96"
    os.environ["SGLANG_OSCAR_V_CLIP_RATIO"] = "0.92"
    os.environ["SGLANG_OSCAR_ABSORB_V_ROTATION"] = "0"

    from sglang.srt.distributed import (
        init_distributed_environment, initialize_model_parallel)
    init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="nccl",
                                 distributed_init_method="tcp://127.0.0.1:29593")
    initialize_model_parallel(tensor_model_parallel_size=1)

    from sglang.QuantKernel.gpu_flush_int2 import (
        gpu_flush_int2_apply, gpu_flush_int2_plan)
    from sglang.srt.mem_cache.common import _mixed_extend_layout_counts
    from sglang.srt.mem_cache.unified_kv_allocator import UnifiedInt2HPKVAllocator
    from sglang.srt.mem_cache.unified_kv_pool import UnifiedInt2HPKVPool

    device = "cuda:0"
    torch.manual_seed(0)
    bs, seq = args.bs, args.seq
    pages = max(64, ceil_align(seq * bs * 3, N_Q) // N_Q)
    hp_prefix_slots = ceil_align(HP_PREFIX, N_Q) * (bs + 2)
    max_req = bs + 2

    pool = UnifiedInt2HPKVPool(
        num_quant_pages=pages, hp_dtype=torch.bfloat16, hp_prefix_tokens=HP_PREFIX,
        hp_recent_tokens=HP_RECENT, dtype="int2", head_num=H, head_dim=D,
        layer_num=LAYERS, device=device, enable_memory_saver=False,
        max_req_slots=max_req, start_layer=0, end_layer=LAYERS,
        model_dtype=torch.bfloat16, num_hp_prefix_slots=hp_prefix_slots)
    alloc = UnifiedInt2HPKVAllocator(
        num_quant_pages=pages, quant_tokens_per_page=N_Q, hp_prefix_tokens=HP_PREFIX,
        hp_recent_tokens=HP_RECENT, hp_recent_ring_size=HP_RECENT + N_Q,
        max_req_slots=max_req, num_hp_prefix_slots=hp_prefix_slots, dtype="int2",
        hp_dtype=torch.bfloat16, device=device, kvcache=pool, need_sort=False)

    max_ctx = seq + args.steps + 16
    rtt = torch.zeros((max_req, max_ctx), dtype=torch.int32, device=device)
    reqs = torch.arange(bs, dtype=torch.int64, device=device)

    # --- prefill each request ------------------------------------------------
    hp_n, rec_n, q_n, q_alloc, counter_init = _mixed_extend_layout_counts(
        0, seq, HP_PREFIX, HP_RECENT, N_Q, is_final_chunk=True)
    for i in range(bs):
        r = reqs[i : i + 1]
        qs = alloc.alloc_quant(q_alloc)
        hp = alloc.alloc_hp_prefix(r, [ceil_align(hp_n, N_Q)])
        rc = alloc.alloc_hp_recent(r, [rec_n])
        loc = torch.cat([hp[:hp_n], qs[:q_n], rc[:rec_n]])
        rtt[i, :seq] = loc.to(torch.int32)
    print(f"bs={bs} seq={seq}: hp_prefix={hp_n} quant={q_n} hp_recent={rec_n}")

    # Stagger the per-request flush counters so that on most steps SOME requests
    # flush and others do not -- the bs>1-only case this gate exists for.
    ctr = pool._flush_counter
    for i in range(bs):
        ctr[i] = (counter_init + i * (N_Q // max(bs, 1))) % N_Q

    cur = torch.full((bs,), seq, dtype=torch.int64, device=device)
    failures, mixed_steps = [], 0

    for step in range(args.steps):
        counters = ctr[reqs]
        flush_mask = counters == 0
        ctr[reqs] = torch.where(flush_mask, torch.full_like(counters, N_Q - 1),
                                counters - 1)
        nflush = int(flush_mask.sum())
        if 0 < nflush < bs:
            mixed_steps += 1

        step_loc = alloc.alloc_hp_recent(reqs, [1] * bs)
        dst = alloc.alloc_quant(bs * N_Q)
        if dst is None:
            failures.append(f"step {step}: alloc_quant({bs*N_Q}) exhausted")
            break

        plan = gpu_flush_int2_plan(
            seq_lens=cur.to(torch.int32), prefix_lens=torch.zeros(bs, dtype=torch.int32, device=device),
            req_pool_indices=reqs, dst_quant_slots=dst, req_to_token=rtt,
            flush_mask=flush_mask, hp_prefix_tokens=pool.hp_prefix_tokens,
            hp_recent_tokens=pool.hp_recent_tokens,
            hp_global_offset=pool.hp_global_offset, flush_interval=N_Q)

        if plan is not None:
            # The remap is what makes the invariant true: it rewrites
            # req_to_token from the demoted HP slot to its new quant slot.
            # Checking before applying it just re-reads the pre-flush mapping
            # and reports every demoted slot as a violation.
            gpu_flush_int2_apply(
                plan, req_pool_indices=reqs, req_to_token=rtt,
                hp_k_ptrs=pool._flush_hp_k_ptrs, hp_v_ptrs=pool._flush_hp_v_ptrs,
                quant_k_ptrs=pool._flush_quant_k_ptrs,
                quant_v_ptrs=pool._flush_quant_v_ptrs,
                k_sz_ptrs=pool._flush_k_sz_ptrs, v_sz_ptrs=pool._flush_v_sz_ptrs,
                hp_k_sample=pool.hp_k_buffer[0], hp_v_sample=pool.hp_v_buffer[0],
                quant_k_sample=pool.k_buffer[0], quant_v_sample=pool.v_buffer[0],
                k_sz_sample=pool.k_scales_zeros[0], v_sz_sample=pool.v_scales_zeros[0],
                hp_k_strides=pool._flush_hp_k_stride, hp_v_strides=pool._flush_hp_v_stride,
                quant_k_strides=pool._flush_quant_k_stride,
                quant_v_strides=pool._flush_quant_v_stride,
                k_sz_strides=pool._flush_k_sz_stride, v_sz_strides=pool._flush_v_sz_stride,
                num_heads=pool.head_num, head_dim=pool.head_dim,
                v_head_dim=pool.v_head_dim,
                k_num_scale_groups=pool.k_num_scale_groups,
                v_num_scale_groups=pool.v_num_scale_groups,
                num_layers=pool.layer_num, k_clip_ratio=pool._k_clip_ratio,
                v_clip_ratio=pool._v_clip_ratio, k_vq=False, v_vq=False)
            torch.cuda.synchronize()
            returned = plan.returned_slot_ids
            # THE INVARIANT: nothing handed back to the free pool may still be
            # referenced by any request's req_to_token.
            live = set()
            for i in range(bs):
                live.update(rtt[i, : int(cur[i])].tolist())
            ret = [s for s in returned.tolist() if s >= 0]
            clash = sorted(set(ret) & live)
            if clash:
                failures.append(
                    f"step {step} (flushing {nflush}/{bs}): {len(clash)} freed "
                    f"slot(s) STILL referenced by req_to_token, e.g. {clash[:5]}")
                if len(failures) >= 3:
                    break
            # a returned slot must not appear twice
            if len(ret) != len(set(ret)):
                dup = len(ret) - len(set(ret))
                failures.append(f"step {step}: {dup} duplicate id(s) in returned_slot_ids")

        for i in range(bs):
            rtt[i, int(cur[i])] = int(step_loc[i])
        cur += 1

    print(f"ran {args.steps} steps; {mixed_steps} had a MIX of flushing and "
          f"non-flushing requests")
    print()
    if failures:
        print(f"FAIL ({len(failures)} violation(s)):")
        for f in failures[:5]:
            print(f"  {f}")
        return 1
    print("ALL PASS - no freed slot was still referenced, no duplicate returns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
