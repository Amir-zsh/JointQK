"""Round-trip gate for the mixed HP+int2 pool at gpt-oss shapes.

The decode kernel and both sink corrections are verified elsewhere
(``verify_gptoss_int2_decode.py``) — they are numerically exact. What is not
covered by any gate is what the *write* path leaves in the arena: an 8K
prefill splits into [HP-prefix 64][int2 middle ~7.9K][HP-recent 256], and a
position that lands in the wrong slot, is never written, or is stored in a
rotation basis the decode query does not share, produces exactly the observed
failure — fluent local text (the SWA layers and the HP band are intact) with
zero long-range retrieval.

The gate is a correspondence test, not a fidelity test. int2 is lossy by
design, so absolute error says little; what must hold is that reconstructed
slot p still matches *true position p* better than any other position. That
is precisely the needle property NIAH measures.

Run:
  PYTHONPATH=vendor/OSCAR-vq/sglang-research/python \
      .venv-oscar/bin/python pipelines/oscar_e2e/verify_gptoss_mixed_pool.py --gpu 1
"""

import argparse
import os

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ROT_DIR = os.path.join(REPO, "artifacts/oscar_gptoss20b/rotations_gpqa198")

# gpt-oss-20B full-attention side: 12 dense layers, 8 kv heads, head_dim 64.
LAYERS, H, D, N_Q = 12, 8, 64, 8
HP_PREFIX, HP_RECENT = 64, 256


def ceil_align(x, a):
    return ((int(x) + a - 1) // a) * a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--probes", type=int, default=256)
    ap.add_argument("--decode-steps", type=int, default=64)
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    os.environ["SGLANG_OSCAR_K_ROTATION_PATH"] = f"{ROT_DIR}/k_rotation_qqt_r_h_pbr.pt"
    os.environ["SGLANG_OSCAR_V_ROTATION_PATH"] = f"{ROT_DIR}/v_rotation_sst_r_h_pbr.pt"
    os.environ["SGLANG_OSCAR_K_CLIP_RATIO"] = "0.96"
    os.environ["SGLANG_OSCAR_V_CLIP_RATIO"] = "0.92"
    os.environ["SGLANG_OSCAR_ABSORB_V_ROTATION"] = "0"

    # The pool shards the VQ codebook by TP rank, so it needs a TP group even
    # for a single-process gate.
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        world_size=1, rank=0, local_rank=0, backend="nccl",
        distributed_init_method="tcp://127.0.0.1:29577",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)

    from sglang.srt.mem_cache.common import _mixed_extend_layout_counts
    from sglang.srt.mem_cache.unified_kv_allocator import UnifiedInt2HPKVAllocator
    from sglang.srt.mem_cache.unified_kv_pool import UnifiedInt2HPKVPool

    device = "cuda:0"
    torch.manual_seed(0)
    seq = args.seq
    L = args.layer

    num_quant_pages = ceil_align(seq * 2, N_Q) // N_Q
    num_hp_prefix_slots = ceil_align(HP_PREFIX, N_Q) * 4
    max_req_slots = 4
    hp_recent_ring = HP_RECENT + N_Q

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
        max_req_slots=max_req_slots,
        start_layer=0,
        end_layer=LAYERS,
        model_dtype=torch.bfloat16,
        num_hp_prefix_slots=num_hp_prefix_slots,
    )
    alloc = UnifiedInt2HPKVAllocator(
        num_quant_pages=num_quant_pages,
        quant_tokens_per_page=N_Q,
        hp_prefix_tokens=HP_PREFIX,
        hp_recent_tokens=HP_RECENT,
        hp_recent_ring_size=hp_recent_ring,
        max_req_slots=max_req_slots,
        num_hp_prefix_slots=num_hp_prefix_slots,
        dtype="int2",
        hp_dtype=torch.bfloat16,
        device=device,
        kvcache=pool,
        need_sort=False,
    )

    # --- replicate _alloc_for_extend_mixed's layout for one final-chunk req --
    hp_pre_n, hp_rec_n, quant_n, quant_alloc_n, counter_init = _mixed_extend_layout_counts(
        0, seq, HP_PREFIX, HP_RECENT, N_Q, is_final_chunk=True
    )
    print(f"layout seq={seq}: hp_prefix={hp_pre_n} quant={quant_n} "
          f"hp_recent={hp_rec_n} (sum {hp_pre_n + quant_n + hp_rec_n})")
    assert hp_pre_n + quant_n + hp_rec_n == seq, "layout does not cover the sequence"

    req_idx = torch.tensor([0], dtype=torch.int64, device=device)
    quant_slots = alloc.alloc_quant(quant_alloc_n)
    hp_pre_slots = alloc.alloc_hp_prefix(req_idx, [ceil_align(hp_pre_n, N_Q)])
    hp_rec_slots = alloc.alloc_hp_recent(req_idx, [hp_rec_n])
    assert quant_slots is not None, "quant allocation failed"
    loc = torch.cat(
        [hp_pre_slots[:hp_pre_n], quant_slots[:quant_n], hp_rec_slots[:hp_rec_n]]
    )
    assert loc.numel() == seq, f"out_cache_loc {loc.numel()} != seq {seq}"

    failures = []
    uniq = torch.unique(loc).numel()
    if uniq != seq:
        failures.append(f"slot collision: {seq - uniq} duplicate slot ids in loc")
    print(f"slot ids: {uniq}/{seq} unique, hp_global_offset={pool.hp_global_offset}")

    # --- write one layer's K/V through the real pool write path -------------
    k = torch.randn(seq, H, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(seq, H, D, device=device, dtype=torch.bfloat16)
    pool.set_kv_buffer(
        None, loc, k, v,
        layer_id_override=L,
        already_hadamard_transformed=False,  # pool applies the OSCAR rotation
        is_decode=False,
    )

    # --- read back exactly as the decode path sees it -----------------------
    # Stored space is rotated: quant tier dequantizes to k @ R_k, HP tier holds
    # k @ R_k in bf16. The decode query is rotated by the same R_k, so the
    # comparison target is the rotated truth.
    R_k = pool._R_k[L].to(torch.float32)
    k_true = (k.to(torch.float32) @ R_k)

    from pipelines.oscar_e2e.verify_gptoss_int2_decode import dequantize_int2

    hp_off = pool.hp_global_offset
    is_hp = loc >= hp_off
    k_hat = torch.zeros(seq, H, D, device=device, dtype=torch.float32)
    if is_hp.any():
        hp_rows = (loc[is_hp] - hp_off).to(torch.int64)
        k_hat[is_hp] = pool.get_hp_key_buffer(L)[hp_rows].to(torch.float32)
    if (~is_hp).any():
        q_rows = loc[~is_hp].to(torch.int64)
        k_hat[~is_hp] = dequantize_int2(
            pool.get_raw_key_buffer(L)[q_rows],
            pool.get_key_scales_zeros(L)[q_rows].to(torch.float32),
            D,
        )

    # --- correspondence: does slot p still identify position p? -------------
    torch.manual_seed(1)
    probes = torch.randperm(seq, device=device)[: args.probes]
    for tier_name, mask in (("hp", is_hp), ("quant", ~is_hp)):
        sel = probes[mask[probes]]
        if sel.numel() == 0:
            continue
        # score every probe's reconstruction against all true positions, per head
        hits, cos_self = 0, []
        for h in range(H):
            s = k_hat[sel, h] @ k_true[:, h].T  # [n_probe, seq]
            hits += (s.argmax(dim=-1) == sel).sum().item()
            cos_self.append(
                torch.nn.functional.cosine_similarity(
                    k_hat[sel, h], k_true[sel, h], dim=-1
                ).mean().item()
            )
        total = sel.numel() * H
        acc = hits / total
        mean_cos = sum(cos_self) / len(cos_self)
        tol = 0.98 if tier_name == "hp" else 0.90
        ok = acc >= tol
        if not ok:
            failures.append(
                f"{tier_name} tier: self-retrieval {acc:.3f} < {tol} "
                f"(mean cos {mean_cos:.3f}) — stored content does not match "
                f"its own position"
            )
        print(f"  {tier_name:5s} tier: n={sel.numel():4d} self-retrieval@1 "
              f"{acc:.4f}  mean cos(k_hat, k_true) {mean_cos:.4f}  "
              f"{'PASS' if ok else 'FAIL'}")

    # --- stage 2: do the prefill positions survive decode-time flushes? -----
    # Each flush demotes an aged HP-recent block into a fresh quant page and
    # rewrites req_to_token. The documented failure mode is a page returned to
    # the free pool while still referenced — it then gets handed out again and
    # overwrites live K/V. Prefill positions are what a needle lives in, so
    # re-run the correspondence check against req_to_token after N steps.
    if args.decode_steps > 0:
        from sglang.QuantKernel.gpu_flush_int2 import (
            gpu_flush_int2_apply,
            gpu_flush_int2_plan,
        )

        max_ctx = seq + args.decode_steps + 8
        req_to_token = torch.zeros(
            (max_req_slots, max_ctx), dtype=torch.int32, device=device
        )
        req_to_token[0, :seq] = loc.to(torch.int32)
        # Production seeds this from the extend layout; starting at 0 instead
        # fires a premature flush whose demote window straddles the
        # quant/HP-recent boundary — a real failure mode, but not the one the
        # scheduler actually produces.
        flush_counter = torch.full(
            (max_req_slots,), counter_init, dtype=torch.int32, device=device
        )
        cur_len = seq
        n_flushes = 0

        for _ in range(args.decode_steps):
            counters = flush_counter[req_idx]
            flush_mask = counters == 0
            flush_counter[req_idx] = torch.where(
                flush_mask, torch.full_like(counters, N_Q - 1), counters - 1
            )
            step_loc = alloc.alloc_hp_recent(req_idx, [1])
            dst_quant = alloc.alloc_quant(N_Q)
            assert dst_quant is not None, "decode quant allocation failed"
            plan = gpu_flush_int2_plan(
                seq_lens=torch.tensor([cur_len], dtype=torch.int32, device=device),
                prefix_lens=torch.zeros(1, dtype=torch.int32, device=device),
                req_pool_indices=req_idx,
                dst_quant_slots=dst_quant,
                req_to_token=req_to_token,
                flush_mask=flush_mask,
                hp_prefix_tokens=pool.hp_prefix_tokens,
                hp_recent_tokens=pool.hp_recent_tokens,
                hp_global_offset=pool.hp_global_offset,
                flush_interval=N_Q,
            )
            if plan is not None:
                n_flushes += 1
                alloc.free(plan.returned_slot_ids)
                gpu_flush_int2_apply(
                    plan,
                    req_pool_indices=req_idx,
                    req_to_token=req_to_token,
                    hp_k_ptrs=pool._flush_hp_k_ptrs,
                    hp_v_ptrs=pool._flush_hp_v_ptrs,
                    quant_k_ptrs=pool._flush_quant_k_ptrs,
                    quant_v_ptrs=pool._flush_quant_v_ptrs,
                    k_sz_ptrs=pool._flush_k_sz_ptrs,
                    v_sz_ptrs=pool._flush_v_sz_ptrs,
                    hp_k_sample=pool.hp_k_buffer[0],
                    hp_v_sample=pool.hp_v_buffer[0],
                    quant_k_sample=pool.k_buffer[0],
                    quant_v_sample=pool.v_buffer[0],
                    k_sz_sample=pool.k_scales_zeros[0],
                    v_sz_sample=pool.v_scales_zeros[0],
                    hp_k_strides=pool._flush_hp_k_stride,
                    hp_v_strides=pool._flush_hp_v_stride,
                    quant_k_strides=pool._flush_quant_k_stride,
                    quant_v_strides=pool._flush_quant_v_stride,
                    k_sz_strides=pool._flush_k_sz_stride,
                    v_sz_strides=pool._flush_v_sz_stride,
                    num_heads=pool.head_num,
                    head_dim=pool.head_dim,
                    v_head_dim=pool.v_head_dim,
                    k_num_scale_groups=pool.k_num_scale_groups,
                    v_num_scale_groups=pool.v_num_scale_groups,
                    num_layers=pool.layer_num,
                    k_clip_ratio=pool._k_clip_ratio,
                    v_clip_ratio=pool._v_clip_ratio,
                    k_vq=False,
                    v_vq=False,
                )
            # the new token's own K/V (content irrelevant; only its slot must
            # not collide with a live prefill slot)
            pool.set_kv_buffer(
                None, step_loc,
                torch.randn(1, H, D, device=device, dtype=torch.bfloat16),
                torch.randn(1, H, D, device=device, dtype=torch.bfloat16),
                layer_id_override=L, already_hadamard_transformed=False,
                is_decode=True,
            )
            req_to_token[0, cur_len] = step_loc.to(torch.int32)
            cur_len += 1

        print(f"\nafter {args.decode_steps} decode steps ({n_flushes} flushes):")
        post = req_to_token[0, :seq].to(torch.int64)
        uniq_post = torch.unique(post).numel()
        if uniq_post != seq:
            failures.append(
                f"post-decode slot collision: {seq - uniq_post} prefill positions "
                f"now share a slot (a live page was freed and re-handed-out)"
            )
        print(f"  prefill slot ids: {uniq_post}/{seq} unique")
        if uniq_post != seq:
            vals, counts = torch.unique(post, return_counts=True)
            for s in vals[counts > 1].tolist():
                pos = (post == s).nonzero().flatten().tolist()
                tier = "HP" if s >= hp_off else "quant"
                print(f"    slot {s} ({tier}) shared by positions {pos[:8]}")

        is_hp2 = post >= hp_off
        k_hat2 = torch.zeros(seq, H, D, device=device, dtype=torch.float32)
        if is_hp2.any():
            k_hat2[is_hp2] = pool.get_hp_key_buffer(L)[
                (post[is_hp2] - hp_off)
            ].to(torch.float32)
        if (~is_hp2).any():
            rows = post[~is_hp2]
            k_hat2[~is_hp2] = dequantize_int2(
                pool.get_raw_key_buffer(L)[rows],
                pool.get_key_scales_zeros(L)[rows].to(torch.float32),
                D,
            )
        all_pos = torch.arange(seq, device=device)
        hits = 0
        for h in range(H):
            for c0 in range(0, seq, 2048):
                c = all_pos[c0 : c0 + 2048]
                s = k_hat2[c, h] @ k_true[:, h].T
                hits += (s.argmax(dim=-1) == c).sum().item()
        acc2 = hits / (seq * H)
        ok2 = acc2 >= 0.90
        if not ok2:
            failures.append(
                f"post-decode self-retrieval {acc2:.3f} < 0.90 — prefill K/V was "
                f"corrupted by the flush path"
            )
        print(f"  prefill self-retrieval@1 {acc2:.4f}  {'PASS' if ok2 else 'FAIL'}")

    print()
    if failures:
        for f in failures:
            print("FAIL:", f)
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
