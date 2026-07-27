"""Validate the opt-in CUDA vq2 stage-1 through the ENGINE's own entry point.

The standalone benchmark (`cuda_vq2_stage1.py`) feeds the kernel contiguous fp16
tensors. The engine does neither:

  * `att_out` / `att_lse` are SLICES of a [bs, heads, total_splits, L] scratch
    (`attn_logits[:, :, hp_max_kv_splits:, :]`), so the head stride is
    total_splits*L, not n_splits*L. A kernel that assumes contiguity writes the
    wrong cells -- silently, since the shapes still match.
  * q is bfloat16 for Qwen3, not fp16.

So this test drives `_decode_grouped_att_m_fwd_quant_vq2` itself, twice, with
SGLANG_VQ2_CUDA off then on, over the same inputs, and diffs the results. It
checks the *unwritten* region of the scratch too: a stride bug typically shows up
as corruption outside the quant slice rather than as a wrong value inside it.

  docker exec -e CUDA_VISIBLE_DEVICES=0 oscar-ab bash -lc 'cd <repo> && \
    PYTHONPATH=vendor/OSCAR-vq/sglang-research/python \
    /opt/venv-oscar/bin/python pipelines/throughput/kernel_study/test_vq2_cuda_engine.py'
"""
import os

import torch
import triton

BATCH = int(os.environ.get("BS", 8))
SEQ = int(os.environ.get("CTX", 4096))
HP_SPLITS = int(os.environ.get("HP_SPLITS", 4))     # leading splits owned by the HP tier
Q_SPLITS = int(os.environ.get("Q_SPLITS", 8))
H_Q, H_KV, L, NG, KC = 32, 8, 128, 32, 256
SM_SCALE = 1.0 / (L ** 0.5)


def build(dtype, dev="cuda", seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    total, cache = BATCH * SEQ, BATCH * SEQ + 64
    q = torch.randn(BATCH, H_Q, L, generator=g, device=dev, dtype=torch.float32).to(dtype)
    k_idx = torch.randint(0, KC, (cache, H_KV, NG), generator=g, device=dev, dtype=torch.uint8)
    cent = torch.randn(H_KV, NG, KC, 4, generator=g, device=dev, dtype=torch.float32) * (L ** -0.5)
    cb = cent.to(torch.float8_e5m2).view(torch.int32).squeeze(-1).contiguous()
    v_buf = torch.randint(0, 256, (cache, H_KV, L // 4), generator=g, device=dev, dtype=torch.uint8)
    k_sz = torch.rand(cache, H_KV, 2, generator=g, device=dev) * 0.5 + 0.75
    v_sz = torch.empty(cache, H_KV, 2, device=dev, dtype=torch.float32)
    v_sz[..., 0] = torch.rand(cache, H_KV, generator=g, device=dev) * 0.1 + 0.05
    v_sz[..., 1] = 1.5
    kv_indptr = torch.arange(0, BATCH + 1, device=dev, dtype=torch.int32) * SEQ
    # PAGING LOCALITY. The kernel study used a random permutation -- worst-case
    # scattered pages. A real server allocates pages largely in order, so the
    # K_Idx/V streams are near-contiguous. Which one you benchmark changes the
    # answer, so it is a knob here rather than an assumption.
    if os.environ.get("PAGING", "random") == "sequential":
        kv_indices = torch.arange(total, device=dev, dtype=torch.int32)
    else:
        kv_indices = torch.randperm(cache, generator=g, device=dev)[:total].to(torch.int32)
    splits = torch.empty(BATCH, device=dev, dtype=torch.int32)
    seq_lens = torch.full((BATCH,), SEQ, device=dev, dtype=torch.int32)
    from sglang.srt.layers.attention.triton_backend import get_num_kv_splits_triton
    cores = torch.cuda.get_device_properties(0).multi_processor_count
    get_num_kv_splits_triton[(1,)](
        splits, seq_lens, BATCH, 1, H_Q, H_KV, Q_SPLITS, cores,
        MAX_NUM_SEQ=256 if BATCH < 256 else triton.next_power_of_2(BATCH))
    return locals()


def run(d, use_cuda: bool):
    """Allocate the engine's combined scratch and drive only the quant slice."""
    total_splits = HP_SPLITS + Q_SPLITS
    # Sentinel, not zeros: any cell the kernel wrongly writes is then visible.
    logits = torch.full((BATCH, H_Q, total_splits, L), -7.5, device="cuda",
                        dtype=torch.float32)
    lse = torch.full((BATCH, H_Q, total_splits), float("-inf"), device="cuda",
                     dtype=torch.float32)
    q_logits = logits[:, :, HP_SPLITS:, :]      # non-contiguous view
    q_lse = lse[:, :, HP_SPLITS:]
    assert q_logits.stride(1) == total_splits * L, "expected the engine's stride"

    os.environ["SGLANG_VQ2_CUDA"] = "1" if use_cuda else "0"
    from sglang.srt.layers.attention.triton_ops import decode_attention as D
    D._decode_grouped_att_m_fwd_quant_vq2(
        d["q"], d["k_idx"], d["cb"], d["v_buf"], d["k_sz"], d["v_sz"],
        q_logits, q_lse, d["kv_indptr"], d["kv_indices"], d["splits"],
        Q_SPLITS, SM_SCALE, 0.0)
    torch.cuda.synchronize()
    return logits, lse


def main() -> int:
    print(f"bs={BATCH} ctx={SEQ} hp_splits={HP_SPLITS} quant_splits={Q_SPLITS}")
    ok_all = True
    # kv_indices is int64 in the engine and int32 in the standalone benchmark.
    # The first version of this test only covered int32, so a hardcoded
    # data_ptr<int>() slipped through and crashed mid CUDA-graph capture.
    for dtype, idx_dt in ((torch.float16, torch.int32),
                          (torch.bfloat16, torch.int32),
                          (torch.bfloat16, torch.int64)):
        d = build(dtype)
        d["kv_indices"] = d["kv_indices"].to(idx_dt)
        lo_t, ls_t = run(d, use_cuda=False)
        lo_c, ls_c = run(d, use_cuda=True)

        # 1. the quant slice must agree
        a = lo_t[:, :, HP_SPLITS:, :]
        b = lo_c[:, :, HP_SPLITS:, :]
        fin = torch.isfinite(ls_t[:, :, HP_SPLITS:])
        eo = (b - a).abs().max().item() / max(a.abs().max().item(), 1e-9)
        el = ((ls_c[:, :, HP_SPLITS:][fin] - ls_t[:, :, HP_SPLITS:][fin]).abs().max().item()
              / max(ls_t[:, :, HP_SPLITS:][fin].abs().max().item(), 1e-9))
        # 2. the HP region must be untouched by either path
        hp_t = (lo_t[:, :, :HP_SPLITS, :] == -7.5).all().item()
        hp_c = (lo_c[:, :, :HP_SPLITS, :] == -7.5).all().item()
        lse_hp = torch.isinf(ls_c[:, :, :HP_SPLITS]).all().item()
        ok = eo < 5e-3 and el < 5e-3 and hp_t and hp_c and lse_hp
        ok_all &= ok
        print(f"  q={str(dtype).split('.')[-1]:9s} kvidx={str(idx_dt).split('.')[-1]:5s} "
              f"out rel {eo:.2e}  lse rel {el:.2e}  "
              f"hp-region intact triton={hp_t} cuda={hp_c} lse={lse_hp}  "
              f"{'PASS' if ok else 'FAIL'}")
    # Time both paths through the same entry point, so an in-server speedup
    # that exceeds this ratio cannot be coming from the kernel.
    if os.environ.get("TIME", "1") == "1":
        import time
        d = build(torch.bfloat16)
        d["kv_indices"] = d["kv_indices"].to(torch.int64)
        for use_cuda in (False, True):
            run(d, use_cuda)                       # warm
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                run(d, use_cuda)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / 5 * 1e3
            print(f"  {'CUDA  ' if use_cuda else 'Triton'} stage-1: {ms:8.3f} ms/call")
    print("\n" + ("ALL PASS" if ok_all else "FAILURES -- do not benchmark this"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
