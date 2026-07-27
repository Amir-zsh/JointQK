"""Verify the HP/quant index split at bs>1 — the one path no gate covers.

Empirically the served failure needs concurrency: 48 sequential requests are
clean, 12 concurrent ones produce non-finite logits. Every existing gate runs
bs=1 (or hand-built indices), so the per-request scatter has never been checked
against a reference at bs>1.

`_build_mixed_kv_indices` walks each request's req_to_token row, classifies each
slot as HP (`slot >= hp_global_offset`) or quant, and scatters the two streams
into shared buffers at offsets given by cumsum'd per-request lengths. A bug in
those offsets is invisible at bs=1 (one request, offset 0) and corrupts which
KV a request attends to at bs>1 -- which is exactly the observed signature.

The reference is deliberately dumb: a Python loop doing the same classification
per request. Any disagreement is the engine's.

  PYTHONPATH=vendor/OSCAR-vq/sglang-research/python:. \
      .venv-oscar/bin/python pipelines/oscar_e2e/verify_mixed_indices_bs.py --gpu 3
"""

import argparse
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--max-bs", type=int, default=8)
    ap.add_argument("--trials", type=int, default=40)
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        world_size=1, rank=0, local_rank=0, backend="nccl",
        distributed_init_method="tcp://127.0.0.1:29591",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)

    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

    device = "cuda:0"
    torch.manual_seed(0)
    HP_OFFSET = 16384          # matches the gpt-oss pool geometry
    MAX_CTX = 4096
    NUM_REQ = 16

    # A bare object carrying only what _build_mixed_kv_indices touches.
    class Shim:
        pass

    shim = Shim()
    shim.req_to_token = torch.zeros((NUM_REQ, MAX_CTX), dtype=torch.int32, device=device)
    shim.mixed_hp_global_offset = HP_OFFSET
    shim._build_mixed_kv_indices = TritonAttnBackend._build_mixed_kv_indices.__get__(shim)

    failures = []
    print(f"{'bs':>3} {'seq lens':>26} {'hp/quant split':>18}  result")
    for trial in range(args.trials):
        bs = int(torch.randint(1, args.max_bs + 1, (1,)).item())
        seq_lens = torch.randint(64, 1024, (bs,), device=device, dtype=torch.int64)
        req_pool = torch.randperm(NUM_REQ, device=device)[:bs].to(torch.int64)

        # Populate req_to_token with a realistic mixed layout per request:
        # [HP-prefix][quant middle][HP-recent], slot ids unique across requests.
        shim.req_to_token.zero_()
        expect_hp, expect_q = [], []
        base_q, base_h = 1, HP_OFFSET + 1
        for i in range(bs):
            T = int(seq_lens[i])
            npre = min(64, T)
            nrec = min(256, max(0, T - npre))
            nq = T - npre - nrec
            slots = []
            for _ in range(npre):
                slots.append(base_h); base_h += 1
            for _ in range(nq):
                slots.append(base_q); base_q += 1
            for _ in range(nrec):
                slots.append(base_h); base_h += 1
            shim.req_to_token[int(req_pool[i]), :T] = torch.tensor(
                slots, dtype=torch.int32, device=device)
            expect_hp.append([s - HP_OFFSET for s in slots if s >= HP_OFFSET])
            expect_q.append([s for s in slots if s < HP_OFFSET])

        total = int(seq_lens.sum())
        hp_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        q_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        hp_idx = torch.full((total,), -1, dtype=torch.int64, device=device)
        q_idx = torch.full((total,), -1, dtype=torch.int64, device=device)
        shim._build_mixed_kv_indices(req_pool, seq_lens, hp_indptr, hp_idx,
                                     q_indptr, q_idx, bs)
        torch.cuda.synchronize()

        ok = True
        detail = ""
        for i in range(bs):
            h0, h1 = int(hp_indptr[i]), int(hp_indptr[i + 1])
            q0, q1 = int(q_indptr[i]), int(q_indptr[i + 1])
            got_hp = hp_idx[h0:h1].tolist()
            got_q = q_idx[q0:q1].tolist()
            if got_hp != expect_hp[i]:
                ok = False
                detail = (f"req{i} HP mismatch: got {len(got_hp)} want "
                          f"{len(expect_hp[i])}; first diff at "
                          f"{next((k for k,(a,b) in enumerate(zip(got_hp, expect_hp[i])) if a!=b), 'len')}")
                break
            if got_q != expect_q[i]:
                ok = False
                detail = (f"req{i} QUANT mismatch: got {len(got_q)} want "
                          f"{len(expect_q[i])}; first diff at "
                          f"{next((k for k,(a,b) in enumerate(zip(got_q, expect_q[i])) if a!=b), 'len')}")
                break
        # every destination slot must have been written
        if ok and ((hp_idx[: int(hp_indptr[bs])] < 0).any()
                   or (q_idx[: int(q_indptr[bs])] < 0).any()):
            ok = False
            detail = "unwritten (-1) entries inside the used range"

        if not ok:
            failures.append((bs, detail))
        if trial < 12 or not ok:
            lens = ",".join(str(int(x)) for x in seq_lens[: min(bs, 4)])
            print(f"{bs:>3} {lens:>26} {int(hp_indptr[bs]):>8}/{int(q_indptr[bs]):<9} "
                  f"{'PASS' if ok else 'FAIL  ' + detail}")

    print()
    if failures:
        print(f"FAIL: {len(failures)}/{args.trials} trials disagree with the reference")
        for bs, d in failures[:5]:
            print(f"  bs={bs}: {d}")
        return 1
    print(f"ALL PASS ({args.trials} trials, bs 1..{args.max_bs})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
