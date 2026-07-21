#!/usr/bin/env python3
"""Long-position calibration capture over a mixed-domain segment corpus.

Adapted from third_party/samuel_vq/capture_gpqa_concat.py (Samuel's
OSCAR-faithful concat capture — see third_party/samuel_vq/PROVENANCE.md);
only the corpus construction differs: instead of the GPQA CSV, consumes the
{domain, text} jsonl from build_mixed_corpus.py, shuffles segments with a
fixed seed so domains interleave inside every sequence, and cycles the
stream into exactly n_sequences x target_ctx tokens. Emits the identical
{sigma_q, sigma_k, k_mean, k_cov, meta} basis + query_stats-format pool that
train_group_vq_alloc.py consumes.

    .venv/bin/python pipelines/oscar_e2e/capture_mixed_concat.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --corpus artifacts/oscar_llama31_8b/mixed_corpus.jsonl \
        --target-ctx 65536 --n-sequences 8 \
        --out-basis <basis.pt> --out-pool <pool_dir>
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import _bootstrap  # noqa: E402,F401
from pipelines.calibration.capture_raw import run_prefill_qkv_capture  # noqa: E402
from kvq.capture.model import get_model_device, load_model_and_tokenizer  # noqa: E402

TCHUNK = 4096


def accumulate(q, k, acc, dev):
    L, Hq, T, d = q.shape
    Hkv = k.shape[1]
    if acc["sumq"] is None:
        acc["sumq"] = torch.zeros(L, Hq, d, dtype=torch.float64, device=dev)
        acc["sumk"] = torch.zeros(L, Hkv, d, dtype=torch.float64, device=dev)
        acc["sq2"] = torch.zeros(L, Hq, d, d, dtype=torch.float64, device=dev)
        acc["sk2"] = torch.zeros(L, Hkv, d, d, dtype=torch.float64, device=dev)
        acc["shape"] = (L, Hq, Hkv, d)
    for s in range(0, T, TCHUNK):
        qc = q[:, :, s:s + TCHUNK, :].to(dev, non_blocking=True).double()
        kc = k[:, :, s:s + TCHUNK, :].to(dev, non_blocking=True).double()
        acc["sumq"] += qc.sum(2)
        acc["sumk"] += kc.sum(2)
        acc["sq2"] += torch.einsum("lhtd,lhte->lhde", qc, qc)
        acc["sk2"] += torch.einsum("lhtd,lhte->lhde", kc, kc)
        del qc, kc
    acc["ntok"] += T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--target-ctx", type=int, default=65536)
    ap.add_argument("--n-sequences", type=int, default=8)
    ap.add_argument("--pool-stride", type=int, default=4)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out-basis", required=True)
    ap.add_argument("--out-pool", required=True)
    args = ap.parse_args()

    segs = [json.loads(l) for l in open(REPO / args.corpus)]
    rng = random.Random(args.seed)
    rng.shuffle(segs)
    from collections import Counter
    print(f"loaded {len(segs)} segments {dict(Counter(s['domain'] for s in segs))}", flush=True)

    model, tok = load_model_and_tokenizer(args.model, device_map="auto",
                                          dtype_name="float16")
    dev = get_model_device(model)

    delim_ids = tok("\n\n---\n\n", add_special_tokens=False)["input_ids"]
    seg_ids = [tok(s["text"], add_special_tokens=False)["input_ids"] for s in segs]
    cyclic = []
    for pids in seg_ids:
        cyclic += (delim_ids if cyclic else []) + pids
    total = len(cyclic)
    reps = args.target_ctx * args.n_sequences / total
    expanded = cyclic * (args.target_ctx // total + 2)
    sequences = []
    for i in range(args.n_sequences):
        offset = (i * total) % len(expanded)
        seq = expanded[offset:offset + args.target_ctx]
        if len(seq) < args.target_ctx:
            seq = seq + expanded[:args.target_ctx - len(seq)]
        sequences.append(seq)
    print(f"corpus_toks={total}, {args.n_sequences} x {args.target_ctx} "
          f"sequences (total cycling {reps:.2f}x)", flush=True)

    acc = {"sumq": None, "sumk": None, "sq2": None, "sk2": None,
           "ntok": 0, "shape": None}
    pool = Path(args.out_pool)
    (pool / "examples").mkdir(parents=True, exist_ok=True)
    examples = []
    for i, ids_list in enumerate(sequences):
        ids = torch.tensor([ids_list], dtype=torch.long)
        out = run_prefill_qkv_capture(model, ids)
        accumulate(out["q_post"], out["k_post"], acc, dev)
        # .contiguous(): torch.save of a strided view persists the FULL
        # underlying storage — without it each example is 4x larger on disk.
        kp = (out["k_post"][:, :, ::args.pool_stride, :]
              .to(torch.float16).cpu().contiguous())
        T2 = int(kp.shape[2])
        torch.save({"k_post": kp, "prompt_length": T2,
                    "config": "mixed_concat", "row_index": i},
                   pool / f"examples/ex_{i:04d}.pt")
        examples.append({"file": f"examples/ex_{i:04d}.pt", "prompt_length": T2,
                         "config": "mixed_concat", "row_index": i})
        print(f"  [{i+1}/{len(sequences)}] ctx={len(ids_list)} pool_T={T2} "
              f"ntok={acc['ntok']}", flush=True)
        del out
    json.dump({"examples": examples, "num_examples": len(examples)},
              open(pool / "manifest.json", "w"), indent=0)

    L, Hq, Hkv, d = acc["shape"]
    gs = Hq // Hkv
    ntok = acc["ntok"]
    Eqq = acc["sq2"] / ntok
    Ekk = acc["sk2"] / ntok
    mk = acc["sumk"] / ntok
    sigma_q = Eqq.reshape(L, Hkv, gs, d, d).sum(2)
    k_cov = Ekk - torch.einsum("lhd,lhe->lhde", mk, mk)
    meta = dict(n_layers=L, n_q_heads=Hq, n_kv_heads=Hkv, d_head=d, group_size=gs)
    torch.save({"sigma_q": sigma_q.float().cpu(), "sigma_k": Ekk.float().cpu(),
                "k_mean": mk.float().cpu(), "k_cov": k_cov.float().cpu(),
                "meta": meta, "ntok": ntok, "n_prompts": len(sequences)},
               args.out_basis)
    sq = 0.5 * (sigma_q[1:] + sigma_q[1:].transpose(-1, -2))
    mineig = torch.linalg.eigvalsh(sq.double()).min().item()
    print(f"SAVED basis {args.out_basis} | ntok={ntok} L={L} Hq={Hq} Hkv={Hkv} "
          f"d={d} | sigma_q min-eig(l>=1)={mineig:.3e}", flush=True)
    print(f"SAVED pool {args.out_pool} | {len(sequences)} sequences", flush=True)


if __name__ == "__main__":
    main()
