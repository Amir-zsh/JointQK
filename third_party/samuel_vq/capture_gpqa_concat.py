#!/usr/bin/env python3
"""OSCAR-faithful *long-position* calibration: concatenate the same 198 GPQA-Diamond prompts
into one long ~30k-token input (or a few of them), so the model sees keys at positions 0-30k
instead of 0-~150. Same corpus, same total token budget as capture_gpqa.py, same OSCAR-style
prompt template — only the delivery is long-context. This gives the codebook RoPE-position
coverage matching the eval regime without leaving GPQA's domain.

Emits a drop-in {sigma_q, sigma_k, k_mean, k_cov, meta} basis + a query_stats-format pool
consumable by train_group_vq_alloc.py and build_method_bundles.py."""
import argparse, json, os, sys
from pathlib import Path
import pandas as pd
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # our depth: third_party/samuel_vq/
import _bootstrap  # noqa
from pipelines.calibration.capture_raw import run_prefill_qkv_capture
from kvq.capture.model import get_model_device, load_model_and_tokenizer

QUERY_TEMPLATE_MULTICHOICE = (
"Answer the following multiple choice question. The last line of your response should be "
"of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. "
"Think step by step before answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
)
TCHUNK = 4096


def accumulate(q, k, acc, dev):
    L, Hq, T, d = q.shape; Hkv = k.shape[1]
    if acc["sumq"] is None:
        acc["sumq"] = torch.zeros(L, Hq, d, dtype=torch.float64, device=dev)
        acc["sumk"] = torch.zeros(L, Hkv, d, dtype=torch.float64, device=dev)
        acc["sq2"] = torch.zeros(L, Hq, d, d, dtype=torch.float64, device=dev)
        acc["sk2"] = torch.zeros(L, Hkv, d, d, dtype=torch.float64, device=dev)
        acc["shape"] = (L, Hq, Hkv, d)
    for s in range(0, T, TCHUNK):
        qc = q[:, :, s:s + TCHUNK, :].to(dev, non_blocking=True).double()
        kc = k[:, :, s:s + TCHUNK, :].to(dev, non_blocking=True).double()
        acc["sumq"] += qc.sum(2); acc["sumk"] += kc.sum(2)
        acc["sq2"] += torch.einsum("lhtd,lhte->lhde", qc, qc)
        acc["sk2"] += torch.einsum("lhtd,lhte->lhde", kc, kc)
        del qc, kc
    acc["ntok"] += T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--csv", default="artifacts/gpqa_diamond/gpqa_diamond.csv")
    ap.add_argument("--target-ctx", type=int, default=32768, help="target tokens per concat sequence")
    ap.add_argument("--n-sequences", type=int, default=0, help="if >0, emit exactly N sequences of EXACTLY target-ctx tokens by cycling the corpus (repetition allowed). Default 0 = pack greedy, no repetition.")
    ap.add_argument("--out-basis", required=True)
    ap.add_argument("--out-pool", required=True)
    ap.add_argument("--pool-stride", type=int, default=4, help="subsample k_post positions for k-means pool")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    prompts = [QUERY_TEMPLATE_MULTICHOICE.format(
        Question=r["Question"], A=r["Correct Answer"],
        B=r["Incorrect Answer 1"], C=r["Incorrect Answer 2"], D=r["Incorrect Answer 3"]) for _, r in df.iterrows()]
    print(f"loaded {len(prompts)} GPQA-Diamond prompts", flush=True)

    model, tok = load_model_and_tokenizer(args.model, device_map="auto", dtype_name="float16")
    dev = get_model_device(model)

    # Pre-tokenize each prompt with a "\n\n---\n\n" delimiter between them
    delim_ids = tok("\n\n---\n\n", add_special_tokens=False)["input_ids"]
    prompt_ids = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    print(f"per-prompt tokens: min/med/max = {min(map(len,prompt_ids))}/{sorted(map(len,prompt_ids))[len(prompt_ids)//2]}/{max(map(len,prompt_ids))}", flush=True)

    if args.n_sequences > 0:
        # Build one long cyclic token stream, then chunk into exactly n_sequences × target_ctx tokens.
        # Each sequence starts at a different offset in the cyclic stream to add some variety.
        cyclic = []
        for pids in prompt_ids:
            cyclic += (delim_ids if cyclic else []) + pids
        total = len(cyclic)
        # If corpus < target_ctx, we must repeat the stream to fill each sequence.
        reps_needed = args.target_ctx // total + 2
        expanded = cyclic * reps_needed
        sequences = []
        for i in range(args.n_sequences):
            offset = (i * total) % len(expanded)
            seq = expanded[offset:offset + args.target_ctx]
            if len(seq) < args.target_ctx:  # ran off end, wrap
                seq = seq + expanded[:args.target_ctx - len(seq)]
            sequences.append(seq)
        print(f"cycled corpus into {len(sequences)} × {args.target_ctx} sequences (corpus_toks={total}, repeats/seq≈{args.target_ctx/total:.1f})", flush=True)
    else:
        # Pack prompts greedily into sequences of up to target-ctx tokens (no repetition)
        sequences, cur = [], []
        for pids in prompt_ids:
            add = (delim_ids + pids) if cur else pids
            if cur and len(cur) + len(add) > args.target_ctx:
                sequences.append(cur); cur = pids
            else:
                cur += add
        if cur: sequences.append(cur)
        print(f"packed into {len(sequences)} sequences, lengths: {[len(s) for s in sequences]}", flush=True)

    acc = {"sumq": None, "sumk": None, "sq2": None, "sk2": None, "ntok": 0, "shape": None}
    pool = Path(args.out_pool); (pool / "examples").mkdir(parents=True, exist_ok=True)
    examples = []
    for i, seq_ids in enumerate(sequences):
        ids = torch.tensor([seq_ids], dtype=torch.long)
        out = run_prefill_qkv_capture(model, ids)
        accumulate(out["q_post"], out["k_post"], acc, dev)
        kp = out["k_post"][:, :, ::args.pool_stride, :].to(torch.float16).cpu()
        T2 = int(kp.shape[2])
        torch.save({"k_post": kp, "prompt_length": T2, "config": "gpqa_concat", "row_index": i},
                   pool / f"examples/ex_{i:04d}.pt")
        examples.append({"file": f"examples/ex_{i:04d}.pt", "prompt_length": T2,
                         "config": "gpqa_concat", "row_index": i})
        print(f"  [{i+1}/{len(sequences)}] ctx={len(seq_ids)} pool_T={T2} ntok={acc['ntok']}", flush=True)
        del out
    json.dump({"examples": examples, "num_examples": len(examples)},
              open(pool / "manifest.json", "w"), indent=0)

    L, Hq, Hkv, d = acc["shape"]; gs = Hq // Hkv; ntok = acc["ntok"]
    Eqq = acc["sq2"] / ntok; Ekk = acc["sk2"] / ntok; mk = acc["sumk"] / ntok
    sigma_q = Eqq.reshape(L, Hkv, gs, d, d).sum(2)
    k_cov = Ekk - torch.einsum("lhd,lhe->lhde", mk, mk)
    meta = dict(n_layers=L, n_q_heads=Hq, n_kv_heads=Hkv, d_head=d, group_size=gs)
    torch.save({"sigma_q": sigma_q.float().cpu(), "sigma_k": Ekk.float().cpu(),
                "k_mean": mk.float().cpu(), "k_cov": k_cov.float().cpu(),
                "meta": meta, "ntok": ntok, "n_prompts": len(sequences)}, args.out_basis)
    sq = 0.5 * (sigma_q[1:] + sigma_q[1:].transpose(-1, -2))
    mineig = torch.linalg.eigvalsh(sq.double()).min().item()
    print(f"SAVED basis {args.out_basis} | ntok={ntok} L={L} Hq={Hq} Hkv={Hkv} d={d} | sigma_q min-eig(l>=1)={mineig:.3e}", flush=True)
    print(f"SAVED pool {args.out_pool} | {len(sequences)} sequences", flush=True)


if __name__ == "__main__":
    main()
