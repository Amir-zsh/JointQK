#!/usr/bin/env python3
"""Stream Qwen3-8B prefill q/k over a calibration split and accumulate the SAME
uncentered, GQA-sum-pooled second moments that run_pca_ec_deadzone.calib_moments
returns -- but over hundreds of prompts, pooled-only (no per-token retention), so
the 400-prompt basis matches the Llama EC study's n400 protocol without ~1TB of
per-example captures.

Convention (bit-for-bit calib_moments): sigma_q = sum_over_group E[q q^T] per
kv-head (L,Hkv,d,d); sigma_k = E[k k^T] (L,Hkv,d,d); k_mean = E[k]; k_cov =
E[kk^T] - E[k]E[k]^T. Emits a drop-in {sigma_q, sigma_k, k_mean, k_cov, meta}.

Sharded by prompt across GPUs; each shard writes float64 partial sums
(sumq,sumk,sq2,sk2,ntok); `--merge` finalizes. T-chunked einsum caps peak GPU mem
so 32k-token compact8 prompts fit alongside the model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

from pipelines.calibration.capture_raw import run_prefill_qkv_capture  # noqa: E402
from pipelines.calibration.common import read_json, rows_for_shard, select_rows, validate_split  # noqa: E402
from kvq.data.base import fetch_example  # noqa: E402
from kvq.data.kvpress_adapter import build_kvpress_dataset_spec  # noqa: E402
from kvq.capture.model import get_model_device, load_model_and_tokenizer  # noqa: E402

TCHUNK = 4096  # token-block size for the moment einsums (peak-mem cap)


def _accumulate(q_post, k_post, acc, dev):
    """q_post (L,Hq,T,d) fp16 CPU, k_post (L,Hkv,T,d) fp16 CPU -> update acc in fp64 on dev."""
    L, Hq, T, d = q_post.shape
    Hkv = k_post.shape[1]
    if acc["sumq"] is None:
        acc["sumq"] = torch.zeros(L, Hq, d, dtype=torch.float64, device=dev)
        acc["sumk"] = torch.zeros(L, Hkv, d, dtype=torch.float64, device=dev)
        acc["sq2"] = torch.zeros(L, Hq, d, d, dtype=torch.float64, device=dev)
        acc["sk2"] = torch.zeros(L, Hkv, d, d, dtype=torch.float64, device=dev)
        acc["shape"] = (L, Hq, Hkv, d)
    for s in range(0, T, TCHUNK):
        q = q_post[:, :, s:s + TCHUNK, :].to(dev, non_blocking=True).double()
        k = k_post[:, :, s:s + TCHUNK, :].to(dev, non_blocking=True).double()
        acc["sumq"] += q.sum(2)
        acc["sumk"] += k.sum(2)
        acc["sq2"] += torch.einsum("lhtd,lhte->lhde", q, q)
        acc["sk2"] += torch.einsum("lhtd,lhte->lhde", k, k)
        del q, k
    acc["ntok"] += T


def capture_shard(args):
    dev = "cuda"
    split = read_json(args.split_manifest)
    validate_split(split)
    rows = rows_for_shard(select_rows(split, smoke=False), args.num_shards, args.shard_id)
    # basis corpus = TRAIN rows only (compact8 train == the JointQK calibration split)
    rows = [r for r in rows if r.get("split", "train") == "train"]
    spec = build_kvpress_dataset_spec(
        name=args.dataset, config_names=tuple(split["config"]["tasks"]),
        metadata_fields=("task", "answers", "length", "all_classes"))
    model, tok = load_model_and_tokenizer(args.model, device_map="auto", dtype_name="float16")
    dev = get_model_device(model)
    print(f"[shard {args.shard_id}/{args.num_shards}] {len(rows)} train rows | model on {dev}", flush=True)

    acc = {"sumq": None, "sumk": None, "sq2": None, "sk2": None, "ntok": 0, "shape": None}
    seen = []
    for i, row in enumerate(rows):
        ex = fetch_example(spec, row["config"], int(row["row_index"]), tok)
        out = run_prefill_qkv_capture(model, ex.input_ids)
        _accumulate(out["q_post"], out["k_post"], acc, dev)
        seen.append({"config": row["config"], "row_index": int(row["row_index"]),
                     "split": row.get("split", "train"), "T": int(out["q_post"].shape[2])})
        if (i + 1) % 10 == 0 or i == len(rows) - 1:
            print(f"  [shard {args.shard_id}] {i+1}/{len(rows)} ntok={acc['ntok']}", flush=True)
        del out
    L, Hq, Hkv, d = acc["shape"]
    torch.save({"sumq": acc["sumq"].cpu(), "sumk": acc["sumk"].cpu(),
                "sq2": acc["sq2"].cpu(), "sk2": acc["sk2"].cpu(), "ntok": acc["ntok"],
                "shape": acc["shape"], "seen": seen,
                "n_layers": L, "n_q_heads": Hq, "n_kv_heads": Hkv, "d_head": d},
               args.out)
    print(f"[shard {args.shard_id}] saved {args.out} | ntok={acc['ntok']} rows={len(seen)}", flush=True)


def merge(args):
    parts = [torch.load(p, map_location="cpu", weights_only=False) for p in args.shards]
    sumq = sum(p["sumq"].double() for p in parts)
    sumk = sum(p["sumk"].double() for p in parts)
    sq2 = sum(p["sq2"].double() for p in parts)
    sk2 = sum(p["sk2"].double() for p in parts)
    ntok = sum(p["ntok"] for p in parts)
    L, Hq, Hkv, d = parts[0]["shape"]
    gs = Hq // Hkv
    Eqq = sq2 / ntok
    Ekk = sk2 / ntok
    mk = sumk / ntok
    sigma_q = Eqq.reshape(L, Hkv, gs, d, d).sum(2)       # GQA sum-pool -> (L,Hkv,d,d)
    sigma_k = Ekk
    k_mean = mk
    k_cov = Ekk - torch.einsum("lhd,lhe->lhde", mk, mk)
    meta = dict(n_layers=L, n_q_heads=Hq, n_kv_heads=Hkv, d_head=d, group_size=gs)
    seen = [s for p in parts for s in p["seen"]]
    torch.save({"sigma_q": sigma_q.float(), "sigma_k": sigma_k.float(),
                "k_mean": k_mean.float(), "k_cov": k_cov.float(),
                "meta": meta, "ntok": ntok, "n_prompts": len(seen), "seen": seen},
               args.out)
    # min-eig sanity on Sigma_Q over deployed layers (l>=1), like the EC pipeline expects
    sq = 0.5 * (sigma_q[1:] + sigma_q[1:].transpose(-1, -2))
    mineig = torch.linalg.eigvalsh(sq.double()).min().item()
    print(f"MERGED -> {args.out}", flush=True)
    print(f"  prompts={len(seen)} ntok={ntok} | L={L} Hq={Hq} Hkv={Hkv} d={d} (GQA {gs})", flush=True)
    print(f"  sigma_q min-eig (l>=1) = {mineig:.3e} (PSD expected)", flush=True)
    tasks = {}
    for s in seen:
        tasks[s["config"]] = tasks.get(s["config"], 0) + 1
    print(f"  per-task prompt counts: {tasks}", flush=True)


def codepool(args):
    """Capture k_post per example (query_stats format) for the EC/VQ code fit. Only
    k_post + prompt_length are needed by _codes_for_idx, so q/v are dropped (cheap).
    Shardable: each shard writes ex_{gid:04d}.pt at the row's GLOBAL index (no
    collisions); run `codepool_merge` after to build manifest.json in global order."""
    import json
    split = read_json(args.split_manifest)
    all_rows = split["rows"] if "rows" in split else split["examples"]
    idxs = list(range(args.shard_id, len(all_rows), args.num_shards))
    spec = build_kvpress_dataset_spec(
        name=args.dataset, config_names=tuple(sorted({r["config"] for r in all_rows})),
        metadata_fields=("task", "answers", "length", "all_classes"))
    model, tok = load_model_and_tokenizer(args.model, device_map="auto", dtype_name="float16")
    out = Path(args.out); (out / "examples").mkdir(parents=True, exist_ok=True)
    for n, gid in enumerate(idxs):
        row = all_rows[gid]
        ex = fetch_example(spec, row["config"], int(row["row_index"]), tok)
        input_ids = ex.input_ids
        if args.max_tokens and input_ids.shape[-1] > args.max_tokens:
            input_ids = input_ids[..., :args.max_tokens]   # cap seq len (bounds q_post storage)
        o = run_prefill_qkv_capture(model, input_ids)
        T = int(o["k_post"].shape[2])
        rec = {"k_post": o["k_post"].to(torch.float16), "prompt_length": T,
               "config": row["config"], "row_index": int(row["row_index"])}
        if args.save_q:                      # q_post too, for attention-KL fine-tuning
            rec["q_post"] = o["q_post"].to(torch.float16)
        torch.save(rec, out / f"examples/ex_{gid:04d}.pt")
        print(f"  [codepool s{args.shard_id}] {n+1}/{len(idxs)} gid={gid} {row['config']} T={T}", flush=True)
        del o
    print(f"CODEPOOL shard {args.shard_id} done ({len(idxs)} examples)", flush=True)


def codepool_merge(args):
    """Build manifest.json from all ex_*.pt in <out>/examples (global sorted order)."""
    import json
    out = Path(args.out)
    files = sorted((out / "examples").glob("ex_*.pt"))
    examples = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        examples.append({"file": f"examples/{f.name}", "prompt_length": int(d["prompt_length"]),
                         "config": d.get("config"), "row_index": d.get("row_index")})
    json.dump({"examples": examples, "num_examples": len(examples)},
              open(out / "manifest.json", "w"), indent=0)
    print(f"CODEPOOL merged {out} | {len(examples)} examples", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    cp = sub.add_parser("codepool")
    cp.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    cp.add_argument("--dataset", default="longbench")
    cp.add_argument("--split-manifest", required=True)
    cp.add_argument("--out", required=True)
    cp.add_argument("--num-shards", type=int, default=1)
    cp.add_argument("--shard-id", type=int, default=0)
    cp.add_argument("--save-q", action="store_true", help="also save q_post (for attention-KL fine-tuning)")
    cp.add_argument("--max-tokens", type=int, default=0, help="truncate prompt to first N tokens (0=off)")
    cpm = sub.add_parser("codepool_merge")
    cpm.add_argument("--out", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--model", default="Qwen/Qwen3-8B")
    c.add_argument("--dataset", default="longbench")
    c.add_argument("--split-manifest", required=True)
    c.add_argument("--num-shards", type=int, default=4)
    c.add_argument("--shard-id", type=int, required=True)
    c.add_argument("--out", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--shards", nargs="+", required=True)
    m.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "capture":
        capture_shard(args)
    elif args.cmd == "codepool":
        codepool(args)
    elif args.cmd == "codepool_merge":
        codepool_merge(args)
    else:
        merge(args)
