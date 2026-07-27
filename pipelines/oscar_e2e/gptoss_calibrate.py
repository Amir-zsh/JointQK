#!/usr/bin/env python3
"""gpt-oss calibration captures — full-attention layers only.

gpt-oss alternates sliding-window (128-token) and full-attention layers;
HF's cache truncates SWA layers' K/V to the window, so per-layer stacking
is ragged and — more importantly — SWA layers stay bf16 in our serving
policy (their KV is bounded; nothing to compress). Both subcommands
therefore capture ONLY the full-attention layers, indexed in pool-local
order (0..N_full-1), and write a layer_map.json recording local -> global.

  dump    per-prompt QKV dump (rotation calibration; compute_kv_rotation
          layout: <out>/layer_<local>/{q,k,v}/<chunk>.pt)
  concat  cycled long-context capture (codebook calibration; emits
          basis_moments + query_stats pool like capture_mixed_concat)

Both use the base transformer (no lm_head logits) and the model's chat
template for dump prompts (harmony format for gpt-oss).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import _bootstrap  # noqa: E402,F401
from pipelines.calibration.capture_raw import (  # noqa: E402
    _assemble_per_layer,
    capture_rope_qk,
)
from kvq.capture.model import get_model_device, load_model_and_tokenizer  # noqa: E402

TMPL = ("Answer the following multiple choice question. The last line of your response "
        "should be of the following format: 'Answer: $LETTER' (without quotes) where "
        "LETTER is one of ABCD. Think step by step before answering.\n\n{Question}\n\n"
        "A) {A}\nB) {B}\nC) {C}\nD) {D}")
TCHUNK = 4096


def full_layer_ids(model) -> list[int]:
    lt = model.config.layer_types
    return [i for i, t in enumerate(lt) if t == "full_attention"]


def capture_full_layers(model, input_ids, full_ids, prefill_chunk: int = 4096):
    """Like run_prefill_qkv_capture but returns only full-attention layers
    (SWA layers' cache is window-truncated and unused by our policy).
    Prefill is CHUNKED: a single 64K forward OOMs on MoE expert activations
    (transformers integrations.moe._grouped_linear intermediates scale with
    token count); 4K chunks with the cache threaded between calls keep
    activations bounded while the hooks concatenate q chunks per layer."""
    dev = get_model_device(model)
    input_ids = input_ids.to(dev)
    T = int(input_ids.shape[-1])
    n_layers = int(model.config.num_hidden_layers)
    with capture_rope_qk(model) as (_qpre, q_post_chunks, _kpre, _kpost):
        cache = None
        with torch.inference_mode():
            for s0 in range(0, T, prefill_chunk):
                out = model(input_ids=input_ids[:, s0:s0 + prefill_chunk],
                            past_key_values=cache, use_cache=True)
                cache = out.past_key_values
    q_all = _assemble_per_layer(q_post_chunks, n_layers)          # [L, Hq, T, d]
    ks, vs = [], []
    for l in full_ids:
        k = cache.layers[l].keys.detach().to("cpu", dtype=torch.float16).squeeze(0)
        v = cache.layers[l].values.detach().to("cpu", dtype=torch.float16).squeeze(0)
        assert k.shape[1] == T, f"layer {l}: cache T={k.shape[1]} != {T} (not a full layer?)"
        ks.append(k)
        vs.append(v)
    return {
        "q_post": q_all[full_ids].to(torch.float16),
        "k_post": torch.stack(ks),
        "v": torch.stack(vs),
        "prompt_length": T,
    }


def _checkpoint_is_mxfp4(model_id: str) -> bool:
    from transformers import AutoConfig
    qc = getattr(
        AutoConfig.from_pretrained(model_id, trust_remote_code=True),
        "quantization_config", None,
    )
    if qc is None:
        return False
    fmt = qc.get("quant_method") if isinstance(qc, dict) else getattr(
        qc, "quant_method", None
    )
    return str(fmt).lower().endswith("mxfp4")


def load(args):
    # Attention implementation, by elimination: gpt-oss has no sdpa
    # (_supports_sdpa=False — sinks), flex_attention's compiled path breaks
    # across a multi-GPU device_map (dynamo fake-tensor device mismatch),
    # and sinks must stay correct (they shape attention outputs and thus
    # every downstream layer's K statistics). So: EAGER + small prefill
    # chunks — per-chunk fp32 attention weights are [H, chunk, T] ≈ 8.6 GB
    # transient at chunk=512, T=64K.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # The official openai/gpt-oss-20b checkpoint ships MXFP4 MoE experts. Its
    # packed-weight path dies with an illegal memory access while materializing
    # across a multi-GPU device_map, and calibration wants bf16 statistics
    # anyway, so ask transformers to dequantize at load. Absent on a plain
    # bf16 checkpoint, where quantization_config=None leaves the load unchanged.
    quant_cfg = None
    device_map = "auto"
    if _checkpoint_is_mxfp4(args.model):
        from transformers import Mxfp4Config
        quant_cfg = Mxfp4Config(dequantize=True)
        # Even when dequantizing, sharding this checkpoint over a multi-GPU
        # device_map dies with an illegal memory access while materializing the
        # expert weights. Dequantized it is ~42 GB, which fits one 80 GB card
        # alongside the ~8.6 GB transient eager-attention weights, so pin it.
        device_map = {"": 0}
        print(f"model {args.model}: MXFP4 checkpoint -> dequantizing to bf16, "
              f"single-GPU device_map", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map=device_map, dtype=torch.bfloat16,
        attn_implementation="eager", trust_remote_code=True,
        quantization_config=quant_cfg)
    model.eval()
    base = model.model if hasattr(model, "model") else model
    full_ids = full_layer_ids(model)
    print(f"model {args.model}: {len(full_ids)} full-attention layers {full_ids}", flush=True)
    return base, tok, full_ids


def cmd_dump(args):
    base, tok, full_ids = load(args)
    df = pd.read_csv(args.csv)
    prompts = [TMPL.format(Question=r["Question"], A=r["Correct Answer"],
                           B=r["Incorrect Answer 1"], C=r["Incorrect Answer 2"],
                           D=r["Incorrect Answer 3"]) for _, r in df.iterrows()][: args.num_prompts]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"local_to_global": full_ids, "model": args.model},
              open(out / "layer_map.json", "w"))
    for pi, p in enumerate(prompts):
        text = tok.apply_chat_template([{"role": "user", "content": p}],
                                       add_generation_prompt=True, tokenize=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
        cap = capture_full_layers(base, ids, full_ids)
        chunk = pi + 1
        for li in range(len(full_ids)):
            for name, t in (("q", cap["q_post"][li]), ("k", cap["k_post"][li]),
                            ("v", cap["v"][li])):
                d = out / f"layer_{li}" / name
                d.mkdir(parents=True, exist_ok=True)
                torch.save(t.permute(1, 0, 2).contiguous(), d / f"{chunk}.pt")  # [T,H,d]
        if pi == 0:  # dummy chunk 0 (compute's "all" mode skips chunk 0)
            for li in range(len(full_ids)):
                for name, t in (("q", cap["q_post"][li]), ("k", cap["k_post"][li]),
                                ("v", cap["v"][li])):
                    torch.save(t.permute(1, 0, 2)[:6].contiguous(),
                               out / f"layer_{li}" / name / "0.pt")
        if (pi + 1) % 25 == 0:
            print(f"  dumped {pi+1}/{len(prompts)}", flush=True)
    print(f"DUMP DONE -> {out} ({len(prompts)} prompts, {len(full_ids)} local layers)", flush=True)


def cmd_concat(args):
    base, tok, full_ids = load(args)
    dev = get_model_device(base)
    segs = [json.loads(l) for l in open(REPO / args.corpus)]
    import random
    rng = random.Random(17)
    rng.shuffle(segs)
    delim = tok("\n\n---\n\n", add_special_tokens=False)["input_ids"]
    seg_ids = [tok(s["text"], add_special_tokens=False)["input_ids"] for s in segs]
    cyclic = []
    for pids in seg_ids:
        cyclic += (delim if cyclic else []) + pids
    total = len(cyclic)
    expanded = cyclic * (args.target_ctx // total + 2)
    sequences = []
    for i in range(args.n_sequences):
        off = (i * total) % len(expanded)
        seq = expanded[off:off + args.target_ctx]
        if len(seq) < args.target_ctx:
            seq += expanded[:args.target_ctx - len(seq)]
        sequences.append(seq)
    print(f"corpus_toks={total}, {args.n_sequences} x {args.target_ctx}", flush=True)

    L = len(full_ids)
    acc = {"sumq": None}
    pool = Path(args.out_pool)
    (pool / "examples").mkdir(parents=True, exist_ok=True)
    examples = []
    for i, ids_list in enumerate(sequences):
        ids = torch.tensor([ids_list], dtype=torch.long)
        cap = capture_full_layers(base, ids, full_ids, prefill_chunk=args.prefill_chunk)
        q, k = cap["q_post"], cap["k_post"]                    # [L,H,T,d] fp16
        Hq, Hkv, d = q.shape[1], k.shape[1], k.shape[-1]
        if acc["sumq"] is None:
            acc.update(
                sumq=torch.zeros(L, Hq, d, dtype=torch.float64, device=dev),
                sumk=torch.zeros(L, Hkv, d, dtype=torch.float64, device=dev),
                sq2=torch.zeros(L, Hq, d, d, dtype=torch.float64, device=dev),
                sk2=torch.zeros(L, Hkv, d, d, dtype=torch.float64, device=dev),
                ntok=0, shape=(L, Hq, Hkv, d))
        T = q.shape[2]
        for s in range(0, T, TCHUNK):
            qc = q[:, :, s:s+TCHUNK].to(dev).double()
            kc = k[:, :, s:s+TCHUNK].to(dev).double()
            acc["sumq"] += qc.sum(2); acc["sumk"] += kc.sum(2)
            acc["sq2"] += torch.einsum("lhtd,lhte->lhde", qc, qc)
            acc["sk2"] += torch.einsum("lhtd,lhte->lhde", kc, kc)
            del qc, kc
        acc["ntok"] += T
        kp = k[:, :, ::args.pool_stride].contiguous()
        torch.save({"k_post": kp, "prompt_length": int(kp.shape[2]),
                    "config": "gptoss_concat", "row_index": i},
                   pool / f"examples/ex_{i:04d}.pt")
        examples.append({"file": f"examples/ex_{i:04d}.pt",
                         "prompt_length": int(kp.shape[2]),
                         "config": "gptoss_concat", "row_index": i})
        print(f"  [{i+1}/{len(sequences)}] ctx={len(ids_list)} ntok={acc['ntok']}", flush=True)
        del cap, q, k
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
    torch.save({"sigma_q": sigma_q.float().cpu(), "sigma_k": Ekk.float().cpu(),
                "k_mean": mk.float().cpu(), "k_cov": k_cov.float().cpu(),
                "meta": dict(n_layers=L, n_q_heads=Hq, n_kv_heads=Hkv, d_head=d,
                             group_size=gs, local_to_global=full_ids),
                "ntok": ntok, "n_prompts": len(sequences)}, args.out_basis)
    print(f"SAVED basis {args.out_basis} | ntok={ntok} L={L} d={d}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/gpt-oss-20b-BF16")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--csv", default="artifacts/prompt_rows/gpqa_diamond.csv")
    d.add_argument("--num-prompts", type=int, default=198)
    d.add_argument("--out", required=True)
    c = sub.add_parser("concat")
    c.add_argument("--corpus", default="artifacts/oscar_llama31_8b/gpqa_only_corpus.jsonl")
    c.add_argument("--target-ctx", type=int, default=65536)
    c.add_argument("--n-sequences", type=int, default=8)
    c.add_argument("--pool-stride", type=int, default=4)
    c.add_argument("--prefill-chunk", type=int, default=512,
                   help="eager attention fp32 weights are [H, chunk, T]: 512 fits 64K captures, use 256 at 128K")
    c.add_argument("--out-basis", required=True)
    c.add_argument("--out-pool", required=True)
    args = ap.parse_args()
    {"dump": cmd_dump, "concat": cmd_concat}[args.cmd](args)


if __name__ == "__main__":
    main()
