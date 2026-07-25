"""Limitation or bug? Simulate OSCAR int2 in pure HF, bypassing the engine.

The engine's int2 collapses on gpt-oss the moment any token is quantized, but
every offline gate of the engine's kernels and pool passes, and gpt-oss keys
quantize *better* than llama's. Two explanations remain:

  BUG        the engine mis-invokes the int2 path somewhere no gate reaches
  LIMITATION OSCAR int2 genuinely breaks this architecture, and the engine is
             faithfully reproducing that

This decides it without touching the engine. The model runs under plain
HuggingFace; after prefill we overwrite the KV cache with its own
quantize->dequantize roundtrip, exactly as OSCAR defines it (rotate into the
calibrated basis, per-token affine int2, rotate back), on the same layers and
the same token band the engine quantizes:

  * only full-attention layers (gpt-oss: odd global ids; llama: all)
  * only positions [hp_prefix, T - hp_recent) -- sinks and the recent window
    stay bf16, as in the mixed HP+int2 pool

Rotations are orthogonal, so quantizing in the rotated basis and rotating back
is equivalent to what attention computes against the stored rotated keys; no
attention code needs modifying.

llama is the control. It serves int2 at 84.3, so if the simulator degrades
llama too, the simulator is wrong -- not the method.

  python dbg_int2_hf_simulation.py --model unsloth/gpt-oss-20b-BF16 \
      --rot artifacts/oscar_gptoss20b/rotations_gpqa198 --gpus 0,3 --tag gptoss
"""

import argparse
import json
import os
import re

import torch

NEEDLE_RE = re.compile(r"One of the special magic numbers for [\w-]+ is: \d+\.?", re.I)
HP_PREFIX, HP_RECENT = 64, 256


def quant_dequant_int2(x, num_groups=1):
    """Per-token affine min-max int2 (matches _groupwise_affine_quantize)."""
    T, H, D = x.shape
    gs = D // num_groups
    g = x.float().reshape(T, H, num_groups, gs)
    vmin, vmax = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    scale = (vmax - vmin).clamp_min(1e-8) / 3.0
    zero = -vmin / scale
    q = torch.clamp(g / scale + zero + 0.5, 0, 3).floor()
    return ((q - zero) * scale).reshape(T, H, D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rot", required=True)
    ap.add_argument("--gpus", default="0,3")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--gen", type=int, default=96)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--which", default="full",
                    help="full=quantize full-attn layers (engine behaviour); "
                         "swa=quantize the sliding-window layers instead; "
                         "firstN=only the first N full-attn layers")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", dtype=torch.bfloat16, trust_remote_code=True
    )
    model.eval()

    lm = os.path.join(args.rot, "layer_map.json")
    local_to_global = (
        json.load(open(lm))["local_to_global"]
        if os.path.exists(lm)
        else list(range(model.config.num_hidden_layers))
    )
    Rk = torch.load(os.path.join(args.rot, "k_rotation_qqt_r_h_pbr.pt"),
                    map_location="cpu")["layers"]
    Rv = torch.load(os.path.join(args.rot, "v_rotation_sst_r_h_pbr.pt"),
                    map_location="cpu")["layers"]

    rows = [json.loads(l) for l in open("artifacts/prompt_rows/niah_8192_gptoss_t1.jsonl")
            if l.strip()]
    rows = [r for r in rows if NEEDLE_RE.search(r["prompt"])][: args.rows]

    def build(row, target):
        p = row["prompt"]
        m = NEEDLE_RE.search(p)
        needle = m.group(0)
        number = re.search(r"(\d+)", needle).group(1)
        head, tail = p[: min(400, m.start())], p[-400:]
        budget = max(0, target * 4 - len(head) - len(needle) - len(tail))
        return head + needle + " " + p[m.end(): m.end() + budget] + tail, number

    all_layers = list(range(model.config.num_hidden_layers))
    swa_layers = [l for l in all_layers if l not in local_to_global]

    def target_layers():
        # Which layers get quantized. 'swa' isolates the claim that the
        # full-attention layers are the only long-range channel: the
        # sliding-window layers see 128 tokens, so degrading them should not
        # touch needle retrieval at all.
        if args.which == "swa":
            return [(i, l) for i, l in enumerate(swa_layers)]
        if args.which.startswith("first"):
            n = int(args.which[5:])
            return [(i, l) for i, l in enumerate(local_to_global)][:n]
        return list(enumerate(local_to_global))

    def run(prompt, quantize):
        ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
        with torch.inference_mode():
            out = model(input_ids=ids, use_cache=True)
            cache = out.past_key_values
            T = ids.shape[1]
            if quantize:
                lo, hi = HP_PREFIX, max(HP_PREFIX, T - HP_RECENT)
                for local, gid in target_layers():
                    lay = cache.layers[gid]
                    for buf, R in ((lay.keys, Rk), (lay.values, Rv)):
                        if hi <= lo:
                            continue
                        Rm = R[local]["rotation"].to(buf.device, torch.float32)
                        # [B,H,T,D] -> [T,H,D] band, rotate, quantize, rotate back
                        band = buf[0, :, lo:hi].permute(1, 0, 2).float()
                        qd = quant_dequant_int2(band @ Rm) @ Rm.T
                        buf[0, :, lo:hi] = qd.permute(1, 0, 2).to(buf.dtype)
            past = cache
            new = []
            for _ in range(args.gen):
                nxt = out.logits[:, -1:].argmax(-1)
                new.append(nxt.item())
                out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
        return tok.decode(new)

    print(f"# {args.tag}: which={args.which} -> {len(target_layers())} "
          f"quantized layers, band=[{HP_PREFIX}, T-{HP_RECENT}), "
          f"target~{args.tokens} tok")
    for mode in ("bf16", "int2"):
        hits = 0
        first = None
        for r in rows:
            prompt, number = build(r, args.tokens)
            txt = run(prompt, quantize=(mode == "int2"))
            hits += number in txt
            if first is None:
                first = txt
        print(f"[{args.tag}/{mode}] needle {hits}/{len(rows)}")
        print(f"    {first[:200]!r}")


if __name__ == "__main__":
    main()
