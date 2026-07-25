"""Does MoE routing amplify KV quantization noise? gpt-oss vs dense llama.

int2 reconstructs gpt-oss keys *better* than llama's (cos 0.923 vs 0.898) yet
destroys gpt-oss and leaves llama intact, so the damage is not in the KV error
itself but in what the model does with it downstream. gpt-oss is a Mixture of
Experts; llama is dense. Router top-k selection is a discrete function of the
hidden state, so an arbitrarily small perturbation can flip an expert and swap
an entire FFN -- an amplifier a dense model does not have.

Two measurements, teacher-forced so both runs consume identical tokens and
stay comparable (free-running generations diverge and confound everything):

  hidden drift  per-layer relative error between the bf16 and int2 runs.
                Dense models should show this growing smoothly with depth;
                a discrete amplifier shows jumps.
  expert flips  fraction of (token, layer) where the router's top-k set
                changes. Dense llama has no routers, so this is gpt-oss only
                and is interpreted against its own hidden drift.

  python dbg_moe_amplification.py --model unsloth/gpt-oss-20b-BF16 \
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
    ap.add_argument("--steps", type=int, default=24)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", dtype=torch.bfloat16, trust_remote_code=True
    )
    model.eval()
    layers = model.model.layers
    nL = len(layers)

    lm = os.path.join(args.rot, "layer_map.json")
    local_to_global = (
        json.load(open(lm))["local_to_global"]
        if os.path.exists(lm) else list(range(nL))
    )
    Rk = torch.load(os.path.join(args.rot, "k_rotation_qqt_r_h_pbr.pt"),
                    map_location="cpu")["layers"]
    Rv = torch.load(os.path.join(args.rot, "v_rotation_sst_r_h_pbr.pt"),
                    map_location="cpu")["layers"]

    row = next(json.loads(l) for l in
               open("artifacts/prompt_rows/niah_8192_gptoss_t1.jsonl")
               if l.strip() and NEEDLE_RE.search(json.loads(l)["prompt"]))
    p = row["prompt"]
    m = NEEDLE_RE.search(p)
    head, tail = p[: min(400, m.start())], p[-400:]
    budget = max(0, args.tokens * 4 - len(head) - len(m.group(0)) - len(tail))
    prompt = head + m.group(0) + " " + p[m.end(): m.end() + budget] + tail
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)

    # --- capture hooks: per-layer hidden output, and router top-k if present --
    cap = {"hidden": {}, "topk": {}, "attn": {}, "mlp": {}}

    def mk_hidden_hook(i):
        def h(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            cap["hidden"][i] = t.detach()[:, -1].float()
        return h

    def mk_router_hook(i):
        # GptOssTopKRouter.forward returns (router_logits, router_scores,
        # router_indices). router_indices IS the expert selection; taking topk
        # of router_scores instead yields 0..top_k-1 and looks like "no flips
        # ever", which is what the first version of this probe reported.
        def h(_m, _inp, out):
            if isinstance(out, tuple) and len(out) >= 3:
                cap["topk"][i] = out[2].detach()[-1].flatten().tolist()
        return h

    def mk_sub_hook(i, name):
        def h(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(t):
                cap[name][i] = t.detach().reshape(-1, t.shape[-1])[-1].float()
        return h

    handles = []
    n_routers = 0
    for i, lay in enumerate(layers):
        handles.append(lay.register_forward_hook(mk_hidden_hook(i)))
        router = getattr(getattr(lay, "mlp", None), "router", None)
        if router is not None:
            handles.append(router.register_forward_hook(mk_router_hook(i)))
            n_routers += 1
        # Within-layer breakdown: if drift grows across the MLP rather than
        # across attention, the FFN (MoE for gpt-oss, dense for llama) is the
        # amplifier rather than the attention read itself.
        for nm in ("self_attn", "mlp"):
            sub = getattr(lay, nm, None)
            if sub is not None:
                key = "attn" if nm == "self_attn" else "mlp"
                handles.append(sub.register_forward_hook(mk_sub_hook(i, key)))

    def prefill(quantize):
        # The cache tensors are inference tensors; they can only be mutated
        # inside the same inference_mode context that produced them.
        with torch.inference_mode():
            out = model(input_ids=ids, use_cache=True)
            cache = out.past_key_values
            T = ids.shape[1]
            if quantize:
                lo, hi = HP_PREFIX, max(HP_PREFIX, T - HP_RECENT)
                for local, gid in enumerate(local_to_global):
                    lay = cache.layers[gid]
                    for buf, R in ((lay.keys, Rk), (lay.values, Rv)):
                        Rm = R[local]["rotation"].to(buf.device, torch.float32)
                        band = buf[0, :, lo:hi].permute(1, 0, 2).float()
                        buf[0, :, lo:hi] = (
                            (quant_dequant_int2(band @ Rm) @ Rm.T)
                            .permute(1, 0, 2).to(buf.dtype)
                        )
        return out, cache

    out_ref, cache_ref = prefill(False)
    _, cache_q = prefill(True)

    # Reference continuation drives BOTH runs (teacher forcing).
    toks, past, out = [], cache_ref, out_ref
    with torch.inference_mode():
        for _ in range(args.steps):
            nxt = out.logits[:, -1:].argmax(-1)
            toks.append(nxt)
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values

    drift = [[] for _ in range(nL)]
    drift_attn = [[] for _ in range(nL)]
    drift_mlp = [[] for _ in range(nL)]
    flips = [0] * nL
    counts = [0] * nL
    _, cache_ref2 = prefill(False)
    pa, pb = cache_ref2, cache_q
    with torch.inference_mode():
        for t in toks:
            model(input_ids=t, past_key_values=pa, use_cache=True)
            ha = {i: v.clone() for i, v in cap["hidden"].items()}
            aa = {i: v.clone() for i, v in cap["attn"].items()}
            ma = {i: v.clone() for i, v in cap["mlp"].items()}
            ka = {i: list(v) for i, v in cap["topk"].items()}
            model(input_ids=t, past_key_values=pb, use_cache=True)
            for i in range(nL):
                if i in ha and i in cap["hidden"]:
                    a, b = ha[i], cap["hidden"][i]
                    drift[i].append(((a - b).norm() / a.norm().clamp_min(1e-9)).item())
                for src, dst in ((aa, drift_attn), (ma, drift_mlp)):
                    key = "attn" if src is aa else "mlp"
                    if i in src and i in cap[key]:
                        a, b = src[i], cap[key][i]
                        dst[i].append(((a - b).norm() / a.norm().clamp_min(1e-9)).item())
                if i in ka and i in cap["topk"]:
                    counts[i] += 1
                    flips[i] += int(set(ka[i]) != set(cap["topk"][i]))

    # Hook sanity: a top-k capture that never varies across layers/steps means
    # the hook is reading a constant, not routing. Without this check a
    # flip-rate of 0.000 is indistinguishable from a broken probe.
    if n_routers:
        seen = {i: tuple(v) for i, v in cap["topk"].items()}
        distinct = len(set(seen.values()))
        print(f"# router sanity: {n_routers} routers, {distinct} distinct top-k "
              f"patterns across layers; sample L1={seen.get(1)} L11={seen.get(11)}")

    for h in handles:
        h.remove()

    print(f"# {args.tag}: {nL} layers, {n_routers} routers, "
          f"{len(toks)} teacher-forced steps, prompt={ids.shape[1]} tok")
    print(f"{'layer':>5} {'hidden':>9} {'attn_out':>9} {'mlp_out':>9} {'flip':>7}")
    for i in range(nL):
        def av(x):
            return sum(x[i]) / len(x[i]) if x[i] else float("nan")
        f = (flips[i] / counts[i]) if counts[i] else float("nan")
        print(f"{i:>5} {av(drift):>9.4f} {av(drift_attn):>9.4f} "
              f"{av(drift_mlp):>9.4f} {f:>7.3f}")
    dd = [sum(x) / len(x) for x in drift if x]
    nz = [d for d in dd if d > 1e-6]
    # Layer 0 is a sliding-window layer with an unquantized cache, so its drift
    # is exactly 0 -- ratios against it are meaningless. Report the profile.
    print(f"MEAN {args.tag}: drift_firstnonzero={nz[0]:.4f} drift_last={dd[-1]:.4f} "
          f"drift_mean={sum(dd)/len(dd):.4f} "
          f"expert_flip={sum(flips) / max(sum(counts), 1):.3f}")


if __name__ == "__main__":
    main()
