"""Measure int2 reconstruction fidelity on REAL keys and values, gpt-oss vs llama.

The served ablation showed int2 tracks bf16 while the context fits the bf16 HP
band and collapses the moment the quant tier engages — but the offline kernel
and pool gates are numerically exact on synthetic Gaussian keys, and llama
serves int2 at 84.3 through the same code. So the question is whether real
gpt-oss keys survive 2-bit quantization in the OSCAR rotated basis at all.

This captures post-RoPE K from a live forward pass (the same tensor the engine
caches), applies the model's own OSCAR rotation, quantizes int2 exactly as the
pool does, and reports:

  cos        per-token cosine between K_hat and K in the rotated basis
  logit_r    correlation of attention logits q.K_hat vs q.K  -- what attention
             actually consumes; a basis can look lossy per-coordinate and
             still preserve ranking, or vice versa
  top1       fraction of queries whose argmax key is unchanged (needle test)
  v_cos      per-token cosine on the value side
  o_err      relative error of the attention output softmax(qK) @ V -- how the
             value error actually reaches the residual stream

Run per model; compare. llama is the working reference, so its numbers define
"good enough to serve".

  python dbg_capture_k_fidelity.py --model unsloth/gpt-oss-20b-BF16 \
      --rot artifacts/oscar_gptoss20b/rotations_gpqa198 --gpus 3,2 --tag gptoss
"""

import argparse
import json
import os

import torch


def quant_dequant_int2(x, num_groups):
    """Per-token affine min-max int2, exactly as _groupwise_affine_quantize."""
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
    ap.add_argument("--rot", required=True, help="rotation dir")
    ap.add_argument("--gpus", default="3,2")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tokens", type=int, default=2048)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", dtype=torch.bfloat16, trust_remote_code=True
    )
    model.eval()

    row = json.loads(open("artifacts/prompt_rows/niah_8192_gptoss_t1.jsonl").readline())
    ids = tok(row["prompt"], return_tensors="pt").input_ids[:, : args.tokens]
    with torch.inference_mode():
        out = model(input_ids=ids.to(model.device), use_cache=True)
    cache = out.past_key_values

    lm = os.path.join(args.rot, "layer_map.json")
    local_to_global = (
        json.load(open(lm))["local_to_global"]
        if os.path.exists(lm)
        else list(range(model.config.num_hidden_layers))
    )
    R = torch.load(
        os.path.join(args.rot, "k_rotation_qqt_r_h_pbr.pt"), map_location="cpu"
    )["layers"]
    Rv = torch.load(
        os.path.join(args.rot, "v_rotation_sst_r_h_pbr.pt"), map_location="cpu"
    )["layers"]

    print(f"# {args.tag}: {len(local_to_global)} quantized layers, "
          f"T={ids.shape[1]}")
    print(f"{'layer':>5} {'groups':>7} {'cos':>7} {'logit_r':>8} {'top1':>6} "
          f"{'v_cos':>7} {'o_err':>8}")
    rows = []
    for local, g in enumerate(local_to_global):
        k = cache.layers[g].keys.detach()[0].transpose(0, 1).float().cpu()  # [T,H,D]
        Rk = R[local]["rotation"].float()
        kr = k @ Rk
        T, H, D = kr.shape
        q = torch.randn(64, H, D)  # proxy queries in the same rotated basis
        for ng in (1, 4):
            kh = quant_dequant_int2(kr, ng)
            cos = torch.nn.functional.cosine_similarity(kh, kr, dim=-1).mean().item()
            rs, t1 = [], []
            for h in range(H):
                a = q[:, h] @ kr[:, h].T
                b = q[:, h] @ kh[:, h].T
                rs.append(torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item())
                t1.append((a.argmax(-1) == b.argmax(-1)).float().mean().item())
            r = sum(rs) / len(rs)
            top1 = sum(t1) / len(t1)
            # V side: error lands directly in the attention output, so score it
            # as the model consumes it -- o = softmax(qK) @ V vs the same
            # weights against the dequantized V.
            v = cache.layers[g].values.detach()[0].transpose(0, 1).float().cpu()
            vr = v @ Rv[local]["rotation"].float()
            vh = quant_dequant_int2(vr, ng)
            vcos = torch.nn.functional.cosine_similarity(vh, vr, dim=-1).mean().item()
            oerrs = []
            for h in range(H):
                p = torch.softmax((q[:, h] @ kr[:, h].T) / (D ** 0.5), dim=-1)
                o, oh = p @ vr[:, h], p @ vh[:, h]
                oerrs.append(((o - oh).norm() / o.norm().clamp_min(1e-9)).item())
            oerr = sum(oerrs) / len(oerrs)
            rows.append((local, ng, cos, r, top1, vcos, oerr))
            print(f"{local:>5} {ng:>7} {cos:>7.3f} {r:>8.3f} {top1:>6.2f} "
                  f"{vcos:>7.3f} {oerr:>8.3f}")
    for ng in (1, 4):
        sel = [x for x in rows if x[1] == ng]
        print(f"MEAN {args.tag} groups={ng}: cos={sum(x[2] for x in sel)/len(sel):.3f} "
              f"logit_r={sum(x[3] for x in sel)/len(sel):.3f} "
              f"top1={sum(x[4] for x in sel)/len(sel):.2f} "
              f"v_cos={sum(x[5] for x in sel)/len(sel):.3f} "
              f"o_err={sum(x[6] for x in sel)/len(sel):.3f}")


if __name__ == "__main__":
    main()
