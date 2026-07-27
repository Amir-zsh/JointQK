"""What happens AT the first quantized layer — the logit->distribution->output chain.

At each model's first quantized layer both runs receive provably identical
inputs (for gpt-oss, layer 0 is sliding-window with an unquantized cache and
its measured drift is exactly 0.000). Same queries, same hidden state, only the
KV quantized. Yet:

    logit error   gpt-oss 0.71  vs  llama 0.21   ->  3.4x
    attn output   gpt-oss 1.207 vs  llama 0.075  ->   16x

The output gap is ~5x larger than the logit gap that supposedly causes it, so
most of the amplification happens BETWEEN them — in the softmax and the
value-weighting — and nothing so far has measured that step.

This decomposes the chain at that single layer:

  entropy / max_p  how peaked the true attention is. A sharply peaked head
                   reorders its top keys under a small logit perturbation; a
                   flat one averages the error away.
  TV(p, p_hat)     how far the attention distribution actually moves.
  rank churn       fraction of the true top-8 keys that leave the top-8.
  ||o-o_hat||/||o|| the endpoint, for reference.

Interpretation: if gpt-oss's TV shift is ~3.4x llama's (tracking the logits)
but its output damage is 16x, the amplification is in the value-weighting. If
TV is itself ~16x, the softmax is the amplifier and peakedness is the cause.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def quant_dequant_int2(x, num_groups=1):
    T, H, D = x.shape
    gs = D // num_groups
    g = x.double().reshape(T, H, num_groups, gs)
    vmin, vmax = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    scale = (vmax - vmin).clamp_min(1e-8) / 3.0
    zero = -vmin / scale
    q = torch.clamp(g / scale + zero + 0.5, 0, 3).floor()
    return ((q - zero) * scale).reshape(T, H, D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rot", required=True)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--groups", type=int, default=1)
    ap.add_argument("--csv", default="artifacts/prompt_rows/gpqa_diamond.csv")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    import pandas as pd
    import _bootstrap  # noqa: F401
    from pipelines.calibration.capture_raw import _assemble_per_layer, capture_rope_qk
    from kvq.capture.model import get_model_device
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TMPL = ("Answer the following multiple choice question. The last line of your response "
            "should be of the following format: 'Answer: $LETTER' (without quotes) where "
            "LETTER is one of ABCD. Think step by step before answering.\n\n{Question}\n\n"
            "A) {A}\nB) {B}\nC) {C}\nD) {D}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", dtype=torch.bfloat16, trust_remote_code=True)
    model.eval()
    dev = get_model_device(model)

    lm = os.path.join(args.rot, "layer_map.json")
    l2g = (json.load(open(lm))["local_to_global"] if os.path.exists(lm)
           else list(range(model.config.num_hidden_layers)))
    Rk = torch.load(os.path.join(args.rot, "k_rotation_qqt_r_h_pbr.pt"),
                    map_location="cpu")["layers"]
    Rv = torch.load(os.path.join(args.rot, "v_rotation_sst_r_h_pbr.pt"),
                    map_location="cpu")["layers"]

    local, g = 0, l2g[0]          # THE first quantized layer
    df = pd.read_csv(args.csv)
    prompts = [TMPL.format(Question=r["Question"], A=r["Correct Answer"],
                           B=r["Incorrect Answer 1"], C=r["Incorrect Answer 2"],
                           D=r["Incorrect Answer 3"])
               for _, r in df.iterrows()][: args.prompts]

    ent, maxp, tv, churn, oerr, lerr = [], [], [], [], [], []
    nL = int(model.config.num_hidden_layers)
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p}],
                                       add_generation_prompt=True, tokenize=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        with capture_rope_qk(model) as (_a, qch, _b, _c):
            with torch.inference_mode():
                out = model(input_ids=ids, use_cache=True)
        q_all = _assemble_per_layer(qch, nL)
        cache = out.past_key_values
        k = cache.layers[g].keys.detach()[0].transpose(0, 1).double().cpu()
        v = cache.layers[g].values.detach()[0].transpose(0, 1).double().cpu()
        q = q_all[g].transpose(0, 1).double().cpu()
        T, Hk, D = k.shape
        grp = q.shape[1] // Hk
        Bk = Rk[local]["rotation"].double()
        Bv = Rv[local]["rotation"].double()
        khat = (quant_dequant_int2((k.reshape(-1, D) @ Bk).reshape(T, Hk, D),
                                   args.groups).reshape(-1, D) @ Bk.T).reshape(T, Hk, D)
        vhat = (quant_dequant_int2((v.reshape(-1, D) @ Bv).reshape(T, Hk, D),
                                   args.groups).reshape(-1, D) @ Bv.T).reshape(T, Hk, D)
        attn = model.model.layers[g].self_attn
        sk = getattr(attn, "sinks", None)
        qsel = q[-32:] if T > 32 else q
        for h in range(Hk):
            qh = qsel[:, h * grp:(h + 1) * grp].reshape(-1, D)
            a = (qh @ k[:, h].T) / (D ** 0.5)
            b = (qh @ khat[:, h].T) / (D ** 0.5)
            lerr.append(((a - b).norm() / a.norm().clamp_min(1e-12)).item())
            if sk is not None:
                sv = sk.detach().double().cpu()[h * grp:(h + 1) * grp]
                sv = sv.repeat_interleave(qsel.shape[0])[:, None]
                pa = torch.softmax(torch.cat([a, sv], 1), -1)[:, :-1]
                pb = torch.softmax(torch.cat([b, sv], 1), -1)[:, :-1]
            else:
                pa, pb = torch.softmax(a, -1), torch.softmax(b, -1)
            pn = pa / pa.sum(-1, keepdim=True).clamp_min(1e-30)
            ent.append((-(pn * pn.clamp_min(1e-30).log()).sum(-1)).mean().item())
            maxp.append(pa.max(-1).values.mean().item())
            tv.append((0.5 * (pa - pb).abs().sum(-1)).mean().item())
            kk = min(8, T)
            ta = a.topk(kk, -1).indices
            tb = b.topk(kk, -1).indices
            same = torch.stack([torch.isin(ta[i], tb[i]).sum() for i in range(ta.shape[0])])
            churn.append(1.0 - (same.double().mean() / kk).item())
            o, oh = pa @ v[:, h], pb @ vhat[:, h]
            oerr.append(((o - oh).norm() / o.norm().clamp_min(1e-12)).item())

    m = lambda x: sum(x) / len(x)
    print(f"\n# {args.tag}: FIRST quantized layer (global {g}), groups={args.groups}, "
          f"{len(prompts)} prompts")
    print(f"  entropy(p)        {m(ent):8.4f}   nats  (ln T ~ flat)")
    print(f"  max p             {m(maxp):8.4f}   (peakedness)")
    print(f"  logit err         {m(lerr):8.4f}")
    print(f"  TV(p, p_hat)      {m(tv):8.4f}   <- distribution shift")
    print(f"  top-8 churn       {m(churn):8.4f}   (fraction of top keys displaced)")
    print(f"  attn output err   {m(oerr):8.4f}")
    print(f"  amplification     logit->TV {m(tv)/max(m(lerr),1e-9):6.2f}x   "
          f"TV->output {m(oerr)/max(m(tv),1e-9):6.2f}x")


if __name__ == "__main__":
    main()
