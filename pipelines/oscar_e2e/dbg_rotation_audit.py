"""Does the shipped OSCAR rotation actually help, measured with REAL queries?

Two earlier fidelity probes were flawed in the same direction and both made the
calibration look fine:

  * attention-output error was computed with the SAME softmax weights for the
    quantized and reference V, so the K error never reached the distribution;
  * logit fidelity used random Gaussian queries, which is structurally blind to
    a basis that is misaligned against the real query distribution -- and
    aligning error away from real queries is the entire purpose of a Q-weighted
    rotation.

This measures the thing the rotation exists to optimize, with queries captured
from the model on the same prompt family used for calibration: the relative
error of the attention logits q·K under int2, in competing bases.

  shipped   the calibrated rotation actually served
  identity  no rotation at all
  hadamard  rotation-free energy spreading (the calibration-free control)
  random    a random orthogonal basis

A correct calibration should put `shipped` clearly best. If it lands at or
below `identity`/`random`, the rotation is not helping on real queries and the
calibration is suspect -- which per-tensor cosine similarity cannot reveal,
because cosine is basis-invariant under an orthogonal map.

Run llama too: it serves int2 at 84.3, so its shipped rotation defines what
"working" looks like on this metric.
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
    """Per-token affine min-max int2, as the pool does it."""
    T, H, D = x.shape
    gs = D // num_groups
    g = x.double().reshape(T, H, num_groups, gs)
    vmin, vmax = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    scale = (vmax - vmin).clamp_min(1e-8) / 3.0
    zero = -vmin / scale
    q = torch.clamp(g / scale + zero + 0.5, 0, 3).floor()
    return ((q - zero) * scale).reshape(T, H, D)


def hadamard(n, device):
    H = torch.ones(1, 1, device=device, dtype=torch.float64)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    assert H.shape[0] == n, f"head_dim {n} must be a power of two"
    return H / (n ** 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rot", required=True)
    ap.add_argument("--gpus", default="5,6")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--csv", default="artifacts/prompt_rows/gpqa_diamond.csv")
    ap.add_argument("--noise", action="store_true",
                    help="add a matched-magnitude Gaussian arm and report ||o|| "
                         "and sink softmax share")
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
        args.model, device_map="auto", dtype=torch.bfloat16, trust_remote_code=True
    )
    model.eval()
    dev = get_model_device(model)

    lm = os.path.join(args.rot, "layer_map.json")
    local_to_global = (
        json.load(open(lm))["local_to_global"] if os.path.exists(lm)
        else list(range(model.config.num_hidden_layers))
    )
    R = torch.load(os.path.join(args.rot, "k_rotation_qqt_r_h_pbr.pt"),
                   map_location="cpu")["layers"]

    df = pd.read_csv(args.csv)
    prompts = [TMPL.format(Question=r["Question"], A=r["Correct Answer"],
                           B=r["Incorrect Answer 1"], C=r["Incorrect Answer 2"],
                           D=r["Incorrect Answer 3"])
               for _, r in df.iterrows()][: args.prompts]

    # accumulate logit error per basis, over prompts and layers
    acc = {k: [] for k in ("shipped", "identity", "hadamard", "random", "matched-noise")}
    onorm, sinkshare = [], []
    torch.manual_seed(0)
    n_layers_total = int(model.config.num_hidden_layers)

    for pi, p in enumerate(prompts):
        text = tok.apply_chat_template([{"role": "user", "content": p}],
                                       add_generation_prompt=True, tokenize=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        with capture_rope_qk(model) as (_qpre, q_post_chunks, _kpre, _kpost):
            with torch.inference_mode():
                out = model(input_ids=ids, use_cache=True)
        q_all = _assemble_per_layer(q_post_chunks, n_layers_total)   # [L, Hq, T, d]
        cache = out.past_key_values

        for local, g in enumerate(local_to_global):
            k = cache.layers[g].keys.detach()[0].transpose(0, 1).double().cpu()  # [T,Hk,d]
            q = q_all[g].transpose(0, 1).double().cpu()                          # [T,Hq,d]
            T, Hk, D = k.shape
            Hq = q.shape[1]
            grp = Hq // Hk
            Rk = R[local]["rotation"].double()
            bases = {
                "shipped": Rk,
                "identity": torch.eye(D, dtype=torch.float64),
                "hadamard": hadamard(D, k.device).to(torch.float64),
                "random": torch.linalg.qr(torch.randn(D, D, dtype=torch.float64))[0],
            }
            # a fixed slice of queries keeps cost bounded and is basis-agnostic
            qsel = q[-64:] if T > 64 else q

            if args.noise:
                # How much softmax mass does the learned sink absorb, and how
                # big is the resulting attention output? Drift > 1 was
                # ATTRIBUTED to a small ||o|| but never measured.
                attn = model.model.layers[g].self_attn
                sk_par = getattr(attn, "sinks", None)
                v = cache.layers[g].values.detach()[0].transpose(0, 1).double().cpu()
                for h in range(Hk):
                    qh = qsel[:, h * grp:(h + 1) * grp].reshape(-1, D)
                    lg = (qh @ k[:, h].T) / (D ** 0.5)
                    if sk_par is not None:
                        sv = sk_par.detach().double().cpu()
                        s_h = sv[h * grp:(h + 1) * grp].repeat_interleave(qsel.shape[0])
                        full = torch.cat([lg, s_h[:, None]], dim=1)
                        pr = torch.softmax(full, -1)
                        sinkshare.append(pr[:, -1].mean().item())
                        pv = pr[:, :-1]
                    else:
                        pv = torch.softmax(lg, -1)
                        sinkshare.append(0.0)
                    o = pv @ v[:, h]
                    # ||o|| relative to the typical value-vector norm: <<1 means
                    # the head is emitting almost nothing, so small absolute
                    # perturbations become large RELATIVE ones.
                    onorm.append((o.norm(dim=-1).mean() /
                                  v[:, h].norm(dim=-1).mean().clamp_min(1e-12)).item())

            for name, B in bases.items():
                kr = k.reshape(-1, D) @ B
                kh = quant_dequant_int2(kr.reshape(T, Hk, D)).reshape(-1, D)
                khat = (kh @ B.T).reshape(T, Hk, D)
                errs = []
                for h in range(Hk):
                    qh = qsel[:, h * grp:(h + 1) * grp].reshape(-1, D)
                    a = qh @ k[:, h].T
                    b = qh @ khat[:, h].T
                    errs.append(((a - b).norm() / a.norm().clamp_min(1e-12)).item())
                acc[name].append(sum(errs) / len(errs))
                if args.noise and name == "shipped":
                    # Matched-magnitude control: Gaussian noise with exactly the
                    # perturbation norm int2 produced in this basis. If it does
                    # the same damage, the MAGNITUDE of the error is what breaks
                    # the model and the quantizer is not doing anything special.
                    kr3 = kr.reshape(T, Hk, D)
                    kh3 = kh.reshape(T, Hk, D)
                    gnoise = torch.randn_like(kr3)
                    gnoise *= ((kh3 - kr3).norm() / gnoise.norm().clamp_min(1e-12))
                    kn = ((kr3 + gnoise).reshape(-1, D) @ B.T).reshape(T, Hk, D)
                    ne = []
                    for h in range(Hk):
                        qh = qsel[:, h * grp:(h + 1) * grp].reshape(-1, D)
                        a = qh @ k[:, h].T
                        b = qh @ kn[:, h].T
                        ne.append(((a - b).norm() / a.norm().clamp_min(1e-12)).item())
                    acc["matched-noise"].append(sum(ne) / len(ne))
        if (pi + 1) % 4 == 0:
            print(f"  {pi+1}/{len(prompts)} prompts", flush=True)

    print(f"\n# {args.tag}: relative attention-LOGIT error under int2, real queries")
    print(f"#   {len(prompts)} prompts x {len(local_to_global)} quantized layers")
    base = sum(acc["identity"]) / len(acc["identity"])
    for name in ("shipped", "identity", "hadamard", "random", "matched-noise"):
        if not acc[name]:
            continue
        v = sum(acc[name]) / len(acc[name])
        print(f"  {name:13s} logit_err {v:.4f}   ({v/base:5.2f}x identity)")
    if onorm:
        print(f"  ||o|| / ||v||  {sum(onorm)/len(onorm):.4f}   "
              f"(small => head emits little, so absolute error is relatively large)")
        print(f"  sink mass      {sum(sinkshare)/len(sinkshare):.4f}   "
              f"(fraction of softmax absorbed by the learned sink)")


if __name__ == "__main__":
    main()
