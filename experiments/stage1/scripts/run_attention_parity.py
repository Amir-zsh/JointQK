"""Tier 1.2 / 1.3 — Per-layer attention-score parity for TurboQuant and JointQK.

Loads a target model, captures KV cache from a needle-in-haystack prompt at
multiple context lengths, then for each layer:
  - computes fp16 reference attention scores: q[-1] · K^T
  - compresses K,V with the chosen press, recomputes attention scores
  - reports cosine similarity + top-1 + top-5 match per (layer, kv_head) and
    aggregates per (model, press, K, V, layer0_full, ctx)

Tier 1.2 reproduces upstream's validate_v3 protocol with TurboQuantPress to
verify our wrapper matches direct upstream usage.
Tier 1.3 runs both methods on Qwen3-8B (the model we have calibration for)
to measure JointQK out-of-distribution generalization.

Note: the same FILLER text and needle as upstream's validate_v3.py.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Upstream prompt — same as validate_v3.py and generation_test_v2.py.
NEEDLE = "The secret project code name is AURORA-7749."
QUESTION = "What is the secret project code name?"
FILLER = (
    "The quarterly financial review meeting covered several topics including\n"
    "budget allocations for the upcoming fiscal year, departmental spending reports, and projected\n"
    "revenue streams from various business units. The committee discussed infrastructure upgrades\n"
    "planned for the western regional offices and noted that maintenance schedules should be\n"
    "coordinated with the facilities management team. Several action items were assigned to team\n"
    "leads for follow-up before the next meeting cycle.\n\n"
)


MODEL_REGISTRY = {
    "qwen25_3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "llama31_8b": "meta-llama/Llama-3.1-8B-Instruct",
}


def build_prompt(tokenizer, target_tokens, needle_pos=0.5, model_tag="qwen25_3b"):
    filler_len = len(tokenizer.encode(FILLER))
    n_reps = max(1, target_tokens // filler_len)
    needle_idx = int(n_reps * needle_pos)
    parts = []
    for i in range(n_reps):
        if i == needle_idx:
            parts.append(f"\n--- Memo ---\n{NEEDLE}\n--- End ---\n\n")
        parts.append(FILLER)
    haystack = "".join(parts)
    if model_tag.startswith("qwen"):
        return (f"<|im_start|>user\n{haystack}\nQuestion: {QUESTION}<|im_end|>\n"
                f"<|im_start|>assistant\n")
    return f"User: {haystack}\nQuestion: {QUESTION}\nAssistant: "


def load_model(model_tag, dtype, use_bnb_4bit, device_map):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = MODEL_REGISTRY[model_tag]
    tok = AutoTokenizer.from_pretrained(name)
    kwargs = {"device_map": device_map, "dtype": dtype}
    if use_bnb_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval()
    return model, tok


@dataclass
class FakeModule:
    layer_idx: int


def build_press(press_name, model, k_bits, v_bits, layer0_full):
    """Return (press, label_extras_dict). label_extras may include 'layer0_full'."""
    if press_name == "turboquant":
        from experiments.stage1.toolkit.turboquant_press import TurboQuantPress
        press = TurboQuantPress(
            k_bits=k_bits, v_bits=v_bits,
            protected_layers=0, residual_window=0, seed=42,
        )
        press.post_init_from_model(model)
        return press, {}
    if press_name == "jointqk":
        from experiments.stage1.toolkit.jointqk_press import JointQKPress
        # Pick calibration based on model
        cca_path = REPO / "artifacts/stage1/cca_vs_waterfill_study/cca_stats.pt"
        v_path = REPO / "artifacts/stage1/v_method_study/v_stats.pt"
        # NOTE: hardcoded to Qwen3-8B calibration. Llama would need different paths.
        press = JointQKPress(
            cca_stats_path=str(cca_path),
            v_stats_path=str(v_path),
            v_method="v_eigen_uniform",
            k_method="r_sym_waterfill",
            k_bits=k_bits, v_bits=v_bits, rank=64,
            layer0_full_precision=layer0_full,
        )
        press.post_init_from_model(model)
        return press, {"layer0_full": layer0_full}
    if press_name == "direct_turboquant":
        # Direct path used for sanity: not a press, uses upstream TurboQuantV3 directly.
        from turboquant_pytorch.compressors_v3 import TurboQuantV3
        return ("DIRECT", TurboQuantV3, k_bits, v_bits), {}
    raise ValueError(press_name)


def compress_kv_for_layer(press_obj, K, V, layer_idx, n_layers):
    """Apply compression to (B,H,S,D) K,V on the same device. Return (Kr, Vr)."""
    if isinstance(press_obj, tuple) and press_obj[0] == "DIRECT":
        _, TQ, k_bits, v_bits = press_obj
        tq = TQ(head_dim=K.shape[-1], key_bits=k_bits, value_bits=v_bits,
                residual_window=0, layer_idx=layer_idx, n_layers=n_layers,
                protected_layers=0, seed=42, device=str(K.device))
        ck, cv = tq.compress_kv(K, V)
        Kr, Vr = tq.decompress_kv(ck, cv)
        return Kr, Vr
    module = FakeModule(layer_idx=layer_idx)
    return press_obj.compress(module, hidden_states=None, keys=K, values=V,
                              attentions=None, kwargs={})


def attention_metrics(real_scores, tq_scores):
    """Return (cos, top1_pct, top5_pct) averaged across (B, H)."""
    B, H, _ = real_scores.shape
    cos = []
    top1 = 0
    top5 = 0
    n = 0
    for b in range(B):
        for h in range(H):
            rs = real_scores[b, h].float()
            ts = tq_scores[b, h].float()
            cos.append(F.cosine_similarity(rs.unsqueeze(0), ts.unsqueeze(0)).item())
            r1 = rs.argmax().item()
            t1 = ts.argmax().item()
            top1 += int(r1 == t1)
            top5 += int(r1 in ts.topk(5).indices.tolist())
            n += 1
    return sum(cos) / len(cos), 100.0 * top1 / n, 100.0 * top5 / n


def run_one_config(model, tok, press_name, k_bits, v_bits, layer0_full,
                   contexts, model_tag, out_csv, log):
    press, extras = build_press(press_name, model, k_bits, v_bits, layer0_full)

    n_layers = model.config.num_hidden_layers

    for ctx in contexts:
        prompt = build_prompt(tok, ctx, model_tag=model_tag)
        inputs = tok(prompt, return_tensors="pt", truncation=True,
                     max_length=ctx + 256).to(model.device)
        seq_len = inputs["input_ids"].shape[1]
        log(f"[{press_name} K={k_bits} V={v_bits} layer0_full={layer0_full}] ctx={seq_len}")

        with torch.no_grad():
            outputs = model(**inputs, use_cache=True, output_attentions=False)
        cache = outputs.past_key_values

        cos_sums, t1_hits, t5_hits, n_layers_done = 0.0, 0, 0, 0
        n_total = 0  # heads × layers
        for layer_idx in range(n_layers):
            keys = cache.layers[layer_idx].keys
            values = cache.layers[layer_idx].values
            query = keys[:, :, -1:, :]   # use last-token key as query proxy (matches validate_v3 pattern)

            real = torch.matmul(query.float(), keys.float().transpose(-2, -1)).squeeze(-2)
            kr, vr = compress_kv_for_layer(press, keys, values, layer_idx, n_layers)
            tqs = torch.matmul(query.float(), kr.float().transpose(-2, -1)).squeeze(-2)

            cos, t1, t5 = attention_metrics(real, tqs)
            cos_sums += cos * real.shape[1]
            t1_hits += int(t1 / 100.0 * real.shape[1])
            t5_hits += int(t5 / 100.0 * real.shape[1])
            n_total += real.shape[1]
            n_layers_done += 1

            del kr, vr, real, tqs
            torch.cuda.empty_cache()

        avg_cos = cos_sums / n_total
        avg_t1 = 100.0 * t1_hits / n_total
        avg_t5 = 100.0 * t5_hits / n_total
        log(f"  → cos={avg_cos:.6f}  top1={avg_t1:.1f}%  top5={avg_t5:.1f}%  "
            f"(layers={n_layers_done}, heads_total={n_total})")

        with out_csv.open("a") as f:
            w = csv.writer(f)
            w.writerow([model_tag, press_name, k_bits, v_bits,
                        ("True" if layer0_full else "False") if press_name == "jointqk" else "",
                        seq_len, n_layers_done, n_total,
                        f"{avg_cos:.6f}", f"{avg_t1:.2f}", f"{avg_t5:.2f}"])

        del outputs, cache
        gc.collect()
        torch.cuda.empty_cache()

    # Free press state
    del press
    gc.collect()
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_REGISTRY), required=True)
    p.add_argument("--presses", default="direct_turboquant,turboquant,jointqk",
                   help="comma list: direct_turboquant, turboquant, jointqk")
    p.add_argument("--k-bits", default="4")
    p.add_argument("--v-bits", default="2")
    p.add_argument("--contexts", default="2048,4096,8192")
    p.add_argument("--layer0-full", default="both",
                   choices=["true", "false", "both"],
                   help="for jointqk: which layer0_full settings to test")
    p.add_argument("--bnb-4bit", action="store_true",
                   help="load model with bnb-4bit (matches upstream validate_v3)")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--output-csv", required=True)
    p.add_argument("--output-log", required=True)
    args = p.parse_args()

    out_csv = Path(args.output_csv)
    out_log = Path(args.output_log)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_log.parent.mkdir(parents=True, exist_ok=True)

    if not out_csv.exists():
        with out_csv.open("w") as f:
            csv.writer(f).writerow(
                ["model", "press", "k_bits", "v_bits", "layer0_full",
                 "ctx", "n_layers", "heads_total", "cos", "top1_pct", "top5_pct"])

    log_handle = out_log.open("a")
    def log(msg):
        print(msg, flush=True)
        log_handle.write(msg + "\n")
        log_handle.flush()

    presses = [p.strip() for p in args.presses.split(",") if p.strip()]
    k_bits_list = [int(x) for x in args.k_bits.split(",")]
    v_bits_list = [int(x) for x in args.v_bits.split(",")]
    contexts = [int(x) for x in args.contexts.split(",")]

    log(f"Loading {args.model} (bnb_4bit={args.bnb_4bit}, dtype={args.dtype})...")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]
    model, tok = load_model(args.model, dtype, args.bnb_4bit, args.device_map)
    log(f"  loaded. n_layers={model.config.num_hidden_layers}")

    for press_name in presses:
        for k_bits in k_bits_list:
            for v_bits in v_bits_list:
                if press_name == "jointqk":
                    layer0_options = (
                        [True, False] if args.layer0_full == "both"
                        else [args.layer0_full == "true"])
                else:
                    layer0_options = [True]   # ignored for non-jointqk
                for l0 in layer0_options:
                    run_one_config(model, tok, press_name, k_bits, v_bits, l0,
                                   contexts, args.model, out_csv, log)

    log_handle.close()
    log(f"Done. CSV: {out_csv}")


if __name__ == "__main__":
    main()
