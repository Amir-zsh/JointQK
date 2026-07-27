"""Find the FIRST layer and step where a non-finite value appears, in-process.

The served failure is `probability tensor contains inf, nan or element < 0` at
torch.multinomial -- the sampler is merely the first consumer that checks.
Black-box bisection of the server kept hitting confounds (every "surviving"
config turned out to be below the load threshold), and the code audit closed
the obvious NaN routes: the per-token RMS scale is clamp_min(1e-8), the quant
arenas are zero-initialised, invalid flush rows route to a trash slot, and the
req_to_token remap early-returns on valid==0.

So instead of guessing, drive the engine in-process and hook every decoder
layer. On the first non-finite tensor we report layer index, decode step, and
which sublayer (attention vs MLP) produced it -- turning "NaN somewhere" into a
specific kernel to read.

Runs the same shape as the failing case: concurrent requests, sampled decoding,
long prompts. Uses sglang's Engine API so the hooks live in the same process as
the model.

  python dbg_nan_localize.py --gpus 3,4 --rows 24 --concurrency 4
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="3,4")
    ap.add_argument("--rows", type=int, default=24)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--model", default="unsloth/gpt-oss-20b-BF16")
    ap.add_argument("--codebook", default="artifacts/oscar_gptoss20b/"
                    "vqa_gptoss20b_G4_strat_flat_ptn_gpqacc128k_fp8.pt")
    ap.add_argument("--rot", default="artifacts/oscar_gptoss20b/rotations_gpqa198")
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["PYTHONPATH"] = str(REPO / "vendor/OSCAR-vq/sglang-research/python")
    sys.path.insert(0, str(REPO / "vendor/OSCAR-vq/sglang-research/python"))
    # serve-time config identical to the failing runs
    os.environ["SGLANG_ENABLE_MIXED_KV_WINDOWS"] = "1"
    os.environ["SGLANG_VQ_CODEBOOK_PATH"] = str(REPO / args.codebook)
    os.environ["SGLANG_OSCAR_K_ROTATION_PATH"] = str(REPO / args.rot / "k_rotation_qqt_r_h_pbr.pt")
    os.environ["SGLANG_OSCAR_V_ROTATION_PATH"] = str(REPO / args.rot / "v_rotation_sst_r_h_pbr.pt")
    os.environ["SGLANG_OSCAR_K_CLIP_RATIO"] = "0.96"
    os.environ["SGLANG_OSCAR_V_CLIP_RATIO"] = "0.92"
    os.environ["SGLANG_OSCAR_ABSORB_V_ROTATION"] = "0"
    os.environ["SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE"] = "0"

    import torch
    import sglang as sgl

    engine = sgl.Engine(
        model_path=args.model,
        kv_cache_dtype="int2",
        tp_size=2 if "," in args.gpus else 1,
        context_length=16384,
        max_total_tokens=42000,
        attention_backend="triton",
        prefill_attention_backend="triton",
        decode_attention_backend="triton",
        sampling_backend="pytorch",
        disable_radix_cache=True,
        mem_fraction_static=0.78,
        log_level="warning",
    )

    # Reach the live model through the engine's runner and hook every layer.
    runner = engine.tokenizer_manager.scheduler_info  # placeholder if unavailable
    state = {"step": 0, "hits": []}

    def hook_factory(idx, name):
        def hook(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(t) and t.dtype.is_floating_point:
                if not torch.isfinite(t).all():
                    if not state["hits"]:
                        nz = (~torch.isfinite(t)).sum().item()
                        print(f"\n*** FIRST NON-FINITE: layer {idx} {name} "
                              f"at decode step {state['step']} "
                              f"({nz}/{t.numel()} elements) ***", flush=True)
                    state["hits"].append((state["step"], idx, name))
        return hook

    try:
        model = engine.scheduler.tp_worker.model_runner.model  # in-process path
        layers = model.model.layers
        for i, lay in enumerate(layers):
            lay.register_forward_hook(hook_factory(i, "layer"))
            for nm in ("self_attn", "mlp"):
                sub = getattr(lay, nm, None)
                if sub is not None:
                    sub.register_forward_hook(hook_factory(i, nm))
        print(f"hooked {len(layers)} layers x 3 points", flush=True)
    except Exception as e:  # engine internals vary; fail loudly rather than silently
        print(f"COULD NOT HOOK MODEL IN-PROCESS: {e}", flush=True)
        print("Fall back to: run the server with --enable-nan-detection and "
              "correlate the crash step from the scheduler log.", flush=True)
        return 2

    rows = [json.loads(l) for l in
            open(REPO / "artifacts/prompt_rows/niah_8192_gptoss_t1.jsonl")][: args.rows]
    prompts = [r["prompt"] for r in rows]
    sp = {"temperature": 1.0, "top_p": 1.0, "top_k": -1,
          "max_new_tokens": args.max_new}

    print(f"driving {len(prompts)} prompts, concurrency {args.concurrency}", flush=True)
    out = engine.generate(prompts, sp)
    print(f"generated {len(out)} responses", flush=True)

    if state["hits"]:
        first = state["hits"][0]
        print(f"\nSUMMARY: first non-finite at step {first[0]}, layer {first[1]}, "
              f"{first[2]}; {len(state['hits'])} total hits")
        seen = sorted({(h[1], h[2]) for h in state["hits"]})
        print(f"  distinct sites: {seen[:10]}")
    else:
        print("\nSUMMARY: no non-finite values observed (load may be below "
              "threshold -- raise --rows/--concurrency)")
    engine.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
