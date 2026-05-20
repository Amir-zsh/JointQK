#!/usr/bin/env python3
"""Capture decode-time q's, then measure top-1 retention against compressed k.

Tests hypothesis A2: calibration measures top-1 of `q_prefill ⊤ k_prefill`,
but real attention at decode time uses `q_decode[t] ⊤ k_prefill`. If decode-q
distribution differs from prefill-q (different RoPE phase, hidden-state shift),
JointQK's basis (fitted to prefill-q distribution) is suboptimal for decode-q.

Method:
1. Load Qwen3-8B and a small subset of test prompts (one per task).
2. For each prompt: prefill → capture k_prefill (post-RoPE) and q_decode (post-RoPE)
   for the first N decode tokens.
3. For each (layer, head): compute top-1 of:
     true:   argmax_j q_decode[t] @ k_prefill[j]
     compressed: argmax_j q_decode[t] @ recon(k_prefill[j])
   for each compression method. Compare top-1 retention vs prefill-q top-1.

Scope:
- 4 prompts (1 per task: hotpotqa, musique, qasper, qmsum from 80-prompt test set)
- 8 decode tokens per prompt
- b=4 only (this is where the disconnect is sharpest)
- Methods: v3, jointqk (NEW basis)

Output:
  artifacts/stage1/calibration/longbench_compact8_qkv/05_reports/decode_q_top1.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from experiments.stage1.calibration.analyze_bases import jointqk_basis
from experiments.stage1.calibration.common import RunPaths
from experiments.stage1.toolkit.capture import capture_rope_qk
from experiments.stage1.toolkit.model import get_model_device, load_model_and_tokenizer
from experiments.stage1.toolkit.per_coord_quantization import PerCoordCompressor, round_bits_to_integer
from experiments.stage1.toolkit.metric_transform import water_fill
from experiments.stage1.toolkit.quantization import Stage1MSECompressor


def ts() -> str:
    return time.strftime("%H:%M:%S")


def allocate_bits_waterfill(scores: torch.Tensor, b_avg: int, max_coord_bits: int = 8) -> torch.Tensor:
    d = int(scores.shape[-1])
    total_bits = b_avg * d
    flat = scores.detach().float().cpu().reshape(-1, d).clamp_min(1e-30)
    cont = water_fill(flat, total_bits=float(total_bits))
    bits_int = round_bits_to_integer(cont, total_bits=total_bits)
    out = bits_int.clone()
    for i in range(out.shape[0]):
        excess = int((out[i] - max_coord_bits).clamp_min(0).sum().item())
        out[i].clamp_(max=max_coord_bits)
        while excess > 0:
            below = (out[i] < max_coord_bits).nonzero(as_tuple=True)[0]
            if below.numel() == 0:
                break
            min_val = out[i, below].min()
            cands = below[out[i, below] == min_val]
            take = min(int(cands.numel()), excess)
            out[i, cands[:take]] += 1
            excess -= take
    return out.reshape(scores.shape)


@torch.inference_mode()
def capture_prefill_decode_qk(
    model, tokenizer, prompt_text: str, num_decode_tokens: int = 8
) -> dict[str, Any]:
    """Run prefill on `prompt_text`, then generate `num_decode_tokens` tokens
    while hooking q_post / k_post chunks. Returns per-layer:
      - k_prefill (n_layers, n_kv_heads, prompt_len, d)  fp16
      - q_decode  (n_layers, n_q_heads,  num_decode,  d)  fp16
      - prompt_length (int)
    """
    device = get_model_device(model)
    n_layers = int(model.config.num_hidden_layers)

    enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=8000)
    input_ids = enc["input_ids"].to(device)
    prompt_len = int(input_ids.shape[-1])

    with capture_rope_qk(model) as (_q_pre, q_post_chunks, _k_pre, _k_post):
        outputs = model(input_ids=input_ids, use_cache=True)
        # q_post_chunks now has prefill q's. K_post is in outputs.past_key_values.
        # Snapshot prefill q-chunks so we can later split off decode chunks.
        prefill_chunks_per_layer = len(q_post_chunks) // n_layers
        prefill_q_count = len(q_post_chunks)

        past_kv = outputs.past_key_values

        # Decode loop with hooks active.
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        decoded_ids = [int(next_token.item())]
        for _ in range(num_decode_tokens - 1):
            out = model(input_ids=next_token, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            decoded_ids.append(int(next_token.item()))

        # Split q_post_chunks: prefill chunks first (one per layer), then decode chunks
        # (one chunk per layer per decode step).
        all_q_chunks = list(q_post_chunks)

    # Prefill q: cat the first n_layers chunks per layer.
    # In practice capture_rope_qk emits one chunk per forward call, so the first
    # `n_layers` chunks correspond to layer 0..n_layers-1 of the prefill forward.
    n_total_calls = 1 + (num_decode_tokens - 1) + 1  # prefill + (n-1) decode + 1 first-decode-from-logits
    # Actually we did: 1 prefill forward + (num_decode_tokens-1) decode forwards.
    # Each forward pass emits n_layers chunks (one per layer).
    expected = (1 + (num_decode_tokens - 1)) * n_layers
    if len(all_q_chunks) != expected:
        raise RuntimeError(
            f"q_post_chunks count {len(all_q_chunks)} != expected {expected} "
            f"(prefill + {num_decode_tokens - 1} decode forwards × {n_layers} layers)"
        )

    # First num_decode_tokens-1 chunks per layer are decode (the first decode
    # token came from prefill's last logit; subsequent tokens come from decode).
    # Wait: the FIRST forward pass IS the prefill, so its q_post chunks have the
    # prefill q's (shape (1, n_q_heads, prompt_len, d)). The next N-1 forwards
    # are decode steps with shape (1, n_q_heads, 1, d) each.
    # Hooks were registered before the prefill forward, so chunks[0..n_layers-1]
    # are prefill (each (1, n_q_heads, prompt_len, d)), and the rest are decode
    # (each (1, n_q_heads, 1, d)).
    prefill_q_per_layer = []
    for L in range(n_layers):
        # Prefill emits one chunk per layer; in interleaved order across one forward.
        # Order is layer 0 → layer 1 → ... → layer n_layers-1 (since fwd is sequential).
        # So all_q_chunks[L] is layer L's prefill q.
        prefill_q_per_layer.append(all_q_chunks[L])

    # Decode chunks: indices n_layers .. expected-1, in order:
    #   [layer 0 decode-step 1, layer 1 decode-step 1, ..., layer 0 decode-step 2, ...]
    decode_q_per_layer: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    idx = n_layers
    for step in range(num_decode_tokens - 1):
        for L in range(n_layers):
            decode_q_per_layer[L].append(all_q_chunks[idx])
            idx += 1

    # Stack decode q's: (n_layers, n_q_heads, num_decode-1, d)
    q_decode = []
    for L in range(n_layers):
        if not decode_q_per_layer[L]:
            continue
        # Each decode chunk is (1, n_q_heads, 1, d). Cat along seq dim (dim=2).
        cat = torch.cat(decode_q_per_layer[L], dim=2).squeeze(0).cpu().to(torch.float16)
        q_decode.append(cat)
    q_decode = torch.stack(q_decode, dim=0) if q_decode else torch.empty(0)

    # k_prefill from outputs (after prefill, before any decode).
    # past_kv already includes decode k's; we want only the first prompt_len.
    k_prefill = []
    # outputs.past_key_values is from the FIRST prefill forward.
    for L in range(n_layers):
        kl = outputs.past_key_values.layers[L].keys.detach().to("cpu", dtype=torch.float16).squeeze(0)
        # Take only prefill positions (first prompt_len).
        k_prefill.append(kl[:, :prompt_len, :])
    k_prefill = torch.stack(k_prefill, dim=0)  # (n_layers, n_kv_heads, prompt_len, d)

    # prefill q
    prefill_q = []
    for L in range(n_layers):
        chunk = prefill_q_per_layer[L].detach().to("cpu", dtype=torch.float16).squeeze(0)
        prefill_q.append(chunk)
    prefill_q = torch.stack(prefill_q, dim=0)  # (n_layers, n_q_heads, prompt_len, d)

    return {
        "k_prefill": k_prefill,
        "q_decode": q_decode,  # (n_layers, n_q_heads, num_decode-1, d)
        "q_prefill": prefill_q,  # (n_layers, n_q_heads, prompt_len, d)
        "prompt_length": prompt_len,
        "decoded_token_ids": decoded_ids,
    }


def measure_top1(
    q: torch.Tensor,           # (n_q_heads, T_q, d)
    k_full: torch.Tensor,      # (n_kv_heads, T_k, d)
    k_compressed: torch.Tensor,# (n_kv_heads, T_k, d)
    n_kv_heads: int,
    device: torch.device,
) -> dict[str, float]:
    """Per-(query, head_group) top-1 retention. Returns dict {top1, top5}."""
    q = q.to(device).float()
    k_full = k_full.to(device).float()
    k_compressed = k_compressed.to(device).float()
    group = q.shape[0] // n_kv_heads
    top1_num, top1_den = 0, 0
    top5_num, top5_den = 0, 0
    for h in range(n_kv_heads):
        q_h = q[h * group : (h + 1) * group]     # (group, T_q, d)
        k_t_full = k_full[h]                     # (T_k, d)
        k_t_comp = k_compressed[h]
        # (group * T_q, d) @ (d, T_k) → (group * T_q, T_k)
        gT, d = q_h.shape[0] * q_h.shape[1], q_h.shape[-1]
        scores_full = q_h.reshape(gT, d) @ k_t_full.T
        scores_comp = q_h.reshape(gT, d) @ k_t_comp.T
        true_argmax = scores_full.argmax(dim=-1)
        comp_argmax = scores_comp.argmax(dim=-1)
        top1_num += int((true_argmax == comp_argmax).sum().item())
        top1_den += int(true_argmax.numel())
        k_top = min(5, scores_full.shape[-1])
        comp_top5 = scores_comp.topk(k_top, dim=-1).indices
        top5_num += int((comp_top5 == true_argmax.unsqueeze(-1)).any(dim=-1).sum().item())
        top5_den += int(true_argmax.numel())
    return {
        "top1": top1_num / max(1, top1_den),
        "top5": top5_num / max(1, top5_den),
    }


@torch.inference_mode()
def reconstruct_jointqk(
    k_full: torch.Tensor,    # (n_kv_heads, T, d)
    sigma_q: torch.Tensor,   # (n_kv_heads, d, d) for this layer
    sigma_k: torch.Tensor,   # (n_kv_heads, d, d) for this layer
    bits: int,
    device: torch.device,
    max_coord_bits: int = 8,
) -> torch.Tensor:
    n_kv_heads, T, d = k_full.shape
    R_sym = jointqk_basis(sigma_q.unsqueeze(0), sigma_k.unsqueeze(0), eps=1e-4).squeeze(0)  # (n_kv_heads, d, d)
    train_q_diag = (R_sym.transpose(-1, -2) @ sigma_q @ R_sym).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    train_k_diag = (R_sym.transpose(-1, -2) @ sigma_k @ R_sym).diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    alloc = allocate_bits_waterfill(train_q_diag * train_k_diag, bits, max_coord_bits=max_coord_bits)
    train_k_std = torch.sqrt(train_k_diag)

    recon = torch.empty_like(k_full, dtype=torch.float32)
    for h in range(n_kv_heads):
        comp = PerCoordCompressor(
            bits_per_coord=alloc[h].cpu(),
            std_per_coord=train_k_std[h].cpu(),
            forward_map=R_sym[h].cpu(),
            inverse_map=R_sym[h].transpose(-1, -2).cpu(),
        ).to(device)
        recon[h] = comp.roundtrip(k_full[h].to(device).float()).cpu().to(recon.dtype)
        del comp
    return recon


@torch.inference_mode()
def reconstruct_v3(k_full: torch.Tensor, bits: int, seed: int, device: torch.device) -> torch.Tensor:
    n_kv_heads, T, d = k_full.shape
    comp = Stage1MSECompressor(head_dim=int(d), bits=int(bits), seed=int(seed), device=device)
    recon = torch.empty_like(k_full, dtype=torch.float32)
    for h in range(n_kv_heads):
        kt = k_full[h].to(device).float()
        recon[h] = comp.roundtrip(kt.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).cpu().to(recon.dtype)
    return recon


def main() -> None:
    artifact_root = REPO / "artifacts/stage1/calibration"
    paths = RunPaths.from_args(artifact_root, "longbench_compact8_qkv")
    agg = torch.load(paths.stats_dir / "aggregate.pt", map_location="cpu", weights_only=False)
    per_example = agg["per_example"]
    test_examples = [p for p in per_example if p["split"] == "test"]
    # Pick one test prompt per task (first one).
    by_task = {}
    for p in test_examples:
        by_task.setdefault(p["config"], p)
    target_tasks = ["hotpotqa", "musique", "qasper", "qmsum"]
    chosen = [by_task[t] for t in target_tasks if t in by_task]
    print(f"[{ts()}] picked {len(chosen)} test prompts: {[p['config'] for p in chosen]}", flush=True)

    # Load pooled train_stats (N=400) for computing the calibration moments.
    from experiments.stage1.calibration.analyze_bases import combine_stats
    train_indices = [int(p["index"]) for p in per_example if p["split"] == "train"]
    print(f"[{ts()}] pooling train stats over {len(train_indices)} examples", flush=True)
    train_stats = combine_stats(paths, per_example, train_indices, torch.device("cpu"))
    sigma_q_full = train_stats["sigma_q"].float()
    sigma_k_full = train_stats["sigma_k"].float()
    n_layers, n_kv_heads, head_dim, _ = sigma_q_full.shape

    # Load model.
    print(f"[{ts()}] loading Qwen3-8B...", flush=True)
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen3-8B", device_map="auto", dtype_name="float16")
    device = get_model_device(model)
    print(f"[{ts()}] model on {device}", flush=True)

    bits = 4
    num_decode = 8

    results: dict[str, dict[str, dict[str, float]]] = {}
    for prompt_meta in chosen:
        task = prompt_meta["config"]
        # Read raw text from the per-example raw file (we have q_post / k_post / v).
        # For decode capture we need the prompt text. Reconstruct via the example's row index.
        # Easier: load the row from the split manifest and re-fetch the prompt.
        from experiments.stage1.data.kvpress_adapter import build_kvpress_dataset_spec
        from experiments.stage1.data.base import fetch_example
        spec = build_kvpress_dataset_spec(
            name="longbench",
            config_names=tuple({prompt_meta["config"]}),
            metadata_fields=("task", "answers", "length", "all_classes"),
        )
        example = fetch_example(spec, prompt_meta["config"], int(prompt_meta["row_index"]), tokenizer)
        prompt_text = example.prompt_text
        print(f"[{ts()}] === {task} (row {prompt_meta['row_index']}) ===", flush=True)

        captured = capture_prefill_decode_qk(model, tokenizer, prompt_text, num_decode_tokens=num_decode)
        k_prefill = captured["k_prefill"]   # (n_layers, n_kv_heads, T_k, d)
        q_prefill = captured["q_prefill"]   # (n_layers, n_q_heads, T_k, d)
        q_decode = captured["q_decode"]     # (n_layers, n_q_heads, num_decode-1, d)
        T_k = int(k_prefill.shape[2])
        T_q = int(q_decode.shape[2])
        print(f"[{ts()}] prompt_len={T_k} decode_steps={T_q} decoded_ids={captured['decoded_token_ids']}", flush=True)

        # Compute per-layer top-1 for each method.
        per_method = {"v3": {"prefill": [], "decode": []},
                      "jointqk": {"prefill": [], "decode": []}}

        for L in range(n_layers):
            if L == 0:
                continue  # match the layer-0-excluded headline convention
            k_full_l = k_prefill[L]  # (n_kv_heads, T_k, d)

            # v3 reconstruction
            recon_v3 = reconstruct_v3(k_full_l, bits=bits, seed=20260505, device=device)
            # jointqk reconstruction (with NEW pooled basis)
            recon_jq = reconstruct_jointqk(
                k_full_l, sigma_q_full[L], sigma_k_full[L], bits=bits, device=device,
            )

            # Top-1 against prefill q (calibration metric)
            pq = q_prefill[L]
            dq = q_decode[L]
            for method, recon in [("v3", recon_v3), ("jointqk", recon_jq)]:
                m_p = measure_top1(pq, k_full_l, recon, n_kv_heads, device)
                m_d = measure_top1(dq, k_full_l, recon, n_kv_heads, device)
                per_method[method]["prefill"].append(m_p)
                per_method[method]["decode"].append(m_d)

            del recon_v3, recon_jq
            torch.cuda.empty_cache()

        def avg(lst, key):
            return sum(d[key] for d in lst) / max(1, len(lst))

        task_result = {}
        for method in ("v3", "jointqk"):
            task_result[method] = {
                "prefill_top1": avg(per_method[method]["prefill"], "top1"),
                "prefill_top5": avg(per_method[method]["prefill"], "top5"),
                "decode_top1": avg(per_method[method]["decode"], "top1"),
                "decode_top5": avg(per_method[method]["decode"], "top5"),
            }
        results[task] = task_result
        print(f"[{ts()}] {task} results:", flush=True)
        for method in ("v3", "jointqk"):
            r = task_result[method]
            print(f"  {method}: prefill_top1={r['prefill_top1']:.4f}  decode_top1={r['decode_top1']:.4f}  "
                  f"(decode-prefill={r['decode_top1']-r['prefill_top1']:+.4f})", flush=True)
        print(f"  jointqk-v3 prefill_top1: {task_result['jointqk']['prefill_top1']-task_result['v3']['prefill_top1']:+.4f}", flush=True)
        print(f"  jointqk-v3 decode_top1:  {task_result['jointqk']['decode_top1']-task_result['v3']['decode_top1']:+.4f}", flush=True)

    out = REPO / "artifacts/stage1/calibration/longbench_compact8_qkv/05_reports/decode_q_top1.json"
    out.write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(f"\n[{ts()}] wrote {out}", flush=True)

    print(f"\n=== Summary: prefill-q vs decode-q top-1 (b={bits}, layer-0-excluded) ===")
    print(f"{'task':<14} {'method':<10} {'prefill_top1':>12} {'decode_top1':>12} {'Δ':>8}")
    for task in target_tasks:
        if task not in results: continue
        for method in ("v3", "jointqk"):
            r = results[task][method]
            print(f"{task:<14} {method:<10} {r['prefill_top1']:>12.4f} {r['decode_top1']:>12.4f} "
                  f"{r['decode_top1']-r['prefill_top1']:>+8.4f}")


if __name__ == "__main__":
    main()
