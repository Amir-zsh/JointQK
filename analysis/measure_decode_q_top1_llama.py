#!/usr/bin/env python3
"""Llama-3.1-8B decode-q top-1 measurement (mirror of Qwen3 measure_decode_q_top1.py).

Loads sigma_q/sigma_k directly from the prebuilt
`cca_stats_llama31_8b_longbench_compact8_n400.pt` (we don't have a local
aggregate.pt for Llama, so we skip the combine_stats path).

Compares JointQK vs TurboQuant (Stage1MSECompressor) on decode-time queries
against compressed prefill keys.

Output:
  artifacts/calibration/longbench_compact8_qkv_llama31_8b/05_reports/decode_q_top1_llama.json
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402, F401

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipelines.calibration.analyze_bases import jointqk_basis
from kvq.capture.hooks import capture_rope_qk
from kvq.capture.model import get_model_device, load_model_and_tokenizer
from kvq.compression.per_coord import PerCoordCompressor, round_bits_to_integer
from kvq.compression.metric_transform import water_fill
from kvq.compression.lloyd_max import Stage1MSECompressor


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
    model, tokenizer, prompt_text: str, num_decode_tokens: int = 8, max_prompt_tokens: int = 8000
) -> dict[str, Any]:
    device = get_model_device(model)
    n_layers = int(model.config.num_hidden_layers)

    enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_tokens)
    input_ids = enc["input_ids"].to(device)
    prompt_len = int(input_ids.shape[-1])

    with capture_rope_qk(model) as (_q_pre, q_post_chunks, _k_pre, _k_post):
        outputs = model(input_ids=input_ids, use_cache=True)
        past_kv = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        decoded_ids = [int(next_token.item())]
        for _ in range(num_decode_tokens - 1):
            out = model(input_ids=next_token, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            decoded_ids.append(int(next_token.item()))

        all_q_chunks = list(q_post_chunks)

    expected = (1 + (num_decode_tokens - 1)) * n_layers
    if len(all_q_chunks) != expected:
        raise RuntimeError(
            f"q_post_chunks count {len(all_q_chunks)} != expected {expected}"
        )

    prefill_q_per_layer = [all_q_chunks[L] for L in range(n_layers)]
    decode_q_per_layer: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    idx = n_layers
    for step in range(num_decode_tokens - 1):
        for L in range(n_layers):
            decode_q_per_layer[L].append(all_q_chunks[idx])
            idx += 1

    q_decode_layers = []
    for L in range(n_layers):
        if not decode_q_per_layer[L]:
            continue
        cat = torch.cat(decode_q_per_layer[L], dim=2).squeeze(0).cpu().to(torch.float16)
        q_decode_layers.append(cat)
    q_decode = torch.stack(q_decode_layers, dim=0) if q_decode_layers else torch.empty(0)

    k_prefill = []
    for L in range(n_layers):
        kl = outputs.past_key_values.layers[L].keys.detach().to("cpu", dtype=torch.float16).squeeze(0)
        k_prefill.append(kl[:, :prompt_len, :])
    k_prefill = torch.stack(k_prefill, dim=0)

    prefill_q = []
    for L in range(n_layers):
        chunk = prefill_q_per_layer[L].detach().to("cpu", dtype=torch.float16).squeeze(0)
        prefill_q.append(chunk)
    prefill_q = torch.stack(prefill_q, dim=0)

    return {
        "k_prefill": k_prefill,
        "q_decode": q_decode,
        "q_prefill": prefill_q,
        "prompt_length": prompt_len,
        "decoded_token_ids": decoded_ids,
    }


def measure_top1(
    q: torch.Tensor,
    k_full: torch.Tensor,
    k_compressed: torch.Tensor,
    n_kv_heads: int,
    device: torch.device,
) -> dict[str, float]:
    q = q.to(device).float()
    k_full = k_full.to(device).float()
    k_compressed = k_compressed.to(device).float()
    group = q.shape[0] // n_kv_heads
    top1_num, top1_den = 0, 0
    top5_num, top5_den = 0, 0
    for h in range(n_kv_heads):
        q_h = q[h * group : (h + 1) * group]
        k_t_full = k_full[h]
        k_t_comp = k_compressed[h]
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
    k_full: torch.Tensor,
    sigma_q: torch.Tensor,
    sigma_k: torch.Tensor,
    bits: int,
    device: torch.device,
    max_coord_bits: int = 8,
) -> torch.Tensor:
    n_kv_heads, T, d = k_full.shape
    R_sym = jointqk_basis(sigma_q.unsqueeze(0), sigma_k.unsqueeze(0), eps=1e-4).squeeze(0)
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


_FNAME_RE = re.compile(
    r"^longbench__(?P<config>[A-Za-z0-9_\-]+)__row(?P<row>\d+)__(?P<split>train|test)\.pt$"
)


def pick_test_prompts(raw_root: Path, target_tasks: list[str]) -> list[dict[str, Any]]:
    """Return one prompt per target task (first match across shards)."""
    chosen: dict[str, dict[str, Any]] = {}
    shard_dirs = sorted([p for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("shard_")])
    for shard in shard_dirs:
        for f in sorted(shard.iterdir()):
            m = _FNAME_RE.match(f.name)
            if not m:
                continue
            if m.group("split") != "test":
                continue
            config = m.group("config")
            if config not in target_tasks or config in chosen:
                continue
            chosen[config] = {
                "config": config,
                "row_index": int(m.group("row")),
                "raw_path": str(f),
            }
        if len(chosen) == len(target_tasks):
            break
    return [chosen[t] for t in target_tasks if t in chosen]


def main() -> None:
    cca_path = REPO / "artifacts/bases/cca_stats_llama31_8b_longbench_compact8_n400.pt"
    raw_root = REPO / "artifacts/calibration/longbench_compact8_qkv_llama31_8b/01_raw"
    out_path = REPO / "artifacts/calibration/longbench_compact8_qkv_llama31_8b/05_reports/decode_q_top1_llama.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[{ts()}] loading Llama cca_stats from {cca_path.name}", flush=True)
    cca = torch.load(cca_path, map_location="cpu", weights_only=False)
    sigma_q_full = cca["sigma_q"].float()  # (n_layers, n_kv_heads, d, d)
    sigma_k_full = cca["sigma_k"].float()
    n_layers, n_kv_heads, head_dim, _ = sigma_q_full.shape
    print(f"[{ts()}] sigma shapes: n_layers={n_layers} n_kv_heads={n_kv_heads} d={head_dim}", flush=True)

    target_tasks = ["hotpotqa", "musique", "qasper", "qmsum"]
    chosen = pick_test_prompts(raw_root, target_tasks)
    print(f"[{ts()}] picked {len(chosen)} test prompts: "
          f"{[(p['config'], p['row_index']) for p in chosen]}", flush=True)

    print(f"[{ts()}] loading Llama-3.1-8B-Instruct...", flush=True)
    model, tokenizer = load_model_and_tokenizer(
        "meta-llama/Llama-3.1-8B-Instruct", device_map="auto", dtype_name="float16"
    )
    device = get_model_device(model)
    print(f"[{ts()}] model on {device}", flush=True)

    from kvq.data.kvpress_adapter import build_kvpress_dataset_spec
    from kvq.data.base import fetch_example

    bits = 2
    num_decode = 8

    results: dict[str, dict[str, dict[str, float]]] = {}
    for prompt_meta in chosen:
        task = prompt_meta["config"]
        spec = build_kvpress_dataset_spec(
            name="longbench",
            config_names=tuple({task}),
            metadata_fields=("task", "answers", "length", "all_classes"),
        )
        example = fetch_example(spec, task, int(prompt_meta["row_index"]), tokenizer)
        prompt_text = example.prompt_text
        print(f"[{ts()}] === {task} (row {prompt_meta['row_index']}) ===", flush=True)

        captured = capture_prefill_decode_qk(model, tokenizer, prompt_text, num_decode_tokens=num_decode)
        k_prefill = captured["k_prefill"]
        q_prefill = captured["q_prefill"]
        q_decode = captured["q_decode"]
        T_k = int(k_prefill.shape[2])
        T_q = int(q_decode.shape[2])
        print(f"[{ts()}] prompt_len={T_k} decode_steps={T_q} decoded_ids={captured['decoded_token_ids']}", flush=True)

        per_method = {"v3": {"prefill": [], "decode": []},
                      "jointqk": {"prefill": [], "decode": []}}

        for L in range(n_layers):
            if L == 0:
                continue
            k_full_l = k_prefill[L]

            recon_v3 = reconstruct_v3(k_full_l, bits=bits, seed=20260505, device=device)
            recon_jq = reconstruct_jointqk(
                k_full_l, sigma_q_full[L], sigma_k_full[L], bits=bits, device=device,
            )

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

        # Save incrementally so partial progress isn't lost.
        out_path.write_text(json.dumps(results, indent=2, default=str) + "\n")

    print(f"\n[{ts()}] wrote {out_path}", flush=True)
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
