from __future__ import annotations

import importlib
from contextlib import contextmanager
from typing import Iterator

import torch

from experiments.stage1.toolkit.model import get_model_device


@contextmanager
def capture_rope_qk(
    model: torch.nn.Module,
) -> Iterator[tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]]:
    module_path = model.__class__.__module__
    modeling_module = importlib.import_module(module_path)
    target_function = "apply_rotary_pos_emb"
    if not hasattr(modeling_module, target_function):
        raise AttributeError(f"Model module '{module_path}' does not expose '{target_function}'.")

    original_function = getattr(modeling_module, target_function)
    q_pre_chunks: list[torch.Tensor] = []
    q_post_chunks: list[torch.Tensor] = []
    k_pre_chunks: list[torch.Tensor] = []
    k_post_chunks: list[torch.Tensor] = []

    def patched_function(q, k, *args, **kwargs):
        q_pre_chunks.append(q.detach().to("cpu", dtype=torch.float16))
        k_pre_chunks.append(k.detach().to("cpu", dtype=torch.float16))
        q_out, k_out = original_function(q, k, *args, **kwargs)
        q_post_chunks.append(q_out.detach().to("cpu", dtype=torch.float16))
        k_post_chunks.append(k_out.detach().to("cpu", dtype=torch.float16))
        return q_out, k_out

    setattr(modeling_module, target_function, patched_function)
    try:
        yield q_pre_chunks, q_post_chunks, k_pre_chunks, k_post_chunks
    finally:
        setattr(modeling_module, target_function, original_function)


@torch.inference_mode()
def run_prefill_and_capture(model: torch.nn.Module, input_ids: torch.Tensor):
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"Expected input_ids shape (1, L); got {tuple(input_ids.shape)}")
    input_ids = input_ids.to(get_model_device(model))
    with capture_rope_qk(model) as (q_pre_chunks, q_post_chunks, _k_pre_chunks, _k_post_chunks):
        outputs = model(input_ids=input_ids, use_cache=True)
    pre = torch.cat(q_pre_chunks, dim=0).float()
    post = torch.cat(q_post_chunks, dim=0).float()
    return outputs, pre, post


def _assemble_per_layer(chunks: list[torch.Tensor], n_layers: int) -> torch.Tensor:
    if len(chunks) % n_layers != 0:
        raise RuntimeError(
            f"RoPE hook fired {len(chunks)} times, which is not a multiple of n_layers={n_layers}. "
            "Either the model skipped some layers or the patch missed some calls."
        )
    per_layer = [torch.cat(chunks[layer::n_layers], dim=2) for layer in range(n_layers)]
    stacked = torch.stack(per_layer, dim=0)
    return stacked.squeeze(1)


def _extract_cache_kv(cache, n_layers: int) -> tuple[torch.Tensor, torch.Tensor]:
    keys = []
    values = []
    for layer in range(n_layers):
        layer_keys = cache.layers[layer].keys.detach().to("cpu", dtype=torch.float16)
        layer_values = cache.layers[layer].values.detach().to("cpu", dtype=torch.float16)
        keys.append(layer_keys.squeeze(0))
        values.append(layer_values.squeeze(0))
    return torch.stack(keys, dim=0), torch.stack(values, dim=0)


def _extract_cache_keys(cache, n_layers: int) -> torch.Tensor:
    keys = []
    for layer in range(n_layers):
        layer_keys = cache.layers[layer].keys.detach().to("cpu", dtype=torch.float16)
        keys.append(layer_keys.squeeze(0))
    return torch.stack(keys, dim=0)


@torch.inference_mode()
def run_prefill_qk_post_capture(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
) -> dict[str, torch.Tensor | int]:
    """Capture prefill-only post-RoPE Q and K tensors.

    This is the minimal artifact needed for K-basis calibration studies. It avoids
    decode generation and does not materialize V cache tensors on CPU.
    """
    device = get_model_device(model)
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"Expected input_ids shape (1, L); got {tuple(input_ids.shape)}")
    input_ids = input_ids.to(device)
    prompt_length = int(input_ids.shape[-1])
    n_layers = int(model.config.num_hidden_layers)

    with capture_rope_qk(model) as (_q_pre_chunks, q_post_chunks, _k_pre_chunks, _k_post_chunks):
        outputs = model(input_ids=input_ids, use_cache=True)

    q_post = _assemble_per_layer(q_post_chunks, n_layers)
    k_post_cache = _extract_cache_keys(outputs.past_key_values, n_layers)
    captured_length = int(q_post.shape[2])
    if captured_length != prompt_length or int(k_post_cache.shape[2]) != prompt_length:
        raise RuntimeError(
            f"Prefill capture length mismatch: prompt={prompt_length}, "
            f"q_post={captured_length}, k_post={int(k_post_cache.shape[2])}"
        )

    return {
        "q_post": q_post,
        "k_post": k_post_cache,
        "prompt_length": prompt_length,
        "total_length": prompt_length,
        "captured_length": captured_length,
        "generated_token_ids": torch.empty(0, dtype=torch.long),
    }


@torch.inference_mode()
def run_generation_and_capture(
    model: torch.nn.Module,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> dict[str, torch.Tensor | int]:
    device = get_model_device(model)
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"Expected input_ids shape (1, L); got {tuple(input_ids.shape)}")
    input_ids = input_ids.to(device)
    prompt_length = int(input_ids.shape[-1])
    n_layers = int(model.config.num_hidden_layers)
    pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    with capture_rope_qk(model) as (q_pre_chunks, q_post_chunks, k_pre_chunks, k_post_chunks):
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            use_cache=True,
            pad_token_id=pad_token_id,
        )

    sequences = outputs.sequences
    total_length = int(sequences.shape[-1])
    n_generated = total_length - prompt_length
    expected_calls = n_layers * n_generated
    if len(q_post_chunks) != expected_calls:
        raise RuntimeError(
            f"RoPE hook fired {len(q_post_chunks)} times; expected {expected_calls} "
            f"(n_layers={n_layers}, {n_generated} forward passes: 1 prefill + {max(n_generated - 1, 0)} decode). "
            "The patch may be bypassed by a kernel-fused path."
        )

    q_pre = _assemble_per_layer(q_pre_chunks, n_layers)
    q_post = _assemble_per_layer(q_post_chunks, n_layers)
    k_post_hook = _assemble_per_layer(k_post_chunks, n_layers)
    k_post_cache, v_cache = _extract_cache_kv(outputs.past_key_values, n_layers)

    captured_length = int(q_post.shape[2])
    tensors = {"q_pre": q_pre, "q_post": q_post, "k_post_hook": k_post_hook, "k_post_cache": k_post_cache, "v": v_cache}
    for name, tensor in tensors.items():
        if tensor.shape[2] != captured_length:
            raise RuntimeError(
                f"Seq-len disagreement across captured tensors: {name}={tensor.shape[2]} vs q_post={captured_length}. "
                f"Per-tensor shapes: {[(n, tuple(t.shape)) for n, t in tensors.items()]}"
            )
    if captured_length != total_length - 1 and captured_length != total_length:
        raise RuntimeError(
            f"Captured seq_len={captured_length} does not match expected total_length={total_length} "
            f"or total_length-1={total_length - 1} (prompt_length={prompt_length}, n_generated={n_generated})"
        )

    generated_token_ids = sequences[0, prompt_length:].detach().cpu()

    return {
        "q_pre": q_pre,
        "q_post": q_post,
        "k_post": k_post_cache,
        "v": v_cache,
        "prompt_length": prompt_length,
        "total_length": total_length,
        "captured_length": captured_length,
        "generated_token_ids": generated_token_ids,
    }
