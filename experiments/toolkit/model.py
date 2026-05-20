from __future__ import annotations

import torch

from experiments.toolkit.io import torch_dtype_from_name


def get_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def load_model_and_tokenizer(
    model_name: str,
    device_map: str = "auto",
    dtype_name: str = "float16",
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch_dtype_from_name(dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer
