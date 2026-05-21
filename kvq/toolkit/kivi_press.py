"""kvpress press wrapping KIVI: per-channel int4 K + per-token int4 V."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from kvpress.presses.base_press import BasePress
from kvpress.utils import extract_keys_and_values

from kvq.toolkit.kivi_quantizer import (
    kivi_quantize_keys,
    kivi_quantize_values,
)


@dataclass
class KIVIPress(BasePress):
    k_bits: int = 4
    v_bits: int = 4
    group_size: int = 128
    compress_decode: bool = False
    layer0_full_precision: bool = True  # skip K/V quantization at layer 0 (anomalous attention sink). Default True per 2026-05-06 fairness convention; pass False to compress layer 0 too.
    compression_ratio: float = 0.0

    def post_init_from_model(self, model):
        # KIVI is stateless per-call; nothing to pre-build.
        pass

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if module.layer_idx == 0 and self.layer0_full_precision:
            return keys, values
        keys_recon = kivi_quantize_keys(keys, bits=self.k_bits, group_size=self.group_size)
        values_recon = kivi_quantize_values(values, bits=self.v_bits, group_size=self.group_size)
        return keys_recon, values_recon

    def forward_hook(self, module, args, kwargs, output):
        if not self.compress_decode:
            return super().forward_hook(module, args, kwargs, output)

        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        cache_layer = cache.layers[module.layer_idx]
        q_len = hidden_states.shape[1]
        is_prefill = kwargs["cache_position"][-1] <= q_len
        keys, values = extract_keys_and_values(cache, module.layer_idx)
        if is_prefill:
            keys, values = self.compress(module, hidden_states, keys, values,
                                         output[1] if len(output) > 1 else None, kwargs)
        else:
            new_k = keys[:, :, -1:, :]
            new_v = values[:, :, -1:, :]
            new_k_recon, new_v_recon = self.compress(module, hidden_states, new_k, new_v, None, kwargs)
            keys = torch.cat([keys[:, :, :-1, :], new_k_recon], dim=2)
            values = torch.cat([values[:, :, :-1, :], new_v_recon], dim=2)
        cache_layer.keys = keys
        cache_layer.values = values
        return output
