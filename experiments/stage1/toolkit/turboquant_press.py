"""kvpress press wrapping TurboQuant V3 (random Hadamard + Lloyd-Max, K and V).

Uses `turboquant_pytorch.compressors_v3.TurboQuantV3` directly. residual_window=0 since
we manage decode scope via compress_decode flag.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import sys
from pathlib import Path

import torch
from kvpress.presses.base_press import BasePress
from kvpress.utils import extract_keys_and_values

# Ensure turboquant-pytorch is importable
_TQ_DIR = Path(__file__).resolve().parents[3] / "turboquant-pytorch"
if str(_TQ_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_TQ_DIR.parent))

from turboquant_pytorch.compressors_v3 import TurboQuantV3  # type: ignore


@dataclass
class TurboQuantPress(BasePress):
    k_bits: int = 4
    v_bits: int = 2
    protected_layers: int = 0          # we don't carve out first/last layers
    residual_window: int = 0            # decode managed via compress_decode flag
    seed: int = 42
    compress_decode: bool = False
    compression_ratio: float = 0.0

    _tq: dict = field(default_factory=dict, init=False, repr=False)

    def post_init_from_model(self, model):
        head_dim = getattr(model.config, "head_dim", None)
        if head_dim is None:
            head_dim = model.config.hidden_size // model.config.num_attention_heads
        n_layers = model.config.num_hidden_layers
        for L in range(n_layers):
            self._tq[L] = TurboQuantV3(
                head_dim=head_dim,
                key_bits=self.k_bits,
                value_bits=self.v_bits,
                residual_window=self.residual_window,
                layer_idx=L,
                n_layers=n_layers,
                protected_layers=self.protected_layers,
                seed=self.seed,
                device="cpu",
            )

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        L = module.layer_idx
        tq = self._tq[L]
        device = keys.device
        # MSECompressor inside TurboQuantV3 holds Pi/centroids on CPU; move to compress device.
        if tq.key_compressor.Pi.device != device:
            tq.key_compressor.Pi = tq.key_compressor.Pi.to(device)
            tq.key_compressor.centroids = tq.key_compressor.centroids.to(device)
            tq.val_compressor.Pi = tq.val_compressor.Pi.to(device)
            tq.val_compressor.centroids = tq.val_compressor.centroids.to(device)
            tq.key_compressor.device = str(device)
            tq.val_compressor.device = str(device)

        # MSECompressor materializes a (N, D, K) diffs tensor that OOMs on long contexts.
        # Chunk along the seq dim so each call sees at most CHUNK_TOKENS tokens.
        B, H, S, D = keys.shape
        CHUNK_TOKENS = 2048
        if S <= CHUNK_TOKENS:
            ck, cv = tq.compress_kv(keys, values)
            return tq.decompress_kv(ck, cv)

        keys_recon = torch.empty_like(keys)
        values_recon = torch.empty_like(values)
        for start in range(0, S, CHUNK_TOKENS):
            end = min(start + CHUNK_TOKENS, S)
            ck, cv = tq.compress_kv(keys[:, :, start:end], values[:, :, start:end])
            kr, vr = tq.decompress_kv(ck, cv)
            keys_recon[:, :, start:end] = kr
            values_recon[:, :, start:end] = vr
            del ck, cv, kr, vr
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
