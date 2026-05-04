"""kvpress press wrapping JointQK (K side) + V_BUILDERS (V side).

Designed to support all of Phase 1's per-side ablations through the same code path:
- quantize_k=True, quantize_v=False:  K-only sweep (V at fp16)
- quantize_k=False, quantize_v=True:  V-only sweep (K at fp16)
- quantize_k=True, quantize_v=True:   combined K+V (Phase 1C and downstream)
- compress_decode=True:                Mode B (Phase 6) — also compress decode-step keys/values
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from kvpress.presses.base_press import BasePress
from kvpress.utils import extract_keys_and_values

from experiments.stage1.toolkit.per_coord_quantization import (
    PerCoordCompressor,
    build_method_compressor,
)
from experiments.stage1.toolkit.v_compressor_adapter import build_v_compressor


@dataclass
class JointQKPress(BasePress):
    cca_stats_path: str = ""
    v_stats_path: str = ""
    v_method: str = "v_eigen_waterfill"
    k_method: str = "r_sym_waterfill"
    k_bits: int = 4
    v_bits: int = 2
    rank: int = 64
    layer0_full_precision: bool = False  # apples-to-apples: compress every layer like TurboQuant / KIVI
    quantize_k: bool = True
    quantize_v: bool = True
    compress_decode: bool = False
    eps: float = 1e-6  # whitening regularization (matches Stage-1E default)
    # kvpress evaluate.py reads .compression_ratio for logging; quantization presses
    # don't reduce seq_len, so this is purely informational (bits-per-coord proxy).
    compression_ratio: float = 0.0

    # populated in post_init_from_model
    _k_compressors: dict = field(default_factory=dict, init=False, repr=False)
    _v_compressors: dict = field(default_factory=dict, init=False, repr=False)
    _n_layers: int = field(default=0, init=False, repr=False)
    _n_kv_heads: int = field(default=0, init=False, repr=False)

    def post_init_from_model(self, model):
        """Load calibration stats and pre-build per-(layer, head) compressors."""
        if self.quantize_k:
            if not self.cca_stats_path:
                raise ValueError("quantize_k=True requires cca_stats_path")
            cca = torch.load(self.cca_stats_path, map_location="cpu", weights_only=False)
            n_layers, n_kv_heads, d, _ = cca["sigma_q"].shape
            self._n_layers = n_layers
            self._n_kv_heads = n_kv_heads
            for L in range(n_layers):
                if L == 0 and self.layer0_full_precision:
                    continue
                for h in range(n_kv_heads):
                    comp = build_method_compressor(
                        method=self.k_method,
                        sigma_q_for_head=cca["sigma_q"][L, h],
                        sigma_k_for_head=cca["sigma_k"][L, h],
                        cca={
                            "P_K": cca["P_K"][L, h],
                            "P_K_inv": cca["P_K_inv"][L, h],
                            "rho": cca["rho"][L, h],
                        },
                        mq_eigvals=cca["mq_eigvals"][L, h] if "mq_eigvals" in cca else None,
                        mq_eigvecs=cca["mq_eigvecs"][L, h] if "mq_eigvecs" in cca else None,
                        b_avg=float(self.k_bits),
                        r=self.rank,
                        seed=42,
                        head_dim=d,
                        V_h=cca["V_h"][L, h] if "V_h" in cca else None,
                        R_sym=cca["R_sym"][L, h] if "R_sym" in cca else None,
                    )
                    self._k_compressors[(L, h)] = comp

        if self.quantize_v:
            if not self.v_stats_path:
                raise ValueError("quantize_v=True requires v_stats_path")
            v_stats = torch.load(self.v_stats_path, map_location="cpu", weights_only=False)
            sigma_v = v_stats["sigma_v"]  # (n_layers, n_kv_heads, d, d)
            n_layers_v, n_kv_heads_v, d_v, _ = sigma_v.shape
            if self._n_layers and self._n_layers != n_layers_v:
                raise ValueError(
                    f"V stats n_layers={n_layers_v} != cca_stats n_layers={self._n_layers}"
                )
            self._n_layers = n_layers_v
            self._n_kv_heads = n_kv_heads_v
            for L in range(n_layers_v):
                for h in range(n_kv_heads_v):
                    self._v_compressors[(L, h)] = build_v_compressor(
                        method=self.v_method,
                        sigma_v_for_head=sigma_v[L, h],
                        head_dim=d_v,
                        bits=int(self.v_bits),
                        seed=42 + L * 1000 + h,
                    )

    def _quantize_k(self, keys: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Apply K compression per-head. keys: (batch, n_kv_heads, seq_len, head_dim)."""
        if layer_idx == 0 and self.layer0_full_precision:
            return keys
        out = torch.empty_like(keys)
        for h in range(keys.shape[1]):
            comp = self._k_compressors[(layer_idx, h)].to(keys.device)
            out[:, h] = comp.roundtrip(keys[:, h])
        return out

    def _quantize_v(self, values: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Apply V compression per-head. values: (batch, n_kv_heads, seq_len, head_dim)."""
        out = torch.empty_like(values)
        for h in range(values.shape[1]):
            comp = self._v_compressors[(layer_idx, h)].to(values.device)
            out[:, h] = comp.roundtrip(values[:, h])
        return out

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        """Compress full K/V tensors (used at prefill and, in Mode B, on the last token)."""
        L = module.layer_idx
        keys_recon = self._quantize_k(keys, L) if self.quantize_k else keys
        values_recon = self._quantize_v(values, L) if self.quantize_v else values
        return keys_recon, values_recon

    def forward_hook(self, module, args, kwargs, output):
        """Override BasePress.forward_hook to optionally fire on decode steps too.

        Default kvpress behavior (Mode A): early-return when past prefill, so only the
        prefill K/V cache is compressed; new keys/values stored at fp16.

        Mode B (compress_decode=True): on decode steps, compress only the LAST token of
        K and V (the new one), leaving previously-quantized history untouched.
        """
        if not self.compress_decode:
            return super().forward_hook(module, args, kwargs, output)

        # Mode B: handle prefill + per-step decode
        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        cache_layer = cache.layers[module.layer_idx]
        q_len = hidden_states.shape[1]

        is_prefill = kwargs["cache_position"][-1] <= q_len
        keys, values = extract_keys_and_values(cache, module.layer_idx)

        if is_prefill:
            keys, values = self.compress(module, hidden_states, keys, values, output[1] if len(output) > 1 else None, kwargs)
        else:
            # Compress only the last token (the just-appended decode key/value)
            new_k = keys[:, :, -1:, :]
            new_v = values[:, :, -1:, :]
            new_k_recon, new_v_recon = self.compress(module, hidden_states, new_k, new_v, None, kwargs)
            keys = torch.cat([keys[:, :, :-1, :], new_k_recon], dim=2)
            values = torch.cat([values[:, :, :-1, :], new_v_recon], dim=2)

        cache_layer.keys = keys
        cache_layer.values = values
        return output
