"""kvpress press wrapping JointQK (K side) + V_BUILDERS (V side).

Designed to support all of Phase 1's per-side ablations through the same code path:
- quantize_k=True, quantize_v=False:  K-only sweep (V at fp16)
- quantize_k=False, quantize_v=True:  V-only sweep (K at fp16)
- quantize_k=True, quantize_v=True:   combined K+V (Phase 1C and downstream)
- compress_decode=True:                Mode B (Phase 6) — also compress decode-step keys/values
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from kvpress.presses.base_press import BasePress
from kvpress.utils import extract_keys_and_values

from kvq.compression.ec_roundtrip import load_ec_compressors_from_bundle
from kvq.compression.per_coord import (
    PerCoordCompressor,
    batched_roundtrip,
    build_jointqk_compressor,
    stack_per_head,
)
from kvq.compression.v_compressor_adapter import build_v_compressor


@dataclass
class JointQKPress(BasePress):
    cca_stats_path: str = ""
    v_stats_path: str = ""
    v_method: str = "v_eigen_waterfill"
    k_method: str = "r_sym_waterfill"
    # For k_method "ec_*" (entropy-coded K): path to a fit_ec_bundle.py bundle.
    # Per-(layer, head) compressors are precomputed offline (deadzone delta +
    # frozen coder model fit on calibration rows); the press only loads them.
    ec_bundle_path: str = ""
    k_bits: int = 4
    v_bits: int = 2
    rank: int = 64
    layer0_full_precision: bool = True  # skip K/V quantization at layer 0 (anomalous attention sink). Default True per 2026-05-06 fairness convention; pass False to compress layer 0 too.
    quantize_k: bool = True
    quantize_v: bool = True
    compress_decode: bool = False
    eps: float = 1e-6  # whitening regularization (matches Stage-1E default)
    # kvpress evaluate.py reads .compression_ratio for logging; quantization presses
    # don't reduce seq_len, so this is purely informational (bits-per-coord proxy).
    compression_ratio: float = 0.0
    # Disk cache for compressors. If set, post_init writes/reads a .pt file keyed by
    # the calibration stats path mtime + press kwargs. Saves the ~100s cold rebuild
    # across worker restarts. Default: artifacts/_compressor_cache (auto-mkdir);
    # pass empty string "" to disable.
    compressor_cache_dir: str = "artifacts/_compressor_cache"

    # populated in post_init_from_model
    _k_compressors: dict = field(default_factory=dict, init=False, repr=False)
    _v_compressors: dict = field(default_factory=dict, init=False, repr=False)
    _n_layers: int = field(default=0, init=False, repr=False)
    _n_kv_heads: int = field(default=0, init=False, repr=False)
    # Lazy per-layer batched stacks (Lever 3). Built on first compress() call;
    # replaces the per-(layer, head) Python loop with one batched op per layer.
    # Disabled by default: in fp32, batched bmm and per-head mm CUDA kernels
    # have different reduction orders (~1e-6 fp32 diff per element), enough to
    # flip argmin at 2-bit quantisation and shift F1 by ~0.7 pp on small
    # samples. Forcing parity by changing the canonical per-head path's
    # precision (we tried fp16 round, fp64 matmul, fp16 throughout) all
    # degrade the per-head baseline F1, which is worse than living with the
    # speedup gap. Enable only if you don't need F1 parity with prior runs.
    _k_batched: dict = field(default_factory=dict, init=False, repr=False)
    _v_batched: dict = field(default_factory=dict, init=False, repr=False)
    use_batched_compress: bool = False

    def _cache_key(self) -> str:
        """Stable hash of all inputs that affect compressor construction.
        Include calibration-file mtime so a re-calibration invalidates the cache.
        """
        parts = [self.k_method, str(self.k_bits), str(self.rank),
                 self.v_method, str(self.v_bits),
                 "qk" if self.quantize_k else "_",
                 "qv" if self.quantize_v else "_",
                 "l0fp" if self.layer0_full_precision else "_"]
        for p in (self.cca_stats_path, self.v_stats_path, self.ec_bundle_path):
            if p and Path(p).exists():
                parts.append(str(Path(p).resolve()))
                parts.append(str(int(Path(p).stat().st_mtime)))
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    def _cache_path(self) -> Optional[Path]:
        if not self.compressor_cache_dir:
            return None
        return Path(self.compressor_cache_dir) / f"jointqk_{self._cache_key()}.pt"

    def _try_load_cache(self) -> bool:
        cache_path = self._cache_path()
        if cache_path is None or not cache_path.exists():
            return False
        try:
            blob = torch.load(cache_path, map_location="cpu", weights_only=False)
            self._k_compressors = blob.get("k_compressors", {})
            self._v_compressors = blob.get("v_compressors", {})
            self._n_layers = blob["n_layers"]
            self._n_kv_heads = blob["n_kv_heads"]
            return True
        except Exception:
            return False

    def _save_cache(self) -> None:
        cache_path = self._cache_path()
        if cache_path is None:
            return
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
            torch.save({
                "k_compressors": self._k_compressors,
                "v_compressors": self._v_compressors,
                "n_layers": self._n_layers,
                "n_kv_heads": self._n_kv_heads,
            }, tmp)
            os.replace(tmp, cache_path)  # atomic; tolerates racing workers
        except Exception:
            # Cache failure is never fatal — fall back to in-memory only.
            pass

    def post_init_from_model(self, model):
        """Load calibration stats and pre-build per-(layer, head) compressors.

        Idempotent: kvpress's BasePress.__call__ context manager calls this on
        every pipeline invocation, but compressor construction is expensive
        (~100s: torch.load 170 MB cca_stats + 256 K eigendecompositions + 256
        V eigendecompositions + Lloyd-Max codebook builds). Skip if already
        initialised.

        Disk-cache: if compressor_cache_dir is set, try loading a previously-
        built cache file keyed by (cca_stats_path mtime, k_method, k_bits, rank,
        v_method, v_bits, layer0_full_precision). Save on first build so future
        worker restarts skip the rebuild.
        """
        if (self._k_compressors or not self.quantize_k) and \
           (self._v_compressors or not self.quantize_v):
            return

        if self._try_load_cache():
            return

        if self.quantize_k and self.k_method.startswith("pgq_"):
            # Paged per-token-precision K (page_quant study). Reuses the
            # ec_bundle_path field (a bundle path either way; k_method
            # disambiguates, and _cache_key already mtimes it). k_bits is the
            # page budget in bits/coord (float allowed, e.g. 1.5).
            if not self.ec_bundle_path:
                raise ValueError(f"k_method={self.k_method!r} requires ec_bundle_path")
            from kvq.compression.page_quant import load_pgq_compressors_from_bundle
            self._k_compressors, pgq_meta = load_pgq_compressors_from_bundle(
                self.ec_bundle_path, self.k_method, float(self.k_bits))
            if not self.layer0_full_precision:
                raise ValueError("pgq_* bundles assume layer0_full_precision=True")
            self._n_layers = int(pgq_meta["n_layers"])
            self._n_kv_heads = int(pgq_meta["n_kv_heads"])
        elif self.quantize_k and self.k_method.startswith("ec_"):
            if not self.ec_bundle_path:
                raise ValueError(f"k_method={self.k_method!r} requires ec_bundle_path")
            self._k_compressors, ec_meta = load_ec_compressors_from_bundle(self.ec_bundle_path)
            if f"ec_{ec_meta['basis']}" != self.k_method:
                raise ValueError(
                    f"k_method={self.k_method!r} but bundle {self.ec_bundle_path} "
                    f"was fit with basis={ec_meta['basis']!r}")
            if not self.layer0_full_precision:
                raise ValueError("ec_* bundles are fit under layer0_full_precision=True")
            self._n_layers = int(ec_meta["n_layers"])
            self._n_kv_heads = int(ec_meta["n_kv_heads"])
        elif self.quantize_k:
            if not self.cca_stats_path:
                raise ValueError("quantize_k=True requires cca_stats_path")
            cca = torch.load(self.cca_stats_path, map_location="cpu", weights_only=False)
            n_layers, n_kv_heads, d, _ = cca["sigma_q"].shape
            self._n_layers = n_layers
            self._n_kv_heads = n_kv_heads
            # QPCA fields (optional). Bundle may contain them alongside R_sym when
            # both bases were built at calibration time. When k_method = "qpca_*",
            # they MUST be present; build_jointqk_compressor errors otherwise.
            has_qpca = all(k in cca for k in ("qpca_forward", "qpca_inverse", "qpca_eigvals"))
            has_rsym = "R_sym" in cca
            for L in range(n_layers):
                if L == 0 and self.layer0_full_precision:
                    continue
                for h in range(n_kv_heads):
                    self._k_compressors[(L, h)] = build_jointqk_compressor(
                        method=self.k_method,
                        sigma_q_for_head=cca["sigma_q"][L, h],
                        sigma_k_for_head=cca["sigma_k"][L, h],
                        R_sym=cca["R_sym"][L, h] if has_rsym else None,
                        b_avg=float(self.k_bits),
                        r=self.rank,
                        head_dim=d,
                        qpca_forward=cca["qpca_forward"][L, h] if has_qpca else None,
                        qpca_inverse=cca["qpca_inverse"][L, h] if has_qpca else None,
                        qpca_eigvals=cca["qpca_eigvals"][L, h] if has_qpca else None,
                    )

        if self.quantize_v:
            if self.v_method == "v_turboquant":
                if not self._n_layers:
                    self._n_layers = model.config.num_hidden_layers
                n_layers_v = self._n_layers
                n_kv_heads_v = getattr(model.config, "num_key_value_heads", None)
                if n_kv_heads_v is None:
                    n_kv_heads_v = model.config.num_attention_heads
                d_v = getattr(model.config, "head_dim", None)
                if d_v is None:
                    d_v = model.config.hidden_size // model.config.num_attention_heads
                self._n_kv_heads = n_kv_heads_v
                for L in range(n_layers_v):
                    if L == 0 and self.layer0_full_precision:
                        continue
                    for h in range(n_kv_heads_v):
                        self._v_compressors[(L, h)] = build_v_compressor(
                            method=self.v_method,
                            sigma_v_for_head=None,
                            head_dim=d_v,
                            bits=int(self.v_bits),
                            # Match TurboQuantV3's value-compressor seed for the layer.
                            seed=42 + L * 1000 + 500,
                        )
                # Persist before returning — otherwise repeated worker restarts
                # rebuild the K side from scratch when v_method=v_turboquant.
                self._save_cache()
                return

            if not self.v_stats_path:
                raise ValueError("quantize_v=True requires v_stats_path")
            v_stats = torch.load(self.v_stats_path, map_location="cpu", weights_only=False)
            # New (v2) calibration stores cov_v + mu_v; legacy (v1) only has
            # uncentered sigma_v. Prefer the centered form, fall back gracefully.
            cov_v = v_stats.get("cov_v", v_stats.get("sigma_v"))
            mu_v = v_stats.get("mu_v")  # None for legacy; centered compressor falls back to uncentered behaviour
            n_layers_v, n_kv_heads_v, d_v, _ = cov_v.shape
            if self._n_layers and self._n_layers != n_layers_v:
                raise ValueError(
                    f"V stats n_layers={n_layers_v} != cca_stats n_layers={self._n_layers}"
                )
            self._n_layers = n_layers_v
            self._n_kv_heads = n_kv_heads_v
            for L in range(n_layers_v):
                if L == 0 and self.layer0_full_precision:
                    continue
                for h in range(n_kv_heads_v):
                    self._v_compressors[(L, h)] = build_v_compressor(
                        method=self.v_method,
                        cov_v_for_head=cov_v[L, h],
                        mu_v_for_head=(mu_v[L, h] if mu_v is not None else None),
                        head_dim=d_v,
                        bits=int(self.v_bits),
                        seed=42 + L * 1000 + h,
                    )

        # Persist for future worker restarts (no-op if compressor_cache_dir disabled).
        self._save_cache()

    def _quantize_layer(self, x: torch.Tensor, layer_idx: int,
                        comps: dict, layer_cache: dict) -> torch.Tensor:
        """Round-trip one layer's heads through their compressors.

        Layout: x is (B, n_heads, S, d). Uses the batched stack when
        `use_batched_compress` is on AND every head is a PerCoordCompressor;
        otherwise falls back to a per-head Python loop. The fallback decision
        is cached as `None` so we don't re-check `isinstance` per call.
        """
        n_heads = x.shape[1]
        if self.use_batched_compress and layer_idx not in layer_cache:
            head_comps = [comps[(layer_idx, h)] for h in range(n_heads)]
            if all(isinstance(c, PerCoordCompressor) for c in head_comps):
                layer_cache[layer_idx] = tuple(t.to(x.device) for t in stack_per_head(head_comps))
            else:
                layer_cache[layer_idx] = None  # cache the fallback decision
        if self.use_batched_compress and layer_cache[layer_idx] is not None:
            return batched_roundtrip(x, *layer_cache[layer_idx])
        out = torch.empty_like(x)
        for h in range(n_heads):
            out[:, h] = comps[(layer_idx, h)].to(x.device).roundtrip(x[:, h])
        return out

    def _quantize_k(self, keys: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if layer_idx == 0 and self.layer0_full_precision:
            return keys
        return self._quantize_layer(keys, layer_idx, self._k_compressors, self._k_batched)

    def _quantize_v(self, values: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if layer_idx == 0 and self.layer0_full_precision:
            return values
        return self._quantize_layer(values, layer_idx, self._v_compressors, self._v_batched)

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
