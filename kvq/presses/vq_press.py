"""kvpress press for group-VQ K-compression (downstream LongBench/RULER F1).

Reuses JointQKPress's whole machinery — layer-0 skip, V-side compressor
(v_turboquant @ v_bits, identical to jointqk_k2_v2 for a fair K-only comparison),
prefill/decode hooks — and only swaps the K compressors: instead of the scalar
per-coord QPCA quantizer, it loads a trained group-VQ codebook bundle
(train_group_vq_alloc.py / train_group_vq.py output) and builds one
GroupVQCompressor per (layer, kv_head). GroupVQCompressor.roundtrip already
matches the interface JointQKPress._quantize_layer expects.

press_kwargs:
    vq_codebook_path : path to a codebook payload (dict with forward/inverse/mean,
                       codebooks[(l,h)], bounds). REQUIRED.
    v_stats_path / v_method / v_bits : V-side (default v_turboquant @ 2b).
    layer0_full_precision : default True (headline convention).

The codebook's basis (forward/inverse/mean) is baked into the payload, so this
press needs no cca_stats_path. IMPORTANT: the codebook must be calibrated on the
SAME model whose K-cache it compresses (Qwen3-8B here) — and, for a fair vs-scalar
F1 comparison, ideally the same calibration corpus the scalar baseline used.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

from kvq.presses.jointqk_press import JointQKPress

# GroupVQCompressor lives in entropy_coding/ (not a package); add it to the path.
_EC_DIR = Path(__file__).resolve().parents[2] / "entropy_coding"
if str(_EC_DIR) not in sys.path:
    sys.path.insert(0, str(_EC_DIR))
from group_vq_codec import GroupVQCompressor, SinkRecentWrap, OutlierProtectWrap  # noqa: E402


@dataclass
class VQPress(JointQKPress):
    vq_codebook_path: str = ""
    # Small fp16 outlier-protection band: keep the first `vq_sink` and last
    # `vq_recent` prompt tokens unquantized (0/0 = protect nothing, original VQ).
    # Sized far below OSCAR's 64/256 so the bit-rate cost stays small.
    vq_sink: int = 0
    vq_recent: int = 0
    # Content-based outlier protection: restore the worst-reconstructed `vq_outlier_frac`
    # fraction of tokens to fp8 (stacks on top of the positional band). 0 = off.
    vq_outlier_frac: float = 0.0

    _vq_loaded: bool = field(default=False, init=False, repr=False)

    def post_init_from_model(self, model):
        # Build the K-side group-VQ compressors from the trained codebook bundle,
        # then defer to JointQKPress for the V side (and its caches / n_layers).
        if not self._vq_loaded and self.quantize_k:
            if not self.vq_codebook_path:
                raise ValueError("VQPress requires vq_codebook_path")
            payload = torch.load(self.vq_codebook_path, map_location="cpu", weights_only=False)
            F, inv, mean = payload["forward"], payload["inverse"], payload["mean"]
            bounds, cbs = payload["bounds"], payload["codebooks"]
            L, Hkv = F.shape[0], F.shape[1]
            self._n_layers, self._n_kv_heads = L, Hkv
            ptn = bool(payload.get("pertoken_norm", False))
            for l in range(L):
                if l == 0 and self.layer0_full_precision:
                    continue
                for h in range(Hkv):
                    gvq = GroupVQCompressor(
                        F[l, h], inv[l, h], mean[l, h], list(cbs[(l, h)]), bounds,
                        pertoken_norm=ptn)
                    comp = (SinkRecentWrap(gvq, self.vq_sink, self.vq_recent)
                            if (self.vq_sink or self.vq_recent) else gvq)
                    if self.vq_outlier_frac > 0:
                        comp = OutlierProtectWrap(comp, self.vq_outlier_frac)
                    self._k_compressors[(l, h)] = comp
            self._vq_loaded = True

        # V side + bookkeeping: run the parent with quantize_k masked off so it only
        # builds V. Disable the parent's compressor disk-cache first: its cache key
        # doesn't include vq_codebook_path, so a stale jointqk cache could otherwise
        # clobber the VQ K-compressors we just loaded.
        qk_saved, cache_saved = self.quantize_k, self.compressor_cache_dir
        self.quantize_k = False
        self.compressor_cache_dir = ""
        try:
            super().post_init_from_model(model)
        finally:
            self.quantize_k, self.compressor_cache_dir = qk_saved, cache_saved
