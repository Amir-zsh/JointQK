"""Generic K-compressor-bundle press for the downstream F1 comparison.

Loads a pickled {(layer, kv_head): compressor} dict (any object exposing
.roundtrip(k) and .to(device)) into the K side, and reuses JointQKPress for the V
side (v_turboquant @ v_bits, identical to the other JQ-family cells) and the
prefill/decode hooks. Used for the scalar-INT2, OSCAR, and EC (rANS/Exp-Golomb)
bundles built by entropy_coding/build_method_bundles.py, so every method in the
report is benched under one calibration and V setup. (VQ has its own VQPress.)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

from kvq.presses.jointqk_press import JointQKPress

_EC_DIR = Path(__file__).resolve().parents[2] / "entropy_coding"
if str(_EC_DIR) not in sys.path:
    sys.path.insert(0, str(_EC_DIR))
# import the modules whose classes the bundles pickle, so unpickling resolves them
import run_pca_ec_deadzone  # noqa: E402,F401
import oscar_codec          # noqa: E402,F401


@dataclass
class KBundlePress(JointQKPress):
    k_bundle_path: str = ""
    _kb_loaded: bool = field(default=False, init=False, repr=False)

    def post_init_from_model(self, model):
        if not self._kb_loaded and self.quantize_k:
            if not self.k_bundle_path:
                raise ValueError("KBundlePress requires k_bundle_path")
            blob = torch.load(self.k_bundle_path, map_location="cpu", weights_only=False)
            comps = blob["comps"]                      # {(l, h): compressor}
            self._k_compressors = dict(comps)
            ls = sorted({l for (l, _) in comps}); hs = sorted({h for (_, h) in comps})
            self._n_layers = max(ls) + 1; self._n_kv_heads = max(hs) + 1
            self._kb_loaded = True
        qk_saved, cache_saved = self.quantize_k, self.compressor_cache_dir
        self.quantize_k = False; self.compressor_cache_dir = ""
        try:
            super().post_init_from_model(model)
        finally:
            self.quantize_k, self.compressor_cache_dir = qk_saved, cache_saved
