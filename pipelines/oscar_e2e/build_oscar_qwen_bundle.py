#!/usr/bin/env python3
"""Build a minimal pgq3-style bundle whose oscar_mixer is the OSCAR authors'
RotationZoo k-rotation for Qwen3-8B (per-layer, shared across kv heads,
applied rows @ R — same convention as OscarArmCompressor's forward_map).

This makes the pgq_oscar_uni arm a PRODUCTION-PARAMETERIZED emulation on
Qwen (authors' calibrated rotation + their serve defaults: group 128, clip
0.96, 64/256 windows), which is also the pre-registered O4 fallback."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

ROT = REPO / ("artifacts/oscar_e2e/rotzoo/Qwen3-8B/"
              "seq20000_prompt83_group128/k_rotation_qqt_r_h_pbr.pt")
OUT = REPO / "artifacts/page_quant2/pgq3_bundle__oscar_rotzoo__qwen3_8b.pt"
L, H, D, PTOK = 36, 8, 128, 64

rot = torch.load(ROT, map_location="cpu", weights_only=False)
layers = rot["layers"]
assert len(layers) == L, (len(layers), L)
mixer = torch.stack([layers[i]["rotation"].float() for i in range(L)])
orth = (mixer @ mixer.transpose(-1, -2) - torch.eye(D)).abs().max()
assert orth < 1e-3, f"rotation not orthogonal: {orth}"

blob = {
    "pgq_version": 4, "model_tag": "qwen3_8b", "basis": "oscar_rotzoo",
    "ptok": PTOK, "n_layers": L, "n_kv_heads": H, "head_dim": D,
    # mu_q is only consumed by omega-weighted (ea) variants; the oscar arm
    # runs uniform mode, so zeros are inert.
    "mu_q": torch.zeros(L, H, D),
    "oscar_mixer": mixer,                     # (L, d, d) authors' rotation
    "oscar_widths": [2, 3], "oscar_group": 128, "oscar_clip_q": 0.96,
    "omega_tau_by_rate": {}, "omega_clamp_bits": 4.0,
    "rotzoo_source": str(ROT.relative_to(REPO)),
    "rotzoo_sha8": hashlib.sha256(ROT.read_bytes()).hexdigest()[:8],
}
OUT.parent.mkdir(parents=True, exist_ok=True)
torch.save(blob, OUT)
print(f"saved {OUT} sha8={hashlib.sha256(OUT.read_bytes()).hexdigest()[:8]} "
      f"| mixer {tuple(mixer.shape)} orth_err {float(orth):.2e}")
