"""Entropy-coded (EC) deadzone roundtrip compressor for the K side.

The quantizer is a per-coordinate deadzone uniform quantizer (step `delta`,
deadzone parameter `dz`) whose integer indices are entropy-coded by the paged
rANS codec with a frequency model FROZEN on calibration. F1 experiments only
need the reconstruction; the real codec (entropy_coding/kvq_codec.py) decodes
to the SAME indices, so this roundtrip must replicate two codec behaviours
exactly:

1. Deadzone round / dequant (entropy_coding/run_pca_ec_deadzone.py
   `_dz_round` / `_dz_dequant`).
2. **Snap-to-support**: indices unseen in calibration cannot be encoded; the
   codec snaps them to the nearest value in the frozen per-coord alphabet
   (kvq_codec._CoordModel.snap / coded_bits_eval: searchsorted, tie -> left).
   Skipping the snap would score reconstructions the real codec cannot emit.

Compressors are built offline by pipelines/ec/fit_ec_bundle.py into a bundle
.pt; JointQKPress loads them via `load_ec_compressors_from_bundle` when
`k_method` starts with "ec_". This module lives in kvq/ so the press disk
cache (torch.save of compressor objects) re-imports cleanly.
"""
from __future__ import annotations

from pathlib import Path

import torch

BUNDLE_VERSION = 1


def dz_round(r: torch.Tensor, delta: torch.Tensor, dz: float) -> torch.Tensor:
    """Deadzone uniform quantizer index. dz=0.5 -> round-to-nearest."""
    return torch.sign(r) * torch.floor(r.abs() / delta + dz)


def dz_dequant(idx: torch.Tensor, delta: torch.Tensor, dz: float) -> torch.Tensor:
    """Bin-centroid reconstruction; reduces to idx*delta at dz=0.5."""
    return (idx + torch.sign(idx) * (0.5 - dz)) * delta


class SnappedDeadzoneECCompressor:
    """Per-(layer, kv_head) EC roundtrip: center -> rotate -> deadzone-quantize
    -> snap to frozen alphabet -> dequantize -> rotate back -> uncenter.

    Interface matches what JointQKPress._quantize_layer needs: `.to(device)`
    and `.roundtrip(x)` with x (..., d).
    """

    def __init__(
        self,
        forward_map: torch.Tensor,   # (d, d); transformed = (x - mu) @ forward_map
        inverse_map: torch.Tensor,   # (d, d)
        mu: torch.Tensor,            # (d,)
        delta: torch.Tensor,         # (d,)
        dz: float,
        support_vals: torch.Tensor,  # (d, Vmax) float32, ascending, +inf padded
        support_lens: torch.Tensor,  # (d,) long
    ) -> None:
        self.d = int(forward_map.shape[0])
        self.fwd = forward_map.float()
        self.inv = inverse_map.float()
        self.mu = mu.reshape(1, -1).float()
        self.delta = delta.reshape(1, -1).float().clamp_min(1e-12)
        self.dz = float(dz)
        self.support_vals = support_vals.float()
        self.support_lens = support_lens.long()
        # kvpress press code reads .forward_map on PerCoordCompressor; mirror it.
        self.forward_map = self.fwd

    def to(self, device: str | torch.device) -> "SnappedDeadzoneECCompressor":
        self.fwd = self.fwd.to(device)
        self.inv = self.inv.to(device)
        self.mu = self.mu.to(device)
        self.delta = self.delta.to(device)
        self.support_vals = self.support_vals.to(device)
        self.support_lens = self.support_lens.to(device)
        self.forward_map = self.fwd
        return self

    def snap_indices(self, idx: torch.Tensor) -> torch.Tensor:
        """Snap (N, d) deadzone indices onto the frozen per-coord alphabet.

        Exact replica of coded_bits_eval's symbol mapping (numpy searchsorted
        side='left'; nearest value, tie -> left). Constant coords (alphabet
        size <= 1) always emit their single value — the codec spends 0 bits on
        them and the decoder reproduces the constant.
        """
        n = idx.shape[0]
        col = idx.transpose(0, 1).contiguous()                      # (d, N)
        pos = torch.searchsorted(self.support_vals, col)            # (d, N)
        last = (self.support_lens - 1).clamp_min(0).unsqueeze(1)    # (d, 1)
        pos = torch.minimum(pos, last)
        left = (pos - 1).clamp_min(0)
        v_pos = torch.gather(self.support_vals, 1, pos)
        v_left = torch.gather(self.support_vals, 1, left)
        choose_left = (v_left - col).abs() <= (v_pos - col).abs()
        snapped = torch.where(choose_left, v_left, v_pos)           # (d, N)
        # Constant coords: alphabet has a single value; emit it unconditionally.
        const = (self.support_lens <= 1).unsqueeze(1)
        snapped = torch.where(const, self.support_vals[:, :1].expand(-1, n), snapped)
        return snapped.transpose(0, 1)

    @torch.no_grad()
    def roundtrip(self, states: torch.Tensor) -> torch.Tensor:
        if states.shape[-1] != self.d:
            raise ValueError(f"states last-dim {states.shape[-1]} != {self.d}")
        if self.fwd.device != states.device:
            self.to(states.device)
        leading = states.shape[:-1]
        flat = states.reshape(-1, self.d).float()
        r = (flat - self.mu) @ self.fwd
        idx = dz_round(r, self.delta, self.dz)
        idx = self.snap_indices(idx)
        q = dz_dequant(idx, self.delta, self.dz)
        out = q @ self.inv + self.mu
        return out.reshape(leading + (self.d,))


def load_ec_compressors_from_bundle(
    path: str | Path,
) -> tuple[dict[tuple[int, int], SnappedDeadzoneECCompressor], dict]:
    """Load per-(layer, head) EC compressors from a fit_ec_bundle.py artifact.

    Layer 0 is skipped: bundles are fit under the layer0_full_precision
    convention and the press never quantizes layer 0.
    """
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    if blob.get("version") != BUNDLE_VERSION:
        raise ValueError(f"EC bundle {path}: version {blob.get('version')} != {BUNDLE_VERSION}")
    L, Hkv = int(blob["n_layers"]), int(blob["n_kv_heads"])
    comps: dict[tuple[int, int], SnappedDeadzoneECCompressor] = {}
    for l in range(1, L):
        for h in range(Hkv):
            comps[(l, h)] = SnappedDeadzoneECCompressor(
                forward_map=blob["forward"][l, h],
                inverse_map=blob["inverse"][l, h],
                mu=blob["mu"][l, h],
                delta=blob["delta"][l, h],
                dz=float(blob["dz"]),
                support_vals=blob["support_vals"][l, h],
                support_lens=blob["support_lens"][l, h],
            )
    meta = {k: blob[k] for k in blob if not isinstance(blob[k], torch.Tensor)}
    return comps, meta


def bundle_model_to_dict(blob: dict) -> dict:
    """Convert padded bundle tensors back to the {(l,h): [(vals, p), ...]} form
    consumed by entropy_coding's coded_bits_eval / kvq_codec ladder builders."""
    import numpy as np

    L, Hkv, d = blob["delta"].shape
    vals_t = blob["support_vals"]
    lens_t = blob["support_lens"]
    probs_t = blob["support_probs"]
    model = {}
    for l in range(L):
        for h in range(Hkv):
            per = []
            for j in range(d):
                n = int(lens_t[l, h, j])
                vals = vals_t[l, h, j, :n].numpy().astype(np.int64)
                p = probs_t[l, h, j, :n].numpy().astype(np.float64)
                p = p / max(p.sum(), 1e-300)
                per.append((vals, p))
            model[(l, h)] = per
    return model
