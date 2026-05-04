"""Per-coordinate scalar quantizer for Stage 1E variants requiring non-uniform bit allocation.

Unlike `Stage1MSECompressor` (which unit-normalizes vectors before quantizing), this compressor:
- Optionally rotates by a per-head matrix Pi.
- Per-coordinate, scalar-quantizes using a 1D Lloyd-Max codebook scaled to per-coordinate std.
- Coordinates with `bits[j] = 0` collapse to zero output.

Storage cost is not modeled (we keep dequantized floats); this module measures distortion only.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from einops import rearrange

from experiments.stage1.toolkit.quantization import solve_lloyd_max


_UNIT_GAUSSIAN_CACHE: dict[int, torch.Tensor] = {}


def unit_gaussian_centroids(bits: int) -> torch.Tensor:
    """Return Lloyd-Max centroids for a unit-variance Gaussian with `bits` bits per sample.

    Cached per `bits`. Reuses `solve_lloyd_max` then rescales to unit variance:
    solve_lloyd_max(d, bits) returns centroids for sigma² = 1/d, so multiplying by sqrt(d)
    yields centroids for unit-variance Gaussian.
    """
    if bits <= 0:
        return torch.zeros(1)
    if bits in _UNIT_GAUSSIAN_CACHE:
        return _UNIT_GAUSSIAN_CACHE[bits].clone()
    d_canon = 64
    cents, _ = solve_lloyd_max(d_canon, bits)
    cents = cents * math.sqrt(d_canon)
    _UNIT_GAUSSIAN_CACHE[bits] = cents.clone()
    return cents


class PerCoordCompressor:
    """Per-coordinate scalar quantizer with optional pre/post linear maps.

    Args:
        bits_per_coord: int tensor of shape (d,). Coordinates with 0 bits → quantize to zero.
        std_per_coord:  float tensor of shape (d,). Per-coord standard deviation in the transformed basis.
        forward_map:    float tensor of shape (d, d) or None. If provided: transformed = states @ forward_map.
        inverse_map:    float tensor of shape (d, d) or None. After quantization: out = quantized @ inverse_map.
                        If None but forward_map is given, defaults to forward_map.T (correct for orthogonal forward_map).
                        For non-orthogonal forward_map (CCA's P_K.T), inverse_map MUST be the matching back-projection.

    Convention follows Stage 1D's `apply_headwise_linear`:
        transformed = states @ M_forward
        recon       = quantized @ M_inverse

    For an orthogonal forward_map (V eigenbasis with eigvecs as columns):
        forward_map = V         (states @ V projects onto eigvec basis)
        inverse_map = V.T       (states @ V @ V.T = states)

    For CCA (non-orthogonal):
        forward_map = P_K.T     (states @ P_K.T projects onto canonical-key basis: row form of P_K @ k_col)
        inverse_map = P_K_inv.T

    The compressor batches over leading dims of the input tensor.
    """

    def __init__(
        self,
        bits_per_coord: torch.Tensor,
        std_per_coord: torch.Tensor,
        forward_map: torch.Tensor | None = None,
        inverse_map: torch.Tensor | None = None,
    ) -> None:
        if bits_per_coord.dim() != 1 or std_per_coord.dim() != 1:
            raise ValueError("bits_per_coord and std_per_coord must be 1D")
        if bits_per_coord.shape != std_per_coord.shape:
            raise ValueError("shape mismatch between bits_per_coord and std_per_coord")
        self.d = int(bits_per_coord.shape[0])
        if forward_map is not None and forward_map.shape != (self.d, self.d):
            raise ValueError(f"forward_map must be (d, d) = ({self.d}, {self.d}), got {tuple(forward_map.shape)}")

        self.bits_int = bits_per_coord.round().clamp_min(0).to(torch.long)
        self.stds = std_per_coord.float()
        self.forward_map = forward_map.float() if forward_map is not None else None
        if forward_map is not None and inverse_map is None:
            inverse_map = forward_map.transpose(-1, -2)  # default: orthogonal case
        self.inverse_map = inverse_map.float() if inverse_map is not None else None

        # Build per-coord scaled codebook (variable-length per j).
        self.codebooks_padded: torch.Tensor  # (d, K_max)
        self.codebook_lengths: torch.Tensor  # (d,) int
        self._build_codebooks()

    def _build_codebooks(self) -> None:
        K_max = max(int(2 ** int(b)) for b in self.bits_int.tolist()) if self.bits_int.max() > 0 else 1
        K_max = max(K_max, 1)
        self.codebooks_padded = torch.full((self.d, K_max), float("inf"))
        self.codebook_lengths = torch.zeros(self.d, dtype=torch.long)
        for j in range(self.d):
            b = int(self.bits_int[j])
            if b == 0:
                self.codebooks_padded[j, 0] = 0.0
                self.codebook_lengths[j] = 1
            else:
                cents = unit_gaussian_centroids(b) * float(self.stds[j])
                k = cents.numel()
                self.codebooks_padded[j, :k] = cents
                self.codebook_lengths[j] = k

    def to(self, device: str | torch.device) -> "PerCoordCompressor":
        self.bits_int = self.bits_int.to(device)
        self.stds = self.stds.to(device)
        self.codebooks_padded = self.codebooks_padded.to(device)
        self.codebook_lengths = self.codebook_lengths.to(device)
        if self.forward_map is not None:
            self.forward_map = self.forward_map.to(device)
        if self.inverse_map is not None:
            self.inverse_map = self.inverse_map.to(device)
        return self

    @torch.no_grad()
    def roundtrip(self, states: torch.Tensor) -> torch.Tensor:
        """states: (..., d). Returns reconstruction of same shape."""
        if states.shape[-1] != self.d:
            raise ValueError(f"states last-dim {states.shape[-1]} != compressor.d {self.d}")
        device = states.device
        if self.codebooks_padded.device != device:
            self.to(device)
        leading_shape = states.shape[:-1]
        flat = states.reshape(-1, self.d).float()

        if self.forward_map is not None:
            transformed = flat @ self.forward_map
        else:
            transformed = flat

        # Vectorized per-coord nearest-centroid lookup, chunked along N to bound memory.
        # codebooks_padded[j, k] = +inf for k >= codebook_lengths[j], so argmin naturally
        # skips invalid entries. transformed: (N, d); codebooks_padded: (d, K_max).
        # The (N, d, K_max) diffs tensor blows up when K_max is large (water-fill methods
        # can put 10+ bits into a single coord -> K_max >= 1024). Chunk N to cap memory
        # at ~1 GB per chunk.
        N = transformed.shape[0]
        K_max = self.codebooks_padded.shape[1]
        BYTES_PER_ELEM = 4  # fp32
        MEM_BUDGET = 1 << 30  # 1 GiB
        chunk = max(1, MEM_BUDGET // (self.d * K_max * BYTES_PER_ELEM))

        cb_unsq = self.codebooks_padded.unsqueeze(0)  # (1, d, K_max)
        out = torch.empty_like(transformed)
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            chunk_t = transformed[start:end]                                    # (B, d)
            diffs = (chunk_t.unsqueeze(-1) - cb_unsq).abs()                     # (B, d, K_max)
            idx = diffs.argmin(dim=-1)                                          # (B, d)
            out[start:end] = torch.gather(
                cb_unsq.expand(end - start, -1, -1), 2, idx.unsqueeze(-1)
            ).squeeze(-1)
            del diffs

        if self.inverse_map is not None:
            out = out @ self.inverse_map
        return out.reshape(leading_shape + (self.d,))


def _per_head_bmm(x: torch.Tensor, maps: torch.Tensor, B: int, S: int) -> torch.Tensor:
    """Apply per-head linear map: `out[b, h, s, :] = x[b, h, s, :] @ maps[h]`.
    Uses one bmm by repacking the batch dim into the seq axis.
    """
    flat = rearrange(x, "b h s d -> h (b s) d")
    return rearrange(torch.bmm(flat, maps), "h (b s) d -> b h s d", b=B, s=S)


@torch.no_grad()
def batched_roundtrip(
    states: torch.Tensor,        # (B, H, S, d)
    forward_maps: torch.Tensor,  # (H, d, d)
    inverse_maps: torch.Tensor,  # (H, d, d)
    codebooks: torch.Tensor,     # (H, d, K_max)  — pad slots are +inf so argmin skips them
) -> torch.Tensor:
    """All `n_heads` heads of one layer's PerCoordCompressor as one op.

    Apply `forward_maps[h]`, look up the per-(head, coord) nearest centroid in
    `codebooks`, then apply `inverse_maps[h]`. Equivalent to looping `roundtrip`
    over heads but with batched ops.

    NOT byte-identical to the per-head loop. fp32 `mm` (used by per-head) and
    fp32 `bmm` (used here) are different CUDA kernels with different reduction
    orders; on A100 they differ by ~1e-6 per element. At 2-bit quantisation
    that's enough to flip argmin near centroid boundaries on a small fraction
    of elements, giving a few-pp F1 drift on tiny samples and ~0.1 pp at
    fraction=1.0. Higher bits (3+) are effectively identical. Per-head is the
    canonical path; this is gated behind `use_batched_compress` in JointQKPress.
    """
    B, H, S, d = states.shape
    K_max = codebooks.shape[-1]

    # Forward map: (B, H, S, d) → (B, H, S, d).
    transformed = _per_head_bmm(states.float(), forward_maps, B, S)

    # Per-coord nearest-centroid lookup. Chunk along S to bound the (B, H,
    # chunk_S, d, K_max) diff tensor to ~1 GiB at fp32.
    chunk_S = max(1, (1 << 30) // (max(B, 1) * H * d * K_max * 4))
    cb = rearrange(codebooks, "h d k -> 1 h 1 d k")  # broadcasts over (B, S)
    out = torch.empty_like(transformed)
    for s0 in range(0, S, chunk_S):
        s1 = min(s0 + chunk_S, S)
        chunk = transformed[:, :, s0:s1]                            # (B, H, c, d)
        idx = (chunk.unsqueeze(-1) - cb).abs().argmin(-1)           # (B, H, c, d)
        out[:, :, s0:s1] = torch.take_along_dim(cb, idx.unsqueeze(-1), dim=-1).squeeze(-1)

    # Inverse map.
    recon = _per_head_bmm(out, inverse_maps, B, S)
    return recon.to(states.dtype)


def stack_per_head(comps: list["PerCoordCompressor"]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack n_heads PerCoordCompressors into (forward_maps, inverse_maps, codebooks).

    Codebooks are padded along K_max to the per-layer max with +inf in unused
    slots so argmin naturally skips them.
    """
    d = comps[0].d
    K_max = max(int(c.codebooks_padded.shape[1]) for c in comps)
    cb = torch.full((len(comps), d, K_max), float("inf"))
    for h, c in enumerate(comps):
        k = c.codebooks_padded.shape[1]
        cb[h, :, :k] = c.codebooks_padded
    fwd = torch.stack([c.forward_map for c in comps], dim=0)
    inv = torch.stack([c.inverse_map for c in comps], dim=0)
    return fwd, inv, cb


def round_bits_to_integer(bits: torch.Tensor, total_bits: int | None = None) -> torch.Tensor:
    """Round continuous bit allocations to non-negative integers, optionally preserving the sum.

    bits: (..., d) continuous bit allocations.
    total_bits: if provided, the output sums to total_bits along the last dim (largest-remainder rounding).
    """
    bits = bits.clamp_min(0.0)
    if total_bits is None:
        return bits.round().to(torch.long)
    leading_shape = bits.shape[:-1]
    flat = bits.reshape(-1, bits.shape[-1])
    floor_bits = flat.floor()
    remainder = flat - floor_bits
    target = int(total_bits)
    out = floor_bits.clone()
    for i in range(flat.shape[0]):
        deficit = target - int(out[i].sum().item())
        if deficit <= 0:
            continue
        # Top-k indices by remainder
        _, top_idx = torch.topk(remainder[i], k=deficit)
        out[i, top_idx] += 1
    return out.reshape(leading_shape + (flat.shape[-1],)).to(torch.long)


def build_method_compressor(
    method: str,
    sigma_q_for_head: torch.Tensor,
    sigma_k_for_head: torch.Tensor,
    cca: dict[str, torch.Tensor] | None,
    mq_eigvals: torch.Tensor | None,
    mq_eigvecs: torch.Tensor | None,
    b_avg: float,
    r: int | None,
    seed: int,
    head_dim: int,
    V_h: torch.Tensor | None = None,
    R_sym: torch.Tensor | None = None,
) -> PerCoordCompressor | None:
    """Construct a PerCoordCompressor for the requested method on a single head.

    Returns None for the V3 baseline (caller uses Stage1MSECompressor for that path).

    Inputs are per-head (no batch dim). cca contains per-head P_K, P_K_inv, rho, etc.
    V_h: orthogonal CCA basis (P_K @ Σ_K^(1/2)) for cca_orth_* methods.
    R_sym: joint Q-K eigenbasis (eigvec_descending((Σ_Q Σ_K + Σ_K Σ_Q)/2)) for r_sym_* methods.
    """
    if method == "v3":
        return None

    total_bits = b_avg * head_dim

    if method.startswith("v_"):
        if mq_eigvecs is None or mq_eigvals is None:
            raise ValueError("V-basis methods require mq_eigvals and mq_eigvecs")
        # V (= mq_eigvecs) is orthogonal: V.T @ V = I. Forward map (row conv) = V; inverse = V.T.
        forward_map = mq_eigvecs
        inverse_map = mq_eigvecs.transpose(-1, -2)
        Sk_in_basis = mq_eigvecs.transpose(-1, -2) @ sigma_k_for_head @ mq_eigvecs
        sigma_k_diag = Sk_in_basis.diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
        weights = mq_eigvals.clamp_min(1e-30)
    elif method.startswith("cca_orth_"):
        if V_h is None:
            raise ValueError("cca_orth_* methods require V_h")
        # V_h is orthogonal: V_h V_h.T = I. Row form: canonical_row = k_row @ V_h.T.
        forward_map = V_h.transpose(-1, -2)
        inverse_map = V_h
        Sk_in_basis = V_h @ sigma_k_for_head @ V_h.transpose(-1, -2)
        sigma_k_diag = Sk_in_basis.diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
        Mq_in_basis = V_h @ sigma_q_for_head @ V_h.transpose(-1, -2)
        weights = Mq_in_basis.diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    elif method.startswith("r_sym_"):
        if R_sym is None:
            raise ValueError("r_sym_* methods require R_sym")
        # R_sym is orthogonal (eigvec of symmetric matrix). Row form: c_row = k_row @ R_sym.
        forward_map = R_sym
        inverse_map = R_sym.transpose(-1, -2)
        Sk_in_basis = R_sym.transpose(-1, -2) @ sigma_k_for_head @ R_sym
        sigma_k_diag = Sk_in_basis.diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
        Mq_in_basis = R_sym.transpose(-1, -2) @ sigma_q_for_head @ R_sym
        weights = Mq_in_basis.diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    elif method.startswith("cca_"):
        if cca is None:
            raise ValueError("CCA methods require cca payload")
        # P_K maps k_col → canonical_col. Row form: canonical_row = k_row @ P_K.T.
        forward_map = cca["P_K"].transpose(-1, -2)
        inverse_map = cca["P_K_inv"].transpose(-1, -2)
        Sk_in_basis = cca["P_K"] @ sigma_k_for_head @ cca["P_K"].transpose(-1, -2)
        sigma_k_diag = Sk_in_basis.diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
        # CCA's P_K is non-orthogonal, so Q-weighted reconstruction MSE is not weighted
        # by rho^2. Use the trace-formula metric in canonical-K coordinates:
        # diag((P_K_inv)^T Sigma_Q P_K_inv). This matches the corrected E2 simulation
        # and the F8/F11 Monte-Carlo derivation.
        metric_in_basis = (
            cca["P_K_inv"].transpose(-1, -2)
            @ sigma_q_for_head
            @ cca["P_K_inv"]
        )
        weights = metric_in_basis.diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    else:
        raise ValueError(f"Unknown method '{method}'")

    if method.endswith("_truncate") or method.endswith("_uniform"):
        if r is None or r <= 0 or r > head_dim:
            raise ValueError(f"Method '{method}' requires 1 <= r <= head_dim, got r={r}")
        per_coord_bits_int = max(1, int(round(total_bits / r)))
        bits = torch.zeros(head_dim, dtype=torch.long)
        bits[:r] = per_coord_bits_int
        # Adjust to exact total
        deficit = int(total_bits) - int(bits.sum())
        if deficit > 0:
            extra_idx = 0
            while deficit > 0 and extra_idx < r:
                bits[extra_idx] += 1
                deficit -= 1
                extra_idx += 1
        elif deficit < 0:
            for j in range(r - 1, -1, -1):
                if bits[j] > 0 and deficit < 0:
                    bits[j] -= 1
                    deficit += 1
                if deficit == 0:
                    break
        bits_int = bits
    elif method.endswith("_waterfill"):
        from experiments.stage1.toolkit.metric_transform import water_fill
        wf_input = (weights * sigma_k_diag).unsqueeze(0)  # add batch dim of 1
        bits_continuous = water_fill(wf_input, total_bits=total_bits).squeeze(0)
        bits_int = round_bits_to_integer(bits_continuous.unsqueeze(0), total_bits=int(total_bits)).squeeze(0)
        # Cap per-coord bits at 8 to bound K_max=256 (Lloyd-Max at 8b is ~continuous for
        # Gaussian sources; saving these bits and redistributing to lower-bit coords).
        MAX_BITS = 8
        excess = (bits_int - MAX_BITS).clamp_min(0).sum().item()
        bits_int = bits_int.clamp_max(MAX_BITS)
        # Redistribute saved bits to coords with lowest current allocation (until they hit MAX_BITS).
        while excess > 0:
            below = (bits_int < MAX_BITS).nonzero(as_tuple=True)[0]
            if below.numel() == 0:
                break
            min_val = bits_int[below].min()
            candidates = below[bits_int[below] == min_val]
            take = min(int(candidates.numel()), int(excess))
            bits_int[candidates[:take]] += 1
            excess -= take
    else:
        raise ValueError(f"Unknown allocation in method '{method}'")

    coord_stds = torch.sqrt(sigma_k_diag.float())
    return PerCoordCompressor(
        bits_per_coord=bits_int,
        std_per_coord=coord_stds,
        forward_map=forward_map.float(),
        inverse_map=inverse_map.float(),
    )
