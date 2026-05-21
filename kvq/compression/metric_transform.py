from __future__ import annotations

import torch

from kvq.compression.lloyd_max import Stage1MSECompressor


TRANSFORM_FAMILIES = [
    "baseline_raw",
    "basis_only",
    "full_metric",
    "trace_matched_full_metric",
    "per_token_norm_matched_full_metric",
]


def apply_headwise_linear(states: torch.Tensor, matrices: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhsd,hdf->bhsf", states, matrices)


def repeat_kv_states(keys: torch.Tensor, num_query_heads: int) -> torch.Tensor:
    bsz, num_kv_heads, seq_len, dim = keys.shape
    if num_query_heads == num_kv_heads:
        return keys
    group_size = num_query_heads // num_kv_heads
    expanded = keys[:, :, None, :, :].expand(bsz, num_kv_heads, group_size, seq_len, dim)
    return expanded.reshape(bsz, num_query_heads, seq_len, dim)


def whitening_factor(cov: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric whitening: returns (W, W_inv) such that W @ cov @ W^T ≈ I.

    Uses eigh-based formulation, robust to near-singular covariance matrices.
    Adds eps * trace(cov)/d * I for Tikhonov regularization.
    """
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    d = cov.shape[-1]
    if cov.dim() == 2:
        scale = (cov.diagonal().sum() / d).clamp_min(1e-12)
        reg = eps * scale * torch.eye(d, dtype=cov.dtype, device=cov.device)
        cov_reg = cov + reg
        eigvals, eigvecs = torch.linalg.eigh(cov_reg)
        eigvals = eigvals.clamp_min(eps * scale)
        sqrt_inv = torch.diag(1.0 / torch.sqrt(eigvals))
        sqrt_pos = torch.diag(torch.sqrt(eigvals))
        W = sqrt_inv @ eigvecs.transpose(-1, -2)
        W_inv = eigvecs @ sqrt_pos
        return W, W_inv
    eigvals_list = []
    eigvecs_list = []
    for layer_cov in cov:
        scale = (layer_cov.diagonal().sum() / d).clamp_min(1e-12)
        reg = eps * scale * torch.eye(d, dtype=layer_cov.dtype, device=layer_cov.device)
        ev, vec = torch.linalg.eigh(layer_cov + reg)
        eigvals_list.append(ev.clamp_min(eps * scale))
        eigvecs_list.append(vec)
    eigvals = torch.stack(eigvals_list, dim=0)
    eigvecs = torch.stack(eigvecs_list, dim=0)
    sqrt_inv = torch.diag_embed(1.0 / torch.sqrt(eigvals))
    sqrt_pos = torch.diag_embed(torch.sqrt(eigvals))
    W = sqrt_inv @ eigvecs.transpose(-1, -2)
    W_inv = eigvecs @ sqrt_pos
    return W, W_inv


def compute_cca_basis(
    sigma_q: torch.Tensor,
    sigma_k: torch.Tensor,
    cross_qk: torch.Tensor,
    eps: float = 1e-4,
) -> dict[str, torch.Tensor]:
    """Per-head CCA basis from second moments.

    Solves SVD of W_Q @ C_QK @ W_K^T where W_X = Σ_X^{-1/2}, returning canonical
    correlations ρ and projection matrices P_K, P_Q such that
        - P_K maps a key vector k to its canonical-coordinates representation
          (i.e. P_K @ k has unit-variance whitened CCA components).
        - P_Q analogously for queries.
        - The canonical pairs (P_Q @ q)_i, (P_K @ k)_i are uncorrelated for i != j
          and have correlation ρ_i.

    Inputs are batched over heads:
        sigma_q: (H, d, d) — Σ_Q per kv-head (Q is GQA-pooled).
        sigma_k: (H, d, d) — Σ_K per kv-head.
        cross_qk: (H, d, d) — C_QK per kv-head (E[q k^T]).
        eps: regularization scaling for whitening.

    Returns dict with keys:
        - rho: (H, d) canonical correlations sorted descending in [0, 1] up to noise.
        - P_K: (H, d, d) where P_K[h] @ k projects key onto canonical-K basis.
        - P_Q: (H, d, d) projection onto canonical-Q basis.
        - P_K_inv: (H, d, d) inverse mapping.
        - W_K: (H, d, d) whitening factor for K (Σ_K^{-1/2}).
        - W_K_inv: (H, d, d) un-whitening (Σ_K^{1/2}).
    """
    if sigma_q.dim() == 2:
        sigma_q = sigma_q.unsqueeze(0)
        sigma_k = sigma_k.unsqueeze(0)
        cross_qk = cross_qk.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    sigma_q = sigma_q.float()
    sigma_k = sigma_k.float()
    cross_qk = cross_qk.float()

    W_Q, _W_Q_inv = whitening_factor(sigma_q, eps)
    W_K, W_K_inv = whitening_factor(sigma_k, eps)

    M = W_Q @ cross_qk @ W_K.transpose(-1, -2)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    rho = S.clamp(min=0.0, max=1.0 + 1e-3)

    P_K = Vh @ W_K
    P_K_inv = W_K_inv @ Vh.transpose(-1, -2)
    P_Q = U.transpose(-1, -2) @ W_Q

    out = {
        "rho": rho,
        "P_K": P_K,
        "P_K_inv": P_K_inv,
        "P_Q": P_Q,
        "W_K": W_K,
        "W_K_inv": W_K_inv,
    }
    if squeeze:
        out = {k: v.squeeze(0) for k, v in out.items()}
    return out


def _water_fill_unconstrained_row(log2_var: torch.Tensor, total_bits: float) -> torch.Tensor:
    """Solve unconstrained reverse water-fill on one row, returning continuous b_j >= 0.

    Returns bits in original coord order (not sorted). No max_bits cap.
    """
    d = log2_var.shape[0]
    sorted_log2, sorted_idx = torch.sort(log2_var, descending=True)
    cum = torch.cumsum(sorted_log2, dim=0)
    valid_b: torch.Tensor | None = None
    for k in range(1, d + 1):
        log_theta = cum[k - 1] / k - 2.0 * total_bits / k
        if sorted_log2[k - 1] <= log_theta:
            continue
        if k < d and sorted_log2[k] > log_theta:
            continue
        valid_b = (0.5 * (sorted_log2 - log_theta)).clamp_min(0.0)
        break
    if valid_b is None:
        # Fallback: spread budget uniformly (shouldn't happen for sane inputs)
        valid_b = torch.full_like(sorted_log2, total_bits / d).clamp_min(0.0)
    b_unsorted = torch.zeros_like(valid_b)
    b_unsorted[sorted_idx] = valid_b
    return b_unsorted


def water_fill(variances: torch.Tensor, total_bits: float, max_bits: float = 16.0) -> torch.Tensor:
    """Continuous reverse water-filling on per-coord 'variances' (could be λ_j σ_j² or σ_j²).

    Returns per-coord bit allocations b_j ∈ [0, max_bits] satisfying sum(b_j) = total_bits, allocating
    more bits to coords with higher variance. Bits are continuous (non-integer); caller rounds if
    the downstream quantizer requires integers.

    Mathematically: b_j = max(0, 0.5 * log2(variances_j / θ)) where θ is chosen so sum(b_j) = total_bits.
    Coords with variance below θ get zero bits. When the unconstrained solution would assign more than
    `max_bits` to some coord, that coord is fixed at `max_bits` and water-fill is recomputed on the
    remaining coordinates with reduced budget — iterating until no further saturation. This preserves
    the total bit budget (up to `min(total_bits, d * max_bits)`).
    """
    variances = variances.float().clamp_min(1e-30)
    if variances.dim() == 1:
        variances = variances.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    H, d = variances.shape

    capped_total = min(float(total_bits), d * float(max_bits))
    log2_var = torch.log2(variances)
    out = torch.zeros_like(variances)

    for h in range(H):
        active = torch.ones(d, dtype=torch.bool, device=variances.device)
        bits_h = torch.zeros(d, dtype=variances.dtype, device=variances.device)
        remaining_budget = capped_total

        for _iteration in range(d + 1):
            n_active = int(active.sum().item())
            if n_active == 0 or remaining_budget <= 0:
                break

            active_idx = torch.nonzero(active, as_tuple=True)[0]
            sub_log2 = log2_var[h, active_idx]
            sub_bits = _water_fill_unconstrained_row(sub_log2, remaining_budget)

            saturated_local = sub_bits >= max_bits
            if not saturated_local.any():
                bits_h[active_idx] = sub_bits.clamp_max(max_bits)
                break

            saturated_global = active_idx[saturated_local]
            bits_h[saturated_global] = max_bits
            n_sat = int(saturated_local.sum().item())
            active[saturated_global] = False
            remaining_budget -= n_sat * float(max_bits)
        else:
            # Should not happen — would require more iterations than coords.
            raise RuntimeError("water_fill iterative loop did not converge")

        out[h] = bits_h

    if squeeze:
        out = out.squeeze(0)
    return out


def factorize_metric(metric: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    sym = 0.5 * (metric + metric.transpose(-1, -2))
    sym = sym + eps * torch.eye(sym.shape[-1], dtype=sym.dtype, device=sym.device)
    try:
        factor = torch.linalg.cholesky(sym)
        inverse = torch.linalg.inv(factor)
    except RuntimeError:
        eigenvalues, eigenvectors = torch.linalg.eigh(sym)
        clipped = eigenvalues.clamp_min(eps)
        sqrt_vals = torch.sqrt(clipped)
        inv_sqrt_vals = 1.0 / sqrt_vals
        factor = eigenvectors @ torch.diag(sqrt_vals)
        inverse = torch.diag(inv_sqrt_vals) @ eigenvectors.transpose(-1, -2)
    return factor, inverse


def factorize_metric_batch(metrics: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    factors = []
    inverses = []
    for metric in metrics:
        factor, inverse = factorize_metric(metric.float(), eps)
        factors.append(factor)
        inverses.append(inverse)
    return torch.stack(factors, dim=0), torch.stack(inverses, dim=0)


def eigendecompose_metric_batch(metrics: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    sym = 0.5 * (metrics.float() + metrics.float().transpose(-1, -2))
    eye = torch.eye(sym.shape[-1], dtype=sym.dtype, device=sym.device).unsqueeze(0)
    sym = sym + eps * eye
    eigenvalues, eigenvectors = torch.linalg.eigh(sym)
    clipped = eigenvalues.clamp_min(eps)
    return clipped, eigenvectors


def _build_scaled_eigen_transform(
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    coord_scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale_embed = torch.diag_embed(coord_scales)
    inverse_embed = torch.diag_embed(1.0 / coord_scales.clamp_min(1e-12))
    transform = eigenvectors @ scale_embed
    inverse = inverse_embed @ eigenvectors.transpose(-1, -2)
    return transform, inverse


def build_metric_transform(
    metrics: torch.Tensor,
    variant: str,
    eps: float,
    gamma: float | None = None,
) -> dict[str, torch.Tensor | float | str]:
    if variant not in TRANSFORM_FAMILIES and variant != "gamma_sweep":
        raise ValueError(f"Unsupported transform variant '{variant}'.")

    if variant == "baseline_raw":
        dim = metrics.shape[-1]
        identity = torch.eye(dim, dtype=metrics.dtype, device=metrics.device).unsqueeze(0).repeat(metrics.shape[0], 1, 1)
        ones = torch.ones(metrics.shape[0], dtype=metrics.dtype, device=metrics.device)
        return {
            "variant": variant,
            "transform": identity,
            "inverse": identity,
            "eigenvalues": torch.ones(metrics.shape[0], dim, dtype=metrics.dtype, device=metrics.device),
            "eigenvectors": identity,
            "coord_scales": torch.ones(metrics.shape[0], dim, dtype=metrics.dtype, device=metrics.device),
            "trace_scale_alpha": ones,
        }

    eigenvalues, eigenvectors = eigendecompose_metric_batch(metrics, eps)
    dim = eigenvalues.shape[-1]
    ones = torch.ones(eigenvalues.shape[0], dtype=eigenvalues.dtype, device=eigenvalues.device)

    if variant == "basis_only":
        coord_scales = torch.ones_like(eigenvalues)
        trace_scale_alpha = ones
    elif variant == "full_metric":
        coord_scales = torch.sqrt(eigenvalues)
        trace_scale_alpha = ones
    elif variant == "trace_matched_full_metric":
        base_scales = torch.sqrt(eigenvalues)
        trace_scale_alpha = torch.sqrt(torch.tensor(float(dim), dtype=eigenvalues.dtype, device=eigenvalues.device) / eigenvalues.sum(dim=-1).clamp_min(1e-12))
        coord_scales = base_scales * trace_scale_alpha.unsqueeze(-1)
    elif variant == "per_token_norm_matched_full_metric":
        coord_scales = torch.sqrt(eigenvalues)
        trace_scale_alpha = ones
    elif variant == "gamma_sweep":
        if gamma is None:
            raise ValueError("gamma must be provided for gamma_sweep transforms.")
        base_scales = eigenvalues.pow(gamma / 2.0)
        gamma_trace = eigenvalues.pow(gamma).sum(dim=-1).clamp_min(1e-12)
        trace_scale_alpha = torch.sqrt(torch.tensor(float(dim), dtype=eigenvalues.dtype, device=eigenvalues.device) / gamma_trace)
        coord_scales = base_scales * trace_scale_alpha.unsqueeze(-1)
    else:
        raise ValueError(f"Unhandled transform variant '{variant}'.")

    transform, inverse = _build_scaled_eigen_transform(eigenvalues, eigenvectors, coord_scales)
    result: dict[str, torch.Tensor | float | str] = {
        "variant": variant,
        "transform": transform,
        "inverse": inverse,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "coord_scales": coord_scales,
        "trace_scale_alpha": trace_scale_alpha,
    }
    if gamma is not None:
        result["gamma"] = float(gamma)
    return result


def match_transformed_token_norms(
    reference_states: torch.Tensor,
    transformed_states: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    reference_norms = torch.linalg.vector_norm(reference_states.float(), dim=-1, keepdim=True)
    transformed_norms = torch.linalg.vector_norm(transformed_states.float(), dim=-1, keepdim=True)
    scales = reference_norms / transformed_norms.clamp_min(eps)
    return transformed_states * scales, scales


def prepare_variant_states(
    states: torch.Tensor,
    metrics: torch.Tensor,
    variant: str,
    eps: float,
    gamma: float | None = None,
) -> dict[str, torch.Tensor | float | str]:
    transform_payload = build_metric_transform(metrics.to(states.device), variant=variant, eps=eps, gamma=gamma)
    transform = transform_payload["transform"]
    inverse = transform_payload["inverse"]
    transformed = apply_headwise_linear(states.float(), transform)
    compressor_input = transformed
    norm_match_scales: torch.Tensor | None = None
    if variant == "per_token_norm_matched_full_metric":
        compressor_input, norm_match_scales = match_transformed_token_norms(states, transformed, eps)
    return {
        **transform_payload,
        "transformed": transformed,
        "compressor_input": compressor_input,
        "norm_match_scales": norm_match_scales,
        "input_norms": torch.linalg.vector_norm(states.float(), dim=-1),
        "transformed_norms": torch.linalg.vector_norm(transformed.float(), dim=-1),
    }


def geometry_aware_roundtrip(
    states: torch.Tensor,
    metrics: torch.Tensor,
    bits: int,
    seed: int,
    eps: float,
) -> torch.Tensor:
    factors, inverses = factorize_metric_batch(metrics.to(states.device), eps)
    transformed = apply_headwise_linear(states.float(), factors)
    compressor = Stage1MSECompressor(states.shape[-1], bits, seed=seed, device=states.device)
    transformed_recon = compressor.roundtrip(transformed)
    return apply_headwise_linear(transformed_recon, inverses)
