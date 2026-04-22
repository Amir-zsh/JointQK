from __future__ import annotations

from unittest import mock

import torch

from experiments.stage1.toolkit import (
    Stage1MSECompressor,
    apply_headwise_linear,
    compute_query_moments,
    factorize_metric_batch,
    geometry_aware_roundtrip,
)


def test_compute_query_moments_shapes_and_covariance():
    queries = torch.randn(3, 4, 11, 8)
    mean, cov, second = compute_query_moments(queries)
    assert mean.shape == (3, 4, 8)
    assert cov.shape == (3, 4, 8, 8)
    assert second.shape == (3, 4, 8, 8)
    recomposed = second - torch.einsum("lhd,lhe->lhde", mean, mean)
    assert torch.allclose(cov, recomposed, atol=1e-5)


def test_factorize_metric_batch_roundtrip():
    base = torch.randn(2, 6, 6)
    metrics = torch.einsum("hij,hkj->hik", base, base) + 1e-2 * torch.eye(6).unsqueeze(0)
    factors, inverses = factorize_metric_batch(metrics, eps=1e-6)
    restored = torch.matmul(factors, factors.transpose(-1, -2))
    identity = torch.matmul(factors, inverses)
    assert torch.allclose(restored, metrics, atol=1e-4)
    assert torch.allclose(identity, torch.eye(6).expand_as(identity), atol=1e-4)


def test_apply_headwise_linear_identity():
    states = torch.randn(1, 3, 5, 4)
    identity = torch.eye(4).repeat(3, 1, 1)
    transformed = apply_headwise_linear(states, identity)
    assert torch.allclose(states, transformed)


def test_stage1_mse_compressor_roundtrip_shape():
    states = torch.randn(1, 2, 7, 8)
    compressor = Stage1MSECompressor(head_dim=8, bits=2, seed=7, device="cpu")
    reconstructed = compressor.roundtrip(states)
    assert reconstructed.shape == states.shape


def test_factorize_metric_batch_fallback_inverse_is_correct():
    base = torch.randn(2, 5, 5)
    metrics = torch.einsum("hij,hkj->hik", base, base) + 1e-2 * torch.eye(5).unsqueeze(0)
    with mock.patch("torch.linalg.cholesky", side_effect=RuntimeError("fallback")):
        factors, inverses = factorize_metric_batch(metrics, eps=1e-6)
    identity = torch.matmul(factors, inverses)
    assert torch.allclose(identity, torch.eye(5).expand_as(identity), atol=1e-4)


def test_geometry_aware_roundtrip_keeps_float32_reconstruction():
    states = torch.randn(1, 2, 6, 8, dtype=torch.float16)
    metrics = torch.eye(8).repeat(2, 1, 1)
    reconstructed = geometry_aware_roundtrip(states, metrics, bits=3, seed=11, eps=1e-6)
    assert reconstructed.shape == states.shape
    assert reconstructed.dtype == torch.float32
