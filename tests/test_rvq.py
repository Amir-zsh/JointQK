"""Unit tests for kvq/compression/rvq.py (pgq2 Arm B)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

from kvq.compression.page_quant import HEADER_BITS  # noqa: E402
from kvq.compression.rvq import (  # noqa: E402
    NORM_BITS, ResidualVQPagedCompressor, nearest_codewords,
)

D = 32
PTOK = 64
N_SUB = 4
SD = 8
K = 16
S = 3
SEED = 20260707


def make_comp(mode="rdo", b_page=1.0, tau=0.0, uniform_stages=None,
              seed=SEED):
    g = torch.Generator().manual_seed(seed)
    fwd = torch.linalg.qr(torch.randn(D, D, generator=g))[0]
    inv = fwd.t().contiguous()
    mu = torch.randn(D, generator=g) * 0.1
    mu_q = torch.randn(D, generator=g)
    # stage-1 codebook sampled from real unit-direction subvectors (1-iter
    # k-means stand-in) so distortion actually drops with stages; deeper
    # stages are residual-scale gaussians
    calib = torch.randn(2048, D, generator=g) @ fwd
    u = calib / calib.norm(dim=1, keepdim=True).clamp_min(1e-8)
    perm = torch.randperm(D, generator=g)
    up = u[:, perm].reshape(-1, N_SUB, SD).permute(1, 0, 2)
    sel = torch.randperm(2048, generator=g)[:K]
    cbs = [up[:, sel, :].contiguous()]
    cbs += [torch.randn(N_SUB, K, SD, generator=g) * (0.08 * 0.5 ** s)
            for s in range(S - 1)]
    return ResidualVQPagedCompressor(
        fwd, inv, mu, mu_q, cbs, perm, b_page=b_page, ptok=PTOK,
        mode=mode, uniform_stages=uniform_stages, omega_tau=tau,
        omega_clamp_bits=4.0)


def test_rate_ladder():
    c = make_comp()
    # stage_bits = N_SUB * log2(K) = 4*4 = 16 each
    assert c.stage_bits == [16, 16, 16]
    assert c.rate_bits.tolist() == [0.0, 16 + NORM_BITS, 32 + NORM_BITS,
                                    48 + NORM_BITS]


def test_reconstruction_matches_manual_lookup_with_permutation():
    c = make_comp(mode="uniform", uniform_stages=2, b_page=8.0)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, PTOK, D, generator=g)
    out = c.roundtrip(x).squeeze(0)
    # manual: rotate, fp16 norm, permute, 2 RVQ stages, inverse-permute
    r = (x.squeeze(0) - c.mu) @ c.forward_map
    n16 = r.norm(dim=1).to(torch.float16).float()
    u = r / n16.clamp_min(1e-8).unsqueeze(1)
    up = u[:, c.perm].reshape(-1, N_SUB, SD).permute(1, 0, 2)
    acc = torch.zeros_like(up)
    resid = up.clone()
    for s in range(2):
        q = nearest_codewords(resid.contiguous(), c.codebooks[s])
        acc = acc + q
        resid = resid - q
    uh = acc.permute(1, 0, 2).reshape(-1, D)[:, c.inv_perm]
    manual = (uh * n16.unsqueeze(1)) @ c.inverse_map + c.mu
    # non-sink rows (sinks forced to max stage = 3, not 2)
    assert torch.allclose(out[4:], manual[4:], atol=1e-4)


def test_stage0_evicts_to_mu():
    c = make_comp(mode="uniform", uniform_stages=0)
    g = torch.Generator().manual_seed(2)
    x = torch.randn(1, 2 * PTOK, D, generator=g)
    out = c.roundtrip(x).squeeze(0)
    assert torch.allclose(out[4:], c.mu.expand_as(out[4:]), atol=1e-5)
    # only sinks pay: absolute 8-bit direction grid + norm
    assert abs(c.bits_payload - 4 * (8 * D + NORM_BITS)) < 1e-6
    assert c.bits_side == 2 * HEADER_BITS


def test_budget_and_overflow():
    # b=1.5*D*PTOK page budget comfortably fits the 8-bit sink grid at
    # test scale (production b=0.75 at d=128 fits it too: 4160 < 5856)
    c = make_comp(mode="rdo", b_page=1.5)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(1, 6 * PTOK + 9, D, generator=g)
    c.roundtrip(x)
    assert c.pages_overflow == 0
    assert (c.bits_payload + c.bits_side) <= 1.5 * D * c.tokens_total + 1e-6
    assert sum(c.rung_hist) == c.tokens_total
    assert c.rung_hist[-1] >= 4  # forced sinks


def test_sink_page_overflow_counted_when_budget_tiny():
    # at test scale the sink grid (272 bits/token) exceeds a 0.6 b/c page-0
    # budget: page 0 must be COUNTED as overflow, never silently dropped
    c = make_comp(mode="rdo", b_page=0.6)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(1, 6 * PTOK + 9, D, generator=g)
    c.roundtrip(x)
    assert c.pages_overflow >= 1


def test_monotone_distortion_in_stages():
    c = make_comp(mode="uniform", uniform_stages=1)
    g = torch.Generator().manual_seed(4)
    x = torch.randn(1, 2 * PTOK, D, generator=g)
    errs = []
    for s in (1, 2, 3):
        cs = make_comp(mode="uniform", uniform_stages=s)
        o = cs.roundtrip(x.clone()).squeeze(0)
        errs.append(float((o[4:] - x.squeeze(0)[4:]).square().mean()))
    assert errs[0] >= errs[1] >= errs[2]  # more stages, less error


def test_omega_changes_allocation():
    g = torch.Generator().manual_seed(5)
    x = torch.randn(1, 8 * PTOK, D, generator=g)
    ca = make_comp(mode="rdo", tau=0.0, b_page=1.5)
    cb = make_comp(mode="rdo", tau=1.5, b_page=1.5)
    ca.roundtrip(x)
    cb.roundtrip(x.clone())
    assert ca.rung_hist != cb.rung_hist


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cpu_gpu_parity():
    g = torch.Generator().manual_seed(6)
    x = torch.randn(1, 4 * PTOK, D, generator=g)
    c_cpu = make_comp(mode="rdo", tau=0.5, b_page=1.0)
    o_cpu = c_cpu.roundtrip(x)
    c_gpu = make_comp(mode="rdo", tau=0.5, b_page=1.0)
    c_gpu.to(torch.device("cuda:0"))
    o_gpu = c_gpu.roundtrip(x.cuda()).cpu()
    assert c_cpu.rung_hist == c_gpu.rung_hist
    assert (o_cpu - o_gpu).abs().max() < 1e-4
