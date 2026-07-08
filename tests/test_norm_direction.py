"""Unit tests for kvq/compression/norm_direction.py (pgq2 Arm A)."""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

from kvq.compression.norm_direction import (  # noqa: E402
    NORM_BITS, NormDirectionPagedCompressor,
)
from kvq.compression.page_quant import HEADER_BITS  # noqa: E402

D = 32
PTOK = 64
SEED = 20260707


def make_comp(mode="rdo", b_page=1.5, tau=0.0, uniform_rung=None,
              gate_theta=0.0, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    fwd = torch.linalg.qr(torch.randn(D, D, generator=g))[0]
    inv = fwd.t().contiguous()
    mu = torch.randn(D, generator=g) * 0.1
    mu_q = torch.randn(D, generator=g)
    calib = torch.randn(4096, D, generator=g) @ fwd
    n = calib.norm(dim=1, keepdim=True).clamp_min(1e-8)
    std_dir = (calib / n).std(0)
    profiles = torch.stack([
        torch.zeros(D, dtype=torch.long),
        torch.full((D,), 1, dtype=torch.long),
        torch.full((D,), 2, dtype=torch.long),
        torch.full((D,), 4, dtype=torch.long),
    ])
    return NormDirectionPagedCompressor(
        fwd, inv, mu, mu_q, std_dir, profiles, b_page=b_page, ptok=PTOK,
        mode=mode, uniform_rung=uniform_rung, omega_tau=tau,
        omega_clamp_bits=4.0, gate_theta=gate_theta, head_m_spread=1.0)


def test_rate_accounting_includes_norm_bits():
    c = make_comp(mode="uniform", uniform_rung=2)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 4 * PTOK, D, generator=g)
    c.roundtrip(x)
    T = 4 * PTOK
    # 4 sink tokens at rung 3 (4 bits), rest at rung 2 (2 bits), all + norm
    expect_payload = 4 * (4 * D + NORM_BITS) + (T - 4) * (2 * D + NORM_BITS)
    assert abs(c.bits_payload - expect_payload) < 1e-6
    assert c.bits_side == 4 * HEADER_BITS  # uniform: header only, no ids


def test_fp16_norm_used_in_reconstruction():
    c = make_comp(mode="uniform", uniform_rung=3, b_page=8.0)
    g = torch.Generator().manual_seed(2)
    x = torch.randn(1, PTOK, D, generator=g)
    out = c.roundtrip(x).squeeze(0)
    r = (x.squeeze(0) - c.mu) @ c.forward_map
    n_fp32 = r.norm(dim=1)
    n_fp16 = n_fp32.to(torch.float16).float()
    u = r / n_fp16.clamp_min(1e-8).unsqueeze(1)
    uh = c._quant_rung(u, 3)
    r_hat = (out - c.mu) @ c.forward_map
    # decoder contract: reconstruction == codebook direction * fp16 norm...
    assert torch.allclose(r_hat[4:], (uh * n_fp16.unsqueeze(1))[4:],
                          atol=1e-4)
    # ...and NOT the fp32 norm (fp16 rounding is visibly present: rel err
    # up to 4.9e-4 on norms ~ sqrt(D) => absolute diffs > 1e-4 somewhere)
    diff32 = (r_hat[4:] - (uh * n_fp32.unsqueeze(1))[4:]).abs().max()
    assert diff32 > 1e-4


def test_rung0_evicts_to_mu_at_zero_bits():
    c = make_comp(mode="uniform", uniform_rung=0)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(1, 2 * PTOK, D, generator=g)
    out = c.roundtrip(x).squeeze(0)
    # non-sink tokens reconstruct exactly to mu, cost 0 bits
    assert torch.allclose(out[4:], c.mu.expand_as(out[4:]), atol=1e-5)
    expect_payload = 4 * (4 * D + NORM_BITS)  # only the forced sinks pay
    assert abs(c.bits_payload - expect_payload) < 1e-6


def test_forced_sinks_max_rung():
    c = make_comp(mode="rdo", b_page=0.5)  # starvation: bulk should evict
    g = torch.Generator().manual_seed(4)
    x = torch.randn(1, 4 * PTOK, D, generator=g)
    c.roundtrip(x)
    assert c.rung_hist[-1] >= 4  # sinks are on the max rung regardless


def test_budget_respected_no_overflow():
    c = make_comp(mode="rdo", b_page=1.0)
    g = torch.Generator().manual_seed(5)
    x = torch.randn(1, 8 * PTOK + 5, D, generator=g)
    c.roundtrip(x)
    assert c.pages_overflow == 0
    total = c.bits_payload + c.bits_side
    assert total <= 1.0 * D * c.tokens_total + 1e-6
    assert sum(c.rung_hist) == c.tokens_total


def test_tau_zero_deterministic():
    g = torch.Generator().manual_seed(6)
    x = torch.randn(1, 4 * PTOK, D, generator=g)
    o1 = make_comp(mode="rdo", tau=0.0).roundtrip(x)
    o2 = make_comp(mode="rdo", tau=0.0).roundtrip(x.clone())
    assert torch.equal(o1, o2)


def test_omega_changes_allocation():
    g = torch.Generator().manual_seed(7)
    x = torch.randn(1, 8 * PTOK, D, generator=g)
    ca = make_comp(mode="rdo", tau=0.0, b_page=0.8)
    cb = make_comp(mode="rdo", tau=1.0, b_page=0.8)
    ca.roundtrip(x)
    cb.roundtrip(x.clone())
    assert ca.rung_hist != cb.rung_hist


def test_gate_full_theta_equals_plain():
    # theta huge -> gate always fires -> omega neutralized everywhere
    g = torch.Generator().manual_seed(8)
    x = torch.randn(1, 6 * PTOK, D, generator=g)
    plain = make_comp(mode="rdo", tau=0.0, b_page=0.8)
    gated = make_comp(mode="rdo", tau=1.0, b_page=0.8, gate_theta=1e9)
    o1 = plain.roundtrip(x)
    o2 = gated.roundtrip(x.clone())
    assert plain.rung_hist == gated.rung_hist
    assert torch.equal(o1, o2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cpu_gpu_parity():
    g = torch.Generator().manual_seed(9)
    x = torch.randn(1, 6 * PTOK, D, generator=g)
    c_cpu = make_comp(mode="rdo", tau=0.5, b_page=1.0)
    o_cpu = c_cpu.roundtrip(x)
    c_gpu = make_comp(mode="rdo", tau=0.5, b_page=1.0)
    c_gpu.to(torch.device("cuda:0"))
    o_gpu = c_gpu.roundtrip(x.cuda()).cpu()
    assert c_cpu.rung_hist == c_gpu.rung_hist
    assert (o_cpu - o_gpu).abs().max() < 1e-4
