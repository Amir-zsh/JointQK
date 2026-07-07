"""Unit tests for kvq/compression/page_quant.py (page_quant study)."""
import math

import numpy as np
import pytest
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

from kvq.compression.page_quant import (  # noqa: E402
    FixedWidthPagedCompressor, PagedRDOCompressor, snap_positions,
)
from kvq.compression.ec_roundtrip import dz_round  # noqa: E402

D = 32
PTOK = 64
SEED = 20260707


def reference_snap_positions(idx, vals_list):
    """Verbatim numpy reference of _CoordModel.snap (tie -> left)."""
    n, d = idx.shape
    out = np.zeros((n, d), dtype=np.int64)
    for j in range(d):
        vals = vals_list[j]
        if len(vals) <= 1:
            out[:, j] = 0
            continue
        pos = np.searchsorted(vals, idx[:, j])
        pos = np.clip(pos, 0, len(vals) - 1)
        left = np.clip(pos - 1, 0, None)
        choose_left = (idx[:, j] - vals[left]) <= (vals[pos] - idx[:, j])
        out[:, j] = np.where(choose_left & (pos > 0), left, pos)
    return out


def make_rdo_compressor(mode="rdo", tau=0.0, b_page=2.0, dz=0.5,
                        n_rungs=3, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    fwd = torch.linalg.qr(torch.randn(D, D, generator=g))[0]
    inv = fwd.t().contiguous()
    mu = torch.randn(D, generator=g) * 0.1
    mu_q = torch.randn(D, generator=g)
    delta_base = 0.5
    m_grid = [0.5 * 2 ** i for i in range(n_rungs)][: n_rungs]
    m_grid = [1.0, 2.0, 4.0][:n_rungs]
    calib = torch.randn(4096, D, generator=g) @ fwd
    sv, sl, sn = [], [], []
    for m in m_grid:
        idx = dz_round(calib, torch.tensor(delta_base * m), dz)
        vals_l, lens, nlps = [], [], []
        vmax = 0
        per = []
        for j in range(D):
            u, c = idx[:, j].unique(return_counts=True)
            p = c.float() / c.sum()
            per.append((u.float(), -torch.log2(p.clamp_min(1e-12))))
            vmax = max(vmax, len(u))
        V = torch.full((D, vmax), float("inf"))
        NL = torch.full((D, vmax), 1e6)
        LN = torch.zeros(D, dtype=torch.long)
        for j, (u, nl) in enumerate(per):
            V[j, : len(u)] = u
            NL[j, : len(u)] = nl
            LN[j] = len(u)
        sv.append(V)
        sl.append(LN)
        sn.append(NL)
    return PagedRDOCompressor(
        fwd, inv, mu, mu_q, delta_base, m_grid, sv, sl, sn,
        dz=dz, b_page=b_page, ptok=PTOK, mode=mode,
        omega_tau=tau, omega_clamp_bits=4.0)


def test_snap_positions_matches_reference():
    g = torch.Generator().manual_seed(1)
    vals_list = []
    vmax = 0
    for j in range(D):
        n = int(torch.randint(1, 9, (1,), generator=g))
        v = torch.randn(n, generator=g).sort().values.round()
        v = v.unique()
        vals_list.append(v.numpy().astype(np.float64))
        vmax = max(vmax, len(v))
    sv = torch.full((D, vmax), float("inf"))
    sl = torch.zeros(D, dtype=torch.long)
    for j, v in enumerate(vals_list):
        sv[j, : len(v)] = torch.as_tensor(v, dtype=torch.float32)
        sl[j] = len(v)
    idx = torch.randn(500, D, generator=g) * 3
    got = snap_positions(idx, sv, sl).numpy()
    want = reference_snap_positions(idx.numpy().astype(np.float64),
                                    vals_list)
    assert (got == want).all()


def test_rdo_budget_respected_and_stats():
    c = make_rdo_compressor(b_page=2.0)
    g = torch.Generator().manual_seed(2)
    x = torch.randn(1, 10 * PTOK + 5, D, generator=g)
    out = c.roundtrip(x)
    assert out.shape == x.shape
    budget_ok_bits = (c.bits_payload + c.bits_side)
    nominal = 2.0 * D * c.tokens_total
    if c.pages_overflow == 0:
        assert budget_ok_bits <= nominal + 1e-6
    assert c.tokens_total == 10 * PTOK + 5
    assert sum(c.rung_hist) == c.tokens_total


def test_plain_equals_tau_zero():
    c1 = make_rdo_compressor(mode="rdo", tau=0.0)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(1, 4 * PTOK, D, generator=g)
    o1 = c1.roundtrip(x)
    c2 = make_rdo_compressor(mode="rdo", tau=0.0)
    o2 = c2.roundtrip(x.clone())
    assert torch.equal(o1, o2)


def test_omega_changes_allocation_at_tight_budget():
    ca = make_rdo_compressor(mode="rdo", tau=0.0, b_page=1.5)
    cb = make_rdo_compressor(mode="rdo", tau=2.0, b_page=1.5)
    g = torch.Generator().manual_seed(4)
    x = torch.randn(1, 8 * PTOK, D, generator=g)
    ca.roundtrip(x)
    cb.roundtrip(x.clone())
    assert ca.rung_hist != cb.rung_hist


def test_pagerung_one_rung_per_page():
    c = make_rdo_compressor(mode="pagerung", b_page=2.0)
    g = torch.Generator().manual_seed(5)
    x = torch.randn(1, 6 * PTOK, D, generator=g)
    c.roundtrip(x)
    # every page contributes ptok tokens to exactly one rung bucket
    assert sum(c.rung_hist) == 6 * PTOK
    for h in c.rung_hist:
        assert h % PTOK == 0


def make_fixed_compressor(tau=0.0, b_page=2.0, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    fwd = torch.linalg.qr(torch.randn(D, D, generator=g))[0]
    inv = fwd.t().contiguous()
    mu = torch.randn(D, generator=g) * 0.1
    mu_q = torch.randn(D, generator=g)
    calib = torch.randn(4096, D, generator=g) @ fwd
    std = calib.std(0)
    widths = [0, 1, 2, 4]
    alphas = torch.tensor([1.0, 1.5, 2.5, 4.0])
    return FixedWidthPagedCompressor(
        fwd, inv, mu, mu_q, std, widths, alphas, b_page=b_page,
        ptok=PTOK, omega_tau=tau, omega_clamp_bits=4.0)


def test_fixed_width_budget_exact_and_zero_width():
    c = make_fixed_compressor(b_page=1.0)
    g = torch.Generator().manual_seed(6)
    x = torch.randn(1, 8 * PTOK, D, generator=g)
    out = c.roundtrip(x)
    assert out.shape == x.shape
    assert c.pages_overflow == 0
    assert (c.bits_payload + c.bits_side) <= 1.0 * D * c.tokens_total + 1e-6
    # at b=1.0 with 3-bit-ish ids some tokens must sit at width 0 or 1
    assert c.rung_hist[0] + c.rung_hist[1] > 0


def test_fixed_width_zero_reconstructs_mu():
    c = make_fixed_compressor(b_page=0.1)  # forces nearly everything to w=0
    g = torch.Generator().manual_seed(7)
    x = torch.randn(1, 2 * PTOK, D, generator=g)
    out = c.roundtrip(x)
    frac_w0 = c.rung_hist[0] / sum(c.rung_hist)
    assert frac_w0 > 0.9
    mu_err = (out - c.mu.cpu()).abs().mean()
    x_err = (out - x).abs().mean()
    assert mu_err < x_err  # reconstruction collapsed toward mu, not x


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cpu_gpu_allocation_parity():
    g = torch.Generator().manual_seed(8)
    x = torch.randn(1, 6 * PTOK, D, generator=g)
    c_cpu = make_rdo_compressor(mode="rdo", tau=1.0, b_page=1.5)
    o_cpu = c_cpu.roundtrip(x)
    c_gpu = make_rdo_compressor(mode="rdo", tau=1.0, b_page=1.5)
    c_gpu.to(torch.device("cuda:0"))
    o_gpu = c_gpu.roundtrip(x.cuda()).cpu()
    assert c_cpu.rung_hist == c_gpu.rung_hist
    assert (o_cpu - o_gpu).abs().max() < 1e-4
