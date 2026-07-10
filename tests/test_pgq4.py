"""pgq4 FoldedScalarPagedCompressor contract tests (plan4).

Covers: the fold identity (the kernel story the study rests on), exact rate
accounting, sink bypass, gain norm exactness, profile rungs, page-rung mode,
loader routing, and the Mode-B' flush-range logic.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402, F401

import math  # noqa: E402

import torch  # noqa: E402

from kvq.compression.pgq4_folded import (  # noqa: E402
    GAIN_BITS, SINK_BITS, WIDTH_LADDER, FoldedScalarPagedCompressor,
    uniform_quant,
)
from kvq.compression.page_quant import (  # noqa: E402
    HEADER_BITS, load_pgq_compressors_from_bundle,
)

torch.manual_seed(0)
D = 32


def make_comp(gain=False, mode="rdo", uniform_rung=None, b_page=2.0,
              grid="uniform", profiles=None, omega_tau=0.0,
              force_recent_pages=0, seed=1):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(D, D, generator=g)
    F = torch.linalg.qr(A)[0] + 0.05 * torch.randn(D, D, generator=g)
    G = torch.linalg.inv(F)
    mu = torch.randn(D, generator=g) * 0.1
    mu_q = torch.randn(D, generator=g)
    code_std = torch.rand(D, generator=g) * 0.5 + 0.75
    if profiles is None:
        profiles = torch.tensor([[w] * D for w in WIDTH_LADDER])
    alphas = torch.tensor([0.9, 0.55, 0.34])
    sink_scale = code_std * 24.0 / 127.0
    lm2 = torch.tensor([-1.51, -0.45, 0.45, 1.51])
    lm2[1] = 0.0                                     # forced zero level
    lm3 = torch.linspace(-2.2, 2.2, 8)
    lm3[lm3.abs().argmin()] = 0.0
    return FoldedScalarPagedCompressor(
        forward_map=F, inverse_map=G, mu=mu, mu_q=mu_q, code_std=code_std,
        profiles=profiles, alphas=alphas, sink_scale=sink_scale,
        b_page=b_page, grid=grid, lm_cents=[lm2, lm3], gain=gain,
        ptok=16, mode=mode, uniform_rung=uniform_rung,
        omega_tau=omega_tau, force_recent_pages=force_recent_pages)


def gauss_keys(T, comp, scale=1.0, seed=2):
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(T, D, generator=g) * comp.code_std * scale
    return r @ torch.linalg.inv(comp.forward_map) + comp.mu


def test_fold_identity():
    """q.k_hat == ((q @ G^T) * s).i + q.mu for the uniform grid — the
    decode-on-codes identity that folds basis and scales into the query."""
    comp = make_comp(mode="uniform", uniform_rung=3)   # w=4 everywhere
    k = gauss_keys(40, comp)
    k_hat = comp.roundtrip(k.clone()).float()

    w = WIDTH_LADDER[-1]
    s = comp.alphas[-1] * comp.code_std
    r = (k.float() - comp.mu) @ comp.forward_map
    lim = (1 << (w - 1)) - 1
    codes = torch.round(r[4:] / s).clamp(-lim, lim)    # skip sink bypass rows
    q = torch.randn(8, D)
    lhs = q @ k_hat[4:].T
    q_fold = (q @ comp.inverse_map.T) * s
    rhs = q_fold @ codes.T + (q @ comp.mu).unsqueeze(1)
    assert torch.allclose(lhs, rhs, atol=1e-3), \
        (lhs - rhs).abs().max().item()


def test_rate_exact_and_fill():
    # b=2.5 keeps page 0 solvent after the 8-bit sink escape at the test's
    # small (d=32, ptok=16) geometry; production (d=128, ptok=64) is solvent
    # from b=1.0
    comp = make_comp(b_page=2.5)
    k = gauss_keys(160, comp)
    comp.roundtrip(k)
    rate = (comp.bits_payload + comp.bits_side) / comp.tokens_total / D
    assert comp.pages_overflow == 0
    assert rate <= 2.5 + 1e-6
    assert rate >= 2.5 * 0.88, rate                    # RDO fills the pages


def test_uniform_rate_formula():
    comp = make_comp(mode="uniform", uniform_rung=2)   # w=3
    T = 40
    k = gauss_keys(T, comp)
    comp.roundtrip(k)
    P = (T + comp.ptok - 1) // comp.ptok
    expect = 3 * D * (T - 4) + 4 * SINK_BITS * D
    assert abs(comp.bits_payload - expect) < 1e-6
    assert abs(comp.bits_side - HEADER_BITS * P) < 1e-6


def test_sink_bypass_and_start_pos():
    comp = make_comp(b_page=1.0)
    k = gauss_keys(64, comp)
    k[:4] *= 6.0                                        # sink-like outliers
    k_hat = comp.roundtrip(k.clone()).float()
    rel = (k_hat[:4] - k[:4]).norm() / k[:4].norm()
    assert rel < 0.02, rel                              # 8-bit absolute grid
    # start_pos > 0: no bypass — same rows now go through bulk rungs
    comp2 = make_comp(b_page=1.0)
    k_hat2 = comp2.roundtrip(k.clone(), start_pos=64).float()
    rel2 = (k_hat2[:4] - k[:4]).norm() / k[:4].norm()
    assert rel2 > rel


def test_gain_norm_exact():
    comp = make_comp(gain=True, b_page=2.5)
    k = gauss_keys(96, comp)
    k_hat = comp.roundtrip(k.clone()).float()
    n0 = (k.float() - comp.mu).norm(dim=1)
    n1 = (k_hat - comp.mu).norm(dim=1)
    kept = n1 > 1e-6
    ratio = (n1[kept] / n0[kept])
    assert (ratio - 1).abs().max() < 2e-3, ratio        # fp16 gain tolerance
    # gain bits are charged
    assert comp.bits_payload > 96 * GAIN_BITS


def test_profile_rungs():
    nblk = 4
    blocks = torch.tensor([[0, 0, 0, 0], [2, 0, 0, 0], [3, 3, 0, 0],
                           [4, 4, 4, 4]])
    profiles = blocks.repeat_interleave(D // nblk, dim=1)
    comp = make_comp(profiles=profiles, mode="uniform", uniform_rung=2)
    k = gauss_keys(24, comp)
    k_hat = comp.roundtrip(k.clone()).float()
    r_hat = (k_hat - comp.mu) @ comp.forward_map
    # rung 2 = (3,3,0,0): tail half of coords must be exactly zero in code
    # domain (up to the inverse round-trip noise of the non-orthogonal maps)
    assert r_hat[4:, D // 2:].abs().max() < 1e-4
    expect_rate = 3 * (D // 2)
    assert abs(comp.bits_payload
               - (expect_rate * 20 + 4 * SINK_BITS * D)) < 1e-6


def test_pagerung_one_rung_per_page():
    comp = make_comp(mode="pagerung", b_page=2.0, omega_tau=0.5)
    k = gauss_keys(64, comp)                            # 4 pages of 16
    comp.roundtrip(k)
    # page 0 forced to top rung: its 12 non-sink tokens land in rung 3
    assert comp.rung_hist[-1] >= 12
    # honest rate at or under budget
    rate = (comp.bits_payload + comp.bits_side) / comp.tokens_total / D
    assert rate <= 2.0 + 1e-6


def test_force_recent_pages():
    comp = make_comp(b_page=1.0, force_recent_pages=2)
    k = gauss_keys(80, comp)                            # 5 pages of 16
    comp.roundtrip(k)
    # last 2 pages (32 tokens) forced to top rung
    assert comp.rung_hist[-1] >= 32
    rate = (comp.bits_payload + comp.bits_side) / comp.tokens_total / D
    assert rate > 1.0                                   # honest: forcing costs


def test_loader_routing(tmp_path):
    L, H, d = 3, 2, D
    g = torch.Generator().manual_seed(3)
    F = torch.randn(L, H, d, d, generator=g) * 0.1 + torch.eye(d)
    Finv = torch.linalg.inv(F)
    Rlay = torch.linalg.qr(torch.randn(L, d, d, generator=g))[0]
    stats_one = {
        "code_std": torch.rand(L, H, d, generator=g) + 0.5,
        "alphas": torch.tensor([0.9, 0.55, 0.34]).expand(L, H, 3).clone(),
        "sink_scale": torch.rand(L, H, d, generator=g) * 0.1 + 0.05,
    }
    blob = {
        "pgq_version": 5, "model_tag": "test", "ptok": 16,
        "n_layers": L, "n_kv_heads": H, "head_dim": d,
        "mu": torch.zeros(L, H, d), "mu_q": torch.randn(L, H, d, generator=g),
        "bases": {
            "qpca_unc": {"forward": F, "inverse": Finv},
            "oscar": {"forward": Rlay,
                      "inverse": Rlay.transpose(1, 2).contiguous()},
            "r_sym": {"forward": F, "inverse": Finv},
        },
        "stats": {k: stats_one for k in ("qpca_unc", "oscar", "r_sym")},
        "lm_cents": [torch.tensor([-1.5, 0.0, 0.5, 1.5]),
                     torch.linspace(-2.0, 2.0, 8)],
        "prof_head": torch.tensor([[0, 0, 0, 0], [4, 4, 4, 4]])
        .expand(L, H, 2, 4).clone(),
        "prof_layer": torch.tensor([[0, 0, 0, 0], [4, 4, 4, 4]])
        .expand(L, 2, 4).clone(),
        "prof_share": "layer",
        "px_profiles": torch.tensor([[0, 0, 0, 0], [4, 0, 0, 0],
                                     [4, 4, 4, 4]]),
        "prof_share_penalty": 0.01,
        "omega_tau_by_rate": {"2": 0.5},
        "omega_clamp_bits": 4.0,
    }
    p = tmp_path / "pgq4_test.pt"
    torch.save(blob, p)

    for method, nr in [("pgq_fold_rdo", 4), ("pgq_foldg_ea", 4),
                       ("pgq_foldlm_rdo", 4), ("pgq_foldob_rdo", 4),
                       ("pgq_foldrs_pgr", 4), ("pgq_prof_rdo", 2),
                       ("pgq_profpx_uni", 3), ("pgq_foldrw_rdo", 4)]:
        kb = 1 if method.endswith("uni") else 2.0
        comps, meta = load_pgq_compressors_from_bundle(str(p), method, kb)
        assert len(comps) == (L - 1) * H
        c = comps[(1, 0)]
        assert c.n_rungs == nr, (method, c.n_rungs)
        k = torch.randn(20, d)
        out = c.roundtrip(k)
        assert out.shape == k.shape
    c = load_pgq_compressors_from_bundle(str(p), "pgq_foldrw_rdo", 2.0)[0][
        (1, 0)]
    assert c.force_recent_pages == 4
    c = load_pgq_compressors_from_bundle(str(p), "pgq_foldg_ea", 2.0)[0][
        (1, 0)]
    assert c.gain and c.omega_tau == 0.5


def test_decode_flush_ranges():
    from kvq.presses.jointqk_press import decode_flush_ranges
    # W=0, chunk=8: flush as soon as 8 unquantized tokens accumulate
    assert decode_flush_ranges(qlen=100, total=108, recent=0, chunk=8) == \
        [(100, 108)]
    assert decode_flush_ranges(qlen=100, total=107, recent=0, chunk=8) == []
    # W=32: keep at least 32 fp16; flush aged chunks only
    assert decode_flush_ranges(qlen=100, total=140, recent=32, chunk=8) == \
        [(100, 108)]
    assert decode_flush_ranges(qlen=100, total=139, recent=32, chunk=8) == []
    # multiple chunks aged out at once (e.g. after a long stall)
    assert decode_flush_ranges(qlen=100, total=160, recent=32, chunk=8) == \
        [(100, 108), (108, 116), (116, 124)]
