"""Unit tests for kvq/compression/tcq.py (pgq3 families a/b/e)."""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

from kvq.compression.page_quant import (  # noqa: E402
    HEADER_BITS, build_hadamard, bit_reversal_perm,
)
from kvq.compression.tcq import (  # noqa: E402
    NORM_BITS, TCQPagedCompressor, fit_warped_lm_tables, sparse_k_quantize,
    tcq_greedy, tcq_viterbi,
)

D = 32
PTOK = 64
WIDTHS = [1, 2]
N_STATES = 4
SPARSE_KS = [4]
MAG_BITS = 8
SEED = 20260707


def unit_dirs(n, generator, d=D):
    x = torch.randn(n, d, generator=generator)
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)


def make_tables(gen, n_states=N_STATES, p=1.0, mixer=None):
    u = unit_dirs(4096, gen)
    if mixer is not None:
        u = u @ mixer
    return [fit_warped_lm_tables(
        u, 2 ** (w + 1) if n_states > 1 else 2 ** w, p) for w in WIDTHS]


def make_comp(mode="rdo", b_page=1.5, tau=0.0, uniform_rung=None,
              n_states=N_STATES, mixer=None, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    fwd = torch.linalg.qr(torch.randn(D, D, generator=g))[0]
    inv = fwd.t().contiguous()
    mu = torch.randn(D, generator=g) * 0.1
    mu_q = torch.randn(D, generator=g)
    tables = make_tables(g, n_states=n_states, mixer=mixer)
    return TCQPagedCompressor(
        fwd, inv, mu, mu_q, tables, WIDTHS, n_states, SPARSE_KS, MAG_BITS,
        b_page=b_page, ptok=PTOK, mode=mode, uniform_rung=uniform_rung,
        omega_tau=tau, omega_clamp_bits=4.0, mixer=mixer)


def test_rate_ladder():
    c = make_comp()
    idx_bits = math.ceil(math.log2(D))              # 5 at test scale
    assert c.rate_bits.tolist() == [
        0.0, NORM_BITS + 1 * D, NORM_BITS + 2 * D,
        NORM_BITS + 4 * (idx_bits + MAG_BITS)]


def test_zero_level_in_every_table():
    g = torch.Generator().manual_seed(SEED)
    for t in make_tables(g):
        assert (t == 0.0).any(dim=1).all()


def test_viterbi_beats_greedy():
    g = torch.Generator().manual_seed(1)
    table = make_tables(g)[1]                       # width-2, 8 levels
    u = unit_dirs(256, g)
    dv = (u - tcq_viterbi(u, table, N_STATES)).square().sum()
    dg = (u - tcq_greedy(u, table, N_STATES)).square().sum()
    assert dv <= dg + 1e-6
    assert dv < dg                                  # strict on real data


def test_one_state_is_scalar_nn():
    g = torch.Generator().manual_seed(2)
    table = make_tables(g, n_states=1)[0]           # width-1, 2 levels
    u = unit_dirs(64, g)
    out = tcq_viterbi(u, table, 1)
    idx = (u.unsqueeze(2) - table.unsqueeze(0)).abs().argmin(2)
    manual = table.unsqueeze(0).expand(64, -1, -1).gather(
        2, idx.unsqueeze(2)).squeeze(2)
    assert torch.equal(out, manual)


def test_uniform_reconstruction_matches_manual():
    c = make_comp(mode="uniform", uniform_rung=2, b_page=8.0)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(1, PTOK, D, generator=g)
    out = c.roundtrip(x).squeeze(0)
    r = (x.squeeze(0) - c.mu) @ c.forward_map
    n16 = r.norm(dim=1).to(torch.float16).float()
    u = r / n16.clamp_min(1e-8).unsqueeze(1)
    uh = tcq_viterbi(u, c.tables[1], N_STATES)      # rung 2 = width 2
    uh = uh / uh.norm(dim=1, keepdim=True).clamp_min(1e-8)  # decoder renorm
    manual = (uh * n16.unsqueeze(1)) @ c.inverse_map + c.mu
    assert torch.allclose(out[4:], manual[4:], atol=1e-4)


def test_rung0_evicts_to_mu():
    c = make_comp(mode="uniform", uniform_rung=0)
    g = torch.Generator().manual_seed(4)
    x = torch.randn(1, 2 * PTOK, D, generator=g)
    out = c.roundtrip(x).squeeze(0)
    assert torch.allclose(out[4:], c.mu.expand_as(out[4:]), atol=1e-5)
    assert abs(c.bits_payload - 4 * (8 * D + NORM_BITS)) < 1e-6
    assert c.bits_side == 2 * HEADER_BITS


def test_sparse_k_quantize_shape_and_grid():
    g = torch.Generator().manual_seed(5)
    u = unit_dirs(32, g)
    q = sparse_k_quantize(u, 4, MAG_BITS)
    assert int((q != 0).sum(1).max()) <= 4
    lim = (1 << (MAG_BITS - 1)) - 1
    assert torch.allclose(q * lim, (q * lim).round(), atol=1e-5)


def test_monotone_distortion_in_widths():
    g = torch.Generator().manual_seed(6)
    x = torch.randn(1, 2 * PTOK, D, generator=g)
    errs = []
    for rung in (1, 2):                             # widths 1, 2
        c = make_comp(mode="uniform", uniform_rung=rung)
        o = c.roundtrip(x.clone()).squeeze(0)
        errs.append(float((o[4:] - x.squeeze(0)[4:]).square().mean()))
    assert errs[0] >= errs[1]


def test_budget_and_overflow():
    c = make_comp(mode="rdo", b_page=1.5)
    g = torch.Generator().manual_seed(7)
    x = torch.randn(1, 6 * PTOK + 9, D, generator=g)
    c.roundtrip(x)
    assert c.pages_overflow == 0
    assert (c.bits_payload + c.bits_side) <= 1.5 * D * c.tokens_total + 1e-6
    assert sum(c.rung_hist) == c.tokens_total
    assert c.rung_hist[-1] >= 4                     # forced sinks


def test_sink_page_overflow_counted_when_budget_tiny():
    c = make_comp(mode="rdo", b_page=0.6)
    g = torch.Generator().manual_seed(7)
    x = torch.randn(1, 6 * PTOK + 9, D, generator=g)
    c.roundtrip(x)
    assert c.pages_overflow >= 1


def test_omega_changes_allocation():
    g = torch.Generator().manual_seed(8)
    x = torch.randn(1, 8 * PTOK, D, generator=g)
    ca = make_comp(mode="rdo", tau=0.0, b_page=1.0)
    cb = make_comp(mode="rdo", tau=1.5, b_page=1.0)
    ca.roundtrip(x)
    cb.roundtrip(x.clone())
    assert ca.rung_hist != cb.rung_hist


def test_hadamard_helpers_orthonormal():
    h = build_hadamard(D)
    assert torch.allclose(h @ h.t(), torch.eye(D), atol=1e-5)
    p = bit_reversal_perm(D)
    assert sorted(p.tolist()) == list(range(D))


def test_load_pgq3_bundle_all_families(tmp_path):
    from kvq.compression.page_quant import (
        PGQ3_BUNDLE_VERSION, load_pgq_compressors_from_bundle,
    )
    g = torch.Generator().manual_seed(SEED)
    L, H = 3, 2
    fwd = torch.linalg.qr(torch.randn(L, H, D, D, generator=g).reshape(
        -1, D, D))[0].reshape(L, H, D, D)
    tabs = make_tables(torch.Generator().manual_seed(SEED))
    h = build_hadamard(D)
    blob = {
        "pgq_version": PGQ3_BUNDLE_VERSION, "ptok": PTOK,
        "n_layers": L, "n_kv_heads": H, "head_dim": D,
        "forward": fwd, "inverse": fwd.transpose(-1, -2).contiguous(),
        "mu": torch.randn(L, H, D, generator=g) * 0.1,
        "mu_q": torch.randn(L, H, D, generator=g),
        "tcq_tables": [t.expand(L, H, -1, -1).contiguous() for t in tabs],
        "tcq_widths": WIDTHS, "tcq_states": N_STATES, "tcq_mixer": h,
        "sparse_ks": SPARSE_KS, "mag_bits": MAG_BITS, "warp_p": 1.0,
        "e8_profiles": torch.ones(L, H, 2, D // 8, dtype=torch.long),
        "e8_beta": torch.full((L, H), 0.5), "e8_mixer": h,
        "oscar_mixer": h, "oscar_widths": [2, 3], "oscar_group": D,
        "oscar_clip_q": 0.96,
        "omega_tau_by_rate": {"1.5": 0.5}, "omega_clamp_bits": 4.0,
    }
    path = tmp_path / "pgq3_bundle_test.pt"
    torch.save(blob, path)

    x = torch.randn(1, 2 * PTOK, D, generator=g)
    for method, kb in (("pgq_tcq_rdo", 1.5), ("pgq_tcq_ea", 1.5),
                       ("pgq_tcq_uni", 2), ("pgq_e8_rdo", 1.5),
                       ("pgq_e8_uni", 1), ("pgq_oscar_uni", 0)):
        comps, meta = load_pgq_compressors_from_bundle(str(path), method, kb)
        assert set(comps) == {(l, hh) for l in range(1, L)
                              for hh in range(H)}
        out = comps[(1, 0)].roundtrip(x.clone())
        assert out.shape == x.shape and torch.isfinite(out).all()
    assert meta["pgq_version"] == PGQ3_BUNDLE_VERSION


def test_mixed_domain_roundtrip_energy_consistent():
    # mixer rungs decode through H^T: reconstruction error in raw code
    # space equals the mixed-domain table error (orthogonality)
    h = build_hadamard(D)
    c = make_comp(mode="uniform", uniform_rung=2, mixer=h, seed=SEED)
    g = torch.Generator().manual_seed(9)
    x = torch.randn(1, PTOK, D, generator=g)
    out = c.roundtrip(x).squeeze(0)
    r = (x.squeeze(0) - c.mu) @ c.forward_map
    n16 = r.norm(dim=1).to(torch.float16).float()
    u = r / n16.clamp_min(1e-8).unsqueeze(1)
    uh_m = tcq_viterbi(u @ h, c.tables[1], N_STATES)
    uh_r = uh_m @ h.t()
    uh_r = uh_r / uh_r.norm(dim=1, keepdim=True).clamp_min(1e-8)
    manual = (uh_r * n16.unsqueeze(1)) @ c.inverse_map + c.mu
    assert torch.allclose(out[4:], manual[4:], atol=1e-4)
