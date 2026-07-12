"""pgq8 PageDCTCompressor contract tests (plan8).

Covers: identity-page equivalence with the parent codec (protected pages and
decode chunks are bit-identical to pgq4), the DCT win on redundant data at
matched rate, exact rate accounting, sink escape, orthogonality, and loader
routing. Scale maps are FITTED from matching data (like fit_pgq8_stats) —
a flat map mis-scales the DC row by design, that's what the fit is for."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402, F401

import pytest  # noqa: E402
import torch  # noqa: E402

from kvq.compression.page_quant import load_pgq_compressors_from_bundle  # noqa: E402
from kvq.compression.pgq8_dct import PageDCTCompressor, dct_matrix  # noqa: E402
from test_pgq4 import D, gauss_keys, make_comp  # noqa: E402

torch.manual_seed(0)


def fit_dct_std(base, r_sample):
    """Per-(coefficient-row, coord) std, exactly like fit_pgq8_stats."""
    dm = dct_matrix(base.ptok)
    P = r_sample.shape[0] // base.ptok
    pg = r_sample[: P * base.ptok].reshape(P, base.ptok, D)
    y = torch.einsum("st,ptd->psd", dm, pg)
    return y.square().mean(0).sqrt().clamp_min(1e-6)


def make_dct_comp(dct_std=None, **kw):
    """PageDCTCompressor sharing every constant with test_pgq4's make_comp."""
    base = make_comp(**kw)
    if dct_std is None:
        dct_std = base.code_std.unsqueeze(0).repeat(base.ptok, 1)
    comp = PageDCTCompressor(
        dct_std=dct_std, forward_map=base.forward_map,
        inverse_map=base.inverse_map, mu=base.mu, mu_q=base.mu_q,
        code_std=base.code_std, profiles=base.profiles, alphas=base.alphas,
        sink_scale=base.sink_scale, b_page=base.b_page, grid=base.grid,
        lm_cents=base.lm_cents, ptok=base.ptok, mode="rdo",
        force_recent_pages=base.force_recent_pages)
    return comp, base


def r_to_k(comp, r):
    return (r @ torch.linalg.inv(comp.forward_map) + comp.mu)


def code_err(comp, k, out):
    r = (k.float() - comp.mu) @ comp.forward_map
    r_hat = (out.float() - comp.mu) @ comp.forward_map
    return r, r_hat


def test_dct_matrix_orthonormal():
    m = dct_matrix(64)
    assert torch.allclose(m @ m.T, torch.eye(64), atol=1e-5)
    assert torch.allclose(m[0], torch.full((64,), 1 / 8.0), atol=1e-6)


def test_decode_chunks_identical_to_parent():
    """start_pos > 0 (Mode-B' flush) pages never transform."""
    comp, base = make_dct_comp(grid="lm", b_page=2.0)
    k = gauss_keys(48, comp)
    a = comp.roundtrip(k.clone(), start_pos=100)
    b = base.roundtrip(k.clone(), start_pos=100)
    assert torch.equal(a, b)
    assert comp.bits_payload == base.bits_payload


def test_protected_pages_identical_to_parent():
    """With ptok=16, T=36, rw=2: pages are [sink][rw][rw-partial] — every
    page identity -> whole sequence bit-identical to pgq4."""
    comp, base = make_dct_comp(grid="lm", b_page=2.0, force_recent_pages=2)
    k = gauss_keys(36, comp)
    a = comp.roundtrip(k.clone())
    b = base.roundtrip(k.clone())
    assert torch.equal(a, b)
    assert comp.bits_payload == base.bits_payload
    assert comp.bits_side == base.bits_side


def test_sink_escape_exact():
    """Sink rows at exactly +-15 sigma (inside the 24-sigma escape grid)
    reconstruct near-exactly even though page 0 shares its budget."""
    comp, _ = make_dct_comp(grid="lm", b_page=2.0)
    g = torch.Generator().manual_seed(5)
    r = torch.randn(64, D, generator=g) * comp.code_std
    sgn = torch.where(torch.rand(4, D, generator=g) > 0.5, 1.0, -1.0)
    r[:4] = sgn * 15.0 * comp.code_std
    k = r_to_k(comp, r)
    out = comp.roundtrip(k.clone()).float()
    _, r_hat = code_err(comp, k, out)
    rel = (r[:4] - r_hat[:4]).norm() / r[:4].norm()
    assert rel < 0.02, float(rel)


def redundant_keys(comp, T, noise=0.15, seed=7):
    """Slow token-axis drift + small iid noise, scaled to the codec stats."""
    g = torch.Generator().manual_seed(seed)
    steps = torch.randn(T, D, generator=g) * 0.12
    drift = steps.cumsum(0)
    drift = drift - drift.mean(0)
    drift = drift / drift.std(0).clamp_min(1e-6)
    r = (drift + noise * torch.randn(T, D, generator=g)) * comp.code_std
    return r_to_k(comp, r), r


def test_redundant_pages_win_at_matched_rate():
    """Correlated tokens: the DCT arm must clearly beat the parent codec's
    code-space SE at the same honest rate (interior pages only)."""
    base_probe = make_comp(grid="lm", b_page=1.0)
    k_fit, r_fit = redundant_keys(base_probe, 4096, seed=11)
    dstd = fit_dct_std(base_probe, r_fit)
    comp, base = make_dct_comp(dct_std=dstd, grid="lm", b_page=1.0)
    k, r = redundant_keys(comp, 96, seed=7)

    out_d = comp.roundtrip(k.clone()).float()
    out_p = base.roundtrip(k.clone()).float()
    _, rh_d = code_err(comp, k, out_d)
    _, rh_p = code_err(comp, k, out_p)
    se_d = (rh_d - r)[16:].square().sum()
    se_p = (rh_p - r)[16:].square().sum()
    assert se_d < 0.7 * se_p, (float(se_d), float(se_p))
    rate_d = comp.bits_payload + comp.bits_side
    rate_p = base.bits_payload + base.bits_side
    assert abs(rate_d - rate_p) / rate_p < 0.02
    # toy geometry: page 0's sink escape exceeds its budget for BOTH codecs
    assert comp.pages_overflow == base.pages_overflow


def test_rate_budget_respected_no_sinks():
    """Decode chunks have no sink overshoot: all-in rate must sit at or
    under the nominal budget."""
    comp, _ = make_dct_comp(grid="lm", b_page=1.5)
    k = gauss_keys(128, comp)
    comp.roundtrip(k.clone(), start_pos=100)
    rate = (comp.bits_payload + comp.bits_side) / comp.tokens_total / D
    assert rate <= 1.5 + 1e-9, rate
    assert comp.pages_overflow == 0


def test_prefill_rate_matches_parent():
    """Prefill (incl. the toy-geometry page-0 sink overshoot) charges exactly
    what the parent charges."""
    comp, base = make_dct_comp(grid="lm", b_page=1.5)
    k = gauss_keys(128, comp)
    comp.roundtrip(k.clone())
    base.roundtrip(k.clone())
    assert comp.bits_side == base.bits_side
    r_d = comp.bits_payload / comp.tokens_total / D
    r_p = base.bits_payload / base.tokens_total / D
    assert abs(r_d - r_p) < 0.02, (r_d, r_p)


def test_orthogonality_roundtrip_high_budget():
    """Ample budget: transformed pages reconstruct within top-grid error —
    the inverse transform must not amplify anything."""
    probe = make_comp(grid="lm", b_page=8.0)
    g = torch.Generator().manual_seed(9)
    r_fit = torch.randn(4096, D, generator=g) * probe.code_std
    comp, base = make_dct_comp(dct_std=fit_dct_std(probe, r_fit),
                               grid="lm", b_page=8.0)
    r = torch.randn(48, D, generator=g) * comp.code_std
    k = r_to_k(comp, r)
    out = comp.roundtrip(k.clone()).float()
    out_p = base.roundtrip(k.clone()).float()
    _, rh = code_err(comp, k, out)
    _, rh_p = code_err(comp, k, out_p)
    rel = (r[16:32] - rh[16:32]).norm() / r[16:32].norm()
    rel_p = (r[16:32] - rh_p[16:32]).norm() / r[16:32].norm()
    assert rel < max(0.15, 2.0 * float(rel_p)), (float(rel), float(rel_p))


def test_loader_routing(tmp_path):
    L, H, d, ptok = 3, 2, D, 16
    g = torch.Generator().manual_seed(3)
    eye = torch.eye(d)
    blob = {
        "pgq_version": 5, "n_layers": L, "n_kv_heads": H, "head_dim": d,
        "ptok": ptok, "omega_clamp_bits": 4.0, "omega_tau_by_rate": {},
        "prof_share": "layer", "prof_width_ladder": [0, 2, 3, 4],
        "prof_layer": torch.tensor([[[0, 0], [2, 0], [3, 2], [4, 3]]] * L),
        "mu": torch.zeros(L, H, d), "mu_q": torch.zeros(L, H, d),
        "bases": {"qpca_unc": {"forward": eye.expand(L, H, d, d).clone(),
                               "inverse": eye.expand(L, H, d, d).clone()}},
        "stats": {"qpca_unc": {
            "code_std": torch.ones(L, H, d),
            "alphas": torch.rand(L, H, 3, generator=g) + 0.3,
            "sink_scale": torch.ones(L, H, d) * 0.2}},
        "lm_cents": [torch.tensor([-1.51, 0.0, 0.45, 1.51]),
                     torch.linspace(-2.2, 2.2, 8)],
        "dct_std": torch.ones(L, H, ptok, d, dtype=torch.float16),
    }
    p = tmp_path / "b.pt"
    torch.save(blob, p)
    comps, meta = load_pgq_compressors_from_bundle(str(p), "pgq_dctlm_rdo", 2.0)
    assert isinstance(comps[(1, 0)], PageDCTCompressor)
    assert comps[(1, 0)].dct_std.shape == (ptok, d)
    with pytest.raises(ValueError):
        load_pgq_compressors_from_bundle(str(p), "pgq_dctob_rdo", 2.0)
    del blob["dct_std"]
    torch.save(blob, p)
    with pytest.raises(ValueError):
        load_pgq_compressors_from_bundle(str(p), "pgq_dctlm_rdo", 2.0)
