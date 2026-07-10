"""Unit tests for kvq/compression/e8.py and oscar_arm.py (pgq3 c/f)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

from kvq.compression.page_quant import HEADER_BITS, build_hadamard  # noqa: E402
from kvq.compression.e8 import (  # noqa: E402
    E8PagedCompressor, e8_nearest, voronoi_roundtrip,
)
from kvq.compression.oscar_arm import (  # noqa: E402
    SCALE_ZERO_BITS, OscarArmCompressor,
)

D = 32
PTOK = 64
NSUB = D // 8
SEED = 20260707
# rung 0 evict, then water-filled-style lattice profiles
PROFILES = torch.tensor([[0] * NSUB, [1] * NSUB, [2, 1, 1, 1], [2] * NSUB])


def make_comp(mode="rdo", b_page=1.5, tau=0.0, uniform_rung=None,
              beta=0.35, mixer=None, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    fwd = torch.linalg.qr(torch.randn(D, D, generator=g))[0]
    inv = fwd.t().contiguous()
    mu = torch.randn(D, generator=g) * 0.1
    mu_q = torch.randn(D, generator=g)
    return E8PagedCompressor(
        fwd, inv, mu, mu_q, PROFILES, beta, mixer, b_page=b_page,
        ptok=PTOK, mode=mode, uniform_rung=uniform_rung, omega_tau=tau,
        omega_clamp_bits=4.0)


def test_e8_nearest_recovers_lattice_points():
    # integer-part sums must be even: [1.5, .5*6, -.5] - 1/2 = [1,0..0,-1]
    pts = torch.tensor([[0.] * 8,
                        [1., 1., 0., 0., 0., 0., 0., 0.],     # D8
                        [.5] * 8,                             # D8 + 1/2
                        [1.5, .5, .5, .5, .5, .5, .5, -.5]])
    noisy = pts + 0.05 * torch.randn(4, 8, generator=torch.Generator()
                                     .manual_seed(1))
    assert torch.allclose(e8_nearest(noisy), pts, atol=1e-6)


def test_e8_nearest_is_valid_lattice_member():
    g = torch.Generator().manual_seed(2)
    y = e8_nearest(torch.randn(512, 8, generator=g) * 2.0)
    # integer coords in the generator basis => genuine E8 points
    from kvq.compression.e8 import _E8_GEN_INV
    a = y.double() @ _E8_GEN_INV
    assert (a - a.round()).abs().max() < 1e-6


def test_voronoi_code_is_bijective_and_idempotent():
    from kvq.compression.e8 import _E8_GEN
    # m=1: enumerate ALL 2^8 codewords — decode must be injective (a true
    # 256-point codebook), every codeword in E8, and decode a fixed point
    idx = torch.cartesian_prod(*([torch.arange(2.)] * 8))
    cw = (idx.double() @ _E8_GEN)
    dec = voronoi_roundtrip(cw, 1)
    assert torch.unique(dec, dim=0).shape[0] == 256
    a = dec @ torch.linalg.inv(_E8_GEN)
    assert (a - a.round()).abs().max() < 1e-6
    assert torch.allclose(voronoi_roundtrip(dec, 1), dec, atol=1e-9)


def test_voronoi_idempotent_inside_region():
    # small-norm E8 points sit strictly inside Voronoi(4*E8)
    # (inradius 2*sqrt(2)): quotient encode/decode must be the identity
    pts = torch.tensor([[0.] * 8,
                        [1., 1., 0., 0., 0., 0., 0., 0.],
                        [1., -1., 0., 0., 0., 0., 0., 0.],
                        [.5] * 8]).double()
    assert torch.allclose(voronoi_roundtrip(pts, 2), pts, atol=1e-9)


def test_voronoi_wraps_outside_region():
    y = torch.tensor([[8., 0., 0., 0., 0., 0., 0., 0.]])   # far outside m=1
    y_hat = voronoi_roundtrip(y, 1)
    assert not torch.allclose(y_hat, y)
    assert torch.allclose(e8_nearest(y_hat), y_hat, atol=1e-6)  # still E8
    assert voronoi_roundtrip(y, 0).abs().max() == 0


def test_rate_ladder_and_escape():
    # lattice rungs carry the 16b raw norm; the all-zero profile is a true
    # evict rung at 0 bits; escape rung has no norm sideband (exact fp16 r)
    c = make_comp()
    assert c.rate_bits.tolist() == [0.0,
                                    16.0 + 8.0 * NSUB,
                                    16.0 + 8.0 * (2 + 1 + 1 + 1),
                                    16.0 + 16.0 * NSUB, 16.0 * D]
    assert c.n_rungs == PROFILES.shape[0] + 1


def test_monotone_distortion_in_profiles():
    # data scaled so m=1 subvectors stay inside the Voronoi region (at
    # production scale the fit's q999 beta plays this role for m=2; heavy
    # m=1 overload wrapping legitimately beats eviction only sometimes,
    # and the RDO — not this test — is what handles that regime)
    g = torch.Generator().manual_seed(4)
    x = torch.randn(1, 2 * PTOK, D, generator=g) * 0.3
    errs = []
    for rung in (0, 1, 3, 4):                       # evict, m=1, m=2, fp16
        c = make_comp(mode="uniform", uniform_rung=rung, beta=1.0)
        o = c.roundtrip(x.clone()).squeeze(0)
        errs.append(float((o[4:] - x.squeeze(0)[4:]).square().mean()))
    assert errs[0] >= errs[1] >= errs[2] >= errs[3]
    assert errs[3] < 1e-4                           # fp16 escape ~exact


def test_evict_rung_reconstructs_mu():
    c = make_comp(mode="uniform", uniform_rung=0)
    g = torch.Generator().manual_seed(5)
    x = torch.randn(1, 2 * PTOK, D, generator=g)
    out = c.roundtrip(x).squeeze(0)
    assert torch.allclose(out[4:], c.mu.expand_as(out[4:]), atol=1e-5)
    assert abs(c.bits_payload - 4 * 16 * D) < 1e-6  # sinks pay fp16 escape
    assert c.bits_side == 2 * HEADER_BITS


def test_sinks_forced_to_escape_rung():
    c = make_comp(mode="rdo", b_page=1.5)
    g = torch.Generator().manual_seed(6)
    x = torch.randn(1, 4 * PTOK, D, generator=g) * 3.0   # hot data
    out = c.roundtrip(x).squeeze(0)
    err = (out[:4] - x.squeeze(0)[:4]).norm() / x.squeeze(0)[:4].norm()
    assert float(err) < 1e-3                        # fp16-exact sinks
    assert c.rung_hist[-1] >= 4


def test_budget_and_overflow():
    c = make_comp(mode="rdo", b_page=1.5)
    g = torch.Generator().manual_seed(7)
    x = torch.randn(1, 6 * PTOK + 9, D, generator=g)
    c.roundtrip(x)
    assert c.pages_overflow == 0
    assert (c.bits_payload + c.bits_side) <= 1.5 * D * c.tokens_total + 1e-6
    assert sum(c.rung_hist) == c.tokens_total


def test_mixer_roundtrip_matches_unmixed_energy():
    # the mixer is orthogonal: distortion at matched profile must be on the
    # same scale mixed vs unmixed (allocation comparability)
    g = torch.Generator().manual_seed(8)
    x = torch.randn(1, 2 * PTOK, D, generator=g)
    e_plain = make_comp(mode="uniform", uniform_rung=3)
    e_mixed = make_comp(mode="uniform", uniform_rung=3,
                        mixer=build_hadamard(D))
    d_plain = float((e_plain.roundtrip(x.clone()).squeeze(0)[4:]
                     - x.squeeze(0)[4:]).square().mean())
    d_mixed = float((e_mixed.roundtrip(x.clone()).squeeze(0)[4:]
                     - x.squeeze(0)[4:]).square().mean())
    assert d_mixed < 3.0 * d_plain + 1e-6


# ---- family (f): OSCAR-emulation arm ------------------------------------

def make_oscar(width=2, group=D, clip_q=0.96):
    g = torch.Generator().manual_seed(SEED)
    h = build_hadamard(D)
    mu_q = torch.randn(D, generator=g)
    return OscarArmCompressor(h, h.t().contiguous(), torch.zeros(D), mu_q,
                              width, group, clip_q, b_page=8.0, ptok=PTOK)


def test_oscar_rate_charged_honestly():
    # windows off: pure bulk-rate contract (2.25 b/c at d=128, one group)
    c = make_oscar(width=2, group=D)
    c.sink_tokens = 0
    c.recent_tokens = 0
    g = torch.Generator().manual_seed(9)
    x = torch.randn(1, 2 * PTOK, D, generator=g)
    c.roundtrip(x)
    T = 2 * PTOK
    assert abs(c.bits_payload - T * (2 * D + SCALE_ZERO_BITS)) < 1e-6
    assert c.bits_side == 2 * HEADER_BITS
    assert abs(c.bits_payload / (T * D)
               - (2 + SCALE_ZERO_BITS / D)) < 1e-9


def test_oscar_windows_fp16_and_charged():
    # published (S0, W) protection: fp16 passthrough, 16 b/c charged
    c = make_oscar(width=2, group=D)
    c.sink_tokens = 64
    c.recent_tokens = 256
    g = torch.Generator().manual_seed(10)
    T = 6 * PTOK   # 384: 64 sink + 64 bulk + 256 recent
    x = torch.randn(1, T, D, generator=g)
    out = c.roundtrip(x).squeeze(0)
    r = x.squeeze(0) @ c.forward_map
    ref = (r.to(torch.float16).float() @ c.inverse_map)
    assert torch.allclose(out[:64], ref[:64], atol=1e-3)
    assert torch.allclose(out[-256:], ref[-256:], atol=1e-3)
    n_bulk = T - 320
    want = n_bulk * (2 * D + SCALE_ZERO_BITS) + 320 * 16 * D
    assert abs(c.bits_payload - want) < 1e-6


def test_oscar_constant_group_is_exact():
    c = make_oscar(width=2, group=D, clip_q=1.0)
    x = torch.full((1, PTOK, D), 0.75)
    out = c.roundtrip(x)
    # min == max => zero scale, reconstruction = fp16(zero-point)
    assert (out - x).abs().max() < 1e-3


def test_oscar_error_scales_with_width():
    g = torch.Generator().manual_seed(10)
    x = torch.randn(1, 2 * PTOK, D, generator=g)
    e2 = float((make_oscar(width=2).roundtrip(x.clone()) - x)
               .square().mean())
    e3 = float((make_oscar(width=3).roundtrip(x.clone()) - x)
               .square().mean())
    assert e2 >= e3


def test_oscar_unclipped_wide_grid_is_near_exact():
    # clip_q=1.0 isolates the grid: at 8 bits the only residual error is
    # fp16 scale/zero rounding
    g = torch.Generator().manual_seed(11)
    x = torch.randn(1, 2 * PTOK, D, generator=g)
    e8b = float((make_oscar(width=8, clip_q=1.0).roundtrip(x.clone()) - x)
                .square().mean())
    assert e8b < 1e-4
