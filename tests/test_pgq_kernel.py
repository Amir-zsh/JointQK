"""pgq7 K2 v0 kernel parity tests: Triton stage-1 logits vs golden refs.

Real geometry (d=128, ptok=64, the shipping LM ladder), synthetic keys plus
a real-bundle case. Gate: max rel error <= 1e-2 wrt the fp32 reference
built from the emitted quantized rows (plan7 K2)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402, F401

import pytest  # noqa: E402
import torch  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")

LLAMA_BUNDLE = (REPO / "artifacts/page_quant2/pgq8_bundle__llama31_8b.pt")


def real_geometry_comp(dct=True, seed=1, b_page=2.0, rw=0):
    """d=128, ptok=64 compressor with LM grids and monotone profiles —
    kernel-real geometry, synthetic stats."""
    from kvq.compression.pgq4_folded import FoldedScalarPagedCompressor
    from kvq.compression.pgq8_dct import PageDCTCompressor, dct_matrix
    from kvq.compression.per_coord import unit_gaussian_centroids

    g = torch.Generator().manual_seed(seed)
    d, ptok = 128, 64
    A = torch.randn(d, d, generator=g)
    F = torch.linalg.qr(A)[0] + 0.03 * torch.randn(d, d, generator=g)
    G = torch.linalg.inv(F)
    mu = torch.randn(d, generator=g) * 0.1
    code_std = torch.rand(d, generator=g) * 0.6 + 0.7
    profiles = torch.tensor([
        [0] * 128,
        [2] * 32 + [0] * 96,
        [2] * 64 + [2] * 32 + [0] * 32,
        [3] * 32 + [2] * 64 + [0] * 32,
        [3] * 64 + [2] * 64,
        [4] * 32 + [3] * 64 + [2] * 32,
        [4] * 64 + [3] * 64,
        [6] * 32 + [4] * 32 + [4] * 32 + [3] * 32,
    ])
    alphas = torch.tensor([0.9, 0.6, 0.42, 0.28])
    cents = []
    for w in (2, 3):
        c = unit_gaussian_centroids(w).sort().values
        c[c.abs().argmin()] = 0.0
        cents.append(c)
    c4 = unit_gaussian_centroids(4).sort().values
    c4[c4.abs().argmin()] = 0.0
    cents.append(c4)
    kw = dict(forward_map=F, inverse_map=G, mu=mu,
              mu_q=torch.randn(d, generator=g), code_std=code_std,
              profiles=profiles, alphas=alphas,
              sink_scale=code_std * 24.0 / 127.0, b_page=b_page, grid="lm",
              lm_cents=cents, ptok=ptok, mode="rdo",
              force_recent_pages=rw, width_ladder=(0, 2, 3, 4, 6))
    if not dct:
        return FoldedScalarPagedCompressor(**kw)
    dstd = code_std.unsqueeze(0).repeat(ptok, 1)
    dstd[0] *= 8.0                                    # DC rows are hot
    dstd[1:4] *= 3.0
    return PageDCTCompressor(dct_std=dstd, **kw)


def corr_keys(comp, T, seed=3):
    g = torch.Generator().manual_seed(seed)
    steps = torch.randn(T, 128, generator=g) * 0.35
    r = steps.cumsum(0)
    r = (r - r.mean(0)) / r.std(0).clamp_min(1e-6) * comp.code_std
    return r @ torch.linalg.inv(comp.forward_map) + comp.mu


def run_case(comp, T, Hg=4, seed=5):
    from kvq.kernels.golden import build_golden
    from kvq.kernels.pgq_pack import pack_sequence
    from kvq.kernels.pgq_decode_attn import page_logits

    k = corr_keys(comp, T)
    em = {}
    comp.roundtrip(k.clone(), emit=em)
    packed = pack_sequence(em["codes"], em["assign"], comp.profiles,
                           comp.ptok, nsink=em["nsink"],
                           sink_codes=em["sink_codes"])
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(Hg, 128, generator=g)
    gold = build_golden(comp, em, packed, q, device="cuda")
    z = page_logits(gold)
    ref = gold["z_ref"]
    m = ~torch.isnan(ref)
    assert int(m.sum()) > 0
    scale = ref[m].abs().max().clamp_min(1e-6)
    err = (z[m] - ref[m]).abs().max() / scale
    return float(err), gold


@cuda
def test_kernel_parity_dct_pages():
    comp = real_geometry_comp(dct=True, b_page=1.5)
    err, gold = run_case(comp, T=64 * 12)
    assert bool((gold["page_kind"] == 2).any())
    assert err < 1e-2, err


@cuda
def test_kernel_parity_identity_pages():
    comp = real_geometry_comp(dct=False, b_page=1.5)
    err, gold = run_case(comp, T=64 * 12)
    assert bool((gold["page_kind"] == 1).all())
    assert err < 1e-2, err


@cuda
def test_kernel_parity_mixed_rw():
    comp = real_geometry_comp(dct=True, b_page=2.0, rw=4)
    err, gold = run_case(comp, T=64 * 12 + 17)      # partial tail excluded
    kinds = gold["page_kind"]
    assert bool((kinds == 1).any()) and bool((kinds == 2).any())
    assert err < 1e-2, err


def run_attn_case(comp, T, Hg=4, seed=5, pps=8):
    from kvq.kernels.golden import build_golden
    from kvq.kernels.pgq_pack import pack_sequence
    from kvq.kernels.pgq_decode_attn import page_attention

    k = corr_keys(comp, T)
    em = {}
    comp.roundtrip(k.clone(), emit=em)
    packed = pack_sequence(em["codes"], em["assign"], comp.profiles,
                           comp.ptok, nsink=em["nsink"],
                           sink_codes=em["sink_codes"])
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(Hg, 128, generator=g)
    v = torch.randn(T, 128, generator=g)
    gold = build_golden(comp, em, packed, q, device="cuda", v=v)
    o = page_attention(gold, pages_per_split=pps)
    ref = gold["o_ref"]
    return float((o - ref).norm() / ref.norm().clamp_min(1e-9))


@cuda
def test_attention_parity_dct():
    comp = real_geometry_comp(dct=True, b_page=2.0, rw=4)
    err = run_attn_case(comp, T=64 * 13 + 9)         # sink+rw+partial tiers
    assert err < 1e-2, err


@cuda
def test_attention_parity_identity():
    comp = real_geometry_comp(dct=False, b_page=1.5)
    err = run_attn_case(comp, T=64 * 9)
    assert err < 1e-2, err


@cuda
def test_attention_split_invariance():
    comp = real_geometry_comp(dct=True, b_page=2.0)
    e1 = run_attn_case(comp, T=64 * 13, pps=3)
    e2 = run_attn_case(comp, T=64 * 13, pps=64)
    assert e1 < 1e-2 and e2 < 1e-2, (e1, e2)


def build_multi(H=2, T=64 * 9, dct=True, b_page=2.0, rw=0, seed0=11,
                segmented=False, v_int2=False):
    from kvq.kernels.golden import (add_v_int2, build_golden,
                                    quantize_v_int2, segment_layout,
                                    stack_heads)
    from kvq.kernels.pgq_pack import pack_sequence
    golds, packeds = [], []
    for h in range(H):
        comp = real_geometry_comp(dct=dct, seed=seed0 + h, b_page=b_page,
                                  rw=rw)
        k = corr_keys(comp, T, seed=seed0 + 50 + h)
        em = {}
        comp.roundtrip(k.clone(), emit=em)
        packed = pack_sequence(em["codes"], em["assign"], comp.profiles,
                               comp.ptok, nsink=em["nsink"],
                               sink_codes=em["sink_codes"])
        packeds.append(packed)
        g = torch.Generator().manual_seed(seed0 + 100 + h)
        q = torch.randn(4, 128, generator=g)
        v = torch.randn(T, 128, generator=g)
        if v_int2:
            v = quantize_v_int2(v)[3]        # goldens built on v_hat
        golds.append(build_golden(comp, em, packed, q, device="cuda", v=v))
    gm = stack_heads(golds)
    if segmented:
        gm = segment_layout(gm, packeds)
    if v_int2:
        gm = add_v_int2(gm)
    return gm


@cuda
def test_attention_v2_planar_phase_a():
    from kvq.kernels.pgq_decode_attn import page_attention_v2
    gm = build_multi(H=2, T=64 * 9 + 21, dct=True, rw=4, segmented=True)
    o = page_attention_v2(gm, phase_a="pl")
    err = float((o - gm["o_ref"]).norm() / gm["o_ref"].norm())
    assert err < 1e-2, err


@cuda
def test_attention_v2_two_page_tiles():
    from kvq.kernels.pgq_decode_attn import page_attention_v2
    gm = build_multi(H=2, T=64 * 9 + 21, dct=True, rw=4, segmented=True)
    for pps in (3, 16):                      # odd range exercises the tail
        o = page_attention_v2(gm, pps, phase_b="kernel2")
        err = float((o - gm["o_ref"]).norm() / gm["o_ref"].norm())
        assert err < 1e-2, (pps, err)


@cuda
def test_attention_v2_two_page_int2():
    from kvq.kernels.pgq_decode_attn import page_attention_v2
    gm = build_multi(H=2, T=64 * 9 + 21, dct=True, rw=4, segmented=True,
                     v_int2=True)
    o = page_attention_v2(gm, 3, phase_b="kernel2", v_int2=True)
    err = float((o - gm["o_ref"]).norm() / gm["o_ref"].norm())
    assert err < 1e-2, err


@cuda
def test_attention_v2_torch_phase_b():
    from kvq.kernels.pgq_decode_attn import page_attention_v2
    gm = build_multi(H=2, T=64 * 9 + 21, dct=True, rw=4, segmented=True)
    o = page_attention_v2(gm, phase_b="torch")
    err = float((o - gm["o_ref"]).norm() / gm["o_ref"].norm())
    assert err < 1e-2, err


@cuda
def test_attention_v2_int2_values():
    from kvq.kernels.pgq_decode_attn import page_attention_v2
    gm = build_multi(H=2, T=64 * 9 + 21, dct=True, rw=4, segmented=True,
                     v_int2=True)
    o = page_attention_v2(gm, v_int2=True)
    err = float((o - gm["o_ref"]).norm() / gm["o_ref"].norm())
    assert err < 1e-2, err
    o16 = page_attention_v2(gm, v_int2=False)     # fp16 tier on same v_hat
    err16 = float((o - o16).norm() / o16.norm())
    assert err16 < 5e-3, err16


@cuda
def test_attention_v2_parity_segmented():
    from kvq.kernels.pgq_decode_attn import page_attention_v2
    gm = build_multi(H=2, T=64 * 9 + 21, dct=True, rw=4, segmented=True)
    o = page_attention_v2(gm)
    err = float((o - gm["o_ref"]).norm() / gm["o_ref"].norm())
    assert err < 1e-2, err


@cuda
def test_attention_v2_identity_pages():
    from kvq.kernels.pgq_decode_attn import page_attention_v2
    gm = build_multi(H=1, T=64 * 8, dct=False, b_page=1.5, segmented=True)
    o = page_attention_v2(gm, pages_per_split=3)
    err = float((o - gm["o_ref"]).norm() / gm["o_ref"].norm())
    assert err < 1e-2, err


@cuda
def test_attention_v1_parity_multihead():
    from kvq.kernels.pgq_decode_attn import page_attention_v1
    gm = build_multi(H=2, T=64 * 9 + 21, dct=True, rw=4)
    o = page_attention_v1(gm)
    err = float((o - gm["o_ref"]).norm() / gm["o_ref"].norm())
    assert err < 1e-2, err


@cuda
def test_attention_v1_matches_v0():
    from kvq.kernels.pgq_decode_attn import page_attention_v1
    gm = build_multi(H=1, T=64 * 8, dct=True)
    o1 = page_attention_v1(gm, pages_per_split=3)
    o2 = page_attention_v1(gm, pages_per_split=64)
    ref = gm["o_ref"]
    for o in (o1, o2):
        err = float((o - ref).norm() / ref.norm())
        assert err < 1e-2, err


@cuda
@pytest.mark.skipif(not LLAMA_BUNDLE.exists(), reason="bundle not on host")
def test_kernel_parity_real_bundle_cell():
    from kvq.compression.page_quant import load_pgq_compressors_from_bundle
    comps, _ = load_pgq_compressors_from_bundle(
        str(LLAMA_BUNDLE), "pgq_dctlmrw_rdo", 2.0)
    comp = comps[(8, 3)]
    err, gold = run_case(comp, T=64 * 20)
    assert err < 1e-2, err
