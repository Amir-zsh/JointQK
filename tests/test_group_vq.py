"""group_vq port tests: parity vs the third_party snapshot, wrap behaviour,
method grammar, and bundle-loader contracts. CPU-only."""
import importlib.util
import math
from pathlib import Path

import pytest
import torch

from kvq.compression.group_vq import (
    GroupVQCompressor,
    OutlierProtectWrap,
    SinkRecentWrap,
    group_boundaries,
    load_vqg_compressors,
    parse_vqg_method,
    stratified_perm,
    waterfill_continuous,
    group_bit_alloc,
)

REPO = Path(__file__).resolve().parents[1]
SNAP = REPO / "third_party" / "samuel_vq"
QWEN_CB = SNAP / "codebooks" / "vqa_G4_strat_flat_fair_fp8.pt"


def _load_snapshot_module():
    spec = importlib.util.spec_from_file_location(
        "samuel_group_vq_codec", SNAP / "group_vq_codec.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_group_boundaries_g4_exact_rate():
    bounds = group_boundaries(128, 4, bpc=2)
    assert len(bounds) == 32
    assert bounds[0] == (0, 4, 8)
    assert bounds[-1] == (124, 128, 8)
    assert sum(b for _, _, b in bounds) / 128 == 2.0


def test_group_boundaries_remainder_group():
    bounds = group_boundaries(128, 6, bpc=2)
    assert bounds[-1] == (126, 128, 4)
    assert sum(b for _, _, b in bounds) / 128 == 2.0


def test_stratified_perm_spans_spectrum():
    perm = stratified_perm(128, 4)
    assert sorted(perm.tolist()) == list(range(128))
    # first group (positions 0..3) holds ranks {0, 32, 64, 96}: one per stratum
    assert perm[:4].tolist() == [0, 32, 64, 96]


def test_waterfill_monotone_and_budget():
    score = torch.tensor([16.0, 4.0, 1.0, 0.25, 1e-6])
    b = waterfill_continuous(score, total=6.0)
    assert torch.all(b[:-1] >= b[1:])          # more score -> more bits
    assert abs(float(b.sum()) - 6.0) < 1e-3
    gb = group_bit_alloc(score, [(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 4, 0), (4, 5, 0)],
                         avg_bits=1.2, max_k_bits=13)
    assert sum(gb) == 6


def test_parse_vqg_method_grammar():
    assert parse_vqg_method("pgq_vqg_flat") == (False, 0.0, "flat")
    assert parse_vqg_method("pgq_vqgb_flat") == (True, 0.0, "flat")
    assert parse_vqg_method("pgq_vqgbo05_flat") == (True, 0.05, "flat")
    assert parse_vqg_method("pgq_vqgo10_wf") == (False, 0.10, "wf")
    with pytest.raises(ValueError):
        parse_vqg_method("pgq_vqg_rdo")
    with pytest.raises(ValueError):
        parse_vqg_method("pgq_vqgb")


def _tiny_compressor(d=8, G=4, T=64, seed=0):
    gen = torch.Generator().manual_seed(seed)
    F = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))[0]
    bounds = group_boundaries(d, G, bpc=2)
    cbs = [torch.randn(1 << bits, e - s, generator=gen, dtype=torch.float32)
           for (s, e, bits) in bounds]
    mean = torch.randn(d, generator=gen, dtype=torch.float32)
    return GroupVQCompressor(F.float(), F.t().float(), mean, cbs, bounds)


def test_roundtrip_shape_dtype_and_determinism():
    c = _tiny_compressor()
    k = torch.randn(2, 64, 8, dtype=torch.float16)
    out1, out2 = c.roundtrip(k), c.roundtrip(k)
    assert out1.shape == k.shape and out1.dtype == k.dtype
    assert torch.equal(out1, out2)


def test_encode_decode_matches_roundtrip():
    c = _tiny_compressor()
    k = torch.randn(64, 8)
    idx = c.encode_idx(k)
    dec = c.decode_idx(idx, dtype=k.dtype)
    assert torch.allclose(dec, c.roundtrip(k), atol=1e-6)


def test_sink_recent_wrap_protects_band():
    c = _tiny_compressor()
    w = SinkRecentWrap(c, sink=4, recent=8)
    k = torch.randn(1, 64, 8)
    out = w.roundtrip(k)
    assert torch.equal(out[:, :4], k[:, :4])
    assert torch.equal(out[:, -8:], k[:, -8:])
    assert not torch.equal(out[:, 4:-8], k[:, 4:-8])
    # short sequence entirely inside the band passes through untouched
    ks = torch.randn(1, 10, 8)
    assert torch.equal(w.roundtrip(ks), ks)


def test_outlier_wrap_restores_worst_tokens_to_fp8():
    c = _tiny_compressor()
    k = torch.randn(1, 64, 8)
    k[0, 17] *= 40.0                             # far outside codebook coverage
    w = OutlierProtectWrap(c, frac=0.02)         # 1 of 64 tokens
    out = w.roundtrip(k)
    ref_fp8 = k[0, 17].to(torch.float8_e4m3fn).to(out.dtype)
    assert torch.equal(out[0, 17], ref_fp8)


def _tiny_bundle(L=3, H=2, d=8, G=4, alloc="flat"):
    gen = torch.Generator().manual_seed(1)
    bounds = group_boundaries(d, G, bpc=2)
    cbs = {(l, h): [torch.randn(1 << bits, e - s, generator=gen)
                    for (s, e, bits) in bounds]
           for l in range(L) for h in range(H)}
    return {
        "forward": torch.randn(L, H, d, d, generator=gen),
        "inverse": torch.randn(L, H, d, d, generator=gen),
        "mean": torch.randn(L, H, d, generator=gen),
        "codebooks": cbs, "bounds": bounds, "G": G,
        "grouping": "stratified", "allocation": alloc,
        "whiten": False, "pertoken_norm": False, "bits_per_coord": 2.0,
    }


def test_load_vqg_layer0_skip_and_wraps():
    blob = _tiny_bundle()
    comps, meta = load_vqg_compressors(blob, "<mem>", "pgq_vqgbo05_flat", 2.0)
    assert meta["n_layers"] == 3 and meta["n_kv_heads"] == 2
    assert (0, 0) not in comps and (1, 0) in comps and (2, 1) in comps
    top = comps[(1, 0)]
    assert isinstance(top, OutlierProtectWrap)
    assert isinstance(top.inner, SinkRecentWrap)
    assert top.inner.sink == 4 and top.inner.recent == 32


def test_load_vqg_validates_alloc_and_rate():
    blob = _tiny_bundle(alloc="flat")
    with pytest.raises(ValueError, match="allocation"):
        load_vqg_compressors(blob, "<mem>", "pgq_vqg_wf", 2.0)
    with pytest.raises(ValueError, match="b/coord"):
        load_vqg_compressors(blob, "<mem>", "pgq_vqg_flat", 1.5)


@pytest.mark.skipif(not QWEN_CB.exists(), reason="snapshot codebook absent")
def test_parity_vs_snapshot_codec_real_codebook():
    """Bit-exact parity: our port vs Samuel's snapshot codec, his real fp8
    Qwen codebook, several (layer, head) cells, fp16 inputs."""
    snap = _load_snapshot_module()
    payload = torch.load(QWEN_CB, map_location="cpu", weights_only=False)
    F, inv, mean = payload["forward"], payload["inverse"], payload["mean"]
    bounds, cbs = payload["bounds"], payload["codebooks"]
    ptn = bool(payload.get("pertoken_norm", False))
    gen = torch.Generator().manual_seed(7)
    for (l, h) in [(1, 0), (5, 3), (20, 7)]:
        ours = GroupVQCompressor(F[l, h], inv[l, h], mean[l, h],
                                 list(cbs[(l, h)]), bounds, pertoken_norm=ptn)
        his = snap.GroupVQCompressor(F[l, h], inv[l, h], mean[l, h],
                                     list(cbs[(l, h)]), bounds, pertoken_norm=ptn)
        k = (torch.randn(97, F.shape[-1], generator=gen) * 3).to(torch.float16)
        assert torch.equal(ours.roundtrip(k), his.roundtrip(k))
        # sane reconstruction, not identity
        rel = ((ours.roundtrip(k).float() - k.float()).pow(2).sum()
               / k.float().pow(2).sum()).item()
        assert 0.0 < rel < 1.0


@pytest.mark.skipif(not QWEN_CB.exists(), reason="snapshot codebook absent")
def test_qwen_codebook_shape_contract():
    payload = torch.load(QWEN_CB, map_location="cpu", weights_only=False)
    d = payload["forward"].shape[-1]
    assert d == 128 and payload["G"] == 4
    assert payload["allocation"] == "flat" and payload["grouping"] == "stratified"
    bounds = payload["bounds"]
    assert len(bounds) == 32
    cb0 = payload["codebooks"][(1, 0)]
    assert len(cb0) == 32 and cb0[0].shape == (256, 4)
    assert abs(payload["bits_per_coord"] - 2.0) < 1e-6
