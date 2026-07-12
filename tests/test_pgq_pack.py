"""pgq7 K1: packed-format golden-vector tests (plan7 §4).

Pins: (1) the emit kwarg is behavior-neutral, (2) the little-endian bitstream
spec (the golden layout the K2 Triton unpack must match), (3) pack -> unpack
-> dequant is BIT-IDENTICAL to the compressor's reconstruction on synthetic
data and real Qwen selection rows, (4) segment/stride accounting matches the
codec's honest rates and the per-page contiguity invariant (plan7 R2).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402, F401

import pytest  # noqa: E402
import torch  # noqa: E402

from kvq.compression.page_quant import (  # noqa: E402
    load_pgq_compressors_from_bundle,
)
from kvq.kernels.pgq_pack import (  # noqa: E402
    _pack_bits, _unpack_bits, block_widths, dequant_codes, pack_sequence,
    payload_bytes, unpack_sequence,
)
from test_pgq4 import D, gauss_keys, make_comp  # noqa: E402

torch.manual_seed(0)

QWEN_BUNDLE = (REPO / "artifacts/page_quant2/"
               "pgq5_bundle__qpca_unc__qwen3_8b_compact8train12.pt")
QWEN_RAW = (REPO / "artifacts/calibration/"
            "longbench_compact8_qkv_qwen3_8b/01_raw")

# water-filled monotone style: width changes only at 16-coord blocks
MIXED_PROFILES = torch.tensor(
    [[0] * 32, [2] * 16 + [0] * 16, [3] * 16 + [2] * 16, [4] * 16 + [3] * 16])


def roundtrip_emit(comp, k):
    em = {}
    out = comp.roundtrip(k.clone(), emit=em)
    return out, em


def pack_unpack(comp, em, block):
    packed = pack_sequence(em["codes"], em["assign"], comp.profiles,
                           comp.ptok, nsink=em["nsink"],
                           sink_codes=em["sink_codes"], block=block)
    codes, assign, sink = unpack_sequence(packed)
    return packed, codes, assign, sink


def test_emit_is_behavior_neutral():
    for grid in ("uniform", "lm"):
        comp = make_comp(grid=grid, b_page=2.0)
        k = gauss_keys(100, comp)
        out_plain = comp.roundtrip(k.clone())
        out_emit, em = roundtrip_emit(comp, k)
        assert torch.equal(out_plain, out_emit)
        assert em["assign"].shape == (100,) and em["codes"].shape == (100, D)
        assert em["nsink"] == 4 and em["sink_codes"].shape == (4, D)


def test_bitstream_golden_layout():
    # width 2: c0|c1<<2|c2<<4|c3<<6 per byte (quarter-packed like OSCAR's
    # crumbs but sequential, not dim-interleaved)
    codes = torch.zeros(1, 32, dtype=torch.uint8)
    codes[0, :4] = torch.tensor([1, 2, 3, 0], dtype=torch.uint8)
    packed = _pack_bits(codes, 2)
    assert packed.shape == (1, 8)
    assert int(packed[0, 0]) == 1 | (2 << 2) | (3 << 4)
    assert packed[0, 1:].eq(0).all()
    # width 3: bit j of code i at bit position i*3+j
    codes3 = torch.zeros(1, 32, dtype=torch.uint8)
    codes3[0, :2] = torch.tensor([5, 6], dtype=torch.uint8)
    packed3 = _pack_bits(codes3, 3)
    assert packed3.shape == (1, 12)
    assert int(packed3[0, 0]) == 0b110101                 # 5 then 6, LE
    for w in (2, 3, 4, 6, 8):
        c = torch.randint(0, 1 << w, (7, 32), dtype=torch.uint8)
        assert torch.equal(_unpack_bits(_pack_bits(c, w), w, 32), c)


@pytest.mark.parametrize("grid", ["uniform", "lm"])
@pytest.mark.parametrize("rw", [0, 4])
def test_pack_roundtrip_bit_identity(grid, rw):
    comp = make_comp(grid=grid, b_page=2.0, profiles=MIXED_PROFILES,
                     force_recent_pages=rw)
    k = gauss_keys(100, comp)
    out, em = roundtrip_emit(comp, k)
    packed, codes, assign, sink = pack_unpack(comp, em, block=16)
    assert torch.equal(codes, em["codes"])
    assert torch.equal(assign, em["assign"])
    assert torch.equal(sink, em["sink_codes"])
    r_hat = dequant_codes(codes, assign, comp, nsink=em["nsink"],
                          sink_codes=sink)
    assert torch.equal(r_hat, em["r_hat"])
    # and through the inverse map: exactly the compressor's output
    out_ref = (r_hat @ comp.inverse_map + comp.mu).to(out.dtype)
    assert torch.equal(out_ref, out.reshape(100, D))


def test_uniform_mode_bit_identity():
    comp = make_comp(mode="uniform", uniform_rung=2, profiles=MIXED_PROFILES)
    k = gauss_keys(50, comp)
    out, em = roundtrip_emit(comp, k)
    packed, codes, assign, sink = pack_unpack(comp, em, block=16)
    r_hat = dequant_codes(codes, assign, comp, nsink=em["nsink"],
                          sink_codes=sink)
    assert torch.equal(r_hat, em["r_hat"])


def test_segments_and_rates():
    comp = make_comp(grid="lm", b_page=2.0, profiles=MIXED_PROFILES)
    k = gauss_keys(100, comp)
    _, em = roundtrip_emit(comp, k)
    packed, *_ = pack_unpack(comp, em, block=16)
    # strides carry exactly the codec's per-rung payload bits (gain off)
    assert torch.equal(packed["strides"] * 8,
                       comp.rung_rate.to(packed["strides"].dtype))
    for ri, toks in enumerate(packed["rung_tokens"]):
        assert packed["payload"][ri].shape == (toks.numel(),
                                               int(packed["strides"][ri]))
        # per-page contiguity: within a page, indices ascend and same-rung
        # tokens are consecutive slices of the segment list
        pages = toks // comp.ptok
        assert (pages.diff() >= 0).all()
        assert (toks.diff()[pages.diff() == 0] > 0).all()
    nonsink = int((torch.arange(100) >= em["nsink"]).sum())
    expect = (int(packed["strides"][em["assign"][em["nsink"]:]].sum())
              + em["nsink"] * D)
    assert payload_bytes(packed) == expect
    assert nonsink == 100 - em["nsink"]
    # header: 2 bits/token for the 4-rung ladder
    assert packed["id_bits"] == comp.id_bits
    assert packed["rung_ids"].numel() == (100 * packed["id_bits"] + 7) // 8


def test_block_constancy_enforced():
    prof = MIXED_PROFILES.clone()
    prof[3, 7] = 2                        # width change inside a block
    with pytest.raises(ValueError):
        block_widths(prof, 16)


@pytest.mark.skipif(not (QWEN_BUNDLE.exists() and QWEN_RAW.exists()),
                    reason="Qwen pgq5 artifacts not present on this host")
def test_real_qwen_selection_rows():
    from pipelines.ec import fit_ec_bundle as feb
    feb.set_model_tag("qwen3_8b")
    roles = json.loads(Path(feb.ROLES).read_text())
    comps, meta = load_pgq_compressors_from_bundle(
        str(QWEN_BUNDLE), "pgq_proflmrw_rdo", 2.0)
    pool = feb.RawPool(roles["selection"])
    for (cfg, row), (l, h) in zip(pool.rows, [(1, 0), (8, 3), (20, 5),
                                              (35, 7)]):
        art = pool.art((cfg, row))
        T = int(art["prompt_length"])
        k = art["k_post"][l, h, :T]
        comp = comps[(l, h)]
        out, em = roundtrip_emit(comp, k)
        packed, codes, assign, sink = pack_unpack(comp, em, block=32)
        assert torch.equal(codes, em["codes"]), (cfg, row, l, h)
        assert torch.equal(assign, em["assign"])
        r_hat = dequant_codes(codes, assign, comp, nsink=em["nsink"],
                              sink_codes=sink)
        assert torch.equal(r_hat, em["r_hat"]), (cfg, row, l, h)
        assert payload_bytes(packed) > 0
