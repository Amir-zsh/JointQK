"""Snap-parity tests for kvq.compression.ec_roundtrip.

The deployed SnappedDeadzoneECCompressor must map deadzone indices onto the
frozen alphabet exactly the way the rANS codec's symbol mapper does
(entropy_coding coded_bits_eval / kvq_codec._CoordModel.snap: numpy
searchsorted side='left', nearest value, tie -> left).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from kvq.compression.ec_roundtrip import (
    SnappedDeadzoneECCompressor, dz_dequant, dz_round,
)


def reference_snap(vals: np.ndarray, col: np.ndarray) -> np.ndarray:
    """Verbatim symbol mapping from entropy_coding coded_bits_eval."""
    pos = np.searchsorted(vals, col)
    pos = np.clip(pos, 0, vals.size - 1)
    left = np.clip(pos - 1, 0, vals.size - 1)
    choose_left = np.abs(vals[left] - col) <= np.abs(vals[pos] - col)
    sym = np.where(choose_left, left, pos)
    return vals[sym]


def make_comp(d, supports, delta=None, dz=0.375):
    vmax = max(len(s) for s in supports)
    sv = torch.full((d, vmax), float("inf"))
    sl = torch.zeros(d, dtype=torch.long)
    for j, s in enumerate(supports):
        sv[j, : len(s)] = torch.tensor(s, dtype=torch.float32)
        sl[j] = len(s)
    return SnappedDeadzoneECCompressor(
        forward_map=torch.eye(d), inverse_map=torch.eye(d),
        mu=torch.zeros(d), delta=delta if delta is not None else torch.ones(d),
        dz=dz, support_vals=sv, support_lens=sl)


def test_snap_matches_reference_on_random_alphabets():
    rng = np.random.default_rng(0)
    d, n = 16, 5000
    supports = []
    for _ in range(d):
        size = int(rng.integers(2, 40))
        vals = np.unique(rng.integers(-50, 50, size=size))
        supports.append(vals.tolist())
    comp = make_comp(d, supports)
    idx = torch.from_numpy(rng.integers(-80, 80, size=(n, d)).astype(np.float32))
    got = comp.snap_indices(idx)
    for j in range(d):
        want = reference_snap(np.asarray(supports[j], dtype=np.int64),
                              idx[:, j].numpy().astype(np.int64))
        np.testing.assert_array_equal(got[:, j].numpy().astype(np.int64), want)


def test_constant_coordinate_emits_alphabet_value():
    comp = make_comp(2, [[3], [-1, 0, 2]])
    idx = torch.tensor([[10.0, 10.0], [-7.0, -7.0], [0.0, 1.0]])
    got = comp.snap_indices(idx)
    assert torch.all(got[:, 0] == 3.0)
    # 1 is equidistant from 0 and 2 -> tie goes left -> 0.
    np.testing.assert_array_equal(got[:, 1].numpy(), [2.0, -1.0, 0.0])


def test_tie_goes_left():
    comp = make_comp(1, [[0, 2]])
    # idx=1 is equidistant from 0 and 2 -> reference picks left (0).
    got = comp.snap_indices(torch.tensor([[1.0]]))
    assert got.item() == 0.0


def test_roundtrip_in_support_is_pure_deadzone():
    """When every index is already in the alphabet, the roundtrip equals the
    raw deadzone quantizer (UniformECRoundtrip behaviour)."""
    torch.manual_seed(1)
    d, n, dz = 8, 1000, 0.375
    delta = torch.rand(d) * 0.5 + 0.1
    x = torch.randn(n, d)
    idx = dz_round(x, delta.unsqueeze(0), dz)
    supports = [torch.unique(idx[:, j]).tolist() for j in range(d)]
    comp = make_comp(d, supports, delta=delta, dz=dz)
    want = dz_dequant(idx, delta.unsqueeze(0), dz)
    got = comp.roundtrip(x)
    assert torch.allclose(got, want, atol=1e-5)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_cpu_gpu_agree(device):
    rng = np.random.default_rng(2)
    d, n = 8, 2000
    supports = [np.unique(rng.integers(-30, 30, size=12)).tolist() for _ in range(d)]
    comp = make_comp(d, supports)
    x = torch.randn(n, d)
    ref = comp.roundtrip(x.clone())
    comp.to(device)
    got = comp.roundtrip(x.to(device)).cpu()
    assert torch.allclose(ref, got, atol=1e-5)
