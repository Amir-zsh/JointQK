"""Unit tests for kvq/presses/omega_page_press.py (plan3 Thrust A)."""
import math
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

from kvq.presses.omega_page_press import OmegaPagePress  # noqa: E402

L, H, D = 4, 2, 16
PS = 8          # small pages at test scale
SEED = 20260707


@pytest.fixture()
def stats_path(tmp_path):
    g = torch.Generator().manual_seed(SEED)
    p = tmp_path / "omega_stats.pt"
    torch.save({"mu_q": torch.randn(L, H, D, generator=g)}, p)
    return str(p)


def make_press(stats_path, mode="omega_max", ratio=0.5):
    return OmegaPagePress(compression_ratio=ratio,
                          omega_stats_path=stats_path,
                          score_mode=mode, page_size=PS)


def fake_module(layer_idx=2):
    return types.SimpleNamespace(layer_idx=layer_idx, head_dim=D)


def keys_of(t, generator):
    return torch.randn(1, H, t, D, generator=generator)


def test_scores_are_page_major_ranks(stats_path):
    press = make_press(stats_path)
    g = torch.Generator().manual_seed(1)
    k = keys_of(6 * PS, g)
    s = press.score(fake_module(), None, k, k.clone(), None, {})
    assert s.shape == (1, H, 6 * PS)
    # within a page, scores strictly descend by 1; across pages, by PS
    sp = s.reshape(1, H, 6, PS)
    assert torch.all(sp[..., :-1] - sp[..., 1:] == 1)
    assert set((-s).long().flatten().tolist()) == set(range(6 * PS))


def test_sink_and_recent_pages_rank_first(stats_path):
    press = make_press(stats_path)
    g = torch.Generator().manual_seed(2)
    n_pages = 6
    k = keys_of(n_pages * PS, g)
    s = press.score(fake_module(), None, k, k.clone(), None, {})
    top = (-s).argsort(-1)[..., :2 * PS]              # 2 forced pages
    pages = (top // PS)
    for h in range(H):
        assert set(pages[0, h].tolist()) == {0, n_pages - 1}


def test_page_ranking_matches_manual_omega_max(stats_path):
    press = make_press(stats_path)
    g = torch.Generator().manual_seed(3)
    n_pages = 8
    k = keys_of(n_pages * PS, g)
    mod = fake_module()
    s = press.score(mod, None, k, k.clone(), None, {})
    mu = torch.load(stats_path, weights_only=False)["mu_q"][mod.layer_idx]
    m = torch.einsum("bhtd,hd->bht", k, mu) / math.sqrt(D)
    manual = m.reshape(1, H, n_pages, PS).max(-1).values
    manual[..., 0] = manual[..., -1] = torch.inf
    order_manual = manual.argsort(-1, descending=True)
    order_press = (-s.reshape(1, H, n_pages, PS)[..., 0]).argsort(-1)
    # forced pages tie under inf; compare the free pages' relative order
    for h in range(H):
        free_m = [p for p in order_manual[0, h].tolist()
                  if p not in (0, n_pages - 1)]
        free_p = [p for p in order_press[0, h].tolist()
                  if p not in (0, n_pages - 1)]
        assert free_m == free_p


def test_compress_keeps_whole_pages(stats_path):
    press = make_press(stats_path, ratio=0.5)
    g = torch.Generator().manual_seed(4)
    n_pages = 8
    k = keys_of(n_pages * PS, g)
    v = k.clone()
    mod = fake_module()
    s = press.score(mod, None, k, v, None, {})
    n_kept = int(k.shape[2] * 0.5)
    idx = s.topk(n_kept, dim=-1).indices
    # kept tokens tile whole pages: 4 pages x PS tokens each
    for h in range(H):
        kept_pages = torch.unique(idx[0, h] // PS)
        assert kept_pages.numel() == n_pages // 2
        assert idx[0, h].numel() == kept_pages.numel() * PS
    ck, cv = press.compress(mod, None, k, v, None, {})
    assert ck.shape == (1, H, n_kept, D)
    assert torch.equal(ck, k.gather(
        2, idx.unsqueeze(-1).expand(-1, -1, -1, D)))
    assert torch.equal(cv, v.gather(
        2, idx.unsqueeze(-1).expand(-1, -1, -1, D)))


def test_partial_last_page_is_recent_protected(stats_path):
    press = make_press(stats_path)
    g = torch.Generator().manual_seed(5)
    t = 5 * PS + 3
    k = keys_of(t, g)
    s = press.score(fake_module(), None, k, k.clone(), None, {})
    assert s.shape[-1] == t
    top = (-s).argsort(-1)[..., :PS + 3]
    for h in range(H):
        pages = set((top[0, h] // PS).tolist())
        assert pages == {0, 5}                        # sink + partial recent


def test_all_score_modes_run(stats_path):
    g = torch.Generator().manual_seed(6)
    k = keys_of(8 * PS, g)
    outs = {}
    for mode in ("omega_max", "omega_mean", "quest_mu", "random_page"):
        press = make_press(stats_path, mode=mode)
        outs[mode] = press.score(fake_module(), None, k, k.clone(),
                                 None, {})
        assert outs[mode].shape == (1, H, 8 * PS)
    # different modes generally produce different page orders
    assert not torch.equal(outs["omega_max"], outs["random_page"])


def test_quest_mu_matches_manual_boxes(stats_path):
    press = make_press(stats_path, mode="quest_mu")
    g = torch.Generator().manual_seed(7)
    n_pages = 6
    k = keys_of(n_pages * PS, g)
    mod = fake_module()
    raw = press._page_scores(k, mod.layer_idx)
    mu = torch.load(stats_path, weights_only=False)["mu_q"][mod.layer_idx]
    kp = k.reshape(1, H, n_pages, PS, D)
    manual = torch.maximum(mu.view(1, H, 1, D) * kp.min(3).values,
                           mu.view(1, H, 1, D) * kp.max(3).values) \
        .sum(-1) / math.sqrt(D)
    assert torch.allclose(raw, manual, atol=1e-5)
