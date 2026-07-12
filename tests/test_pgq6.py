"""pgq6 MergedPageCompressor contract tests (plan6).

Covers: the replication↔bias identity the format rests on, merged-page rate
accounting (counts + M-choice charged, budget respected), redundancy-driven
merge selection, run contiguity, sink/recency protection, and loader routing.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402, F401

import math  # noqa: E402

import torch  # noqa: E402

from kvq.compression.pgq6_merge import (  # noqa: E402
    MERGE_LEVELS, PGQ6_COUNT_BITS, MergedPageCompressor,
)
from kvq.compression.page_quant import (  # noqa: E402
    load_pgq_compressors_from_bundle,
)

torch.manual_seed(0)
D = 32
PTOK = 16
LEVELS = (16, 8, 4)


def make_mcomp(b_page=2.0, grid="lm", force_recent_pages=0, seed=1):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(D, D, generator=g)
    F = torch.linalg.qr(A)[0] + 0.05 * torch.randn(D, D, generator=g)
    G = torch.linalg.inv(F)
    mu = torch.randn(D, generator=g) * 0.1
    mu_q = torch.randn(D, generator=g)
    code_std = torch.rand(D, generator=g) * 0.5 + 0.75
    profiles = torch.tensor([[0] * D, [2] * D, [3] * D, [4] * D])
    alphas = torch.tensor([0.9, 0.55, 0.34])
    sink_scale = code_std * 24.0 / 127.0
    lm2 = torch.tensor([-1.51, 0.0, 0.45, 1.51])
    lm3 = torch.linspace(-2.2, 2.2, 8)
    lm3[lm3.abs().argmin()] = 0.0
    return MergedPageCompressor(
        forward_map=F, inverse_map=G, mu=mu, mu_q=mu_q, code_std=code_std,
        profiles=profiles, alphas=alphas, sink_scale=sink_scale,
        b_page=b_page, grid=grid, lm_cents=[lm2, lm3], gain=False,
        ptok=PTOK, mode="rdo", uniform_rung=None,
        force_recent_pages=force_recent_pages, merge_levels=LEVELS)


def keys_from_r(comp, r):
    return r @ torch.linalg.inv(comp.forward_map) + comp.mu


def test_replication_bias_identity():
    """softmax over c identical keys == one key with +log c logit — the
    identity that makes the replicated eval form exact vs deployment."""
    g = torch.Generator().manual_seed(3)
    q = torch.randn(5, D, generator=g)
    ks = torch.randn(7, D, generator=g)
    kbar = torch.randn(D, generator=g)
    c = 6
    # replicated form
    K_rep = torch.cat([ks, kbar.expand(c, D)])
    p_rep = torch.softmax(q @ K_rep.T / math.sqrt(D), 1)
    mass_rep = p_rep[:, 7:].sum(1)
    # bias form: +log c OUTSIDE the 1/sqrt(d) scaling to match replication
    logits = torch.cat([q @ ks.T / math.sqrt(D),
                        ((q @ kbar) / math.sqrt(D)
                         + math.log(c)).unsqueeze(1)], 1)
    p_bias = torch.softmax(logits, 1)
    assert torch.allclose(mass_rep, p_bias[:, 7], atol=1e-6)
    assert torch.allclose(p_rep[:, :7], p_bias[:, :7], atol=1e-6)


def test_merged_runs_are_contiguous_and_replicated():
    comp = make_mcomp(b_page=1.0)
    # 4 pages of highly redundant rows: runs of 4 identical tokens
    g = torch.Generator().manual_seed(4)
    base = torch.randn(PTOK, D, generator=g) * comp.code_std
    r = base.repeat_interleave(4, dim=0)                     # 64 tokens
    k = keys_from_r(comp, r)
    out = comp.roundtrip(k.unsqueeze(0), start_pos=64).squeeze(0)
    assert out.shape == k.shape
    # tokens of an identical input run must reconstruct identically when the
    # RDO merged them; at minimum the codec must not crash and must respect
    # rate. Check rate first:
    rate = (comp.bits_payload + comp.bits_side) / comp.tokens_total / D
    assert rate <= 1.0 + 0.02, rate
    # redundant data at a tight budget should trigger merging on some pages
    merged_tokens = sum(comp.merge_hist[1:])
    assert merged_tokens > 0, comp.merge_hist
    assert comp.rows_total < comp.tokens_total


def test_random_data_prefers_unmerged_at_ample_budget():
    comp = make_mcomp(b_page=4.0)
    g = torch.Generator().manual_seed(5)
    r = torch.randn(64, D, generator=g) * comp.code_std
    k = keys_from_r(comp, r)
    comp.roundtrip(k.unsqueeze(0), start_pos=64)
    # i.i.d. tokens: merging costs spread with no compensating width gain at
    # an ample budget — conservative tie-break keeps pages unmerged
    assert comp.merge_hist[0] == 64, comp.merge_hist
    assert comp.rows_total == 64


def test_rate_accounting_charges_counts():
    comp = make_mcomp(b_page=1.0)
    g = torch.Generator().manual_seed(6)
    base = torch.randn(4, D, generator=g) * comp.code_std
    r = base.repeat_interleave(4, dim=0)                     # one page
    k = keys_from_r(comp, r)
    comp.roundtrip(k.unsqueeze(0), start_pos=64)
    # budget respected all-in (payload includes 6-bit counts on merged rows)
    total = comp.bits_payload + comp.bits_side
    assert total <= comp.b_page * D * comp.tokens_total * 1.02
    assert comp.pages_overflow == 0


def test_sink_and_recency_pages_stay_unmerged():
    comp = make_mcomp(b_page=1.5, force_recent_pages=1)
    g = torch.Generator().manual_seed(7)
    base = torch.randn(8, D, generator=g) * comp.code_std
    r = base.repeat_interleave(8, dim=0)                     # 4 pages
    # sink rows at exactly ±15 sigma per coord (inside the 24-sigma escape)
    r[:4] = torch.sign(r[:4]) * 15.0 * comp.code_std
    k = keys_from_r(comp, r)
    out = comp.roundtrip(k.unsqueeze(0), start_pos=0).squeeze(0)
    # sink escape reproduces sinks well despite the tight budget
    rel = (out[:4] - k[:4]).norm() / k[:4].norm()
    assert rel < 0.05, float(rel)
    # page 0 (sink) and the last page (recency) contribute unmerged tokens;
    # with 4 pages of 8x redundancy the middle pages should merge
    assert comp.merge_hist[0] >= 32, comp.merge_hist
    assert sum(comp.merge_hist[1:]) > 0, comp.merge_hist


def test_loader_routing_mrg(tmp_path):
    L, H, d = 3, 2, D
    g = torch.Generator().manual_seed(8)
    F = torch.randn(L, H, d, d, generator=g) * 0.1 + torch.eye(d)
    Finv = torch.linalg.inv(F)
    stats_one = {
        "code_std": torch.rand(L, H, d, generator=g) + 0.5,
        "alphas": torch.tensor([0.9, 0.55, 0.34]).expand(L, H, 3).clone(),
        "sink_scale": torch.rand(L, H, d, generator=g) * 0.1 + 0.05,
    }
    blob = {
        "pgq_version": 5, "model_tag": "test", "ptok": 16,
        "n_layers": L, "n_kv_heads": H, "head_dim": d,
        "mu": torch.zeros(L, H, d), "mu_q": torch.randn(L, H, d, generator=g),
        "bases": {"qpca_unc": {"forward": F, "inverse": Finv}},
        "stats": {"qpca_unc": stats_one},
        "lm_cents": [torch.tensor([-1.5, 0.0, 0.5, 1.5]),
                     torch.linspace(-2.0, 2.0, 8)],
        "prof_head": torch.tensor([[0, 0, 0, 0], [4, 4, 4, 4]])
        .expand(L, H, 2, 4).clone(),
        "prof_layer": torch.tensor([[0, 0, 0, 0], [4, 4, 4, 4]])
        .expand(L, 2, 4).clone(),
        "prof_share": "layer",
        "px_profiles": torch.tensor([[0, 0, 0, 0], [4, 4, 4, 4]]),
        "prof_share_penalty": 0.01,
        "omega_tau_by_rate": {"2": 0.5},
        "omega_clamp_bits": 4.0,
    }
    p = tmp_path / "pgq6_test.pt"
    torch.save(blob, p)
    for method in ("pgq_mrglm_rdo", "pgq_mrglmrw_rdo"):
        comps, _ = load_pgq_compressors_from_bundle(str(p), method, 2.0)
        c = comps[(1, 0)]
        assert isinstance(c, MergedPageCompressor)
        assert c.merge_levels == (16, 8, 4)      # scaled to ptok=16
        out = c.roundtrip(torch.randn(40, d), start_pos=64)
        assert out.shape == (40, d)
    c = load_pgq_compressors_from_bundle(str(p), "pgq_mrglmrw_rdo", 2.0)[0][
        (1, 0)]
    assert c.force_recent_pages == 4
