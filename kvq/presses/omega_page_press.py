"""Page-granular eviction press (plan3 Thrust A).

ScorerPress that scores whole 64-token pages with a calibration-derived
importance signal and evicts the lowest-scoring pages per (layer, kv_head).
Page scores (frozen `score_mode` chosen by the A1 probe,
pipelines/page_quant/probe_page_selection.py):

  omega_max / omega_mean  max / mean over the page of m = k.mu_q/sqrt(d)
  quest_mu                Quest-style per-page coord min/max boxes scored
                          with mu_q
  random_page             seeded random page ranking (ablation control)

The first n_sink_pages and last n_recent_pages are always kept. Token scores
are emitted in page-major rank order (rank-based, not raw-score-based), so
the base ScorerPress.compress() topk keeps whole pages deterministically —
per-head, via its existing gather; no compress() override needed.

mu_q comes from `omega_stats_path` (exported by the A1 probe:
artifacts/page_quant2/omega_stats_llama31_8b.pt, mu_q (L, H_kv, d)).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress

SCORE_MODES = ("omega_max", "omega_mean", "quest_mu", "random_page")


@dataclass
class OmegaPagePress(ScorerPress):
    compression_ratio: float = 0.0
    omega_stats_path: str = ""
    score_mode: str = "omega_max"
    page_size: int = 64
    n_sink_pages: int = 1
    n_recent_pages: int = 1
    seed: int = 20260708
    _mu_q: torch.Tensor | None = field(default=None, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if self.score_mode not in SCORE_MODES:
            raise ValueError(f"score_mode {self.score_mode!r} not in "
                             f"{SCORE_MODES}")

    def _mu(self, layer: int, device) -> torch.Tensor:
        if self._mu_q is None:
            if not self.omega_stats_path:
                raise ValueError("OmegaPagePress requires omega_stats_path")
            blob = torch.load(self.omega_stats_path, map_location="cpu",
                              weights_only=False)
            self._mu_q = blob["mu_q"].float()
        return self._mu_q[layer].to(device)

    def _page_scores(self, keys: torch.Tensor, layer: int) -> torch.Tensor:
        """keys (B, H, T, d) -> (B, H, n_pages) raw page scores."""
        b, h, t, d = keys.shape
        ps = self.page_size
        n_pages = (t + ps - 1) // ps
        pad = n_pages * ps - t
        if pad:
            keys = torch.nn.functional.pad(keys, (0, 0, 0, pad),
                                           value=0.0)
        kp = keys.reshape(b, h, n_pages, ps, d)
        if self.score_mode == "random_page":
            g = torch.Generator().manual_seed(self.seed + layer)
            return (torch.rand(n_pages, generator=g).to(keys.device)
                    .expand(b, h, n_pages).clone())
        mu = self._mu(layer, keys.device)              # (H, d)
        if self.score_mode == "quest_mu":
            kmin = kp.min(3).values                    # (B, H, P, d)
            kmax = kp.max(3).values
            mu_ = mu.view(1, h, 1, d)
            return torch.maximum(mu_ * kmin, mu_ * kmax).sum(-1) \
                / math.sqrt(d)
        m = torch.einsum("bhpsd,hd->bhps", kp, mu) / math.sqrt(d)
        if pad:                                        # padded slots inert
            m[..., -1, ps - pad:] = -torch.inf
        if self.score_mode == "omega_max":
            return m.max(-1).values
        m = m.masked_fill(torch.isinf(m), 0.0)
        denom = torch.full_like(m[..., 0], float(ps))
        if pad:
            denom[..., -1] = ps - pad
        return m.sum(-1) / denom

    def score(self, module: nn.Module, hidden_states: torch.Tensor,
              keys: torch.Tensor, values: torch.Tensor,
              attentions: torch.Tensor, kwargs) -> torch.Tensor:
        b, h, t, d = keys.shape
        ps = self.page_size
        n_pages = (t + ps - 1) // ps
        raw = self._page_scores(keys, module.layer_idx)
        raw[..., :self.n_sink_pages] = torch.inf
        if self.n_recent_pages:
            raw[..., n_pages - self.n_recent_pages:] = torch.inf
        # rank-based token scores: page-major order makes the base topk
        # keep whole pages regardless of raw-score scale
        rank = raw.argsort(-1, descending=True).argsort(-1).float()
        pos = torch.arange(n_pages * ps, device=keys.device)
        page_of = (pos // ps).view(1, 1, -1).expand(b, h, -1)
        tok = -(rank.gather(2, page_of) * ps
                + (pos % ps).view(1, 1, -1))
        return tok[..., :t]
