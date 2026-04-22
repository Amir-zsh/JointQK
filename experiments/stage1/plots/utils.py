from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import torch


def choose_representative_heads(score_matrix: torch.Tensor, sample_heads: int) -> list[tuple[int, int]]:
    layer_count, head_count = score_matrix.shape
    pairs: list[tuple[int, int]] = []
    for layer in torch.linspace(0, layer_count - 1, steps=min(sample_heads, layer_count)).round().long().tolist():
        pairs.append((int(layer), 0))
    remaining = sample_heads - len(pairs)
    if remaining > 0:
        flat_indices = score_matrix.flatten().argsort(descending=True).tolist()
        for index in flat_indices:
            layer = index // head_count
            head = index % head_count
            pair = (int(layer), int(head))
            if pair not in pairs:
                pairs.append(pair)
            if len(pairs) >= sample_heads:
                break
    return pairs[:sample_heads]


def flatten_samples(sample_list: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([sample.float() for sample in sample_list], dim=2)


def standardize_samples(samples: torch.Tensor, dim: int = 2) -> torch.Tensor:
    mean = samples.mean(dim=dim, keepdim=True)
    std = samples.std(dim=dim, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (samples - mean) / std


def apply_figure_title(fig: plt.Figure, title: str, note: str | None = None, fontsize: int = 12) -> None:
    text = f"{title}\n{note}" if note else title
    fig.suptitle(text, fontsize=fontsize)
