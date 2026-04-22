from __future__ import annotations

import torch


def trim_queries(query_tensor: torch.Tensor, n_sink: int) -> torch.Tensor:
    if query_tensor.shape[2] <= n_sink:
        raise ValueError(f"Query tensor has {query_tensor.shape[2]} tokens, which is <= n_sink={n_sink}")
    return query_tensor[:, :, n_sink:, :]


def split_prefix_and_future(
    keys: torch.Tensor,
    future_queries: torch.Tensor,
    prefix_fraction: float,
    min_future_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    seq_len = keys.shape[2]
    split_at = max(1, int(seq_len * prefix_fraction))
    split_at = min(split_at, seq_len - min_future_tokens)
    if split_at <= 0 or split_at >= seq_len:
        raise ValueError(f"Invalid split point {split_at} for sequence length {seq_len}")
    return keys[:, :, :split_at, :], future_queries[:, :, split_at:, :], split_at


def compute_query_moments(query_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = query_tensor.float()
    mean = q.mean(dim=2)
    second_moment = torch.einsum("lhsd,lhse->lhde", q, q) / q.shape[2]
    cov = second_moment - torch.einsum("lhd,lhe->lhde", mean, mean)
    return mean, cov, second_moment


class QueryMomentsAccumulator:
    def __init__(self) -> None:
        self.sum_q: torch.Tensor | None = None
        self.sum_outer: torch.Tensor | None = None
        self.total_count = 0

    def update(self, query_tensor: torch.Tensor) -> None:
        q = query_tensor.float()
        q_sum = q.sum(dim=2)
        q_outer = torch.einsum("lhsd,lhse->lhde", q, q)
        if self.sum_q is None:
            self.sum_q = q_sum.clone()
            self.sum_outer = q_outer.clone()
        else:
            self.sum_q += q_sum
            self.sum_outer += q_outer
        self.total_count += q.shape[2]

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.sum_q is None or self.sum_outer is None or self.total_count == 0:
            raise RuntimeError("No query statistics were accumulated.")
        mean = self.sum_q / self.total_count
        second_moment = self.sum_outer / self.total_count
        cov = second_moment - torch.einsum("lhd,lhe->lhde", mean, mean)
        return mean, cov, second_moment


def compute_grouped_query_second_moment(future_queries: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    bsz, num_query_heads, future_len, dim = future_queries.shape
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"Cannot group {num_query_heads} query heads into {num_kv_heads} KV heads evenly."
        )
    group_size = num_query_heads // num_kv_heads
    grouped = future_queries.view(bsz, num_kv_heads, group_size, future_len, dim)
    flat = grouped.permute(1, 0, 2, 3, 4).reshape(num_kv_heads, -1, dim)
    return torch.einsum("hnd,hne->hde", flat, flat) / flat.shape[1]
