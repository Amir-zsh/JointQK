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


def split_prefill_and_decode(
    queries: torch.Tensor,
    keys: torch.Tensor,
    prompt_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slice captured Q/K tensors into (prefill_Q, decode_Q, prefill_K).

    For Stage 1E E5: decode-phase queries read the prefill cache. The keys produced by
    decode-phase tokens are not used here (no compression target on those).

    Returns (prefill_queries, decode_queries, prefill_keys) where each is the
    L-axis slice of the input. Inputs are shaped (batch, n_heads, seq_len, head_dim).
    """
    seq_len = keys.shape[2]
    if prompt_length <= 0 or prompt_length >= seq_len:
        raise ValueError(
            f"Invalid prompt_length={prompt_length} for seq_len={seq_len}; "
            "expected 0 < prompt_length < seq_len."
        )
    prefill_q = queries[:, :, :prompt_length, :]
    decode_q = queries[:, :, prompt_length:, :]
    prefill_k = keys[:, :, :prompt_length, :]
    return prefill_q, decode_q, prefill_k


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


class CrossMomentsAccumulator:
    """Streaming accumulator for the per-(layer, kv_head) cross-moment C_QK = (1/N) sum_t q_t k_t^T.

    Q is GQA-pooled: query heads belonging to the same kv_head are averaged before the outer product.
    Used by the Stage 1E CCA pipeline.
    """

    def __init__(self) -> None:
        self.sum_qk: torch.Tensor | None = None
        self.total_count = 0

    def update(self, queries: torch.Tensor, keys: torch.Tensor, num_kv_heads: int) -> None:
        """queries shape: (batch, num_query_heads, seq_len, dim).
        keys shape:    (batch, num_kv_heads, seq_len, dim).
        Accumulates sum over t of q_t_pooled k_t^T per kv-head.
        """
        bsz, num_query_heads, seq_len, dim = queries.shape
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(
                f"Cannot group {num_query_heads} query heads into {num_kv_heads} KV heads evenly."
            )
        group_size = num_query_heads // num_kv_heads
        q_grouped = queries.float().view(bsz, num_kv_heads, group_size, seq_len, dim).mean(dim=2)
        k = keys.float()
        sum_qk = torch.einsum("bhsd,bhse->hde", q_grouped, k)
        if self.sum_qk is None:
            self.sum_qk = sum_qk.clone()
        else:
            self.sum_qk += sum_qk
        self.total_count += bsz * seq_len

    def finalize(self) -> torch.Tensor:
        if self.sum_qk is None or self.total_count == 0:
            raise RuntimeError("No cross-moment statistics were accumulated.")
        return self.sum_qk / self.total_count


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
