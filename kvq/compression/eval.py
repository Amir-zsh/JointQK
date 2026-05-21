from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from kvq.compression.metric_transform import repeat_kv_states


def compute_attention_metrics(
    future_queries: torch.Tensor,
    reference_keys: torch.Tensor,
    candidate_keys: torch.Tensor,
) -> dict[str, float]:
    dim = reference_keys.shape[-1]
    repeated_reference = repeat_kv_states(reference_keys, future_queries.shape[1])
    repeated_candidate = repeat_kv_states(candidate_keys, future_queries.shape[1])

    real_logits = torch.matmul(future_queries.float(), repeated_reference.float().transpose(-2, -1)) / math.sqrt(dim)
    approx_logits = torch.matmul(future_queries.float(), repeated_candidate.float().transpose(-2, -1)) / math.sqrt(dim)

    diff = approx_logits - real_logits
    real_flat = real_logits.reshape(-1, real_logits.shape[-1])
    approx_flat = approx_logits.reshape(-1, approx_logits.shape[-1])
    k = min(5, real_flat.shape[-1])
    real_top1 = real_flat.argmax(dim=-1)
    approx_top1 = approx_flat.argmax(dim=-1)
    approx_topk = approx_flat.topk(k, dim=-1).indices
    top1_match = (real_top1 == approx_top1).float().mean().item()
    top5_containment = (approx_topk == real_top1.unsqueeze(-1)).any(dim=-1).float().mean().item()
    cosine = F.cosine_similarity(real_flat, approx_flat, dim=-1).mean().item()
    return {
        "logit_mse": diff.pow(2).mean().item(),
        "logit_cosine": cosine,
        "top1_match": top1_match,
        "top5_containment": top5_containment,
    }


def compute_geometry_distortion(
    reconstructed: torch.Tensor,
    reference: torch.Tensor,
    metrics: torch.Tensor,
) -> float:
    error = (reconstructed.float() - reference.float()).to(metrics.device)
    dist = torch.einsum("bhsd,hde,bhse->bhs", error, metrics.float(), error)
    return (dist / reference.shape[-1]).mean().item()


def summarize_metrics(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    if not metric_rows:
        return {}
    keys = sorted(metric_rows[0].keys())
    summary = {}
    for key in keys:
        values = [row[key] for row in metric_rows]
        summary[key] = float(sum(values) / len(values))
    return summary
