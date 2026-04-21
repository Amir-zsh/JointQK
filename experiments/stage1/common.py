from __future__ import annotations

import importlib
import json
import math
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
KVPRESS_ROOT = REPO_ROOT / "kvpress"
if str(KVPRESS_ROOT) not in sys.path:
    sys.path.insert(0, str(KVPRESS_ROOT))


DEFAULT_TASKS = ["qasper", "hotpotqa", "passage_retrieval_en"]
TRANSFORM_FAMILIES = [
    "baseline_raw",
    "basis_only",
    "full_metric",
    "trace_matched_full_metric",
    "per_token_norm_matched_full_metric",
]
_LLOYD_MAX_CACHE: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}


@dataclass
class PromptRecord:
    task: str
    example_index: int
    context: str
    question: str
    answer_prefix: str


def parse_task_names(task_names: str | list[str] | None) -> list[str]:
    if task_names is None:
        return list(DEFAULT_TASKS)
    if isinstance(task_names, list):
        return task_names
    return [task.strip() for task in task_names.split(",") if task.strip()]


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    def _convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, torch.Tensor):
            return value.tolist()
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_convert(v) for v in value]
        return value

    Path(path).write_text(json.dumps(_convert(payload), indent=2, sort_keys=True))


def torch_dtype_from_name(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype_name}'. Expected one of {sorted(mapping)}")
    return mapping[dtype_name]


def get_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def load_model_and_tokenizer(
    model_name: str,
    device_map: str = "auto",
    dtype_name: str = "float16",
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch_dtype_from_name(dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_longbench_slice(
    task_names: list[str],
    num_examples: int,
    seed: int = 42,
    dataset_name: str = "longbench-e",
) -> list[PromptRecord]:
    from datasets import load_dataset

    examples: list[PromptRecord] = []
    per_task = max(1, math.ceil(num_examples / max(len(task_names), 1)))

    for task in task_names:
        config_name = task
        if dataset_name == "longbench-e" and not task.endswith("_e"):
            config_name = f"{task}_e"
        dataset = load_dataset("Xnhyacinth/LongBench", config_name, split="test")
        if len(dataset) == 0:
            continue

        task_seed = seed + sum(ord(ch) for ch in task)
        rng = random.Random(task_seed)
        indices = list(range(len(dataset)))
        rng.shuffle(indices)

        for idx in indices[:per_task]:
            row = dataset[int(idx)]
            examples.append(
                PromptRecord(
                    task=task,
                    example_index=int(idx),
                    context=row["context"],
                    question=row["question"],
                    answer_prefix=row.get("answer_prefix", ""),
                )
            )
            if len(examples) >= num_examples:
                return examples

    return examples[:num_examples]


def build_prompt(record: PromptRecord, include_answer_prefix: bool = True) -> str:
    parts = [record.context, record.question]
    if include_answer_prefix and record.answer_prefix:
        parts.append(record.answer_prefix)
    return "".join(parts)


@contextmanager
def capture_rotary_queries(model: torch.nn.Module) -> Iterator[tuple[list[torch.Tensor], list[torch.Tensor]]]:
    module_path = model.__class__.__module__
    modeling_module = importlib.import_module(module_path)
    target_function = "apply_rotary_pos_emb"
    if not hasattr(modeling_module, target_function):
        raise AttributeError(f"Model module '{module_path}' does not expose '{target_function}'.")

    original_function = getattr(modeling_module, target_function)
    captured_pre: list[torch.Tensor] = []
    captured_post: list[torch.Tensor] = []

    def patched_function(q_embed, k_embed, *args, **kwargs):
        captured_pre.append(q_embed.detach().cpu())
        q_out, k_out = original_function(q_embed, k_embed, *args, **kwargs)
        captured_post.append(q_out.detach().cpu())
        return q_out, k_out

    setattr(modeling_module, target_function, patched_function)
    try:
        yield captured_pre, captured_post
    finally:
        setattr(modeling_module, target_function, original_function)


@torch.inference_mode()
def run_prefill_and_capture(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_context_length: int,
):
    device = get_model_device(model)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_context_length,
    ).to(device)
    with capture_rotary_queries(model) as (captured_pre, captured_post):
        outputs = model(**inputs, use_cache=True)
    pre = torch.cat(captured_pre, dim=0).float()
    post = torch.cat(captured_post, dim=0).float()
    return outputs, pre, post, inputs


def trim_queries(query_tensor: torch.Tensor, n_sink: int) -> torch.Tensor:
    if query_tensor.shape[2] <= n_sink:
        raise ValueError(f"Query tensor has {query_tensor.shape[2]} tokens, which is <= n_sink={n_sink}")
    return query_tensor[:, :, n_sink:, :]


def sample_queries(query_tensor: torch.Tensor, sample_size: int, seed: int) -> torch.Tensor:
    seq_len = query_tensor.shape[2]
    if seq_len <= sample_size:
        return query_tensor.clone()
    rng = random.Random(seed)
    indices = sorted(rng.sample(list(range(seq_len)), sample_size))
    return query_tensor[:, :, indices, :].clone()


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
            self.sum_q = q_sum
            self.sum_outer = q_outer
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


def gaussian_approx_pdf(x: float, d: int) -> float:
    sigma2 = 1.0 / d
    return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(-(x * x) / (2 * sigma2))


def solve_lloyd_max(d: int, bits: int, max_iter: int = 200, tol: float = 1e-10) -> tuple[torch.Tensor, torch.Tensor]:
    cache_key = (d, bits)
    cached = _LLOYD_MAX_CACHE.get(cache_key)
    if cached is not None:
        centroids, boundaries = cached
        return centroids.clone(), boundaries.clone()

    from scipy import integrate

    n_levels = 2**bits
    sigma = 1.0 / math.sqrt(d)
    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for _ in range(max_iter):
        boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
        edges = [lo * 3] + boundaries + [hi * 3]
        updated = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            numerator, _ = integrate.quad(lambda x: x * gaussian_approx_pdf(x, d), a, b)
            denominator, _ = integrate.quad(lambda x: gaussian_approx_pdf(x, d), a, b)
            updated.append(numerator / denominator if denominator > 1e-15 else centroids[i])
        max_shift = max(abs(updated[i] - centroids[i]) for i in range(n_levels))
        centroids = updated
        if max_shift < tol:
            break

    boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
    centroids_tensor = torch.tensor(centroids, dtype=torch.float32)
    boundaries_tensor = torch.tensor(boundaries, dtype=torch.float32)
    _LLOYD_MAX_CACHE[cache_key] = (centroids_tensor.clone(), boundaries_tensor.clone())
    return centroids_tensor, boundaries_tensor


def generate_rotation_matrix(d: int, seed: int | None = None, device: str | torch.device = "cpu") -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)
    gaussian = torch.randn(d, d, generator=generator)
    q, r = torch.linalg.qr(gaussian)
    diag_sign = torch.sign(torch.diag(r))
    diag_sign[diag_sign == 0] = 1.0
    q = q * diag_sign.unsqueeze(0)
    return q.to(device)


class Stage1MSECompressor:
    """
    V3-equivalent single-stage MSE compressor used for the stage-1 study.
    """

    def __init__(self, head_dim: int, bits: int, seed: int, device: str | torch.device = "cpu") -> None:
        self.head_dim = head_dim
        self.bits = bits
        self.device = str(device)
        self.Pi = generate_rotation_matrix(head_dim, seed=seed, device=device)
        self.centroids, _ = solve_lloyd_max(head_dim, bits)
        self.centroids = self.centroids.to(device)

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> dict[str, Any]:
        bsz, heads, seq_len, dim = states.shape
        flat = states.reshape(-1, dim).float()
        vec_norms = torch.norm(flat, dim=-1)
        flat_norm = flat / (vec_norms.unsqueeze(-1) + 1e-8)
        rotated = flat_norm @ self.Pi.T
        diffs = rotated.unsqueeze(-1) - self.centroids
        indices = diffs.abs().argmin(dim=-1).to(torch.uint8)

        indices_per_byte = 8 // self.bits
        idx_pad = (indices_per_byte - dim % indices_per_byte) % indices_per_byte
        idx_flat = indices.long()
        if idx_pad:
            idx_flat = F.pad(idx_flat, (0, idx_pad))
        n_groups = idx_flat.shape[-1] // indices_per_byte
        idx_powers = torch.tensor(
            [2 ** (self.bits * i) for i in range(indices_per_byte - 1, -1, -1)],
            dtype=torch.long,
            device=idx_flat.device,
        )
        idx_bytes = (idx_flat.reshape(flat.shape[0], n_groups, indices_per_byte) * idx_powers).sum(-1).to(torch.uint8)
        return {
            "idx_bytes": idx_bytes.reshape(bsz, heads, seq_len, n_groups),
            "vec_norms": vec_norms.to(torch.float16).reshape(bsz, heads, seq_len),
            "shape": (bsz, heads, seq_len, dim),
            "idx_pad": idx_pad,
        }

    @torch.no_grad()
    def decompress(self, compressed: dict[str, Any]) -> torch.Tensor:
        bsz, heads, seq_len, dim = compressed["shape"]
        idx_bytes = compressed["idx_bytes"].reshape(-1, compressed["idx_bytes"].shape[-1])
        vec_norms = compressed["vec_norms"].reshape(-1, 1).float()
        idx_pad = compressed["idx_pad"]

        indices_per_byte = 8 // self.bits
        mask = (1 << self.bits) - 1
        idx_shifts = torch.tensor(
            [self.bits * i for i in range(indices_per_byte - 1, -1, -1)],
            dtype=torch.long,
            device=idx_bytes.device,
        )
        indices = ((idx_bytes.long().unsqueeze(-1) >> idx_shifts) & mask).reshape(-1, indices_per_byte * idx_bytes.shape[-1])
        if idx_pad:
            indices = indices[:, :dim]

        reconstructed = (self.centroids[indices] @ self.Pi) * vec_norms
        return reconstructed.reshape(bsz, heads, seq_len, dim)

    @torch.no_grad()
    def roundtrip(self, states: torch.Tensor) -> torch.Tensor:
        return self.decompress(self.compress(states))


def factorize_metric(metric: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    sym = 0.5 * (metric + metric.transpose(-1, -2))
    sym = sym + eps * torch.eye(sym.shape[-1], dtype=sym.dtype, device=sym.device)
    try:
        factor = torch.linalg.cholesky(sym)
        inverse = torch.linalg.inv(factor)
    except RuntimeError:
        raise Exception("Errro occured during factorize_metric")
        eigenvalues, eigenvectors = torch.linalg.eigh(sym)
        clipped = eigenvalues.clamp_min(eps)
        sqrt_vals = torch.sqrt(clipped)
        inv_sqrt_vals = 1.0 / sqrt_vals
        factor = eigenvectors @ torch.diag(sqrt_vals)
        # `apply_headwise_linear` treats vectors as row vectors, so the right inverse
        # must undo `x @ factor` via `x @ factor @ inverse = x`.
        inverse = torch.diag(inv_sqrt_vals) @ eigenvectors.transpose(-1, -2)
    return factor, inverse


def factorize_metric_batch(metrics: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    factors = []
    inverses = []
    for metric in metrics:
        factor, inverse = factorize_metric(metric.float(), eps)
        factors.append(factor)
        inverses.append(inverse)
    return torch.stack(factors, dim=0), torch.stack(inverses, dim=0)


def eigendecompose_metric_batch(metrics: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    sym = 0.5 * (metrics.float() + metrics.float().transpose(-1, -2))
    eye = torch.eye(sym.shape[-1], dtype=sym.dtype, device=sym.device).unsqueeze(0)
    sym = sym + eps * eye
    eigenvalues, eigenvectors = torch.linalg.eigh(sym)
    clipped = eigenvalues.clamp_min(eps)
    return clipped, eigenvectors


def _build_scaled_eigen_transform(
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    coord_scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale_embed = torch.diag_embed(coord_scales)
    inverse_embed = torch.diag_embed(1.0 / coord_scales.clamp_min(1e-12))
    transform = eigenvectors @ scale_embed
    inverse = inverse_embed @ eigenvectors.transpose(-1, -2)
    return transform, inverse


def build_metric_transform(
    metrics: torch.Tensor,
    variant: str,
    eps: float,
    gamma: float | None = None,
) -> dict[str, torch.Tensor | float | str]:
    if variant not in TRANSFORM_FAMILIES and variant != "gamma_sweep":
        raise ValueError(f"Unsupported transform variant '{variant}'.")

    if variant == "baseline_raw":
        dim = metrics.shape[-1]
        identity = torch.eye(dim, dtype=metrics.dtype, device=metrics.device).unsqueeze(0).repeat(metrics.shape[0], 1, 1)
        ones = torch.ones(metrics.shape[0], dtype=metrics.dtype, device=metrics.device)
        return {
            "variant": variant,
            "transform": identity,
            "inverse": identity,
            "eigenvalues": torch.ones(metrics.shape[0], dim, dtype=metrics.dtype, device=metrics.device),
            "eigenvectors": identity,
            "coord_scales": torch.ones(metrics.shape[0], dim, dtype=metrics.dtype, device=metrics.device),
            "trace_scale_alpha": ones,
        }

    eigenvalues, eigenvectors = eigendecompose_metric_batch(metrics, eps)
    dim = eigenvalues.shape[-1]
    ones = torch.ones(eigenvalues.shape[0], dtype=eigenvalues.dtype, device=eigenvalues.device)

    if variant == "basis_only":
        coord_scales = torch.ones_like(eigenvalues)
        trace_scale_alpha = ones
    elif variant == "full_metric":
        coord_scales = torch.sqrt(eigenvalues)
        trace_scale_alpha = ones
    elif variant == "trace_matched_full_metric":
        base_scales = torch.sqrt(eigenvalues)
        trace_scale_alpha = torch.sqrt(torch.tensor(float(dim), dtype=eigenvalues.dtype, device=eigenvalues.device) / eigenvalues.sum(dim=-1).clamp_min(1e-12))
        coord_scales = base_scales * trace_scale_alpha.unsqueeze(-1)
    elif variant == "per_token_norm_matched_full_metric":
        coord_scales = torch.sqrt(eigenvalues)
        trace_scale_alpha = ones
    elif variant == "gamma_sweep":
        if gamma is None:
            raise ValueError("gamma must be provided for gamma_sweep transforms.")
        base_scales = eigenvalues.pow(gamma / 2.0)
        gamma_trace = eigenvalues.pow(gamma).sum(dim=-1).clamp_min(1e-12)
        trace_scale_alpha = torch.sqrt(torch.tensor(float(dim), dtype=eigenvalues.dtype, device=eigenvalues.device) / gamma_trace)
        coord_scales = base_scales * trace_scale_alpha.unsqueeze(-1)
    else:
        raise ValueError(f"Unhandled transform variant '{variant}'.")

    transform, inverse = _build_scaled_eigen_transform(eigenvalues, eigenvectors, coord_scales)
    result: dict[str, torch.Tensor | float | str] = {
        "variant": variant,
        "transform": transform,
        "inverse": inverse,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "coord_scales": coord_scales,
        "trace_scale_alpha": trace_scale_alpha,
    }
    if gamma is not None:
        result["gamma"] = float(gamma)
    return result


def match_transformed_token_norms(
    reference_states: torch.Tensor,
    transformed_states: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    reference_norms = torch.linalg.vector_norm(reference_states.float(), dim=-1, keepdim=True)
    transformed_norms = torch.linalg.vector_norm(transformed_states.float(), dim=-1, keepdim=True)
    scales = reference_norms / transformed_norms.clamp_min(eps)
    return transformed_states * scales, scales


def prepare_variant_states(
    states: torch.Tensor,
    metrics: torch.Tensor,
    variant: str,
    eps: float,
    gamma: float | None = None,
) -> dict[str, torch.Tensor | float | str]:
    transform_payload = build_metric_transform(metrics.to(states.device), variant=variant, eps=eps, gamma=gamma)
    transform = transform_payload["transform"]
    inverse = transform_payload["inverse"]
    transformed = apply_headwise_linear(states.float(), transform)
    compressor_input = transformed
    norm_match_scales: torch.Tensor | None = None
    if variant == "per_token_norm_matched_full_metric":
        compressor_input, norm_match_scales = match_transformed_token_norms(states, transformed, eps)
    return {
        **transform_payload,
        "transformed": transformed,
        "compressor_input": compressor_input,
        "norm_match_scales": norm_match_scales,
        "input_norms": torch.linalg.vector_norm(states.float(), dim=-1),
        "transformed_norms": torch.linalg.vector_norm(transformed.float(), dim=-1),
    }


def apply_headwise_linear(states: torch.Tensor, matrices: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhsd,hdf->bhsf", states, matrices)


def repeat_kv_states(keys: torch.Tensor, num_query_heads: int) -> torch.Tensor:
    bsz, num_kv_heads, seq_len, dim = keys.shape
    if num_query_heads == num_kv_heads:
        return keys
    group_size = num_query_heads // num_kv_heads
    expanded = keys[:, :, None, :, :].expand(bsz, num_kv_heads, group_size, seq_len, dim)
    return expanded.reshape(bsz, num_query_heads, seq_len, dim)


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


def geometry_aware_roundtrip(
    states: torch.Tensor,
    metrics: torch.Tensor,
    bits: int,
    seed: int,
    eps: float,
) -> torch.Tensor:
    factors, inverses = factorize_metric_batch(metrics.to(states.device), eps)
    transformed = apply_headwise_linear(states.float(), factors)
    compressor = Stage1MSECompressor(states.shape[-1], bits, seed=seed, device=states.device)
    transformed_recon = compressor.roundtrip(transformed)
    return apply_headwise_linear(transformed_recon, inverses)


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


def summarize_metrics(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    if not metric_rows:
        return {}
    keys = sorted(metric_rows[0].keys())
    summary = {}
    for key in keys:
        values = [row[key] for row in metric_rows]
        summary[key] = float(sum(values) / len(values))
    return summary


def write_markdown_table(path: str | Path, title: str, sections: dict[str, dict[str, float]]) -> None:
    lines = [f"# {title}", ""]
    for name, metrics in sections.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | ---: |")
        for metric_name, value in metrics.items():
            lines.append(f"| {metric_name} | {value:.6f} |")
        lines.append("")
    Path(path).write_text("\n".join(lines))
