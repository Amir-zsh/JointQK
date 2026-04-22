from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_KVPRESS_ROOT = _REPO_ROOT / "kvpress"
if str(_KVPRESS_ROOT) not in sys.path:
    sys.path.insert(0, str(_KVPRESS_ROOT))

from experiments.stage1.toolkit.capture import (
    capture_rope_qk,
    run_generation_and_capture,
    run_prefill_and_capture,
)
from experiments.stage1.toolkit.eval import (
    compute_attention_metrics,
    compute_geometry_distortion,
    summarize_metrics,
)
from experiments.stage1.toolkit.io import (
    ensure_dir,
    save_json,
    torch_dtype_from_name,
    write_markdown_table,
)
from experiments.stage1.toolkit.metric_transform import (
    TRANSFORM_FAMILIES,
    apply_headwise_linear,
    build_metric_transform,
    eigendecompose_metric_batch,
    factorize_metric,
    factorize_metric_batch,
    geometry_aware_roundtrip,
    match_transformed_token_norms,
    prepare_variant_states,
    repeat_kv_states,
)
from experiments.stage1.toolkit.model import get_model_device, load_model_and_tokenizer
from experiments.stage1.toolkit.moments import (
    QueryMomentsAccumulator,
    compute_grouped_query_second_moment,
    compute_query_moments,
    split_prefix_and_future,
    trim_queries,
)
from experiments.stage1.toolkit.quantization import (
    Stage1MSECompressor,
    generate_rotation_matrix,
    solve_lloyd_max,
)

__all__ = [
    "QueryMomentsAccumulator",
    "Stage1MSECompressor",
    "TRANSFORM_FAMILIES",
    "apply_headwise_linear",
    "build_metric_transform",
    "capture_rope_qk",
    "compute_attention_metrics",
    "compute_geometry_distortion",
    "compute_grouped_query_second_moment",
    "compute_query_moments",
    "eigendecompose_metric_batch",
    "ensure_dir",
    "factorize_metric",
    "factorize_metric_batch",
    "generate_rotation_matrix",
    "geometry_aware_roundtrip",
    "get_model_device",
    "load_model_and_tokenizer",
    "match_transformed_token_norms",
    "prepare_variant_states",
    "repeat_kv_states",
    "run_generation_and_capture",
    "run_prefill_and_capture",
    "save_json",
    "solve_lloyd_max",
    "split_prefix_and_future",
    "summarize_metrics",
    "torch_dtype_from_name",
    "trim_queries",
    "write_markdown_table",
]
