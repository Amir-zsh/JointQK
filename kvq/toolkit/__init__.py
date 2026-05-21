from __future__ import annotations

from kvq.toolkit.capture import (
    capture_rope_qk,
    run_generation_and_capture,
    run_prefill_qk_post_capture,
    run_prefill_and_capture,
)
from kvq.toolkit.eval import (
    compute_attention_metrics,
    compute_geometry_distortion,
    summarize_metrics,
)
from kvq.toolkit.io import (
    ensure_dir,
    save_json,
    torch_dtype_from_name,
    write_markdown_table,
)
from kvq.toolkit.metric_transform import (
    TRANSFORM_FAMILIES,
    apply_headwise_linear,
    build_metric_transform,
    compute_cca_basis,
    eigendecompose_metric_batch,
    factorize_metric,
    factorize_metric_batch,
    geometry_aware_roundtrip,
    match_transformed_token_norms,
    prepare_variant_states,
    repeat_kv_states,
    water_fill,
    whitening_factor,
)
from kvq.toolkit.model import get_model_device, load_model_and_tokenizer
from kvq.toolkit.moments import (
    CrossMomentsAccumulator,
    QueryMomentsAccumulator,
    compute_grouped_query_second_moment,
    compute_query_moments,
    split_prefill_and_decode,
    split_prefix_and_future,
    trim_queries,
)
from kvq.toolkit.per_coord_quantization import (
    PerCoordCompressor,
    build_jointqk_compressor,
    round_bits_to_integer,
    unit_gaussian_centroids,
)
from kvq.toolkit.quantization import (
    Stage1MSECompressor,
    generate_rotation_matrix,
    solve_lloyd_max,
)

__all__ = [
    "CrossMomentsAccumulator",
    "PerCoordCompressor",
    "QueryMomentsAccumulator",
    "Stage1MSECompressor",
    "TRANSFORM_FAMILIES",
    "apply_headwise_linear",
    "build_jointqk_compressor",
    "build_metric_transform",
    "capture_rope_qk",
    "compute_attention_metrics",
    "compute_cca_basis",
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
    "round_bits_to_integer",
    "run_generation_and_capture",
    "run_prefill_qk_post_capture",
    "run_prefill_and_capture",
    "save_json",
    "solve_lloyd_max",
    "split_prefill_and_decode",
    "split_prefix_and_future",
    "summarize_metrics",
    "torch_dtype_from_name",
    "trim_queries",
    "unit_gaussian_centroids",
    "water_fill",
    "whitening_factor",
    "write_markdown_table",
]
