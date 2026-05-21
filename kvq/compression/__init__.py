"""KV-cache compression primitives — bit allocation, scalar quantization, basis math.

No kvpress dependency in this subpackage; it can be reasoned about
without the bench harness in scope.
"""
from kvq.compression.lloyd_max import (
    Stage1MSECompressor,
    generate_rotation_matrix,
    solve_lloyd_max,
)
from kvq.compression.per_coord import (
    PerCoordCompressor,
    build_jointqk_compressor,
    round_bits_to_integer,
    unit_gaussian_centroids,
)
from kvq.compression.metric_transform import (
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
from kvq.compression.v_compressor_adapter import build_v_compressor
from kvq.compression.kivi_quantizer import kivi_quantize_keys, kivi_quantize_values
from kvq.compression.eval import (
    compute_attention_metrics,
    compute_geometry_distortion,
    summarize_metrics,
)

__all__ = [
    "Stage1MSECompressor",
    "PerCoordCompressor",
    "TRANSFORM_FAMILIES",
    "apply_headwise_linear",
    "build_jointqk_compressor",
    "build_metric_transform",
    "build_v_compressor",
    "compute_attention_metrics",
    "compute_cca_basis",
    "compute_geometry_distortion",
    "eigendecompose_metric_batch",
    "factorize_metric",
    "factorize_metric_batch",
    "generate_rotation_matrix",
    "geometry_aware_roundtrip",
    "kivi_quantize_keys",
    "kivi_quantize_values",
    "match_transformed_token_norms",
    "prepare_variant_states",
    "repeat_kv_states",
    "round_bits_to_integer",
    "solve_lloyd_max",
    "summarize_metrics",
    "unit_gaussian_centroids",
    "water_fill",
    "whitening_factor",
]
