"""Model loading + RoPE-aware Q/K/V capture hooks + query moment accumulators.

Consumed by calibration pipelines, analysis scripts, and tests. No
compression / press logic lives here.
"""
from kvq.capture.hooks import (
    capture_rope_qk,
    run_generation_and_capture,
    run_prefill_and_capture,
    run_prefill_qk_post_capture,
)
from kvq.capture.model import get_model_device, load_model_and_tokenizer
from kvq.capture.moments import (
    CrossMomentsAccumulator,
    QueryMomentsAccumulator,
    compute_grouped_query_second_moment,
    compute_query_moments,
    split_prefill_and_decode,
    split_prefix_and_future,
    trim_queries,
)

__all__ = [
    "CrossMomentsAccumulator",
    "QueryMomentsAccumulator",
    "capture_rope_qk",
    "compute_grouped_query_second_moment",
    "compute_query_moments",
    "get_model_device",
    "load_model_and_tokenizer",
    "run_generation_and_capture",
    "run_prefill_and_capture",
    "run_prefill_qk_post_capture",
    "split_prefill_and_decode",
    "split_prefix_and_future",
    "trim_queries",
]
