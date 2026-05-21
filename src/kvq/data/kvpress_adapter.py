from __future__ import annotations

from typing import Callable

from kvq.benchmarks.evaluate_registry import DATASET_REGISTRY
from kvq.data.base import DatasetSpec


def kvpress_row_to_messages(row: dict) -> list[dict]:
    """Assemble kvpress-style prompts: context + question + answer_prefix, wrapped as a single user turn.

    kvpress's own evaluate.py uses a non-chat-templated pipeline ('kv-press-text-generation'),
    but we keep chat templating on our side because Qwen3 requires it and all Stage 1 data was
    collected that way. The prompt *content* (context+question+answer_prefix) is identical.
    """
    parts = [row["context"], row["question"]]
    if row.get("answer_prefix"):
        parts.append(row["answer_prefix"])
    return [{"role": "user", "content": "\n\n".join(parts)}]


def build_kvpress_dataset_spec(
    name: str,
    config_names: tuple[str, ...],
    metadata_fields: tuple[str, ...],
    split: str = "test",
    config_alias: Callable[[str], str] | None = None,
) -> DatasetSpec:
    """Construct a DatasetSpec for a kvpress-vendored benchmark.

    `hf_path` is sourced from the vendored DATASET_REGISTRY. `metadata_fields` always
    includes `max_new_tokens` so the collector can honor kvpress's per-task generation
    budget (see collect_query_stats.py).
    """
    if name not in DATASET_REGISTRY:
        raise KeyError(f"'{name}' not in vendored DATASET_REGISTRY; known keys: {sorted(DATASET_REGISTRY)}")
    return DatasetSpec(
        name=name,
        hf_path=DATASET_REGISTRY[name],
        config_names=config_names,
        split=split,
        row_to_messages=kvpress_row_to_messages,
        config_alias=config_alias,
        metadata_fields=tuple(sorted({"max_new_tokens", *metadata_fields})),
    )
