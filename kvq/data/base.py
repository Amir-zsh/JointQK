from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    hf_path: str
    config_names: tuple[str, ...]
    split: str
    row_to_messages: Callable[[dict], list[dict]]
    config_alias: Callable[[str], str] | None = None
    metadata_fields: tuple[str, ...] = ()


@dataclass
class FilteredExample:
    dataset: str
    config: str
    row_index: int
    messages: list[dict]
    prompt_text: str
    input_ids: torch.Tensor
    prompt_length: int
    metadata: dict[str, Any] = field(default_factory=dict)


DATASETS: dict[str, DatasetSpec] = {}


def register_dataset(spec: DatasetSpec) -> None:
    if spec.name in DATASETS:
        raise ValueError(f"Dataset '{spec.name}' already registered")
    DATASETS[spec.name] = spec


def get_dataset_spec(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Registered: {sorted(DATASETS)}")
    return DATASETS[name]


def _resolve_config(spec: DatasetSpec, raw_config: str) -> str:
    return spec.config_alias(raw_config) if spec.config_alias else raw_config


def _render_and_tokenize(
    spec: DatasetSpec,
    row: dict,
    config: str,
    row_index: int,
    tokenizer,
) -> FilteredExample:
    messages = spec.row_to_messages(row)
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    input_ids = tokenizer(prompt_text, return_tensors="pt", truncation=False)["input_ids"]
    metadata = {field_name: row.get(field_name) for field_name in spec.metadata_fields}
    return FilteredExample(
        dataset=spec.name,
        config=config,
        row_index=int(row_index),
        messages=messages,
        prompt_text=prompt_text,
        input_ids=input_ids,
        prompt_length=int(input_ids.shape[-1]),
        metadata=metadata,
    )


def load_and_filter(
    spec: DatasetSpec,
    *,
    tokenizer,
    num_examples_per_config: int,
    min_tokens: int,
    max_tokens: int,
    seed: int = 42,
    config_names: Sequence[str] | None = None,
) -> list[FilteredExample]:
    from datasets import load_dataset

    configs = tuple(config_names) if config_names is not None else spec.config_names
    selected: list[FilteredExample] = []

    for raw_config in configs:
        config = _resolve_config(spec, raw_config)
        dataset = load_dataset(spec.hf_path, config, split=spec.split)
        if len(dataset) == 0:
            continue

        config_seed = seed + sum(ord(ch) for ch in config)
        indices = list(range(len(dataset)))
        random.Random(config_seed).shuffle(indices)

        taken = 0
        for idx in indices:
            row = dataset[int(idx)]
            example = _render_and_tokenize(spec, row, config=config, row_index=idx, tokenizer=tokenizer)
            if example.prompt_length < min_tokens or example.prompt_length > max_tokens:
                continue
            selected.append(example)
            taken += 1
            if taken >= num_examples_per_config:
                break

    return selected


def fetch_example(
    spec: DatasetSpec,
    config: str,
    row_index: int,
    tokenizer,
) -> FilteredExample:
    from datasets import load_dataset

    resolved_config = _resolve_config(spec, config)
    dataset = load_dataset(spec.hf_path, resolved_config, split=spec.split)
    row = dataset[int(row_index)]
    return _render_and_tokenize(spec, row, config=resolved_config, row_index=row_index, tokenizer=tokenizer)
