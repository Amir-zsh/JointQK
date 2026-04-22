from __future__ import annotations

import pytest
import torch

from experiments.stage1.data import (
    DATASETS,
    DatasetSpec,
    FilteredExample,
    get_dataset_spec,
    register_dataset,
)
from experiments.stage1.data.longbench import (
    LONGBENCH_E,
    _longbench_e_alias,
    _longbench_messages,
)


def test_longbench_e_is_registered():
    assert "longbench-e" in DATASETS
    spec = get_dataset_spec("longbench-e")
    assert spec is LONGBENCH_E
    assert spec.hf_path == "Xnhyacinth/LongBench"
    assert spec.split == "test"
    assert "qasper_e" in spec.config_names
    assert "hotpotqa_e" in spec.config_names
    assert "passage_retrieval_en_e" in spec.config_names


def test_get_dataset_spec_unknown_raises_key_error():
    with pytest.raises(KeyError):
        get_dataset_spec("definitely-not-a-dataset")


def test_register_dataset_duplicate_raises_value_error():
    duplicate = DatasetSpec(
        name="longbench-e",
        hf_path="fake",
        config_names=(),
        split="test",
        row_to_messages=lambda row: [],
    )
    with pytest.raises(ValueError):
        register_dataset(duplicate)


def test_longbench_messages_without_answer_prefix():
    messages = _longbench_messages({"context": "ctx", "question": "q?"})
    assert messages == [{"role": "user", "content": "ctx\n\nq?"}]


def test_longbench_messages_with_empty_answer_prefix():
    messages = _longbench_messages({"context": "ctx", "question": "q?", "answer_prefix": ""})
    assert messages == [{"role": "user", "content": "ctx\n\nq?"}]


def test_longbench_messages_with_answer_prefix():
    messages = _longbench_messages(
        {"context": "ctx", "question": "q?", "answer_prefix": "A:"}
    )
    assert messages == [{"role": "user", "content": "ctx\n\nq?\n\nA:"}]


def test_longbench_e_alias_idempotent():
    assert _longbench_e_alias("qasper") == "qasper_e"
    assert _longbench_e_alias("qasper_e") == "qasper_e"
    assert _longbench_e_alias("hotpotqa") == "hotpotqa_e"


def test_filtered_example_fields():
    example = FilteredExample(
        dataset="longbench-e",
        config="qasper_e",
        row_index=7,
        messages=[{"role": "user", "content": "hi"}],
        prompt_text="hi",
        input_ids=torch.zeros((1, 3), dtype=torch.long),
        prompt_length=3,
        metadata={"answers": ["42"]},
    )
    assert example.dataset == "longbench-e"
    assert example.prompt_length == 3
    assert example.metadata == {"answers": ["42"]}
