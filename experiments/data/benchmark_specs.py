from __future__ import annotations

from experiments.data.base import DatasetSpec
from experiments.data.kvpress_adapter import build_kvpress_dataset_spec


def _longbench_e_alias(name: str) -> str:
    return name if name.endswith("_e") else f"{name}_e"


LONGBENCH_E = build_kvpress_dataset_spec(
    name="longbench-e",
    config_names=("qasper_e", "hotpotqa_e", "passage_retrieval_en_e"),
    metadata_fields=("task", "answers", "length", "all_classes"),
    config_alias=_longbench_e_alias,
)


ALL_SPECS: tuple[DatasetSpec, ...] = (LONGBENCH_E,)
