from __future__ import annotations

from experiments.stage1.data.base import DatasetSpec, register_dataset


def _longbench_messages(row: dict) -> list[dict]:
    parts = [row["context"], row["question"]]
    if row.get("answer_prefix"):
        parts.append(row["answer_prefix"])
    content = "\n\n".join(parts)
    return [{"role": "user", "content": content}]


def _longbench_e_alias(name: str) -> str:
    return name if name.endswith("_e") else f"{name}_e"


LONGBENCH_E = DatasetSpec(
    name="longbench-e",
    hf_path="Xnhyacinth/LongBench",
    config_names=("qasper_e", "hotpotqa_e", "passage_retrieval_en_e"),
    split="test",
    row_to_messages=_longbench_messages,
    config_alias=_longbench_e_alias,
    metadata_fields=("answers", "length", "all_classes"),
)


register_dataset(LONGBENCH_E)
