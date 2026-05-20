from experiments.data.base import (
    DATASETS,
    DatasetSpec,
    FilteredExample,
    fetch_example,
    get_dataset_spec,
    load_and_filter,
    register_dataset,
)
from experiments.data.kvpress_adapter import (
    build_kvpress_dataset_spec,
    kvpress_row_to_messages,
)


def register_all_benchmark_specs() -> None:
    """Register every DatasetSpec in benchmark_specs.ALL_SPECS. Idempotent.

    Opt-in: callers (drivers, tests) must invoke this before using get_dataset_spec.
    Importing experiments.data alone does not mutate DATASETS.
    """
    from experiments.data.benchmark_specs import ALL_SPECS

    for spec in ALL_SPECS:
        if spec.name not in DATASETS:
            register_dataset(spec)


__all__ = [
    "DATASETS",
    "DatasetSpec",
    "FilteredExample",
    "build_kvpress_dataset_spec",
    "fetch_example",
    "get_dataset_spec",
    "kvpress_row_to_messages",
    "load_and_filter",
    "register_all_benchmark_specs",
    "register_dataset",
]
