from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path


DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_DATASET = "Xnhyacinth/LongBench"
DEFAULT_CONFIG = "qasper_e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the Hugging Face environment errors seen in stage 1."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--show-env", action="store_true")
    return parser.parse_args()


def print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def print_env() -> None:
    keys = [
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    ]
    for key in keys:
        print(f"{key}={os.environ.get(key, '<unset>')}")


def reproduce_tokenizer_error(model_name: str) -> None:
    from transformers import AutoTokenizer

    print_header("Tokenizer Hub Lookup")
    print(f"Attempting: AutoTokenizer.from_pretrained({model_name!r}, trust_remote_code=True)")
    try:
        AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        print("Traceback:")
        traceback.print_exc()
    else:
        print("Unexpectedly succeeded.")


def reproduce_offline_dataset_lock_error(dataset_name: str, config_name: str) -> None:
    from datasets import load_dataset

    print_header("Offline Dataset Cache Lock")
    print(
        "Attempting: load_dataset("
        f"{dataset_name!r}, {config_name!r}, split='test') "
        "with HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1"
    )

    previous = {
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
    }
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        load_dataset(dataset_name, config_name, split="test")
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        print("Traceback:")
        traceback.print_exc()
    else:
        print("Unexpectedly succeeded.")
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main() -> None:
    args = parse_args()
    print(f"Working directory: {Path.cwd()}")
    if args.show_env:
        print_header("Environment")
        print_env()
    reproduce_tokenizer_error(args.model)
    reproduce_offline_dataset_lock_error(args.dataset, args.config)


if __name__ == "__main__":
    main()
