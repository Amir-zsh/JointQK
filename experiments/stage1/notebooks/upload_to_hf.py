"""Upload LongBench-E Q/K/V collections to Hugging Face Hub.

Requires `huggingface_hub` and an HF token (run `huggingface-cli login`, or export
`HF_TOKEN`). Each bundle goes to its own dataset repo, because `upload_large_folder`
uploads a whole folder at the repo root.

Usage:
    python experiments/stage1/notebooks/upload_to_hf.py \\
        --repo-prefix <user-or-org>/longbench-qkv-qwen3

By default both 'small' and 'full' bundles are uploaded; use --which to pick one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


BUNDLES = {
    "small": {
        "source": Path("artifacts/stage1/query_stats_longbench_under4k_small"),
        "repo_suffix": "-small",
    },
    "full": {
        "source": Path("artifacts/stage1/query_stats_longbench_under4k"),
        "repo_suffix": "-full",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-prefix",
        required=True,
        help="HF dataset repo prefix, e.g. 'amir/longbench-qkv-qwen3'. '-small' or '-full' is appended.",
    )
    parser.add_argument(
        "--which",
        choices=["small", "full", "both"],
        default="both",
        help="Which bundle(s) to upload. Default: both.",
    )
    parser.add_argument("--private", action="store_true", help="Create dataset repos as private.")
    return parser.parse_args()


def dir_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def upload_bundle(name: str, bundle: dict, repo_prefix: str, private: bool, api: HfApi) -> str:
    source = bundle["source"].resolve()
    repo_id = f"{repo_prefix}{bundle['repo_suffix']}"

    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")

    total_gb = dir_size_bytes(source) / 1e9
    print(f"\n=== Uploading '{name}' bundle ===")
    print(f"  source:  {source}")
    print(f"  repo:    https://huggingface.co/datasets/{repo_id}")
    print(f"  size:    {total_gb:.1f} GB")

    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)
    api.upload_large_folder(
        folder_path=str(source),
        repo_id=repo_id,
        repo_type="dataset",
    )
    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"  done:    {url}")
    return repo_id


def main() -> None:
    args = parse_args()
    api = HfApi()
    try:
        whoami = api.whoami()
        print(f"Authenticated as: {whoami.get('name', whoami)}")
    except Exception as exc:  # pragma: no cover - network/auth error surface
        raise SystemExit(
            f"HF authentication failed: {exc}\n"
            "Run `huggingface-cli login` or export HF_TOKEN."
        )

    names = ["small", "full"] if args.which == "both" else [args.which]
    uploaded = []
    for name in names:
        uploaded.append((name, upload_bundle(name, BUNDLES[name], args.repo_prefix, args.private, api)))

    print("\n=== Done ===")
    for name, repo_id in uploaded:
        print(f"  {name:5s} -> {repo_id}")
    print("\nSet these as SPECS[...]['repo_id'] in experiments/stage1/notebooks/longbench_data_tour.ipynb.")


if __name__ == "__main__":
    main()
