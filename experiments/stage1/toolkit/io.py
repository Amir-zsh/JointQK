from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    def _convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, torch.Tensor):
            return value.tolist()
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_convert(v) for v in value]
        return value

    Path(path).write_text(json.dumps(_convert(payload), indent=2, sort_keys=True))


def torch_dtype_from_name(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype_name}'. Expected one of {sorted(mapping)}")
    return mapping[dtype_name]


def write_markdown_table(path: str | Path, title: str, sections: dict[str, dict[str, float]]) -> None:
    lines = [f"# {title}", ""]
    for name, metrics in sections.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | ---: |")
        for metric_name, value in metrics.items():
            lines.append(f"| {metric_name} | {value:.6f} |")
        lines.append("")
    Path(path).write_text("\n".join(lines))
