from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


DEFAULT_SPLIT = Path("artifacts/stage1/calibration_splits/longbench_compact8_60_seed20260504_2k32k/manifest.json")
DEFAULT_ARTIFACT_ROOT = Path("artifacts/stage1/calibration")
DEFAULT_RUN_ID = "longbench_compact8_qkv"
DEFAULT_K_BITS = (2, 3, 4)
DEFAULT_V_BITS = (2, 3, 4)
DEFAULT_SAMPLE_SIZES = (1, 2, 4, 8, 16, 32, 50)
DEFAULT_RANKS = (16, 32, 64, 96)
RAW_TENSORS = ("q_post", "k_post", "v")

# Raw-tensor retention policy. Default keeps raw only for `split == "test"` examples — train
# examples contribute only to second-moment accumulation and never need to be re-loaded for
# empirical evaluation, so dropping their raw saves ~80% of disk on the standard split.
KEEP_RAW_CHOICES = ("none", "test", "all")
DEFAULT_KEEP_RAW = "test"


def should_keep_raw(row: dict[str, Any], keep_raw: str) -> bool:
    """Return True iff the per-example raw fp16 q/k/v should be persisted to disk."""
    if keep_raw == "all":
        return True
    if keep_raw == "none":
        return False
    if keep_raw == "test":
        return row.get("split") == "test"
    raise ValueError(f"Unknown --keep-raw mode: {keep_raw!r}; expected one of {KEEP_RAW_CHOICES}")


@dataclass(frozen=True)
class RunPaths:
    root: Path
    split_dir: Path
    raw_dir: Path
    stats_dir: Path
    bases_dir: Path
    analysis_dir: Path
    reports_dir: Path
    logs_dir: Path

    @classmethod
    def from_args(cls, artifact_root: str | Path, run_id: str) -> "RunPaths":
        root = Path(artifact_root) / run_id
        return cls(
            root=root,
            split_dir=root / "00_split",
            raw_dir=root / "01_raw",
            stats_dir=root / "02_stats",
            bases_dir=root / "03_bases",
            analysis_dir=root / "04_analysis",
            reports_dir=root / "05_reports",
            logs_dir=root / "logs",
        )

    def ensure(self) -> None:
        for path in (
            self.root,
            self.split_dir,
            self.raw_dir,
            self.stats_dir,
            self.bases_dir,
            self.analysis_dir,
            self.reports_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{sec:02d}s"


def read_json(path: str | Path) -> Any:
    with Path(path).open() as f:
        return json.load(f)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def copy_split_manifest(split_manifest: Path, paths: RunPaths) -> Path:
    paths.ensure()
    out = paths.split_dir / "manifest.json"
    if split_manifest.resolve() != out.resolve():
        shutil.copy2(split_manifest, out)
    return out


def parse_csv_ints(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def parse_gpus(value: str) -> list[str]:
    gpus = [x.strip() for x in value.split(",") if x.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    return gpus


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--smoke", action="store_true", help="Use a tiny 2 train + 2 test subset per task.")
    parser.add_argument("--resume", action="store_true")


def add_keep_raw_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--keep-raw",
        choices=KEEP_RAW_CHOICES,
        default=DEFAULT_KEEP_RAW,
        help=(
            "Which examples to retain raw fp16 q/k/v on disk for. "
            "'none' = stats only (smallest disk, breaks empirical mode); "
            "'test' = only test-split examples (default — keeps empirical mode working since it only "
            "reads test, saves IO on train); "
            "'all' = retain raw for every example (largest disk, full re-analysis flexibility)."
        ),
    )


def add_shard_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)


def validate_shard_args(num_shards: int, shard_id: int) -> None:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= shard_id < num_shards):
        raise ValueError(f"--shard-id must be in [0, {num_shards}), got {shard_id}")


def stable_example_id(row: dict[str, Any]) -> str:
    return f"{row['dataset']}__{row['config']}__row{int(row['row_index']):05d}__{row['split']}"


def raw_example_path(paths: RunPaths, row: dict[str, Any]) -> Path:
    return paths.raw_dir / f"shard_{int(row.get('assigned_shard', 0)):03d}" / f"{stable_example_id(row)}.pt"


def stats_example_path(paths: RunPaths, row: dict[str, Any]) -> Path:
    return paths.stats_dir / f"shard_{int(row.get('assigned_shard', 0)):03d}" / f"{stable_example_id(row)}.pt"


def select_rows(split: dict[str, Any], smoke: bool) -> list[dict[str, Any]]:
    rows = [dict(row) for row in split["rows"]]
    if not smoke:
        return rows
    selected: list[dict[str, Any]] = []
    tasks = split["config"]["tasks"]
    for task in tasks:
        for split_name, limit in (("train", 2), ("test", 2)):
            task_rows = [row for row in rows if row["config"] == task and row["split"] == split_name]
            selected.extend(task_rows[:limit])
    return selected


def assign_shards(rows: Sequence[dict[str, Any]], num_shards: int) -> list[dict[str, Any]]:
    validate_shard_args(num_shards, 0)
    assigned: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        item = dict(row)
        item["global_index"] = idx
        item["assigned_shard"] = idx % num_shards
        assigned.append(item)
    return assigned


def rows_for_shard(rows: Sequence[dict[str, Any]], num_shards: int, shard_id: int) -> list[dict[str, Any]]:
    validate_shard_args(num_shards, shard_id)
    assigned = assign_shards(rows, num_shards)
    return [row for row in assigned if int(row["assigned_shard"]) == shard_id]


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def disk_free_bytes(path: str | Path) -> int:
    path = Path(path)
    probe = path if path.exists() else path.parent
    usage = shutil.disk_usage(probe)
    return int(usage.free)


def estimate_raw_bytes(rows: Sequence[dict[str, Any]], include_v: bool = True) -> int:
    # Qwen3-8B: q=(36,32,T,128), k/v=(36,8,T,128), fp16.
    q_bpt = 36 * 32 * 128 * 2
    kv_bpt = 36 * 8 * 128 * 2
    per_token = q_bpt + kv_bpt + (kv_bpt if include_v else 0)
    return int(sum(int(row["prompt_length"]) for row in rows) * per_token)


def preflight_raw_storage(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    smoke: bool,
    keep_raw_mode: str | None = None,
    min_free_factor: float = 1.15,
) -> None:
    required = estimate_raw_bytes(rows, include_v=True)
    free = disk_free_bytes(path)
    # For full raw retention, require at least 3 TB regardless of split estimate to allow retry overhead.
    hard_min = 0 if smoke or (keep_raw_mode not in {None, "all"}) else 3_000_000_000_000
    threshold = max(int(required * min_free_factor), hard_min)
    if free < threshold:
        raise RuntimeError(
            f"Insufficient free space for raw Q/K/V capture at {path}: "
            f"free={free / 1e12:.2f} TB, required~{threshold / 1e12:.2f} TB "
            f"(estimate={required / 1e12:.2f} TB, smoke={smoke})."
        )


def validate_split(split: dict[str, Any]) -> None:
    rows = split.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Split manifest must contain a non-empty rows list")
    tasks = split.get("config", {}).get("tasks")
    if not tasks:
        raise ValueError("Split manifest config.tasks is missing")
    for task in tasks:
        train = [row for row in rows if row["config"] == task and row["split"] == "train"]
        test = [row for row in rows if row["config"] == task and row["split"] == "test"]
        train_ids = {int(row["row_index"]) for row in train}
        test_ids = {int(row["row_index"]) for row in test}
        if train_ids & test_ids:
            raise ValueError(f"Train/test overlap for {task}: {sorted(train_ids & test_ids)}")


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def summarize_values(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "mean": math.nan, "std": math.nan, "min": math.nan, "p10": math.nan, "p50": math.nan, "p90": math.nan, "max": math.nan}
    t = torch.tensor(list(values), dtype=torch.float64)
    return {
        "n": int(t.numel()),
        "mean": float(t.mean()),
        "std": float(t.std(unbiased=False)) if t.numel() > 1 else 0.0,
        "min": float(t.min()),
        "p10": float(torch.quantile(t, 0.10)),
        "p50": float(torch.quantile(t, 0.50)),
        "p90": float(torch.quantile(t, 0.90)),
        "max": float(t.max()),
    }


def write_stage_manifest(path: Path, *, stage: str, args: argparse.Namespace, rows: Sequence[dict[str, Any]], extra: dict[str, Any] | None = None) -> None:
    payload = {
        "stage": stage,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "num_rows": len(rows),
        "rows": list(rows),
    }
    if extra:
        payload.update(extra)
    write_json(path, payload)


def set_cuda_visible_devices(gpu: str | None) -> None:
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


class StatsCache:
    """In-process LRU cache for per-example stats `.pt` files.

    Analyzer trials repeatedly load overlapping subsets of the per-example stats files
    (~75 MB each); without caching, the analytic pass rereads the same file across many
    sample sizes / repetitions / regimes. This cache lifts that to in-memory dict access.

    Drops `cqk_sum` from cached payloads — calibration analysis no longer reads it (CCA
    methods were dropped), so retaining it would just waste ~25% of the per-entry RAM.

    Args:
        max_entries: cap on cached files. None = no cap (cache every file the analyzer
            asks for; bounded by the total unique-example count). Set to a finite value
            on memory-constrained workers.

    Hit-rate is exposed via `.stats()` for telemetry.
    """

    def __init__(self, max_entries: int | None = None) -> None:
        self._d: "OrderedDict[Path, dict[str, Any]]" = OrderedDict()
        self._max = max_entries
        self.hits = 0
        self.misses = 0

    def load(self, path: Path) -> dict[str, Any]:
        path = Path(path)
        cached = self._d.get(path)
        if cached is not None:
            self._d.move_to_end(path)
            self.hits += 1
            return cached
        self.misses += 1
        item = torch.load(path, map_location="cpu", weights_only=False)
        slim = {k: v for k, v in item.items() if k != "cqk_sum"}
        self._d[path] = slim
        if self._max is not None and len(self._d) > self._max:
            self._d.popitem(last=False)
        return slim

    def stats(self) -> dict[str, int]:
        total = self.hits + self.misses
        return {
            "entries": len(self._d),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else 0.0,
        }


class ProgressTracker:
    """Per-shard progress: structured log lines + machine-readable progress.json.

    On `start_example` log "starting" then `finish_example(extra_bytes=...)` log a structured line
    with this-example duration, rolling avg over the last `window`, ETA, throughput, cumulative MB.
    Also writes a progress.json snapshot after each finish so external monitors can roll up across
    shards without parsing log lines.

    Use `mark_skipped` when an example is resume-skipped — keeps counts accurate without logging
    a fake duration.
    """

    def __init__(
        self,
        *,
        label: str,
        total: int,
        progress_path: Path,
        window: int = 10,
    ) -> None:
        self.label = label
        self.total = int(total)
        self.progress_path = Path(progress_path)
        self.window = int(window)
        self.started_at = time.time()
        self.completed = 0
        self.skipped = 0
        self.recent_durations: list[float] = []
        self.bytes_by_kind: dict[str, int] = {}
        self.tokens_total: int = 0
        self.current_example: str | None = None
        self._example_start: float | None = None
        self._write_progress(last_duration=None)

    def start_example(self, example_id: str, *, tokens: int | None = None, idx_one_based: int | None = None) -> None:
        self.current_example = example_id
        self._example_start = time.time()
        idx_str = (
            f"{idx_one_based}/{self.total} ({100 * idx_one_based / max(self.total, 1):.0f}%)"
            if idx_one_based is not None
            else f"_/{self.total}"
        )
        tok_str = f" tokens={tokens}" if tokens is not None else ""
        log(f"[{self.label} | {idx_str}] capturing {example_id}{tok_str}")
        if tokens is not None:
            self.tokens_total += int(tokens)

    def finish_example(self, *, bytes_by_kind: dict[str, int] | None = None) -> None:
        if self._example_start is None:
            raise RuntimeError("finish_example called without matching start_example")
        duration = time.time() - self._example_start
        self.recent_durations.append(duration)
        if len(self.recent_durations) > self.window:
            self.recent_durations.pop(0)
        if bytes_by_kind:
            for k, v in bytes_by_kind.items():
                self.bytes_by_kind[k] = self.bytes_by_kind.get(k, 0) + int(v)
        self.completed += 1
        self._log_finish(duration, bytes_by_kind or {})
        self._write_progress(last_duration=duration)
        self._example_start = None

    def mark_skipped(self, example_id: str) -> None:
        self.completed += 1
        self.skipped += 1
        log(f"[{self.label} | {self.completed}/{self.total}] skip-existing {example_id}")
        self._write_progress(last_duration=None)

    def _log_finish(self, duration: float, bytes_by_kind: dict[str, int]) -> None:
        elapsed = time.time() - self.started_at
        avg = sum(self.recent_durations) / len(self.recent_durations) if self.recent_durations else duration
        remaining = max(0, self.total - self.completed)
        eta = avg * remaining
        rate_per_min = 60.0 * (self.completed - self.skipped) / elapsed if elapsed > 0 else 0.0
        bytes_str = " ".join(f"{k}={v / 1e6:.1f}MB" for k, v in sorted(bytes_by_kind.items()) if v > 0) or "—"
        cum_str = " ".join(f"{k}={v / 1e6:.0f}MB" for k, v in sorted(self.bytes_by_kind.items()) if v > 0) or "—"
        log(
            f"[{self.label} | {self.completed}/{self.total} ({100 * self.completed / max(self.total, 1):.0f}%)] "
            f"done in {fmt_duration(duration)} | avg {fmt_duration(avg)} | ETA {fmt_duration(eta)} "
            f"| rate {rate_per_min:.1f}/min | wrote {bytes_str} | total {cum_str}"
        )

    def _write_progress(self, last_duration: float | None) -> None:
        elapsed = time.time() - self.started_at
        avg = (
            sum(self.recent_durations) / len(self.recent_durations)
            if self.recent_durations
            else 0.0
        )
        remaining = max(0, self.total - self.completed)
        eta = avg * remaining
        payload = {
            "label": self.label,
            "total": self.total,
            "completed": self.completed,
            "skipped": self.skipped,
            "fraction": (self.completed / self.total) if self.total else 0.0,
            "current_example": self.current_example,
            "tokens_total": self.tokens_total,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started_at)),
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": elapsed,
            "elapsed_pretty": fmt_duration(elapsed),
            "avg_seconds_per_example": avg,
            "eta_seconds": eta,
            "eta_pretty": fmt_duration(eta),
            "rate_per_minute": (60.0 * (self.completed - self.skipped) / elapsed) if elapsed > 0 else 0.0,
            "last_duration_seconds": last_duration,
            "bytes_by_kind": dict(self.bytes_by_kind),
        }
        write_json(self.progress_path, payload)

    def finalize(self) -> None:
        elapsed = time.time() - self.started_at
        cum_str = " ".join(f"{k}={v / 1e6:.0f}MB" for k, v in sorted(self.bytes_by_kind.items()) if v > 0) or "—"
        log(
            f"[{self.label}] DONE | {self.completed}/{self.total} (skipped={self.skipped}) | "
            f"elapsed {fmt_duration(elapsed)} | total {cum_str}"
        )
        self._write_progress(last_duration=None)
