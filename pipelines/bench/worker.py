#!/usr/bin/env python3
"""Phase 7 persistent-worker dispatcher.

One Python process per (GPU, slot) loads a target model once via kvpress's
EvaluationRunner, then consumes (press_name, press_kwargs, task) work items
from a shared queue and runs each in-process. Avoids the per-job model-load
hit of the bash-subprocess launcher when many configs share a model.

Work items file format: one JSON object per line, matching the keys of
EvaluationConfig that we override. Lines starting with '#' are ignored.
Required keys: press_name, dataset, data_dir, output_dir.
Optional: press_kwargs, fraction, compression_ratio, max_new_tokens, ...

Usage:
    worker.py \
        --model 'Qwen/Qwen3-8B' \
        --commands-file <jsonl> \
        --log-dir <dir> \
        --gpus 0,1,2,3,4,5 \
        --jobs-per-gpu 2

All workers in one invocation share the same model; for multi-model sweeps
launch one worker.py per model with disjoint --gpus pools.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _bootstrap  # noqa: E402, F401

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
KVPRESS_EVAL = REPO_ROOT / "vendor" / "kvpress" / "evaluation"
for p in (str(REPO_ROOT), str(REPO_ROOT / "vendor" / "kvpress"), str(KVPRESS_EVAL)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _canonical_results_subdir(config) -> str:
    """Mirrors `EvaluationConfig.get_results_dir`'s component logic but without
    the "if exists, append numbered subdir" behaviour. Used to detect already-
    completed cells so reruns of the same command file skip them instead of
    spawning a fresh `1/`, `2/`, ... subdir and re-running inference.
    """
    components = [
        config.dataset,
        str(config.data_dir) if config.data_dir else "",
        config.model.replace("/", "--"),
        config.press_name,
        f"{config.compression_ratio:.2f}",
    ]
    if config.threshold is not None:
        components[-1] = f"{config.threshold:.2f}"
    if config.fraction < 1.0:
        components.append(f"fraction{config.fraction:.3f}")
    if config.max_context_length is not None:
        components.append(f"max_context{config.max_context_length}")
    if config.query_aware:
        components.append("query_aware")
    if config.key_channel_compression_ratio is not None:
        components.append(f"key_channel_cr{config.key_channel_compression_ratio:.2f}")
    if config.needle_depth is not None and config.dataset == "needle_in_haystack":
        components.append(f"needle_depth{config.needle_depth}")
    return "__".join(filter(None, components))


def gpu_persistent_worker(gpu: int, slot: int, model_name: str,
                           in_queue: mp.Queue, out_queue: mp.Queue, log_dir: str):
    """Worker: load model once on `gpu`, then drain (job_idx, label, config_dict) items."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # Imports happen inside the worker so CUDA initialises against the masked GPU.
    from evaluate import EvaluationConfig, EvaluationRunner

    log_dir_p = Path(log_dir)
    init_log = log_dir_p / f"_worker_init_gpu{gpu}s{slot}.log"

    # Bootstrap: load model + tokenizer once via a no_press sentinel runner.
    bootstrap_t0 = time.time()
    sentinel = EvaluationConfig(
        model=model_name,
        press_name="no_press",
        compression_ratio=0.0,
        dataset="longbench",
        data_dir="samsum",  # any valid task; not actually evaluated
        output_dir=str(log_dir_p / f"_sentinel_gpu{gpu}s{slot}"),
        fraction=1.0,
    )
    cached = EvaluationRunner(sentinel)
    cached._setup_press()
    cached._setup_model_pipeline()
    bootstrap_dt = time.time() - bootstrap_t0
    init_log.write_text(
        f"# worker GPU {gpu} slot {slot}\n"
        f"# model={model_name}\n"
        f"# bootstrap_seconds={bootstrap_dt:.1f}\n"
    )

    # Cache press instances by (press_name, kwargs hash). JointQK's
    # post_init_from_model takes ~100s to build 256+256 calibrated compressors;
    # by reusing the same press object across cells with identical kwargs we
    # pay this cost once per (config-class), not once per (config × task).
    press_cache: dict = {}

    def _press_key(press_name: str, press_kwargs):
        # press_kwargs values are JSON-friendly; sort to make the key stable.
        if press_kwargs is None:
            return (press_name, ())
        return (press_name, tuple(sorted(press_kwargs.items())))

    while True:
        item = in_queue.get()
        if item is None:
            break
        job_idx, label, cfg_dict, attempt = item
        job_log = log_dir_p / f"job_{job_idx:03d}_gpu{gpu}s{slot}_{label}_a{attempt}.log"
        start = time.time()
        rc = 0
        is_oom = False
        err_msg = ""
        with job_log.open("w") as f:
            f.write(f"# job {job_idx} | GPU {gpu} slot {slot} | label {label} | "
                    f"attempt {attempt} | start {time.ctime(start)}\n")
            f.write(f"# config: {json.dumps(cfg_dict)}\n\n")
            f.flush()
            try:
                config = EvaluationConfig(model=model_name, **cfg_dict)
                runner = EvaluationRunner(config)
                # Reuse the cached pipeline (and its tokenizer / model) instead of reloading.
                runner.pipeline = cached.pipeline

                # Reuse a previously-built press instance if we've seen this
                # (press_name, kwargs) before. Otherwise let _setup_press() build
                # a fresh one and cache it. The press's own post_init_from_model
                # is now idempotent (jointqk_press.py), so subsequent calls into
                # it through kvpress's context manager are near-instant.
                key = _press_key(config.press_name, config.press_kwargs)
                cached_press = press_cache.get(key)
                if cached_press is not None:
                    runner.press = cached_press
                    f.write(f"press cache hit for {key[0]}\n")
                else:
                    runner._setup_press()
                    press_cache[key] = runner.press
                    f.write(f"press cache miss → built fresh {key[0]}\n")

                output_dir = runner._setup_directories()
                # Check the canonical (un-numbered) results dir BEFORE calling
                # config.get_results_dir, which would otherwise spawn a fresh
                # `<dir>/1/` subdir whenever the canonical one already exists
                # (and we'd then run inference again, missing the skip).
                canonical_dir = output_dir / _canonical_results_subdir(config)
                if (canonical_dir / "predictions.csv").exists() and (canonical_dir / "metrics.json").exists():
                    f.write(f"existing results found at {canonical_dir} — skipping\n")
                else:
                    results_dir = config.get_results_dir(output_dir)
                    predictions = results_dir / "predictions.csv"
                    metrics = results_dir / "metrics.json"
                    cfg_yaml = results_dir / "config.yaml"
                    runner._load_and_prepare_dataset()
                    runner._run_inference()
                    runner._save_results(predictions)
                    runner._calculate_and_save_metrics(metrics)
                    config.save_config(cfg_yaml)
                    f.write(f"wrote {predictions}\n")
                    f.write(f"wrote {metrics}\n")
            except Exception as e:
                err_msg = str(e)
                tb = traceback.format_exc()
                f.write(f"FAIL: {err_msg}\n{tb}\n")
                rc = 1
                # Detect CUDA OOM via either torch's typed exception or
                # message string (older torch / cuBLAS surface as RuntimeError).
                lower = err_msg.lower() + tb.lower()
                if ("out of memory" in lower or "cuda oom" in lower
                        or "cudaerrormemoryallocation" in lower):
                    is_oom = True
                # Also recover the press cache key — if the failure happened
                # mid-inference the cached press might be in a bad state. Drop
                # it so a retry rebuilds fresh.
                try:
                    if 'key' in locals():
                        press_cache.pop(key, None)
                except Exception:
                    pass
            finally:
                # Free press-specific tensors before next item.
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        out_queue.put((job_idx, gpu, slot, rc, time.time() - start, str(job_log),
                       attempt, is_oom, err_msg[:200]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="HuggingFace model id; one model shared by all workers")
    p.add_argument("--commands-file", required=True,
                   help="JSONL: one config dict per line. # comments allowed. "
                        "Optional trailing field 'label' for log-filename suffix; "
                        "default label derives from press_name + data_dir.")
    p.add_argument("--log-dir", required=True)
    p.add_argument("--gpus", default="0,1,2,3,4,5")
    p.add_argument("--jobs-per-gpu", type=int, default=1)
    p.add_argument("--max-retries", type=int, default=1,
                   help="On OOM, requeue up to this many times. Other failures are not retried.")
    args = p.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    gpus = [int(g) for g in args.gpus.split(",")]
    n_workers = len(gpus) * args.jobs_per_gpu

    items: list[tuple[str, dict]] = []
    for line in Path(args.commands_file).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cfg = json.loads(line)
        label = cfg.pop("_label", None) or f"{cfg.get('press_name', 'press')}_{cfg.get('data_dir', '')}"
        # Sanitise label for filename use.
        label = label.replace("/", "_").replace(" ", "_")
        items.append((label, cfg))

    overview = log_dir / "_overview.log"
    overview.write_text(
        f"# {len(items)} jobs across GPUs {gpus} (jobs-per-gpu={args.jobs_per_gpu}, "
        f"workers={n_workers}) | model={args.model} | start {time.ctime()}\n\n"
    )

    in_q: mp.Queue = mp.Queue()
    out_q: mp.Queue = mp.Queue()
    for i, (label, cfg) in enumerate(items):
        in_q.put((i, label, cfg, 0))  # attempt 0

    workers = []
    for gpu in gpus:
        for slot in range(args.jobs_per_gpu):
            w = mp.Process(
                target=gpu_persistent_worker,
                args=(gpu, slot, args.model, in_q, out_q, str(log_dir)),
            )
            w.start()
            workers.append(w)

    pending = len(items)        # job-completion accounting (decremented on terminal outcome)
    fail = 0
    retries = 0
    while pending > 0:
        job_idx, gpu, slot, rc, elapsed, log_file, attempt, is_oom, err = out_q.get()
        if rc == 0:
            status = "OK  "
            note = ""
        elif is_oom and attempt < args.max_retries:
            status = "OOM "
            note = f" -> requeue (attempt {attempt + 1})"
            # Look up the original cfg for this job_idx to requeue.
            label, cfg = items[job_idx]
            in_q.put((job_idx, label, cfg, attempt + 1))
            retries += 1
        else:
            status = "FAIL" if not is_oom else "OOM!"
            note = f" (attempt {attempt}, err: {err})" if err else ""
        line = (
            f"[{time.strftime('%H:%M:%S')}] job {job_idx:03d} GPU {gpu}s{slot} {status} "
            f"{elapsed:6.0f}s {Path(log_file).name}{note}"
        )
        print(line, flush=True)
        with overview.open("a") as f:
            f.write(line + "\n")
        if rc == 0 or not (is_oom and attempt < args.max_retries):
            # Terminal outcome: count down pending.
            pending -= 1
            if rc != 0:
                fail += 1

    # All jobs accounted for; drain workers.
    for _ in range(n_workers):
        in_q.put(None)
    for w in workers:
        w.join()

    summary_line = (
        f"\n# done: {len(items) - fail}/{len(items)} succeeded, "
        f"{fail} failed, {retries} retries"
    )
    print(summary_line, flush=True)
    with overview.open("a") as f:
        f.write(summary_line + "\n")

    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
