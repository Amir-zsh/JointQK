#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipelines.calibration.common import (
    ProgressTracker,
    RunPaths,
    add_common_args,
    add_keep_raw_arg,
    add_shard_args,
    copy_split_manifest,
    log,
    messages_hash,
    preflight_raw_storage,
    prompt_hash,
    raw_example_path,
    read_json,
    rows_for_shard,
    select_rows,
    set_cuda_visible_devices,
    should_keep_raw,
    stable_example_id,
    stats_example_path,
    tokens_scale_from_sample,
    validate_shard_args,
    validate_split,
    write_stage_manifest,
)
from pipelines.calibration.compute_stats import (
    build_stats_artifact,
    compute_moments_from_tensors,
)
from kvq.data.base import fetch_example
from kvq.data.kvpress_adapter import build_kvpress_dataset_spec
from kvq.toolkit.capture import capture_rope_qk
from kvq.toolkit.model import get_model_device, load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture prefill q_post/k_post/v and compute per-example second-moment stats inline."
    )
    add_common_args(parser)
    add_shard_args(parser)
    add_keep_raw_arg(parser)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset", default="longbench")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--gpu", default=None, help="Optional single GPU id for this worker.")
    return parser.parse_args()


def _assemble_per_layer(chunks: list[torch.Tensor], n_layers: int) -> torch.Tensor:
    if len(chunks) % n_layers != 0:
        raise RuntimeError(f"RoPE hook fired {len(chunks)} times, not a multiple of n_layers={n_layers}")
    per_layer = [torch.cat(chunks[layer::n_layers], dim=2) for layer in range(n_layers)]
    return torch.stack(per_layer, dim=0).squeeze(1)


def _extract_cache_kv(cache: Any, n_layers: int) -> tuple[torch.Tensor, torch.Tensor]:
    keys = []
    values = []
    for layer in range(n_layers):
        layer_keys = cache.layers[layer].keys.detach().to("cpu", dtype=torch.float16).squeeze(0)
        layer_values = cache.layers[layer].values.detach().to("cpu", dtype=torch.float16).squeeze(0)
        keys.append(layer_keys)
        values.append(layer_values)
    return torch.stack(keys, dim=0), torch.stack(values, dim=0)


@torch.inference_mode()
def run_prefill_qkv_capture(model: torch.nn.Module, input_ids: torch.Tensor) -> dict[str, torch.Tensor | int]:
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"Expected input_ids shape (1, L); got {tuple(input_ids.shape)}")
    device = get_model_device(model)
    input_ids = input_ids.to(device)
    prompt_length = int(input_ids.shape[-1])
    n_layers = int(model.config.num_hidden_layers)

    with capture_rope_qk(model) as (_q_pre_chunks, q_post_chunks, _k_pre_chunks, _k_post_chunks):
        outputs = model(input_ids=input_ids, use_cache=True)

    q_post = _assemble_per_layer(q_post_chunks, n_layers)
    k_post, v = _extract_cache_kv(outputs.past_key_values, n_layers)
    for name, tensor in (("q_post", q_post), ("k_post", k_post), ("v", v)):
        if int(tensor.shape[2]) != prompt_length:
            raise RuntimeError(f"{name} seq length {tensor.shape[2]} != prompt length {prompt_length}")
        if tensor.dtype != torch.float16:
            raise RuntimeError(f"{name} dtype {tensor.dtype} != fp16")
    return {
        "q_post": q_post,
        "k_post": k_post,
        "v": v,
        "prompt_length": prompt_length,
        "total_length": prompt_length,
        "captured_length": prompt_length,
        "generated_token_ids": torch.empty(0, dtype=torch.long),
    }


def validate_existing_raw(path: Path, row: dict[str, Any]) -> bool:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["config"] != row["config"] or int(payload["row_index"]) != int(row["row_index"]):
            return False
        for name in ("q_post", "k_post", "v"):
            tensor = payload[name]
            if tensor.dtype != torch.float16 or int(tensor.shape[2]) != int(payload["prompt_length"]):
                return False
        return True
    except Exception:
        return False


def validate_existing_stats(path: Path, row: dict[str, Any]) -> bool:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["config"] != row["config"] or int(payload["row_index"]) != int(row["row_index"]):
            return False
        for name in ("sq_sum", "sk_sum", "cqk_sum", "sv_sum", "sum_q", "sum_k", "sum_v"):
            if name not in payload or not torch.isfinite(payload[name]).all():
                return False
        return True
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    validate_shard_args(args.num_shards, args.shard_id)
    set_cuda_visible_devices(args.gpu)
    paths = RunPaths.from_args(args.artifact_root, args.run_id)
    paths.ensure()

    # Run-root model guard. The first capture invocation under a fresh run-id
    # writes _run_meta.json with the model name; subsequent invocations refuse
    # to write into a directory captured against a *different* model. This
    # prevents accidentally clobbering an existing model's calibration when
    # the agent forgets to pass --run-id <model>-tagged-id.
    run_meta_path = paths.root / "_run_meta.json"
    if run_meta_path.exists():
        existing = read_json(run_meta_path)
        if existing.get("model") != args.model:
            raise RuntimeError(
                f"refusing to capture into run-id={args.run_id!r} (root {paths.root}): "
                f"existing _run_meta.json has model={existing.get('model')!r}, "
                f"current --model={args.model!r}. Use a distinct --run-id for the "
                f"new model (e.g. {args.run_id}_<model_tag>) or delete the run-root "
                f"to repurpose it."
            )
    else:
        from pipelines.calibration.common import write_json as _write_json
        _write_json(run_meta_path, {
            "model": args.model,
            "run_id": args.run_id,
            "created_by": "capture_raw.py",
        })

    split_manifest = copy_split_manifest(args.split_manifest, paths)
    split = read_json(split_manifest)
    validate_split(split)
    all_rows = select_rows(split, args.smoke)
    rows = rows_for_shard(all_rows, args.num_shards, args.shard_id)
    raw_shard_dir = paths.raw_dir / f"shard_{args.shard_id:03d}"
    stats_shard_dir = paths.stats_dir / f"shard_{args.shard_id:03d}"
    raw_shard_dir.mkdir(parents=True, exist_ok=True)
    stats_shard_dir.mkdir(parents=True, exist_ok=True)

    rows_keeping_raw = [r for r in rows if should_keep_raw(r, args.keep_raw)]
    rows_dropping_raw = len(rows) - len(rows_keeping_raw)

    # Load tokenizer + model config first (cheap, no weights) so the disk preflight
    # can use the actual capture model's shape and an empirical tokens-per-prompt
    # scale factor. This avoids over-estimating disk by ~30% on Llama against a
    # Qwen-tokenized manifest.
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model_shape = {
        "n_layers": int(config.num_hidden_layers),
        "n_q_heads": int(config.num_attention_heads),
        "n_kv_heads": int(getattr(config, "num_key_value_heads", config.num_attention_heads)),
        "head_dim": int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)),
    }
    spec = build_kvpress_dataset_spec(
        name=args.dataset,
        config_names=tuple(split["config"]["tasks"]),
        metadata_fields=("task", "answers", "length", "all_classes"),
    )

    # Empirically derive the manifest→capture-tokenizer scale (1.0 if same model).
    def _fetch(row):
        return fetch_example(spec, row["config"], int(row["row_index"]), tokenizer)
    tokens_scale = tokens_scale_from_sample(rows_keeping_raw or rows, _fetch, n_samples=5)
    log(f"model shape: {model_shape} | tokens_scale (capture/manifest): {tokens_scale:.3f}")

    if args.keep_raw != "none":
        preflight_raw_storage(
            paths.raw_dir,
            rows_keeping_raw,
            smoke=args.smoke,
            keep_raw_mode=args.keep_raw,
            model_shape=model_shape,
            tokens_scale=tokens_scale,
        )
    log(
        f"shard {args.shard_id}/{args.num_shards} | examples={len(rows)} "
        f"| keep-raw mode={args.keep_raw} (raw written for {len(rows_keeping_raw)}, dropped for {rows_dropping_raw})"
    )
    log(f"loading model={args.model} device_map={args.device_map} dtype={args.dtype}")
    model, _ = load_model_and_tokenizer(args.model, device_map=args.device_map, dtype_name=args.dtype)
    device = get_model_device(model)
    log(f"model loaded on {device}; starting capture")

    progress = ProgressTracker(
        label=f"capture shard {args.shard_id}/{args.num_shards}",
        total=len(rows),
        progress_path=raw_shard_dir / "progress.json",
    )

    completed: list[dict[str, Any]] = []
    t_all = time.time()
    for local_idx, row in enumerate(rows):
        row = dict(row)
        row["assigned_shard"] = args.shard_id
        keep_raw_for_row = should_keep_raw(row, args.keep_raw)
        raw_path = raw_example_path(paths, row)
        stats_path = stats_example_path(paths, row)

        # Resume: stats must exist + validate; raw conditionally on keep policy.
        stats_ok = stats_path.exists() and validate_existing_stats(stats_path, row)
        raw_ok = (not keep_raw_for_row) or (raw_path.exists() and validate_existing_raw(raw_path, row))
        if args.resume and stats_ok and raw_ok:
            progress.mark_skipped(stable_example_id(row))
            completed.append({
                **row,
                "stats_file": str(stats_path.relative_to(paths.root)),
                "raw_file": (str(raw_path.relative_to(paths.root)) if keep_raw_for_row and raw_path.exists() else None),
            })
            continue

        # Fetch example FIRST so we can pass the actual capture-time prompt_length
        # to the progress tracker. The manifest's prompt_length is tokenizer-locked
        # to whichever model created the split (typically Qwen3) and would be wrong
        # by ~30% under Llama.
        example = fetch_example(spec, row["config"], int(row["row_index"]), tokenizer)
        progress.start_example(
            stable_example_id(row),
            tokens=int(example.prompt_length),
            idx_one_based=local_idx + 1,
        )
        phash = prompt_hash(example.prompt_text)
        # Hash the canonical messages list (tokenizer-independent), not the
        # rendered prompt_text, so the same manifest works across models with
        # different chat templates (Qwen3 / Llama-3.1 / etc.). Falls back to
        # prompt_sha256 for legacy manifests that don't have messages_sha256.
        mhash = messages_hash(example.messages)
        if "messages_sha256" in row:
            if mhash != row["messages_sha256"]:
                raise RuntimeError(
                    f"Messages hash mismatch for {stable_example_id(row)}: "
                    f"manifest={row['messages_sha256']} fetched={mhash} "
                    f"(messages content drifted; the LongBench dataset row may have changed upstream)"
                )
        elif "prompt_sha256" in row and phash != row["prompt_sha256"]:
            raise RuntimeError(
                f"Prompt hash mismatch for {stable_example_id(row)}: "
                f"manifest={row['prompt_sha256']} fetched={phash} "
                f"(legacy prompt_sha256 check; this is tokenizer-locked. Re-generate "
                f"the manifest with messages_sha256 to support cross-model captures.)"
            )
        captured = run_prefill_qkv_capture(model, example.input_ids)

        # Compute moments inline while q/k/v are still in memory — fused stats stage.
        moments = compute_moments_from_tensors(
            captured["q_post"],
            captured["k_post"],
            captured["v"],
            prompt_length=int(captured["prompt_length"]),
            device=device,
        )
        stats_artifact = build_stats_artifact(moments, row, prompt_sha256=phash)
        stats_artifact["messages_sha256"] = mhash
        stats_tmp = stats_path.with_suffix(stats_path.suffix + f".tmp.{args.shard_id}.{local_idx}")
        torch.save(stats_artifact, stats_tmp)
        stats_tmp.replace(stats_path)
        stats_bytes = stats_path.stat().st_size

        raw_bytes = 0
        if keep_raw_for_row:
            raw_artifact = {
                "q_post": captured["q_post"],
                "k_post": captured["k_post"],
                "v": captured["v"],
                "prompt_length": int(captured["prompt_length"]),
                "total_length": int(captured["total_length"]),
                "captured_length": int(captured["captured_length"]),
                "generated_token_ids": captured["generated_token_ids"],
                "dataset": row["dataset"],
                "config": row["config"],
                "task": row["task"],
                "split": row["split"],
                "row_index": int(row["row_index"]),
                "prompt_sha256": phash,
                "messages_sha256": mhash,
                "metadata": example.metadata,
                "collection": "final_calibration_prefill_qpost_kpost_v",
            }
            raw_tmp = raw_path.with_suffix(raw_path.suffix + f".tmp.{args.shard_id}.{local_idx}")
            torch.save(raw_artifact, raw_tmp)
            raw_tmp.replace(raw_path)
            raw_bytes = raw_path.stat().st_size
            del raw_artifact

        completed.append({
            **row,
            "stats_file": str(stats_path.relative_to(paths.root)),
            "raw_file": (str(raw_path.relative_to(paths.root)) if keep_raw_for_row else None),
        })
        progress.finish_example(bytes_by_kind={"raw": raw_bytes, "stats": stats_bytes})

        del captured, moments, stats_artifact, example
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    progress.finalize()
    write_stage_manifest(
        raw_shard_dir / "manifest.json",
        stage="raw_capture",
        args=args,
        rows=completed,
        extra={"elapsed_seconds": time.time() - t_all, "keep_raw_mode": args.keep_raw},
    )
    write_stage_manifest(
        stats_shard_dir / "manifest.json",
        stage="compute_stats_inline",
        args=args,
        rows=completed,
        extra={"elapsed_seconds": time.time() - t_all},
    )


if __name__ == "__main__":
    main()
