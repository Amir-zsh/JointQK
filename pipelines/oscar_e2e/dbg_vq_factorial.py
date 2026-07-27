#!/usr/bin/env python3
"""Separate VQ-K loss, INT2-V loss, and runtime codebook resnapping.

This is an engine-free GPT-OSS control. It performs one prefill with plain
HuggingFace, modifies only the middle KV band once, and then greedily decodes.
The four primary arms are a 2x2 factorial:

    bf16                 exact K, exact V
    vq_k                 deployed VQ K, exact V
    int2_v               exact K, deployed scalar-INT2 V
    vq_k_int2_v          deployed VQ K and scalar-INT2 V

``vq_k_source`` is diagnostic: it uses the centroids stored in the artifact
before the SGLang loader's native-E5M2 resnap. It is not a deployable arm, but
measures whether the artifact/loader format boundary materially changes output.
With ``--raw-bundle``, ``vq_k_raw`` instead applies the runtime E5M2 snap once
to the unprocessed training artifact.

Only the initial middle band is quantized. The default generation length is
below HP_RECENT, so generated tokens do not age into the quantized tier and no
token is quantized twice.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
NEEDLE_RE = re.compile(r"One of the special magic numbers for [\w-]+ is: \d+\.?", re.I)
INSTRUCTION_END = "I will quiz you about the number afterwards."
QUESTION_START = "\nWhat is the special magic number for "
HP_PREFIX = 64
HP_RECENT = 256
PRIMARY_ARMS = ("bf16", "vq_k", "int2_v", "vq_k_int2_v")
DIAGNOSTIC_ARMS = ("vq_k_source", "vq_k_raw")


def _clean_prefix(text: str, budget: int) -> str:
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text.strip()
    return text[:budget].rsplit(maxsplit=1)[0].strip()


def _build_prompt(row: dict, target_tokens: int) -> tuple[str, str]:
    prompt = row["prompt"]
    match = NEEDLE_RE.search(prompt)
    if match is None:
        raise ValueError("row does not contain a recognized needle")
    needle = match.group(0)
    number = re.search(r"(\d+)", needle).group(1)

    instruction_end = prompt.find(INSTRUCTION_END)
    question_start = prompt.rfind(QUESTION_START)
    if instruction_end < 0 or question_start <= match.end():
        raise ValueError("prompt does not match the expected NIAH task template")
    body_start = instruction_end + len(INSTRUCTION_END)

    preamble = prompt[:body_start].rstrip()
    tail = prompt[question_start:].lstrip()
    before = prompt[body_start : match.start()].strip()
    after = prompt[match.end() : question_start].strip()
    fixed_chars = len(preamble) + len(needle) + len(tail) + 4
    filler_budget = max(0, target_tokens * 4 - fixed_chars)
    total_filler = len(before) + len(after)
    before_budget = (
        round(filler_budget * len(before) / total_filler) if total_filler else 0
    )
    after_budget = filler_budget - before_budget
    short_before = _clean_prefix(before, before_budget)
    short_after = _clean_prefix(after, after_budget)

    return (
        "\n".join(
            part for part in (preamble, short_before, needle, short_after, tail) if part
        ),
        number,
    )


def _quant_dequant_int2(
    x: torch.Tensor, groups: int = 1, clip_ratio: float = 0.0
) -> torch.Tensor:
    """Per-token affine INT2 used by the existing engine-free control."""
    tokens, heads, dim = x.shape
    if dim % groups:
        raise ValueError(f"head dim {dim} is not divisible by {groups} groups")
    values = x.float()
    if clip_ratio > 0:
        clip_index = min(int(clip_ratio * dim), dim - 1)
        threshold = values.abs().sort(dim=-1).values[..., clip_index : clip_index + 1]
        values = values.clamp(-threshold, threshold)
    grouped = values.reshape(tokens, heads, groups, dim // groups)
    vmin = grouped.amin(-1, keepdim=True)
    vmax = grouped.amax(-1, keepdim=True)
    scale = (vmax - vmin).clamp_min(1e-8) / 3.0
    zero = -vmin / scale
    quant = torch.clamp(grouped / scale + zero + 0.5, 0, 3).floor()
    return ((quant - zero) * scale).reshape(tokens, heads, dim)


class LayerVQ:
    def __init__(self, blob: dict, layer: int, device: torch.device):
        self.forward = blob["forward"][layer].to(device, torch.bfloat16)
        self.inverse = blob["inverse"][layer].to(device, torch.bfloat16)
        self.mean = blob["mean"][layer].to(device, torch.bfloat16)
        self.bounds = blob["bounds"]
        self.pertoken_norm = bool(blob.get("pertoken_norm", False))

        source = torch.stack(
            [
                torch.stack(
                    [c.to(torch.float16) for c in blob["codebooks"][(layer, h)]]
                )
                for h in range(self.forward.shape[0])
            ]
        ).to(device)
        self.source_cb = source
        self.runtime_cb = source.to(torch.float8_e5m2).to(torch.float16)
        self.runtime_sq = 0.5 * self.runtime_cb.float().pow(2).sum(-1)
        self.source_sq = 0.5 * self.source_cb.float().pow(2).sum(-1)

    def roundtrip(
        self, keys: torch.Tensor, *, runtime_resnap: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kd = keys.to(torch.bfloat16)
        residual = torch.einsum(
            "thd,hde->the", kd - self.mean.unsqueeze(0), self.forward
        ).contiguous()
        rf = residual.float()
        if self.pertoken_norm:
            scale = rf.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
            normalized = (rf / scale).to(torch.float16)
        else:
            scale = torch.ones(
                (*rf.shape[:2], 1), device=rf.device, dtype=torch.float32
            )
            normalized = rf.to(torch.float16)

        cb = self.runtime_cb if runtime_resnap else self.source_cb
        cb_sq = self.runtime_sq if runtime_resnap else self.source_sq
        heads, groups, codewords, group_dim = cb.shape
        grouped = normalized.view(normalized.shape[0], heads, groups, group_dim)
        scores = torch.einsum("thgc,hgkc->thgk", grouped, cb).float()
        indices = (scores - cb_sq.unsqueeze(0)).argmax(-1)
        h_ids = torch.arange(heads, device=keys.device).view(1, heads, 1)
        g_ids = torch.arange(groups, device=keys.device).view(1, 1, groups)
        reconstructed = cb[h_ids, g_ids, indices].reshape_as(residual).float()
        reconstructed *= scale
        original_space = torch.einsum(
            "thd,hde->the",
            reconstructed.to(torch.bfloat16),
            self.inverse,
        ) + self.mean.unsqueeze(0)
        return original_space.to(keys.dtype), indices


def _artifact_summary(blob: dict) -> dict:
    source = torch.cat(
        [c.float().reshape(-1) for groups in blob["codebooks"].values() for c in groups]
    )
    runtime = source.to(torch.float8_e5m2).float()
    rel = (
        (runtime - source).pow(2).sum() / source.pow(2).sum().clamp_min(1e-30)
    ).sqrt()
    return {
        "declared_fp8_fmt": blob.get("fp8_fmt"),
        "runtime_resnap_rel_rmse": float(rel),
        "runtime_resnap_changed_fraction": float((runtime != source).float().mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/gpt-oss-20b-BF16")
    parser.add_argument(
        "--bundle",
        default="artifacts/oscar_gptoss20b/"
        "vqa_gptoss20b_G4_strat_flat_ptn_gpqacc128k_fp8.pt",
    )
    parser.add_argument(
        "--raw-bundle",
        help="optional unprocessed training bundle for a single-snap vq_k_raw arm",
    )
    parser.add_argument("--rot", default="artifacts/oscar_gptoss20b/rotations_gpqa198")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument(
        "--rows-file",
        default="artifacts/prompt_rows/niah_8192_gptoss_t1.jsonl",
    )
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--gen", type=int, default=96)
    parser.add_argument("--int2-groups", type=int, default=1)
    parser.add_argument("--v-clip-ratio", type=float, default=0.92)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=list(PRIMARY_ARMS) + ["vq_k_source"],
        choices=list(PRIMARY_ARMS) + list(DIAGNOSTIC_ARMS),
    )
    parser.add_argument("--out", default="artifacts/oscar_e2e/vq_factorial.json")
    parser.add_argument(
        "--artifact-only",
        action="store_true",
        help="print the artifact/loader resnap diagnostic without loading a model",
    )
    args = parser.parse_args()
    if args.gen > HP_RECENT:
        parser.error(
            f"--gen must be <= {HP_RECENT}; longer runs need decode-aging simulation"
        )

    bundle_path = REPO / args.bundle
    blob = torch.load(bundle_path, map_location="cpu", weights_only=False)
    raw_blob = (
        torch.load(REPO / args.raw_bundle, map_location="cpu", weights_only=False)
        if args.raw_bundle
        else None
    )
    if "vq_k_raw" in args.arms and raw_blob is None:
        parser.error("the vq_k_raw arm requires --raw-bundle")
    if args.artifact_only:
        print(json.dumps(_artifact_summary(blob), indent=2))
        return 0

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    from transformers import AutoModelForCausalLM, AutoTokenizer

    layer_map_path = REPO / args.rot / "layer_map.json"
    local_to_global = json.loads(layer_map_path.read_text())["local_to_global"]
    value_rotations = torch.load(
        REPO / args.rot / "v_rotation_sst_r_h_pbr.pt",
        map_location="cpu",
        weights_only=False,
    )["layers"]

    rows = [
        json.loads(line)
        for line in (REPO / args.rows_file).read_text().splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if NEEDLE_RE.search(row["prompt"])][: args.rows]
    if not rows:
        raise RuntimeError("no usable needle rows found")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    def run_arm(prompt: str, arm: str) -> tuple[str, dict]:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        stats = {"k_rel_rmse": [], "v_rel_rmse": [], "assignment_flip": []}
        with torch.inference_mode():
            output = model(input_ids=ids, use_cache=True)
            cache = output.past_key_values
            lo, hi = HP_PREFIX, max(HP_PREFIX, ids.shape[1] - HP_RECENT)
            use_vq = arm in {
                "vq_k",
                "vq_k_int2_v",
                "vq_k_source",
                "vq_k_raw",
            }
            use_int2_v = arm in {"int2_v", "vq_k_int2_v"}
            runtime_resnap = arm != "vq_k_source"
            codec_blob = raw_blob if arm == "vq_k_raw" else blob
            if hi > lo and (use_vq or use_int2_v):
                for local, global_layer in enumerate(local_to_global):
                    layer = cache.layers[global_layer]
                    if use_vq:
                        keys = layer.keys[0, :, lo:hi].permute(1, 0, 2)
                        codec = LayerVQ(codec_blob, local, keys.device)
                        reconstructed, runtime_idx = codec.roundtrip(
                            keys, runtime_resnap=runtime_resnap
                        )
                        if runtime_resnap:
                            _, source_idx = codec.roundtrip(keys, runtime_resnap=False)
                            stats["assignment_flip"].append(
                                float((runtime_idx != source_idx).float().mean())
                            )
                        stats["k_rel_rmse"].append(
                            float(
                                (reconstructed.float() - keys.float()).norm()
                                / keys.float().norm().clamp_min(1e-12)
                            )
                        )
                        layer.keys[0, :, lo:hi] = reconstructed.permute(1, 0, 2)
                    if use_int2_v:
                        values = layer.values[0, :, lo:hi].permute(1, 0, 2)
                        rotation = value_rotations[local]["rotation"].to(
                            values.device, torch.float32
                        )
                        reconstructed_v = (
                            _quant_dequant_int2(
                                values.float() @ rotation,
                                args.int2_groups,
                                args.v_clip_ratio,
                            )
                            @ rotation.T
                        ).to(values.dtype)
                        stats["v_rel_rmse"].append(
                            float(
                                (reconstructed_v.float() - values.float()).norm()
                                / values.float().norm().clamp_min(1e-12)
                            )
                        )
                        layer.values[0, :, lo:hi] = reconstructed_v.permute(1, 0, 2)

            generated = []
            for _ in range(args.gen):
                next_token = output.logits[:, -1:].argmax(-1)
                generated.append(next_token.item())
                output = model(
                    input_ids=next_token,
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = output.past_key_values
        aggregate = {
            key: (sum(values) / len(values) if values else None)
            for key, values in stats.items()
        }
        return tokenizer.decode(generated), aggregate

    result = {
        "model": args.model,
        "bundle": args.bundle,
        "artifact": _artifact_summary(blob),
        "raw_artifact": (_artifact_summary(raw_blob) if raw_blob is not None else None),
        "hp_prefix": HP_PREFIX,
        "hp_recent": HP_RECENT,
        "tokens": args.tokens,
        "gen": args.gen,
        "int2_groups": args.int2_groups,
        "v_clip_ratio": args.v_clip_ratio,
        "arms": {},
    }
    for arm in args.arms:
        hits = 0
        samples = []
        metrics = []
        for row in rows:
            prompt, number = _build_prompt(row, args.tokens)
            text, arm_metrics = run_arm(prompt, arm)
            hit = number in text
            hits += int(hit)
            samples.append({"needle": number, "hit": hit, "text": text})
            metrics.append(arm_metrics)
        metric_mean = {}
        for key in metrics[0]:
            values = [item[key] for item in metrics if item[key] is not None]
            metric_mean[key] = sum(values) / len(values) if values else None
        result["arms"][arm] = {
            "hits": hits,
            "rows": len(rows),
            "metrics": metric_mean,
            "samples": samples,
        }
        print(
            f"[{arm}] needle {hits}/{len(rows)} metrics={metric_mean} "
            f"first={samples[0]['text'][:160]!r}",
            flush=True,
        )

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
