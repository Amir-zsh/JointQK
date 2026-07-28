#!/usr/bin/env python3
"""Export the exact evaluation rows + final prompt strings a bench cell uses.

Reproduces vendor/kvpress evaluate.py's row selection byte-for-byte (dataset
registry, exclude_indices_file drop, fraction sampling at the same seed) and
its prompt assembly (chat template applied to user content = context +
question via the separator-split trick, + answer_prefix), so an external
serving stack (the OSCAR SGLang server) can score IDENTICAL rows with
IDENTICAL inputs -> row-paired comparisons against our harness cells.

Output JSONL per row: rid (post-prep dataframe position), all dataset columns
except context, and `prompt` (the final untemplated-input text; feed to a raw
/generate endpoint, NOT a chat endpoint) + `max_new_tokens`.

    .venv/bin/python pipelines/eval/export_prompt_rows.py \
        --model Qwen/Qwen3-8B --dataset ruler --data-dir 32768 \
        --out artifacts/prompt_rows/ruler_32768_qwen.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "vendor" / "kvpress"))

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from kvq.benchmarks.evaluate_registry import DATASET_REGISTRY  # noqa: E402


def prepare_df(dataset, data_dir, fraction, seed, exclude_indices_file):
    if dataset == "gpqa":
        from kvq.benchmarks.gpqa_adapter import load_gpqa_df
        # data_dir carries the variant (default diamond); row selection is
        # whole-set, so fraction/exclude machinery below still applies.
        df = load_gpqa_df(variant=data_dir or "diamond")
    else:
        df = load_dataset(DATASET_REGISTRY[dataset], data_dir=data_dir or None,
                          split="test").to_pandas()
    if exclude_indices_file:
        exclude_map = json.loads(Path(exclude_indices_file).read_text())
        task_key = data_dir if data_dir is not None else dataset
        drop = set(int(i) for i in exclude_map.get(task_key, []))
        if drop:
            df = df.loc[~df.index.isin(drop)].reset_index(drop=True)
    if fraction < 1.0:
        df = df.sample(frac=fraction, random_state=seed)
    return df


def build_prompt(tokenizer, context, question, answer_prefix,
                 enable_thinking=False, model=""):
    separator = "#" * (len(context) + 10)
    templ = tokenizer.apply_chat_template(
        [{"role": "user", "content": context + separator}],
        add_generation_prompt=True, tokenize=False,
        enable_thinking=enable_thinking)
    ctx_part, question_suffix = templ.split(separator)
    # gpt-oss's harmony format requires every message to declare a channel
    # (its own system prompt states this) but apply_chat_template's
    # generation prompt is a bare "<|start|>assistant" with no channel tag.
    # A non-empty answer_prefix glued directly onto that is malformed input:
    # the model burns tokens self-recovering the channel structure before it
    # can answer, which on tight budgets (e.g. NIAH's 128) truncates before
    # it ever emits a formatted response -- go straight to the final channel.
    # Gated on answer_prefix specifically: tasks with an EMPTY answer_prefix
    # (math500, aime25 -- open-ended reasoning, no forced continuation) don't
    # have this failure mode (verified: math500 was already 0% garbage with
    # NO fix, large token budget lets the model self-recover into its own
    # analysis-then-final flow naturally) and forcing straight-to-final would
    # cut off the analysis phase a reasoning task needs. Don't apply there.
    if "gpt-oss" in model.lower() and answer_prefix:
        question_suffix += "<|channel|>final<|message|>"
    return ctx_part + (question or "") + question_suffix + (answer_prefix or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-indices-file", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="override; default = per-row dataset value")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="open the <think> block in the chat template "
                         "(Qwen3 thinking mode) for long-horizon tasks")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    df = prepare_df(args.dataset, args.data_dir, args.fraction, args.seed,
                    args.exclude_indices_file)

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as fh:
        for rid, row in df.iterrows():
            prompt = build_prompt(tok, row["context"], row.get("question", ""),
                                  row.get("answer_prefix", ""),
                                  enable_thinking=args.enable_thinking,
                                  model=args.model)
            rec = {k: v for k, v in row.items() if k != "context"}
            for k, v in list(rec.items()):
                if hasattr(v, "tolist"):
                    rec[k] = v.tolist()
            mnt = args.max_new_tokens or int(row.get("max_new_tokens", 128))
            rec.update({"rid": int(rid), "prompt": prompt,
                        "max_new_tokens": mnt,
                        "dataset": args.dataset, "data_dir": args.data_dir})
            fh.write(json.dumps(rec) + "\n")
            n += 1
    print(f"[export] wrote {n} rows -> {out}")


if __name__ == "__main__":
    main()
