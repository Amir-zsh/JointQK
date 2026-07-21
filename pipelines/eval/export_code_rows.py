#!/usr/bin/env python3
"""Export HumanEval / LiveCodeBench-v6 rows for the served eval, mirroring
export_prompt_rows.py's output schema (one JSONL row: rid, prompt, max_new_tokens,
dataset, plus scorer fields) so run_prompts_client.py consumes them unchanged.

Code tasks don't use the context+question kvpress prompt trick, so this is a
separate exporter: the whole problem is one user turn, chat-templated (thinking
mode optional), and the fields the code scorers need are carried on the row.

    python pipelines/eval/export_code_rows.py --model Qwen/Qwen3-8B \
        --dataset humaneval --enable-thinking \
        --out artifacts/prompt_rows/humaneval_qwen.jsonl

    python pipelines/eval/export_code_rows.py --model Qwen/Qwen3-8B \
        --dataset livecodebench --version-tag release_v6 --enable-thinking \
        --out artifacts/prompt_rows/lcb_v6_qwen.jsonl
"""
from __future__ import annotations

import argparse
import base64
import json
import pickle
import zlib
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

# Exact instruction + system message from OpenAI simple-evals' HumanEval (the
# harness OSCAR's eval stack is built on; their runner wraps simple-evals and
# defaults the system message to "You are a helpful assistant."). Matching the
# wording is the closest we can get to their prompt -- the OSCAR paper does not
# publish the HumanEval/LCB prompts, and their code-task clients were separate
# and are not vendored here.
HUMANEVAL_SYSTEM = "You are a helpful assistant."
HUMANEVAL_INSTRUCTION = (
    "Read the following function signature and docstring, and fully implement "
    "the function described. Your response should only contain the code for "
    "this function.\n"
)

# LCB's own generation prompt (system framing + per-format instruction).
LCB_SYSTEM = (
    "You are an expert Python programmer. You will be given a question (problem "
    "specification) and will generate a correct Python program that matches the "
    "specification and passes all tests."
)
LCB_STDIN_INSTR = (
    "Read the inputs from stdin, solve the problem, and write the answer to "
    "stdout (do not directly test on the sample inputs). Enclose your code "
    "within delimiters as follows:\n```python\n# YOUR CODE HERE\n```"
)
LCB_FUNCTIONAL_INSTR = (
    "You will use the following starter code to write the solution to the "
    "problem and enclose your code within delimiters.\n```python\n{starter}\n```"
)


def _decode_private(s: str):
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(s.encode("utf-8")))))


def build_humaneval_rows(tok, enable_thinking, max_new_tokens):
    ds = load_dataset("openai_humaneval", split="test")
    rows = []
    for i, ex in enumerate(ds):
        user = HUMANEVAL_INSTRUCTION + ex["prompt"]
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": HUMANEVAL_SYSTEM},
             {"role": "user", "content": user}],
            add_generation_prompt=True, tokenize=False,
            enable_thinking=enable_thinking)
        rows.append({
            "rid": i,
            "task_id": ex["task_id"],
            "entry_point": ex["entry_point"],
            "he_stub": ex["prompt"],
            "he_test": ex["test"],
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "dataset": "humaneval",
        })
    return rows


def _load_lcb_shards(version_tag):
    n = int(version_tag.split("_v")[-1])  # release_v6 -> 6
    shards = ["test.jsonl"] + [f"test{k}.jsonl" for k in range(2, n + 1)]
    recs = []
    for name in shards:
        p = hf_hub_download("livecodebench/code_generation_lite", name, repo_type="dataset")
        with open(p) as f:
            recs.extend(json.loads(line) for line in f)
    return recs


def build_lcb_rows(tok, enable_thinking, max_new_tokens, version_tag):
    recs = _load_lcb_shards(version_tag)
    rows = []
    for i, ex in enumerate(recs):
        meta = json.loads(ex.get("metadata") or "{}")
        fn_name = meta.get("func_name", "")
        starter = ex.get("starter_code", "") or ""
        instr = (LCB_FUNCTIONAL_INSTR.format(starter=starter) if fn_name
                 else LCB_STDIN_INSTR)
        user = f"### Question:\n{ex['question_content']}\n\n### Instructions:\n{instr}"
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": LCB_SYSTEM},
             {"role": "user", "content": user}],
            add_generation_prompt=True, tokenize=False,
            enable_thinking=enable_thinking)
        # Tests are NOT embedded in the row: 28k cases across v6 would balloon
        # predictions.csv (and again per seed). The scorer reloads them by
        # task_id from the same version tag (see code_scorers.load_lcb_tests).
        rows.append({
            "rid": i,
            "task_id": ex["question_id"],
            "lcb_fn_name": fn_name,
            "lcb_version": version_tag,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "dataset": "livecodebench",
        })
    return rows


def load_lcb_tests(version_tag):
    """{question_id: [ {input, output, testtype}, ... ]} for a v-tag. Used by
    the scorer so rows/predictions stay small."""
    out = {}
    for ex in _load_lcb_shards(version_tag):
        out[ex["question_id"]] = (json.loads(ex["public_test_cases"])
                                  + _decode_private(ex["private_test_cases"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, choices=["humaneval", "livecodebench"])
    ap.add_argument("--version-tag", default="release_v6", help="LCB only")
    ap.add_argument("--max-new-tokens", type=int, default=32768)
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.dataset == "humaneval":
        rows = build_humaneval_rows(tok, args.enable_thinking, args.max_new_tokens)
    else:
        rows = build_lcb_rows(tok, args.enable_thinking, args.max_new_tokens, args.version_tag)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"[export] wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
