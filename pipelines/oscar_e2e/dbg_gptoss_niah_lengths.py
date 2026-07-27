"""Valid length sweep: real NIAH task, shortened haystack, scored by needle match.

The earlier raw-text probe was uncontrolled — a truncated haystack plus an
unrelated question makes gpt-oss ramble in bf16 exactly as it does in int2, so
it could not attribute anything. This version keeps the task intact: the
instruction preamble, the needle sentence, and the trailing question are all
preserved; only filler is removed to hit a target length. Success is an exact
match on the magic number, so "degenerate" and "wrong" are distinguishable
from "correct".

The point of the sweep is the mixed-KV band boundary. With hp_prefix=64 and
hp_recent=256, a prompt under ~320 tokens is served entirely from the bf16 HP
band (quant=0) — if int2 fails there, the defect is not in the quantizer.
Above it the int2 tier carries a growing share of the context.

Usage (server already running):
  python dbg_gptoss_niah_lengths.py --port 30941 --mode int2
"""

import argparse
import json
import re
import urllib.request

NEEDLE_RE = re.compile(r"One of the special magic numbers for [\w-]+ is: \d+\.?", re.I)
INSTRUCTION_END = "I will quiz you about the number afterwards."
QUESTION_START = "\nWhat is the special magic number for "


def _clean_prefix(text, budget):
    """Take at most budget characters without ending inside a word."""
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text.strip()
    return text[:budget].rsplit(maxsplit=1)[0].strip()


def build(row, target_tokens):
    """Shorten only the haystack while preserving the complete task template."""
    p = row["prompt"]
    m = NEEDLE_RE.search(p)
    if not m:
        return None, None
    needle = m.group(0)
    number = re.search(r"(\d+)", needle).group(1)

    instruction_end = p.find(INSTRUCTION_END)
    question_start = p.rfind(QUESTION_START)
    if instruction_end < 0 or question_start <= m.end():
        raise ValueError("prompt does not match the expected NIAH task template")
    body_start = instruction_end + len(INSTRUCTION_END)

    preamble = p[:body_start].rstrip()
    tail = p[question_start:].lstrip()
    before = p[body_start : m.start()].strip()
    after = p[m.end() : question_start].strip()

    # Keep the needle at approximately its original relative depth. Restrict
    # shortening to the filler body; never cut the instruction, question,
    # assistant prefix, or a word.
    fixed_chars = len(preamble) + len(needle) + len(tail) + 4
    filler_budget = max(0, target_tokens * 4 - fixed_chars)
    total_filler = len(before) + len(after)
    before_budget = (
        round(filler_budget * len(before) / total_filler) if total_filler else 0
    )
    after_budget = filler_budget - before_budget
    short_before = _clean_prefix(before, before_budget)
    short_after = _clean_prefix(after, after_budget)

    prompt = "\n".join(
        part for part in (preamble, short_before, needle, short_after, tail) if part
    )
    return prompt, number


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--rows", default="artifacts/prompt_rows/niah_8192_gptoss_t1.jsonl")
    ap.add_argument(
        "--long-rows",
        default="artifacts/prompt_rows/niah_16384_gptoss_t1.jsonl",
        help="source rows for targets above 8192 tokens",
    )
    ap.add_argument("--lens", default="128,256,512,1024,2048,4096,8192,16384")
    ap.add_argument("--n", type=int, default=5, help="rows per length")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    args = ap.parse_args()

    row_cache = {}

    def load_rows(path):
        if path not in row_cache:
            with open(path) as fh:
                rows = [json.loads(l) for l in fh if l.strip()]
            row_cache[path] = [r for r in rows if NEEDLE_RE.search(r["prompt"])][
                : args.n
            ]
        return row_cache[path]

    for tgt in [int(x) for x in args.lens.split(",")]:
        row_path = args.long_rows if tgt > 8192 else args.rows
        rows = load_rows(row_path)
        hits = capped = 0
        toks = []
        first_out = None
        for r in rows:
            prompt, number = build(r, tgt)
            if prompt is None:
                continue
            for _ in range(args.samples):
                body = {
                    "text": prompt,
                    "sampling_params": {
                        "temperature": args.temperature,
                        "top_p": 1.0,
                        "top_k": -1,
                        "max_new_tokens": args.max_new_tokens,
                    },
                }
                req = urllib.request.Request(
                    f"http://127.0.0.1:{args.port}/generate",
                    json.dumps(body).encode(),
                    {"Content-Type": "application/json"},
                )
                d = json.loads(urllib.request.urlopen(req, timeout=600).read())
                out, meta = d["text"], d.get("meta_info", {})
                hits += number in out
                fr = meta.get("finish_reason")
                capped += int(isinstance(fr, dict) and fr.get("type") == "length")
                toks.append(meta.get("completion_tokens", 0))
                if first_out is None:
                    first_out = (out, meta.get("prompt_tokens"))
        n = len(rows) * args.samples
        avg = sum(toks) / len(toks) if toks else 0
        print(
            f"[{args.mode}] target~{tgt:>5} prompt_tokens={first_out[1]} "
            f"needle_found={hits}/{n} cap={capped}/{n} mean_gen={avg:.0f}",
            flush=True,
        )
        print(f"      {first_out[0][:150]!r}", flush=True)


if __name__ == "__main__":
    main()
