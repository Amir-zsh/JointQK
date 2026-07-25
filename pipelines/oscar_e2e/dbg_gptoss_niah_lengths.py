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

NEEDLE_RE = re.compile(
    r"One of the special magic numbers for [\w-]+ is: \d+\.?", re.I
)


def build(row, target_tokens):
    """Rebuild the row's prompt at ~target_tokens, needle and question intact."""
    p = row["prompt"]
    m = NEEDLE_RE.search(p)
    if not m:
        return None, None
    needle = m.group(0)
    number = re.search(r"(\d+)", needle).group(1)
    # Preamble = everything before the haystack body; tail = the question.
    head = p[: min(400, m.start())]
    tail = p[-400:]
    filler_budget = max(0, target_tokens * 4 - len(head) - len(needle) - len(tail))
    filler = p[m.end() : m.end() + filler_budget]
    return head + needle + " " + filler + tail, number


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--rows", default="artifacts/prompt_rows/niah_8192_gptoss_t1.jsonl")
    ap.add_argument("--lens", default="128,256,512,1024,2048,4096,8192")
    ap.add_argument("--n", type=int, default=5, help="rows per length")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.rows) if l.strip()]
    rows = [r for r in rows if NEEDLE_RE.search(r["prompt"])][: args.n]

    for tgt in [int(x) for x in args.lens.split(",")]:
        hits = capped = 0
        toks = []
        first_out = None
        for r in rows:
            prompt, number = build(r, tgt)
            if prompt is None:
                continue
            body = {
                "text": prompt,
                "sampling_params": {"temperature": 0.0, "max_new_tokens": 160},
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
        n = len(rows)
        avg = sum(toks) / len(toks) if toks else 0
        print(
            f"[{args.mode}] target~{tgt:>5} prompt_tokens={first_out[1]} "
            f"needle_found={hits}/{n} cap={capped}/{n} mean_gen={avg:.0f}",
            flush=True,
        )
        print(f"      {first_out[0][:150]!r}", flush=True)


if __name__ == "__main__":
    main()
