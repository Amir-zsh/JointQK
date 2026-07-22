#!/usr/bin/env python3
"""Generate the full RULER NIAH suite (8 subtasks) at arbitrary context lengths (incl. 128K),
in the schema the vendored ruler scorer + worker consume. simonjegou/ruler caps at 16K; this
reproduces RULER-NIAH at 32K/64K/128K for the OSCAR Table-3 comparison. Haystacks (PG essays +
noise) are extracted from the real simonjegou/ruler data (artifacts/niah_corpus/), so distractor
structure matches RULER — the reason BF16 itself degrades at long ctx (single-needle can't show that).

Subtasks (RULER definitions, verified against simonjegou/ruler):
  single_1  noise haystack,  1 number needle
  single_2  essay haystack,  1 number needle
  single_3  essay haystack,  1 uuid   needle
  multikey_1 essay haystack, 1 target + 3 distractor number needles
  multikey_2 needle haystack (filled w/ distractor number needles) + 1 target
  multikey_3 needle haystack (uuid distractors) + 1 target uuid
  multivalue essay haystack, 1 key with 4 values (retrieve all)
  multiquery essay haystack, 4 keys queried (retrieve all)
"""
import argparse, random, uuid, re
from pathlib import Path
from transformers import AutoTokenizer
from datasets import Dataset

ADJ = ("solid abashed unsightly grumpy tiny ancient rough abaft lazy efficient stale calm bright "
       "fast deep red blue green gold silver long short new old happy quiet brave clever eager fancy "
       "gentle jolly kind lively proud silly witty zany bold crisp dark early fair giant huge icy").split()
NOUN = ("few geometry patty cornerstone summary orchard blueberry daily pursuit government river moon "
        "track lake fox hill tea coin key road dawn tree idea alluvium melody canyon harbor meadow "
        "anchor beacon cipher domain ember fable galaxy harvest island jungle kernel lantern marble").split()
CORPUS = Path("artifacts/niah_corpus")
ESSAY = CORPUS.joinpath("essays.txt").read_text()
NOISE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.\n"

CFG = {  # haystack, value-type, n_keys, n_values(per key), n_queried
    "niah_single_1":   ("noise", "num", 1, 1, 1),
    "niah_single_2":   ("essay", "num", 1, 1, 1),
    "niah_single_3":   ("essay", "uuid", 1, 1, 1),
    "niah_multikey_1": ("essay", "num", 4, 1, 1),
    "niah_multikey_2": ("needle", "num", 1, 1, 1),
    "niah_multikey_3": ("needle", "uuid", 1, 1, 1),
    "niah_multivalue": ("essay", "num", 1, 4, 1),
    "niah_multiquery": ("essay", "num", 4, 1, 4),
}


def rval(vt, rng):
    return str(uuid.UUID(int=rng.getrandbits(128))) if vt == "uuid" else str(rng.randint(1_000_000, 9_999_999))


def rkey(rng, used):
    while True:
        k = f"{rng.choice(ADJ)}-{rng.choice(NOUN)}"
        if k not in used:
            used.add(k); return k


def needle(key, val, vt):
    return f"One of the special magic {'uuids' if vt=='uuid' else 'numbers'} for {key} is: {val}.\n"


def build(tok, ctx, task, rng, essay_ids, noise_ids):
    hs, vt, n_k, n_v, n_q = CFG[task]
    word = "uuid" if vt == "uuid" else "number"
    used = set()
    keys = [rkey(rng, used) for _ in range(n_k)]
    # target key(s): multiquery queries all n_k; others query the first key
    q_keys = keys if n_q > 1 else keys[:1]
    # values: multivalue gives n_v values to the (single) queried key
    kv = {}                                   # key -> list of values (target needles)
    for k in keys:
        kv[k] = [rval(vt, rng) for _ in range(n_v if k in q_keys else 1)]
    answers = [v for k in q_keys for v in kv[k]]
    target_needles = [needle(k, v, vt) for k in keys for v in kv[k]]

    # haystack budget
    intro = (f"Some special magic {word}s are hidden within the following text. Make sure to memorize "
             f"them. I will quiz you about the {word}s afterwards.\n" if n_q > 1 or n_v > 1 else
             f"A special magic {word} is hidden within the following text. Make sure to memorize it. "
             f"I will quiz you about the {word} afterwards.\n")
    overhead = len(tok(intro + "".join(target_needles))["input_ids"]) + 40
    budget = max(ctx - overhead, 0)

    if hs == "needle":                        # haystack = distractor needles (pre-size, no O(n^2))
        per = max(len(tok(needle("aa-bb", rval(vt, rng), vt))["input_ids"]), 1)
        tgt = set(keys)                        # distractors need only differ from the target key(s),
        def dkey():                            # not be globally unique (key space is only ~1.8k combos)
            while True:
                k = f"{rng.choice(ADJ)}-{rng.choice(NOUN)}"
                if k not in tgt:
                    return k
        parts = [needle(dkey(), rval(vt, rng), vt) for _ in range(budget // per + 8)]
        filler = tok.decode(tok("".join(parts))["input_ids"][:budget])
    else:                                     # slice pre-tokenized corpus token ids (O(budget))
        base = noise_ids if hs == "noise" else essay_ids
        reps = budget // max(len(base), 1) + 2
        filler = tok.decode((base * reps)[:budget])

    # insert target needles at random depths
    for nd in target_needles:
        cut = rng.randint(0, len(filler))
        filler = filler[:cut] + nd + filler[cut:]
    context = intro + filler + "\n"

    kq = ", ".join(q_keys[:-1]) + (", and " + q_keys[-1] if len(q_keys) > 1 else q_keys[0])
    plural = "s" if (n_q > 1 or n_v > 1) else ""
    question = f"What {'are all' if plural else 'is'} the special magic {word}{plural} for {kq} mentioned in the provided text? "
    prefix = f"The special magic {word}{plural} for {kq} mentioned in the provided text {'are' if plural else 'is'}"
    return context, question, prefix, answers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--n", type=int, default=50, help="samples PER subtask")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    rng = random.Random(args.seed)
    essay_ids = tok(ESSAY)["input_ids"]          # tokenize corpora ONCE
    noise_ids = tok(NOISE * 200)["input_ids"]
    rows = []
    for task in CFG:
        for i in range(args.n):
            ctx, q, pref, ans = build(tok, args.ctx, task, rng, essay_ids, noise_ids)
            rows.append(dict(_id=f"{task}_{args.ctx}_{args.seed}_{i}", context=ctx, question=q,
                             answer_prefix=pref, answer=ans, task=task, max_new_tokens=128))
    ds = Dataset.from_list(rows)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(args.out)
    ntok = [len(tok(r["context"])["input_ids"]) for r in rng.sample(rows, 5)]
    print(f"SAVED {args.out} | n={len(rows)} ({len(CFG)} tasks x {args.n}) | ctx~{args.ctx} | sample tokens={ntok}", flush=True)


if __name__ == "__main__":
    main()
