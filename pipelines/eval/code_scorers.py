"""Code-execution scorers for HumanEval and LiveCodeBench v6.

Registry scorers (aime25, math500, ...) live in Amir's read-only kvq tree; these
two benchmarks have no scorer there and the tree is off-limits to edit, so the
client (`run_prompts_client.py`) imports these directly via a local override.

Interface matches the registry scorers: `calculate_metrics(df) -> dict` reading
`predicted_answer` plus per-row fields carried from the exported rows. Execution
goes through the subprocess sandbox in `code_exec.py`.
"""
from __future__ import annotations

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

# Import the sibling sandbox by its own directory rather than the `pipelines.eval`
# package path: Amir's read-only tree (on PYTHONPATH) ships a regular `pipelines`
# package that shadows ours, so the dotted path can't resolve when the client
# runs under his env.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from code_exec import exec_asserts, exec_stdin  # noqa: E402

# Score rows concurrently: each candidate spawns a subprocess, so this is
# process/IO-bound and threads overlap the waits.
_MAX_WORKERS = 16


def extract_code(text: str) -> str:
    """Pull the solution code from a chat completion.

    Qwen3 thinking traces are wrapped in <think>...</think>; drop everything up
    to the last </think>. Then take the last fenced code block (```python or
    bare ```); if there's no fence, return the post-think text as-is (some
    completions emit raw code).
    """
    if text is None:
        return ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if blocks:
        return blocks[-1]
    # No fence: some completions emit bare code, possibly after a prose lead-in
    # ("Here is the solution:\n\ndef ..."). Slice from the first top-level code
    # token so the prose doesn't turn into a SyntaxError.
    m = re.search(r"(?m)^(def |class |import |from |@)", text)
    return text[m.start():] if m else text


def _score_parallel(items, run_one):
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        return list(ex.map(run_one, items))


# ---------------------------------------------------------------------------
# HumanEval
# ---------------------------------------------------------------------------

def _humaneval_one(row) -> bool:
    code = extract_code(row["predicted_answer"])
    entry = row["entry_point"]
    # Instruct models usually redefine the full function; if the extracted code
    # doesn't define the entry point, fall back to completing the raw stub.
    if f"def {entry}" not in code:
        code = row["he_stub"] + "\n" + code
    program = code + "\n" + row["he_test"] + f"\ncheck({entry})\n"
    return exec_asserts(program, timeout=8.0)["passed"]


def humaneval_scorer(df: pd.DataFrame) -> dict:
    rows = [r for _, r in df.iterrows()]
    passed = _score_parallel(rows, _humaneval_one)
    correct = int(sum(passed))
    return {"correct": correct, "accuracy": correct / len(df), "total": len(df)}


# ---------------------------------------------------------------------------
# LiveCodeBench v6
# ---------------------------------------------------------------------------
# Two problem formats (LCB's own convention):
#   - functional: LeetCode-style. Code defines `class Solution` with a method
#     `fn_name`; each test input is newline-joined json args, output is the
#     json-encoded expected return. Driver calls `Solution().fn_name(*args)`.
#   - stdin: the program reads stdin and prints; compare stdout to expected.
# Per-row fields: `lcb_tests` (JSON list of {input, output, testtype}),
# `lcb_fn_name` (str or "").

# LCB solutions assume these are in scope without importing them (their harness
# injects the same set).
_PREAMBLE = (
    "from typing import *\n"
    "import collections, math, heapq, bisect, itertools, functools, re, string, sys\n"
    "from collections import *\n"
    "from math import *\n"
)


def _lcb_functional_program(code: str, fn_name: str, tests) -> str:
    payload = json.dumps([{"input": t.get("input", ""), "output": t.get("output", "")}
                          for t in tests])
    driver = f"""
import json as _json
_cases = _json.loads({json.dumps(payload)})
_sol = Solution()
for _c in _cases:
    _args = [_json.loads(_l) for _l in _c["input"].split("\\n") if _l.strip() != ""]
    _exp = _json.loads(_c["output"])
    _got = _sol.{fn_name}(*_args)
    assert _got == _exp, f"got {{_got!r}} exp {{_exp!r}}"
"""
    return _PREAMBLE + "\n" + code + "\n" + driver


# task_id -> tests, loaded once per version tag (decoding all v6 shards is slow).
_LCB_TESTS_CACHE = {}


def _lcb_tests(version_tag, task_id):
    if version_tag not in _LCB_TESTS_CACHE:
        from export_code_rows import load_lcb_tests  # sibling; our dir is on sys.path
        _LCB_TESTS_CACHE[version_tag] = load_lcb_tests(version_tag)
    return _LCB_TESTS_CACHE[version_tag].get(task_id)


def _lcb_one(row) -> bool:
    code = extract_code(row["predicted_answer"])
    fn_name = row.get("lcb_fn_name") or ""
    tests = row.get("lcb_tests")  # legacy embedded rows; else reload by task_id
    tests = json.loads(tests) if isinstance(tests, str) else \
        _lcb_tests(row.get("lcb_version", "release_v6"), row["task_id"])
    if not tests:
        return False
    if fn_name:
        # All functional tests in one subprocess (loops internally).
        prog = _lcb_functional_program(code, fn_name, tests)
        return exec_asserts(prog, timeout=12.0)["passed"]
    # stdin/stdout: one subprocess per test (each needs its own stdin feed).
    prog = _PREAMBLE + "\n" + code
    for t in tests:
        if not exec_stdin(prog, stdin=t.get("input", ""),
                          expected=t.get("output", ""), timeout=8.0)["passed"]:
            return False  # all-or-nothing per problem, as LCB scores it
    return True


def livecodebench_scorer(df: pd.DataFrame) -> dict:
    rows = [r for _, r in df.iterrows()]
    # Warm the test cache single-threaded (decoding shards is slow) before the
    # worker pool starts, unless rows carry embedded tests.
    if "lcb_tests" not in df.columns and len(df):
        _lcb_tests(rows[0].get("lcb_version", "release_v6"), rows[0]["task_id"])
    passed = _score_parallel(rows, _lcb_one)
    correct = int(sum(passed))
    return {"correct": correct, "accuracy": correct / len(df), "total": len(df)}


CODE_SCORERS = {
    "humaneval": humaneval_scorer,
    "livecodebench": livecodebench_scorer,
}
