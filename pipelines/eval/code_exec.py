"""Self-contained sandbox for executing model-generated code against tests.

We don't depend on the `human_eval` or `livecodebench` packages (neither is
installed, and both pull heavy deps); instead each candidate program runs in a
fresh subprocess with a wall-clock timeout and RLIMIT caps, so a hang, an OOM,
or a crash in generated code can't take down the scorer. Two execution modes
cover both benchmarks:

- assert mode (HumanEval, LCB functional): the program ends in `assert`/`check`
  calls that raise on a wrong answer; pass == exit code 0.
- stdin mode (LCB stdin/stdout): the program is a script; we feed `stdin` and
  compare captured stdout to the expected output.

Only stdlib is used here. Generated solutions run under `sys.executable` (the
project venv), so they can import numpy/typing etc. like the reference harnesses.
"""
from __future__ import annotations

import resource
import subprocess
import sys
import tempfile
from pathlib import Path

# Per-process caps for the sandboxed child. Memory cap is generous because some
# HumanEval/LCB solutions build large lists; the wall-clock timeout is the real
# guard against runaway loops.
_MEM_LIMIT_BYTES = 4 * 1024 * 1024 * 1024  # 4 GB address space
_CPU_LIMIT_SECONDS = 30  # hard CPU cap; wall-clock timeout is usually tighter


def _set_limits():
    # Runs in the child between fork and exec. Best-effort: on systems where a
    # limit can't be lowered we skip it rather than fail the whole run.
    for res, val in (
        (resource.RLIMIT_AS, _MEM_LIMIT_BYTES),
        (resource.RLIMIT_CPU, _CPU_LIMIT_SECONDS),
    ):
        try:
            resource.setrlimit(res, (val, val))
        except (ValueError, OSError):
            pass


def _run(program: str, stdin: str | None, timeout: float) -> subprocess.CompletedProcess | None:
    # Write to a temp file rather than `-c` so tracebacks carry real line
    # numbers and very long programs don't hit ARG_MAX. Returns None on timeout.
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as fh:
        fh.write(program)
        fh.flush()
        try:
            return subprocess.run(
                [sys.executable, fh.name],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=_set_limits,
            )
        except subprocess.TimeoutExpired:
            return None


def exec_asserts(program: str, timeout: float = 8.0) -> dict:
    """Run an assert-style program. passed == clean exit (code 0)."""
    proc = _run(program, stdin=None, timeout=timeout)
    if proc is None:
        return {"passed": False, "reason": "timeout"}
    if proc.returncode == 0:
        return {"passed": True, "reason": "ok"}
    return {"passed": False, "reason": (proc.stderr or "").strip()[-500:] or f"exit={proc.returncode}"}


def exec_stdin(program: str, stdin: str, expected: str, timeout: float = 8.0) -> dict:
    """Run a script with `stdin`; passed == stdout matches `expected`.

    Comparison is line-wise after stripping trailing whitespace on each line and
    trailing blank lines, which is how LCB's own checker tolerates formatting.
    """
    proc = _run(program, stdin=stdin, timeout=timeout)
    if proc is None:
        return {"passed": False, "reason": "timeout"}
    if proc.returncode != 0:
        return {"passed": False, "reason": (proc.stderr or "").strip()[-500:] or f"exit={proc.returncode}"}
    if _norm_stdout(proc.stdout) == _norm_stdout(expected):
        return {"passed": True, "reason": "ok"}
    return {"passed": False, "reason": "wrong_output"}


def _norm_stdout(s: str) -> list[str]:
    lines = [ln.rstrip() for ln in (s or "").replace("\r\n", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return lines
