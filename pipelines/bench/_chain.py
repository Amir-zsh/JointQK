#!/usr/bin/env python3
"""Wait for Phase 6 (12 OK in overview) done, run aggregate_decode_scope.py,
then launch Phase 7 LongBench (Qwen first, then Llama)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _bootstrap  # noqa: E402, F401

import datetime as dt
import re
import subprocess
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
LOG = REPO / "logs/phase7_chain.log"
LOG.parent.mkdir(parents=True, exist_ok=True)


def log(msg):
    line = f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}\n"
    LOG.open("a").write(line)
    print(line, end="", flush=True)


def count_oks(overview):
    if not Path(overview).exists():
        return 0
    return len(re.findall(r"\bOK\b", Path(overview).read_text()))


def main():
    LOG.write_text("")
    log("waiting for Phase 6 (12 OK)")

    while True:
        n = count_oks(REPO / "logs/phase6/_overview.log")
        log(f"phase6_OKs={n}/12")
        if n >= 12:
            log("Phase 6 done; aggregating")
            break
        time.sleep(60)

    # Read existing decode-scope decision (manually locked to WINNER=A)
    decision_file = REPO / "artifacts/downstream/qwen3_8b/decode_scope/decode_decision.txt"
    if decision_file.exists():
        log(f"decode decision (preserved): {decision_file.read_text().strip()}")
    else:
        log("running aggregate_decode_scope (no pre-existing decision)")
        subprocess.run([
            "bash", "-c",
            "source ~/miniconda3/etc/profile.d/conda.sh && conda activate efficient-llm && "
            "python -u -m pipelines.eval.aggregate_decode_scope "
            "--input-dir artifacts/downstream/qwen3_8b/decode_scope "
            "--output-decision artifacts/downstream/qwen3_8b/decode_scope/decode_decision.txt"
        ], cwd=REPO)

    # Launch Phase 7 LongBench Qwen
    log("launching Phase 7 LongBench Qwen3-8B")
    subprocess.run([
        "bash", str(REPO / "pipelines/bench/launch_longbench.sh"),
        "--gpus", "0,1,2,3,4,5",
        "--model", "qwen3_8b",
        "--fraction", "0.5",
    ], cwd=REPO)

    log("Phase 7 LongBench Qwen done; launching Llama")
    subprocess.run([
        "bash", str(REPO / "pipelines/bench/launch_longbench.sh"),
        "--gpus", "0,1,2,3,4,5",
        "--model", "llama31_8b",
        "--fraction", "0.5",
    ], cwd=REPO)

    log("Phase 7 LongBench all done; launching RULER")
    subprocess.run([
        "bash", str(REPO / "pipelines/bench/launch_ruler.sh"),
        "--gpus", "0,1,2,3,4,5",
        "--ks", "4",
        "--ctxs", "4096,8192,16384",
    ], cwd=REPO)

    log("Phase 7 RULER done; running aggregators")
    for cmd in [
        "python -u -m pipelines.eval.aggregate_longbench "
        "--qwen-dir artifacts/downstream/qwen3_8b "
        "--llama-dir artifacts/downstream/llama31_8b "
        "--output artifacts/downstream/longbench_summary.json",
        "python -u -m pipelines.eval.aggregate_ruler "
        "--qwen-dir artifacts/downstream/qwen3_8b "
        "--llama-dir artifacts/downstream/llama31_8b "
        "--output artifacts/downstream/ruler_summary.json",
    ]:
        full = (
            "source ~/miniconda3/etc/profile.d/conda.sh && conda activate efficient-llm && "
            + cmd
        )
        result = subprocess.run(["bash", "-c", full], cwd=REPO, capture_output=True, text=True)
        log(f"aggregator: {result.stdout.strip()[:500]}")
    log("Phase 7 ALL DONE")


if __name__ == "__main__":
    main()
