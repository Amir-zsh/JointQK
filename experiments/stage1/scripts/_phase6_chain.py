#!/usr/bin/env python3
"""Wait for Phase 1C (3 OK across both launchers) AND Phase 5 (3 summary.json) done,
then launch Phase 6 on all 6 GPUs."""
import datetime as dt
import re
import subprocess
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
LOG = REPO / "experiments/stage1/logs/phase6_chain.log"
LOG.parent.mkdir(parents=True, exist_ok=True)


def log(msg):
    line = f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}\n"
    LOG.open("a").write(line)
    print(line, end="", flush=True)


def count_oks(overview):
    if not Path(overview).exists():
        return 0
    txt = Path(overview).read_text()
    return len(re.findall(r"\bOK\b", txt))


def phase5_summaries():
    base = REPO / "experiments/stage1/logs"
    return sum(1 for b in (2, 3, 4) if (base / f"e3_b{b}_r64.summary.json").exists())


def main():
    LOG.write_text("")  # truncate
    log("chain waiting for Phase 1C (3 OK total) + Phase 5 (3 summary.json)")

    while True:
        n_p1c = count_oks(REPO / "experiments/stage1/logs/phase1c/_overview.log") \
              + count_oks(REPO / "experiments/stage1/logs/phase1c_rerun/_overview.log")
        n_p5 = phase5_summaries()
        log(f"phase1c_OKs={n_p1c}/3, phase5_summaries={n_p5}/3")
        if n_p1c >= 3 and n_p5 >= 3:
            log("both done; proceeding")
            break
        time.sleep(60)

    # Aggregate Phase 1C
    log("aggregating Phase 1C combined sanity")
    out = REPO / "artifacts/stage1/v_method_study/sweep_combined"
    for kb in (2, 3, 4):
        d = out / f"jointqk_k{kb}_v3"
        metrics = list(d.glob("**/metrics.json"))
        if metrics:
            f1 = float(metrics[0].read_text())
            log(f"  K{kb}/V3 combined F1={f1:.2f}")

    # Run Llama gates (Phase 5 W1 gate)
    log("running Llama gates")
    for gate in ("gate_e3", "gate_e5"):
        result = subprocess.run([
            "bash", "-c",
            f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate efficient-llm && "
            f"python -m experiments.stage1.gates.{gate} "
            f"--output-dir artifacts/stage1/cca_vs_waterfill_study/llama31_8b"
        ], cwd=REPO, capture_output=True, text=True)
        log(f"{gate} stdout: {result.stdout.strip()}")
        if result.returncode != 0:
            log(f"{gate} STDERR: {result.stderr.strip()}")

    # Launch Phase 6 on all 6 GPUs
    log("launching Phase 6 on GPUs 0,1,2,3,4,5")
    subprocess.run([
        "bash", str(REPO / "experiments/stage1/scripts/launch_phase6_decode_scope.sh"),
        "--gpus", "0,1,2,3,4,5",
        "--fraction", "0.3",
    ], cwd=REPO)

    log("Phase 6 launcher exited")


if __name__ == "__main__":
    main()
