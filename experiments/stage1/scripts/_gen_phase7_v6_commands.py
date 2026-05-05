"""Emit Phase 7 v6 JSONL commands for phase7_worker.py.

Phase 7 v6: Qwen3-8B / 8 KIVI tasks / fraction=1.0 / Mode A (compress_decode=False).
Layer 0 skipped for K and V on every compressed method (`layer0_full_precision=True`).
JointQK V uses centered v_random with budgets V∈{2,3}; TurboQuant duplicated at the
same V budgets for fair comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT_BASE = REPO / "artifacts/stage1/downstream_v6/qwen3_8b"
CCA = REPO / "artifacts/stage1/cca_vs_waterfill_study/cca_stats.pt"
VST = REPO / "artifacts/stage1/v_method_study/v_stats.pt"
LOG_DIR = REPO / "experiments/stage1/logs/phase7_v6_qwen3_8b"

TASKS = ["qasper", "qmsum", "multi_news", "trec", "triviaqa",
         "samsum", "lcc", "repobench-p"]
K_BUDGETS = [2, 3, 4]
V_BUDGETS = [2, 3]
V_METHOD = "v_random"
COMPRESS_DECODE = False  # Mode A
LAYER0_FP = True


def emit(rows, label, press_name, press_kwargs, task):
    rows.append({
        "press_name": press_name,
        "press_kwargs": press_kwargs,
        "dataset": "longbench",
        "data_dir": task,
        "fraction": 1.0,
        "output_dir": str(OUT_BASE / f"{label}_{task}"),
        "_label": f"{label}_{task}",
    })


def main():
    rows = []
    for task in TASKS:
        # 1. Oracle
        emit(rows, "full_precision", "no_press", {"compression_ratio": 0.0}, task)
        # 2. JointQK × {K, V}
        for kb in K_BUDGETS:
            for vb in V_BUDGETS:
                kw = {
                    "cca_stats_path": str(CCA),
                    "v_stats_path": str(VST),
                    "v_method": V_METHOD,
                    "k_bits": kb,
                    "v_bits": vb,
                    "compress_decode": COMPRESS_DECODE,
                    "quantize_k": True,
                    "quantize_v": True,
                    "layer0_full_precision": LAYER0_FP,
                }
                emit(rows, f"jointqk_k{kb}_v{vb}", "jointqk", kw, task)
        # 3. TurboQuant × {K, V}
        for kb in K_BUDGETS:
            for vb in V_BUDGETS:
                kw = {
                    "k_bits": kb,
                    "v_bits": vb,
                    "compress_decode": COMPRESS_DECODE,
                    "layer0_full_precision": LAYER0_FP,
                }
                emit(rows, f"turboquant_k{kb}_v{vb}", "turboquant", kw, task)
        # 4. KIVI int{2, 3, 4}
        for kbits in (2, 3, 4):
            emit(rows, f"kivi_int{kbits}", "kivi", {
                "k_bits": kbits, "v_bits": kbits, "group_size": 128,
                "compress_decode": COMPRESS_DECODE,
                "layer0_full_precision": LAYER0_FP,
            }, task)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cmds = LOG_DIR / "commands.jsonl"
    with cmds.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"# wrote {len(rows)} jobs to {cmds}")
    print(f"# tasks: {len(TASKS)}; configs/task: {len(rows) // len(TASKS)}")


if __name__ == "__main__":
    main()
