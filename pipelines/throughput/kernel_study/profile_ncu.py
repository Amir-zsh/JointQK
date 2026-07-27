"""Nsight Compute comparison of the vq2 / int2 / variant stage-1 kernels.

The earlier round of this study inferred the bottleneck by ablation (delete a
load, diff the runtime) because ncu was unavailable: the driver ships with
`RmProfilingAdminOnly: 1`, so counters need CAP_SYS_ADMIN, which the normal
`oscar-ab` container does not have. A *privileged sibling container* gets them
without touching the host driver:

  docker run -d --name oscar-prof --gpus all --privileged --shm-size=32g \
    -e NVIDIA_DRIVER_CAPABILITIES=all -e HF_HOME=/root/.cache/huggingface \
    -v <repo>:<repo> -v /raid/amir/.cache/huggingface:/root/.cache/huggingface \
    -w <repo> oscar-cu13:latest sleep infinity

Then, from the host:

  python pipelines/throughput/kernel_study/profile_ncu.py

`sectors/request` is the number this study lives on: it is the L1 address
divergence of the codebook gather. 32 means every lane in a warp landed in a
distinct 32 B sector -- the worst case, and one warp instruction costs 32
wavefronts through the LSU.
"""
import argparse
import csv
import io
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CONTAINER = os.environ.get("PROF_CONTAINER", "oscar-prof")

METRICS = [
    "gpu__time_duration.sum",
    "launch__registers_per_thread",
    "launch__occupancy_limit_registers",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "lts__t_sectors.sum",
    "dram__bytes_read.sum",
    "launch__occupancy_limit_shared_mem",
    "launch__shared_mem_per_block_driver",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    "smsp__inst_executed_op_shared_ld.sum",
    "smsp__inst_executed_op_local_ld.sum",
    "smsp__inst_executed_op_local_st.sum",
    "smsp__inst_executed.sum",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
]

ROWS = [
    ("time", "gpu__time_duration.sum", lambda v: f"{v/1e3:,.0f} us"),
    ("regs/thread", "launch__registers_per_thread", lambda v: f"{v:,.0f}"),
    ("occ limit (regs) CTA/SM", "launch__occupancy_limit_registers", lambda v: f"{v:,.0f}"),
    ("warps active %", "sm__warps_active.avg.pct_of_peak_sustained_active", lambda v: f"{v:.1f}%"),
    ("global ld requests", "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", lambda v: f"{v/1e6:,.1f} M"),
    ("global ld sectors", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", lambda v: f"{v/1e6:,.1f} M"),
    ("SECTORS / REQUEST", None, None),
    ("L1 throughput %", "l1tex__throughput.avg.pct_of_peak_sustained_active", lambda v: f"{v:.1f}%"),
    ("L1 hit rate", "l1tex__t_sector_hit_rate.pct", lambda v: f"{v:.1f}%"),
    ("L2 hit rate", "lts__t_sector_hit_rate.pct", lambda v: f"{v:.1f}%"),
    ("L2 sectors", "lts__t_sectors.sum", lambda v: f"{v/1e6:,.1f} M"),
    ("DRAM read", "dram__bytes_read.sum", lambda v: f"{v/1e9:.2f} GB"),
    ("inst executed", "smsp__inst_executed.sum", lambda v: f"{v/1e6:,.0f} M"),
    ("shared/block", "launch__shared_mem_per_block_driver", lambda v: f"{v/1024:,.0f} KB"),
    ("occ limit (smem) CTA/SM", "launch__occupancy_limit_shared_mem", lambda v: f"{v:,.0f}"),
    ("shared ld inst", "smsp__inst_executed_op_shared_ld.sum", lambda v: f"{v/1e6:,.1f} M"),
    ("LOCAL ld (spill)", "smsp__inst_executed_op_local_ld.sum", lambda v: f"{v/1e6:,.1f} M"),
    ("LOCAL st (spill)", "smsp__inst_executed_op_local_st.sum", lambda v: f"{v/1e6:,.1f} M"),
    ("shared BANK CONFLICTS", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum", lambda v: f"{v/1e6:,.1f} M"),
    ("stall long_scoreboard", "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio", lambda v: f"{v:.2f}"),
    ("stall short_scoreboard", "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio", lambda v: f"{v:.2f}"),
    ("stall mio_throttle", "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio", lambda v: f"{v:.2f}"),
    ("stall math_pipe", "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio", lambda v: f"{v:.2f}"),
    ("stall barrier", "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio", lambda v: f"{v:.2f}"),
    ("stall not_selected", "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio", lambda v: f"{v:.2f}"),
    ("stall wait", "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio", lambda v: f"{v:.2f}"),
]


def run(script: str, kernel_regex: str, env: dict, python: str,
        launch_count: int, argv: list) -> dict:
    envs = sum((["-e", f"{k}={v}"] for k, v in env.items()), [])
    cmd = [
        "docker", "exec", "-e", "CUDA_VISIBLE_DEVICES=0",
        "-e", "PYTHONPATH=vendor/OSCAR-vq/sglang-research/python", *envs, CONTAINER,
        "ncu", "--target-processes", "all", "-k", f"regex:{kernel_regex}",
        "--metrics", ",".join(METRICS), "--csv",
        "--launch-count", str(launch_count),
        python, script, *argv,
    ]
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    lines = [l for l in p.stdout.splitlines() if l.startswith('"')]
    if not lines:
        print(p.stdout[-3000:], file=sys.stderr)
        print(p.stderr[-3000:], file=sys.stderr)
        raise SystemExit("ncu produced no CSV rows")
    out: dict = {}
    for row in csv.DictReader(io.StringIO("\n".join(lines))):
        name = row.get("Kernel Name") or row.get("Function Name")
        metric, val = row["Metric Name"], row["Metric Value"]
        try:
            v = float(val.replace(",", ""))
        except ValueError:
            continue
        out.setdefault(name, {})[metric] = v
    return out


def report(data: dict, order: list[str]) -> None:
    cols = [k for k in order if k in data] + [k for k in data if k not in order]
    w = max(len(c) for c in cols) + 2
    print(f"\n{'metric':<26}" + "".join(f"{c[:w-1]:>{w}}" for c in cols))
    print("-" * (26 + w * len(cols)))
    for label, metric, fmt in ROWS:
        cells = []
        for c in cols:
            if metric is None:                    # derived: sectors per request
                r = data[c].get("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum")
                s = data[c].get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")
                cells.append(f"{s/r:.2f}" if r else "-")
            else:
                v = data[c].get(metric)
                cells.append(fmt(v) if v is not None else "-")
        print(f"{label:<26}" + "".join(f"{x:>{w}}" for x in cells))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="pipelines/throughput/kernel_study/vq2_variants.py")
    ap.add_argument("--kernel", default="stage1")
    ap.add_argument("--env", action="append", default=[],
                    help="KEY=VALUE passed into the container")
    ap.add_argument("--python", default="/opt/venv-oscar/bin/python")
    ap.add_argument("--launch-count", type=int, default=3)
    ap.add_argument("--argv", nargs="*", default=[])
    a = ap.parse_args()
    env = {"PROFILE": "1"}
    env.update(dict(kv.split("=", 1) for kv in a.env))
    data = run(a.script, a.kernel, env, a.python, a.launch_count, a.argv)
    report(data, ["_fwd_grouped_kernel_stage1_quant_int2",
                  "_fwd_grouped_kernel_stage1_quant_vq2", "_stage1_gm"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
