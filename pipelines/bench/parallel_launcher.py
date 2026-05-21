#!/usr/bin/env python3
"""Distribute shell jobs across GPUs with per-job logs.

Each GPU hosts `--jobs-per-gpu N` worker slots; a worker only ever runs one
job at a time. Jobs are picked from a FIFO queue as workers finish. This avoids
the round-robin collision where job index N+k could land on a GPU still busy
with job k. With N=1 (default) behaviour matches the prior single-job-per-GPU
semantics; with N=2 two jobs share one A100, sharing CUDA_VISIBLE_DEVICES.

Reads a commands file (one bash command per line, '#' comments allowed).
Trailing '# label=foo' on a command sets the log filename suffix for that job.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _bootstrap  # noqa: E402, F401

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path


def gpu_worker(gpu, slot, in_queue, out_queue, log_dir):
    """Pull jobs from in_queue, run each with CUDA_VISIBLE_DEVICES=gpu, post results.

    `slot` distinguishes co-located workers on the same GPU when --jobs-per-gpu>1
    so the per-job log filenames remain unique.
    """
    log_dir = Path(log_dir)
    while True:
        item = in_queue.get()
        if item is None:
            break
        job_idx, cmd, label = item
        log_file = log_dir / f"job_{job_idx:03d}_gpu{gpu}s{slot}_{label}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        start = time.time()
        with log_file.open("w") as f:
            f.write(f"# job {job_idx} | GPU {gpu} slot {slot} | label {label} | start {time.ctime(start)}\n")
            f.write(f"# cmd: {cmd}\n\n")
            f.flush()
            rc = subprocess.run(["bash", "-c", cmd], env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        out_queue.put((job_idx, gpu, slot, rc, time.time() - start, str(log_file)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--commands-file", required=True)
    p.add_argument("--log-dir", required=True)
    p.add_argument("--gpus", default="0,1,2,3,4,5")
    p.add_argument("--jobs-per-gpu", type=int, default=1,
                   help="Number of worker slots per GPU. With N=2, two jobs share each GPU.")
    p.add_argument("--label-from-comment", action="store_true")
    args = p.parse_args()
    if args.jobs_per_gpu < 1:
        raise ValueError(f"--jobs-per-gpu must be >= 1, got {args.jobs_per_gpu}")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    gpus = [int(g) for g in args.gpus.split(",")]
    n_workers = len(gpus) * args.jobs_per_gpu

    cmds, labels = [], []
    for line in Path(args.commands_file).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        label = "job"
        if args.label_from_comment and "# label=" in line:
            cmd, _, lbl = line.rpartition("# label=")
            cmd, label = cmd.rstrip(), lbl.strip()
        else:
            cmd = line
        cmds.append(cmd)
        labels.append(label)

    overview = log_dir / "_overview.log"
    overview.write_text(
        f"# {len(cmds)} jobs across GPUs {gpus} (jobs-per-gpu={args.jobs_per_gpu}, "
        f"workers={n_workers}) | start {time.ctime()}\n\n"
    )

    in_q: mp.Queue = mp.Queue()
    out_q: mp.Queue = mp.Queue()
    for i, (c, l) in enumerate(zip(cmds, labels)):
        in_q.put((i, c, l))
    for _ in range(n_workers):
        in_q.put(None)  # poison pills, one per worker

    workers = []
    for gpu in gpus:
        for slot in range(args.jobs_per_gpu):
            w = mp.Process(target=gpu_worker, args=(gpu, slot, in_q, out_q, str(log_dir)))
            w.start()
            workers.append(w)

    fail = 0
    for _ in range(len(cmds)):
        job_idx, gpu, slot, rc, elapsed, log_file = out_q.get()
        status = "OK  " if rc == 0 else "FAIL"
        line = (
            f"[{time.strftime('%H:%M:%S')}] job {job_idx:03d} GPU {gpu}s{slot} {status} "
            f"{elapsed:6.0f}s {Path(log_file).name}"
        )
        print(line, flush=True)
        with overview.open("a") as f:
            f.write(line + "\n")
        if rc != 0:
            fail += 1

    for w in workers:
        w.join()
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
