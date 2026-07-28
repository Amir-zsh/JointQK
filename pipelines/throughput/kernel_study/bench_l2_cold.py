"""Does a cold L2 explain the 28% microbenchmark-vs-server gap on vq2 stage-1?

The gap: in the server, vq2 stage-1 costs 2311 us/call against 1803 us in the
microbenchmark (+28%), while int2 matches (1571 vs 1649). Ruled out already:
output stride, contention, grid shape, codebook residency, int64 indices, bf16
q, and split count.

What is left is cache STATE. The benchmark replays one kernel 200x back to back,
so the codebook and a good slice of the index stream stay resident. The server
walks 36 per-layer index arenas (614 MB each for Qwen3-8B), the V arena, the
scales, the HP arena and the model weights between two calls to the same layer's
kernel -- everything is evicted by the time it comes round again.

That should hurt vq2 much more than int2: vq2's inner loop gathers from a
256-entry codebook it wants held in cache, whereas int2 streams packed bytes and
is closer to DRAM-bandwidth-bound. If flushing L2 between launches reproduces
~28% on vq2 and ~0% on int2, the gap is explained and it is not a kernel defect.

  docker exec -e CUDA_VISIBLE_DEVICES=5 oscar-ab bash -lc 'cd <repo> && \
    PYTHONPATH=vendor/OSCAR-vq/sglang-research/python \
    /opt/venv-oscar/bin/python -u pipelines/throughput/kernel_study/bench_l2_cold.py'
"""
import sys

import torch

sys.path.insert(0, "pipelines/throughput/kernel_study")
import bench_stage1 as B                                    # noqa: E402
from sglang.srt.layers.attention.triton_ops import decode_attention as D  # noqa: E402

# H100 has 50 MB L2; 256 MB of writes evicts it comfortably without the memset
# itself dominating the measured interval.
FLUSH_MB = 256


def timed(run, flush_buf, reps=50):
    """Time `run`, optionally with an L2 flush between iterations.

    The flush is timed separately and subtracted, so the reported number is the
    kernel alone under the two cache states -- not kernel + memset.
    """
    for _ in range(3):
        run()
    torch.cuda.synchronize()

    def loop(with_flush):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(reps):
            if with_flush:
                flush_buf.zero_()
            run()
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b) / reps * 1e3

    def flush_only():
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(reps):
            flush_buf.zero_()
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b) / reps * 1e3

    warm = loop(False)
    cold = loop(True) - flush_only()
    return warm, cold


def main() -> int:
    flush = torch.empty(FLUSH_MB * 1024 * 1024, dtype=torch.uint8, device="cuda")
    print(f"{torch.cuda.get_device_name(0)}  ctx={B.SEQ}  L2 flush={FLUSH_MB} MB\n")
    print(f"{'arm':>6} {'bs':>4} {'splits':>7} {'warm L2':>10} {'cold L2':>10} "
          f"{'cold/warm':>10}")

    for bs in (8, 32, 64):
        B.BATCH = bs
        d = B.build()
        for arm in ("vq2", "int2"):
            s = int(d["splits"].min())
            if arm == "vq2":
                def run():
                    D._decode_grouped_att_m_fwd_quant_vq2(
                        d["q"], d["k_idx"], d["cb"], d["v_buf"], d["k_sz"],
                        d["v_sz"], d["out"], d["lse"], d["kv_indptr"],
                        d["kv_indices"], d["splits"], s, B.SM_SCALE, 0.0)
            else:
                def run():
                    D._decode_grouped_att_m_fwd_quant_int2(
                       d["q"], d["k_int2"], d["v_buf"], d["k_sz"], d["v_sz"],
                       d["out"], d["lse"], d["kv_indptr"], d["kv_indices"],
                       d["splits"], s, B.SM_SCALE, 0.0)
            try:
                warm, cold = timed(run, flush)
            except Exception as e:
                print(f"{arm:>6} {bs:>4} {s:>7}   skipped: "
                      f"{type(e).__name__}: {str(e).splitlines()[0][:60]}")
                continue
            print(f"{arm:>6} {bs:>4} {s:>7} {warm:>9.0f}u {cold:>9.0f}u "
                  f"{cold / warm:>9.2f}x")
        del d
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
