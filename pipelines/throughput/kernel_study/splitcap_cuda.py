"""Split sweep for the CUDA fp32 kernel across batch: is the engine heuristic
(cap=48) already the right adaptive schedule for it?"""
import os, sys, torch, triton
os.environ["SGLANG_VQ2_CUDA"] = "1"
os.environ["SGLANG_VQ2_CUDA_FP32"] = "1"
sys.path.insert(0, "pipelines/throughput/kernel_study")
import bench_stage1 as B
from sglang.srt.layers.attention.triton_ops import decode_attention as D
from sglang.srt.layers.attention.triton_backend import get_num_kv_splits_triton

CAP = 48
SPL = (2, 4, 8, 16, 24, 32, 48)
cores = torch.cuda.get_device_properties(0).multi_processor_count
print(f"ctx={B.SEQ} CUDA fp32 kernel  (us, stage-1 only)")
print(f"{'bs':>4} {'heur':>5} | " + "  ".join(f"s={s:<3}" for s in SPL) + " |  best | heur-time")
for bs in (1, 2, 4, 8, 16, 32, 64):
    B.BATCH = bs
    d = B.build()
    nsp = torch.empty(bs, device="cuda", dtype=torch.int32)
    seq = torch.full((bs,), B.SEQ, device="cuda", dtype=torch.int32)
    get_num_kv_splits_triton[(1,)](nsp, seq, bs, 1, B.H_Q, B.H_KV, CAP, cores,
                                   MAX_NUM_SEQ=256 if bs < 256 else triton.next_power_of_2(bs))
    heur = int(nsp.min())
    times = {}
    row = []
    for s in SPL + ((heur,) if heur not in SPL else ()):
        f = torch.full_like(nsp, s)
        out = torch.zeros(bs, B.H_Q, s, B.L, device="cuda", dtype=torch.float32)
        lse = torch.zeros(bs, B.H_Q, s, device="cuda", dtype=torch.float32)
        def run():
            D._decode_grouped_att_m_fwd_quant_vq2(
                d["q"], d["k_idx"], d["cb"], d["v_buf"], d["k_sz"], d["v_sz"],
                out, lse, d["kv_indptr"], d["kv_indices"], f, s, B.SM_SCALE, 0.0)
        try:
            times[s] = B.bench(run) * 1e3
        except Exception:
            times[s] = float("nan")
        if s in SPL:
            row.append(f"{times[s]:7.0f}")
    b = min((s for s in times if times[s] == times[s]), key=lambda s: times[s])
    print(f"{bs:>4} {heur:>5} | " + "  ".join(row) +
          f" |  s={b} @ {times[b]:.0f} | s={heur} @ {times[heur]:.0f}"
          f"  (+{100*(times[heur]/times[b]-1):.0f}%)")
    del d
    torch.cuda.empty_cache()
