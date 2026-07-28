"""Dump stage-1 inputs + the shipped Triton vq2 kernel's outputs as golden data.

The TileLang venv has no sglang, so parity is checked file-to-file: this runs in
the oscar venv and saves everything; the TileLang side loads and compares.
Att_Out/Lse are only compared on VALID splits (split_kv_start < seq_len) --
beyond them the Triton kernel stores 0/0 = NaN by design and stage 2 never
reads those slots.
"""
import os, sys, torch
sys.path.insert(0, "pipelines/throughput/kernel_study")
import bench_stage1 as B
from sglang.srt.layers.attention.triton_ops import decode_attention as D

out_path = os.environ["OUT"]
B.BATCH = int(os.environ.get("BS", 2))
B.SEQ = int(os.environ.get("CTX", 1000))
d = B.build()
s = int(d["splits"].min())
D._decode_grouped_att_m_fwd_quant_vq2(
    d["q"], d["k_idx"], d["cb"], d["v_buf"], d["k_sz"], d["v_sz"],
    d["out"], d["lse"], d["kv_indptr"], d["kv_indices"], d["splits"],
    B.SPLITS, B.SM_SCALE, 0.0)
torch.cuda.synchronize()
torch.save({k: d[k].cpu() for k in
            ("q", "k_idx", "cb", "v_buf", "k_sz", "v_sz", "kv_indptr",
             "kv_indices", "splits", "out", "lse")} |
           {"BATCH": B.BATCH, "SEQ": B.SEQ, "SPLITS": B.SPLITS,
            "SM_SCALE": B.SM_SCALE, "splits_used": s}, out_path)
print(f"saved {out_path}: bs={B.BATCH} ctx={B.SEQ} max_splits={B.SPLITS} "
      f"engine_splits={s} out{tuple(d['out'].shape)}")
