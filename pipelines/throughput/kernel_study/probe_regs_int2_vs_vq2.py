"""Register pressure / occupancy of the Triton int2 vs vq2 decode kernels.

The vq2 kernel compiles to 255 registers with spills -- pinned at the ceiling.
If int2 fits in fewer, it gets more CTAs/SM, and the vq2-vs-int2 gap would be an
OCCUPANCY effect rather than the gather-throughput effect I assumed all session.
"""
import sys
sys.path.insert(0, "logs")
import torch
from test_tl_vq2 import build, BATCH, H_Q, SPLITS, L, SM_SCALE
from sglang.srt.layers.attention.triton_ops import decode_attention as D

d = build()
out = torch.zeros(BATCH, H_Q, SPLITS, L, device="cuda", dtype=torch.float32)
lse = torch.zeros(BATCH, H_Q, SPLITS, device="cuda", dtype=torch.float32)

D._decode_grouped_att_m_fwd_quant_vq2(
    d["q"], d["k_idx"], d["cb"], d["v_buf"], d["k_sz"], d["v_sz"],
    out, lse, d["kv_indptr"], d["kv_indices"], d["splits"], SPLITS, SM_SCALE, 0.0)

# int2: K is packed INT2 like V (head_dim//4 bytes), with affine scale/zero.
k_int2 = torch.randint(0, 256, (d["cache"], 8, L // 4), device="cuda", dtype=torch.uint8)
D._decode_grouped_att_m_fwd_quant_int2(
    d["q"], k_int2, d["v_buf"], d["k_sz"], d["v_sz"],
    out, lse, d["kv_indptr"], d["kv_indices"], d["splits"], SPLITS, SM_SCALE, 0.0)

for name, K in (("vq2 ", D._fwd_grouped_kernel_stage1_quant_vq2),
                ("int2", D._fwd_grouped_kernel_stage1_quant_int2)):
    dc = getattr(K, "device_caches", None) or {}
    for dev, tup in dc.items():
        kc = tup[0] if isinstance(tup, (tuple, list)) else tup
        for key, ck in list(kc.items())[:1]:
            m = ck.metadata
            thr = m.num_warps * 32
            by_reg = 65536 // max(ck.n_regs * thr, 1)
            by_smem = (228 * 1024) // max(m.shared, 1)
            print(f"{name}: warps={m.num_warps} threads={thr} shared={m.shared}B "
                  f"regs={ck.n_regs} spills={ck.n_spills} -> "
                  f"CTAs/SM: reg={by_reg} smem={by_smem} => {min(by_reg, by_smem)}")
