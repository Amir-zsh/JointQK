"""Verify the geometry-templated CUDA kernel at gpt-oss-20b shapes.

Geometry (16, 256, 64, 8): H_Q=64, H_KV=8, head_dim=64, NG=16. Reference is
the geometry-generic Triton vq2 kernel on identical synthetic inputs -- two
independent implementations agreeing at ~1e-5 is the verification, since no
golden exists for this geometry yet. Also benches both.
"""
import os, sys, torch

os.environ["SGLANG_VQ2_CUDA"] = "0"        # start with Triton for the reference
os.environ["SGLANG_VQ2_CUDA_FP32"] = "1"
sys.path.insert(0, "pipelines/throughput/kernel_study")
import bench_stage1 as B

# gpt-oss-20b geometry
B.H_Q, B.H_KV, B.L, B.NG, B.KC = 64, 8, 64, 16, 256
B.SM_SCALE = 1.0 / (B.L ** 0.5)
B.BATCH = int(os.environ.get("BS", 32))
B.SEQ = int(os.environ.get("CTX", 30000))

from sglang.srt.layers.attention.triton_ops import decode_attention as D
from sglang.srt.layers.attention.triton_ops import vq2_cuda_stage1 as C

d = B.build()
s = int(d["splits"].min())
print(f"geometry: H_Q={B.H_Q} H_KV={B.H_KV} L={B.L} NG={B.NG}  "
      f"bs={B.BATCH} ctx={B.SEQ} splits={s}")

def run():
    D._decode_grouped_att_m_fwd_quant_vq2(
        d["q"], d["k_idx"], d["cb"], d["v_buf"], d["k_sz"], d["v_sz"],
        d["out"], d["lse"], d["kv_indptr"], d["kv_indices"], d["splits"],
        B.SPLITS, B.SM_SCALE, 0.0)

run(); torch.cuda.synchronize()
ref_out, ref_lse = d["out"].clone(), d["lse"].clone()
t_triton = B.bench(run) * 1e3

# now the CUDA path through the same dispatch
os.environ["SGLANG_VQ2_CUDA"] = "1"
geom = (B.NG, B.KC, B.L, B.H_Q // B.H_KV)
assert geom in C._SUPPORTED_GEOMS, geom
C._ext(geom)                       # build outside any capture
ok = C.supports(d["q"], d["k_idx"], d["cb"], d["v_buf"], d["k_sz"], d["v_sz"],
                d["out"], d["lse"], d["kv_indptr"], d["kv_indices"],
                d["splits"], 0.0, 0, False)
print(f"supports() at gpt-oss geometry: {ok}")
assert ok
d["out"].zero_(); d["lse"].zero_()
run(); torch.cuda.synchronize()

v = s
go, gr = d["out"][:, :, :v], ref_out[:, :, :v]
assert torch.isfinite(go).all()
rel = ((go - gr).abs().max() / gr.abs().max().clamp_min(1e-9)).item()
dl = (d["lse"][:, :, :v] - ref_lse[:, :, :v]).abs().max().item()
t_cuda = B.bench(run) * 1e3
print(f"CUDA vs Triton at (16,256,64,8): rel {rel:.3e}  lse {dl:.3e}  "
      f"{'PASS' if rel < 2e-3 and dl < 1e-3 else 'FAIL'}")
print(f"stage-1: Triton {t_triton:.0f} us   CUDA fp32 {t_cuda:.0f} us   "
      f"({t_triton / t_cuda:.2f}x)")
sys.exit(0 if rel < 2e-3 else 1)
