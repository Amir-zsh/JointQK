# Kernel tuning: Samuel's fused VQ8 vs the OSCAR authors' kernel (A100, bs=1)

2026-07-17, branch pgq10. One machine (A100-SXM4-40GB, GPU 5), one timing
harness (CUDA-graphed, 30-iter medians after warmup), both K **and** V at
2-bit in every arm. VQ8 = Samuel's fused single-kernel design (commit
3c65507; our attribution port `kvq/kernels/vq8_fused.py`): fp8e5 codebook,
one int32 load per 4-coord codeword, fused online softmax + INT2-V, whole
model per launch (288 head-programs), per-layer = total / 36. OSCAR = the
authors' vendored Triton kernel, K+V INT2 with per-token scale/zero.

## Sweeps

- OSCAR: splits {8,16,32,64} x BLOCK_N {64,128} x BLOCK_H {8,16} x warps
  {2,4,8} x num_stages {2,3,4} via their SGL_INT2_* env overrides
  (`artifacts/kernels/tune_sweep.json`).
- VQ8: splits {16,32,64,128} x BT {64,128,256} x warps {2,4,8} x
  num_stages {2,3,4} (`artifacts/kernels/tune_vq8.json`).
- Coarse ungraphed ranking, top-4 re-timed under CUDA graphs.

## Results (ms per layer decode step)

| kernel | 32K default | 32K tuned | 128K default | 128K tuned |
|---|---|---|---|---|
| OSCAR (authors') | 0.079 | **0.075** | 0.261 | **0.233** |
| VQ8 (Samuel's)   | 0.079 | **0.073** | 0.302 | 0.281 |
| VQ8 / OSCAR (tuned) | | **0.97x — VQ8 faster** | | 1.21x |

Best configs:
- OSCAR: splits=64 everywhere; 32K -> BLOCK_N=64/BLOCK_H=8/W=4/S=3;
  128K -> BLOCK_N=128/BLOCK_H=16/W=4/S=4. (Their shipped defaults are
  H100-tuned; A100 gains ~5-11%.)
- VQ8: BT=64, warps=2, stages=2 at both contexts; splits=64 @32K,
  splits=32 @128K. (Samuel's hand defaults were within 7% of tuned;
  uint8 indices tested separately and REJECTED — int16 gathers vectorize
  better.)

## Conclusions

1. **Tuned-vs-tuned, VQ decode is at parity with OSCAR at 32K (0.073 vs
   0.075 — VQ8 marginally ahead) and within 21% at 128K** (0.281 vs
   0.233). The residual 128K gap tracks the codebook gather's cache
   pressure as the K stream grows; smaller BT + fewer warps + shallower
   pipeline consistently win for VQ8 (less register/L1 contention with the
   gather), the opposite direction from OSCAR's optimum — the two kernels
   want opposite tunings, so a shared-default comparison mis-serves one of
   them by construction.
2. This completes the correction arc recorded in report10 A10-2: the
   original "VQ is 3.7x behind OSCAR" measured our two-phase harness, not
   the format. In the right kernel architecture the VQ format costs
   ~0-21% decode speed vs OSCAR while carrying zero per-token metadata and
   the +2.4 F1 (SIG) quality margin at fewer bits.
3. History of the estimate, for honesty: 3.7x (our two-phase kernel) ->
   1.16x (Samuel's fused kernel vs OSCAR defaults) -> **0.97-1.21x
   (both tuned)**. Caveats: bs=1, A100, synthetic buffers (timing is
   value-independent), V=INT2 for both; the plan11 engine integration adds
   paged-indices indirection to VQ8, which will be re-measured under V3.
4. pgq two-phase is retired from this comparison (0.635 tuned @128K; its
   gap is architecture, not format — plan9's unfinished full-fusion is the
   known fix, out of scope here).

Sweep artifacts: `artifacts/kernels/tune_sweep.json`,
`artifacts/kernels/tune_vq8.json`.
