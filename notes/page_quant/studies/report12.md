# Report 12 — Llama-3.1-8B five-method grid, Qwen sweep completion, and the VQ-V ablation

**Status: results COMPLETE (lambda6 GPU-pool run, night of 2026-07-20 → 21);
VQ-V mechanism resolved offline (section 4).**

All cells served on the vq2-extended OSCAR engine (vendor/OSCAR-vq @
`6f78cd6cd`, gates G1–G6 ALL PASS), one stack for every method, identical
exported rows per cell, sharded across 6 A100s by the pool scheduler.

## 1. Llama-3.1-8B: five methods, served

Onboarding built fresh this run: OSCAR rotations from the authors' own
pipeline (50 GPQA-prompt QKV dump → `compute_kv_rotation` qqt_sst/r_h_pbr,
orth err ~2e-8) and a ptn/64K-concat GPQA codebook via Samuel's trainer
(stratified, flat, 2 b/coord). Sampling for reasoning/code: K=5 seeds,
T=0.6 / top-p 0.9 / no top-k (Llama-3.1 generation_config), non-thinking
caps (gpqa/math 4096, aime 8192 — the 2048 defaults breached the ≤15%
cap-hit rule in short-validation and were re-exported before any full cell).

**NIAH mean (800 rows/cell, greedy):**

| ctx | bf16 | **vq2** | int2 (OSCAR) | QuaRot-INT2 | Naive-INT2 |
|---|---|---|---|---|---|
| 8K  | 98.2 | 92.3 | 84.3 | 37.3 | 0.3 |
| 16K | 98.5 | 91.2 | 82.7 | 35.5 | 0.2 |
| 32K | 97.4 | 91.7 | 80.0 | 37.3 | 0.1 |
| 64K | 97.4 | **92.1** | 76.7 | 33.6 | 0.0 |

**Reasoning / code (avg@5):**

| task | bf16 | vq2 | int2 | QuaRot | Naive |
|---|---|---|---|---|---|
| math500 (full 500) | 44.3 | 44.3 | 44.5 | 6.4 | 3.9 |
| HumanEval | 63.8 | 63.7 | 64.8 | 7.0 | 0.0 |
| GPQA-diamond | 25.1 | 24.6 | 21.8 | 2.9 | 1.0 |
| AIME25 | ≈0 for every method — Llama-3.1-8B scores ~0 as published; uninformative row |

Findings:

1. **vq2 is context-flat on Llama (~92 from 8K to 64K); production OSCAR
   degrades monotonically (84.3 → 76.7).** At 64K vq2 leads by **+15.4**.
   Same qualitative shape as Qwen, with earlier and larger separation.
2. **vq2's ~6-pt gap to bf16 is context-independent** → codebook capacity
   (calibration quality), not a streaming/position effect. The Llama
   codebook is first-cut (GPQA-only corpus, ~51K unique tokens cycled to
   64K); a richer mixed-domain corpus is the obvious lever.
3. **int2's transfer to Llama is poor even at 8K** (84.3 vs Qwen's 97.6).
   Caveats recorded: our rotations calibrate on 50 prompts (authors used
   83 / 20K-token budget) and the 0.96/0.92 clips are Qwen-tuned defaults —
   some of the gap may be calibration, not method. Still, vq2 uses the
   same 50-prompt-derived stack for V and a same-corpus codebook for K, so
   the *relative* ordering stands on equal footing.
4. **Reasoning/code: bf16 ≈ vq2 ≈ int2** (within seed noise) — quantization
   is free where retrieval pressure is low, on Llama as on Qwen. GPQA is
   near chance for this model (extraction ~72–80%; absolute ~25%), AIME is
   zero — both retained for protocol completeness, neither informative.
5. **QuaRot (~35) and Naive (~0) reproduce OSCAR's Table-2 baseline
   hierarchy** on a model family the paper never tested.

## 2. Qwen3-8B served NIAH sweep — now complete

| ctx | bf16 | vq2 (gpqacc64k) | int2 (production OSCAR) |
|---|---|---|---|
| 8K  | 99.9 | 99.4 | 97.6 |
| 16K | 99.5 | 98.6 | 95.8 |
| 32K | 98.3 | 96.6 | 74.2 |
| 64K | 86.3 | 76.8 | 23.4 |

(8/16K vq2 cells added this run; 32/64K from the plan11/V2 record.)

## 3. VQ-V ablation — closed, negative

Samuel's group-VQ applied to V (strided engine-basis codebook,
`vqv_G4_strided_gpqa_engine.pt`), on top of VQ-K, vs the shipping
VQ-K + INT2-V, served, same rows:

| served NIAH (Qwen) | 32K | 64K |
|---|---|---|
| VQ-K + INT2-V | 96.6 | 76.8 |
| VQ-K + **VQ-V** | 95.56 | **74.0** |

**VQ-V underperforms scalar INT2-V at both contexts and the deficit grows
with context** (−1.0 → −2.8), concentrated in multikey subtasks — despite
the offline de-risk showing 0.49× INT2-V's reconstruction MSE with fp16
centroids. These numbers are trustworthy in a way the first attempt's were
not: gates G4–G6 prove every V read path decodes exactly what the write
path stores (see §5). Verdict: **ablation row, not a flagship change.**

## 4. Mechanism: why VQ-V loses served despite 2× better offline MSE

Three checks (`vqv_e5m2_check.py` + inline variants; live-captured Qwen V,
same R_v the engine loads, layer-0 excluded):

1. **e5m2 snap — refuted.** rel-MSE: INT2-V 0.124, VQ-V fp16 0.058 (0.47×,
   reproduces Samuel's 0.49×), VQ-V e5m2 0.060 (0.48×). The sm80 snap costs
   only +3.3% — VQ-V keeps a >2× average advantage as the engine runs it.
2. **Tail error — refuted.** Per-token relMSE quantiles: VQ-V dominates
   INT2-V at every quantile including the worst token (VQ max 0.12–0.14 <
   INT2 median ≈0.125). Not a rare-token blowup.
3. **Data domain — confirmed.** On NIAH-haystack V (the served regime; the
   codebook is GPQA-calibrated) the ratio collapses from **0.48 → 0.76–0.80**:
   INT2-V is domain-invariant (0.127→0.128) while the codebook degrades +65%
   out of domain.

**Conclusion**: the served deficit is codebook domain-sensitivity — the same
property measured on the K side (his-vs-our codebook, 8–12 NIAH pts) — but V
has no softmax to absorb it, so the OOD penalty converts directly to
retrieval accuracy. Constructive next step: a **mixed-domain V codebook**
(and K codebook — same recipe) is the single lever most likely to flip both
the VQ-V ablation and the Llama vq2 gap.

## 5. Bugs found and fixed this run (both gated, tracker updated)

1. **VQ-V prefix dequant (Samuel's handoff patch, clone `6f78cd6cd`)**:
   `dequantize_prefix_kv` decoded the V index arena as packed int2 crumbs
   during chunked prefill — garbage V for every prompt > 1 chunk, invisible
   to all single-chunk smokes and to gates G4/G5 (a third reader of the
   format, never updated). Fixed + gate **G6**. Write-up:
   https://claude.ai/code/artifact/02c62657-a7de-42e9-a48e-ca6343b7417f
2. **merge_shards scoring (ours)**: CSV round-trip stringified NIAH answer
   lists; the ruler scorer then matched *characters* ("['1679215']" → 7/11
   → 63.64%). All sharded NIAH cells were mis-scored on first merge
   (bf16-32K read 71.4; true 97.4). Caught by a physical-impossibility
   check (baselines "improving" with context) before anything was recorded;
   fixed (literal_eval before scoring) and re-merged.

Also fixed en route: pool-runner stale-bootlog race (readiness grep matched
the previous server's banner).

## 6. Infrastructure delta

Sharded GPU-pool scheduler (`build_pool_queue.py`, `gpu_pool_runner.sh`,
`merge_shards.py`): flock-protected job queue, one worker per GPU,
arm-affinity claiming (server survives across same-arm shards), MAX_REQS +
CUDA-graph batch raised 8 → 24 with per-cell client threads scaled to
context. First production night: 117 jobs, 6 GPUs, zero idle GPU-hours,
2 transient failures auto-requeued. Client resume (io_log-based) made every
interruption cheap.

## Artifacts

Llama: `artifacts/oscar_llama31_8b/{rotations,vqa_*fp8.pt,grid/}`;
Qwen extras: `artifacts/oscar_e2e/lh/pool/`; VQ-V cells:
`artifacts/oscar_e2e/lh/{vqv_niah_32768,pool/vqv_niah_65536}`. All mirrored
lambda6 ↔ lambda7. Engine: `vendor/OSCAR-vq` @ `6f78cd6cd`.
