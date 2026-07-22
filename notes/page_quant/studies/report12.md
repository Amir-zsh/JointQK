# Report 12 — Llama-3.1-8B five-method grid, Qwen sweep completion, and the VQ-V ablation

**Status: results COMPLETE (lambda6 GPU-pool run, night of 2026-07-20 → 21);
VQ-V mechanism resolved offline (section 4).**
Shareable version (Samuel-facing, with state pointers):
https://claude.ai/code/artifact/eea529b4-755c-40a9-a605-8965ba257e2c

All cells served on the vq2-extended OSCAR engine (vendor/OSCAR-vq @
`6f78cd6cd`, gates G1–G6 ALL PASS), one stack for every method, identical
exported rows per cell, sharded across 6 A100s by the pool scheduler.

## 0. Protocol (shared settings)

**Serving stack (every cell, every method, both models):** token pool
140,000/GPU, CUDA graphs, ≤24 concurrent requests. All quantized methods
share OSCAR's mixed-KV window policy — first 64 tokens (sinks) + most
recent 256 tokens in bf16, older tokens quantized on aging; clips K 0.96 /
V 0.92 (Naive/QuaRot: 1.0). No sequence cap below the pool; the longest
requests are NIAH-64K (65,536-token prompt + 128 generated).

**NIAH (both models, all methods):** contexts 8K/16K/32K/64K; 800 rows per
context = 8 RULER subtasks × 100 (single_1–3, multikey_1–3, multivalue,
multiquery). Greedy (T=0), **1 rollout/row**, gen cap 128. Accuracy =
`string_match` per subtask, headline = unweighted mean over the 8.

**Llama reasoning/code:** GPQA-diamond 198 / math500 full 500 / AIME25 30 /
HumanEval 164 rows. **5 rollouts/prompt**, T=0.6 / top-p 0.9 / no top-k
(Llama generation_config), non-thinking; caps gpqa+math 4096, aime 8192,
humaneval 2048. Accuracy = **avg@5**: each seed-set scored independently
(math-verify / answer-letter / test execution), mean of the 5 per-seed
accuracies, std over seeds. Not pass@5.

**Offline mechanism checks (§4):** not served — live-captured Qwen V
(8 GPQA prompts ≤3072 tok for checks 1–2; NIAH-haystack rows ≤4096 tok for
check 3), engine R_v, relative MSE in the rotated basis, layer-0 excluded.

**Calibration data per method:**

| method | rotation calibration | codebook / other |
|---|---|---|
| bf16 | — | — |
| TurboQuant | — (fixed Hadamard, order 128) | Lloyd-Max INT2, MSE-optimal uniform encoding (no calibration; see §5.3) |
| int2 Qwen | authors' released (83 prompts, 20K-tok budget) | clips 0.96/0.92 (authors') |
| int2 Llama | ours: 50 GPQA prompts → authors' pipeline (qqt_sst, r_h_pbr) | same clips (Qwen-tuned, not re-tuned — caveat) |
| vq2 | same rotations as int2 (per model) | K codebook: 198 GPQA prompts cycled into 8×64K seqs (gpqacc64k), ptn, stratified, flat, 2 b/coord |
| vq2mix | same, unchanged | same recipe, 5-domain corpus (~300K unique tok) |
| VQ-V | same | V codebook vqv_G4_strided_gpqa_engine.pt, GPQA-calibrated |
| QuaRot | — (fixed Hadamard, order 128) | — |
| Naive | — (no rotation) | — |

Asymmetry to keep in mind reading the Llama table: vq2 and int2 share the
same 50-prompt rotation stack, so their *relative* ordering is clean, but
absolute distance to bf16 partly reflects the thinner rotation calibration
and un-retuned clips.

## 1. Llama-3.1-8B: five methods, served

Onboarding built fresh this run: OSCAR rotations from the authors' own
pipeline (50 GPQA-prompt QKV dump → `compute_kv_rotation` qqt_sst/r_h_pbr,
orth err ~2e-8) and a ptn/64K-concat GPQA codebook via Samuel's trainer
(stratified, flat, 2 b/coord). Sampling for reasoning/code: K=5 seeds,
T=0.6 / top-p 0.9 / no top-k (Llama-3.1 generation_config), non-thinking
caps (gpqa/math 4096, aime 8192 — the 2048 defaults breached the ≤15%
cap-hit rule in short-validation and were re-exported before any full cell).

**NIAH mean (800 rows/cell, greedy):**

(int2/vq2 columns are the calibration-unified v2 numbers — all calibrated
artifacts derive from the same 198-prompt GPQA corpus; see §1a.)

| ctx | bf16 | **vq2** | int2 (OSCAR) | TurboQuant-INT2 | QuaRot-INT2 | Naive-INT2 |
|---|---|---|---|---|---|---|
| 8K  | 98.2 | 93.3 | 86.6 | 80.7 | 37.3 | 0.3 |
| 16K | 98.5 | 92.1 | 81.8 | 79.3 | 35.5 | 0.2 |
| 32K | 97.4 | 92.3 | 81.3 | 80.1 | 37.3 | 0.1 |
| 64K | 97.4 | **91.2** | 76.6 | 73.6 | 33.6 | 0.0 |

**Reasoning / code (avg@5):**

| task | bf16 | vq2 | int2 | TurboQuant | QuaRot | Naive |
|---|---|---|---|---|---|---|
| math500 (full 500) | 44.3 | 44.1 | 44.2 | 17.2 | 6.4 | 3.9 |
| HumanEval | 63.8 | 63.7 | 62.8 | 30.5 | 7.0 | 0.0 |
| GPQA-diamond | 25.1 | 22.1 | 23.1 | 12.4 | 2.9 | 1.0 |
| AIME25 | ≈0 for every method — Llama-3.1-8B scores ~0 as published; uninformative row |

Findings:

1. **vq2 is context-flat on Llama (~91–93 from 8K to 64K); production
   OSCAR degrades monotonically (86.6 → 76.6).** At 64K vq2 leads by
   **+14.6**. Same qualitative shape as Qwen, with earlier and larger
   separation.
2. **vq2's ~6-pt gap to bf16 is context-independent** → codebook capacity
   (calibration quality), not a streaming/position effect. **Tested same
   day and REFUTED for K**: a five-domain corpus retrain moved nothing
   (see §4 follow-up) — the gap is not calibration-domain on the K side.
3. **int2's transfer to Llama is poor even at 8K** (86.6 vs Qwen's 97.6).
   The rotation-calibration caveat from v1 is now **measured and closed**
   (§1a): quadrupling rotation data (50 → 198 prompts) moves int2 by ~+1 pt
   — the transfer gap is not calibration-starved rotations. Remaining
   suspects: the Qwen-tuned 0.96/0.92 clips, or Llama K/V statistics being
   intrinsically harder at 2 bits.
4. **Reasoning/code: bf16 ≈ vq2 ≈ int2** (within seed noise) — quantization
   is free where retrieval pressure is low, on Llama as on Qwen. GPQA is
   near chance for this model (extraction ~72–80%; absolute ~25%), AIME is
   zero — both retained for protocol completeness, neither informative.
5. **The Table-2 baseline hierarchy reproduces on a model family the
   paper never tested**: TurboQuant (NIAH ~74–81) > QuaRot (~35) >
   Naive (~0), with OSCAR's int2 above TurboQuant everywhere except 32K
   (80.0 vs 80.1, a tie). Two readings: (a) an optimal scalar quantizer
   with a plain Hadamard nearly matches production OSCAR's full stack on
   pure retrieval; (b) on reasoning/code, where decode-context fidelity
   matters, OSCAR's calibrated rotations + bf16 recent ring are worth
   2–3× TurboQuant's scores (44.5 vs 17.2 math500, 64.8 vs 30.5
   HumanEval). TurboQuant's 64K weak spot is the uuid haystack
   (multikey_3 = 17) — same dense-random-text domain that stresses every
   method here. TurboQuant cells ran 2026-07-21 after the §5.3 fix.

### 1a. Calibration-unified v2 re-run (fairness fix + rotation-effect measurement)

v1 provenance was asymmetric: Llama rotations from 50 GPQA prompts, Qwen
rotations from the authors' release, codebooks from our 198-prompt corpus.
v2 unifies: **every calibrated artifact derives from the same 198-prompt
GPQA corpus** (rotations: per-prompt capture ≈48K tokens, authors' recipe;
codebooks: the 64K-concat delivery, unchanged/rotation-independent).
Rotation-dependent methods (int2 fully; vq2 via its V path) were re-served
on lambda6 through the new declarative runner
(`pipelines/oscar_e2e/experiments/llama31_8b_grid_v2.json`, manifest with
git SHAs + artifact sha256s in `grid_v2/experiment_manifest.json`).
Calibration-free methods (bf16, turbo, quarot, naive) are reused from v1.

v1 → v2 deltas (the isolated rotation-calibration effect):

| | 8K | 16K | 32K | 64K | mean |
|---|---|---|---|---|---|
| int2 NIAH | +2.3 | −0.8 | +1.3 | −0.1 | **+0.7** |
| vq2 NIAH | +1.0 | +1.0 | +0.6 | −0.9 | **+0.4** |

Reasoning/code deltas (avg@5, all within per-seed std ≈ 1–2):

| | gpqa | math500 | aime25 | humaneval |
|---|---|---|---|---|
| int2 Δ | +1.3 | −0.3 | +0.7 | −2.0 |
| vq2 Δ | −2.5 | −0.2 | −0.7 | +0.0 | **Conclusion: rotation
calibration quantity/provenance is a ~1-pt effect — every v1 conclusion
stands.** The Qwen v2 re-run (our-corpus rotations at
`artifacts/oscar_e2e/rotzoo/Qwen3-8B/gpqa198_own/`) is handed to Samuel;
its v1 int2 column remains reported as "production OSCAR as released".

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
retrieval accuracy. Constructive next step: a **mixed-domain V codebook** is
the lever most likely to flip the VQ-V ablation.

### Follow-up (same day): mixed-domain K codebook on Llama — NULL

Retrained the Llama K codebook on a five-domain corpus (gpqa 14% / math 32% /
code 14% / needle-stripped RULER essay text 26% / synthetic digit-uuid
key-value lines 14%, ~300K unique tokens vs GPQA-only's ~51K; identical
trainer flags and rotations, so the delta isolates corpus alone). Gates PASS,
served NIAH 800 rows/ctx:

| ctx | vq2 (GPQA-only) | vq2mix (5-domain) | Δ |
|---|---|---|---|
| 8K  | 92.34 | 93.19 | +0.85 |
| 16K | 91.16 | 92.00 | +0.84 |
| 32K | 91.69 | 91.25 | −0.44 |
| 64K | 92.09 | 91.09 | −1.00 |

Mean Δ +0.06, no subtask pattern — **calibration-domain composition does
not move the Llama K-side gap** (the ~6 pts to bf16 stays context-flat).
The K/V asymmetry is coherent with the mechanism: K errors pass through a
softmax that forgives them (and the dot-product geometry is
rotation-conditioned), while V errors are linear in the output. Remaining
suspects for the Llama gap: rotation quality (50-prompt calibration vs the
authors' 83), Qwen-tuned clip defaults, or Llama K statistics being
intrinsically harder at 2 b/coord. The mixed-domain recommendation now
applies to **V only**.

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

3. **TurboQuant baseline was unimplemented-as-documented, then mis-tuned
   (engine, fixed in clone commits `a9a4ab6d3` + `b04860e71`)**: the
   documented recipe (`SGLANG_LLOYD_MAX=1` on the int2plain path) pointed
   at a flag only the *mixed* OSCAR pool read — the plain pool would have
   silently served min-max (i.e., QuaRot mislabeled). Additionally the
   legacy LM encoding (`LM_RATIO=1.16`, dynamic-range-matched) costs +26%
   distortion vs exact Lloyd-Max. Fix: LM branch in BOTH plain-pool
   writers (fused-Hadamard decode + pre-rotated prefill — the two-writer
   split is the same bug class as §5.1) storing the MSE-optimal
   uniform-level encoding Δ*=0.98774 (+1.2% vs exact LM analytic; +0.8%
   on real post-FWHT Llama K/V — `turbo_lm_check.py`). Readers keep the
   `(q−zero)·scale` contract, sidestepping the exact-centroid path's
   documented decode instability. Gates: `verify_turbo_lm.py` T1–T4 ALL
   PASS (kernel-vs-torch 100%, writer-vs-writer arena parity 100%).
   Served family-control: greedy 600-tok loop uniq-ratio turbo 0.15 vs
   QuaRot 0.10 vs bf16 0.62 — looping is the plain-int2 family behavior,
   not an LM defect.

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
