# Report 11 — Long-horizon evaluation on the vq2-extended OSCAR engine (plan11)

**Status: IN PROGRESS (overnight run of 2026-07-16 → 17). Sections marked
[MORNING] fill in when the C1 waves land.**

## What this study asks

pgq10 established the short-generation story (NIAH/LongBench F1, kernel
speed). The one regime never measured: **long-horizon generation** — thinking
traces of thousands of tokens where decode-time quantization compounds. Three
arms, all served on the SAME engine stack (vendor/OSCAR-vq clone, branch
`vq2-longhorizon`), same sampler, same rows:

- **bf16** — serving-stack control.
- **int2** — production OSCAR (authors' rotation, K+V INT2, 64/256 HP band).
- **vq2** — group-VQ K tier inside the OSCAR engine (this study's build):
  Samuel's fused-gather decode kernel, his gpqacc64k ptn codebook, V on
  OSCAR's INT2 path.

Tasks (OSCAR authors' protocol: T=1.0, top_p=0.95, top_k=40; K=4 samples;
avg@4 primary): GPQA-diamond 198 rows @8192, math500 first-200 @4096
(math-verify), AIME-25 30 rows @16384. Thinking mode on (Qwen3 `<think>`).

## Pre-registered bars

- **B-LH1**: avg@4 Δ(vq2 − bf16) CI-lower ≥ −4 pts on GPQA and math500.
- **B-LH2**: Δ(int2 − bf16) — does OSCAR's streaming penalty recur at
  generation time? Either sign is a finding.
- **B-LH3**: paired vq2 − int2 contrast (the comparison-page row).

## The vq2 engine build (Phase E)

`--kv-cache-dtype int2` + mixed-KV windows + `SGLANG_VQ_CODEBOOK_PATH` →
the unified pool's K quant tier becomes group-VQ:

- **Storage space**: both K tiers hold the residual `r = (k − mean) @ F`
  (per-head forward map). Queries map with `q @ inverse.T`, so quant and HP
  scores share the same per-query `−q·mean` shift — softmax-invariant, no
  mean bias anywhere in kernels.
- **Arenas**: K quant = int16 indices `[slots+1, 8, 32]` (stacked across
  layers with per-layer views + one trash row for sync-free shape-static
  writes); ptn RMS scale reuses `k_scales_zeros` slot 0 (bf16 — slightly
  better than Samuel's fp8 scale; charged in honest rates).
- **Decode kernel**: Samuel's fused VQ8 gather core (one int32 load per
  4-coord codeword, fp8 bitcast planes, join-interleave) transplanted into
  the int2 unified stage-1 structure (paged kv_indices, split-K scratch,
  HP+quant two-launch + tier-agnostic stage-2). ptn scale folds into the
  score columns post-dot. **fp8 is e5m2, not e4m3** — sm80 Triton only
  bitcasts fp8e5 (this is also exactly Samuel's measured kernel). Encode
  assigns against e5m2-snapped centroids, so the encoder is nearest-neighbor
  for what the decoder reconstructs; snap cost ≈ +0.0003–0.0014 absolute K
  distortion.
- **Write side**: prefill/extend K encode = chunked torch bmm-argmax per
  layer; decode aging reuses the int2 flush plan/apply (K_VQ constexpr skips
  the in-kernel K pack) + a batched all-layer torch VQ encode over the plan
  tensors. V untouched (int2 clip path, absorbed R_v).
- Window policy unchanged: prefix 64 / recent 256 (their HP band) = the
  user's "quantize after a window, in groups" policy (N_Q=8 pages).

## Gates (Phase V)

**V1 integrity — PASS** (`pipelines/oscar_e2e/verify_vq_engine.py`):
- G1 roundtrip vs `GroupVQCompressor` reference: distortion ratio ≤ 1.0014,
  worst per-token excess ≤ 0.9%. (Elementwise identity is ill-posed for a
  dense 4-dim/256 codebook — fp32-vs-double flips a few % of assignments
  between near-equidistant centroids at zero distortion cost; the gate is
  rate-distortion parity, elementwise rel reported as info.)
- G2 stored-space softmax equivalence (mixed quant+HP): fp32 1.9e-4,
  bf16 (served dtype) 2.8e-4.
- G3 fused kernel vs torch reference: PASS.

**V3 efficiency — PASS** (bs=1, prefill-subtracted, A100, default server
splits=8):

| decode tok/s | ~8K ctx | ~32K ctx |
|---|---|---|
| bf16   | 67.7 | 49.2 |
| int2   | 67.5 | 57.0 |
| vq2    | 60.9 | 44.6 |
| vq2/int2 | **0.90×** | **0.78×** |

Gate ≥ 0.75× of int2: met. The residual gap vs the tuned-kernel parity of
report10 is engine-side (per-head q-map einsum + torch flush encode every 8
steps), not the decode kernel. Prefill is ~1.45× int2's (chunked torch VQ
encode) — acceptable for the study.

**Codebook decision (A10-1 repro, landed tonight)**: Mode-A NIAH f=0.25,
same harness, 2 bits K:

| codebook | 32K | 64K |
|---|---|---|
| Samuel gpqacc64k (b277bddf) | **95.9** | **77.6** |
| ours flat_ptn (d4209fca)    | 87.6 | 65.8 |

His long-context calibration wins by 8–12 pts → **the vq2 arm ships his
codebook**. (Also confirms his reported 96.6/79.5 within quarter-sample
noise.)

**V2 served-quality — PASS, with a headline**: our-codebook NIAH-32K
**800-row served = 93.7** vs Mode-A same-codebook 87.6 (f=0.25). The served
engine — chunked prefill attending over already-quantized keys, decode-time
aging, the full streaming regime — is 6 points BETTER than the post-hoc
harness cell (the 64/256 HP band accounts for it). Contrast production
OSCAR int2 in the same stack: 74.2 served vs ~96 post-hoc Mode-A.
**Streaming quantization is where OSCAR's int2 loses 20+ points and VQ
loses nothing.** His-codebook full 800-row served NIAH-32K: **96.6** vs 95.9 Mode-A
(f=0.25) — within 0.7 pts (bar was 5), again slightly better served than
post-hoc, and exactly matching Samuel's HF-side 96.6. Served triptych at
32K: bf16 98.3, **vq2 96.6**, production OSCAR int2 74.2. (The interim
100-row gate slice was single-subtask only — superseded by this cell,
`artifacts/oscar_e2e/lh/v2_vq2gpqacc_niah800`.)

## C0-lite gates

- 30 bf16-served GPQA thinking traces: FAILED at the original 8192 cap
  (cap-hit 30% swallows the answer line, extraction 70%); the OSCAR
  authors' own GPQA eval uses max_tokens=32768 — at corrected caps
  (gpqa/aime 32768, math500 8192) the gate PASSES clean: extraction 100%,
  cap-hit 0%, accuracy 66.7% (mean trace 6.7K tokens).
- W (recent-tokens) shipped at 256 = production OSCAR's band, keeping the
  int2 comparison apples-to-apples. The {64, 1024} sweep was DEFERRED by
  user decision (2026-07-17 evening) — servers were booted and torn down;
  rerun later via `RECENT_TOKENS=<W> serve_oscar.sh --vq2` on 50
  math500@16K rows per point.

## Incidents (all fixed, in the open)

1. **fp8e4nv unsupported on sm80** → e5m2 packing (matches Samuel's kernel).
2. **vq_map_k einsum non-contiguous** → CUDA graph capture failure at boot;
   `.contiguous()`.
3. **flashinfer sampling JIT** dies on missing `curand.h` at the FIRST
   temperature>0 request (greedy never touches it — why every pgq10 run was
   fine) → `--sampling-backend pytorch` for all arms.
4. **math-verify scorer bug (pre-existing)**: bare-latex golds misparse
   truthy (`\left(3, \frac{\pi}{2}\right)` → `3`), so the `$...$` fallback
   never fired; tuple/interval answers scored 0. Fixed (try both gold
   readings). Retro impact: served bf16 math500 re-scores 0.748 → **0.838**;
   same-direction bias across arms, so pgq10 deltas stand, but absolute math
   numbers in older cells were deflated. Old cells re-scorable offline.

## C1 results [MORNING]

Early landings (raw, CIs pending the full aggregation):

| avg@4 | gpqa | math500@8K | aime25 | 3-task mean |
|---|---|---|---|---|
| bf16 | 0.586 (cap 0%) | 0.901 (cap 19.9%) | 0.667 (cap 15.8%) | 71.8 |
| int2 | 0.542 (cap 1%)  | 0.898 (cap 21.0%) | 0.650 (cap 15.8%) | 69.7 |
| vq2  | 0.577 (cap 1.3%) | 0.870 (cap 21.6%) | **0.683** (cap 15.0%) | 71.0 |

Paired contrasts (10k row-bootstrap):

| Δ avg@4 | gpqa | math500@8K | aime25 (n=30, descriptive) |
|---|---|---|---|
| int2 − bf16 (B-LH2) | **−4.4 [−7.8, −1.1] SIG** | −0.4 [−2.0, +1.4] | −1.7 [−10.0, +5.8] |
| vq2 − bf16 (B-LH1)  | −0.9 [−4.0, +2.3] | **−3.1 [−5.1, −1.3] SIG** | +1.7 [−6.7, +9.2] |
| vq2 − int2 (B-LH3)  | **+3.5 [+0.1, +6.9] SIG** | **−2.8 [−5.0, −0.6] SIG** | +3.3 [−3.3, +10.0] |

On the OSCAR-style multi-task mean, int2 sits −2.1 under bf16 and vq2
−0.8 — vq2 recovers ~60% of production OSCAR's quality gap while being the
numerically best method on AIME (above the anchor). int2's traces run
longer than bf16's on every task (gpqa +507, math +16, aime +1847 mean
tokens) — a consistent mild-degeneration signature even where its accuracy
holds; vq2 shows a smaller version of the same (+806/−72/+925).

Cap-robustness reruns at 16384 (fixes the ~20% math500 cap-hit):

| math500@16K avg@4 | value | cap |
|---|---|---|
| bf16 | 0.961 | 5.0% |
| int2 | 0.950 | 5.8% |
| vq2  | [pending — last cell] | |

**The two quantizers fail in different places.** int2 pays significantly on
GPQA and nothing on math500; vq2 is at the bf16 anchor on GPQA and pays
significantly on math500. Cap-hit rates are near-identical across arms, so
it is not a truncation artifact. Leading hypothesis (ties to review finding
F2): the VQ codebook is k-means on GPQA-domain keys — codebooks are more
domain-specific than an orthogonal rotation, so vq2 excels in-calibration-
domain and pays out-of-domain, while int2's rotation generalizes flat.
Testable next: a math-domain (or mixed-domain) codebook retrain.

On the pre-registered bars: B-LH1 is borderline on gpqa (CI-lower −4.04 vs
the −4 bar) and NOT met on math500; B-LH2 is a confirmed int2 penalty on
gpqa; B-LH3 is significant in opposite directions by task.

Caveat: math500's 8192 cap hits 19.9% of bf16 traces (the C0 probe was
gpqa-only; math traces run longer than its mean suggested). Same rows/cap
across arms, so paired deltas stand; absolutes are deflated a few points.
[MORNING option: 16384-cap rerun of math500 on all three arms.]

- avg@4 / pass@4 tables per task × arm, aggregate_acck.py output.
- B-LH1/2/3 verdicts with paired bootstrap CIs.
- Honest rates: vq2 K = 2.0 b/coord + bf16 ptn scale/token/head + ring;
  int2 = 2 b/coord + scales; both + 64/256 HP band amortized.
- Telemetry: cap-hit, mean trace length per arm (degeneration signal).

## Mid-run implementation review

A line-by-line review of the engine edits, eval plumbing, and arm symmetry
was done while the waves ran: see
[`plan11_review_findings.md`](plan11_review_findings.md). Nothing
invalidates the comparison; the two mandatory disclosures are **F1** (vq2 K
memory is 4.5 b/coord as-built vs 2.25 b/coord information — int16 indices
are a speed choice, packable to parity) and **F2** (both arms' calibrations
are GPQA-domain — symmetric). F6 lists two engine hardening TODOs (extend
fallback assert; retraction instead of hard-crash on pool exhaustion).

## Artifacts

- Engine: `vendor/OSCAR-vq` @ `vq2-longhorizon` (70ba321a8, 6fa08ce3c).
- Gates: `pipelines/oscar_e2e/verify_vq_engine.py` (run log in lh/ dir).
- Waves: `logs/lh_{bf16,int2}.log`, `logs/lh_vq2_chain.log`,
  outputs `artifacts/oscar_e2e/lh/<arm>/<task>/`.
