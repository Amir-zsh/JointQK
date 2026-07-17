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

**V2 served-quality**: [MORNING] our-codebook NIAH-800 served vs Mode-A
87.6; his-codebook NIAH-100 served vs Mode-A 95.9 (5-pt bar). Both cells
under `artifacts/oscar_e2e/{vq2,lh/v2_vq2gpqacc_niah100}`.

## C0-lite gates

- 30 bf16-served GPQA thinking traces @8192: [MORNING — extraction ≥90%,
  cap-hit ≤15%].
- W (recent-tokens) shipped at 256 = production OSCAR's band, keeping the
  int2 arm apples-to-apples; {64, 1024} sweep runs as an informational
  side-cell if GPU 5 frees up.

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

- avg@4 / pass@4 tables per task × arm, aggregate_acck.py output.
- B-LH1/2/3 verdicts with paired bootstrap CIs.
- Honest rates: vq2 K = 2.0 b/coord + bf16 ptn scale/token/head + ring;
  int2 = 2 b/coord + scales; both + 64/256 HP band amortized.
- Telemetry: cap-hit, mean trace length per arm (degeneration signal).

## Artifacts

- Engine: `vendor/OSCAR-vq` @ `vq2-longhorizon` (70ba321a8, 6fa08ce3c).
- Gates: `pipelines/oscar_e2e/verify_vq_engine.py` (run log in lh/ dir).
- Waves: `logs/lh_{bf16,int2}.log`, `logs/lh_vq2_chain.log`,
  outputs `artifacts/oscar_e2e/lh/<arm>/<task>/`.
