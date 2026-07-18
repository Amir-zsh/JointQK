# VQ inside the OSCAR engine ("vq2") — integration notes

Shareable summary of how Samuel's group-VQ K compression was integrated
into the OSCAR authors' SGLang stack as a first-class KV-cache tier, how to
run it, and every caveat / fairness consideration we are tracking.
Written 2026-07-17 during the plan11 long-horizon evaluation.

## Where it lives

- **Engine**: `vendor/OSCAR-vq` — a git clone of the vendored OSCAR
  sglang-research tree, branch **`vq2-longhorizon`** (the vendored original
  is untouched). Main commits: `70ba321a8` (the vq2 tier), `6fa08ce3c`
  (fp8-e5m2 packing). The serve script points `PYTHONPATH` at the clone,
  so all three evaluated configurations (bf16 / OSCAR int2 / vq2) run the
  same engine build.
- **Activation**: env var `SGLANG_VQ_CODEBOOK_PATH=<bundle.pt>` on top of
  the normal OSCAR configuration (`--kv-cache-dtype int2` + mixed-KV
  windows). When unset, the engine behaves exactly as before — the vq2
  additions are inert, which is how the bf16 and int2 baselines run on the
  same build.

File-by-file (all paths inside `vendor/OSCAR-vq/sglang-research/python/`):

| file | change |
|---|---|
| `sglang/srt/mem_cache/vq_codebook.py` | **new** — bundle loader (your trainer's schema: forward/inverse/mean per (layer, head), 32 codebooks of 256×4 per head, `pertoken_norm`), fp8 packing, per-head q/k maps, torch nearest-centroid encode |
| `sglang/srt/mem_cache/unified_kv_pool.py` | K quant arena becomes int16 indices `[slots+1, 8 heads, 32 groups]` (stacked across layers with per-layer views + one "trash" row); per-token-norm scale reuses slot 0 of the existing `k_scales_zeros` layout; per-head rotation at HP write; VQ encode on prefill/extend and on decode-time aging (`vq_flush_k`) |
| `sglang/QuantKernel/gpu_flush_int2.py` | `K_VQ` constexpr: the fused flush kernel skips its in-kernel K int2 pack (V + the req_to_token remap unchanged); the caller then VQ-encodes the same flush plan in torch, all-layers-at-once, shape-static (no host sync) |
| `sglang/srt/layers/attention/triton_ops/decode_attention.py` | `_fwd_grouped_kernel_stage1_quant_vq2` — **your fused gather core verbatim** (one int32 load per 4-coord codeword, four fp8 bitcast planes, the (p0,p2),(p1,p3) join-interleave, `tl.dot` online softmax) transplanted into their unified stage-1 shell (paged kv_indices, split-K scratch, HP+quant two-launch, tier-agnostic stage-2). Per-token-norm scale folds into the score columns after the dot. Tuning knobs `SGL_VQ2_{BLOCK_N,BLOCK_H,NUM_WARPS,NUM_STAGES}`, defaults from our A100 sweep (64/8/2/2) |
| `sglang/srt/layers/attention/quantized_kv_prefill.py` | per-head q/k maps for chunked prefill; VQ prefix K dequant (mixed HP+quant, token-chunked) |
| `sglang/srt/layers/attention/triton_backend.py` | decode dispatch: per-head q map + the vq2 unified path next to the int2 one |
| `sglang/srt/model_executor/pool_configurator.py`, `sglang/srt/environ.py` | memory accounting for the int16 arena; env registration |

Repo-side (main repo, committed on `pgq10`): `serve_oscar.sh --vq2`,
`verify_vq_engine.py` (integrity gates), the long-horizon task/scoring
stack, `plan11_review_findings.md`, `report11.md`.

## The design in one page

**Storage space.** Both K tiers (bf16 recent-window ring AND the quantized
tier) store the residual `r = (k − mean) @ forward` per head. Queries are
mapped with `q @ inverse.T`. Then for every key, quantized or not,
`score = q·k − q·mean`: the mean term is constant across keys for a given
query, so softmax is invariant and **no mean bias appears anywhere in the
kernels** — your kernel body runs unmodified. (Constraint: this trick
requires no tanh logit-capping, which holds for Qwen3; a Gemma-class port
would need the mean folded pre-cap.)

**Quantized K format.** 32 × int16 group indices + one per-token-norm RMS
scale per (token, head) in the existing scale slot (stored bf16/fp32 — a
touch better than an fp8 scale; disclosed in the rate accounting).
V stays on OSCAR's int2 path (clip 0.92, absorbed V rotation), identical
between the int2 and vq2 configurations — so the two differ **only** in
the K tier, which is the comparison we want.

**fp8 format: e5m2, not e4m3.** A100 (sm80) Triton can only bitcast
`fp8e5` — e4m3 bitcast is sm89+. This is also exactly what your
bench_vint2 kernel used, so the speed provenance is unchanged. The encoder
assigns nearest-neighbor against the e5m2-*snapped* centroids (what the
decoder actually reconstructs), keeping encode/decode consistent. Measured
snap cost on our bundle: +0.0003–0.0014 absolute K distortion.

**Write paths.** Prefill/extend: per-head rotate + chunked bmm-argmax
assign in torch (~1 ms per 8K-token chunk across all heads). Decode-time
aging reuses the engine's flush plan (which HP slots demote into which
quant page) and VQ-encodes all 36 layers in a single batched einsum;
invalid rows are diverted to a trash row so everything stays shape-static
in the CUDA-graph-adjacent scheduler path.

## How to run

```bash
# integrity gates (roundtrip parity vs the reference compressor, mixed-tier
# softmax equivalence, kernel vs torch reference):
PYTHONPATH=vendor/OSCAR-vq/sglang-research/python \
  .venv-oscar/bin/python pipelines/oscar_e2e/verify_vq_engine.py --gpu 1

# serve (bf16 / int2 / vq2 all from the same build):
bash pipelines/oscar_e2e/serve_oscar.sh --bf16 --gpu 3 --port 30803
bash pipelines/oscar_e2e/serve_oscar.sh        --gpu 2 --port 30802   # OSCAR int2
bash pipelines/oscar_e2e/serve_oscar.sh --vq2  --gpu 1 --port 30801 \
    --vq-codebook third_party/samuel_vq/codebooks/vqa_G4_strat_flat_ptn_gpqacc64k_fp8.pt

# drive exported rows (greedy or K-sample):
.venv/bin/python pipelines/oscar_e2e/run_prompts_client.py \
    --rows artifacts/prompt_rows/niah_32768_qwen.jsonl --port 30801 \
    --out artifacts/oscar_e2e/vq2/niah_32768
# long-horizon (K=4, T=1.0/top_p .95/top_k 40 — the OSCAR driver protocol):
#   add --samples 4; aggregate with pipelines/eval/aggregate_acck.py
```

Your codebook bundle loads as-is (fp16 or fp8-e4m3 codebooks both
handled; snapped to e5m2 at load).

## Status snapshot (as of writing; full numbers in report11)

- **Integrity gates**: all green. Engine reconstruction within 0.14%
  distortion of the reference compressor; mixed-tier softmax equivalence
  2e-4; fused kernel matches a torch reference.
- **Served NIAH-32K, 800 rows, greedy**: vq2 (our retrained codebook)
  **93.7** vs 87.6 for the same codebook applied post-hoc — VQ *gains*
  under real streaming (chunked prefill over already-quantized keys),
  where production OSCAR int2 drops ~96 → 74.2 in the same stack. Your
  gpqacc64k codebook reproduced at 95.9/77.6 (32K/64K) in our harness and
  ships as the evaluated configuration; its full served NIAH run is in
  flight.
- **Decode speed** (bs=1, A100): int2 67.5→57.0 tok/s at 8K→32K
  (1.0×→1.16× of bf16); vq2 60.9→44.6 (0.90×/0.78× of int2). The pure
  kernel is at int2 parity (report10); the gap is integration overhead —
  per-head q-map einsum (int2's per-layer rotation is one GEMM), the
  torch flush encode every 8 steps, untuned splits. All fixable.
- **Long-horizon (avg@4, K=4 sampled thinking traces)**: GPQA-diamond —
  bf16 58.6, int2 54.2 (**−4.4, significant**), vq2 57.7 (−0.9, at the
  anchor; **+3.5 over int2, significant**). math500 — bf16 90.1, int2
  89.8, vq2 87.0 (**−3.1, significant**). The two methods fail in
  different places; see fairness note 3.

## Caveats and fairness / integrity notes

1. **Memory vs information rate.** As built, vq2's K tier uses **4.5
   bits/coord of memory** (int16 indices, matching your bench_vint2 at
   the pinned commit 3c65507 — the 0.303 ms kernel we ported — plus scale
   slots) while carrying **2.25 bits/coord of information** (2.0 code +
   fp32 scale; no zero-point) vs int2's 2.5. Your live tree has since
   made uint8 the default ("the correct/deployable packing"); our A/B
   measured uint8 ~5% slower on decode (0.302→0.317 ms/layer). Switching
   the engine arena to uint8 is a two-line change with identical accuracy
   (stored values unchanged) — queued post-study. Until then,
   memory-footprint claims must carry this footnote.
2. **e5m2 snap.** Decode reconstructs e5m2-snapped centroids, not the
   fp16/e4m3 values your HF evaluations used. Snap cost measured tiny
   (see above) and the encoder is consistent with it, but absolute
   numbers are not bit-comparable to your HF-side runs.
3. **Calibration domain.** gpqacc64k is calibrated on GPQA-concat traces —
   in-domain for the GPQA eval. We checked the symmetry: the OSCAR
   authors' rotations are *also* calibrated from GPQA prompt dumps (their
   own recipe), so both methods are in-domain on GPQA and equally
   out-of-domain on math. The long-horizon results fit this: vq2 is at
   the bf16 anchor on GPQA and pays ~3 pts on math500, while int2 shows
   the opposite pattern — consistent with codebooks being more
   domain-specific than orthogonal rotations. A math/mixed-domain
   codebook retrain is the obvious follow-up.
4. **Sampling.** All configurations share one engine build, one sampler
   backend (pytorch — flashinfer's sampling JIT is broken on this box),
   identical prompt files, identical caps. No per-request seeds (SGLang
   native API limitation): K samples are independent draws, paired
   analysis is by row, not by seed.
5. **Answer extraction.** GPQA uses the *last* `Answer: X` match (thinking
   traces restate the format mid-reasoning; simple-evals uses the first
   match). Identical across configurations, but absolute GPQA numbers are
   not bit-comparable to simple-evals-scored runs.
6. **A scorer bug we fixed mid-study** (affects older math numbers
   everywhere, in every configuration equally): math-verify misparses
   bare-latex gold answers (a tuple gold parses truthy as its first
   scalar), so tuple/interval answers scored 0. Fixed; served bf16
   math500 re-scores 0.748 → 0.838. Deltas between methods stand; old
   absolute math numbers were deflated.
7. **Engine sharp edges** (pre-existing, hit during the run): the mixed-KV
   pool hard-crashes on quant-slot exhaustion instead of retracting like
   vanilla SGLang — bound `concurrency × max_new_tokens` under the token
   pool (we run AIME-length generations at concurrency 4 on the
   40 GB A100). And the non-quantized prefill fallback path would read
   the raw quant arena as values (garbage for int2 AND vq2) — unreachable
   in our configs, assert pending.
8. **Softcap constraint.** The no-mean-bias trick relies on softmax shift
   invariance; models with tanh logit-capping need the mean handled
   in-kernel instead.

Full review record: `notes/page_quant/studies/plan11_review_findings.md`.
Study report (bars, CIs, telemetry): `notes/page_quant/studies/report11.md`.
