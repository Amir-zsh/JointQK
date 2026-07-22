# VQ inside the OSCAR engine ("vq2") — integration notes

Shareable summary of how Samuel's group-VQ K compression was integrated
into the OSCAR authors' SGLang stack as a first-class KV-cache tier, how to
run it, and every caveat / fairness consideration we are tracking.
Written 2026-07-17 during the plan11 long-horizon evaluation.

> **State pointer (2026-07-21, for Samuel).** The canonical code lives on
> **lambda7** at `/vault/amir/efficient-llm/teamily-project` — main repo
> branch `pgq10`, engine clone `vendor/OSCAR-vq` branch `vq2-longhorizon`
> @ `6f78cd6cd`. Your 2026-07-20 handoff (F1 uint8 arena, F2 kernel parity,
> VQ-V tier, Naive/QuaRot baselines, protocol tooling) is fully merged, plus
> on top of it: V integrity gates **G4/G5/G6** in `verify_vq_engine.py`
> (ALL PASS), a **fix to your VQ-V patch** — `dequantize_prefix_kv` decoded
> the V index arena as packed int2 crumbs during chunked prefill (clone
> commit `6f78cd6cd`; write-up:
> https://claude.ai/code/artifact/02c62657-a7de-42e9-a48e-ca6343b7417f),
> your debug probes removed per the gate-landing condition, and a missing
> stream-capture guard on the VQV read-back probe. Experiments now run on
> **lambda6** (byte-identical mirror at the same path; Llama-3.1-8B
> rotations + ptn/64K codebook under `artifacts/oscar_llama31_8b/`, five-arm
> grid in flight). lambda7 GPUs are yours.

**All paths are on the shared filesystem, repo root =
`/vault/amir/efficient-llm/teamily-project` (branch `pgq10`). Run commands from that root.**

| what | absolute path |
|---|---|
| engine clone (branch `vq2-longhorizon`) | `/vault/amir/efficient-llm/teamily-project/vendor/OSCAR-vq/` |
| serve script | `/vault/amir/efficient-llm/teamily-project/pipelines/oscar_e2e/serve_oscar.sh` |
| integrity gates | `/vault/amir/efficient-llm/teamily-project/pipelines/oscar_e2e/verify_vq_engine.py` |
| client / wave driver | `/vault/amir/efficient-llm/teamily-project/pipelines/oscar_e2e/run_prompts_client.py`, `run_longhorizon_wave.sh` |
| your codebook (pinned snapshot) | `/vault/amir/efficient-llm/teamily-project/third_party/samuel_vq/codebooks/vqa_G4_strat_flat_ptn_gpqacc64k_fp8.pt` |
| exported eval rows | `/vault/amir/efficient-llm/teamily-project/artifacts/prompt_rows/` |
| all result cells (metrics.json per cell) | `/vault/amir/efficient-llm/teamily-project/artifacts/oscar_e2e/lh/<config>/<task>/`, `/vault/amir/efficient-llm/teamily-project/artifacts/oscar_e2e/{bf16,int2,vq2}/niah_*/` |
| study report / review / kernel A/B | `/vault/amir/efficient-llm/teamily-project/notes/page_quant/studies/report11.md`, `plan11_review_findings.md`, `/vault/amir/efficient-llm/teamily-project/artifacts/kernels/idx_dtype_ab.json` |

## Where it lives

- **Engine**: `/vault/amir/efficient-llm/teamily-project/vendor/OSCAR-vq` — a git clone of the vendored OSCAR
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

File-by-file (all paths inside `/vault/amir/efficient-llm/teamily-project/vendor/OSCAR-vq/sglang-research/python/`):

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
#   add --samples 4
# aggregate acc@K with paired bootstrap CIs vs the bf16 anchor:
.venv/bin/python pipelines/eval/aggregate_acck.py \
    --cells bf16=... int2=... vq2=... --anchor bf16 --out <summary.json>
```

Operational notes:
- **Resume**: an interrupted client run (io_log.jsonl present, metrics.json
  absent) resumes automatically — completed (row, sample) pairs are skipped
  and the log is appended. Kill/restart is safe.
- **Decode window**: `RECENT_TOKENS=<W>` env on serve_oscar.sh overrides the
  bf16 recent-ring length (default 256 = the OSCAR band). W∈{64,1024} sweep
  is set up but deferred.
- **Pool budget**: the mixed-KV pool hard-crashes (no retraction) when
  concurrency × max_new_tokens outruns `--max-total-tokens` (140K on the
  A100-40GB). For 32K-token generations run ≤4 concurrent requests; 64K
  prompts fit only ~2 at a time.
- **One wave per arm**: `pipelines/oscar_e2e/run_longhorizon_wave.sh --arm
  {bf16,int2,vq2} --gpu N --port P` runs gpqa+math500+aime25 end to end
  (resumable per task).

Your codebook bundle loads as-is (fp16 or fp8-e4m3 codebooks both
handled; snapped to e5m2 at load).

## Results (final, 2026-07-17 — full record in report11.md)

**Integrity gates**: all green (roundtrip distortion within 0.14% of the
reference codec; mixed-tier softmax equivalence 2e-4; kernel vs torch
reference exact).

**Long-horizon, served, one engine, identical rows** (thinking mode, K=4
samples at T=1.0/top_p .95/top_k 40, avg@4, paired row-bootstrap CIs;
GPQA/AIME capped at 32768 per the OSCAR authors' protocol, math500 at 16384
after an 8K cap breached our cap-hit telemetry rule):

| avg@4 | GPQA-diamond | math500 (n=200) | AIME-25 | 3-task mean |
|---|---|---|---|---|
| bf16 | 58.6 | 96.1 | 66.7 | 73.8 |
| OSCAR INT2 | 54.2 | 95.0 | 65.0 | 71.4 |
| **vq2 (your codebook)** | **57.7** | **95.3** | **68.3** | **73.8** |

- vq2 is statistically indistinguishable from bf16 on all three tasks
  (GPQA −0.9 [−4.0, +2.3]; math −0.9 [−2.1, +0.4]; AIME +1.7 [−6.7, +9.2]).
- Production OSCAR pays **−4.4 [−7.8, −1.1] (significant)** on GPQA and
  shows a longer-trace degeneration signature on every task.
- Head-to-head vq2 ≥ INT2 everywhere: +3.5 GPQA (SIG), +0.3 math, +3.3 AIME.
- Cap sensitivity worth knowing: at a tight 8K math cap vq2 lost −3.1 (SIG)
  — pure truncation interaction (vq2's traces run slightly longer), gone at
  16K. Tight generation caps punish VQ before they punish INT2.

**NIAH, served, 800 rows, greedy**:

| mean string-match | 32K | 64K |
|---|---|---|
| bf16 | 98.3 | 86.3 |
| OSCAR INT2 | 74.2 | 23.4 |
| **vq2 (your codebook)** | **96.6** | **76.8** |

vq2-served matches its own post-hoc numbers (95.9/77.6) within a point at
both contexts — **the streaming-quantization regime that costs production
OSCAR 24–63 points costs VQ nothing**. The served 32K number also equals
your HF-side 96.6 exactly. vq2's remaining 64K gap to bf16 (−9.5) sits in
multikey_2/3, same as post-hoc — codebook capacity at extreme positions,
not a serving effect.

**Decode speed** (bs=1, A100): INT2 67.5→57.0 tok/s at 8K→32K
(1.0×→1.16× of bf16); vq2 60.9→44.6 (0.90×/0.78× of INT2). The pure kernel
is at parity (report10 tuned-vs-tuned); the gap is integration overhead —
per-head q-map GEMMs + the torch flush encode — both optimizable.

**Queued next** (relevant to you): mixed-domain 64K-concat codebook
(GPQA+math+code) as the paper's primary — your gpqacc64k stays as the
domain ablation; HumanEval + LiveCodeBench v6 for OSCAR-suite parity;
uint8 index arena (accuracy-identical; our dtype A/B says the speed
difference is config-dependent second-order —
artifacts/kernels/idx_dtype_ab.json).

## Speed & timing caveats

How every number was measured, and what it does / does not mean:

1. **Served decode tok/s are end-to-end, not kernel numbers**: bs=1, HTTP
   client wall time with prefill subtracted (max_new_tokens=1 baseline vs
   513), after warmup, CUDA-graphed decode, default server split count
   (`triton_attention_num_kv_splits=8`). They include all engine overheads
   by design — that's serving reality — and are NOT comparable to the
   tuned per-kernel ms/layer numbers in report10.
2. **vq2's 0.90×/0.78× of INT2 is integration overhead, not the kernel.**
   Tuned-vs-tuned at the kernel level (report10/A10-2): your VQ8 kernel is
   at parity with the authors' INT2 kernel at 32K (0.073 vs 0.075
   ms/layer) and 1.21× at 128K. The served gap comes from the per-head
   q-map GEMMs (INT2's per-layer rotation is one GEMM; vq2 does 8 small
   ones per layer per step), the eager torch flush-encode every 8 decode
   steps, and untuned `SGL_VQ2_*` splits. All three are optimizable;
   none affect accuracy.
3. **Never compare under shared defaults.** The two kernels tune in
   opposite directions (INT2 wants big tiles + deep pipelining, VQ wants
   BT=64/warps=2 around the gather); each has its own env knobs
   (`SGL_INT2_*` / `SGL_VQ2_*`). Any shared-default comparison
   structurally mis-serves one side — this is exactly how the retracted
   "VQ is 3.7× behind, architectural" claim happened (two-phase harness +
   untuned config; corrected in report10 Amendment A10-2).
4. **Quantized decode beats bf16 only past ~8–16K context.** Decode
   attention is bandwidth-bound and grows with context; at 8K the smaller
   reads roughly cancel dequant overhead (INT2 1.00× of bf16), at 32K
   INT2 is 1.16× and the gap keeps widening. Below that, the case for
   quantization is memory capacity, not speed (~144 KB/token bf16 vs
   ~10× less quantized on Qwen3-8B).
5. **Prefill**: vq2's chunked torch VQ-encode makes prefill ~1.45× INT2's.
   At 64K the wall-clock is dominated by something else entirely: only
   ~2 concurrent 65K prompts fit the 140K-token pool, so throughput is
   pool-bound, not kernel-bound (TP=2 or an 80 GB card is the fix).
6. **Wave wall-times reflect both speed and trace length**: GPQA 792
   traces took bf16 ~4.4 h, INT2 ~4.9 h, vq2 ~7.3 h — vq2's extra time is
   part decode speed, part its traces averaging ~800 tokens longer. The
   bs=1 tok/s table understates wave throughput (waves ran continuous
   batching at ≤8 requests).
7. **Hardware scope**: A100-40GB, triton/triton backends (fa3 prefill is
   Hopper-only) — the authors' published numbers are H100. Absolute
   timings also drift with machine load (our idx-dtype A/B absolutes
   shifted ~15% with 3 neighbor GPUs busy; within-run relatives held), so
   compare only within one measurement session.
8. **int16 vs uint8 indices**: speed difference is second-order and
   config-dependent (±5–11% either way,
   `artifacts/kernels/idx_dtype_ab.json`); memory difference (4 vs 2
   b/coord index stream) is first-order. uint8 is the deployable choice.

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
   out-of-domain on math. Final resolution: the apparent −3.1 math500 loss
   at the 8K cap turned out to be cap truncation, not domain — at 16K vq2
   is at the anchor. Domain sensitivity IS real where we can isolate it
   (swapping to a LongBench-domain codebook costs 3–12 NIAH points), which
   is still the argument for a mixed-domain retrain as the paper primary.
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

## Qwen calibration-unified re-run (v2) — handed to Samuel

Amir's side stopped the Qwen v2 run (2026-07-21) per division of labor; all
assets are ready at identical absolute paths on lambda6/7 under
/vault/amir/efficient-llm/teamily-project:

- **Rotations from OUR corpus** (the fairness fix — v1 int2 used the
  authors' released rotations): `artifacts/oscar_e2e/rotzoo/Qwen3-8B/gpqa198_own/`
  — 198 GPQA-Diamond prompts (same corpus as gpqacc64k), authors' pipeline
  (qqt_sst / r_h_pbr), orth err ~2e-8, validated. Built by
  `pipelines/oscar_e2e/recalibrate_rotations.sh`.
- **Declarative spec** (every argument): `pipelines/oscar_e2e/experiments/qwen3_8b_niah_v2.json`;
  replication recipe in `pipelines/oscar_e2e/experiments/README.md`
  (run_experiment.py → ARMS_DIR workers → merge_shards).
- **Partial results**: 4 of 32 shard jobs completed before stop, kept under
  `artifacts/oscar_e2e/lh/grid_v2/` (re-running the spec resumes — completed
  shards are skipped); queue rows marked HANDOFF in `logs/pool_queue_qwen_v2.tsv`.
- Served smoke on the new rotations passed (int2, coherent generation).

## Shareable artifacts (2026-07-21)

- **Report 12** — Llama six-method grid (calibration-unified v2), Qwen sweep,
  VQ-V ablation + mechanism, protocol, TurboQuant handoff:
  https://claude.ai/code/artifact/eea529b4-755c-40a9-a605-8965ba257e2c
- **Report 13** — position-coverage cliff (RULER-128K), TP serving,
  R1-Distill long-CoT grid:
  https://claude.ai/code/artifact/0cb61471-28cf-41f2-959a-009ee4cd6228
- **Integration handoff** — calibration unification, TurboQuant-INT2 fix,
  TP support; engine cherry-picks `a9a4ab6d3` `b04860e71` `0ecb1ab7c`,
  gates, absolute paths, Qwen v2 resume command:
  https://claude.ai/code/artifact/7e1b0e79-93b7-4b11-9336-b63f39a0c0cf
