# Report 13 — Codebook position coverage (gpqacc64k vs gpqacc128k), RULER-128K, and the R1-Distill long-CoT axis

**Status: position-coverage study COMPLETE; R1-Distill-Llama-8B onboarding
+ grid in flight (section 4 updates as cells land).**

Engine vendor/OSCAR-vq @ `0ecb1ab7c` (TP rank-sharded codebooks; custom
all-reduce root-cause-fixed — see §3). All Llama cells on
rotations_gpqa198, out_root `artifacts/oscar_llama31_8b/grid_v2/`,
declarative specs + manifests per run.

## 1. Headline: codebook position coverage is a cliff, not a slope

gpqacc128k = the identical 198-prompt GPQA corpus, identical 524K-token
budget, identical trainer flags and rotations as gpqacc64k — cycled into
4 × 131072-token sequences instead of 8 × 65536. The only difference is
how far out in RoPE-position space the codebook sees keys.

**In-range (NIAH 8–64K): equivalent.**

| ctx | vq2 (64k-calib) | vq2_128k (128k-calib) | Δ |
|---|---|---|---|
| 8K  | 93.31 | 93.34 | +0.03 |
| 16K | 92.12 | 91.25 | −0.88 |
| 32K | 92.28 | 91.88 | −0.41 |
| 64K | 91.19 | 92.25 | +1.06 |

Doubling calibrated position range buys nothing inside the shared range
(the +1.06 at 64K — the 64k codebook's coverage boundary — is the only
hint, within noise).

**Out-of-range (RULER-131072, 800 rows, TP=2 served): catastrophic.**

| method | 128K NIAH mean | per-subtask |
|---|---|---|
| bf16 (ceiling) | 76.12 | mk 75/79/77 · mq 92 · mv 80 · s1 6 · s2 100 · s3 100 |
| vq2, gpqacc64k | **0.16** | zeros across the board |
| vq2, gpqacc128k | **68.22** | mk 60/62/64 · mq 80 · mv 66 · s1 19 · s2 99 · s3 96 |

The 64k-calibrated codebook does not degrade at 2× its calibrated
positions — it **collapses to zero**. Keys at unseen RoPE positions
encode to garbage codewords and retrieval dies wholesale. The
128k-calibrated codebook, identical in every other respect, serves 128K
at 68.2 — the usual ~7-pt distance below bf16, consistent with the
context-flat gap seen at every other length.

**Design rule for the paper**: codebook calibration must cover the entire
serving position range. It costs nothing (same corpus, same token budget
— just longer concat sequences), and the failure mode for under-coverage
is not graceful.

Notes: (a) `single_1` (the repeated-sentence noise haystack) collapses for
*everyone* at 128K including bf16 (6.0) — a base-model failure mode on
degenerate text at extreme length, not quantization; amusingly vq2_128k
scores higher (19) there, quantization noise apparently breaking the
repetition attractor. Headline means include it for all methods equally.
(b) The 131072 row set is Llama-tokenizer-budgeted (ctx 130048 + 1024
headroom) so no request exceeds native RoPE — generator snapshot
`third_party/samuel_vq/gen_niah.py`, rows
`artifacts/prompt_rows/niah_131072_llama.jsonl`.

## 2. Infrastructure this enabled: TP serving

The 128K cells are the first production use of tensor parallelism (a
131K-token bf16 KV cache is ~34 GB — impossible on one A100-40GB):
bf16/vq2/vq2_128k each served TP=2 on a GPU pair. Two fixes made this
work (tracker `tp-1`): rank-sharded codebook loading (clone `0ecb1ab7c`),
and the custom all-reduce crash root-caused to our own
`expandable_segments` allocator setting (cudaIpc cannot export expandable
VA ranges; upstream vllm#42609) — serve_oscar.sh now drops it for TP>1
and keeps custom AR, measured **+7.8% decode vs the NCCL fallback**
(132.0 vs 122.5 tok/s, TP=2 bs=1).

## 3. Also closed this cycle

- **TurboQuant-INT2** implemented + corrected (tracker `turbo-1`), grid
  column complete — see report12 §5.3.
- **Calibration-unified v2** canonical; rotation-provenance effect ~1 pt
  — see report12 §1a.
- **Qwen v2 + all handoffs** with Samuel:
  https://claude.ai/code/artifact/7e1b0e79-93b7-4b11-9336-b63f39a0c0cf

## 4. R1-Distill-Llama-8B: the long-CoT axis [IN FLIGHT]

Motivation: every reasoning cell in the Llama grid ties (near-chance
model) — we have never tested quantization under thousands of decode
tokens of chain-of-thought, the regime where production OSCAR's streaming
quantization collapsed on NIAH. R1-Distill-Llama-8B is architecturally
identical to Llama-3.1-8B (full stack reuse, zero engine work) and thinks
at length (~50% AIME).

Setup (spec `r1distill_llama8b.json`): calibration from the unified
198-prompt corpus (rotations + gpqacc64k-recipe codebook, built by
`build_r1distill.sh`, gated); grid bf16/int2/vq2 × (NIAH 8–64K greedy +
AIME25 avg@4 + math500-subset avg@2 at 32K thinking caps, T=0.6 /
top-p 0.95); short-val gate before full cells (`</think>` closure,
extraction, cap-hit ≤15%).

[RESULTS PENDING — sections filled as the overnight chain lands.]
