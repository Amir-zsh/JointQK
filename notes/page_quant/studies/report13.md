# Report 13 — Codebook position coverage (gpqacc64k vs gpqacc128k), RULER-128K, and the R1-Distill long-CoT axis

**Status: position-coverage study COMPLETE; R1-Distill-Llama-8B onboarding
+ grid in flight (section 4 updates as cells land).**
Shareable version: https://claude.ai/code/artifact/0cb61471-28cf-41f2-959a-009ee4cd6228

Engine vendor/OSCAR-vq @ `0ecb1ab7c` (TP rank-sharded codebooks; custom
all-reduce root-cause-fixed — see §3). All Llama cells on
rotations_gpqa198, out_root `artifacts/oscar_llama31_8b/grid_v2/`,
declarative specs + manifests per run.

## 1. Headline: codebook position coverage is a cliff, not a slope

*Setup — model: meta-llama/Llama-3.1-8B-Instruct (native RoPE 131072, no
YaRN); calibration: rotations_gpqa198 (198 GPQA prompts, per-prompt dump)
+ K codebooks gpqacc64k (8×65536 concat) vs gpqacc128k (4×131072 concat),
equal 524K-token budget; max seq: 8K–64K in-range evals, 131072
out-of-range eval (prompts ≤130106 + 128 gen, TP=2).*

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

**Cross-check: Samuel's Qwen 128K result (bf16 83.4 / vq2 76.6 /
OSCAR-INT2 25.9, 80 rows) looks different — it isn't. The difference is
YaRN.** His vq2 uses the same 64K-calibrated gpqacc64k codebook yet scores
76.6 where ours scored 0.16, because the two models reach 128K
differently. Background: RoPE encodes each position as rotation angles on
Q/K, and models are trained on a bounded position range. Qwen3-8B's
trained range is 32768, so Samuel serves 128K with YaRN ×4 (his server
config: rope_type yarn, factor 4.0), which rescales rotation frequencies
so positions 0–131072 are *compressed into the trained angular range* —
position 128K gets roughly the angles of native position 32K. Llama-3.1
is natively 131072, so our run uses native RoPE, where positions past 64K
produce angle combinations that exist nowhere below 64K. A codebook
quantizes post-RoPE keys, so what it must cover is the *angular*
distribution it meets at serving time, not position numbers: under YaRN
everything stays inside what a ≤64K calibration has seen (→ his usual
−6.8 gap, matching our −7.9 with the 128K-calibrated codebook); under
native RoPE, keys past 64K snap to meaningless codewords (→ our 0.16).
Same law from both sides: **the cliff is governed by post-RoPE angular
coverage, not raw position index.** Falsification test proposed to
Samuel: Qwen 128K *without* YaRN — predicts his 64K codebook collapses
like ours did. (His OSCAR-INT2 25.9 fits too: scalar quantizers have no
codebook to go out-of-distribution — it just continues its context decay,
23.4 → 25.9.)

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

*Setup — model: Llama-3.1-8B-Instruct; calibration as §1; max seq 131072
served TP=2 (custom-AR A/B at TP=2 bs=1).*

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

*Setup — model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B (Llama-3.1-8B
arch, native 131072); calibration: rotations from the 198-prompt GPQA
corpus via the R1 chat template + gpqacc64k-recipe codebook (8×65536
concat) — both under artifacts/oscar_r1d_llama8b/; max seq: NIAH 8K–64K
+128 gen; reasoning = short prompts + 32768-token generation cap (measured
~4–5K math / ~14K AIME thinking tokens).*

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

Onboarding: build gated clean (rotations orth ~2e-8, codebook G1 PASS,
three thinking-mode smokes correct); short-val: acc 0.90 (math-verify),
cap-hit 0.0%, mean 4,580 thinking tokens, 100% </think> closure.

**NIAH (800 rows, greedy) — quantization hurts R1-distill far more than
base Llama:**

| ctx | bf16 | vq2 | int2 | vq2−int2 |
|---|---|---|---|---|
| 8K  | 94.7 | 79.0 | 78.9 | +0.1 |
| 16K | 95.0 | 72.6 | 67.8 | +4.8 |
| 32K | 90.8 | 67.5 | 52.7 | +14.8 |
| 64K | 80.9 | 55.6 | 33.1 | +22.5 |

The vq2 > int2 hierarchy holds with the separation *growing* with context
(same shape as every other model), but both quantized methods sit far
below bf16 (vq2 −15.7 at 8K, vs −6 on base Llama-3.1-8B with the same
recipe). The reasoning fine-tune evidently reshapes K statistics away
from what GPQA-calibrated artifacts capture — or retrieval-during-
thinking is intrinsically more error-sensitive. Follow-up candidate:
recalibrate rotations+codebook on R1's own generations rather than raw
GPQA prompts.

**Long-CoT reasoning (avg@K, 32K caps) — the axis this model was added
for:**

| cell | bf16 | int2 | vq2 |
|---|---|---|---|
| math500-sub (100 rows, avg@2, ~4–5K think-tokens) | 87.0 ±5.7 | 88.0 ±2.8 | 88.0 ±0.0 |
| AIME25 (30 rows, avg@4, ~14–15K think-tokens) | 34.2 ±3.2 | 28.3 ±4.3 | 30.8 ±7.4 |

At ~5K thinking tokens, 2-bit K quantization is completely free (ties).
At ~14K thinking tokens the ordering matches the project-wide hierarchy —
bf16 34.2 > vq2 30.8 > int2 28.3 — but the magnitudes are small and
within noise at n=30 (vq2 −3.3, int2 −5.8 vs bf16; SE of the difference
≈ 3–4; vq2's ±7.4 spread includes one 20.0 seed). Honest verdict:
**long-CoT reasoning is far more quantization-tolerant than retrieval,
even at 14K decode tokens** — the dramatic separation lives in NIAH, not
in reasoning accuracy. A significance-grade answer needs more seeds/rows
(AIME24+25 × 8 seeds is the cheap extension).

The mechanism behind R1's large NIAH deficits (both methods, see table
above) is under active diagnosis: cap-512 re-slice (verbosity/truncation
channel) and clip-1.0 int2 cell (Qwen-tuned percentile clips vs
heavier-tailed R1 activations) — results land in a report13 addendum.
