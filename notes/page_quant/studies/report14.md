# Report 14 — Why OSCAR INT2 fails on gpt-oss-20B

*A method limitation, not an engine bug.*

*Models: `unsloth/gpt-oss-20b-BF16` (24 layers, alternating
sliding-window(128)/full attention, MoE 32 experts top-4, learned attention
sinks, head_dim 64) and `meta-llama/Llama-3.1-8B-Instruct` (dense, uniform
full attention, head_dim 128, 32 layers) as the working reference. Serving:
vendored OSCAR engine (`vendor/OSCAR-vq`, branch `vq2-longhorizon`), TP=2,
A100-40GB. Investigated 2026-07-24.*

---

## 1. Background

### What the project does

Transformer inference caches a key and a value vector for every token it has
read. At long context that cache dominates memory and bandwidth: a 128K-token
prompt on an 8B model costs tens of gigabytes in bf16, and decode speed is
bound by streaming it. This project compresses the **key** side (and, in the
OSCAR baseline, the value side too) to roughly **2 bits per coordinate**.

The pipeline is:

1. **Calibrate.** Run a small prompt corpus and accumulate per-(layer, kv_head)
   second moments of the post-RoPE keys and queries.
2. **Rotate.** Choose a per-layer orthogonal basis from those statistics, so
   that the energy of a key vector concentrates in fewer coordinates and
   quantization error lands where queries are least sensitive.
3. **Quantize.** Scalar-quantize each coordinate independently in the rotated
   basis, then decompress on the fly when future queries arrive.

Because the rotation is orthogonal, `q·k` is preserved under rotation: the
query is rotated by the same matrix and attention scores are unchanged. All
the loss comes from the quantizer.

### The two arms

**OSCAR INT2** is the published baseline — the authors' production method,
vendored from their `sglang-research` fork. Keys and values are quantized to
2 bits with a **per-token affine min-max** scheme in the calibrated rotated
basis, plus an optional per-row clip. With one scale group per head, each
token's 64 (or 128) coordinates share a single scale and zero-point, giving
four representable levels.

**vq2** is our method: the key tier is replaced by a **group vector-quantizer**
with an fp8 codebook, learned from the same calibration corpus. The value tier
stays on OSCAR's INT2 path unchanged, so the two arms differ only in how keys
are encoded.

### The mixed HP+INT2 cache

Neither method quantizes everything. The pool keeps a **high-precision band**
in bf16 — the first 64 tokens (attention sinks) and the most recent 256 —
and quantizes only the middle. As generation proceeds, tokens age out of the
recent window and are flushed into the quantized tier in `N_Q = 8`-token
pages. This matters for the whole investigation: **a prompt shorter than
~320 tokens never touches the quantized tier at all** and is served entirely
from bf16.

### Where the work stood

The method was validated on dense models. On Llama-3.1-8B, INT2 scores
84.3 on NIAH-8K and vq2 scores 92.3, both against a bf16 ceiling of 98.2.
Qwen3-8B INT2 reaches 97.6. The task this week ("P2") was extending both arms
to a new and much stranger model.

---

## 2. How gpt-oss-20B is different

gpt-oss is not a scaled-up Llama. Four architectural properties separate it
from every model this stack had served, and each one required new code:

| property | Llama-3.1-8B | gpt-oss-20B |
|---|---|---|
| attention pattern | uniform full attention, all 32 layers | **alternating**: 12 sliding-window (128 tokens) + 12 full-attention |
| FFN | dense | **Mixture of Experts**, 32 experts, top-4 per token |
| softmax | standard | **learned attention sinks** (per-head bias in the denominator) |
| head_dim | 128 | **64** |
| KV cache | one uniform pool | **hybrid**: quantized pool for full-attn layers, bf16 for the window layers |

**Hybrid attention** is the consequential one. Half of gpt-oss's layers can
only see the last 128 tokens. Quantizing them is pointless — they hold almost
no cache — so only the 12 full-attention layers are compressed. That demands a
wrapper pool that routes per layer, a dual allocator, an index translation
between the two pools, and a dense re-indexing so global layer ids
`[1,3,5,…,23]` map onto the 12-entry calibration bundle.

**Learned sinks** add a per-head constant to the softmax denominator, letting
a head attend to "nothing". They had to be threaded into the quantized decode
kernel and, for prefill, applied as an exact rescale of a sink-less attention
output, `o · sigmoid(lse − sink)`.

**MoE** means each token's FFN is selected discretely by a router. This turns
out to matter enormously, for reasons that are the subject of §6.

---

## 3. The problem

Once the plumbing worked, the numbers were catastrophic — but only for INT2:

| arm | NIAH-8K | cap-hit rate | mean tokens generated |
|---|---|---|---|
| bf16 | **88.4** | 0.40 | 166 |
| INT2 | **0.0** | **1.00** | **256 (every row)** |

The model did not answer wrongly; it stopped producing language. Every single
row ran to the token cap without emitting a stop token. A representative
generation:

```
' \nThe question is question.\n\n\nWe get squ. ……..?\n\nThe input in order to
**mona?\n\n\n\n\nOk. \nWe have a user sending a message that says: "Trish?"
\n\n?? error???\n\n##???  \n\nThis task is over-...…'
```

Against bf16 on the identical row, which finds the needle and stops at 94
tokens:

```
' 346 …? We need to answer.assistantanalysisWe need to answer: "What is the
special magic number for proud-kernel mentioned in the provided text?" The
text includes: "One of the special magic numbers for proud-kernel is:
3465603." So answer: 3465603.'
```

Note that hitting the cap is *normal* for this model — bf16 does it 40% of the
time, because gpt-oss reasons in an analysis channel before answering. Hitting
it on 100% of rows with degenerate text is not.

---

## 4. Localizing the failure

### It is binary in exactly one variable

INT2 tracks bf16 while the context fits the bf16 band and collapses the moment
*any* token is quantized. Real NIAH prompts, needle-exact scoring, 5 rows per
cell:

| prompt tokens | tokens quantized | INT2 | vq2 | bf16 |
|---|---|---|---|---|
| ~234 | **0** | 3/5 | 5/5 | 3/5 |
| ~524 | 192 | **0/5** | 4/5 | 2/5 |
| ~1107 | 704 | **0/5** | 4/5 | 3/5 |
| ~2271 | 1855 | **0/5** | 4/5 | 2/5 |
| 8192 (served) | 7872 | **0.0** | — | 88.4 |

There is no gradual decay and no threshold beyond that boundary. And nothing
else moves it:

| setting | varied | effect |
|---|---|---|
| decode splits | 8 vs 48 | none — INT2 0/5 at both, vq2 4/5 at both |
| clip ratio | 0.96/0.92 vs 1.0/1.0 | none — bit-identical outputs |
| context length | 518 → 8192 | none |
| concurrency | 6 threads vs sequential | none |

The split-count row closed a real confound: `serve_oscar.sh` defaults INT2 to
8 decode splits and vq2 to 48, so the two arms had differed in more than the
codec until they were crossed.

Grouped scales (4 groups instead of 1) could not be tested — the engine
rejects `--kv-cache-quant-group-size` for hybrid SWA models. That remains the
one untested knob.

### The engine is not at fault

Every component of the INT2 path was verified against an independent reference
at gpt-oss shapes. All pass, and they are retained as regression gates:

| gate | scope | result |
|---|---|---|
| `verify_gptoss_int2_decode.py` | unified HP+INT2 decode kernel, D=64, sinks | exact (≤1.5e-2) |
| `verify_gptoss_mixed_pool.py` | pool write + 256 decode flushes | exact; 8192/8192 unique slots |
| `verify_gptoss_int2_e2e.py` | decode over pool-written buffers, real index split | exact (≤4.6e-3) |
| prefill sink LSE rescale | vs exact sink-softmax reference | 0.2% |
| rotations | orthogonality, dense layer map | 2.4e-07, correct |

TP=2, radix caching, request retraction and concurrency are all excluded by
the fact that **vq2 serves correctly under identical conditions**, sharing the
pool, allocator, index construction, rotations, sinks and layer mapping.

The calibration data was audited separately. The moments' `local_to_global`
matches the serving layer map; `ntok = 524288` from 8×64K sequences, so the
position-coverage cliff (report 13) cannot explain an 8K failure; and the two
models use *different capture scripts* (Llama: `capture_qkv_dump.py`;
gpt-oss: our `gptoss_calibrate.py`, written for the eager/sink constraints),
but both emit the same `[T, H, d]` fp16 post-RoPE convention into the same
rotation builder. One real gap: the rotation bundles carry **no provenance** —
no corpus, prompt count, context length, or capture commit — so they cannot be
audited from their own contents.

### It is not reconstruction fidelity — but the first evidence for that was unsound

Real post-RoPE keys and values captured from live forward passes, rotated into
each model's own OSCAR basis and quantized exactly as the pool does:

| model | K cos | V cos | serves INT2? |
|---|---|---|---|
| Llama-3.1-8B | 0.898 | 0.897 | **yes (84.3)** |
| gpt-oss-20B | **0.923** | **0.915** | no (0.0) |

gpt-oss's tensors reconstruct *better* than Llama's, so per-tensor cosine —
the standard proxy for judging a KV quantizer — does not predict survival.

**Two further columns originally reported here were withdrawn.** A "logit
corr" computed against *random Gaussian* queries is structurally blind to a
basis misaligned with the real query distribution, which is precisely what a
Q-weighted rotation is for; and an "attn output err" of 0.210/0.174 was
computed with the **same softmax weights** for the quantized and reference V,
so the K error never reached the attention distribution at all. It measured
the V half only. Neither supports the claim it was used for.

Replaced by a measurement of the quantity that matters, using the model's own
queries and letting the distribution move — relative error of the attention
logits `q·K` under int2, in competing bases (8 GPQA prompts per model):

| basis | gpt-oss | Llama | gap |
|---|---|---|---|
| identity (no rotation) | 1.376 | 0.322 | 4.3× |
| **shipped (calibrated)** | **0.710** | **0.208** | **3.4×** |
| hadamard (calibration-free) | 0.633 | 0.205 | 3.1× |
| random orthogonal | 0.768 | 0.209 | 3.7× |

**gpt-oss's attention logits are ~3.4× more damaged by int2 than Llama's, in
every basis tested** — including no rotation, a random basis, and a
calibration-free Hadamard. Even the best basis leaves gpt-oss at 63% logit
error where Llama sits at 20%. No choice of basis closes the gap, which is
what "not a calibration bug" looks like. A 71% logit error scrambles the
softmax distribution, which is why the attention output decorrelates (§6).

The same table also **retires a false alarm**: gpt-oss's shipped rotation is
barely better than random (0.710 vs 0.768) and worse than a plain Hadamard
(0.633), which looked like a broken calibration — until the control showed
Llama's shipped rotation *also* ties random (0.2082 vs 0.2087) and Hadamard
(0.2046) while serving at 84.3. "Shipped ≈ random" is not a gpt-oss defect.

### The quantizer is not producing pathological error

The strongest test of "the quantization is simply done wrong" is *within*
gpt-oss, with no cross-model confound: perturb K with Gaussian noise of
**exactly the norm int2's error had**, in the same basis, and compare.

| perturbation on K, shipped basis | gpt-oss | Llama |
|---|---|---|
| int2 | **0.718** | **0.209** |
| matched-magnitude Gaussian noise | **0.909** | **0.213** |

**Random noise of int2's own error magnitude does *more* damage than int2
does, in both models.** The quantizer's error is better-structured than an
equivalent random perturbation — the opposite of what a broken quantizer looks
like — and the ordering is general, not a gpt-oss quirk. Same model, same keys,
same basis; only the error's structure varies. The damage tracks error
*magnitude*, and gpt-oss does not tolerate perturbation at 2-bit scale.

### The decisive test

If the engine is innocent, the failure should reproduce with no engine at all.
Running both models under plain HuggingFace and overwriting the KV cache with
its own quantize→dequantize roundtrip, on the same layers and the same band
the engine uses:

| model | layers quantized | bf16 | INT2 (HF simulation) |
|---|---|---|---|
| Llama-3.1-8B | 32 (all) | 10/10 | **10/10** |
| gpt-oss-20B | 12 (full-attn) | 4/10 | **0/10** |

The simulator reproduces the failure on gpt-oss and leaves Llama completely
untouched. **This is a limitation of the method on this architecture.** The P2
implementation was faithfully delivering what OSCAR INT2 does to this model.

---

## 5. Which layers carry the damage

Same simulation, varying *which* layers are quantized (n=10, bf16 = 4/10):

| quantized layers | INT2 needle |
|---|---|
| 12 sliding-window layers | **7/10** — no harm |
| 4 full-attention layers | 3/10 |
| 12 full-attention layers | **0/10** |

Quantizing the sliding-window layers is harmless; damage scales with the
number of **full-attention** layers quantized. The structural reading: gpt-oss's
window layers see only 128 tokens, so the 12 full-attention layers are the
**sole channel for long-range information**, with no redundancy to average
quantization noise against. Llama spreads the same information across 32
full-attention layers, any one of which can compensate for another.

---

## 6. Mechanism: where the error grows

The 3.4× logit-error gap of §4 is the input to this section: a 71% error on
attention logits scrambles the softmax distribution, and the damage propagates
from there. Teacher-forced comparison of the bf16 and INT2 runs (identical
token stream so the two stay comparable, 48 decode steps), measuring relative
drift of hidden states and of each sublayer's output:

| | gpt-oss | Llama |
|---|---|---|
| attn output drift, **first quantized layer** | **1.207** | 0.145 |
| mlp output drift, first quantized layer | 1.120 | 0.137 |
| hidden drift, final layer | **0.890** | 0.387 |
| hidden drift, mean over layers | 0.550 | 0.273 |
| expert top-4 flip rate | **0.816** | n/a (dense) |

At the first quantized layer both models receive **identical inputs** —
gpt-oss's layer 0 is a sliding-window layer with an unquantized cache, and its
measured drift is exactly 0.000. Same queries, only the KV quantized. And
gpt-oss's attention output comes back **fully decorrelated** (relative drift
above 1.0) where Llama's is off by 15%. Drift above 1.0 means the quantized
output is further from the truth than emitting zeros would be, which needs
explaining.

Two drafts of this report explained it by sink-compressed output norm — the
sinks absorb the softmax mass, `o = Σpᵢvᵢ` comes out small, and two
similar-magnitude vectors pointing differently reach √2. **Measurement kills
that explanation.** gpt-oss's sink mass is 0.30 (not "most"), and its output
norm is **0.38** of a typical value vector — while **Llama's is 0.232, with no
sinks at all**, and Llama's drift is 0.145. The model with the *smaller*
attention output is the one that is fine, so output compression cannot be the
differentiator.

What remains is the logit error itself: at 71% (vs Llama's 20%) the softmax
distribution `p` is reordered rather than perturbed, so `o` moves to a
genuinely different convex combination of the same value vectors. A modest
`‖o‖` then permits the ratio to exceed 1, but the *driver* is the 3.4× logit
damage of §4, not the sinks.

From there it compounds. gpt-oss reaches 0.890 relative drift at the output
layer — the residual stream is essentially unrelated to the true one, which is
precisely the word salad observed. Llama saturates near 0.39 and stays
coherent enough to score 84.3.

Per-layer hidden drift (sampled layers, normalized depth):

| depth | gpt-oss | Llama |
|---|---|---|
| layer 0 | 0.000 | 0.059 |
| layer 1 | 0.385 | 0.093 |
| layer 2 | 0.373 | 0.125 |
| layer 5 | 0.480 | 0.169 |
| layer 11 | 0.576 | 0.286 |
| layer 17 | 0.705 | 0.314 |
| layer 23 | 0.890 | 0.320 |
| layer 29 | — | 0.354 |

**82% of all expert-routing decisions flip** between the bf16 and INT2 runs,
rising to 97.9% at the final layer. Each flip swaps an entire FFN — a discrete
amplification channel that a dense model structurally lacks.

---

## 7. What this means for the project

- **vq2 on gpt-oss works, and the grid is unblocked.** vq2 serves this model
  correctly through the same plumbing, at both split settings.
- **The result is stronger than the Llama grid shows.** On gpt-oss, 2-bit
  scalar quantization fails outright while vq2 holds — a cleaner separation
  than any dense-model comparison, on an architecture class (MoE + sinks +
  hybrid attention) that the OSCAR paper never evaluated.
- **There is currently no valid INT2 baseline on gpt-oss.** Comparisons must
  be against bf16 until INT2 is made to work, and this report suggests it may
  not be makeable at 2 bits scalar.
- The natural follow-up is a **fidelity-threshold** framing: measure vq2's
  reconstruction on the same tensors, and turn "vq2 is better" into "gpt-oss
  requires ≥X fidelity; INT2 delivers 0.92, vq2 delivers Y."

---

## 8. Confidence and caveats

**Solid.** Limitation-not-bug: the HF simulation reproduces with zero engine
code and the Llama control is clean at 10/10 → 10/10. The binary length
boundary. The fidelity comparison. The drift profiles. Each has a working
reference point in both the passing and failing direction.

**Suggestive, underpowered.** The dose-response and the window-vs-full-attention
split are n=10 at a single 1064-token setting. bf16 itself scores only 4/10
there, because the reconstructed prompts are hard and the 96-token generation
cap truncates gpt-oss's reasoning — so headroom is small. Directionally clear,
not publishable as-is.

**Not established.** *Why* gpt-oss's attention output decorrelates at the first
quantized layer given better KV fidelity. Attention sinks are the leading
suspect — sink-dominated heads leave little softmax mass on content, so a
perturbation is relatively larger — but this was not isolated. The 82%
expert-flip rate is *consistent* with MoE amplification but does not separate
cause from consequence; flips could be downstream of an already-large drift.
The clean test is a **matched-noise control**: inject random KV noise into
Llama at a magnitude that produces drift 1.2 at layer 1, and see whether Llama
also collapses. That separates "gpt-oss is fragile" from "this much drift
kills any model."

**Process note.** Five probes in this investigation produced striking results
that were instrumentation errors, each caught by adding a known-good control.
The pattern is uniform: a measurement was read as a finding before it was shown
able to distinguish a passing case from a failing one.

- a raw-text length probe with no bf16 control — bf16 produced *identical*
  "degenerate" output, so the probe could not attribute anything;
- an end-to-end gate that passed global slot ids where the HP tier expects
  local ones, giving an apparent 100% error and a false reproduction of the
  bug;
- a router hook reading `router_scores` instead of `router_indices`, reporting
  0% expert flips when the true rate is 82%;
- an attention-output error computed with frozen softmax weights, so the K
  error never entered the distribution — it measured the V half and was used
  to argue fidelity was fine;
- "the shipped rotation barely beats random", which looked like a broken
  gpt-oss calibration until Llama's shipped rotation was found to tie its own
  random and Hadamard baselines while serving at 84.3.

Any single measurement in this area should be treated as provisional until it
reproduces both a passing and a failing case.

---

## 9. Reproduction

```bash
# limitation vs bug: HF simulation, no engine code
python pipelines/oscar_e2e/dbg_int2_hf_simulation.py \
    --model unsloth/gpt-oss-20b-BF16 \
    --rot artifacts/oscar_gptoss20b/rotations_gpqa198 --gpus 0,3 \
    --tag gptoss --rows 10 --which full     # 0/10;  --which swa -> 7/10

# llama control (must stay 10/10)
python pipelines/oscar_e2e/dbg_int2_hf_simulation.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --rot artifacts/oscar_llama31_8b/rotations_gpqa198 --gpus 0 \
    --tag llama --rows 10

# mechanism: per-layer drift, sublayer breakdown, expert flips
python pipelines/oscar_e2e/dbg_moe_amplification.py \
    --model unsloth/gpt-oss-20b-BF16 \
    --rot artifacts/oscar_gptoss20b/rotations_gpqa198 --gpus 0,3 \
    --tag gptoss --steps 48

# real-tensor quantization fidelity (K and V)
python pipelines/oscar_e2e/dbg_capture_k_fidelity.py \
    --model <model> --rot <rot-dir> --gpus 0,3 --tag <tag>

# engine gates (all must pass)
PYTHONPATH=vendor/OSCAR-vq/sglang-research/python:. .venv-oscar/bin/python \
    pipelines/oscar_e2e/verify_gptoss_int2_decode.py --gpu 1
PYTHONPATH=vendor/OSCAR-vq/sglang-research/python:. .venv-oscar/bin/python \
    pipelines/oscar_e2e/verify_gptoss_mixed_pool.py --gpu 1
PYTHONPATH=vendor/OSCAR-vq/sglang-research/python:. .venv-oscar/bin/python \
    pipelines/oscar_e2e/verify_gptoss_int2_e2e.py --gpu 1

# served length sweep (against a running server)
python pipelines/oscar_e2e/dbg_gptoss_niah_lengths.py \
    --port <port> --mode <int2|vq2|bf16>
```

Logs on lambda-server6 (`10.137.32.78`): `logs/hfsim_n10.log`,
`logs/moeamp3_{gptoss,llama}.log`, `logs/kvfid_{gptoss,llama}.log`,
`logs/splits.log`, `logs/codec.log`, `logs/2x2.log`.

Related: `notes/page_quant/fixes_to_apply.md` entries **p2-1** (premature
mixed-KV flush double-assigns a quant slot) and **p2-2** (`SWAKVPool` missing
`release_req_slab` delegate) — both latent, neither the cause of this failure.
