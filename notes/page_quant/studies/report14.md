# Report 14 — Why OSCAR INT2 fails on gpt-oss-20B

*Three separable findings: a scalar-INT2 fidelity limitation, a now-fixed
piecewise-prefill graph bug shared by INT2 and vq2, and a now-fixed
GPT-OSS-specific VQ learned-sink bug. Corrected VQ is materially better but
still below bf16. The remaining loss begins when the combined VQ-K/INT2-V tier
activates; attribution within that tier is still open.*

*Models: `unsloth/gpt-oss-20b-BF16` (24 layers, alternating
sliding-window(128)/full attention, MoE 32 experts top-4, learned attention
sinks, head_dim 64) and `meta-llama/Llama-3.1-8B-Instruct` (dense, uniform
full attention, head_dim 128, 32 layers) as the working reference. Serving:
vendored OSCAR engine (`vendor/OSCAR-vq`, branch `vq2-longhorizon`), TP=2,
A100-40GB. Investigated 2026-07-24—26.*

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

The first failure was catastrophic quality degradation under scalar INT2:

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

This finite-but-degenerate quality failure is distinct from a second failure
found later: sampled concurrent serving under vq2 or INT2 could produce
non-finite logits and terminate in `torch.multinomial`. The second failure was
an engine bug; §4 separates it from the scalar-quantization result.

### Post-fix reproduction

The table above was originally measured while the piecewise-prefill padding
bug (§4) was still active, so it carried a question: how much of it was the bug
rather than the scalar-INT2 limitation? Re-run on the same stratified 24-row
slice and protocol as the post-fix throughput cells, with the padding fix in
place:

| arm | sampled 4×24 NIAH-8K | cap-hit |
|---|---|---|
| INT2 | **0.0** | 0.96 |
| vq2, before learned-sink fix | 70.8 | 0.79 |

Both headline numbers survive: INT2 is still 0.0 (still generating clean,
non-crashing, content-free text — cap-hit 0.96, not the 1.00/NaN-terminated
signature of the padding bug), and vq2 is unchanged within noise of its
pre-fix 66.7. This established that neither result came from the graph-padding
bug, but the later learned-sink isolation below supersedes the absolute VQ
number. The old short-context sweep is excluded; §4 explains why.

### Final isolation: quantization side effect versus engine bug

The post-fix VQ result above still contained a separate correctness bug. VQ
stores a mean-centered key residual

```
r = (k - mean) @ forward
q_m = q @ inverse.T
```

so every real-key logit is shifted by the same query-dependent constant:

```
scale * q_m·r = scale * q·k - c
c = scale * q·mean
```

Subtracting `c` from every key is softmax-invariant only when the denominator
contains real keys alone. GPT-OSS adds a learned per-head sink logit `s` to the
denominator with no value-vector contribution. The VQ path transformed the
keys but left `s` unchanged, making the sink effectively `c` stronger relative
to every key. The correct stored-space sink is `s - c`.

This explains all four otherwise-puzzling observations:

1. the problem is GPT-OSS-specific because Llama has no learned sink;
2. it is VQ-specific because scalar OSCAR rotation does not subtract a mean;
3. it survives in the BF16 HP tier because HP VQ rows are still mean-centered;
4. the old VQ verification gate passed because it compared softmax over real
   keys only and never included the extra sink denominator term.

The repair adjusts the sink per query during quantized prefill. During decode,
the existing unified stage-2 Triton reduction computes `q·mean` and shifts the
sink inside the same kernel, so it adds no CUDA launch. A synthetic A100 gate
against original-space PyTorch attention measured **19.8% relative output
error before the shift** and **3.6e-7 after it**. The fused work adds 3.33 us
to the isolated stage-2 reducer at batch 4 / 96 total splits
(34.01→37.34 us, 9.8% of that small reducer), or about 0.04 ms across all 12
compressed layers per decode step.

Matched NIAH-8K results use the same 24 rows, four independent samples, cap
256, four concurrent requests, split 48, piecewise and decode graphs enabled,
and `temperature=1, top_p=1, top_k=-1`:

| arm | accuracy mean ± sample SD | cap-hit | mean completion |
|---|---:|---:|---:|
| bf16 | **98.96 ± 2.09** | 5.21% | 118.5 |
| VQ before sink fix | 61.46 ± 6.25 | 62.50% | 199.0 |
| VQ after sink fix | **79.17 ± 3.40** | 39.58% | 192.9 |

The bug therefore accounts for roughly **17.7 accuracy points**, but not the
whole bf16 gap. Corrected VQ still produced 38/96 cap hits; several were the
same punctuation loops and circular reasoning seen before the fix.

The decisive controls then removed actual quantization. They used
`RECENT_TOKENS=16384`, one 8.26K request at a time, one 16K final prefill
chunk, and piecewise prefill disabled. This detail is load-bearing:
`_mixed_extend_layout_counts` deliberately puts a **non-final** chunk's middle
into the quant tier, so merely increasing the recent-window cap while retaining
the default 4096-token chunks is not an all-HP experiment.

| verified all-BF16-cache arm | result | cap-hit | mean completion |
|---|---:|---:|---:|
| scalar OSCAR with identity K/V transforms | **8/8** | 12.5% | 130.9 |
| corrected VQ (mean-centered BF16 residuals, no codebook rows) | **8/8** | 0% | 130.3 |

Both are healthy. Thus the pool, hybrid layer mapping, sink correction,
mean-centered BF16 representation, and decode integration work when the
quantized tier is removed. The residual corrected-VQ gap begins only when
middle rows enter that tier.

That control does **not** yet prove that VQ K is the cause. Default VQ2 uses
group-VQ for K but still uses scalar INT2 for V. Moving all rows to the BF16
tier removes both interventions at once. The control therefore excludes the
mean-centered representation and common integration plumbing, but it cannot
separate VQ-K loss, INT2-V loss, their interaction, or a defect specific to
quantized-tier encode/read.

There is also an unisolated artifact-format boundary. The canonical GPT-OSS
bundle declares `fp8_fmt=e5m2`, but `make_fp8.py` stores per-group-scaled E5M2
centroids after dequantizing them back to FP16. The runtime loader then snaps
those values again to unscaled native E5M2 for packed decode. On the canonical
bundle this second snap changes 99.8% of centroid values and has 4.77% relative
centroid RMSE. Encoder and decoder remain mutually consistent because both use
the second-snap values, so this is not by itself an arithmetic correctness bug;
its end-to-end fidelity cost still needs an A/B against a raw, single-snap
bundle.

The implemented follow-up is a real-tensor 2x2 factorial:
BF16-K/BF16-V, VQ-K/BF16-V, BF16-K/INT2-V and deployed VQ-K/INT2-V, plus a
source-centroid diagnostic. Its GPU measurement is still pending. Until that
completes, the correct conclusion is narrower:
**corrected serving still has a quantized-tier gap, whose ownership is not yet
established.**

---

## 4. Localizing the failure

### The original short-context sweep was invalid

The first shortening helper retained only the first 400 prompt characters,
then inserted the needle. It cut the instruction in the middle of a sentence,
producing text such as `"A special magic number is hidden within the following
One of..."`, and joined arbitrary partial filler words to the question. It also
used five greedy generations capped at 160 tokens, whereas the headline 8K
result used 24 rows × four samples at `temperature=1` with a 256-token cap.
The resulting 3/5 BF16 and INT2 cells cannot locate a quantization boundary.

The repaired helper shortens only the haystack body, preserves the complete
instruction/question/assistant prefix and approximately preserves needle
depth. A matched run used ten rows × four samples, `temperature=1`, a 256-token
generation cap and the same reconstructed prompts for all arms:

| target | actual prompt tokens (first row) | BF16 | scalar INT2 | old VQ2 (sink bug) | corrected VQ2 |
|---:|---:|---:|---:|---:|---:|
| 256 | 238 | **39/40 (97.5%)** | **40/40 (100%)** | **31/40 (77.5%)** | **39/40 (97.5%)** |
| 1024 | 1057 | **37/40 (92.5%)** | **0/40 (0%)** | **32/40 (80.0%)** | **37/40 (92.5%)** |
| 2048 | 2150 | **38/40 (95.0%)** | **2/40 (5%)** | **31/40 (77.5%)** | **37/40 (92.5%)** |
| 4096 | 4335 | **38/40 (95.0%)** | **1/40 (2.5%)** | **20/40 (50.0%)** | **35/40 (87.5%)** |
| 8192 | 8262 | **40/40 (100%)** | **0/40 (0%)** | **28/40 (70.0%)** | **37/40 (92.5%)** |
| 16384 | 16455 | **39/40 (97.5%)** | **0/40 (0%)** | **24/40 (60.0%)** | **34/40 (85.0%)** |

Cap hits for BF16 were 1/40, 4/40, 4/40, 3/40, 1/40 and 2/40; scalar
INT2 16/40, 40/40, 40/40, 39/40, 40/40 and 40/40; old VQ2 34/40, 30/40,
28/40, 29/40, 24/40 and 19/40; corrected VQ2 4/40, 5/40, 5/40, 5/40, 7/40
and 17/40. The 16K row uses a newly exported true GPT-OSS
`niah_16384_gptoss_t1.jsonl` source; the shorter rows use the same 8K source
as the earlier repaired sweep.

At the 238-token prompt there are no quantized prompt rows, but old VQ2 is
still not the original BF16 attention. Its HP rows are exact BF16 values in
the VQ mean-centered coordinate system, so real-key logits are shifted by
`-scale * q·mean`; that shift cancels among real keys but not against the
GPT-OSS learned sink denominator term. Corrected VQ2 shifts the sink by the
same amount. Long completions at 238 tokens can also cross the 320-token
HP-window boundary during decode, but the old-VQ drop at this row is already
explained by the all-HP sink mismatch.

Thus shorter context does not cause the earlier 60% result. Corrected VQ2
tracks BF16 through 2K, then shows a moderate but real residual gap at 4K-16K
(7.5, 7.5 and 12.5 points in this matched sweep). Scalar INT2 collapses once a
substantial middle tier is quantized and remains near-zero through 16K.

Grouped scales (4 groups instead of 1) could not be tested — the engine
rejects `--kv-cache-quant-group-size` for hybrid SWA models. That remains the
one untested knob.

### The decode and cache plumbing are not the source of the quality collapse

Every component of the INT2 path was verified against an independent reference
at gpt-oss shapes. All pass, and they are retained as regression gates:

| gate | scope | result |
|---|---|---|
| `verify_gptoss_int2_decode.py` | unified HP+INT2 decode kernel, D=64, sinks | exact (≤1.5e-2) |
| `verify_gptoss_mixed_pool.py` | pool write + 256 decode flushes | exact; 8192/8192 unique slots |
| `verify_gptoss_int2_e2e.py` | decode over pool-written buffers, real index split | exact (≤4.6e-3) |
| prefill sink LSE rescale | vs exact sink-softmax reference | 0.2% |
| rotations | orthogonality, dense layer map | 2.4e-07, correct |

These gates exclude the quantized decode kernels, pool writes, allocator,
index construction, rotations, sinks and layer mapping as the source of the
finite scalar-INT2 quality collapse. The earlier inference that they excluded
*all* engine faults because vq2 served the same short sweeps was too broad:
vq2 shares the same quantized prefill path, and a separate concurrency-dependent
bug in that path was subsequently reproduced.

### A separate engine bug: piecewise prefill graph padding

The later crash required sampled decoding (`temperature=1`) and concurrent
long prompts. The sampler asserted because its probability tensor contained
NaN or Inf; greedy decoding had silently hidden the same class of failure by
using argmax.

The root cause is a shape/metadata contract violation during piecewise CUDA
graph replay:

1. `PiecewiseCudaGraphRunner` rounds a real prefill token count up to a captured
   bucket and pads `q/k/v`; examples observed were 1→4, 283→288 and 68→80.
2. `_forward_extend_quantized_dense` passed the padded `q` tensor to
   `flash_attn_varlen_func`, but built `cu_seqlens_q` from the real requests.
   Consequently, `q.shape[0]` was the bucket size while
   `cu_seqlens_q[-1]` was the smaller real-token count.
3. The caller declared no sequence covering the trailing rows, so
   FlashAttention was not required to define them. Direct probes found exactly
   that padding footprint non-finite:
   `3×32×64 = 6144` values for 1→4, `5×32×64 = 10240` for 283→288 and
   `12×32×64 = 24576` for 68→80.
4. Fixing only that call was insufficient. An isolated dummy sequence made
   the quantized full-attention output finite, but the intervening bf16
   sliding-window attention backend again left its metadata-excluded padding
   rows undefined. The captured residual stream then carried those rows into
   later attention and MoE segments.
5. GPT-OSS makes the leak reliably fatal. Its learned-sink correction consumes
   both the attention output and softmax LSE, and its captured MoE computation
   processes padded token rows. Dense Llama did not turn the same undefined
   padding into a sampled-logit failure in the tested runs.

`DISABLE_CUDA_GRAPH=1` initially appeared to rule graphs out, but that setting
disabled only decode CUDA graphs. Server arguments from the supposedly
graphs-off failure showed `disable_cuda_graph=True` and
`disable_piecewise_cuda_graph=False`; prefill replay was still captured.

This bug affects **vq2 as well as plain INT2**. In this engine, `--vq2` still
sets `kv_cache_dtype=int2`; the codebook changes the K tier, while V remains
INT2 and both modes use the same quantized prefill implementation.

This is an integration-layer bug, not a FlashAttention implementation bug.
FlashAttention is a stateless kernel: it receives Q/K/V and cumulative sequence
offsets, and cannot infer whether tensor rows omitted from those offsets are
graph padding, another request or invalid input. The SGLang/OSCAR adapter owns
the tensor-to-metadata contract and the lifetime of static graph buffers, so
the repair belongs at that boundary; no FlashAttention code is changed.

Completely disabling piecewise prefill, or eagerly running only non-exact
tails, avoids the bug but gives up precisely the launch-overhead reduction
needed for a fair throughput comparison. The production fix therefore keeps
the padded replay path and makes its padding well-defined:

1. The quantized full-attention path appends an isolated dummy query sequence
   for the rounded-up rows, with zero Q/K/V, and extends **both**
   `cu_seqlens_q` and `cu_seqlens_k`. FlashAttention now sees
   `cu_seqlens_q[-1] == q.shape[0]` while the dummy rows cannot attend to a
   real request.
2. The shared attention boundary zeros all rows beyond the real token count
   after every backend call. This contains padding across the alternating
   quantized full-attention and bf16 sliding-window layers before captured
   residual and MoE segments consume it.

The exact 24-row, four-thread, sampled VQ2 reproduction then completed 24/24
in 44 s with **both decode and piecewise prefill graphs enabled**. Every one
of 69 logged prefill launches reported `cuda graph: True`, including ragged
totals of 72, 80, 144 and 288 tokens as well as all 48 bulk 4096-token chunks;
every decode launch was graphed too.

The same full-graph regression under scalar INT2 completed 24/24 in 45 s.
All 55 logged prefill launches were graphed, including 288- and 296-token
ragged totals, every decode launch was graphed, and neither run produced a
scheduler exception, sampler assertion or non-finite probability diagnostic.
INT2 still scored 0.0 and every row reached the 256-token cap, independently
confirming that fixing serving correctness does not fix scalar-INT2 fidelity.
Because sampled completion lengths vary, these wall-clock and throughput
figures validate the fast path but are not a controlled performance benchmark.

A final pre-commit VQ2 rerun exposed and then closed a separate hybrid-SWA
allocator leak. The HP-recent ring had 263 slots (`256 + N_Q - 1`), but decode
allocates its next HP slot before the flush plan releases the oldest `N_Q=8`
slots. On the first wrap, the new full→SWA mapping overwrote slot 0's still-live
mapping; the flush then freed the new mapping, permanently losing the old SWA
slot. The observed leak count tracked requests that crossed this wrap boundary.

The ring bug is **not specific to GPT-OSS**, nor is it owned by an attention
kernel. It requires the mixed HP+quantized cache, the `SWAKVPool`
full-slot-to-SWA-slot mapping, and a request long enough to wrap the recent-HP
ring. GPT-OSS exposes that combination reliably because it alternates full and
sliding-window attention layers. Dense full-attention models have no SWA
mapping and therefore cannot suffer this particular mapping leak; any other
hybrid-attention model using the same cache configuration could. Scalar INT2
and vq2 are both covered because they share this allocator and lifecycle.
FlashAttention only consumes the resulting KV tensors and indices; it does not
allocate SWA slots, maintain mappings or recycle request state.

The ring now reserves the transient allocation slot (`256 + N_Q = 264`), and
the hybrid wrapper delegates `release_req_slab` so a recycled request slot
starts with clean ring and flush cursors. With strict idle memory checking
enabled, a sampled 24-row run scored 75.0 and completed 24/24 with 12 wrap
crossings. A second fixed-length run forced all 24 requests across the boundary
and scored 91.67. Across the two runs, all 126 prefill launches were graphed
(30 ragged tails), the server remained healthy after returning to idle, and
there was no allocator leak, scheduler exception or non-finite diagnostic.

For context, the interim exact-size guard had already made the crash disappear,
and all-eager controls took 293–377 s in two longer-generation runs. Those
experiments localized the fault; they are not the final execution policy.

### Controlled throughput: no VQ2 speedup at 8K

The sampled regression timings above cannot compare methods because completion
lengths differ. A controlled serving-throughput run therefore used the same 24
NIAH-8K prompts (198,250 prompt tokens per run), concurrency four, greedy
decoding with EOS ignored, and exactly 256 output tokens per request (6,144
total). All arms used TP=2 on the same A100-40GB pair, Triton attention, both
CUDA graph mechanisms, and the same decode split-K factor of **48**. Each
number below is the mean of two warmed runs; the spread is sample standard
deviation:

| KV method | output tok/s | relative to bf16 |
|---|---:|---:|
| bf16 | **184.04 ± 3.24** | **1.000×** |
| scalar INT2 | **146.99 ± 1.59** | **0.799×** |
| VQ2 | **131.91 ± 1.53** | **0.717×** |

Every logged prefill and decode launch was graphed, and no arm produced a
scheduler exception. At this 8K operating point, scalar INT2 is 20.1% slower
than bf16 and VQ2 is 28.3% slower than bf16 and 10.3% slower than scalar INT2.
The current integrated VQ2 path therefore has more total overhead than the KV
bandwidth it saves at this context length. The controlled run does not,
however, identify the fused VQ gather kernel as that overhead: the component
measurements below show that its stage-1 kernel is already competitive with
scalar INT2. **This experiment does not support a VQ2 tok/s speedup claim.**
Longer-context sweeps may cross into a bandwidth-dominated regime, but that
must be measured rather than inferred from compression ratio.

`--triton-attention-num-kv-splits` controls how many KV-length partitions
produce partial attention results before a final reduction. More splits expose
more parallelism but add partial-buffer and reduction overhead. The production
launcher previously defaulted VQ2 to 48 and INT2/bf16 to 8 because their
kernel-specific optima differ. That is a valid best-tuned-system comparison
when labeled as such, but it is not sufficient to attribute a speedup to the
codec. The table above deliberately matches the split count across methods.

### Performance audit: graph-fix cost and VQ2 optimization

The padded-graph repair has a nonzero but small cost, and only on a ragged
piecewise-prefill replay. An exact-size bucket takes the false branch, so the
repair launches no GPU work. On a ragged bucket the current defensive
implementation clones and clears padded Q, creates zero dummy K/V rows,
extends the indptr, executes a small isolated dummy sequence, and clears
padded output rows at attention boundaries. An A100 operation-level
reproduction of the newly added tensor operations (excluding the K/V
concatenations that already existed) measured **3.27–3.66 ms per ragged
prefill batch** across the observed 1→4, 68→80 and 283→288 shapes. This is not
a server A/B measurement and excludes the dummy sequence's FlashAttention
increment, but that increment covers only 3–12 zero tokens in these cases.

The controlled VQ2 run contained 48 exact 4096-token chunks and about 21
ragged launches over roughly 46.6 s. Applying the measured repair cost to
those tails gives about 71 ms, or **~0.15% of wall time**, before the small
dummy-attention increment. The fix cannot explain VQ2's 10.3% gap to INT2.
It could matter more for a workload dominated by very short ragged prefills.

The VQ2 audit found the following at the GPT-OSS TP-rank shape
(`L=12, H_KV=4, H_Q=32, D=64, G=4, K=256`) on A100:

| component | measured cost / observation | assessment |
|---|---|---|
| fused decode stage 1, batch 4, context ~8.25K, split 48 | VQ2 **142.75 µs/layer** vs INT2 **154.98 µs/layer** | not the demonstrated bottleneck |
| post-RoPE VQ query map, batch 4, graph replay | **~10.1 µs/layer**, ~0.12 ms over 12 layers/step | low priority |
| decode aging flush, batch 4 | **1.23 ms every 8 steps**, **0.154 ms/step amortized** | real eager tax; fuse, but too small to explain the full gap |
| prefill VQ assignment, 4096 tokens | **1.69 ms/layer**; 2048-token chunk materializes a **128 MiB fp32 score tensor** | plainly inefficient; high-value TTFT target |
| mixed-prefix VQ reconstruction | **~0.34 ms/layer** at 2K–8K in the isolated test | allocation-heavy but lower priority |

The most straightforward code optimization is one fused Triton
nearest-codeword writer shared by prefill and decode flush. Today both paths
normalize in fp32, materialize `[T,H,NG,256]` scores through `torch.einsum`,
run separate subtraction/argmax/casts, allocate scale tensors, and scatter in
later kernels. A fused kernel can reduce candidates in registers and directly
write the uint8 index and per-token scale, while masking invalid flush-plan
rows. This removes the 128 MiB prefill temporary and the periodic eager flush
chain without changing the codec.

Second, `_vq_prefix_dequantize_k` is high-level advanced indexing: it expands
uint8 indices to int64, reconstructs dense codewords, gathers HP rows even for
quantized slots, then selects with `torch.where`. The scalar-INT2 path already
does the analogous mixed HP/quant operation in one Triton kernel. A VQ version
should read the slot class, index, codebook and scale once and write the dense
prefix directly. This is straightforward, although the isolated cost makes it
less urgent than the encoder.

Third, the split cap needs an end-to-end sweep rather than the hard-coded VQ2
default of 48. The unified stage-2 kernel has `TOTAL_SPLITS` as a compile-time
loop bound and scans every configured HP+quant scratch slot, including unused
ones. In an isolated stage-1 sweep, VQ2 split 16 and split 48 were effectively
tied at this 8K shape (**143.35 vs 142.75 µs/layer**), so split 48 may buy
almost no stage-1 time while making the reducer much larger. A served split-16
run attempted before the hybrid-SWA ring fix terminated after eight requests.
After the fix, the same split-16 configuration completed a strict fixed-length
24/24 correctness run, scored 95.83, and graphed all 57 prefills without a
leak. Its single 146.93 tok/s observation is not a tuned or repeated benchmark,
so the split-count performance question remains open.

Lower-priority cleanup remains possible in the graph repair itself: zero only
the padded views instead of cloning all Q, cache the extended query indptr once
per forward rather than rebuilding it in each quantized layer, and represent
the dummy request in shared PCG attention metadata so all backends consume the
same valid static rows. Those changes should be made only with the sampled
24-row regression retained; their maximum benefit on the measured 8K workload
is on the order of tenths of a percent.

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

### The decisive test for the scalar-INT2 quality failure

If the finite scalar-INT2 quality collapse is independent of serving, it
should reproduce with no engine at all. Running both models under plain
HuggingFace and overwriting the KV cache with its own quantize→dequantize
roundtrip, on the same layers and the same band the engine uses:

| model | layers quantized | bf16 | INT2 (HF simulation) |
|---|---|---|---|
| Llama-3.1-8B | 32 (all) | 10/10 | **10/10** |
| gpt-oss-20B | 12 (full-attn) | 4/10 | **0/10** |

The simulator reproduces the quality failure on gpt-oss and leaves Llama
completely untouched. **This establishes a limitation of scalar INT2 on this
architecture, independent of the piecewise-prefill bug.** The P2
implementation was faithfully delivering what OSCAR INT2 does in the eager/HF
experiment, even though the served graph path also had a separate defect.

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

- **VQ had a learned-sink correctness bug.** Fixing its coordinate
  error raises matched NIAH-8K from 61.46 to 79.17, while a verified all-BF16
  cache scores 8/8. The remaining bf16 gap appears only when rows enter the
  combined VQ-K/INT2-V tier, but the two quantizers have not yet been separated.
- **The result is stronger than the Llama grid shows.** On gpt-oss, 2-bit
  scalar quantization fails outright in the engine-free simulation while
  corrected VQ remains much stronger in served quality sweeps, although neither
  2-bit method matches bf16.
- **VQ2 currently has no demonstrated 8K throughput win on gpt-oss/A100.**
  Under a matched split-48 controlled run, bf16 is 39.5% faster than VQ2 and
  scalar INT2 is 11.4% faster than VQ2. Longer-context and split-count sweeps
  are required before making a tok/s speedup claim.
- **There is currently no valid INT2 baseline on gpt-oss.** Comparisons must
  use the corrected padded graph path, and even then scalar INT2 has the
  independently reproduced quality collapse. Comparisons should remain
  against bf16 unless scalar INT2 fidelity is improved.
- The natural follow-up is a **fidelity-threshold** framing: measure vq2's
  reconstruction on the same tensors, and turn "vq2 is better" into "gpt-oss
  requires ≥X fidelity; INT2 delivers 0.92, vq2 delivers Y."

---

## 8. Confidence and caveats

**Solid.** There are three independent findings. The scalar-INT2 quality collapse
reproduces with zero engine code while the Llama HF control stays 10/10 →
10/10. Separately, the served NaN localizes to padded quantized prefill, its
non-finite count exactly matches the padding footprint, disabling only decode
graphs does not remove it, the all-eager control passes, and exact-size-only
piecewise replay first passed as a diagnostic. The final padded-replay fix then
passed the exact sampled concurrency reproduction under both VQ2 and scalar
INT2 with every logged prefill and decode launch graphed. Finally, the VQ sink
error is derived algebraically, fails a discriminating softmax gate by 3.33e-2
when omitted, matches by 1.04e-4 when shifted, and improves the matched served
run by 17.7 points. Verified all-BF16-cache controls then separate that engine
fix from a residual quantized-tier loss. They do not assign that loss to VQ K
rather than the still-INT2 V tier.

**Repaired length-sweep evidence.** The old dose-response and
window-vs-full-attention probes used the malformed shortening helper and
unmatched greedy/cap settings; they must not be used. The matched repaired
sweep shows BF16 at 92.5–100% through 16K. Corrected VQ2 tracks BF16 through
2K and remains 85–92.5% at 4K–16K, while scalar INT2 falls to 0–5% from
1K onward. The old, unshifted-sink VQ2 path is worse at every length,
including 77.5% at the 238-token row where no prompt rows are quantized. It
establishes both the fixed learned-sink bug and a smaller residual corrected
VQ2 gap, but does not yet locate the minimum failing number of quantized rows.

The throughput comparison is also only two warmed runs per arm at one context
length, concurrency and split count. Its ordering is consistent across both
runs and is sufficient to reject an 8K speedup claim, but it is not a complete
performance characterization.

**Not established.** *Why* gpt-oss's attention output decorrelates at the first
scalar-quantized layer given better KV fidelity. The VQ coordinate error around
learned sinks is now isolated and fixed; it does not explain scalar INT2,
which performs no mean subtraction. The 82%
expert-flip rate is *consistent* with MoE amplification but does not separate
cause from consequence; flips could be downstream of an already-large drift.
The clean test is a **matched-noise control**: inject random KV noise into
Llama at a magnitude that produces drift 1.2 at layer 1, and see whether Llama
also collapses. That separates "gpt-oss is fragile" from "this much drift
kills any model."

**Process note.** Five probes produced instrumentation errors, and two controls
were interpreted too broadly. The pattern is uniform: a measurement was
read as a finding before it was shown able to distinguish a passing case from
a failing one.

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
- `DISABLE_CUDA_GRAPH=1` was labeled "graphs off", but it disabled decode
  graphs only; piecewise prefill graphs remained enabled and were the fault.
- `RECENT_TOKENS=16384` was initially labeled "all HP", but non-final 4096-token
  chunks deliberately quantize their middle. A stale remote launcher also
  ignored the requested one-chunk override. The valid control verified
  `chunked_prefill_size=16384`, disabled piecewise prefill, and ran one 8.26K
  request at a time.

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
PYTHONPATH=vendor/OSCAR-vq/sglang-research/python:. .venv-oscar/bin/python \
    pipelines/oscar_e2e/verify_vq_engine.py --gpu 1 \
    --bundle artifacts/oscar_gptoss20b/vqa_gptoss20b_G4_strat_flat_ptn_gpqacc128k_fp8.pt \
    --v-bundle /nonexistent --layers 0 1 5 11

# engine-free VQ-K x INT2-V factorial; each prefill row is quantized once
.venv/bin/python pipelines/oscar_e2e/dbg_vq_factorial.py \
    --gpus 0,1 --rows 3 --tokens 1024 --gen 96

# artifact-only loader-boundary diagnostic (CPU)
.venv/bin/python pipelines/oscar_e2e/dbg_vq_factorial.py --artifact-only

# live fused-decode audit: set this for an eager server run. Each TP worker
# writes one file; analyze each file independently.
SGLANG_VQ_AUDIT_DUMP=/tmp/vq-audit-{rank}-{pid}.pt \
SGLANG_VQ_AUDIT_LAYER=0 <start-vq2-server-with-decode-graphs-disabled>
.venv/bin/python pipelines/oscar_e2e/audit_vq_live_dump.py \
    /tmp/vq-audit-0-<pid>.pt

# served length sweep (against a running server)
.venv/bin/python pipelines/eval/export_prompt_rows.py \
    --model unsloth/gpt-oss-20b-BF16 --dataset niah --data-dir 16384 \
    --max-new-tokens 256 --out artifacts/prompt_rows/niah_16384_gptoss_t1.jsonl
python pipelines/oscar_e2e/dbg_gptoss_niah_lengths.py \
    --port <port> --mode <int2|vq2|bf16> \
    --lens 256,1024,2048,4096,8192,16384 \
    --n 10 --samples 4 --temperature 1 --max-new-tokens 256
# old VQ2 ablation: start the vq2 server with
# SGLANG_VQ_DISABLE_SINK_CORRECTION=1 and run the same client command with
# --mode old_vq2

# sampled concurrency regression (24 rows, four clients)
head -24 artifacts/prompt_rows/niah_8192_gptoss_t1.jsonl > /tmp/rows.jsonl
.venv/bin/python pipelines/oscar_e2e/run_prompts_client.py \
    --rows /tmp/rows.jsonl --port <port> --threads 4 --timeout 1800 \
    --samples 1 --temperature 1.0 --top-p 1.0 --top-k -1 --out /tmp/out.jsonl

# controlled throughput: start bf16, INT2 and VQ2 servers separately with
# KV_SPLITS=48, warm each, then use a fresh output directory for each run
.venv/bin/python pipelines/oscar_e2e/run_prompts_client.py \
    --rows /tmp/rows.jsonl --port <port> --threads 4 --timeout 1800 \
    --samples 1 --temperature 0 --ignore-eos --out /tmp/throughput-run

# Important: --disable-cuda-graph alone is not the relevant control.
# The fixed engine keeps piecewise graphs for padded INT2/vq2 token counts by
# representing graph padding as an isolated dummy sequence and clearing padded
# output rows at every attention boundary.
```

Logs on lambda-server6 (`10.137.32.78`): `logs/hfsim_n10.log`,
`logs/moeamp3_{gptoss,llama}.log`, `logs/kvfid_{gptoss,llama}.log`,
`logs/splits.log`, `logs/codec.log`, `logs/2x2.log`.

Related: `notes/page_quant/fixes_to_apply.md` entries **p2-1** (premature
mixed-KV flush double-assigns a quant slot, still latent), **p2-2** (the
`SWAKVPool.release_req_slab` delegate, now applied), and **p2-3** (the
allocate-before-flush ring overwrite fixed here). None caused the graph-padding
failure.
