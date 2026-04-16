# Stage 1 Plan: Query Normality and Oracle Geometry Study

This document specifies the first research stage in concrete terms.

The purpose of Stage 1 is to answer two questions:

1. Are query activations approximately normal, as claimed in Expected Attention?
2. If future queries are known, does geometry-aware key quantization beat standard TurboQuant?

We should not proceed to online estimators, value compression, or pruning until these are answered.

---

## Stage 1A: Query Normality Validation

## Objective

Validate whether query activations are well-approximated by a Gaussian distribution at the level needed by the project.

This is not a philosophical question. We do not need perfect normality. We only need the Gaussian approximation to be useful enough to induce a good geometry matrix:

```math
M_q = \mathbb{E}[q q^T]
```

## What to measure

For each chosen model, layer, and head:

- marginal distribution of each query coordinate
- coordinate-wise skewness
- coordinate-wise kurtosis
- goodness-of-fit to a Gaussian after standardization
- covariance structure stability across prompts
- whether the empirical second moment is dominated by a low-rank structure

The most important practical outputs are:

- how Gaussian the marginals look
- whether the estimated second moment is stable and usable

## Models

Use one model first:

- `Qwen/Qwen3-8B` or
- `meta-llama/Llama-3.1-8B-Instruct`

Start with one model only. Expand later if Stage 1 succeeds.

## Data

Use a small but varied long-context set:

- a LongBench-E slice
- a synthetic retrieval prompt
- a code/document-style long context if available

The point is not benchmark coverage yet. The point is to sample realistic query activations from diverse long contexts.

## Experimental procedure

1. Run prefill on each context.
2. Extract pre-RoPE queries and post-RoPE queries for each layer/head.
3. Save a subsample of query vectors by token position.
4. For each layer/head:
   - standardize each coordinate
   - compute skewness and kurtosis
   - compare histograms to Gaussian overlays
   - run a normality test on a manageable subsample
5. Compare:
   - all tokens
   - tokens after removing the first few sink or outlier positions
   - pre-RoPE vs post-RoPE queries

## Decision rule

We should proceed if at least one of the following is true:

- pre-RoPE queries are approximately Gaussian enough across most layers/heads
- post-RoPE queries are approximately Gaussian enough
- even if marginals are imperfect, the empirical second moment `M_q` is stable and well-behaved

We should be cautious if:

- heavy tails dominate most heads
- the second moment is extremely unstable across prompts
- Gaussian approximation fails badly and cannot be rescued by excluding sink tokens or using simple robust preprocessing

## Deliverables

- a short report with plots and summary statistics
- a conclusion on whether to use:
  - Gaussian query model
  - empirical second moment only
  - both

## Recommended implementation

Reuse `kvpress` query extraction logic:

- `kvpress/kvpress/presses/expected_attention_press.py`

In particular:

- use `get_prerope_query_states`
- mirror the split between pre-RoPE statistics and average future-position RoPE application

Output files should include:

- cached query tensors or summary statistics
- plots per representative layer/head
- one markdown summary

---

## Stage 1B: Oracle Geometry-Aware Quantization Study

## Objective

Test the core scientific claim of the project:

**if future queries are known, quantizing keys in the future-query geometry should beat standard TurboQuant at equal average bitrate**

This is an oracle study. We are allowed to use true future queries because this stage is about whether the idea is valid at all.

## Scope

Keep the first study narrow:

- keys only
- no pruning
- token preserving
- fixed bitwidth first
- then variable bit allocation if fixed-bit results are positive

## Baselines

At minimum compare:

1. full precision keys
2. standard TurboQuant baseline in original space
3. geometry-aware quantization using oracle future-query geometry

Prefer using the practical MSE-only TurboQuant variant, not QJL, because:

- QJL is likely harmful for softmax attention
- the local `turboquant-pytorch` repo already treats MSE-only as the practical default

## Geometry-aware method

For each layer/head and each evaluation block:

1. collect future queries `q_1, ..., q_n`
2. compute oracle second moment

```math
M_q = \frac{1}{n}\sum_{j=1}^n q_j q_j^T
```

3. regularize

```math
M_q^{reg} = M_q + \epsilon I
```

4. factor

```math
M_q^{reg} = L^T L
```

5. transform keys

```math
\tilde{k}_i = L k_i
```

6. quantize `\tilde{k}_i` using the same quantizer family used by the baseline
7. compare against standard quantization of `k_i` directly

This isolates the effect of the geometry.

## Metrics

Use both internal and downstream metrics.

### Internal

- key reconstruction MSE in original space
- geometry-aware key distortion

```math
(\hat{k}_i - k_i)^T M_q (\hat{k}_i - k_i)
```

- attention logit MSE on actual future queries
- top-k attention ranking agreement
- attention-output error if easy to compute

### Downstream

Start with one simple downstream task:

- Needle-in-a-Haystack or
- a small LongBench-E slice

Do not over-expand benchmarks in Stage 1. Internal-metric clarity matters more.

## Bitwidths

First compare fixed bitwidths:

- `2`
- `3`
- `4`

If results are positive, add a small variable-bit experiment with:

- `{2, 3, 4, 8}`

## Decision rule

We proceed only if:

- geometry-aware quantization beats standard TurboQuant at equal average bitrate on internal metrics
- and preferably improves downstream performance on at least one task

If geometry-aware quantization does not beat standard TurboQuant in the oracle setting, we should not continue with online geometry estimation yet.

## Deliverables

- internal-metric comparison plots
- one downstream comparison table
- conclusion: whether the oracle geometry effect is real

---

## Code Reuse Strategy

## Query extraction

Use:

- `kvpress/kvpress/presses/expected_attention_press.py`

Main reusable pieces:

- pre-RoPE query extraction
- average-RoPE handling
- head-aware query statistics

## Quantization baseline

Use:

- `turboquant-pytorch/compressors_v3.py`
- `turboquant-pytorch/turboquant.py`

For Stage 1, the practical baseline should be the MSE-only path:

- `MSECompressor`

This avoids conflating the geometry question with the QJL question.

## Suggested code organization

Create a new research workspace under a dedicated directory such as:

```text
experiments/
  stage1/
    collect_queries.py
    analyze_query_normality.py
    run_oracle_quant_study.py
    metrics.py
    plots.py
    outputs/
```

The exact layout can change, but Stage 1 outputs should be reproducible and isolated from the upstream repos.

---

## Recommended Execution Order

1. Implement query extraction and caching.
2. Run a small query-normality study.
3. Decide whether to use Gaussian modeling, empirical `M_q`, or both.
4. Implement oracle geometry-aware key quantization.
5. Compare against standard TurboQuant at fixed bitwidths.
6. Only if positive, add a small variable-bit extension.

---

## Stage 1 Success Criteria

Stage 1 is successful if all of the following hold:

- query activations are Gaussian enough or at least yield a stable usable second moment
- oracle geometry-aware quantization improves over standard TurboQuant at equal bitrate
- the effect is visible on both internal metrics and at least one downstream task

If this fails, revise the modeling assumptions before moving on.
