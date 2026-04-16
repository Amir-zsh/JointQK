# Rate-Distortion Perspectives on KV Cache Compression
## A Query-Aware and Importance-Aware Proposal for Long-Context LLM Inference

---

## 1. Introduction

Large language models are rapidly scaling to context windows of 128K tokens and beyond, enabling applications such as multi-document reasoning, codebase understanding, video analysis, and agentic workflows. In this regime, the central systems bottleneck is no longer only the model weights, which are fixed, but the internal state that grows linearly with sequence length. During autoregressive inference, every processed token contributes a key and value vector to the KV cache, and these cached representations must be stored in memory and repeatedly loaded to compute attention for each newly generated token.

This growth creates multiple production bottlenecks at once. The KV cache can dominate GPU memory usage, directly limiting batch size and throughput. Decoding is often memory-bandwidth-bound rather than compute-bound, so reducing bits per KV entry can directly improve time per output token. In disaggregated serving systems, the KV cache must be transferred between prefill and decode workers, so cache size also determines transfer latency. In prefix-caching systems, the cached state is reused across requests, making token-count-preserving compression especially attractive because it preserves the structure required for reuse.

These observations motivate a broader thesis: KV cache compression should not be viewed merely as a local optimization of tensor storage, but as a principled allocation problem over memory, bandwidth, and downstream model fidelity. Rate-distortion theory provides a natural language for expressing this trade-off.

This document develops that perspective around two foundational questions: what object should be compressed, and what notion of distortion should guide compression? We argue that the KV cache is the right target not only because it is practically important, but also because it admits distortion measures that are internal, local, and comparatively cheap to compute. Unlike delayed metrics such as final task performance or even perplexity, KV-level distortion can be evaluated directly through its effect on attention logits, attention outputs, and residual updates. Within this tractable class of distortion measures, however, generic reconstruction error is still too crude; the more appropriate objective is expected downstream error under future query usage. Recent work on Expected Attention and Attention Matching further suggests that future query usage can be approximated well enough to guide high-quality KV compression, while TurboQuant provides a lightweight online quantization backend with strong distortion-rate behavior. Building on these ingredients, we propose a rate-distortion view of KV cache compression in which pruning and quantization become part of a single allocation problem over bits and downstream error.

---

## 2. The Two Foundational Questions

Any principled compression framework for LLMs must answer two foundational questions.

### 2.1 What are we compressing?

There are several possible objects of compression in the LLM pipeline:

- **Input tokens**, as in prompt compression methods that reduce the context before inference.
- **Model weights**, via quantization or pruning of learned parameters.
- **Intermediate activations**, which are transient internal states.
- **The KV cache**, which stores attention keys and values for all processed tokens.

Each object induces a different optimization problem. Input compression changes what the model sees. Weight compression changes the model itself. Activation compression must deal with rapidly changing states across layers and tokens. The KV cache sits in a distinctive middle ground: it is input-dependent, so adaptive compression can exploit content-specific structure, but it is also stable after computation, since each token's key and value are fixed once produced during prefill.

### 2.2 What is distortion?

Compression quality can also be measured at several levels:

- **Final task performance**, such as accuracy or benchmark score.
- **Perplexity or log-likelihood**, which provide output-level proxies.
- **Internal representation distortion**, such as errors in attention logits, attention outputs, residual-stream updates, or hidden states.

The most meaningful distortion measure is final task performance, but it is also the most delayed and
expensive: evaluating it requires full autoregressive generation and often an external task evaluator.
Perplexity is cheaper than task performance, but it still requires running the model over the full
sequence and remains too expensive to serve as a low-level objective inside a compression procedure.
Internal distortion measures are fundamentally different. They are local to the computation, can often
be evaluated during or immediately after a forward pass, and therefore make optimization tractable.
The key design challenge is to find an internal distortion measure that is cheap enough to optimize yet
meaningful enough to correlate with downstream quality.

---

## 3. Why the KV Cache Is the Right Compression Target

The answers to the two foundational questions point naturally to the KV cache.

First, the KV cache admits distortion measures that are local and tractable. Unlike task performance
or perplexity, KV-level distortion can be measured directly through its effect on attention logits,
attention outputs, and residual updates. This makes optimization feasible.

Second, the KV cache is both stable enough and adaptive enough. Each token's key and value are
computed once and then reused throughout decoding, so the representation itself is much less
transient than generic activations. At the same time, it remains input-specific, so adaptive
compression can still exploit context-dependent structure.

Third, the KV cache is where the systems gains are largest. Reducing bits per KV entry improves
memory usage, decode bandwidth, offloading cost, and inter-node transfer latency. Unlike aggressive
query-dependent compaction, token-count-preserving compression also remains compatible with
prefix caching and multi-request reuse.

For these reasons, KV cache compression is a natural setting in which to ask a rate-distortion question: how should limited memory or bandwidth be allocated across cached representations so as to minimize the model's downstream error?

---

## 4. Background

### 4.1 KV Cache and Attention

For a decoder-only transformer, let $h_i$ denote the hidden state at token position $i$. For one attention head,

$$
q_i = R_i W_Q h_i, \qquad k_i = R_i W_K h_i, \qquad v_i = W_V h_i,
$$

where $R_i$ is the RoPE matrix and $W_Q, W_K, W_V$ are learned projections. At generation step $t$, the attention score assigned by the current query $q_t$ to cached token $i$ is

$$
a_{ti} = \frac{\exp\left(q_t^\top k_i / \sqrt d\right)}{\sum_{j \le t} \exp\left(q_t^\top k_j / \sqrt d\right)}.
$$

The corresponding attention output contributes to the residual stream as

$$
h_t^{\mathrm{out}} = h_t + \sum_{i \le t} a_{ti} W_O v_i.
$$

This decomposition already suggests two distinct channels of compression error:

- compressing a **key** perturbs future attention logits and weights;
- compressing a **value** perturbs the residual update once the token is attended to.

### 4.2 Future-Query Modeling: Expected Attention and Attention Matching

One of the main difficulties in KV compression is that token importance depends on future queries, which are not yet known when compression decisions are made. Recent work suggests that this obstacle can be handled by replacing unknown future queries with an approximation of future query usage.

Expected Attention takes an explicitly probabilistic route. If hidden states are approximately Gaussian, then future queries can also be modeled as Gaussian after projection through $W_Q$ and RoPE:

$$
q \sim \mathcal{N}(\bar\mu_q, \bar\Sigma_q).
$$

For a fixed key $k_i$, the expected unnormalized attention score can then be computed in closed form:

$$
\hat z_i
= \mathbb{E}_q \left[\exp\left(\frac{q^\top k_i}{\sqrt d}\right)\right]
= \exp\left(\frac{\bar\mu_q^\top k_i}{\sqrt d} + \frac{k_i^\top \bar\Sigma_q k_i}{2d}\right).
$$

Normalizing yields an expected attention weight $\hat a_i$, which is used to estimate future token importance. In Expected Attention, this signal is used primarily for pruning: tokens with low expected contribution are evicted. The main conceptual contribution for our purposes is broader: future-query distributions can provide a principled, query-aware notion of importance without requiring access to actual future queries.

Attention Matching takes a different but closely related route. Instead of postulating an explicit distribution, it uses a representative set of reference queries and asks for a compressed cache that preserves the attention behavior induced by those queries. In effect, this replaces an explicit probabilistic model of future queries with an empirical approximation of future query usage. The resulting compaction objective is defined by matching attention outputs and attention mass on the reference-query set.

These two methods share the same high-level lesson: good KV compression requires modeling how the cache will be used by future queries, not just reconstructing keys and values as generic vectors. In this proposal, we use this more abstract idea of **future-query modeling** as the conceptual starting point. When we need a concrete analytical formulation, we will often specialize to Expected Attention because its Gaussian assumption yields tractable closed-form expressions.

### 4.3 TurboQuant

TurboQuant studies online vector quantization for high-dimensional vectors under distortion objectives such as mean-squared error and inner-product error. Its design is data-oblivious: a random rotation spreads vector energy across coordinates, after which coordinate-wise scalar quantization becomes highly effective. This yields a lightweight quantization procedure suitable for online settings such as runtime caching.

TurboQuant is appealing for KV cache compression because it does not require heavy calibration or codebook learning, and because it comes with strong distortion-rate behavior for generic vectors. However, its guarantees are intentionally worst-case and task-agnostic. It does not know which tokens are likely to matter, which key directions future queries will probe, or which value directions most affect the residual stream. In other words, TurboQuant is a strong quantization engine, but not yet a KV-aware one.

---

## 5. Gap and Opportunity

Current KV compression methods expose a gap between pruning-style and quantization-style approaches.

Pruning and compaction methods, including Expected Attention and Attention Matching, exploit future-query information but largely operate through token selection or latent compaction rather than bit allocation. Quantization methods operate on a richer action space by allowing graded precision, but they usually optimize generic reconstruction error such as MSE and therefore do not exploit future-query structure in a principled way. In practice, a moderately important token may be better stored at low precision than either dropped entirely or kept at full precision.

This suggests a missing middle ground. If future query usage can be approximated, whether through an explicit distribution as in Expected Attention or a representative query set as in Attention Matching, then compression should not optimize worst-case reconstruction of keys and values as generic vectors. Instead, it should allocate bits according to how future queries are expected to use the cache. This is exactly the type of allocation problem that rate-distortion theory is designed to formalize.

---

## 6. Problem Formulation

We propose to formulate KV cache compression as a rate-distortion optimization problem.

Let the compression scheme be denoted by $\mathcal{C}$. The general objective is

$$
\min_{\mathcal{C}} \; D(\mathcal{C}) + \lambda R(\mathcal{C}),
$$

where $R(\mathcal{C})$ measures the rate, i.e. the memory or bandwidth cost of the compressed KV cache, and $D(\mathcal{C})$ measures the distortion induced in downstream model behavior. The multiplier $\lambda$ controls the trade-off between quality and compression.

At a finer granularity, let $b_i$ denote the bit budget allocated to cached token $i$, and let $\hat k_i, \hat v_i$ denote its compressed key and value. Then a tokenwise formulation is

$$
\min_{\{b_i, \hat k_i, \hat v_i\}}
\sum_i D_i(\hat k_i, \hat v_i, b_i)
\;+\;
\lambda \sum_i b_i.
$$

The central modeling choice is the distortion term. We argue that it should be future-query-aware.

For keys, if $e_i^{(K)} = \hat k_i - k_i$, then the attention logit perturbation induced by a future query $q$ is

$$
\delta \ell_i = \frac{q^\top e_i^{(K)}}{\sqrt d}.
$$

The natural distribution-aware key distortion is therefore

$$
\mathbb{E}_q[\delta \ell_i^2]
= \frac{1}{d} e_i^{(K)\top} \mathbb{E}[q q^\top] e_i^{(K)}.
$$

More generally, the matrix $\mathbb{E}[q q^\top]$ should be interpreted as the second moment of whichever future-query model is used. Under a Gaussian query model, $\mathbb{E}[q q^\top] = \bar\Sigma_q + \bar\mu_q \bar\mu_q^\top$, so the second moment of future queries defines the geometry of important key-space directions. We use this Gaussian case as the main formulation because it leads to closed-form expressions, but the broader rate-distortion view does not require the estimator itself to be Gaussian.

For values, if $e_i^{(V)} = \hat v_i - v_i$, then a first-order approximation of output distortion is

$$
\Delta h_i^{(V)} \approx a_i W_O e_i^{(V)}.
$$

Replacing unknown future attention with expected attention suggests a value-side distortion of the form

$$
D_i^{(V)} \propto \hat a_i^2 \|W_O e_i^{(V)}\|_2^2.
$$

These expressions imply a unified view: keys should be preserved according to how future queries read them, and values should be preserved according to how strongly they affect the residual stream once read. This moves the objective away from uniform MSE and toward expected downstream distortion under future usage.

In this perspective, pruning is simply the special case $b_i = 0$. Variable-precision quantization fills the continuum between zero bits and full precision. The rate-distortion problem therefore naturally unifies pruning and quantization in a single framework.

---

## 7. Proposed Solution Family

We propose a family of query-aware, importance-aware KV compression methods built from two components:

1. a future-query modeler that provides token importance and key-space geometry,
2. a lightweight quantization backend that can operate online under runtime constraints.

At a high level, the solution family includes three progressively richer ideas.

### 7.1 Importance-Aware Rate Allocation

The first step is to use estimated future importance to allocate bit-widths non-uniformly across cached tokens. Tokens expected to receive more future attention, or to produce larger downstream residual effects when attended, should receive more bits. This turns future-query modeling from a pruning or compaction signal into a rate-allocation signal.

### 7.2 Query-Aware and Residual-Aware Representation Compression

The second step is to move beyond token-level rate allocation and make the representation itself
geometry-aware. On the key side, quantization should preserve directions that future queries are likely
to probe strongly, yielding a key distortion metric based on expected logit error rather than generic
reconstruction error. On the value side, quantization should preserve directions that most affect the
residual stream after attention selection, yielding a value distortion metric based on downstream output
effect rather than raw value MSE.

In this sense, both keys and values should be compressed in task-aware geometries: keys according to
future-query statistics, values according to residual-stream sensitivity. This produces a unified
representation-aware view of KV compression rather than treating key quantization and value
quantization as unrelated problems.

### 7.3 Hybrid Policies

Finally, pruning and quantization can be unified into a single allocation problem. Some tokens may deserve zero bits and be evicted, some may deserve low precision, and some may deserve high precision. This yields a continuous policy rather than a binary keep-or-drop decision.

---

## 8. Why Future-Query Modeling and TurboQuant Fit Together

Future-query modeling methods and TurboQuant play complementary roles.

Expected Attention and Attention Matching both provide the missing task structure. They tell us, through different mechanisms, which cached tokens are likely to matter and which aspects of the cache must be preserved to maintain downstream attention behavior. TurboQuant provides the missing quantization machinery: an online, lightweight, runtime-friendly vector quantizer with strong distortion-rate behavior.

Neither ingredient is sufficient alone. Future-query modeling by itself yields pruning or compaction signals but not a full precision-allocation mechanism. TurboQuant by itself provides strong generic quantization but not a query-aware objective. Combined, they suggest a practical design principle: use a future-query model to define the distortion geometry and importance weights, then use TurboQuant as the low-level engine for compressing the resulting transformed representations. In the present proposal, Expected Attention remains especially useful because its Gaussian assumption produces a clean analytical formulation for this combination.

---

## References

1. Claude E. Shannon. "Coding Theorems for a Discrete Source with a Fidelity Criterion." *IRE National Convention Record*, vol. 7, pt. 4, 1959, pp. 142-163. IEEE Xplore: https://ieeexplore.ieee.org/document/5311476

2. Alessio Devoto, Maximilian Jeblick, and Simon Jégou. "Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution." *arXiv preprint* arXiv:2510.00636, 2025. arXiv: https://arxiv.org/abs/2510.00636. DOI: https://doi.org/10.48550/arXiv.2510.00636

3. Amir Zandieh, Majid Daliri, Majid Hadian, and Vahab Mirrokni. "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate." *arXiv preprint* arXiv:2504.19874, 2025. arXiv: https://arxiv.org/abs/2504.19874. DOI: https://doi.org/10.48550/arXiv.2504.19874

4. Adam Zweiger, Xinghong Fu, Han Guo, and Yoon Kim. "Fast KV Compaction via Attention Matching." *arXiv preprint* arXiv:2602.16284, 2026. arXiv: https://arxiv.org/abs/2602.16284. DOI: https://doi.org/10.48550/arXiv.2602.16284
