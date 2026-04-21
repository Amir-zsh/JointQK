# Stage 1 Experiments and Findings

This note summarizes the stage-1 work so far:

1. Query-distribution validation for the Expected Attention assumption.
2. Oracle key-only geometry-aware quantization versus the V3 baseline.
3. The main challenges uncovered by these experiments.

## Current Takeaway

If someone reads only one Stage 1 note, it should be this one.

The current Stage 1 position is:

- query second moments are worth modeling even though exact Gaussianity is false
- the original full-metric oracle path is not a robust win over the V3 baseline
- outside layer 0, the full-metric path loses even on geometry distortion
- the oracle eigenbasis by itself is not the problem; the harmful component is the anisotropic scaling term
- Stage 1D did **not** isolate token-level norm spread as the dominant mechanism, so that explanation should still be treated as provisional

The corresponding artifacts live under:

- [artifacts/stage1/query_stats](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/query_stats)
- [artifacts/stage1/query_distribution_charts](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/query_distribution_charts)
- [artifacts/stage1/query_coordinate_marginals](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/query_coordinate_marginals)
- [artifacts/stage1/visuals](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/visuals)
- [artifacts/stage1/oracle_v3_study](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/oracle_v3_study)
- [artifacts/stage1/oracle_v3_study_fixed_clean](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/oracle_v3_study_fixed_clean)
- [artifacts/stage1/oracle_norm_spread_study](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/oracle_norm_spread_study)

## 1. Goal of Stage 1

Stage 1 was designed to answer two questions before moving on to online estimators or rate allocation:

1. Are future queries Gaussian enough, or at least stable enough in second moment, to justify query-aware geometry?
2. In an oracle setting, does geometry-aware key quantization beat standard V3-style quantization?

The baseline quantizer is the V3-style single-stage compressor used in the stage-1 code. The oracle method computes a future-query second moment per KV head and quantizes after a geometry transform derived from that second moment.

## 2. Experiment A: Query Distribution Validation

### Setup

We collected pre-RoPE and post-RoPE query activations on a small LongBench-E slice and computed:

- coordinate-wise skew
- coordinate-wise excess kurtosis
- prompt-to-prompt second-moment stability
- representative histograms and QQ plots

The main numerical summary is in [query_stats/analysis/summary.md](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/query_stats/analysis/summary.md:1).

### Main findings

The strict statement "queries are Gaussian" is not supported.

Formal normality tests reject exact Gaussianity on representative heads. But the low-order moment picture is still fairly mild:

- Pre-RoPE mean absolute skew: `0.1986`
- Post-RoPE mean absolute skew: `0.1167`
- Pre-RoPE mean absolute excess kurtosis: `0.4497`
- Post-RoPE mean absolute excess kurtosis: `0.4937`
- Pre-RoPE mean prompt stability: `0.1973`
- Post-RoPE mean prompt stability: `0.2100`

Interpretation:

- Exact Gaussianity: no.
- Approximate Gaussian / second-moment modeling: yes, plausible enough.
- The safer modeling object is the empirical second moment
  \[
  M_q = E[qq^T]
  \]
  rather than a literal multivariate normal claim.

### Relevant charts

The pooled head-level diagnostics are in:

- [pre_query_histograms.png](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/query_distribution_charts/pre_query_histograms.png)
- [post_query_histograms.png](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/query_distribution_charts/post_query_histograms.png)
- [pre_query_qq.png](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/query_distribution_charts/pre_query_qq.png)
- [post_query_qq.png](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/query_distribution_charts/post_query_qq.png)
- [pre_skew_heatmap.png](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/query_distribution_charts/pre_skew_heatmap.png)
- [post_skew_heatmap.png](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/query_distribution_charts/post_skew_heatmap.png)
- [pre_kurtosis_heatmap.png](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/query_distribution_charts/pre_kurtosis_heatmap.png)
- [post_kurtosis_heatmap.png](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/query_distribution_charts/post_kurtosis_heatmap.png)

The individual-coordinate marginals are more informative for the normality question:

- [pre_coordinate_histograms.png](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/query_coordinate_marginals/pre_coordinate_histograms.png)
- [post_coordinate_histograms.png](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/query_coordinate_marginals/post_coordinate_histograms.png)
- [pre_coordinate_qq.png](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/query_coordinate_marginals/pre_coordinate_qq.png)
- [post_coordinate_qq.png](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/query_coordinate_marginals/post_coordinate_qq.png)

These coordinate-level plots show the real picture:

- some coordinates are very close to normal
- some coordinates are clearly skewed
- some coordinates have visibly heavy tails

This is documented in [query_coordinate_marginals/summary.md](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/query_coordinate_marginals/summary.md:1).

## 3. Experiment B: Oracle Geometry-Aware Quantization

### Setup

We compared two key-only methods:

- baseline: standard V3-style MSE quantization
- oracle: geometry-aware quantization using the true future-query second moment

For each layer and KV head, the oracle metric is
\[
M_q = \frac{1}{N}\sum_{j=1}^N q_j q_j^T
\]
where the `q_j` are actual future queries from the same example after the prefix split.

The oracle branch then:

1. factors `M_q`
2. transforms keys into query-aware geometry
3. applies the same V3-style compressor in transformed space
4. maps the reconstruction back

This is an oracle upper-bound experiment in the sense that it uses true future queries, not an online estimator.

### Global results

From [oracle_v3_study_fixed_clean/summary.md](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/oracle_v3_study_fixed_clean/summary.md:1), the oracle method improves the two quadratic metrics it was designed around:

- 2-bit geometry distortion: `2.9138 -> 2.3406`
- 3-bit geometry distortion: `0.8724 -> 0.6911`
- 4-bit geometry distortion: `0.2431 -> 0.1900`

and similarly:

- 2-bit logit MSE: `2.9138 -> 2.3405`
- 3-bit logit MSE: `0.8724 -> 0.6910`
- 4-bit logit MSE: `0.2431 -> 0.1900`

But ranking quality gets worse:

- 2-bit top-1: `0.5234 -> 0.3895`
- 3-bit top-1: `0.6636 -> 0.5604`
- 4-bit top-1: `0.7784 -> 0.7088`

So the high-level result is:

- the oracle method reduces average quadratic error
- but it degrades attention ranking behavior

### Relevant charts

The overview is in:

- [oracle_global_comparison.png](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/visuals/oracle_global_comparison.png)
- [oracle_delta_scatter.png](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/visuals/oracle_delta_scatter.png)

The scatter is especially important:

- left is better on logit MSE
- up is better on top-1

Most of the mass is not in the ideal quadrant. That already shows a mismatch between the optimized objective and ranking-sensitive behavior.

## 4. Layer-Wise Breakdown

The global average is misleading.

When we decomposed the results by layer, a few layers dominated the average, especially layer 0. This is visible in:

- [oracle_per_layer_logit_mse.png](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/visuals/oracle_per_layer_logit_mse.png)
- [oracle_per_layer_top1_delta.png](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/visuals/oracle_per_layer_top1_delta.png)
- [layer0_dominance.png](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/visuals/layer0_dominance.png)

The practical conclusion is that the oracle improvement on average logit MSE is heavily driven by the first layer.

### Excluding layer 0

If we ignore layer 0 and average over the remaining `210` layer-example pairs, the conclusion flips completely.

For geometry distortion and logit MSE:

| Bits | Method | Geometry Dist. | Logit MSE |
| --- | --- | ---: | ---: |
| 2 | Baseline | 1.6412 | 1.6412 |
| 2 | Oracle | 2.3190 | 2.3189 |
| 3 | Baseline | 0.4618 | 0.4618 |
| 3 | Oracle | 0.6845 | 0.6845 |
| 4 | Baseline | 0.1250 | 0.1250 |
| 4 | Oracle | 0.1881 | 0.1881 |

And ranking remains worse as well:

- 2-bit top-1: `0.5380 -> 0.3960`
- 3-bit top-1: `0.6815 -> 0.5657`
- 4-bit top-1: `0.7970 -> 0.7119`

So after removing layer 0:

- the baseline beats the oracle method even on the quadratic objective
- this is no longer only a ranking-vs-MSE issue

## 5. Implementation Audit

We also audited the oracle implementation and found two real code issues:

1. the eigen-fallback inverse for the geometry factorization was wrong
2. the geometry-aware branch had an extra cast-back that made the comparison less clean

Both were fixed, and the oracle study was rerun cleanly into [oracle_v3_study_fixed_clean](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/oracle_v3_study_fixed_clean).

Those fixes did **not** materially change the results. The patched rerun matches the original run up to tiny numerical differences.

So the negative result is not explained by a simple implementation bug.

## 6. What We Learned

### Supported conclusions

The experiments support the following:

1. Query second moments are stable enough to be a meaningful object.
2. Exact Gaussianity is false, but a Gaussian / second-moment approximation is still reasonable as a modeling device.
3. The current oracle geometry-aware quantizer does not improve the behavior we care about in a robust way.

### Important non-conclusion

The experiments do **not** prove that query-aware geometry is a bad idea.

They only show that the current way we combine geometry with the V3/TurboQuant-style backend does not yield a robust win.

## 7. Core Challenges

### Challenge 1: Objective mismatch

The current oracle method is built around a quadratic key distortion proxy:
\[
(k-\hat k)^T M_q (k-\hat k)
\]
or equivalently average future logit MSE.

But the downstream attention behavior is sensitive to:

- ranking
- margins
- softmax competition

These are not well captured by average squared logit error. This is why logit MSE can improve while top-1 worsens.

### Challenge 2: Layer imbalance

The average result is dominated by a small number of layers, especially layer 0.

This makes aggregate metrics hard to trust unless they are accompanied by:

- per-layer curves
- median statistics
- layer-0-excluded summaries

### Challenge 3: Geometry-aware transform versus V3 backend

The current method uses a query-aware transform and then applies the same V3 quantizer. But this does **not** mean we are exactly optimizing geometry distortion.

The backend still performs:

- per-vector normalization
- random rotation
- scalar Lloyd-Max quantization

So the actual method is "apply a V3-style heuristic in transformed space," not "solve geometry distortion minimization exactly."

This matters because a non-orthogonal geometry transform can change the distribution seen by the quantizer in ways that may make the backend less appropriate.

### Challenge 4: Interpreting the role of orthogonal versus non-orthogonal transforms

An orthogonal SVD basis alone is not enough to encode the full metric. The eigenvalue weighting still matters.

So there is a design tension:

- full metric matching requires scaling as well as rotation
- but non-orthogonal scaling may interact badly with the V3 backend

This is one of the main unresolved algorithmic questions.

### Challenge 4A: What Stage 1D resolved and what it did not

The Stage 1D follow-up is documented in [stage1d_norm_spread_ablation_report.md](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1d_norm_spread_ablation_report.md:1).

That follow-up sharpened the Stage 1C diagnosis:

- `basis_only` stays close to baseline outside layer 0
- `full_metric` is worse than baseline outside layer 0
- the harmful component is therefore the anisotropic scaling term, not the oracle eigenbasis by itself

But Stage 1D did **not** establish that token-level norm spread is the dominant mechanism:

- `trace_matched_full_metric` did not recover the loss
- `per_token_norm_matched_full_metric` also did not recover the loss
- the `gamma` sweep was not monotone

So the clean supported conclusion is narrower:

- scaling is implicated
- norm spread is a plausible contributor
- but the current evidence does not isolate norm spread alone as the full explanation

### Challenge 5: Ranking-sensitive alternatives

The current metric is too weak if the end goal is attention preservation.

The next round likely needs one or more of:

- margin-aware logit objectives
- rank-aware objectives
- softmax-aware objectives
- layer-weighted objectives
- orthogonal-basis methods that use eigenvalue weights through allocation rather than through full preconditioning

## 8. Current Bottom Line

The stage-1 bottom line is:

- query-aware second moments are worth modeling
- exact normality is not required and not supported
- the current geometry-aware quantization method is not a robust win over standard V3
- the main problem is not just implementation; it is the current objective-and-backend coupling
- Stage 1D further shows that the oracle eigenbasis is not the problem by itself; the full anisotropic scaling term is the part that breaks the current path
- Stage 1D does not yet prove that token-level norm spread is the main mechanism, especially because the current V3 backend normalizes vectors before scalar quantization

So the project is still alive, but the next step should be:

**refine the way geometry is injected into the quantizer, especially by avoiding direct full-metric preconditioning, rather than moving directly to online estimation or rate allocation.**
