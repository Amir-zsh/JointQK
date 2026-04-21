# Stage 1D Report: Oracle Norm-Spread Ablation

> Current supporting diagnosis note.
>
> This is the most up-to-date Stage 1 diagnostic follow-up, but it should be read together with [stage1_experiments_and_findings.md](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1_experiments_and_findings.md:1). The strongest supported conclusion is about **basis vs scaling**. The norm-spread interpretation remains limited because the current backend normalizes vectors before scalar quantization.

This report summarizes the Stage 1D follow-up experiment run after the Stage 1C diagnosis.

The question for this stage was:

> Does the current oracle path underperform mainly because the non-orthogonal scaling in the query-aware metric transform increases transformed-key norm spread, which the V3/TurboQuant backend then handles badly?

This stage reused the Stage 1 oracle setup:

- model: `Qwen/Qwen3-8B`
- data: `LongBench-E` slice with `qasper`, `hotpotqa`, `passage_retrieval_en`
- keys only
- oracle future-query second moment
- bitwidths: `2`, `3`, `4`

The main artifacts for this stage are:

- [artifacts/stage1/oracle_norm_spread_study/summary.md](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/oracle_norm_spread_study/summary.md:1)
- [artifacts/stage1/oracle_norm_spread_study/metrics.json](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/oracle_norm_spread_study/metrics.json:1)
- [artifacts/stage1/oracle_norm_spread_study/variant_study.pt](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/oracle_norm_spread_study/variant_study.pt:1)

The most useful figures are:

- [variant_grouped_bars.png](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/oracle_norm_spread_study/variant_grouped_bars.png)
- [variant_grouped_bars_layer0_excluded.png](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/oracle_norm_spread_study/variant_grouped_bars_layer0_excluded.png)
- [norm_cv_vs_delta_logit_mse.png](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/oracle_norm_spread_study/norm_cv_vs_delta_logit_mse.png)
- [norm_cv_vs_delta_top1_match.png](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/oracle_norm_spread_study/norm_cv_vs_delta_top1_match.png)
- [gamma_sweep_lines.png](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/oracle_norm_spread_study/gamma_sweep_lines.png)
- [recovery_comparison.png](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/oracle_norm_spread_study/recovery_comparison.png)

## 1. What this stage tested

The transform-family ablation compared:

- `baseline_raw`: no geometry transform
- `basis_only`: oracle eigenbasis only
- `full_metric`: full oracle metric transform
- `trace_matched_full_metric`: full metric with global trace matching
- `per_token_norm_matched_full_metric`: full metric with tokenwise norm matching before compression

It also ran a `gamma` sweep at `3` bits:

- `gamma = 0.0`: basis-only
- `gamma = 1.0`: trace-matched full metric

The primary interpretation for this stage should be based on **geometry distortion**, not generic key MSE, because the point of the experiment was to understand whether the backend can realize the intended oracle geometry objective.

In this harness, geometry distortion and logit MSE are not independent signals. They are numerically the same quadratic error up to tiny floating-point noise because both are built from the same future-query second moment. So the Stage 1D interpretation should keep **geometry distortion** and treat `logit_mse` as redundant reporting.

## 2. Headline result

The follow-up supports two separate conclusions that should not be conflated.

First, there is a genuine positive method result:

> A partial-metric oracle with intermediate `gamma` values, especially around `0.25` to `0.5`, clearly beats the V3 baseline at `3` bits on both geometry distortion and top-1, in both the full-data and layer-0-excluded summaries.

Second, the mechanism conclusion is narrower than the original Stage 1C norm-spread story:

> The harmful part is the anisotropic scaling in the full metric transform, not the oracle eigenbasis by itself.

But the stronger norm-spread claim was **not** established:

> This stage does not show that token-level norm spread is the main mechanism of failure.

More strongly: the current Stage 1D control arms are not sufficient to reject that mechanism, because two of them collapse under the backend algebra.

## 3. Main quantitative results

The clearest evidence comes from the layer-0-excluded summary.

At `3` bits:

- baseline geometry distortion: `0.4618`
- basis-only geometry distortion: `0.4707`
- full metric geometry distortion: `0.6790`

- baseline top-1: `0.6815`
- basis-only top-1: `0.6808`
- full metric top-1: `0.5648`

The same pattern holds at `2` and `4` bits:

- `basis_only` stays close to baseline
- `full_metric` is clearly worse

That is the main negative diagnosis result.

But there is also a strong positive result in the `gamma` sweep at `3` bits.

Full data:

- baseline geometry distortion: `0.8724`
- `gamma = 0.25`: `0.2311`
- `gamma = 0.5`: `0.1962`

- baseline top-1: `0.6636`
- `gamma = 0.25`: `0.7268`
- `gamma = 0.5`: `0.7274`

Layer-0-excluded:

- baseline geometry distortion: `0.4618`
- `gamma = 0.25`: `0.2057`
- `gamma = 0.5`: `0.1906`

- baseline top-1: `0.6815`
- `gamma = 0.25`: `0.7390`
- `gamma = 0.5`: `0.7338`

So Stage 1D should explicitly record that a **partial-metric oracle is a meaningful win** under the current backend.

Because geometry distortion and logit MSE match numerically in this setup, the important interpretation is:

- outside layer 0, the full metric path loses even on the quadratic objective it was supposed to help
- this is not just a ranking-vs-metric mismatch story
- intermediate metric scaling can nevertheless outperform the plain V3 baseline by a large margin

## 4. What happened to norm spread

The norm-dispersion diagnostics behave as expected directionally.

Layer-0-excluded transformed norm CV:

- basis-only: about `0.0975`
- full metric: about `0.3951`
- trace-matched full metric: about `0.3934`
- per-token norm-matched full metric: about `0.0975`

So the full metric transform does create much larger norm dispersion before the compressor.

However, that apparent "failed rescue" evidence has to be interpreted carefully.

- `trace_matched_full_metric` matched `full_metric`
- `per_token_norm_matched_full_metric` also matched `full_metric`

Under the current V3-style compressor, those matches are expected by construction:

- the compressor unit-normalizes every transformed vector before scalar quantization
- `trace_matched_full_metric` differs from `full_metric` only by a per-head scalar factor
- `per_token_norm_matched_full_metric` differs from `full_metric` only by a per-token scalar factor that is later explicitly undone before applying the inverse transform

So these are **degenerate controls** under the current backend. Their failure to recover performance should not be read as evidence that norm spread is unimportant.

And the `gamma` sweep was not monotone:

- intermediate `gamma` values improved geometry distortion before the final `gamma = 1.0` point became worse than baseline outside layer 0

So the evidence does **not** support a simple one-factor story of:

`more norm spread -> directly worse result`

## 5. Why the norm-spread conclusion is limited

The current V3-style backend normalizes each vector before scalar quantization and stores the norm separately.

That means several seemingly natural norm controls are less decisive than they appear:

- global trace matching is canceled by the backend's own norm handling
- tokenwise norm matching before compression is also canceled once the compressor unit-normalizes and the explicit post-decompression undo step is applied

So this stage should be read as:

- a successful test of **basis change vs scaling**
- a successful demonstration that partial metric scaling can help
- but not yet a definitive test of **norm spread as the main mechanism**

## 6. Main conclusion from this stage

This stage supports the following conclusions:

1. The oracle eigenbasis alone is not the problem.
2. The full anisotropic metric scaling is the part that breaks the current oracle path.
3. Partial metric scaling at `gamma ≈ 0.25-0.5` is a genuine positive method result and currently the strongest Stage 1 evidence that geometry-aware structure can help under the V3 backend.
4. This remains true when success is interpreted through geometry distortion rather than duplicate quadratic MSE reporting.
5. The current evidence still does not isolate token-level norm spread as the dominant causal mechanism.
6. The trace-matched and per-token-norm-matched arms should not be used as negative evidence against the norm-spread hypothesis, because they are degenerate under the current backend.

## 7. Implication for next steps

The next step should not be "estimate the same full metric online."

Instead, the method work should focus on one of:

- eigenbasis-aware methods that avoid full non-orthogonal preconditioning
- bit allocation or coordinate weighting in the oracle basis rather than direct full-metric scaling
- backend-aware ways of encoding eigenvalue information that are compatible with the V3 normalization-and-rotation path

So Stage 1D narrows the design space:

- keep the oracle basis as a useful object
- treat full metric scaling as the currently broken component
- treat partial metric scaling as the most promising current Stage 1 method signal
- do not claim that norm spread alone explains the failure yet
- do not treat the current degenerate norm-control arms as evidence against the norm-spread story
