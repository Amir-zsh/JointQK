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

## 2. Headline result

The follow-up supports a narrower and cleaner conclusion than Stage 1C:

> The harmful part is the anisotropic scaling in the full metric transform, not the oracle eigenbasis by itself.

But the stronger norm-spread claim was **not** established:

> This stage does not show that token-level norm spread is the main mechanism of failure.

## 3. Main quantitative result

The clearest evidence comes from the layer-0-excluded summary.

At `3` bits:

- baseline logit MSE / geometry distortion: `0.4618`
- basis-only logit MSE / geometry distortion: `0.4707`
- full metric logit MSE / geometry distortion: `0.6790`

- baseline top-1: `0.6815`
- basis-only top-1: `0.6808`
- full metric top-1: `0.5648`

The same pattern holds at `2` and `4` bits:

- `basis_only` stays close to baseline
- `full_metric` is clearly worse

Because geometry distortion and logit MSE match numerically in this setup, the important interpretation is:

- outside layer 0, the full metric path loses even on the quadratic objective it was supposed to help
- this is not just a ranking-vs-metric mismatch story

## 4. What happened to norm spread

The norm-dispersion diagnostics behave as expected directionally.

Layer-0-excluded transformed norm CV:

- basis-only: about `0.0975`
- full metric: about `0.3951`
- trace-matched full metric: about `0.3934`
- per-token norm-matched full metric: about `0.0975`

So the full metric transform does create much larger norm dispersion before the compressor.

However, the causal rescue tests did **not** recover performance:

- `trace_matched_full_metric` matched `full_metric`
- `per_token_norm_matched_full_metric` also matched `full_metric`

And the `gamma` sweep was not monotone:

- intermediate `gamma` values improved geometry distortion before the final `gamma = 1.0` point became worse than baseline outside layer 0

So the evidence does **not** support a simple one-factor story of:

`more norm spread -> directly worse result`

## 5. Why the norm-spread conclusion is limited

The current V3-style backend normalizes each vector before scalar quantization and stores the norm separately.

That means several seemingly natural norm controls are less decisive than they appear:

- global trace matching can be canceled by the backend's own norm handling
- tokenwise norm matching before compression is also partly neutralized by the same backend structure

So this stage should be read as:

- a successful test of **basis change vs scaling**
- but not yet a definitive test of **norm spread as the main mechanism**

## 6. Main conclusion from this stage

This stage supports the following conclusions:

1. The oracle eigenbasis alone is not the problem.
2. The full anisotropic metric scaling is the part that breaks the current oracle path.
3. This remains true even when success is interpreted through geometry distortion.
4. The current evidence does not isolate token-level norm spread as the dominant causal mechanism.

## 7. Implication for next steps

The next step should not be "estimate the same full metric online."

Instead, the method work should focus on one of:

- eigenbasis-aware methods that avoid full non-orthogonal preconditioning
- bit allocation or coordinate weighting in the oracle basis rather than direct full-metric scaling
- backend-aware ways of encoding eigenvalue information that are compatible with the V3 normalization-and-rotation path

So Stage 1D narrows the design space:

- keep the oracle basis as a useful object
- treat full metric scaling as the currently broken component
- do not claim that norm spread alone explains the failure yet
