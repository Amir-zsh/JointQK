# Stage 1E Report: Partial-Spectrum Generalization Check

> Current method follow-up note.
>
> This report should be read together with [stage1_experiments_and_findings.md](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1_experiments_and_findings.md:1). Stage 1D diagnosed why direct full-metric preconditioning fails. Stage 1E tests the narrower positive method signal from Stage 1D: partial metric scaling with `gamma = 0.25`.

This report summarizes the Stage 1E follow-up experiment run after the Stage 1D norm-spread ablation.

The question for this stage was:

> Does the robust `3`-bit `gamma = 0.25` partial-spectrum result from Stage 1D generalize to `2`-bit and `4`-bit settings?

This stage reused the Stage 1 oracle setup:

- model: `Qwen/Qwen3-8B`
- data: `LongBench-E` slice with `qasper`, `hotpotqa`, `passage_retrieval_en`
- examples: `2` per config, `6` total
- layers: `36`
- keys only
- oracle future-query second moment
- bitwidths: `2`, `3`, `4`

The main artifacts for this stage are:

- [artifacts/stage1/oracle_partial_spectrum_study/summary.md](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/oracle_partial_spectrum_study/summary.md:1)
- [artifacts/stage1/oracle_partial_spectrum_study/metrics.json](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/oracle_partial_spectrum_study/metrics.json:1)
- [artifacts/stage1/oracle_partial_spectrum_study/variant_study.pt](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/oracle_partial_spectrum_study/variant_study.pt:1)

## 1. What this stage tested

The experiment compared:

- `baseline_raw`
- `basis_only`
- `full_metric`
- `trace_matched_full_metric`
- `gamma_sweep` with `gamma = 0.125`
- `gamma_sweep` with `gamma = 0.25`
- `gamma_sweep` with `gamma = 0.5`

Unlike Stage 1D, the gamma variants were evaluated at all three bitwidths: `2`, `3`, and `4`.

The primary metrics are:

- geometry distortion, the quadratic oracle objective
- top-1 match, the ranking-sensitive companion metric
- median deltas and win rates against the raw V3 baseline

As in Stage 1D, geometry distortion and logit MSE are effectively duplicate quadratic signals in this harness, so geometry distortion is the primary reported quadratic metric.

## 2. Headline result

Stage 1E is a strong positive result:

> `gamma = 0.25` beats the raw V3 baseline on both geometry distortion and top-1 match for all three bitwidths, including after excluding layer 0.

The generated study decision is:

- `strong_success`

The result is not layer-0-driven.

## 3. Layer-0-excluded results

The most important numbers are the layer-0-excluded summary.

| Bits | Method | Geometry Dist. | Top-1 |
| --- | --- | ---: | ---: |
| 2 | Baseline | `1.6972` | `0.4847` |
| 2 | Gamma 0.25 | `0.9027` | `0.5486` |
| 3 | Baseline | `0.4727` | `0.6251` |
| 3 | Gamma 0.25 | `0.2218` | `0.6794` |
| 4 | Baseline | `0.1276` | `0.7512` |
| 4 | Gamma 0.25 | `0.0577` | `0.7898` |

Against baseline, the layer-0-excluded deltas and win rates are:

| Bits | Mean Geometry Delta | Median Geometry Delta | Geometry Win | Mean Top-1 Delta | Median Top-1 Delta | Top-1 Win | Both Win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | `-0.7945` | `-0.4189` | `0.990` | `+0.0639` | `+0.0434` | `0.810` | `0.805` |
| 3 | `-0.2509` | `-0.1288` | `1.000` | `+0.0543` | `+0.0227` | `0.776` | `0.776` |
| 4 | `-0.0699` | `-0.0362` | `1.000` | `+0.0386` | `+0.0080` | `0.752` | `0.752` |

Interpretation:

- geometry improves almost universally
- top-1 improves for most layer-example pairs
- the effect remains after removing the layer that distorted the original full-metric aggregate

## 4. Comparison with `gamma = 0.5`

`gamma = 0.5` can produce slightly lower mean geometry distortion in some settings, especially at `4` bits.

But it is less robust on top-1:

- layer-0-excluded `gamma = 0.5` both-win rate is `0.586` at `2` bits
- layer-0-excluded `gamma = 0.5` both-win rate is `0.533` at `3` bits
- layer-0-excluded `gamma = 0.5` both-win rate is `0.419` at `4` bits

By contrast, `gamma = 0.25` has both-win rates of `0.805`, `0.776`, and `0.752`.

So `gamma = 0.25` is the safer current default. It gives up a small amount of quadratic distortion in some cases but preserves ranking behavior much more reliably.

## 5. Main conclusion from this stage

Stage 1E supports the following conclusions:

1. The partial-spectrum method signal from Stage 1D is real across `2`, `3`, and `4` bits on this oracle slice.
2. The positive result is not driven by layer 0.
3. `gamma = 0.25` is the best current default for fixed-bit oracle partial-spectrum key quantization under the V3 backend.
4. Direct full-metric scaling remains the wrong mechanism to estimate online.
5. The next method step can reasonably move from oracle partial-spectrum scaling to estimated-query versions of the same softened geometry, or to allocation methods that use the oracle spectrum without full preconditioning.

## 6. Implication for next steps

The project no longer needs to ask whether softened query-aware geometry can beat the raw V3 baseline on internal key metrics. On the current oracle slice, it can.

The next step should be one of:

- estimate the same softened `gamma = 0.25` query geometry online
- test whether `gamma = 0.25` improves downstream generation quality
- use the oracle eigenbasis and spectrum for bit allocation rather than direct preconditioning

The key constraint remains:

- do not estimate or deploy the failed full-metric `gamma = 1.0` path as the main method.
