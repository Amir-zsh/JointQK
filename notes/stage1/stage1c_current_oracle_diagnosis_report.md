# Stage 1C Report: Diagnosing the Current Oracle Path

> Supporting diagnosis note.
>
> This document captures the Stage 1C diagnosis of the original `Lk` oracle path. It remains useful as a historical diagnosis, but Stage 1D later narrowed the conclusion: the broken component is the anisotropic scaling term, while token-level norm spread was **not** isolated as the dominant mechanism. For the current summary, see [stage1_experiments_and_findings.md](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1_experiments_and_findings.md:1) and [stage1d_norm_spread_ablation_report.md](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1d_norm_spread_ablation_report.md:1).

This report summarizes the follow-up diagnosis performed after the initial stage-1 oracle study.

The question for this stage was:

> Why does the current geometry-aware oracle path underperform the baseline V3 path once layer 0 is excluded?

This stage did **not** test a new method. It diagnosed the **current** oracle implementation exactly as used in stage 1:

- baseline: V3-style MSE quantization on raw keys `k`
- oracle: current Cholesky-based metric transform followed by the same V3 backend on transformed keys `Lk`

The main artifacts for this stage are:

- [artifacts/stage1/current_oracle_diagnosis_gpu0/summary.md](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/current_oracle_diagnosis_gpu0/summary.md:1)
- [artifacts/stage1/current_oracle_diagnosis_gpu0/metrics.json](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/current_oracle_diagnosis_gpu0/metrics.json:1)
- [artifacts/stage1/current_oracle_diagnosis_gpu0/diagnosis.pt](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/current_oracle_diagnosis_gpu0/diagnosis.pt:1)

The most useful diagnosis figures are:

- [global_current_vs_baseline.png](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/current_oracle_diagnosis_gpu0/global_current_vs_baseline.png)
- [layer0_excluded_current_vs_baseline.png](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/current_oracle_diagnosis_gpu0/layer0_excluded_current_vs_baseline.png)
- [transform_condition_number_vs_logit_delta.png](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/current_oracle_diagnosis_gpu0/transform_condition_number_vs_logit_delta.png)
- [norm_cv_vs_logit_delta.png](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/current_oracle_diagnosis_gpu0/norm_cv_vs_logit_delta.png)
- [rotation_variance_spread_vs_logit_delta.png](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/current_oracle_diagnosis_gpu0/rotation_variance_spread_vs_logit_delta.png)

The new representative distribution panels for this stage are:

- [layer_00_panel.png](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/current_oracle_diagnosis_gpu0/panel_charts/layer_00_panel.png)
- [layer_03_panel.png](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/current_oracle_diagnosis_gpu0/panel_charts/layer_03_panel.png)
- [layer_17_panel.png](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/current_oracle_diagnosis_gpu0/panel_charts/layer_17_panel.png)
- [layer_28_panel.png](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/current_oracle_diagnosis_gpu0/panel_charts/layer_28_panel.png)

## 1. What this stage was trying to verify

The working hypothesis was:

\[
\text{V3}(k) \text{ beats } \text{V3}(Lk) \text{ because the current transform } L \text{ introduces anisotropy before the V3 backend.}
\]

More concretely, we wanted to know:

1. Is `Lk` more anisotropic than `k` before quantization?
2. After the same V3 random rotation, is rotated `Lk` still less well-behaved than rotated `k`?
3. Does that backend mismatch plausibly explain the oracle underperformance?

## 2. Headline result

The diagnosis reproduces the original stage-1 oracle result and supports the view that the current transform makes the quantizer's job harder.

The important caveat is that the evidence is **supportive but not perfectly clean**. The script's final label is `supported`, but the correlation evidence is mixed and should not be overstated.

The safe conclusion is:

> The current Cholesky-based transform produces substantially more anisotropic transformed keys, and that backend mismatch is a plausible explanation for why `V3(Lk)` underperforms `V3(k)`.

## 3. Reproduced performance result

The diagnosis first confirms that the previously observed failure is real.

From the full summary, the oracle path still looks better on the average quadratic metrics:

- 2-bit geometry distortion: `2.9138 -> 2.3406`
- 3-bit geometry distortion: `0.8724 -> 0.6911`
- 4-bit geometry distortion: `0.2431 -> 0.1900`

and similarly for logit MSE:

- 2-bit logit MSE: `2.9138 -> 2.3405`
- 3-bit logit MSE: `0.8724 -> 0.6910`
- 4-bit logit MSE: `0.2431 -> 0.1900`

But top-1 match gets worse:

- 2-bit top-1: `0.5234 -> 0.3895`
- 3-bit top-1: `0.6636 -> 0.5604`
- 4-bit top-1: `0.7784 -> 0.7088`

This by itself is not new. The important confirmation comes from the layer-0-excluded summary:

- 2-bit geometry distortion: `1.6412 -> 2.3190`
- 3-bit geometry distortion: `0.4618 -> 0.6845`
- 4-bit geometry distortion: `0.1250 -> 0.1881`

- 2-bit logit MSE: `1.6411 -> 2.3189`
- 3-bit logit MSE: `0.4618 -> 0.6845`
- 4-bit logit MSE: `0.1250 -> 0.1881`

- 2-bit top-1: `0.5380 -> 0.3960`
- 3-bit top-1: `0.6815 -> 0.5657`
- 4-bit top-1: `0.7970 -> 0.7119`

So after removing layer 0, the oracle path is worse than baseline on:

- the quadratic objective
- logit MSE
- ranking quality

This means the problem is not merely "better MSE but worse ranking." Outside layer 0, the current oracle path loses even on the metric it was supposed to help.

## 4. What changed in the distributions

The core of the diagnosis is the comparison between `k` and `Lk`, before and after the V3 backend steps.

### Before rotation

Excluding layer 0, the aggregate diagnostics show:

- baseline `k` pre-transform norm CV: about `0.0975`
- oracle `Lk` pre-transform norm CV: about `0.3951`

- baseline `k` effective rank: about `16.24`
- oracle `Lk` effective rank: about `6.70`

Interpretation:

- `Lk` has much more uneven norm distribution
- `Lk` is concentrated in fewer effective directions
- this is a clear sign of stronger anisotropy before V3 quantization

This effect is widespread, not a few outliers:

- oracle pre-transform norm CV is higher than baseline in `100%` of layer-example rows
- oracle effective rank is lower than baseline in about `78.6%` of rows

### After rotation

After the same V3-style random rotation, the gap gets smaller but does not disappear.

Excluding layer 0:

- baseline rotated variance spread: about `7.04`
- oracle rotated variance spread: about `7.30`

- baseline rotated mean absolute skew: about `0.161`
- oracle rotated mean absolute skew: about `0.247`

- baseline rotated mean absolute excess kurtosis: about `0.285`
- oracle rotated mean absolute excess kurtosis: about `0.427`

Interpretation:

- random rotation does not undo the anisotropy introduced by the current transform
- rotated `Lk` still looks less isotropic and less Gaussian-like than rotated `k`

Again this is not a rare-event story:

- oracle rotated variance spread exceeds baseline in about `71.4%` of rows

## 5. What the representative panel charts show

The 2x2 panel charts were produced specifically to visualize the distributional effect of the current transform:

- top-left: `k`
- top-right: rotated `k`
- bottom-left: `Lk`
- bottom-right: rotated `Lk`

Each panel annotates:

- skew
- excess kurtosis
- condition number

The selected layers were:

- `0` explicitly
- `3` as an early representative layer
- `17` as a middle representative layer
- `28` as a late representative layer

See [panel_charts/summary.md](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/current_oracle_diagnosis_gpu0/panel_charts/summary.md:1) for the exact chosen examples.

The visual pattern across these charts is consistent:

- `k` and rotated `k` are relatively well-behaved
- `Lk` is visibly more stretched and less regular
- rotated `Lk` remains more skewed and heavy-tailed than rotated `k`

Layer 0 is important because it is the one layer that helps the global average. Even there, the transform visibly changes the distribution. The later representative layers matter more for diagnosis, because they are the ones where the current oracle path actually fails.

## 6. Correlation evidence

The correlation analysis is directionally useful, but not decisive on its own.

Reported correlations from [summary.md](/vault/amir/efficient-llm/teamily-project/artifacts/stage1/current_oracle_diagnosis_gpu0/summary.md:1):

- `transformed_norm_cv` vs `delta_logit_mse`: `0.2523`
- `transform_condition_number_mean` vs `delta_logit_mse`: `-0.2229`
- `rotated_variance_spread` vs `delta_logit_mse`: `-0.0164`

- `transform_condition_number_mean` vs `delta_top1_match`: `0.4402`
- `rotated_variance_spread` vs `delta_top1_match`: `0.4303`
- `transformed_effective_rank` vs `delta_top1_match`: `-0.3367`

These should be interpreted carefully. The signs are not uniformly aligned with a single simple narrative, especially for top-1. That is why the diagnosis should not be read as "condition number alone explains everything."

The stronger case comes from combining:

- the consistent distributional shift from `k` to `Lk`
- the failure of the oracle path once layer 0 is excluded
- the widespread increase in norm dispersion and decrease in effective rank
- the representative panel charts

So the evidence is stronger as a structural diagnosis than as a single clean correlation story.

## 7. Main conclusion from this stage

This stage supports the following conclusions:

1. The negative oracle result is real and reproducible.
2. The current Cholesky-based transform makes the key distribution substantially more anisotropic before quantization.
3. V3 random rotation does not remove that mismatch; rotated `Lk` remains less well-behaved than rotated `k`.
4. The most plausible current explanation is that the non-orthogonal metric transform is interacting badly with the V3 backend.

This stage does **not** prove that query-aware geometry is wrong.

It only shows that the **current way** of injecting geometry into the V3 backend is not working robustly.

## 8. Implication for next steps

The next step should be to separate:

- orthogonal basis changes
- anisotropic scaling

That means testing SVD/eigenbasis-based variants explicitly, for example:

- orthogonal-only: `U^T k`
- full metric: `sqrt(\Lambda) U^T k`

The purpose of that next experiment would be to test whether the real problem is the scaling term rather than the eigenbasis rotation.

That follow-up belongs to the next stage-1 investigation step, not to this report.
