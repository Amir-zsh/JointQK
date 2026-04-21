# Stage 1D: Oracle Norm-Spread Study

## Full Summary

- 2-bit logit MSE: Baseline `2.9138`, Basis Only `2.6203`, Full Metric `2.3253`, Trace-Matched `2.3253`, Per-Token Norm-Matched `2.3253`
- 2-bit geometry distortion: Baseline `2.9138`, Basis Only `2.6203`, Full Metric `2.3253`, Trace-Matched `2.3253`, Per-Token Norm-Matched `2.3253`
- 2-bit top-1: Baseline `0.5234`, Basis Only `0.5257`, Full Metric `0.3855`, Trace-Matched `0.3855`, Per-Token Norm-Matched `0.3855`
- 3-bit logit MSE: Baseline `0.8724`, Basis Only `0.8084`, Full Metric `0.6867`, Trace-Matched `0.6867`, Per-Token Norm-Matched `0.6867`
- 3-bit geometry distortion: Baseline `0.8724`, Basis Only `0.8084`, Full Metric `0.6867`, Trace-Matched `0.6867`, Per-Token Norm-Matched `0.6867`
- 3-bit top-1: Baseline `0.6636`, Basis Only `0.6630`, Full Metric `0.5594`, Trace-Matched `0.5594`, Per-Token Norm-Matched `0.5594`
- 4-bit logit MSE: Baseline `0.2431`, Basis Only `0.2266`, Full Metric `0.1858`, Trace-Matched `0.1858`, Per-Token Norm-Matched `0.1858`
- 4-bit geometry distortion: Baseline `0.2431`, Basis Only `0.2266`, Full Metric `0.1858`, Trace-Matched `0.1858`, Per-Token Norm-Matched `0.1858`
- 4-bit top-1: Baseline `0.7784`, Basis Only `0.7779`, Full Metric `0.7092`, Trace-Matched `0.7092`, Per-Token Norm-Matched `0.7092`

## Layer-0-Excluded Summary

- 2-bit logit MSE: Baseline `1.6411`, Basis Only `1.6530`, Full Metric `2.2995`, Trace-Matched `2.2995`, Per-Token Norm-Matched `2.2995`
- 2-bit top-1: Baseline `0.5380`, Basis Only `0.5403`, Full Metric `0.3921`, Trace-Matched `0.3921`, Per-Token Norm-Matched `0.3921`
- 2-bit transformed norm CV: Basis Only `0.0975`, Full Metric `0.3951`, Trace-Matched `0.3934`, Per-Token Norm-Matched `0.0975`
- 3-bit logit MSE: Baseline `0.4618`, Basis Only `0.4707`, Full Metric `0.6790`, Trace-Matched `0.6790`, Per-Token Norm-Matched `0.6790`
- 3-bit top-1: Baseline `0.6815`, Basis Only `0.6808`, Full Metric `0.5648`, Trace-Matched `0.5648`, Per-Token Norm-Matched `0.5648`
- 3-bit transformed norm CV: Basis Only `0.0975`, Full Metric `0.3951`, Trace-Matched `0.3934`, Per-Token Norm-Matched `0.0975`
- 4-bit logit MSE: Baseline `0.1250`, Basis Only `0.1286`, Full Metric `0.1839`, Trace-Matched `0.1839`, Per-Token Norm-Matched `0.1839`
- 4-bit top-1: Baseline `0.7970`, Basis Only `0.7963`, Full Metric `0.7124`, Trace-Matched `0.7124`, Per-Token Norm-Matched `0.7124`
- 4-bit transformed norm CV: Basis Only `0.0975`, Full Metric `0.3951`, Trace-Matched `0.3934`, Per-Token Norm-Matched `0.0975`

## Correlation Evidence

- Full data `transformed_norm_cv` vs `delta_logit_mse`: `-0.0325`
- Full data `transformed_norm_cv` vs `delta_top1_match`: `-0.1457`
- Layer-0-excluded `transformed_norm_cv` vs `delta_logit_mse`: `0.1639`
- Layer-0-excluded `transformed_norm_cv` vs `delta_top1_match`: `-0.2088`

## Spectrum Sweep

- gamma=0.0: norm CV `0.0975`, delta logit MSE `0.0089`, delta top-1 `-0.0007`
- gamma=0.25: norm CV `0.1022`, delta logit MSE `-0.2561`, delta top-1 `0.0576`
- gamma=0.5: norm CV `0.1749`, delta logit MSE `-0.2712`, delta top-1 `0.0524`
- gamma=0.75: norm CV `0.2800`, delta logit MSE `-0.1707`, delta top-1 `-0.0053`
- gamma=1.0: norm CV `0.3934`, delta logit MSE `0.2172`, delta top-1 `-0.1167`

## Conclusion

- If `basis_only` is near baseline while `full_metric` is worse, scaling is implicated: `True`.
- If `trace_matched_full_metric` is still bad, the issue is not only global energy inflation: `True`.
- If `per_token_norm_matched_full_metric` recovers a substantial fraction of the loss, token-level norm spread is the main mechanism: `False` with mean recovery fraction `-0.0000`.
- If performance degrades monotonically with `gamma`, that strengthens the causal claim: `False`.