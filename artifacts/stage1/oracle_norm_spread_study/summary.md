# Stage 1D: Oracle Norm-Spread Study

## Headline

- `geometry_distortion` is the primary quadratic metric for this study. In this harness it numerically matches `logit_mse` because both are computed from the same future-query second moment, so the summary reports geometry distortion only.
- A partial-metric oracle is a real positive result here: the best `3`-bit `gamma` is `0.5` on full data and `0.5` on the layer-0-excluded slice, and both beat the V3 baseline on geometry distortion and top-1.
- The trace-matched and per-token-norm-matched control arms are mathematically degenerate under the current unit-normalizing backend, so they do not provide valid evidence against the norm-spread mechanism.
- That positive result is therefore separate from the original mechanism question. The sweep still does not validate a simple `more norm spread -> worse backend behavior` story, but the degenerate controls also do not falsify it.

## Full Summary

- 2-bit geometry distortion: Baseline `2.9138`, Basis Only `2.6203`, Full Metric `2.3253`, Trace-Matched `2.3253`, Per-Token Norm-Matched `2.3253`
- 2-bit top-1: Baseline `0.5234`, Basis Only `0.5257`, Full Metric `0.3855`, Trace-Matched `0.3855`, Per-Token Norm-Matched `0.3855`
- 3-bit geometry distortion: Baseline `0.8724`, Basis Only `0.8084`, Full Metric `0.6867`, Trace-Matched `0.6867`, Per-Token Norm-Matched `0.6867`
- 3-bit top-1: Baseline `0.6636`, Basis Only `0.6630`, Full Metric `0.5594`, Trace-Matched `0.5594`, Per-Token Norm-Matched `0.5594`
- 4-bit geometry distortion: Baseline `0.2431`, Basis Only `0.2266`, Full Metric `0.1858`, Trace-Matched `0.1858`, Per-Token Norm-Matched `0.1858`
- 4-bit top-1: Baseline `0.7784`, Basis Only `0.7779`, Full Metric `0.7092`, Trace-Matched `0.7092`, Per-Token Norm-Matched `0.7092`

## Layer-0-Excluded Summary

- 2-bit geometry distortion: Baseline `1.6412`, Basis Only `1.6530`, Full Metric `2.2996`, Trace-Matched `2.2996`, Per-Token Norm-Matched `2.2996`
- 2-bit top-1: Baseline `0.5380`, Basis Only `0.5403`, Full Metric `0.3921`, Trace-Matched `0.3921`, Per-Token Norm-Matched `0.3921`
- 2-bit transformed norm CV: Basis Only `0.0975`, Full Metric `0.3951`, Trace-Matched `0.3934`, Per-Token Norm-Matched `0.0975`
- 3-bit geometry distortion: Baseline `0.4618`, Basis Only `0.4707`, Full Metric `0.6791`, Trace-Matched `0.6791`, Per-Token Norm-Matched `0.6791`
- 3-bit top-1: Baseline `0.6815`, Basis Only `0.6808`, Full Metric `0.5648`, Trace-Matched `0.5648`, Per-Token Norm-Matched `0.5648`
- 3-bit transformed norm CV: Basis Only `0.0975`, Full Metric `0.3951`, Trace-Matched `0.3934`, Per-Token Norm-Matched `0.0975`
- 4-bit geometry distortion: Baseline `0.1250`, Basis Only `0.1286`, Full Metric `0.1839`, Trace-Matched `0.1839`, Per-Token Norm-Matched `0.1839`
- 4-bit top-1: Baseline `0.7970`, Basis Only `0.7963`, Full Metric `0.7124`, Trace-Matched `0.7124`, Per-Token Norm-Matched `0.7124`
- 4-bit transformed norm CV: Basis Only `0.0975`, Full Metric `0.3951`, Trace-Matched `0.3934`, Per-Token Norm-Matched `0.0975`

## Correlation Evidence

- Correlations below use only non-degenerate variants (`basis_only`, `full_metric`, and the `gamma` sweep), excluding the trace-matched and per-token-norm-matched control arms.
- Full data `transformed_norm_cv` vs `delta_geometry_distortion`: `-0.0173`
- Full data `transformed_norm_cv` vs `delta_top1_match`: `-0.3905`
- Layer-0-excluded `transformed_norm_cv` vs `delta_geometry_distortion`: `0.3541`
- Layer-0-excluded `transformed_norm_cv` vs `delta_top1_match`: `-0.4803`

## Spectrum Sweep: Full Data

- gamma=0.0: norm CV `0.1024`, delta geometry distortion `-0.0641`, delta top-1 `-0.0006`
- gamma=0.25: norm CV `0.1062`, delta geometry distortion `-0.6414`, delta top-1 `0.0632`
- gamma=0.5: norm CV `0.1776`, delta geometry distortion `-0.6762`, delta top-1 `0.0638`
- gamma=0.75: norm CV `0.2814`, delta geometry distortion `-0.5755`, delta top-1 `0.0070`
- gamma=1.0: norm CV `0.3926`, delta geometry distortion `-0.1857`, delta top-1 `-0.1042`

## Spectrum Sweep: Layer-0-Excluded

- gamma=0.0: norm CV `0.0975`, delta geometry distortion `0.0089`, delta top-1 `-0.0007`
- gamma=0.25: norm CV `0.1022`, delta geometry distortion `-0.2561`, delta top-1 `0.0576`
- gamma=0.5: norm CV `0.1749`, delta geometry distortion `-0.2712`, delta top-1 `0.0524`
- gamma=0.75: norm CV `0.2800`, delta geometry distortion `-0.1707`, delta top-1 `-0.0053`
- gamma=1.0: norm CV `0.3934`, delta geometry distortion `0.2173`, delta top-1 `-0.1167`

## Conclusion

- If `basis_only` is near baseline while `full_metric` is worse, scaling is implicated: `True`.
- Partial-metric scaling is a real method signal: intermediate `gamma` values improve geometry distortion and top-1 over baseline on both full data and the layer-0-excluded slice.
- The trace-matched and per-token-norm-matched variants should not be treated as causal rescue tests here, because unit-normalizing compression makes them effectively degenerate with `full_metric`.
- The original norm-spread mechanism claim still lacks clean support because the non-degenerate `gamma` sweep is not monotone: `False`.