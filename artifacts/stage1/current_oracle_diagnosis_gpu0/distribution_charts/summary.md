# Distribution Diagnostics Summary

These charts use layer-0-excluded, bit-deduplicated rows from `diagnosis.pt`.

- Mean pre-transform norm CV: baseline `0.0975`, oracle `0.3951`
- Mean pre-transform effective rank: baseline `16.24`, oracle `6.70`
- Mean post-rotation variance spread: baseline `7.0360`, oracle `7.2973`
- Mean post-rotation |skew|: baseline `0.1615`, oracle `0.2466`
- Mean post-rotation |excess kurtosis|: baseline `0.2854`, oracle `0.4267`

## Representative Case

- task `qasper`, layer `0`, bits `2`
- transform condition number `658.70`

## Charts

- `pre_transform_norm_cv_comparison.png`
- `pre_transform_effective_rank_comparison.png`
- `post_rotation_variance_spread_comparison.png`
- `post_rotation_skew_comparison.png`
- `post_rotation_excess_kurtosis_comparison.png`
- `post_rotation_skew_vs_kurtosis.png`
- `representative_distribution_moments.png`

Interpretation:

- Higher norm CV and lower effective rank indicate stronger anisotropy before rotation.
- Higher variance spread, |skew|, and |excess kurtosis| indicate a less isotropic / less Gaussian-like distribution after rotation.