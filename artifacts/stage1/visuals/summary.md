# Stage 1 Visualization Summary

## Query Distribution

- Pre-RoPE mean absolute skew: `0.1986`
- Post-RoPE mean absolute skew: `0.1167`
- Pre-RoPE mean prompt stability: `0.1973`
- Post-RoPE mean prompt stability: `0.2100`

## Oracle Study

- 2-bit logit MSE: `2.9138 -> 2.3405`
- 2-bit top-1: `0.5234 -> 0.3895`
- 3-bit logit MSE: `0.8724 -> 0.6910`
- 3-bit top-1: `0.6636 -> 0.5604`
- 4-bit logit MSE: `0.2431 -> 0.1900`
- 4-bit top-1: `0.7784 -> 0.7088`

## Charts

- `query_summary_bars.png`
- `second_moment_stability.png`
- `oracle_global_comparison.png`
- `oracle_per_layer_logit_mse.png`
- `oracle_per_layer_top1_delta.png`
- `oracle_delta_scatter.png`
- `layer0_dominance.png`