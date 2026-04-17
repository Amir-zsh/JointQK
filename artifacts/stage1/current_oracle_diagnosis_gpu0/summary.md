# Current Oracle Diagnosis

## Full Summary

- 2-bit geometry distortion: `2.9138 -> 2.3406`
- 2-bit logit MSE: `2.9138 -> 2.3405`
- 2-bit top-1: `0.5234 -> 0.3895`
- 3-bit geometry distortion: `0.8724 -> 0.6911`
- 3-bit logit MSE: `0.8724 -> 0.6910`
- 3-bit top-1: `0.6636 -> 0.5604`
- 4-bit geometry distortion: `0.2431 -> 0.1900`
- 4-bit logit MSE: `0.2431 -> 0.1900`
- 4-bit top-1: `0.7784 -> 0.7088`

## Layer-0-Excluded Summary

- 2-bit geometry distortion: `1.6412 -> 2.3190`
- 2-bit logit MSE: `1.6411 -> 2.3189`
- 2-bit top-1: `0.5380 -> 0.3960`
- 3-bit geometry distortion: `0.4618 -> 0.6845`
- 3-bit logit MSE: `0.4618 -> 0.6845`
- 3-bit top-1: `0.6815 -> 0.5657`
- 4-bit geometry distortion: `0.1250 -> 0.1881`
- 4-bit logit MSE: `0.1250 -> 0.1881`
- 4-bit top-1: `0.7970 -> 0.7119`

## Correlations

- `rotated_variance_spread__vs__delta_logit_mse`: `-0.0164`
- `rotated_variance_spread__vs__delta_top1_match`: `0.4303`
- `transform_condition_number_mean__vs__delta_logit_mse`: `-0.2229`
- `transform_condition_number_mean__vs__delta_top1_match`: `0.4402`
- `transformed_effective_rank__vs__delta_logit_mse`: `-0.0602`
- `transformed_effective_rank__vs__delta_top1_match`: `-0.3367`
- `transformed_norm_cv__vs__delta_logit_mse`: `0.2523`
- `transformed_norm_cv__vs__delta_top1_match`: `0.0316`

## Decision

- `supported`