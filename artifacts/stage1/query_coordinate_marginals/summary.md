# Individual Coordinate Marginals

These plots show true single-coordinate marginals, not pooled head-level values.

- Coordinates were chosen per selected head as:
  - one most Gaussian-like coordinate
  - one most skewed coordinate
  - one heaviest-tail coordinate

- Skew measures asymmetry. `0` means symmetric.
- Excess kurtosis measures tail-heaviness relative to a Gaussian. `0` means Gaussian-like tails.

Selected heads and coordinates:
- Layer 0, Head 0: Gaussian-like: C15 (|skew|=0.01, |kurt|=0.01), Most skewed: C63 (|skew|=0.73, |kurt|=6.05)
- Layer 12, Head 0: Gaussian-like: C109 (|skew|=0.02, |kurt|=0.01), Most skewed: C98 (|skew|=0.53, |kurt|=0.37), Heaviest tail: C13 (|skew|=0.03, |kurt|=1.36)
- Layer 23, Head 0: Gaussian-like: C29 (|skew|=0.02, |kurt|=0.01), Most skewed: C119 (|skew|=0.71, |kurt|=0.75), Heaviest tail: C49 (|skew|=0.61, |kurt|=1.81)
- Layer 35, Head 0: Gaussian-like: C78 (|skew|=0.01, |kurt|=0.00), Most skewed: C106 (|skew|=1.44, |kurt|=4.31)