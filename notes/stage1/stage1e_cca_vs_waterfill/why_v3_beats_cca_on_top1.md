# Why does V3 beat CCA-waterfill on top-1 retention, despite higher geometry distortion?

> Diagnostic write-up. The puzzle: at b_avg = 3, layer-0-excluded:
>
> | Method | top-1 ↑ | geometry distortion ↓ |
> |---|---:|---:|
> | `v_waterfill` | 0.760 | 0.066 |
> | `v3` | 0.682 | 0.456 |
> | `cca_waterfill` | 0.535 | 0.097 |
>
> CCA has **5× lower** geometry distortion than V3, yet **15 pp lower** top-1. V_waterfill beats both on both. Why?

## TL;DR

Top-1 retention is determined by `|q^T (k_top1 - k̂_top1)|` at the **specific top-1 key**, not by the average reconstruction error norm. Empirically (representative head: layer 12, kv_head 5, b=3):

| Method | mean ‖err‖ / ‖k‖ | mean \|δ\| at top-1 | mean \|δ\| at random key | **\|δ\| amplification at top-1** |
|---|---:|---:|---:|---:|
| `v3` | 0.182 | 0.99 | 0.33 | **3.0×** |
| `v_waterfill` | 0.164 | 0.32 | 0.18 | **1.7×** |
| `cca_waterfill` | **0.146** | **3.34** | 0.22 | **15.1×** |

CCA has the *smallest* total reconstruction error but a *15× amplification* at the top-1 key — leading to the worst top-1 retention. The top-1 key's logit is also pulled systematically *negative* (mean signed δ at top-1 = −3.34 for CCA), pulling it toward the runner-up.

The mechanism is structural, in two parts:

1. **CCA's basis is sorted by Q-K canonical correlation, which makes top-1 keys statistically more extreme in their per-coord values than random keys.** In the CCA basis at this head, the top-1 key's mean per-coord |z|-score is **1.14**, vs **0.80** for a random key. In the V eigenbasis, top-1 and random keys are equally extreme (0.74 vs 0.80). In the original (≈V3 random-rotation input) basis, top-1 keys are *less* extreme than random (0.37 vs 0.81).
2. **Lloyd-Max scalar quantization with a finite codebook has larger error at the distribution tail than at the bulk** (the codebook is designed for the bulk Gaussian). So a basis that puts top-1 keys preferentially in the tail will preferentially clip them.

Combining (1) and (2): CCA's basis, by design, places top-1 keys exactly where its quantizer is least accurate. V3's random rotation is "blind" to top-1 structure — top-1 keys end up in the bulk like random keys. V_waterfill's basis (V eigenvectors of `M_q`) is Q-only and doesn't expose top-1 keys' tails either.

## How we got here

### Hypothesis 1 (refuted): "CCA water-fill discards a low-ρ subspace; the discarded mass corrupts top-1."

At `b_avg = 3` on this head, **CCA_waterfill has 0 zero-bit coords**. The water-fill spreads 3 bits/coord-average across all 128 coords (allocations range 1–7 bits). The bulk discarded-subspace argument doesn't apply at `b ≥ 3`. *(It does apply at `b = 2` for some heads, but the top-1 gap exists at all bit budgets.)*

### Hypothesis 2 (refuted): "CCA's error covariance is shaped like Σ_K, putting noise on the same axes as the keys."

Theoretical prediction: `Cov(err)_CCA ≈ θ · P_K_inv P_K_inv^T = θ · Σ_K`. Empirically, `Cov(err)_CCA` has Frobenius-cosine **0.24** with `Σ_K` — not a strong match. (The match drops because the per-canonical-coord noise is not uniform — water-fill makes it inversely proportional to the Q-weighted variance, which breaks the clean proportional-to-Σ_K shape.)

The variance prediction `q^T Cov(err) q` is consistent across methods (V3=0.017 noise/signal ratio, V_wf=0.006, CCA=0.008) but **fails to predict top-1 retention**. The mean predicted `sqrt(q^T Cov(err) q)` for CCA is 0.28, but the *measured* `|δ| at top-1` is **3.34**. So the noise has structure beyond a Gaussian random covariance — it has a **bias correlated with the specific key being quantized**.

### Hypothesis 3 (confirmed): "The top-1 key's coordinate values are tail outliers in the CCA basis specifically."

Per-coord z-score (centered, scaled by per-coord std) of top-1 keys vs random keys:

| Basis | top-1 mean \|z\| | random mean \|z\| | top-1 max \|z\| | random max \|z\| |
|---|---:|---:|---:|---:|
| **CCA-canonical (sorted by ρ)** | **1.14** | 0.80 | 7.93 | 2.79 |
| V eigenbasis (sorted by λ) | 0.74 | 0.80 | 7.17 | 2.70 |
| original (≈V3 input) | 0.37 | 0.81 | 10.50 | 2.70 |

In the CCA basis the top-1 key is meaningfully *more extreme* than a random key on average (the histogram in `top1_extremeness_per_basis.png` shows a clear heavy-tail shift). In the V eigenbasis they're indistinguishable. In the original basis, top-1 keys are actually *less* extreme on average than random keys.

### Why the CCA basis exposes top-1 tails

A heuristic argument: the CCA basis is the SVD of the whitened cross-moment `Σ_Q^{-1/2} C_QK Σ_K^{-1/2}`. By construction, the top-1 key for query `q` is the one whose canonical-K coordinates `c_k = P_K k` align most strongly with `q`'s canonical-Q coordinates `c_q = P_Q q` on the high-ρ coordinates. Selecting "the key best aligned with `q` on a specific axis" pushes the chosen key toward extreme values on that axis — that's exactly the tail of the per-coord distribution.

In the V eigenbasis (`V = eigvecs(M_q)`), `V` only knows about Q's structure. It doesn't sort coordinates by anything specific to the key distribution, so top-1 selection doesn't preferentially push k's coordinates into V's per-coord tails.

In V3's random rotation, the basis is direction-blind. Top-1 keys end up with typical per-coord values for the same reason that the angular distance between two random vectors concentrates around 90°: averaging over enough random directions destroys structure.

### Why Lloyd-Max scalar quantization makes this matter

Lloyd-Max codebooks are optimized for a Gaussian source and concentrate centroids in the bulk of the distribution. At `b = 3` the codebook has 8 centroids spaced roughly across `±2.5 σ`. Values beyond that range get pushed back toward the nearest extreme centroid — a one-sided clipping bias. For CCA's basis, the top-1 key has |z|-scores reaching 7.9 σ, which gets clipped to ~2.5 σ. The clipping is large in absolute terms because the canonical-K coordinate values are large, and large in *q-direction* terms because the canonical basis is aligned with the q-direction by construction.

## The full mechanism

For each method, the per-key error in coordinate `j` of the method's basis can be decomposed into:

```
err_j(k) ≈ quantization_noise_j  +  clipping_bias_j(k_j)
```

- `quantization_noise_j` is approximately zero-mean Gaussian, variance `σ²_j(c) · 2^{-2 b_j}`. This is what `Cov(err)` and `q^T Cov(err) q` capture.
- `clipping_bias_j(k_j)` is *deterministic in `k_j`*: small in the bulk, large and signed-toward-bulk at the tail.

For random keys (whose `k_j` values are typical), `clipping_bias` is small. For top-1 keys with `|z_j|` in the tail, `clipping_bias` is large and *negative-toward-bulk*. After projecting back through `P_K_inv` and inner-producting with `q`, the bias accumulates coherently into a large negative `δ` at the top-1 key — pulling its logit down toward the runner-up.

This is exactly what the empirical signed-δ measurements show:

| Method | mean signed δ at top-1 | mean signed δ at random |
|---|---:|---:|
| `v3` | −0.99 | −0.06 |
| `v_waterfill` | −0.28 | −0.01 |
| `cca_waterfill` | **−3.34** | +0.04 |

The minus signs at top-1 are clipping bias pulling extreme values toward the bulk centroid. CCA's bias is by far the largest because (a) top-1 keys are more extreme in the CCA basis, and (b) the un-whitening through `P_K_inv` amplifies the bias further in directions of large `Σ_K` — which by joint Q-K correlation are exactly the directions q has mass in.

## Why V_waterfill wins on both metrics

V_waterfill's two advantages:

- **Geometry distortion (vs V3):** V is the optimal orthogonal basis for diagonalizing `M_q`, and water-fill is the optimal allocation given Bennett's high-rate model. Both contribute to a 7× reduction in geometry distortion vs V3 at b=3.
- **Top-1 retention (vs CCA):** V's eigenvectors only depend on Q, not on the joint Q-K correlation structure. So V's basis doesn't selectively push top-1 keys into per-coord tails — top-1 and random keys have comparable tail behavior in V's basis. Lloyd-Max clipping bias hits both classes of keys equally, so the *amplification factor at top-1 is small (1.7×)*.

V_waterfill is in the sweet spot: structure-aware enough to allocate bits well (which fixes the geometry distortion problem), but **not** structure-aware enough to expose the top-1-key tail behavior (which is what hurts CCA).

## Takeaways

1. **Q-weighted distortion (geometry) and top-1 retention can disagree because top-1 cares about per-key tails, not population averages.** A basis that puts top-1 keys in its quantizer's bulk is better than a basis that gives lower expected MSE.
2. **The CCA basis is "too good" at aligning with Q-K joint structure.** It exposes top-1 keys preferentially, where scalar quantization is least accurate.
3. **V3's random rotation is "implicit obfuscation"** that hides the top-1 tail structure from the quantizer. This is *good* for top-1 retention even though it gives up most of the bit-allocation gains.
4. **V_waterfill threads the needle:** smart allocation in a Q-aware basis, but not joint-Q-K-aware, so top-1 tails are not exposed.
5. **Operationally, this favors orthogonal Q-only bases for top-1-critical applications.** CCA-style joint-aligned bases would need a quantizer that handles tails better — uniform-step-size quantization in Σ_K-units, or a signal-conditional codebook — to recover the geometry advantage in top-1 terms.

## Reproducibility

Diagnostics for layer 12, kv_head 5 (a typical mid-tier head):

- `experiments/stage1/scripts/diagnose_top1_mechanism.py` — bit allocation, error norms, top-1 flips, error covariance anisotropy.
- `experiments/stage1/scripts/diagnose_top1_error_alignment.py` — Cov(err) shape match to `M_q⁻¹` / `I` / `Σ_K`; per-query noise/signal ratio.
- `experiments/stage1/scripts/diagnose_top1_specificity.py` — `|δ|` and signed `δ` at top-1 vs runner-up vs median vs random key.
- `experiments/stage1/scripts/diagnose_top1_extreme_coords.py` — per-coord z-scores of top-1 vs random keys per basis.

Output figures and `metrics.json` in `artifacts/stage1/cca_vs_waterfill_study/diagnostics_v3_vs_cca/`.

The four scripts produce the chart inputs for this note. Use `for f in experiments/stage1/scripts/diagnose_top1_*.py; do python -u $f; done` to regenerate.

## Limitations

- Single (layer, kv_head) characterization. The numbers here are from layer 12, kv_head 5 with one example (`ex_005.pt`, qasper). The mechanism is structural, but the magnitude of the effect varies per head — heads with steeper canonical-correlation spectra would show even larger CCA top-1 amplification, while flat-spectrum heads would show less.
- Bennett's high-rate model is approximate at b=3. The clipping-bias decomposition above is qualitative; a full theoretical predictor of top-1 retention from `(b, basis, source distribution)` would require modeling the codebook clipping explicitly.
- The bias at top-1 also depends on the specific `q` distribution. We use real prefill-phase queries here. Decode-phase queries (E5) have peakier attention, so the runner-up gap is larger, which is why the effect is partially masked at decode time even though the per-key clipping bias still exists.
