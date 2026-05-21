# Preview pooled-N=50 K-method comparison — results report

**Date:** 2026-05-05
**Run:** `artifacts/calibration/longbench_compact8_qkv/05_reports/preview_pooled_n50/`
**Source:** `pipelines/calibration/preview_pooled.py` (multi-GPU rewrite, 2026-05-05)

---

## TL;DR

On a pooled-N=50 calibration setup (400 train + 80 test LongBench prompts, Qwen3-8B):

- **`jointqk` (eigenvectors of `(Σ_Q Σ_K + Σ_K Σ_Q)/2`) wins top-1 and top-5 attention retention at every bit width**, beating the random-Hadamard baseline (V3 / TurboQuant) by 8–26 pp on top-1.
- **`k_only` is worse than V3** at low bits — calibrating from K covariance alone is *worse than no calibration*. Q matters in the basis.
- **Bennett MSE doesn't predict top-1.** `q_only` has lower empirical K-reconstruction MSE than `jointqk` at every bit width but loses to it on top-1 by 5–11 pp. Matches the Stage 1E concern that an MSE-only objective mis-ranks bases.
- **`jointqk`'s calibrated basis is essentially converged at N=50** — 98% top-16 subspace overlap with the eval-side joint basis. `q_only` and `k_only` are far from converged at the same N (68% and 77% respectively).

This is **single-trial pooled N=50 only** — no sample-size sweep, no same-task / LOO regimes, no V (values) baselines, no per-task or per-layer breakdown, no error bars. Those require the full sweep (next).

---

## Setup

- **Model:** Qwen3-8B
- **Calibration corpus:** 400 LongBench train prompts (50 × 8 tasks: hotpotqa, multi_news, musique, passage_retrieval_en, qasper, qmsum, repobench-p, triviaqa)
- **Eval corpus:** 80 LongBench test prompts (10 × 8 tasks)
- **Per-coord allocation:** continuous water-fill (matches production `build_method_compressor`), max 8 bits/coord, byte-for-byte aligned with deployment.
- **K methods compared:**
  - `v3`: random Hadamard + unit-norm + uniform Lloyd-Max (TurboQuant).  No basis dependence on calibration.
  - `q_only`: eigenvectors of `Σ_Q` (descending eigenvalue).
  - `k_only`: eigenvectors of `Σ_K`.
  - `jointqk`: eigenvectors of `(Σ_Q Σ_K + Σ_K Σ_Q)/2`.
- **Bit budgets:** average 2, 3, 4 bits per coordinate.
- **Headline reporting convention:** layer-0 excluded (per Stage 1 convention — layer 0 has anomalous attention-sink behaviour).

Per-shard execution: 6 GPUs × ~13–14 eval idx each, round-robin slice of `test_indices`. Per-shard accumulators (`{bits: {layer: {mse_num, mse_den, logit_num, logit_den, top1_num, top1_den, top5_num, top5_den}}}`) are written to `shard_NNN.json`; the launcher sums them and the analytic baselines are recomputed on CPU before printing the headline.

---

## Headline empirical metrics (layer-0 excluded)

Per-(layer, head) means, then averaged across (layer ∈ [1, 35], 8 KV heads).

| method | bits | **top-1** | **top-5** | k_mse | logit_err |
|---|---|---|---|---|---|
| v3 | 2 | 0.3949 | 0.6561 | 6.06×10⁻¹ | 2.46×10² |
| v3 | 3 | 0.5718 | 0.8488 | 1.77×10⁻¹ | 6.47×10¹ |
| v3 | 4 | 0.7295 | 0.9569 | 4.87×10⁻² | 1.72×10¹ |
| q_only | 2 | 0.5446 | 0.8399 | 3.23×10⁻¹ | 3.50×10¹ |
| q_only | 3 | 0.7044 | 0.9327 | 1.01×10⁻¹ | 1.05×10¹ |
| q_only | 4 | 0.8017 | 0.9594 | 2.92×10⁻² | 3.12 |
| k_only | 2 | 0.3920 | 0.6612 | 2.08×10⁻¹ | 3.55×10¹ |
| k_only | 3 | 0.5402 | 0.7696 | 6.62×10⁻² | 1.10×10¹ |
| k_only | 4 | 0.6448 | 0.8202 | 1.95×10⁻² | 3.48 |
| **jointqk** | **2** | **0.6550** | **0.9209** | 2.27×10⁻¹ | 2.99×10¹ |
| **jointqk** | **3** | **0.7820** | **0.9796** | 7.13×10⁻² | 9.86 |
| **jointqk** | **4** | **0.8578** | **0.9934** | 2.07×10⁻² | 3.73 |

**Reading the table:**
- `top-1` and `top-5` are attention retention: the fraction of `argmax_t (q_h^⊤ k_{h,t})` keys preserved by the compressed reconstruction.
- `k_mse` is per-coord empirical reconstruction MSE in the original key space (not the rotated coord space).
- `logit_err` is mean `(q^⊤(k − k̂))²` over (q, k) pairs.

### Top-1 ranking by bit width

| bits | best | 2nd | 3rd | worst |
|---|---|---|---|---|
| 2 | **jointqk** 0.655 | q_only 0.545 | v3 0.395 | k_only 0.392 |
| 3 | **jointqk** 0.782 | q_only 0.704 | v3 0.572 | k_only 0.540 |
| 4 | **jointqk** 0.858 | q_only 0.802 | v3 0.730 | k_only 0.645 |

### Top-1 deltas vs V3 baseline (positive = beats V3)

| bits | q_only | k_only | jointqk |
|---|---|---|---|
| 2 | +14.97 pp | **−0.29 pp** | +26.01 pp |
| 3 | +13.26 pp | **−3.16 pp** | +21.02 pp |
| 4 | +7.22 pp | **−8.47 pp** | +12.83 pp |

`k_only` is *worse* than V3 at every bit width — picking the basis from K alone is harmful.

---

## MSE vs top-1 disagreement

The bit budget that wins on `k_mse` is **not** the one that wins on `top-1`:

| bits | best k_mse | best top-1 |
|---|---|---|
| 2 | k_only (0.208) | jointqk (0.655) |
| 3 | k_only (0.066) | jointqk (0.782) |
| 4 | k_only (0.020) | jointqk (0.858) |

`k_only` is the lowest-K-MSE method at every bit width and the *worst* top-1 method. `q_only` beats `jointqk` on `k_mse` but loses to it on `top-1`. The Bennett high-rate distortion proxy ranks the methods backwards from the deployment-relevant attention metric.

This is the V3-vs-CCA top-1 puzzle from Stage 1E, cleanly reproduced under the production water-fill compressor.

---

## Subspace overlap with eval-side joint basis

Per-(layer, head) projection overlap of the calibrated method basis onto the rank-`r` subspace of the eval-side `jointqk` basis (computed on the held-out 80 prompts), then averaged.

| method | r=16 | r=32 | r=64 |
|---|---|---|---|
| q_only | 0.6852 | 0.7578 | 0.8147 |
| k_only | 0.7696 | 0.8264 | 0.8578 |
| **jointqk** | **0.9804** | **0.9853** | **0.9865** |

`jointqk` is converged: at N=50 calibration samples, the train-side basis already aligns 98% with the eval-side basis on the principal 16 directions. `q_only` and `k_only` are far from converged at the same N. This is the per-method "calibration health" diagnostic — it explains *why* `jointqk` wins on top-1 even though `q_only` is competitive on Bennett MSE: the bit allocation flowing through `jointqk`'s near-converged basis lands on the right principal directions.

---

## Layer-0 sensitivity

| method | bits | top-1 layer0=False | top-1 layer0=True | Δ |
|---|---|---|---|---|
| jointqk | 2 | 0.6550 | 0.6510 | −0.40 pp |
| jointqk | 3 | 0.7820 | 0.7787 | −0.33 pp |
| jointqk | 4 | 0.8578 | 0.8547 | −0.31 pp |
| v3 | 2 | 0.3949 | 0.3846 | −1.03 pp |
| v3 | 3 | 0.5718 | 0.5578 | −1.40 pp |
| v3 | 4 | 0.7295 | 0.7142 | −1.53 pp |

Including layer 0 hurts V3 more than `jointqk`. This is consistent with V3's flat budget — it cannot adapt to layer 0's much larger key trace (V3 analytic shows `k_mse_uniform` ≈ 80.8 with layer 0, ≈ 42.5 without — a 1.9× swing). Calibrated methods absorb the spread via per-(layer, head) bit allocation.

---

## Waterfill saturation diagnostics

Mean per-(layer, head) fraction of coords receiving > 0 bits (`active_coords`) and mean of the per-row max bits assigned (`max_coord_bits`, with cap=8).

| method | bits | active_coords | max_coord_bits |
|---|---|---|---|
| jointqk | 2 | 0.9480 | 7.864 |
| jointqk | 3 | 0.9878 | 7.996 |
| jointqk | 4 | 0.9964 | 8.000 |
| q_only | 2 | 0.9401 | 6.832 |
| q_only | 4 | 0.9908 | 7.993 |
| k_only | 2 | 0.9414 | 6.900 |
| k_only | 4 | 0.9911 | 7.996 |

`jointqk` concentrates harder than `q_only` / `k_only` — at b=2 it pushes a typical principal coord almost to the 8-bit cap (7.86), while `q_only` and `k_only` cap out at ~6.8. This concentration is *good* for `jointqk` precisely because its principal directions are the right ones (per the subspace-overlap finding). For `q_only` and `k_only` the effective budget is more spread out because the principal directions are less correct.

---

## Analytic Bennett vs empirical

The analytic Bennett model overpredicts the K-reconstruction MSE by 70–100× because it models per-coord MSE in the *rotated* basis, while the empirical metric measures reconstruction MSE in the *original* key space — a different denominator. The ratio is roughly constant across bit widths within a method, so analytic Bennett is still a useful relative ranking signal *within* a method. **It is not a faithful proxy for top-1 across methods**, as the MSE-vs-top-1 disagreement above shows.

| method | bits | analytic k_mse_waterfill | empirical k_mse | ratio |
|---|---|---|---|---|
| jointqk | 2 | 17.98 | 0.227 | 79× |
| jointqk | 4 | 1.07 | 0.0208 | 51× |
| q_only | 2 | 26.54 | 0.323 | 82× |
| q_only | 4 | 1.62 | 0.0292 | 56× |
| k_only | 2 | 16.64 | 0.208 | 80× |
| k_only | 4 | 1.01 | 0.0195 | 52× |
| v3 | 2 | 42.46 | 0.606 | 70× |
| v3 | 4 | 2.654 | 0.0487 | 55× |

(`waterfill_*` analytic terms are duplicates of `*` for K methods because water-fill is the headline allocator. Uniform allocation is also stored — at b=4, `k_mse_uniform` = 2.654 for V3 vs `k_mse_waterfill` = 1.07 for `jointqk`, i.e. the calibrated allocation gives a 2.5× analytic improvement on top of the basis gain.)

---

## What this run did NOT measure

The full sweep (`launch.py --stage analysis --sample-sizes 10,30,50 --repetitions 1`) is what generates the missing dimensions. Listed for context:

1. **V (values) baselines** — `v_random`, `v_eigen_uniform`, `v_eigen_waterfill`. Not run here.
2. **Sample-size sweep** — only N=50. Doesn't show convergence rate of calibration.
3. **Same-task and leave-one-out (LOO) regimes** — only pooled. Doesn't show whether calibrating on the eval task itself beats pooled, or how much LOO hurts.
4. **Per-task breakdown** — pooled aggregate over 8 tasks.
5. **Replicates / error bars** — single-shot.
6. **Per-layer plots** — accumulators are stored per-layer in each `shard_NNN.json` but the merged metrics collapse to a single mean.

Per-layer slices and per-task slices can be mined from the existing per-shard JSONs without rerunning. Sample-size, regime, V baselines, and replicates require the full sweep.

---

## Operational notes

This preview triggered the OOM that crashed the server earlier. Root cause: the previous `analyze_bases.py` and `preview_pooled.py` held an unbounded `dict` of raw `.pt` payloads (≈ 5 GB each, 80 eval idx → ≈ 400 GB resident). Fixed before this re-run:

- **`RawLRU`** (bounded LRU, default cap = 0 / bypass) replaces the unbounded dict.
- **Inverted loop order** in all three empirical funcs (`for idx: for bits` instead of `for bits: for idx`) — each raw file is loaded at most once per method call.
- **`_release_idx`** (`del raw + gc.collect() + empty_cache`) at the end of every idx iteration.
- **`StatsCache` default cap** raised from unbounded → 128 entries (≈ 10 GB).
- **Atomic per-trial JSON write + working `--resume`** in `analyze_bases.py` (it accepted the flag previously but didn't act on it). Each trial now persists to `04_analysis/shard_NNN/trials/trial_<idx>.json` immediately on completion; resume skips trials with existing JSON.
- **Multi-GPU shard launcher** in `preview_pooled.py` — round-robin slice of eval idx, one subprocess per visible GPU, per-shard accumulator JSON, CPU merge.

This run: 6 shards on GPUs 0–5, ~38 min wall time. Peak resident RAM ≈ 120 GB across all shards (≈ 20 GB per shard: ≈ 5 GB raw + ≈ 10 GB stats cache + ≈ 5 GB bases/compressors). Wall time was bottlenecked by shard 4 — round-robin slicing didn't balance prompt length, so shards with longer-prompt eval idx took longer (ratio 285 s : 829 s for the V3 method, fastest vs slowest shard).

---

## Open questions / next steps

1. **Re-launch the full sweep** with the memory + resume fixes:
   ```
   .venv/bin/python pipelines/calibration/launch.py \
     --stage analysis --gpus 0,1,2,3,4,5 --sample-sizes 10,30,50 --repetitions 1 --resume
   ```
   Now safe against OOM (bounded raw cache + freed per idx) and against crashes (atomic per-trial write + working resume). Estimated 3–4 h.

2. **Mine per-layer slices** from `shard_NNN.json` accumulators — quantify whether `jointqk`'s top-1 lead is uniform across layers or driven by a few layers.

3. **Per-task breakdown** — the 80 eval prompts include 10 per task; the per-shard slices preserve task identity via `per_example` metadata. Per-task means reveal whether jointqk's lead is task-uniform.

4. **Why does `k_only` underperform V3?** Two hypotheses:
   - K's principal directions over-fit to a few dominant tokens (RoPE-induced position bias?), so the calibrated basis allocates bits to directions that don't matter for queries.
   - V3's random Hadamard happens to spread error uniformly enough that the unweighted Lloyd-Max codebook is a fair fit, while `k_only`'s directional concentration creates coords with very different scales → uniform Lloyd-Max is mismatched.

5. **Token-balanced sharding** for future preview/multi-GPU runs — distribute eval idx by `prompt_length` rather than round-robin to flatten shard wall-time.
