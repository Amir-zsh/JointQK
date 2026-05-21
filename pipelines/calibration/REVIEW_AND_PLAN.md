# Calibration Pipeline — Review and Shared Plan

> Shared workspace between Claude and Codex. Append updates with `## YYYY-MM-DD HH:MM — <author>` headers; don't rewrite each other's history. Keep "Current state" near the top accurate.

## Goal

Systematically determine the **best (basis × allocation) recipe for K and V quantization** under the objective:

1. **Primary:** minimize logit error `E[(qᵀ(k−k̂))²]` (and downstream attention-rank effects).
2. **Secondary:** minimize raw reconstruction MSE (`E[‖k−k̂‖²]`, `E[‖v−v̂‖²]`).

The pipeline must be useful for choosing the deployed method, not just for producing closed-form curves.

## Current state (as of 2026-05-05)

Pipeline lives in `pipelines/calibration/`:

- `capture_raw.py` — fp16 prefill `q_post / k_post / v` per example. The validated smoke capture is 32 rows and is ~146 GB on disk; the full 480-row split will be much larger and should be run with `--artifact-root` pointed at high-capacity storage.
- `compute_stats.py` — per-example second moments (`sq_sum`, `sk_sum`, `cqk_sum`, `sv_sum`, `sum_v`); merges into `02_stats/aggregate.pt`.
- `analyze_bases.py` — sweeps `(basis, allocation, sample_size, regime, bits, layer0)` and writes per-row jsonl + summary.
- `make_charts.py` — only plots `pooled, K=3, layer0=False`.
- `validate_artifacts.py` — shape/existence checks; no math assertions.

**Methods covered:**

| Family | Method | Basis | Allocation |
|---|---|---|---|
| K | `v3` | random Hadamard + per-vector unit-norm | uniform integer bits (TurboQuant baseline) |
| K | `q_only` | eigvecs of Σ_Q | uniform + waterfill (both reported) |
| K | `k_only` | eigvecs of Σ_K | uniform + waterfill (both reported) |
| K | `jointqk` | eigvecs of `(Σ_Q Σ_K + Σ_K Σ_Q)/2` (= R_sym from Stage 1E) | uniform + waterfill (both reported) |
| V | `v_random` | random orthogonal | uniform |
| V | `v_eigen_uniform` | eigvecs of Cov(V) | uniform |
| V | `v_eigen_waterfill` | eigvecs of Cov(V) | waterfill |

CCA-related methods are intentionally out of scope for this calibration — earlier work showed they
don't produce a competitive deployed recipe. The `cqk` cross-moment is no longer surfaced from
`combine_stats` (existing per-example `.pt` files retain `cqk_sum` for backward compatibility but
nothing in the analysis path reads it).

**Headline metrics:**

```
k_mse        = Σ_j k_diag[j] · 2^{-2 b_j}              (Bennett high-rate)
logit_error  = Σ_j q_diag[j] · k_diag[j] · 2^{-2 b_j}  (Bennett, orthogonal-basis only)
v_mse        = Σ_j v_diag[j] · 2^{-2 b_j}              (Bennett)
```

Empirical alternative (`--empirical`) computes real `‖k−k̂‖²` and `mean (qᵀ err)²` via `PerCoordCompressor.roundtrip` on raw tensors. **Gated to smoke runs by default.**

Split: 480 rows = 60 per task × (50 train + 10 test) × 8 tasks (qasper, hotpotqa, musique, qmsum, multi_news, triviaqa, passage_retrieval_en, repobench-p), Qwen3-8B, prompt length 2k–32k.

## Critical issues against the stated goal

### High priority — math/coverage actually misaligned with goal

1. **Logit-error proxy ignores clipping bias.** The closed-form `Σ q·k·2^{−2b}` is Bennett high-rate (zero-mean white noise, uncorrelated with source). At b∈{2,3} the dominant error is signed clipping bias at distribution tails — see `notes/stage1e_cca_vs_waterfill/why_v3_beats_cca_on_top1.md` (mean signed δ at top-1 = −3.34 for CCA-waterfill while Bennett predicts ~0.08). **The headline metric cannot see the effect that decided Stage 1E.**

2. **`cca_orth_waterfill` (V_h) is missing.** Stage 1E's runner-up basis isn't in the sweep. Without it, "best approach" cannot be answered.

3. **Empirical mode is gated to smoke.** The only metric that captures (1) is opt-in and runs on a tiny subset by default. Should be the headline path, not the audit path.

4. **No top-1 / argmax-retention metric.** Logit MSE and top-1 retention can disagree by 15 pp under identical `(method, b)`. Top-1 is the project's de-facto headline. Need to compute it directly: it's one extra line per (layer, head) in the empirical loop (`argmax(q@k.T) == argmax(q@k_hat.T)`).

5. **Calibration's `allocate_bits` ≠ production `water_fill`.** The calibration's bisection + post-hoc cap_bits diverges from `kvq/metric_transform.water_fill`'s iterative-saturation loop and from `build_jointqk_compressor`'s rounding+cap. Calibration winner ≠ deployed winner is possible.

   **Fix:** call `build_jointqk_compressor` (or the same `water_fill`) from the calibration code, not a reimplementation.

### Medium priority — analysis correctness

6. **Headline alias collapses (basis × allocation).** `out["k_mse_k{bits}"] = waterfill_k_mse` hardcodes water-fill, so `analysis_summary.json`'s un-suffixed keys silently pick one allocation. Keep only `_waterfill`/`_uniform` suffixes; remove the alias.

7. **Subspace-overlap reference is hardcoded to jointqk.** Bakes in the answer. More useful: overlap between two random calibration subsets of the **same** basis (sample-size stability), and pairwise overlap across basis families.

8. **`allocate_bits` uses `high` not `mid` after bisection.** Defensible (under-budget then largest-remainder makes it up) but undocumented. Worth a comment + a unit test that `Σ b_j == b·d` after rounding.

9. **`v_rows_for_method` ignores method name for `_random`.** Falls through to the `else: torch.full_like(...)` branch (uniform), which is what was intended for `v_random`, but it's not obvious from the code. A method registry would prevent ambiguity.

10. **Validation gate is shape-only.** No math assertions — Bennett ≈ empirical at high b, budget conservation, basis orthogonality, jointqk equals production R_sym to fp32 precision.

### Low priority — coverage / polish

11. **Charts plot one cell.** `make_charts.py` only renders `pooled, K=3, layer0=False`. Need per-bit-budget panels, regime comparison (same-task / pooled / LOO), analytic-vs-empirical scatter, and subspace-stability-vs-n curves.

12. **V analysis has no attention-output metric.** Pure `‖V−V̂‖²` in eigenbasis. Acceptable given V doesn't enter the logit, but the framing in README should make this explicit.

13. **Raw tensors are very large.** The current validated smoke capture is already ~146 GB for 32 rows, while stats are ~4 GB. Full 480-row raw capture will be much larger. Raw is required for empirical eval, but consider a `--keep-raw` flag or fused capture+stats mode so non-empirical full runs can stream-compute and discard.

14. **No analytic-vs-empirical agreement chart.** Without it, the analytic headline is unfalsifiable.

## What's right and worth keeping

- Stage decomposition (capture → stats → analyze → charts) is clean and resumable.
- Σ_Q convention (per-Q-head outer products, divided by `group·tokens`) matches Stage 1E post-F1.
- V centering (`cov_v = E[vvᵀ] − μμᵀ`, recon around mean) is correct.
- Cross-task / pooled / LOO regime structure is the right generalization-test design.
- Larger split (480 rows × 8 tasks vs Stage 1E's 24 × 3) is a real upgrade.

## Proposed patch set (in priority order)

### P0 — make the headline metric trustworthy

- [x] **P0.1** Add `cca_orth` (V_h) to `K_METHODS`. Routed through production helpers (`compute_cca_basis` + `_derive_vh_rsym`) so calibration's V_h is byte-identical to deployment's. `cqk` now surfaced from `combine_stats`. `cca_waterfill` (non-orthogonal P_K) deferred — Stage 1E's `cca_orth_waterfill` already dominates it.
- [x] **P0.2** `--empirical` flipped to default ON via `argparse.BooleanOptionalAction`; `--no-empirical` opts out. Both `analyze_bases.py` and `launch.py` updated; launcher always forwards an explicit flag. README updated.
- [x] **P0.3** Top-1 **and top-5** retention added to `empirical_k_metrics` (chunked over queries to bound peak memory at ~512 MiB per logit matrix at fp32). New keys: `empirical_top1_waterfill_k{bits}`, `empirical_top5_waterfill_k{bits}` (+ unsuffixed aliases).
- [x] **P0.4** `analyze_bases.allocate_bits` rewritten as a thin wrapper around `metric_transform.water_fill` + `per_coord_quantization.round_bits_to_integer` + cap-and-redistribute loop. Verified byte-identical to `build_jointqk_compressor`'s waterfill path on synthetic input.

### P1 — analysis hygiene

- [ ] **P1.1** Drop the un-suffixed `k_mse_k{bits}` / `logit_error_k{bits}` aliases. Charts and downstream consumers must pick `_waterfill` or `_uniform` explicitly.
- [ ] **P1.2** Subspace overlap: compute (a) within-method stability across paired subsets and (b) cross-method pairwise; report both.
- [ ] **P1.3** Math gates in `validate_artifacts.py`: `|Σb − b·d| ≤ 1`, basis orthogonality (`R Rᵀ = I` to 1e-4), Bennett-vs-empirical agreement at b=6 within 5%, jointqk equals R_sym from `build_jointqk_compressor` to fp32.

### P2 — coverage / polish

- [ ] **P2.1** Charts: per-bit-budget panels (b∈{2,3,4}), regime comparison, analytic-vs-empirical scatter, subspace-stability-vs-n.
- [ ] **P2.2** README clarifies that V optimizes raw recon MSE, not attention-output error.
- [ ] **P2.3** `--keep-raw` flag for capture+stats fused mode that doesn't write raw to disk.

## Open questions

- **Q1.** Should we add `k_uniform` (k_only-eigvecs + uniform allocation) as a method, or keep uniform as a per-method side-metric? Stage 1E showed all uniform-r=64 variants are catastrophic, so probably not worth full method status — but at b∈{2,3,4} flat uniform (no rank cutoff) might survive.
- **Q2.** For the empirical top-1 metric, should we use prefill queries only, decode queries only, or both? Stage 1E E5 showed decode top-1 ≥ prefill top-1; if we want to predict deployment, decode is the truer target but has fewer tokens.
- **Q3.** Is there appetite for adding KIVI's per-channel int4 K-quantization as a method here so calibration can reproduce its operating point? It doesn't fit the basis × allocation framing but is the headline baseline in the paper.

## Decision log

_Append decisions here. Format: `**YYYY-MM-DD — <author>:** <decision> (rationale)`._

## Update log

### 2026-05-05 — Claude (initial review)

Reviewed `capture_raw.py`, `compute_stats.py`, `analyze_bases.py`, `make_charts.py`, `validate_artifacts.py`, `common.py` against the toolkit's `per_coord_quantization.py` and `metric_transform.water_fill`. Wrote up issues 1–14 above. Pipeline is structurally sound but the headline metric (Bennett analytic logit error) cannot detect the clipping-bias mechanism that decided Stage 1E's V3-vs-CCA puzzle. The two highest-leverage changes are P0.2 (empirical-by-default) and P0.4 (single source of truth for water-fill).

No code changes yet; awaiting direction on which P0 items to start with.

### 2026-05-05 — Codex

Reviewed Claude's critique and agree with the main direction: the current pipeline is structurally useful, but the deployed-method decision should not be based on Bennett analytic logit error alone. The P0 items are the right priorities: add empirical top-1/logit metrics as first-class report metrics, add `cca_orth_waterfill`, remove ambiguous unsuffixed K metric aliases, and route allocation/compressor construction through the same production toolkit path used by `JointQKPress` / Stage 1E.

Correction to current-state sizing: the ~146 GB raw artifact size is from the 32-row smoke capture, not the full 480-row split. Full raw Q/K/V capture is expected to be far larger, so full runs should use a large external artifact root or a future fused stats-only mode when empirical raw metrics are not needed.

Implementation note from the chart iteration: explicit uniform-vs-water-fill metrics and charts now exist, but this should be treated as an interim analysis surface until P0.4 is done. The calibration water-fill wrapper still needs to be replaced with the production allocation/compressor path before using these charts to choose the final recipe.

### 2026-05-05 — Claude (P0 patches)

All four P0 items applied to `analyze_bases.py` (+ `launch.py`, `validate_artifacts.py`, `README.md`):

- **P0.4 (allocate_bits)** — `analyze_bases.allocate_bits` is now a thin wrapper around the toolkit's `water_fill` + `round_bits_to_integer` + the MAX_BITS=8 redistribute loop. Removed the old `cap_bits` helper. Cross-checked on a synthetic head: calibration's per-coord allocation matches `build_jointqk_compressor`'s waterfill path **byte-for-byte** (`torch.equal == True`). Sums conserve to `b_avg * d` exactly when caps don't bite.

- **P0.3 (top-1, top-5)** — Added `empirical_top1_waterfill_k{bits}` and `empirical_top5_waterfill_k{bits}` (plus the unsuffixed aliases for parity with the existing K MSE / logit-error keys) to `empirical_k_metrics`. Logits are computed in chunks over queries, capped at ~512 MiB per matrix, so it scales to 32k-token contexts without OOM. Top-5 = "real top-1 contained in approx top-5" (same definition as Stage 1E E3).

- **P0.1 (cca_orth)** — Added `cca_orth_basis()` that calls the production `compute_cca_basis` + `_derive_vh_rsym` directly. `combine_stats` now also returns `cqk` (it was already accumulated, just not returned). `K_METHODS` extended to `("q_only", "k_only", "jointqk", "cca_orth")`. `validate_artifacts.required` updated. Verified on the smoke aggregate: V_h shape `(36, 8, 128, 128)`, orthogonality err 1.7e-4 (well below the 1e-3 threshold), and **byte-identical** to a direct production call.

- **P0.2 (empirical default)** — `--empirical` flipped to default True via `argparse.BooleanOptionalAction` in both `analyze_bases.py` and `launch.py`. `--no-empirical` opts out. Launcher always forwards an explicit flag (no more silent default mismatch). README updated to explain the new default and how to bound cost on full runs (`--empirical-max-eval-examples N`).

**Not done (deferred to P1/P2 per the original plan):**
- Drop the unsuffixed `_k{bits}` aliases (P1.1) — kept for backward compat with current charts; should land alongside chart updates.
- Math gates in `validate_artifacts.py` (P1.3) — basis orthogonality, budget conservation, Bennett-vs-empirical agreement at high b.
- Charts beyond `pooled, K=3, layer0=False` (P2.1).
- `cca_waterfill` (non-orthogonal P_K) — Stage 1E showed `cca_orth_waterfill` strictly dominates it. Adding it would just be for completeness; ask if you want it.

**Suggested next step:** run analyze_bases on the existing 32-row smoke aggregate end-to-end (with `--smoke --empirical`) to confirm the new methods/metrics flow through merge → summary → charts without regressions, then decide whether to extend to the full 480-row capture.

### 2026-05-05 — Claude (scope correction: drop CCA, add v3)

User direction: CCA was an earlier approach that didn't pan out and should be removed from this calibration; instead add a TurboQuant / V3 baseline so we can compare any chosen recipe against the calibration-independent floor.

**Reverted P0.1 (cca_orth):**
- Removed `cca_orth` from `K_METHODS`, deleted `cca_orth_basis()`, removed the `_derive_vh_rsym` / `compute_cca_basis` imports.
- Dropped `cqk` from `combine_stats`'s return (and the in-loop `cqk` accumulator). `compute_stats.py` still writes `cqk_sum` per example, so existing `01_raw/`, `02_stats/` artifacts remain valid — but the analysis path no longer reads it.
- Updated `validate_artifacts.required` to drop `cca_orth`.

**Added v3 / TurboQuant as a K method:**
- `K_METHODS = ("v3", "q_only", "k_only", "jointqk")`.
- New `v3_rows_for_method()` — analytic Bennett with the trace formulas, since V3's noise in original key space is isotropic (random rotation makes it direction-blind):
    - `k_mse = trace(Σ_K) · 2^{-2b}`
    - `logit_error = trace(Σ_Q) · trace(Σ_K) · 2^{-2b} / d`
- New `empirical_v3_metrics()` — uses production `Stage1MSECompressor(head_dim, bits, seed)` (same as the production V3 evaluator) and emits the same four empirical keys (`k_mse`, `logit_error`, `top1`, `top5`) the basis methods now emit.
- Main loop branches on `method == "v3"` — V3 is treated specially because it has no calibrated basis and no allocation choice, so the `(uniform | waterfill)` × basis cross-product doesn't apply. V3 only emits the unsuffixed `_k{bits}` keys.
- V3's metrics are calibration-independent by construction (no dependence on the train subset). On a sample-efficiency curve V3 is intentionally a flat horizontal line — that's the floor every other method must beat.

**Verification on smoke aggregate (32 rows, 4 train / 2 eval examples, b=3, 1 example × 2 layers × 1 head × 256 tokens for empirical):**
- Analytic V3: k_mse @ b=2/3/4 = 41.5 / 10.4 / 2.6 (4× shrink per bit, matches `2^{-2}` Bennett scaling). logit_error/k_mse ratio constant across b (= trace(Σ_Q)/d).
- Empirical V3 @ b=3: k_mse=0.31, logit_error=90.8, top1=0.44, top5=0.86. Ordering of empirical/analytic is consistent with the Lloyd-Max constant `c ≈ π√3/2` that the analytic formula drops.

**Files touched:** `analyze_bases.py`, `validate_artifacts.py`, `REVIEW_AND_PLAN.md`. README still accurate; no changes needed there for the v3 swap.

**Not done:** P1 / P2 items still pending. The chart code (`make_charts.py`) currently emits `bar_latest` from the headline `data` dict which iterates whatever methods appear in the summary — V3 will surface in those bar charts automatically once a full analysis run lands.

### 2026-05-05 — Claude (fused capture+stats, --keep-raw, progress logging)

User direction: cut IO by computing stats inline during capture (no need to dump 50–400 MB of raw fp16 per example to disk just to reread it for moments). Add tri-state `--keep-raw {none, test, all}` defaulting to `test` so empirical eval keeps working out of the box but train examples don't waste disk. Add structured progress logging with ETA so multi-shard runs are easy to follow.

**Architecture change.** Capture now writes per-example **stats** to `02_stats/shard_NNN/<id>.pt` always, and per-example **raw** to `01_raw/shard_NNN/<id>.pt` only when `should_keep_raw(row, --keep-raw)` is true. The standalone `compute_stats.py` per-shard pass is gone from the default flow; `compute_stats.py` now defaults to merge-only and exposes `--rebuild-from-raw` for the legacy path (useful only after changing the moments accumulator or to repair pre-fusion artifacts).

**Files modified:**

- `common.py`:
  - `KEEP_RAW_CHOICES = ("none", "test", "all")`, `DEFAULT_KEEP_RAW = "test"`, `should_keep_raw(row, mode)` predicate, `add_keep_raw_arg(parser)` helper.
  - New `ProgressTracker` class: rolling-window avg duration, ETA, throughput (examples/min), per-kind byte counters (raw/stats), structured per-example log lines, and a `progress.json` snapshot updated after every example for cross-shard monitoring.

- `compute_stats.py`:
  - Extracted `compute_moments_from_tensors(q_post, k_post, v, prompt_length, device)` — pure helper, no I/O. The legacy `compute_example_stats` is now a thin wrapper that loads raw and calls it.
  - New `build_stats_artifact(moments, row, prompt_sha256)` so capture and the rebuild path produce identical on-disk artifacts.
  - `main()` defaults to merge-only; `--rebuild-from-raw` re-enables per-example computation. `--merge-only` retained as a synonym for back-compat.

- `capture_raw.py`:
  - Added `add_keep_raw_arg`. Computes moments inline after the model forward via `compute_moments_from_tensors`. Always writes `02_stats/shard_NNN/<id>.pt`; writes `01_raw/shard_NNN/<id>.pt` only when `should_keep_raw(row, args.keep_raw)`.
  - Resume logic now requires both stats valid AND raw valid (only when raw is supposed to exist). Two new validators (`validate_existing_stats`, `validate_existing_raw`) split the old `validate_existing`.
  - `ProgressTracker` integration produces per-example log lines like `[capture shard 2/4 | 11/80 (14%)] done in 12.6s | avg 8.3s | ETA 9m32s | rate 7.1/min | wrote raw=140MB stats=1MB | total raw=380MB stats=11MB`.
  - Writes a stage manifest into both `01_raw/shard_NNN/manifest.json` and `02_stats/shard_NNN/manifest.json` (the latter records the inline computation).
  - Storage preflight only counts raw bytes for examples that will actually retain raw.

- `launch.py`:
  - New `--keep-raw {none, test, all}` (default `test`), forwarded to capture command lines.
  - Stats stage no longer parallelizes per-shard work — runs `compute_stats.py` once for the merge then validates. The launcher comment explains where to invoke `--rebuild-from-raw` if needed.

- `validate_artifacts.py`:
  - New `--keep-raw` arg; `validate_raw_stage` only requires raw files for examples whose split matches the policy (logs how many were skipped).

- `README.md`: updated stage descriptions, added a `--keep-raw` modes table, made the new disk profile explicit (stats ~16 MB/example, raw 50–400 MB/example).

**Verified:**
- All five modules import cleanly.
- `compute_moments_from_tensors` produces **byte-identical** sums versus the disk-based `compute_example_stats` (max abs diff 0.0 across `sq_sum, sk_sum, cqk_sum, sv_sum, sum_v` on synthetic data with the right GQA shape).
- `should_keep_raw` correctly returns True for test/default, False for train/default, True for train/all, False for test/none.
- `ProgressTracker` end-to-end: starts/finishes examples, handles `mark_skipped` for resume, writes progress.json with full snapshot fields (`avg_seconds_per_example, bytes_by_kind, eta_pretty, rate_per_minute`, etc.), prints structured log lines.
- `--keep-raw` parses on capture, launch, and validate.

**Disk impact (full 480-row split, default `--keep-raw test`):**
- Old: 480 raw files × ~50–400 MB ≈ 25–200 GB raw + 480 × 16 MB ≈ 8 GB stats.
- New: 80 raw files (test only) ≈ 4–32 GB + 8 GB stats. Roughly **5–10× less disk and IO**, more if the per-example tokens skew long.

**Not yet done:**
- End-to-end smoke run to confirm a real capture writes stats + (test-only) raw and the analysis stage still finds what it needs (the import-level checks pass; a small `--smoke --stage all` run is the obvious next verification but requires loading the model).
- Aggregator across shards (`progress.json` rollup) — for now users can `cat artifacts/.../01_raw/shard_*/progress.json | jq` for a snapshot. A small `progress_summary.py` would be useful follow-up.
- If a user runs `analyze --empirical` on a run captured with `--keep-raw none`, the empirical loop will fail when loading raw. The error today is a `torch.load` FileNotFoundError; could be improved with a clearer message (suggest `--no-empirical` or rerun capture with `--keep-raw test|all`).

### 2026-05-05 — Claude (efficiency review + applied fixes)

Pass over the analysis hot path. Top inefficiencies found and resolved:

1. **`combine_stats` reloaded same `.pt` files across thousands of trials.** Added `StatsCache` (LRU dict in `common.py`) that caches per-example stats payloads, dropping `cqk_sum` to save ~25% per entry. `combine_stats` now takes a `cache=` parameter; `analyze_bases.main` instantiates one per shard. New `--stats-cache-entries` arg bounds RAM (default 0 = unbounded). Hit-rate is logged at end-of-shard and recorded in the manifest. **Expected win:** orders of magnitude on analytic-only runs (which were previously disk-bound by repeated stats loads).

2. **`empirical_k_metrics` and `empirical_v_metrics` rebuilt `PerCoordCompressor` per (idx, layer, head).** The compressor only depends on `(bits, layer, head)` — not on the eval example. Hoisted construction out of the idx loop: now built once per `(bits, layer, head)` per call. Layer/head ranges derived from `train_stats.sigma_q.shape` instead of the first raw payload. **Expected win:** ~`len(eval_subset)` × fewer Lloyd-Max codebook solves (so 80× on the default pooled eval).

3. **`raw_cache = {} if smoke and compute_empirical else None`.** Outside smoke runs, K and V empirical passes within the same trial each reloaded raw from disk. Changed gating to `{} if compute_empirical else None` — within-trial K and V now share one raw load per (trial, idx). **Expected win:** ~2× fewer raw loads per trial (V stage no longer reloads what K just loaded).

4. **`torch.trace(qq @ ee)` → `(qq * ee).sum()`.** Both `qq` and `ee` are symmetric (X^T X), so the trace identity holds. Saves the d×d matmul (16K elements) per (layer, head, idx). Verified equivalence to fp32 precision (1.18e-7 relative error on a random seed). Two callsites swapped via `sed`.

**Not implemented (deferred — bigger refactors):**

5. **Batch V3 over (layer, head)**: `Stage1MSECompressor.roundtrip` expects `(B, H, S, D)` and flattens internally — passing the full `(n_layers * n_kv_heads, T, d)` slab in one call would replace 280 small kernel launches with one. Estimated 5–10× faster on GPU for the V3 empirical pass. Doesn't affect basis methods (each has per-(layer, head) compressors).

6. **Per-(trial, idx) outer loop for empirical**: currently each (method, bits) call iterates over eval examples internally, so a single trial with 4 K methods × 3 bits would load each raw 12× absent the raw_cache fix. With the raw_cache fix this drops to 1× per trial, but the loop *structure* still has the bits→idx ordering which is suboptimal for compressor reuse and prefetch. A full restructure (one pass per `(trial, idx)`, all methods × bits inside) would be cleaner but invasive.

7. **Combine `include_layer0 ∈ (True, False)` into one pass**: today the empirical loop runs twice with different layer ranges. Could run once with full layer range and aggregate two ways. Halves the empirical wall time. Also touches multiple call sites.

8. **`compute_moments_from_tensors`**: per-layer `.to(device).float()` could promote the whole `(n_layers, n_q_heads, T, d)` tensor in one transfer if memory allows (~10 GB at fp16 for T=32k × 36 layers — fits on A100 80GB). Not a hot path though (capture is dominated by model forward).

9. **Cross-trial raw cache (LRU by idx)**: `raw_cache` is per-trial and discarded at trial end. Many trials share the same `eval_subset`; an across-trial cache bounded by `--empirical-raw-cache-mb` would eliminate redundant raw loads across the analysis run. Bigger memory implications (raw is 50–400 MB/example).

**Files touched:** `common.py` (StatsCache class), `analyze_bases.py` (compressor hoist in K and V empirical paths, raw_cache gating, trace→elementwise, --stats-cache-entries arg, hit-rate logging).

**Verified:**
- All modules import.
- StatsCache instantiates and exposes hit/miss counters.
- `(qq * ee).sum()` matches `torch.trace(qq @ ee)` to 1.18e-7 relative error on symmetric inputs.
- empirical_k_metrics and empirical_v_metrics still expose the same metric keys; `comps_by_bits` precompute is gated on `bits ∈ k_bits` / `v_bits`.

**Combined expected impact (on the default 480-row split, full pooled eval, b∈{2,3,4}, 4 K methods × 3 V methods × ~1000 trials):**
- Stats cache: ~10–100× less disk I/O on analytic.
- Compressor hoist: ~80× fewer Lloyd-Max codebook solves in empirical.
- Within-trial raw cache: ~2× fewer raw loads.
- (qq * ee).sum(): trivial per-call but adds up across thousands of (layer, head, idx) iterations.

Together these should drop a full empirical sweep from "compute-bound for hours" to "I/O-bound for one pass through raw, then compute-bound at GPU speed". The bigger wins (5, 6, 9) are still on the table if the user wants further speedup.
