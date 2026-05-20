# Stage 1 Final Calibration Pipeline

This folder contains the final calibration pipeline for LongBench Q/K/V basis analysis. The pipeline is split into artifact-producing stages so each step can be validated and rerun independently.

## Stages

1. `capture_raw.py` (fused capture + per-example stats)
   - Reads the fixed LongBench calibration split.
   - Runs the model prefill once per example, captures fp16 `q_post / k_post / v` in memory.
   - Computes per-(layer, kv_head) second-moment sums (`sq_sum`, `sk_sum`, `cqk_sum`, `sv_sum`, `sum_q`, `sum_k`, `sum_v`) inline while the tensors are still hot.
   - Always writes per-example stats to `02_stats/shard_NNN/<id>.pt`.
   - Writes raw fp16 tensors to `01_raw/shard_NNN/<id>.pt` only when `should_keep_raw(row, --keep-raw)` is true (see `--keep-raw` below).
   - Emits structured per-example log lines (shard idx, example idx, tokens, duration, rolling avg, ETA, throughput, MB written) and a per-shard `progress.json` updated after every example for external monitoring.

2. `compute_stats.py` (merge by default)
   - Default: scans all per-shard stats files and merges into `02_stats/aggregate.pt`. No per-example computation — that already happened during capture.
   - `--rebuild-from-raw`: legacy path that reloads raw tensors and recomputes stats per example. Only needed for repairing artifacts captured before the fused pipeline.

3. `analyze_bases.py`
   - K bases: `v3` (random Hadamard / TurboQuant baseline), `q_only` (eigvecs Σ_Q), `k_only` (eigvecs Σ_K), `jointqk` (eigvecs of `(Σ_Q Σ_K + Σ_K Σ_Q)/2`).
   - V bases: `v_random`, `v_eigen_uniform`, `v_eigen_waterfill`.
   - Allocations: water-fill (via the production `metric_transform.water_fill`) and uniform.
   - Regimes: same-task, pooled stratified, leave-one-out.
   - Metrics: analytic Bennett (`k_mse`, `logit_error`, `subspace_overlap`) and **empirical** raw-tensor metrics (`empirical_k_mse`, `empirical_logit_error`, `empirical_top1`, `empirical_top5`) on the test subset of each eval. Empirical is on by default.
   - **Atomic per-trial outputs:** each completed trial writes `04_analysis/shard_NNN/trials/trial_<global_idx>.json` via tmp+rename, then is appended to `shard_NNN/rows.jsonl` at end of shard. Mid-shard crashes lose **at most one trial**.
   - **Resume:** `--resume` skips any trial whose JSON already exists; rows from the cached JSON are spliced into the current shard's `rows.jsonl` at the end.
   - **Memory bounds:** `--stats-cache-entries` (default 128 ≈ ~10 GB) caps the per-shard stats LRU; `--raw-cache-entries` (default 0 = bypass) caps the per-trial raw LRU. The previous unbounded raw cache held all eval idx (~5 GB each → ~400 GB on pooled trials) and crashed the server; the inverted `for idx: for bits` loop order in the empirical funcs ensures each raw file is loaded **at most once per method call** even with the cache disabled.

4. `preview_pooled.py` (multi-GPU preview)
   - Standalone driver for the pooled-N=50 K-method comparison without waiting on the full sweep.
   - **Launcher mode (default):** spawns one shard subprocess per visible GPU. Each shard processes a round-robin slice of the test idx and writes per-shard accumulator JSON. Launcher merges shards on CPU, recomputes analytic baselines, prints the V3-vs-basis headline table, writes `merged.json`.
   - **Shard mode (`--num-shards N --shard-id i`):** internal — single-process worker invoked by the launcher.
   - Output: `05_reports/preview_pooled_n50/` — `shard_NNN.json` (per-shard accumulators), `shard_NNN.log` (per-shard stdout), `merged.json` (final per-method × layer0-mode metrics), `launcher.log`.
   - Memory bounded: `StatsCache` cap=128, `RawLRU` cap=1 per shard. Peak resident ≈ 20 GB per shard.

5. `make_charts.py`
   - Reads merged analysis artifacts.
   - Writes summary charts and a report README under `05_reports/`.

6. `validate_artifacts.py`
   - Validates split, raw, stats, analysis, and chart artifacts.
   - Raw-stage validation honors `--keep-raw`: only requires raw files for examples whose split matches the policy.

## `--keep-raw` modes

Capture's raw fp16 q/k/v tensors are by far the largest disk consumer (1–10 GB per example, depending on prompt length). Stats are tiny (~75 MB per example). `--keep-raw` controls retention:

| Mode | Raw kept for | Disk on full split | Empirical analyze works? |
|---|---|---|---|
| `none` | nothing | stats only (~40 GB) | no — analyze --empirical errors on missing raw |
| `test` (**default**) | only `split == "test"` examples (10 per task × 8 tasks = 80) | stats + ~500 GB raw | yes — empirical eval only reads test |
| `all` | every example (480) | full raw on disk (~3 TB) | yes |

Train examples contribute only to second-moment accumulation, so retaining their raw is wasteful unless you need to rerun stats with a changed accumulator (in which case use `--keep-raw all` once, then drop back to `test`).

## Launching

Use the project's `.venv` (uv-managed Python 3.12) and the multi-GPU launcher.

### Full sweep

```bash
.venv/bin/python experiments/calibration/launch.py \
  --stage all \
  --gpus 0,1,2,3 \
  --jobs-per-gpu 1 \
  --resume
```

`--empirical` is **on by default** (computes raw `‖k − k̂‖²`, mean `(qᵀ err)²`, and top-1 / top-5
attention-rank retention on held-out examples). The analytic Bennett proxy alone cannot detect
clipping bias and disagrees with top-1 ranking — see `REVIEW_AND_PLAN.md` and the preview report
at `notes/preview_pooled_n50_report.md`.

For analytic-only runs (no raw-tensor passes, much cheaper), pass `--no-empirical`. To bound the
cost of the empirical pass on full sweeps, also pass e.g. `--empirical-max-eval-examples 4` (default
0 = use the entire held-out test set per trial).

For a slimmed sweep (3 sample sizes, single rep per cell — preliminary signal in ~3–4 h on 6 GPUs):

```bash
.venv/bin/python experiments/calibration/launch.py \
  --stage analysis \
  --gpus 0,1,2,3 \
  --sample-sizes 10,30,50 \
  --repetitions 1 \
  --resume
```

Resume is safe: re-running with `--resume` after any interruption skips trials whose
`04_analysis/shard_NNN/trials/trial_<idx>.json` already exists.

### Multi-GPU preview (pooled N=50 only)

For a fast headline V3-vs-basis comparison without running the full grid:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  .venv/bin/python experiments/calibration/preview_pooled.py \
  --gpus 0,1,2,3,4,5
```

Spawns one shard per GPU; ~30–40 min wall time on 6 GPUs (shard-imbalance bound — round-robin slicing doesn't balance prompt length). Headline lands in `launcher.log` and `merged.json` under `05_reports/preview_pooled_n50/`.

### Smoke run

```bash
.venv/bin/python experiments/calibration/launch.py \
  --stage all \
  --smoke \
  --gpus 0,1 \
  --jobs-per-gpu 1 \
  --resume
```

Smoke empirical metrics are intentionally bounded: one held-out raw example, two layers, one KV head, 256 tokens, 3-bit K/V, and one empirical trial per shard. Full runs keep the configured bit widths and per-coordinate cap unless overridden.

## Default Artifacts

- Split manifest: `artifacts/calibration_splits/longbench_compact8_60_seed20260504_2k32k/manifest.json`
- Run root: `artifacts/calibration/longbench_compact8_qkv/`
- Raw tensors: `01_raw/shard_NNN/`
- Stats: `02_stats/shard_NNN/` and `aggregate.pt`
- Analysis: `04_analysis/shard_NNN/{trials/trial_*.json, rows.jsonl, manifest.json}` and merged `analysis_rows.jsonl` / `analysis_summary.json`
- Reports: `05_reports/` (charts + `preview_pooled_n50/` from preview)

The full raw Q/K/V capture is large (~500 GB at `--keep-raw test`, ~3 TB at `--keep-raw all`). Use `--artifact-root` to point full runs at storage with enough free space.

## Memory and resume safety (2026-05-05 fixes)

A multi-trial pooled-eval analysis run on 2026-05-05 OOM-crashed the server. Root cause: the per-trial `raw_cache` was an **unbounded `dict`** that accumulated every loaded raw payload (~5 GB each × 80 eval idx = ~400 GB resident on pooled trials). Fixed in `analyze_bases.py` and `preview_pooled.py`:

- **`RawLRU`** (bounded LRU, default `max_entries=0` = bypass) replaces the unbounded dict.
- **`for idx: for bits`** loop order in `empirical_v3_metrics`, `empirical_k_metrics`, `empirical_v_metrics` — each raw file loaded at most once per call (was previously once per `(bits, idx)` with unbounded caching).
- **`_release_idx`** at the bottom of each idx loop: explicit `del raw + gc.collect() + torch.cuda.empty_cache()`.
- **`StatsCache` default cap** raised from unbounded → 128 entries (~10 GB per shard).
- **Atomic per-trial JSON** written via tmp+rename after each completed trial, plus working `--resume` (analyze_bases previously accepted the flag but ignored it).
- **`return_accumulators=True`** mode on the empirical funcs — returns raw per-(bits, layer) accumulators for cross-shard merging in the multi-GPU preview.

See `notes/preview_pooled_n50_report.md` for the first results from the post-fix preview.
