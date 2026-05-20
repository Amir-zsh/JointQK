# Phase 7 v7 on Llama-3.1-8B — runbook for the remote agent

This document is **self-contained** for an agent running the v7 sweep on Llama-3.1-8B-Instruct on a remote server. It assumes you already know the v7 design (otherwise read `notes/jointqk_disconnect_investigation.md` first); here we walk through commands.

The pipeline has three stages:

1. **Capture** — run model prefill on a fixed 480-prompt LongBench split, save fp16 q/k/v for the test-half (~80 prompts) and per-example second-moment stats for all 480.
2. **Build calibration artifacts** — pool 400 train examples into a single `cca_stats.pt` (joint-Q-K eigenbasis) + `v_stats.pt`.
3. **Run v7 sweep** — 192 cells (12 tasks × 16 configs) of LongBench evaluation with TurboQuant baselines, KIVI int{2,3,4}, and JointQK with the new calibration + v_turboquant V.

Total wall time on 6× A100-40GB: **~1 h capture + ~10 min stats + ~15 min build + ~6–10 h v7 sweep** ≈ **~7–11 h end-to-end**. The v7 sweep is by far the longest stage because each cell does prefill *and* autoregressive generation; capture is prefill-only and therefore ~10× faster per prompt.

---

## Cross-model isolation (won't clobber the Qwen3 calibration)

The Llama run uses **distinct paths** at every stage so it cannot accidentally overwrite the existing Qwen3 artifacts at `artifacts/calibration/longbench_compact8_qkv/`:

| stage | Qwen3 path (existing, must not be touched) | Llama path (new, this runbook) |
|---|---|---|
| Capture run-root | `artifacts/calibration/longbench_compact8_qkv/` | `artifacts/calibration/longbench_compact8_qkv_llama31_8b/` |
| K calibration artifact | `artifacts/bases/cca_stats_longbench_compact8_n400.pt` | `artifacts/bases/cca_stats_llama31_8b_longbench_compact8_n400.pt` |
| V calibration artifact | `artifacts/v_bases/v_stats_longbench_compact8_n400.pt` | `artifacts/v_bases/v_stats_llama31_8b_longbench_compact8_n400.pt` |
| Downstream F1 | `artifacts/bench/qwen3_8b/` | `artifacts/bench/llama31_8b/` |
| Logs | `experiments/logs/phase7_v7_qwen3_8b/` | `experiments/logs/phase7_v7_llama31_8b/` |

**Three safety guards are in place** in case a flag gets dropped by accident:

1. **Capture run-root model guard.** `capture_raw.py` writes `_run_meta.json` to the run-root on first capture. Subsequent invocations refuse to write if the recorded `model` differs. The Qwen3 directory has been seeded with `{"model": "Qwen/Qwen3-8B"}`, so any attempt to capture Llama into `longbench_compact8_qkv` (the Qwen3 run-id) will fail with:
   ```
   RuntimeError: refusing to capture into run-id='longbench_compact8_qkv' ... existing _run_meta.json has model='Qwen/Qwen3-8B', current --model='meta-llama/Llama-3.1-8B-Instruct'
   ```

2. **Build script overwrite guard.** `build_calibration_artifacts_from_pool.py` refuses to overwrite an existing output without `--force`. Running it without `--output-suffix` for Llama would target the Qwen3 file, but the existing file blocks the write:
   ```
   RuntimeError: refusing to overwrite existing .../cca_stats_longbench_compact8_n400.pt (use --force to override).
   ```

3. **v7 launcher fail-loud.** `launch.sh --model llama31_8b` checks for the Llama-suffixed calibration files at start. If they don't exist (because step 3 wasn't run yet), it prints the exact build command and exits before queuing any jobs.

Together these mean: **the agent has to make three independent mistakes** — wrong run-id AND wrong output-suffix AND `--force` — before they can clobber the Qwen3 calibration. With the runbook's commands as-written, none of those is possible.

---

## 0. Prerequisites

- A GPU server with at least 6× A100-40GB (or 4× if you reduce parallelism). The v7 sweep hard-codes GPUs `0,1,2,3,4,5` by default — pass `--gpus` to change.
- The repo synced to the remote machine. The local `sync_to_remote.sh` script does this:
  ```bash
  ./sync_to_remote.sh --host <remote-host> --phase7
  ```
  The `--phase7` profile copies:
  - All code, notes, vendored kvpress + turboquant-pytorch (~165 MB project tree).
  - `artifacts/v_bases/v_lock.txt` (selects `v_turboquant`).
  - `artifacts/calibration_splits/longbench_compact8_60_seed20260504_2k32k/` (manifest, train/test row jsonls, exclude file — ~760 KB total). The capture stage reads `manifest.json`; the v7 launcher reads `exclude_train_indices_for_eval.json`.
  - Legacy v6 calibration artifacts (cca_stats.pt, v_stats.pt) for both qwen3_8b and llama31_8b — kept for v6 reproducibility, not used by v7.
  - **NOT included**: multi-hundred-GB raw and per-example stats directories. The Llama v7 calibration files (`cca_stats_llama31_8b_*.pt`, `v_stats_llama31_8b_*.pt`) are built **on the remote** in §2 below from the locally-captured prefill stats.

  Total transfer ~510 MB.
- Python 3.12 + uv. After sync:
  ```bash
  cd /vault/amir/efficient-llm/teamily-project
  uv venv
  source .venv/bin/activate
  uv sync   # installs torch / transformers / kvpress / etc.
  ```
- HuggingFace token with access to `meta-llama/Llama-3.1-8B-Instruct` and `Xnhyacinth/LongBench`:
  ```bash
  export HF_TOKEN=...
  huggingface-cli login --token $HF_TOKEN  # or HF_HUB_OFFLINE=0 + ~/.cache/huggingface/token
  ```

Verify the toolkit imports before going further:
```bash
.venv/bin/python -c "
from experiments.calibration.analyze_bases import combine_stats, jointqk_basis
from experiments.toolkit.jointqk_press import JointQKPress
from experiments.toolkit.turboquant_press import TurboQuantPress
print('OK')
"
```

---

## 0.5. End-to-end smoke test (~15 min, 1 GPU)

Run the **full pipeline** on a tiny subset before committing 22 hours to it. This catches HF auth issues, missing tasks, and toolkit incompatibilities upfront. The smoke run uses a dedicated `--run-id` and `--output-suffix` so its output cannot collide with the real run.

```bash
SMOKE_RUN_ID=longbench_compact8_qkv_llama31_8b_smoke
SMOKE_SUFFIX=llama31_8b_smoke

# 1. Capture: --smoke flag selects the built-in 2-prompt-per-task subset
.venv/bin/python experiments/calibration/launch.py \
    --stage capture --gpus 0 --jobs-per-gpu 1 --keep-raw test --smoke \
    --run-id "$SMOKE_RUN_ID" \
    --model meta-llama/Llama-3.1-8B-Instruct

# 2. Aggregate stats
.venv/bin/python experiments/calibration/launch.py \
    --stage stats --gpus 0 --smoke --run-id "$SMOKE_RUN_ID"

# 3. Build calibration artifacts on the smoke pool (smaller, faster)
.venv/bin/python experiments/scripts/build_calibration_artifacts_from_pool.py \
    --run-id "$SMOKE_RUN_ID" \
    --output-suffix "$SMOKE_SUFFIX"

# 4. Run a 2-cell smoke sweep: full_precision + JointQK on qasper at fraction=0.05
.venv/bin/python experiments/scripts/phase7_worker.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --gpus 0 --jobs-per-gpu 1 --max-retries 1 \
    --log-dir experiments/logs/phase7_v7_smoke \
    --commands-file <(cat <<EOF
{"_label": "smoke_oracle_qasper", "press_name": "no_press", "compression_ratio": 0.0, "dataset": "longbench", "data_dir": "qasper", "fraction": 0.05, "output_dir": "artifacts/bench_smoke/llama31_8b/full_precision_qasper"}
{"_label": "smoke_jointqk_qasper", "press_name": "jointqk", "press_kwargs": {"cca_stats_path": "artifacts/bases/cca_stats_${SMOKE_SUFFIX}.pt", "v_stats_path": "artifacts/v_bases/v_stats_${SMOKE_SUFFIX}.pt", "v_method": "v_turboquant", "k_bits": 2, "v_bits": 3, "compress_decode": false, "layer0_full_precision": true, "quantize_k": true, "quantize_v": true}, "dataset": "longbench", "data_dir": "qasper", "fraction": 0.05, "output_dir": "artifacts/bench_smoke/llama31_8b/jointqk_k2_v3_qasper"}
EOF
)

# 5. Verify outputs
ls artifacts/bench_smoke/llama31_8b/*/longbench__qasper*/metrics.json
cat artifacts/bench_smoke/llama31_8b/full_precision_qasper/longbench__qasper*/metrics.json
cat artifacts/bench_smoke/llama31_8b/jointqk_k2_v3_qasper/longbench__qasper*/metrics.json
```

**What success looks like:**
- Step 1: 16 capture entries (8 tasks × 2 prompts) finish in ~3 min on one GPU. Stats per shard written to `artifacts/calibration/<SMOKE_RUN_ID>/02_stats/`.
- Step 2: `aggregate.pt` written, prints `total=16, train≈12, test≈4` (smoke uses smaller per-task counts; see `select_rows(split, smoke=True)`).
- Step 3: `cca_stats_${SMOKE_SUFFIX}.pt` (~90 MB) + `v_stats_${SMOKE_SUFFIX}.pt` (~36 MB) written.
- Step 4: 2 cells run sequentially in ~10–15 min total (first cell pays the ~13 min compressor build; second hits the in-memory cache).
- Step 5: both `metrics.json` files exist with a single float each (typical qasper F1 at fraction=0.05 is noisy — values 30–60 are normal at ~10 samples).

**If anything fails**, fix it before launching the full sweep. Common smoke failures:
- HF token doesn't have Llama access → step 1 fails on first model load.
- `Xnhyacinth/LongBench` task subdir not in the local HF cache → step 4 fails on `load_dataset`. Pre-pull as in §5.
- The compressor cache dir doesn't exist → JointQK press auto-creates it; if you see permission errors, `mkdir -p artifacts/_compressor_cache && chmod u+w` it.

**Cleanup the smoke run** before launching real:
```bash
rm -rf artifacts/calibration/longbench_compact8_qkv_llama31_8b_smoke \
       artifacts/bases/cca_stats_llama31_8b_smoke.pt \
       artifacts/v_bases/v_stats_llama31_8b_smoke.pt \
       artifacts/bench_smoke \
       experiments/logs/phase7_v7_smoke
```

---

## 1. Capture stage — fp16 q/k/v on 480 LongBench prompts

The capture pipeline reads a fixed split manifest (committed to the repo so the train/test splits match the Qwen run for cross-model comparisons) and runs Llama prefill once per prompt.

**Split manifest** (already in repo):
```
artifacts/calibration_splits/longbench_compact8_60_seed20260504_2k32k/manifest.json
```
This covers 8 LongBench tasks × 60 prompts each (50 train + 10 test) = 480 prompts, 2k–32k tokens.

**Run capture:**
```bash
cd /vault/amir/efficient-llm/teamily-project
.venv/bin/python experiments/calibration/launch.py \
    --stage capture \
    --gpus 0,1,2,3,4,5 \
    --jobs-per-gpu 1 \
    --keep-raw test \
    --run-id longbench_compact8_qkv_llama31_8b \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --resume
```

What this does:
- Spawns one capture worker per GPU.
- Each worker: loads Llama on its GPU, walks its shard of the 480 prompts, runs `model(input_ids, use_cache=True)` to do prefill, hooks q/k/v, computes per-example moments (`sq_sum`, `sk_sum`, `cqk_sum`, `sv_sum`, `sum_q`, `sum_k`, `sum_v`) inline, and saves to `02_stats/shard_NNN/<id>.pt`. Test-split prompts also get their fp16 raw payload saved to `01_raw/shard_NNN/<id>.pt`.
- Per-shard structured progress logs go to `experiments/logs/phase_capture/`. Watch progress with:
  ```bash
  tail -F artifacts/calibration/longbench_compact8_qkv_llama31_8b/logs/capture/_overview.log
  ```

**Disk cost:**
- Stats: ~75 MB × 480 = ~36 GB.
- Raw fp16 (test only): 1–10 GB per prompt × 80 ≈ ~500 GB.
- Total: budget ~600 GB free under `artifacts/calibration/longbench_compact8_qkv_llama31_8b/`.

**Wall-time:** ~40–90 min on 6 GPUs. Capture is **prefill-only** (no generation), so per-prompt cost is just `prefill_time + tensor-to-CPU + disk-write`. For 2k–32k-token LongBench prompts on Llama-3.1-8B at fp16: ~1 sec prefill at 2k → ~30 sec at 32k, plus ~5–10 sec to move q/k/v to CPU and ~10 sec to write the ~5 GB raw .pt for test prompts. Mean ~30 sec/prompt × 80 prompts/worker = ~40 min, with the long-tail 32k prompts pushing the slowest worker to ~60–90 min.

**Resume:** if a worker dies, just rerun the same command with `--resume` and it'll skip examples whose stats + (where required) raw files are already on disk and validate.

**Cross-model hash compatibility:** the manifest's row-integrity check uses `messages_sha256` (a tokenizer-independent hash of the canonical messages list), not `prompt_sha256` (which is tokenizer-locked because chat templates differ between Qwen3 and Llama-3.1). Capture under either model produces the same `messages_sha256`, so the same manifest works across models without spurious mismatches. If a capture fails with `Messages hash mismatch`, the LongBench dataset row content has actually drifted upstream — investigate before re-running.

After capture finishes, compute the merged aggregate:

```bash
.venv/bin/python experiments/calibration/launch.py \
    --stage stats \
    --gpus 0 \
    --run-id longbench_compact8_qkv_llama31_8b
```

This is single-process, takes ~2 min, and writes `02_stats/aggregate.pt` (~2.5 GB — pooled per_example metadata + tensor stats).

**Sanity check:**
```bash
.venv/bin/python -c "
import torch
agg = torch.load('artifacts/calibration/longbench_compact8_qkv_llama31_8b/02_stats/aggregate.pt', map_location='cpu', weights_only=False)
per = agg['per_example']
print('total examples:', len(per))
print('train:', sum(1 for p in per if p['split']=='train'))
print('test:', sum(1 for p in per if p['split']=='test'))
print('first stats keys:', list(torch.load(per[0]['file'], weights_only=False).keys())[:6])
"
```
Expect `total=480, train=400, test=80`.

---

## 2. Build the v7 calibration artifacts

This pools the 400 train examples and emits the two artifact files the press will read.

```bash
.venv/bin/python experiments/scripts/build_calibration_artifacts_from_pool.py \
    --run-id longbench_compact8_qkv_llama31_8b \
    --output-suffix llama31_8b_longbench_compact8_n400
```

Outputs:
- `artifacts/bases/cca_stats_llama31_8b_longbench_compact8_n400.pt` (~90 MB)
- `artifacts/v_bases/v_stats_llama31_8b_longbench_compact8_n400.pt` (~36 MB)

The script:
- Sums per-example second moments across the 400 train indices.
- Computes the joint-Q-K eigenbasis `R_sym = eigvec_descending((Σ_Q Σ_K + Σ_K Σ_Q)/2)` per (layer, kv_head).
- Saves a `cca_stats.pt` payload with `sigma_q`, `sigma_k`, `R_sym`, plus identity / zero placeholders for CCA-only fields the press constructor reads but `r_sym_waterfill` doesn't actually use.
- Saves a `v_stats.pt` with `cov_v`, `mu_v`, `sigma_v` (uncentered legacy field). **Note: `v_turboquant` does not read `v_stats.pt`** (it's calibration-free random Hadamard with a per-layer seed). The file is built for completeness in case you want to ablate `v_random` / `v_eigen_uniform` later.

**Wall-time:** ~10 min CPU (mostly torch.load deserialization of 400 × 75 MB stats files).

---

## 3. Run the v7 sweep

```bash
.venv/bin/python experiments/scripts/launch.sh --model llama31_8b
```

This launches **192 cells** (12 tasks × 16 configs):
- **Tasks (12):** KIVI 8 (qasper, qmsum, multi_news, trec, triviaqa, samsum, lcc, repobench-p) + multi-doc QA (hotpotqa, musique, 2wikimqa, narrativeqa).
- **Configs/task (16):** 1 oracle (`no_press` full precision) + 6 JointQK (K∈{2,3,4} × V∈{2,3}) + 6 TurboQuant (same K×V grid) + 3 KIVI (int2, int3, int4).
- All compressed methods: **`layer0_full_precision=True`** (now press default), **Mode A** (`compress_decode=False`), `fraction=1.0` (~200 samples per task).
- JointQK uses the new pooled-400 K calibration + `v_turboquant` V (per `v_lock.txt`).
- **Calibration train rows are excluded from the F1 evaluation** via `--exclude_indices_file` (the launcher passes this automatically). For the 7 calibration-overlapping tasks (hotpotqa, multi_news, musique, qasper, qmsum, repobench-p, triviaqa), 50/200 rows used to fit Σ_Q / Σ_K are dropped → eval N drops to ~150 on those tasks. The other 5 tasks (2wikimqa, lcc, narrativeqa, samsum, trec) are unaffected (not in the calibration corpus). Exclude file: `artifacts/calibration_splits/longbench_compact8_60_seed20260504_2k32k/exclude_train_indices_for_eval.json` — committed; no regeneration needed.

**Defaults & optional flags:**
- `--gpus 0,1,2,3,4,5` (default — 6 GPUs)
- `--jobs-per-gpu 2` (**default 2** for v7; the launcher uses `phase7_worker.py` which retries OOMs)
- `--max-retries 10` (**default 10** — phase7_worker auto-requeues OOM'd cells up to this many times)
- `--fraction 0.5` (faster preliminary; default `1.0`)
- `--dry-run` (emits the JSONL command list and exits without running anything)

**Wall-time:** ~6–10 h on 6× A100-40GB at fraction=1.0. The v7 launcher uses `phase7_worker.py` which keeps the model loaded across cells (no per-cell ~1 min model-load) and caches the JointQK compressor in process memory (one ~13 min build per unique `(k_bits, v_bits)` per worker, then ~5 sec hits). With 2 jobs/GPU and 6 GPUs that's effectively 12 workers; 192 cells / 12 workers ≈ 16 cells per worker.

**Watch progress:**
```bash
tail -F experiments/logs/phase7_v7_llama31_8b/_overview.log
```

**Output layout:**
```
artifacts/bench/llama31_8b/
├── full_precision_qasper/
│   └── longbench__qasper__meta-llama--Llama-3.1-8B-Instruct__no_press__0.00__fraction1.000/
│       ├── metrics.json
│       ├── predictions.csv
│       └── config.yaml
├── jointqk_k2_v3_qasper/
├── turboquant_k2_v3_qasper/
├── kivi_int4_qasper/
└── ... (192 dirs total)
```

`metrics.json` contains the LongBench F1 score (single float). `predictions.csv` is the per-prompt model output for debugging.

**Resume on failure:** the v7 launcher uses `phase7_worker.py` which is idempotent — re-running it skips any cell whose canonical results dir already has both `predictions.csv` and `metrics.json`. So if a worker crashes, just re-run the same launcher command and it'll pick up where it left off. The `_overview.log` shows status per cell:

| status | meaning |
|---|---|
| `OK` | cell finished, results on disk |
| `OOM` | OOM'd, requeued (counted toward `--max-retries`) |
| `OOM!` | OOM'd `max_retries` times, given up |
| `FAIL` | non-OOM exception (look at the per-cell `job_NNN_*_aN.log`) |

---

## 4. Aggregate and report

After all 192 cells complete, build the F1 grid:

```bash
.venv/bin/python <<'PY'
import json
from pathlib import Path

base = Path("artifacts/bench/llama31_8b")
TASKS = ["qasper", "qmsum", "multi_news", "trec", "triviaqa", "samsum", "lcc",
         "repobench-p", "hotpotqa", "musique", "2wikimqa", "narrativeqa"]
CONFIGS = ["full_precision",
           *[f"jointqk_k{k}_v{v}"   for k in (2,3,4) for v in (2,3)],
           *[f"turboquant_k{k}_v{v}" for k in (2,3,4) for v in (2,3)],
           "kivi_int2", "kivi_int3", "kivi_int4"]

def f1(label, task):
    d = base / f"{label}_{task}"
    sub = next((p for p in d.iterdir() if p.is_dir()), None) if d.exists() else None
    return float((sub/"metrics.json").read_text().strip()) if sub and (sub/"metrics.json").exists() else None

print(f"{'config':<28} | " + " | ".join(f"{t[:10]:>10}" for t in TASKS) + " |   mean")
print("-" * (29 + 13*len(TASKS) + 8))
for cfg in CONFIGS:
    vals = [f1(cfg, t) for t in TASKS]
    pretty = [f"{v:>10.2f}" if v is not None else f"{'—':>10}" for v in vals]
    valid = [v for v in vals if v is not None]
    m = sum(valid)/len(valid) if valid else None
    mstr = f"{m:>6.2f}" if m is not None else f"{'—':>6}"
    print(f"{cfg:<28} | " + " | ".join(pretty) + f" | {mstr}")
PY
```

Expected headlines (drawing from the Qwen3 v7 fair-comparison numbers; Llama may differ but the *shape* should be similar — JointQK should match TurboQuant at K=4 and beat it at K=2, especially on multi-doc QA):

- **K=4 / V=3:** all three (FP, TurboQuant, JointQK) within ~0.5 pp.
- **K=2 / V=3 mean:** JointQK > TurboQuant by a few pp; on `hotpotqa` specifically the gap should be 10–25 pp.
- **KIVI int4** matches FP; **KIVI int2** lags substantially.

Save the grid to `notes/bench_llama31_8b_results_report.md`. Compare to:
- The Qwen3-8B v7 numbers (when available) for cross-model agreement.
- The published v6 Qwen3-8B numbers in `notes/phase7_v6_results_report.md` to see how the calibration + V-method changes translate.

---

## 5. Common failure modes / fixes

- **Capture OOM at long context.** Drop to `--jobs-per-gpu 1` and ensure no other Llama processes are sharing the GPU.
- **`Couldn't find cache for THUDM/LongBench`.** kvpress's evaluate.py uses `Xnhyacinth/LongBench`. If a task download fails, set `HF_DATASETS_OFFLINE=0` and pre-download:
  ```bash
  .venv/bin/python -c "from datasets import load_dataset; load_dataset('Xnhyacinth/LongBench', '2wikimqa', split='test')"
  ```
  Repeat for each missing task.
- **JointQK compressor build hangs at first cell.** It's CPU-bound on 36 × 8 = 288 Lloyd-Max codebook solves per (layer, head). Expected duration ~13 min the first time per worker. Subsequent cells in the same worker process use the in-memory press cache (~5 sec). **Don't kill it.** With `phase7_worker.py` and 12 workers, you'll see ~12 of these initial builds racing in parallel — that's expected.
- **KIVI cells OOM at long context with 2 jobs/GPU.** v7's `--max-retries 10` should auto-requeue them and they'll usually fit once neighbour cells finish and free memory. If a KIVI cell shows up as `OOM!` (final OOM after all retries) in `_overview.log`, rerun just the failed cells at 1/GPU:
  ```bash
  bash experiments/scripts/launch.sh --model llama31_8b --jobs-per-gpu 1
  ```
  (The worker's skip-if-exists guarantees only the failed cells re-execute.)
- **Disk fill-up during capture.** Each test-split raw file is up to 10 GB. Make sure `df -h` shows >600 GB free under the artifact root before launching.

---

## 6. Files / paths summary

| purpose | path |
|---|---|
| Split manifest (immutable) | `artifacts/calibration_splits/longbench_compact8_60_seed20260504_2k32k/manifest.json` |
| Per-example raw fp16 (test-split only) | `artifacts/calibration/longbench_compact8_qkv_llama31_8b/01_raw/shard_NNN/*.pt` |
| Per-example stats | `artifacts/calibration/longbench_compact8_qkv_llama31_8b/02_stats/shard_NNN/*.pt` |
| Aggregated stats | `artifacts/calibration/longbench_compact8_qkv_llama31_8b/02_stats/aggregate.pt` |
| v7 K calibration artifact | `artifacts/bases/cca_stats_llama31_8b_longbench_compact8_n400.pt` |
| v7 V calibration artifact | `artifacts/v_bases/v_stats_llama31_8b_longbench_compact8_n400.pt` |
| `v_lock.txt` (V_METHOD=v_turboquant) | `artifacts/v_bases/v_lock.txt` |
| v7 launcher | `experiments/scripts/launch.sh` |
| Downstream F1 outputs | `artifacts/bench/llama31_8b/<config>_<task>/.../metrics.json` |
| Per-cell logs | `experiments/logs/phase7_v7_llama31_8b/job_*_a0.log` |
| Compressor disk cache (built lazily) | `artifacts/_compressor_cache/*.pt` |

---

## 7. End-to-end command summary (TL;DR)

Set `GPUS` once at the top and every stage uses it. **Edit this line for the remote machine's GPU layout** (e.g., `GPUS=0,1,2,3` if only 4 GPUs are available, or `GPUS=4,5,6,7` if 0-3 are reserved):

```bash
# === Edit this for your machine ===
GPUS=0,1,2,3,4,5
# ===================================

# 0. Setup (once per server)
cd /vault/amir/efficient-llm/teamily-project
uv venv && source .venv/bin/activate && uv sync
huggingface-cli login --token $HF_TOKEN

# 0.5. Smoke test FIRST (~15 min, 1 GPU). See §0.5 for full script. DO NOT SKIP.

# 1. Capture (~40-90 min; prefill-only)
.venv/bin/python experiments/calibration/launch.py \
    --stage capture --gpus "$GPUS" --jobs-per-gpu 1 --keep-raw test \
    --run-id longbench_compact8_qkv_llama31_8b \
    --model meta-llama/Llama-3.1-8B-Instruct --resume

# 2. Aggregate stats (~2 min, single-GPU)
.venv/bin/python experiments/calibration/launch.py \
    --stage stats --gpus "${GPUS%%,*}" \
    --run-id longbench_compact8_qkv_llama31_8b

# 3. Build calibration artifacts (~15 min CPU)
.venv/bin/python experiments/scripts/build_calibration_artifacts_from_pool.py \
    --run-id longbench_compact8_qkv_llama31_8b \
    --output-suffix llama31_8b_longbench_compact8_n400

# 4. v7 sweep (~6-10 h at 2 jobs/GPU + max-retries 10)
nohup bash experiments/scripts/launch.sh \
    --model llama31_8b --gpus "$GPUS" \
    > experiments/logs/phase7_v7_llama31_8b_outer.log 2>&1 &

# 5. Aggregate F1 grid (see Section 4)
```

`${GPUS%%,*}` is the bash expansion that picks the **first** GPU id from the comma-separated list (e.g., `0` from `0,1,2,3,4,5`). The stats stage is single-process so only needs one GPU; every other stage uses the full `$GPUS` set.

Done. Total wall (assuming 6 GPUs): **~7–11 h end-to-end** (smoke ~15 min + capture ~1 h + stats ~2 min + build ~15 min + sweep ~6-10 h). With 4 GPUs the sweep is ~9-15 h; with 8 GPUs it's ~5-7 h. The sweep dominates because it does prefill+generation per cell; capture is prefill-only.
