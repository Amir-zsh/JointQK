# KV-Cache Compression Research

A research codebase studying how to compress the **key** half of the KV
cache in transformer LLMs without losing the attention behaviour the model
relies on, with downstream LongBench / RULER F1 as the headline metric.

---

## What this project is

Modern transformer LLMs cache the keys (`K`) and values (`V`) produced
during prefill so subsequent generated tokens can attend to the prompt
without re-running the forward pass. This **KV cache** grows linearly with
context length and dominates GPU memory at long contexts — gigabytes per
active sequence on 8B-class models with 30k-token prompts. Cheap,
high-quality KV-cache compression is one of the most consequential levers
for serving long-context LLMs.

This project specifically investigates compression of the **K** half:
per-(layer, kv_head) linear bases learned from second-moment statistics,
paired with per-coordinate bit allocations. The headline method, **JointQK
WaterFill**, calibrates a joint Q-K eigenbasis $R_{\text{sym}} =
\text{eigvec}\bigl((\Sigma_Q\Sigma_K + \Sigma_K\Sigma_Q)/2\bigr)$ offline
from a small prompt corpus, then water-fills bits across coordinates by the
Q-weighted reconstruction-MSE metric. V-side compression is benchmarked but
treated separately and is largely orthogonal.

Every method we evaluate fits a common three-phase template:

1. **Calibration (offline, once).** Observe queries and keys on a
   representative corpus; compute per-(layer, kv_head) second moments
   $\Sigma_Q$, $\Sigma_K$, and the cross-moment $C_{QK}$.
2. **Compression (online, per prefill).** Linearly transform each prefill
   key into a calibration-chosen basis. Independently scalar-quantize each
   coordinate to a small Lloyd–Max codebook scaled to that coordinate's
   standard deviation. The compressed keys replace raw keys in the cache.
3. **Reconstruction (online, per generated token).** When a future query
   arrives, dequantize the cached keys and compute attention scores against
   them in place of the originals.

Two design knobs govern this template: **the basis** (random rotation,
Q-eigenbasis, CCA, joint Q-K eigenbasis, …) and **the bit allocation**
(uniform, top-r truncation, reverse water-fill on a per-coordinate weight).
The codebase systematically evaluates them on real LLMs and real
long-context tasks.

### Headline metrics
- **LongBench F1** (8–12 tasks) — per-task end-to-end task accuracy.
- **RULER** (3 context lengths × 4 NIAH variants).
- **Top-1 retention** — fraction of queries whose argmax over the
  compressed keys matches the uncompressed baseline.
- **Q-weighted reconstruction MSE**:
  $\mathbb{E}\!\bigl[(k-\hat k)^\top \Sigma_Q (k-\hat k)\bigr]/d$.

Currently studied models: Qwen3-8B (production) and Llama-3.1-8B (for
F1-inversion investigation).

---

## Setup

Three steps after `git clone`. Total wall-clock: ~5 minutes on a fast link.

### 1. Clone the vendored libraries into `vendor/`

The three third-party libs are intentionally **not** committed into this
repo — each is its own upstream git clone, listed in `.gitignore`. Pull
them at the pinned commits documented in [`vendor/README.md`](vendor/README.md):

```bash
git clone https://github.com/NVIDIA/kvpress.git         vendor/kvpress
git clone https://github.com/IBM/turboquant-pytorch.git vendor/turboquant-pytorch
git clone https://github.com/jy-yuan/KIVI.git           vendor/kivi

# Underscore-name symlink so `import turboquant_pytorch` resolves — the
# hyphen in the source-dir name isn't a valid Python module name:
ln -sfn turboquant-pytorch vendor/turboquant_pytorch
```

Optional but recommended — check out the pinned commits for reproducibility:

```bash
git -C vendor/kvpress            checkout d8349a9
git -C vendor/turboquant-pytorch checkout 03e6112
git -C vendor/kivi               checkout 876b4d2
```

### 2. Create the Python environment

The project's `.venv/` is uv-managed (Python 3.12 + PyTorch CUDA 12.8 +
transformers + the bench dependencies). Conda is **not** used.

```bash
uv venv .venv --python 3.12
uv pip install -r requirements.lock.txt --python .venv/bin/python
```

A pinned lockfile (`requirements.lock.txt`) ships in the repo. The
single-command alternative `uv sync` will work once the upstream's
pyproject is added.

### 3. That's it — no project install needed

This repo deliberately does **not** ship a `pyproject.toml` and does
**not** require `pip install -e .`. Every entry-point script self-bootstraps
its `sys.path` with a four-line header at the top:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[N]))  # N: depth to repo root
import _bootstrap  # noqa: E402, F401  (adds vendor/kvpress to sys.path)
```

That puts the repo root on `sys.path` (so `import kvq.*` and
`import pipelines.*` resolve), and `_bootstrap.py` at the repo root
appends `vendor/kvpress/` (so `import kvpress` resolves through the
upstream vendored copy). No `pip install -e`, no `PYTHONPATH`, no `.pth`
magic — just two `sys.path` entries each script declares for itself.

### Smoke test

```bash
# Run a script directly by file path (no `-m`):
.venv/bin/python pipelines/bench/worker.py --help

# Run the regression fingerprint to confirm artifacts still resolve:
.venv/bin/python -m tests.regression_fingerprint check \
    --baseline tests/baselines/fingerprint_pre.json

# Should print: FINGERPRINT CHECK PASSED — 262 values match …
```

Both work without any pre-step beyond steps 1 and 2.

### GPU allocation

GPUs 0–3 are the project's allocation on the shared host. Launchers default
to `--gpus 0,1,2,3,4,5` but every script accepts `--gpus 0,1,2,3` to stay
within the project's pool.

---

## Repository structure

```
.
├── _bootstrap.py                 # sys.path setup helper (imported by every entry-point script)
├── kvq/                          # Importable library (`from kvq.X import Y`)
│   ├── presses/                  #  kvpress press classes: JointQK, TurboQuant, KIVI
│   ├── compression/              #  Bit allocation + scalar quantization primitives
│   ├── capture/                  #  Model loading + RoPE-aware Q/K/V capture hooks
│   ├── benchmarks/, data/        #  Vendored kvpress data adapters + scorers
│   └── io.py                     #  Small utilities (save_json, ensure_dir, …)
│
├── pipelines/                    # Production data flow (calibrate → bench → aggregate)
│   ├── calibration/              #  Capture K/V + compute pooled second moments
│   ├── scripts/                  #  Calibration-build consumer utilities (basis files, manifests, holdout checks)
│   ├── bench/                    #  Downstream LongBench / RULER F1 sweep (worker + launchers)
│   ├── eval/                     #  Per-cell → summary table aggregators
│   └── calibration_stability/    #  Basis-stability ablation sweeps
│
├── analysis/                     # Ad-hoc investigation scripts (Llama JointQK F1-inversion probes, HTML report)
├── tests/                        # Press parity tests + regression fingerprint
├── notebooks/                    # Exploratory notebooks
├── logs/                         # Run logs (gitignored)
│
├── artifacts/                    # Pipeline outputs (mostly gitignored)
│   ├── calibration/              #  Per-corpus raw captures + pooled stats (>10 GB protected; gitignored)
│   ├── bases/                    #  Joint Q-K basis files: jointqk_<model>_<corpus>_<n>.pt
│   ├── v_bases/                  #  V-side basis files + v_lock.txt (active V method)
│   ├── calibration_splits/       #  Train/test split manifests + exclude_train_indices_for_eval.json
│   ├── bench/                    #  Downstream F1 sweep results (per-cell metrics.json)
│   ├── bench_llama_verify/       #  Llama Mode-A F1 sweep
│   ├── bench_llama_compact9/     #  Llama F1 with compact9-pooled basis
│   ├── bench_llama_lcconly/      #  Llama F1 with lcc-only basis
│   ├── decode_q_captures_llama/  #  Decode-Q tensors for Σ_Q drift analysis
│   ├── q_distribution_shift/     #  Σ_Q top-16 drift JSON + charts
│   ├── _compressor_cache/        #  Pre-built per-(layer, head) compressors keyed by SHA(calibration mtime + kwargs)
│   └── query_stats_longbench_under4k*  #  Captured query stats (large, protected)
│
├── notes/                        # Research write-ups
│   ├── README.md                 #  Index
│   ├── core/                     #  Project framing + execution roadmap + paper plans
│   ├── reference/                #  Standalone derivations
│   ├── bench_results_report.md   #  Latest Qwen3-8B downstream F1 results
│   ├── bench_llama31_8b_results_report.md  #  Llama-3.1-8B F1 results
│   ├── bench_cross_model_comparison.md     #  Qwen3 vs Llama
│   ├── bench_llama_runbook.md    #  Runbook for the Llama bench sweep
│   ├── jointqk_disconnect_investigation.md
│   ├── jointqk_investigation_report.html   #  Self-contained HTML write-up
│   ├── q_distribution_shift.md   #  Σ_Q drift analysis
│   ├── experiments_and_findings.md         #  Cross-cutting synthesis
│   └── figs/                     #  Charts embedded in the notes above
│
├── background/                   # Reference papers (PDFs gitignored; README is the index)
│
├── vendor/                       # Third-party libraries
│   ├── README.md                 #  Clone instructions + pinned commits
│   ├── kvpress/                  #  NVIDIA's kvpress (Apache-2.0); local extensions to evaluate.py + evaluate_registry.py
│   ├── turboquant-pytorch/       #  TurboQuant baseline
│   ├── turboquant_pytorch        #  → turboquant-pytorch (symlink)
│   └── kivi/                     #  KIVI baseline
│
└── paper/                        # Paper drafts (gitignored)
```

### Naming conventions to be aware of

- **`jointqk_*.pt`** files in `artifacts/bases/` store the joint Q-K
  basis (`R_sym`), despite the historical "cca_" prefix. CCA-only fields
  remain in the file but are ignored by the K path.
- **"Bench"** = the downstream LongBench/RULER F1 sweep (formerly called
  "Phase 7" in older notes/logs).
- **"Bases"** = `artifacts/bases/` and `artifacts/v_bases/`, the calibrated
  basis files consumed by the bench worker.

---

## Running benchmarks

The bench pipeline is the headline LongBench/RULER F1 sweep. The
calibration step is rarely re-run because the captures are large; details
are in [the calibration section below](#calibration-rare-rerun). Ad-hoc
investigation scripts live in `analysis/` and have [their own section
further down](#running-analysis-probes).

### Common preconditions

Every bench run requires three artifacts to already exist:

1. **A calibration corpus capture** under
   `artifacts/calibration/<run-id>/02_stats/` (gitignored; >10 GB
   protected).
2. **A K basis file** like `artifacts/bases/jointqk_<model>_<corpus>_<n>.pt`.
3. **A V basis file** like `artifacts/v_bases/v_stats_<model>_<corpus>_<n>.pt`
   plus `artifacts/v_bases/v_lock.txt` (records the active V method).

These are produced once per model+corpus by the calibration pipeline below.
The currently active set is the Qwen3-8B and Llama-3.1-8B compact8/compact9
pools.

### 1. Headline bench sweep (the "v7" config)

```bash
bash pipelines/bench/launch.sh --gpus 0,1,2,3
```

This is the canonical 192-cell sweep documented in
`notes/bench_results_report.md`. It runs, per model:

- 1 full-precision oracle
- 6 JointQK cells: $K\in\{2,3,4\}$ bits × $V\in\{2,3\}$ bits
- 6 TurboQuant cells: same grid
- 3 KIVI baselines: int2, int3, int4

on 12 LongBench tasks (8 KIVI tasks + 4 multi-doc QA). All compressed
methods use `layer0_full_precision=True` and `compress_decode=False` (Mode
A — only prefill keys are compressed; decode-step keys stay fp16).

Useful flags:
- `--fraction 0.1` — sample 10% of each task (smoke test, ≈5 min/cell).
- `--jobs-per-gpu 2` — pack two model copies per GPU (only with ≥40 GB
  VRAM).
- `--dry-run` — print the JSONL of work items without launching workers.
- `--model qwen3_8b` (default) or `--model llama31_8b`.

Output goes to `artifacts/bench/<model_tag>/<cell_name>/longbench__<task>__<model>__<press>__1.00/metrics.json`,
which is a single F1 number per cell.

### 2. Other bench sub-sweeps

| launcher | purpose |
|---|---|
| `bench/launch.sh` | Headline v7 sweep (above). |
| `bench/launch_longbench.sh` | Standalone LongBench-only sweep (KIVI's 8-task subset). |
| `bench/launch_ruler.sh` | RULER NIAH at ctx ∈ {4096, 8192, 16384}. |
| `bench/launch_basis_compare.sh` | Side-by-side: old basis vs newly-built basis. |
| `bench/launch_v_turboquant.sh` | V-method ablation holding K fixed. |
| `bench/launch_v_ablation.sh` | V-method ablation across K∈{2,4}. |
| `bench/launch_per_task_basis.sh` | Per-task K bases (not pooled). |

Every launcher writes per-cell `metrics.json` files under a launcher-named
`artifacts/<dir>/` subtree.

### 3. Single-cell debug

To run one specific (model, basis, K, V, task) combination directly through
the worker, bypassing the launcher's queue:

```bash
# Build a one-line JSONL with the cell config:
cat > /tmp/one_cell.jsonl <<'EOF'
{"press_name": "jointqk", "dataset": "longbench", "data_dir": "qasper", "output_dir": "artifacts/_refactor_smoke/qasper_jq_k2v3", "press_kwargs": {"cca_stats_path": "artifacts/bases/jointqk_llama31_8b_longbench_compact8_n400.pt", "v_stats_path": "artifacts/v_bases/v_stats_llama31_8b_longbench_compact8_n400.pt", "v_method": "v_turboquant", "k_bits": 2, "v_bits": 3, "quantize_k": true, "quantize_v": true, "compress_decode": false, "layer0_full_precision": true}}
EOF

# Run it:
.venv/bin/python pipelines/bench/worker.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --commands-file /tmp/one_cell.jsonl \
    --gpus 0 \
    --jobs-per-gpu 1
```

Useful for reproducing a single F1 number, profiling, or debugging a press
class change without queuing the full sweep.

### 4. Aggregating per-cell results

```bash
.venv/bin/python -m pipelines.eval.aggregate_longbench  --root artifacts/bench
.venv/bin/python -m pipelines.eval.aggregate_ruler      --root artifacts/bench
```

Each aggregator walks the per-cell `metrics.json` files and emits a
canonical summary table for the report notes.

### Verifying nothing regressed

Every commit on the refactor branch is gated by a numerical fingerprint:

```bash
.venv/bin/python -m tests.regression_fingerprint check \
    --baseline tests/baselines/fingerprint_pre.json
```

This walks every `metrics.json` under `artifacts/bench*/` plus the
`q_distribution_shift/per_task_drift.json` and compares values against the
pinned baseline (246 F1 cells + 8 Σ_Q drift tasks, 262 values total). A
clean refactor leaves all 262 values byte-identical.

### Calibration (rare rerun)

If you need to add a new calibration corpus or model:

```bash
# 1. Create a split manifest (which rows go to train vs eval):
.venv/bin/python pipelines/scripts/create_longbench_calibration_split.py \
    --tasks qasper,hotpotqa,musique,qmsum,multi_news,triviaqa,passage_retrieval_en,repobench-p,lcc \
    --samples-per-task 60 --train-per-task 50 --test-per-task 10 \
    --output-dir artifacts/calibration_splits/longbench_compact9_60_seed20260504_2k32k

# 2. Capture K/V across the corpus + compute pooled stats (multi-GPU):
.venv/bin/python -m pipelines.calibration.launch --stage all \
    --split-manifest artifacts/calibration_splits/.../manifest.json \
    --run-id longbench_compact9_qkv_llama31_8b \
    --gpus 0,1,2,3 --num-shards 4

# 3. Build the joint Q-K basis file from the train-split stats:
.venv/bin/python pipelines/scripts/build_calibration_artifacts_from_pool.py \
    --run-id longbench_compact9_qkv_llama31_8b \
    --output-suffix llama31_8b_compact9_n450

# 4. Build the V basis file (sigma_v / cov_v / mu_v):
.venv/bin/python pipelines/scripts/calibrate_sigma_v.py \
    --run-id longbench_compact9_qkv_llama31_8b
```

Step 2 is the expensive one (~30 min on 4 GPUs for 450 prompts at 16k
context). Steps 3–4 take seconds.

---

## Running analysis probes

Ad-hoc scripts under `analysis/` consume existing pipeline outputs (bases
under `artifacts/bases/`, F1 results under `artifacts/bench*/`, decode
captures under `artifacts/decode_q_captures_llama/`) to investigate
specific phenomena — primarily the Llama JointQK F1-inversion documented
in `notes/jointqk_disconnect_investigation.md`. Unlike the bench
pipeline, these scripts are not parameterised for new sweeps; they are
investigation sediment kept for reproducibility.

```bash
# K-fidelity per-prompt: K-MSE, top-1, top-5
.venv/bin/python -m analysis.measure_llama_empirical_kmse_top1_top5

# First-decode logit KL on 4 tasks × 20 prompts
.venv/bin/python -m analysis.measure_logit_kl_llama

# Decode-trajectory KL (teacher-forced per-step KL)
.venv/bin/python -m analysis.measure_decode_trajectory_llama

# Σ_Q top-16 subspace drift across tasks (prefill + decode bins)
.venv/bin/python -m analysis.analyze_q_distribution_shift
.venv/bin/python -m analysis.plot_q_distribution_shift

# Regenerate the consolidated HTML report
.venv/bin/python -m analysis.build_jq_investigation_html \
    --out notes/jointqk_investigation_report.html
```

Direct file-path invocation works too — every script in `analysis/`
ships the same `sys.path` bootstrap header as the pipeline entry points.

---

## Where to read more

- **`notes/core/kv_cache_rate_distortion_proposal.md`** — high-level
  research framing.
- **`notes/core/research_plan.md`** — execution roadmap.
- **`notes/bench_results_report.md`** — latest Qwen3-8B F1 results.
- **`notes/bench_llama31_8b_results_report.md`** — Llama F1 results.
- **`notes/jointqk_disconnect_investigation.md`** + the HTML report — full
  write-up of the K-fidelity-vs-F1 mismatch on Llama.
- **`background/README.md`** — index of reference papers (TurboQuant,
  Expected Attention, Attention Matching, QPTQ).

---

## Vendored dependencies

| Path | Source | Role |
|---|---|---|
| `vendor/kvpress/` | NVIDIA's [kvpress](https://github.com/NVIDIA/kvpress) | Press base classes + the `evaluation/evaluate.py` harness. We extend it with `exclude_indices_file`, `press_kwargs`, and class-instantiation in `_setup_press`. |
| `vendor/turboquant-pytorch/` | IBM's [TurboQuant](https://github.com/IBM/turboquant-pytorch) | Random-Hadamard + Lloyd-Max compressors. Used as a baseline (TurboQuantPress) and as the V-side `v_turboquant` method inside JointQK. |
| `vendor/kivi/` | [jy-yuan/KIVI](https://github.com/jy-yuan/KIVI) | KIVI int2/int3/int4 baseline. Unmodified. |

Each is gitignored from this repo and cloned separately; pinned commits in
`vendor/README.md`.

---
