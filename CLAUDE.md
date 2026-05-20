# CLAUDE.md — Project Guidance for Claude

## What this repo is

A research codebase for **KV-cache key compression** on transformer LLMs. The active line of work calibrates per-(layer, kv_head) second moments from a small prompt corpus, linearly transforms keys into a basis chosen by those statistics, scalar-quantizes each coordinate independently, and decompresses on the fly when future queries arrive. Current focus: comparing (basis × allocation) design choices on long-context benchmarks.

See `README.md` for the public-facing high-level project description and `notes/README.md` for an index of research notes.

## Repo layout (essentials)

- `experiments/`
  - `calibration/` — capture K/V from a prompt corpus, compute pooled second moments (`Σ_Q`, `Σ_K`, `C_QK`), build joint Q-K bases.
  - `bench/` — downstream LongBench/RULER F1 sweep. `worker.py` is the per-cell driver; `launch_*.sh` are the parallel launchers; `parallel_launcher.py` is the GPU pool scheduler; `_chain.py` is the multi-stage orchestrator.
  - `analysis/` — the Llama JointQK F1-inversion probes: K-fidelity measurement, attention-KL, logit-KL, decode-Q trajectory, Σ_Q drift, HTML report builder.
  - `calibration_stability/` — ablation sweeps studying basis stability across calibration corpora (formerly "phase 1AB/1D").
  - `scripts/` — calibration-build utilities (`build_calibration_artifacts_from_pool.py`, `create_longbench_calibration_split.py`, `collect_qk_prefill_stats.py`, ...). Thin consumer-side helpers; capture itself lives in `calibration/`.
  - `toolkit/` — reusable building blocks (`jointqk_press`, `turboquant_press`, `kivi_press`, `per_coord_quantization`, `capture`, `metric_transform`, ...).
  - `eval/` — aggregators (`aggregate_longbench`, `aggregate_ruler`, `aggregate_decode_scope`, `aggregate_phase1`, `aggregate_integrity`).
  - `benchmarks/`, `data/` — vendored kvpress data adapters.
  - `tests/`, `notebooks/`, `logs/` (run logs, gitignored).
- `artifacts/` — per-study output directories. Bench results live under `bench*/`; calibration captures under `calibration/`; Llama analysis under `decode_q_captures_llama/`, `q_distribution_shift/`, etc.
- `notes/` — `core/` (framing), `reference/` (derivations), `<study>/` (per-study write-ups, bug trackers, shareable summaries).
- `vendor/` — vendored Apache-2.0 / similar deps: `vendor/kvpress/`, `vendor/turboquant-pytorch/`, `vendor/kivi/`. Do not modify directly without good reason. (`vendor/turboquant_pytorch` is a symlink so `import turboquant_pytorch` resolves through the hyphen-named source dir.)

## How experiments are organised

The pipeline:

1. **Calibrate.** `experiments/calibration/` captures K/V from a calibration corpus and emits per-(layer, kv_head) second moments. The `scripts/build_calibration_artifacts_from_pool.py` consumer then pools captures into a basis artifact (`cca_stats_*.pt`).
2. **Bench.** `experiments/bench/launch_*.sh` parallelises `bench/worker.py` across GPUs; each worker runs one (method × bits × task) cell of the LongBench/RULER sweep and emits `metrics.json`.
3. **Aggregate.** `experiments/eval/aggregate_*.py` lifts the per-cell JSONs into canonical summary tables.
4. **Analyse.** `experiments/analysis/` runs fidelity probes (K-MSE, top-1, attention-KL, logit-KL, decode trajectory) against the same bases and emits the JQ investigation HTML.

Per-study directories under `artifacts/` follow a consistent pattern: `metrics.json` per cell, optional aggregated `*_summary.json`, optional `report_charts/`.

## Project-specific conventions

- **Headline numbers are layer-0 excluded.** Layer 0 has anomalous norm/condition properties (a finding from earlier in the project). Per-layer arrays are stored full; the headline mean drops index 0.
- **Pre-merge backups.** Before a re-run replaces canonical artifacts, save `.pre_<fix>` copies of every JSON or row-PT it'll overwrite. Existing examples: `*.pre_f11`, `*.pre_newbases`. Large binaries (`*.pt.pre_<fix>`) are `.gitignore`'d; small JSONs are kept under version control for diff/audit.
- **GPU allocation.** GPUs 0–3 are the user's allocation. Do not run on GPUs 4–6; unless asked explicitly.
- **Long studies run autonomously.** When the user asks for a multi-step study, treat it as one logical task: run end-to-end, stream progress to `experiments/logs/<run_name>.log` with a `<run_name>.heartbeat` touched periodically. Validate each step before proceeding to the next; never stack experiments on unvalidated upstream output. Use `python -u` (unbuffered) for any script that emits progress.
- **Bug tracking.** `notes/<study>/fixes_to_apply.md` is for **bugs only** — root cause, fix description, verification result. Do not append "ran cleanly" / activity-log entries; that file is a tracker, not a journal.
- **Regression baselines.** `experiments/tests/baselines/fingerprint_pre.json` pins F1 + Σ_Q drift numbers from the canonical artifacts. Use `python -m experiments.tests.regression_fingerprint check --baseline <path>` to detect drift before a commit lands.
- **New methods plug in at one place.** Method dispatch lives in `build_jointqk_compressor` inside `experiments/toolkit/per_coord_quantization.py`. A new method adds a branch that produces `forward_map`, `inverse_map`, `sigma_k_diag`, and `weights`, then reuses the existing `_uniform` / `_waterfill` allocation tail. Keep additions there rather than scattering them across the driver.

## Common commands

```bash
# Activate the project venv once per shell.
# Use the project's uv-managed `.venv/` (Python 3.12 + torch/transformers/etc.).
# Do NOT use conda — the historical `kv-rd` env no longer exists, and other
# conda envs do not have the project's deps.
source .venv/bin/activate

# Bench sweep (one method × bits × task at a time, parallelised across GPUs):
bash experiments/bench/launch.sh --gpus 0,1,2,3
bash experiments/bench/launch_longbench.sh --gpus 0,1,2,3
bash experiments/bench/launch_ruler.sh --gpus 0,1,2,3

# Single-cell debug — call the worker directly with a JSONL config:
.venv/bin/python experiments/bench/worker.py --commands-file <jsonl> --gpu 0

# Llama JQ investigation probes:
python -m experiments.analysis.measure_logit_kl_llama --tasks lcc hotpotqa
python -m experiments.analysis.analyze_q_distribution_shift --tasks lcc
python -m experiments.analysis.build_jq_investigation_html \
    --out notes/jointqk_investigation_report.html

# Calibration build:
.venv/bin/python experiments/scripts/build_calibration_artifacts_from_pool.py \
    --run-id longbench_compact9_qkv_llama31_8b --output-suffix llama31_8b_compact9_n450

# Calibration-stability sweep:
bash experiments/calibration_stability/launch_ab.sh --gpus 0,1,2,3
```

## Code style

- **Comments only when the *why* is non-obvious.** Hidden constraints, workarounds, surprising invariants — yes. Restating what well-named code does — no. Don't add docstrings that just enumerate parameter types.
- **Trust internal contracts.** Don't add error handling or fallbacks for scenarios that can't happen. Validate at system boundaries (user input, external APIs); inside the toolkit, internal callers are trusted.
- **Edit before creating.** Prefer extending an existing file to introducing a new one. New files are justified when the work doesn't fit anywhere existing.
- **Match existing patterns.** Method-dispatch through `build_jointqk_compressor`, gate-style sanity checks, log/heartbeat conventions in launcher scripts, `*_summary.json` + `*_rows.pt` artifact pairing.

## What NOT to do

- **Don't commit without explicit user approval.** Each commit needs its own "commit now" — a prior approval does not carry across newly added scope.
- **Don't run destructive git operations** (`reset --hard`, `push --force`, `branch -D`, `clean -f`) unless the user explicitly asks.
- **Don't bypass hooks** (`--no-verify`, `--no-gpg-sign`) unless the user explicitly asks.
- **Don't skip validation.** Multi-step pipelines need scripted gate pass/fail between steps. Never assume an upstream step succeeded without checking.
- **Don't pollute trackers.** `fixes_to_apply.md` is for bugs; activity logs go elsewhere or in commit messages.
- **Don't run on GPUs 4–7.**
- **Don't add features the user didn't ask for.** Bug fixes are bug fixes; refactors are a separate ask. Don't generalise an interface "while you're in there."

## Glossary

| Term | Meaning |
|---|---|
| Calibration | Offline computation of per-(layer, kv_head) second moments (`Σ_Q`, `Σ_K`, `C_QK`) from a prompt corpus. |
| `b_avg` | Average bits per coordinate (the compression budget). |
| `r_{95}` | Smallest rank capturing 95% of a spectrum's energy in a given (layer, kv_head). |
| Trace formula | The Q-weighted reconstruction-MSE weight `diag(M^T Σ_Q M)` where `M` is the inverse of the basis forward map. Distinct from canonical correlation `ρ²`; conflating them was a pre-fix bug class (see `notes/fixes_to_apply.md` (when present)). |
| LOO | Leave-one-out cross-validation across calibration examples. |
| GQA | Grouped-Query Attention; queries within a kv-head's group are pooled when forming `Σ_Q` and `C_QK`. |
| Layer-0 excluded | Headline aggregations drop layer 0 because of its anomalous attention-sink behaviour. |
