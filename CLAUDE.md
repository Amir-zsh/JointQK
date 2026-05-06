# CLAUDE.md — Project Guidance for Claude

## What this repo is

A research codebase for **KV-cache key compression** on transformer LLMs. The active line of work calibrates per-(layer, kv_head) second moments from a small prompt corpus, linearly transforms keys into a basis chosen by those statistics, scalar-quantizes each coordinate independently, and decompresses on the fly when future queries arrive. Current focus: comparing (basis × allocation) design choices on long-context benchmarks.

See `README.md` for the public-facing high-level project description and `notes/README.md` for an index of research notes.

## Repo layout (essentials)

- `experiments/<stage>/` — drivers (`run_*.py`), `toolkit/` (reusable building blocks), `scripts/` (launchers, mergers, chart generators), `gates/` (output sanity checks), `notebooks/`, `logs/` (run logs, gitignored).
- `artifacts/<stage>/` — per-study output directories with `*_summary.json`, `*_rows.pt`, `report_charts/`.
- `notes/` — `core/` (framing), `reference/` (derivations), `<stage>/` (per-stage write-ups, bug trackers, shareable summaries).
- `kvpress/`, `turboquant-pytorch/` — vendored Apache-2.0 / similar deps; do not modify directly without good reason.

## How experiments are organised

Each stage follows a common pipeline:

1. A `run_*.py` **driver** consumes a calibration bundle and emits per-(example, layer, kv_head) metrics.
2. **Launcher scripts** under `scripts/` parallelise the driver across GPUs.
3. **Gate scripts** under `gates/` assert basic sanity on the outputs (e.g. all methods produced finite metrics; smoke-test top-1 above thresholds).
4. **Merge / aggregate** scripts lift per-run rows into canonical summary JSONs and chart-ready row PTs.

Per-study directories under `artifacts/<stage>/` follow a consistent pattern: `*_summary.json` (aggregated metrics), `*_rows.pt` (per-row metrics), optional `report_charts/`.

## Project-specific conventions

- **Headline numbers are layer-0 excluded.** Layer 0 has anomalous norm/condition properties (a finding from earlier in Stage 1). Per-layer arrays are stored full; the headline mean drops index 0.
- **Pre-merge backups.** Before a re-run replaces canonical artifacts, save `.pre_<fix>` copies of every JSON or row-PT it'll overwrite. Existing examples: `*.pre_f11`, `*.pre_newbases`. Large binaries (`*.pt.pre_<fix>`) are `.gitignore`'d; small JSONs are kept under version control for diff/audit.
- **Gates are auto-discovery.** `gate_e*.py` iterates over whichever methods appear in each summary JSON rather than a hardcoded list, so adding new methods does not require gate edits — only relaxed thresholds when needed.
- **GPU allocation.** GPUs 0–3 are the user's allocation. Do not run on GPUs 4–6; unless asked explicitly.
- **Long studies run autonomously.** When the user asks for a multi-step study, treat it as one logical task: run end-to-end, stream progress to `experiments/<stage>/logs/<run_name>.log` with a `<run_name>.heartbeat` touched periodically. Validate each step against its gate before proceeding to the next; never stack experiments on unvalidated upstream output. Use `python -u` (unbuffered) for any script that emits progress.
- **Bug tracking.** `notes/<stage>/<study>/fixes_to_apply.md` is for **bugs only** — root cause, fix description, verification result. Do not append "ran cleanly" / activity-log entries; that file is a tracker, not a journal.
- **New methods plug in at one place.** Method dispatch lives in `build_method_compressor` inside `experiments/<stage>/toolkit/per_coord_quantization.py`. A new method adds a branch that produces `forward_map`, `inverse_map`, `sigma_k_diag`, and `weights`, then reuses the existing `_uniform` / `_truncate` / `_waterfill` allocation tail. Keep additions there rather than scattering them across the driver.

## Common commands

```bash
# Activate the project venv once per shell.
# Use the project's uv-managed `.venv/` (Python 3.12 + torch/transformers/etc.).
# Do NOT use conda — the historical `kv-rd` env no longer exists, and other
# conda envs do not have the project's deps.
source .venv/bin/activate
# Or, for one-shot invocations without sourcing:
#     ./.venv/bin/python -m experiments.<stage>.run_<study> ...

# Run a single phase of a stage driver
python -m experiments.<stage>.run_<study> \
    --phase <e3|e4a|e4b|e5> \
    --b-avg 3.0 --rank 64 \
    --methods <comma-separated method names> \
    --output-subdir <subdir>

# Parallel launcher across the user's GPU pool
bash experiments/<stage>/scripts/launch_<study>.sh \
    --phase <phase> --gpus 0,1,2,3

# Run a gate after a study
python -m experiments.<stage>.gates.gate_<phase>
```

Substitute `<stage>` and `<study>` for the active stage you're working in (currently `stage1` / `cca_vs_waterfill_study`; future stages will live under `stage2/`, etc.).

## Code style

- **Comments only when the *why* is non-obvious.** Hidden constraints, workarounds, surprising invariants — yes. Restating what well-named code does — no. Don't add docstrings that just enumerate parameter types.
- **Trust internal contracts.** Don't add error handling or fallbacks for scenarios that can't happen. Validate at system boundaries (user input, external APIs); inside the toolkit, internal callers are trusted.
- **Edit before creating.** Prefer extending an existing file to introducing a new one. New files are justified when the work doesn't fit anywhere existing.
- **Match existing patterns.** Method-dispatch through `build_method_compressor`, gate-style sanity checks, log/heartbeat conventions in launcher scripts, `*_summary.json` + `*_rows.pt` artifact pairing.

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
| Trace formula | The Q-weighted reconstruction-MSE weight `diag(M^T Σ_Q M)` where `M` is the inverse of the basis forward map. Distinct from canonical correlation `ρ²`; conflating them was a pre-fix bug class (see Stage 1's `fixes_to_apply.md`). |
| LOO | Leave-one-out cross-validation across calibration examples. |
| GQA | Grouped-Query Attention; queries within a kv-head's group are pooled when forming `Σ_Q` and `C_QK`. |
| Layer-0 excluded | Headline aggregations drop layer 0 because of its anomalous attention-sink behaviour. |
