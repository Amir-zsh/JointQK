# KV Cache Rate-Distortion Research Plan

This project will be executed in stages. The goal is to move from validating the core modeling assumptions to a minimal publishable result, without mixing too many ideas too early.

The main research target is:

**query-aware, token-preserving KV quantization with variable bit allocation**

This is narrower than general KV compression and is the cleanest way to test the proposal.

## Project Status

Stage 1 is now complete.

The main Stage 1 conclusions are:

- query second moments are stable enough to remain a useful modeling object
- the original full-metric oracle path is not a robust win over the V3 baseline
- the harmful component is the anisotropic scaling term, not the oracle eigenbasis by itself
- softened partial-spectrum scaling with `gamma = 0.25` is a robust positive oracle result across `2`, `3`, and `4` bits on the current slice
- the next step is **not** to estimate the failed full-metric path online

So the immediate research direction has changed. The project should build on the backend-compatible `gamma = 0.25` partial-spectrum oracle path, rather than direct full-metric preconditioning.

---

## Core Questions

We will answer the following questions in order:

1. Do query activations actually look Gaussian enough for the Expected Attention modeling assumptions to be useful?
2. If future queries are known, does geometry-aware quantization beat standard TurboQuant at the same average bitrate?
3. If yes, can we recover most of that gain using estimated future-query statistics instead of oracle queries?
4. After keys are understood, do the same ideas help values?
5. After quantization is stable, does the framework naturally extend to pruning by allowing `b = 0`?

---

## Stage Breakdown

## Stage 1: Validate Query Distribution + Oracle Geometry Study

This stage is complete.

Goals:

- test whether query activations are approximately normal
- test whether oracle geometry-aware key quantization beats standard TurboQuant

Deliverables:

- a query-normality report
- an oracle-study report
- reusable code for collecting queries, keys, and future-query statistics

Success criterion:

- clear empirical evidence that query activations are at least approximately Gaussian or approximately Gaussian after simple preprocessing
- at fixed average bitrate, geometry-aware key quantization beats standard TurboQuant on internal distortion metrics and preferably on at least one downstream long-context task

Go / no-go:

- the original full-metric oracle failed this criterion, but the Stage 1E partial-spectrum oracle with `gamma = 0.25` satisfies the internal-metric version across `2`, `3`, and `4` bits

## Stage 2: Estimated Future-Query Geometry

Goals:

- replace oracle future-query statistics with realistic estimators
- compare Gaussian and empirical estimators

Candidate estimators:

- Expected Attention style mean/covariance from a hidden-state or query buffer
- empirical second moment from a small reference-query buffer
- diagonal-only or low-rank approximations

Deliverables:

- oracle vs estimated-query comparison
- ablation on full, diagonal, and low-rank geometry

Success criterion:

- estimated-query geometry retains a substantial fraction of the oracle gain

## Stage 3: Variable Bit Allocation

Goals:

- move from fixed-bit geometry-aware quantization to rate-distortion allocation
- test whether nonuniform precision beats uniform precision at fixed average bitrate

Candidate bit sets:

- `{2, 3, 4, 8}`
- later `{0, 2, 3, 4, 8}` if pruning is enabled

Deliverables:

- per-token or per-head bit allocation experiments
- comparison of geometry-only vs geometry-plus-importance weighting

Success criterion:

- variable-bit allocation improves over fixed-bit geometry-aware quantization

## Stage 4: Value-Side Geometry

Goals:

- extend the framework from keys to values
- test value-side geometry induced by `W_O^T W_O`

Comparisons:

- keys only
- values only
- keys + values

Success criterion:

- either values help materially, or we conclude the main gain is key-side and keep the story focused

## Stage 5: Hybrid Quantization + Pruning

Goals:

- unify pruning and quantization under one rate-distortion objective
- test whether allowing `b = 0` improves the frontier

Deliverables:

- hybrid policy results against pure quantization and pure pruning

Success criterion:

- a clear frontier improvement or a clear negative result with explanation

---

## Working Principles

To keep the project disciplined:

- start with keys only
- start with token-preserving quantization
- use oracle studies before online approximations
- prefer internal distortion metrics first, then downstream benchmarks
- do not mix latent compaction into the first line of work

---

## Reusable Components in This Repo

We already have useful starting points:

- `kvpress/`
  - query-aware presses, Expected Attention implementation, and evaluation harness
- `turboquant-pytorch/`
  - a working TurboQuant-style quantization codebase and MSE-only variants
- [geometry_aware_kv_math.md](/vault/amir/efficient-llm/teamily-project/notes/reference/geometry_aware_kv_math.md:1)
  - derivation of the geometry-aware objectives

This means the project should build on existing code rather than starting from scratch.

---

## Immediate Next Step

The next concrete task is to convert the Stage 1E oracle partial-spectrum result into a realistic method.

That step should:

- use `gamma = 0.25` as the current fixed-bit oracle default
- avoid estimating or deploying the failed `gamma = 1.0` full-metric path
- test whether estimated future-query statistics retain the oracle partial-spectrum gain
- optionally test downstream generation quality for the oracle `gamma = 0.25` path before investing in estimator complexity

Historical Stage 1 planning details remain in [stage1_plan_historical.md](/vault/amir/efficient-llm/teamily-project/notes/stage1/stage1_plan_historical.md:1).
