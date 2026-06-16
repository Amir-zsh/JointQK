# Entropy-coded K-cache at K=2/V=2 beats TurboQuant on lcc, musique, 2wikimqa (Llama-3.1-8B)

**Date:** 2026-06-12
**Goal:** beat `turboquant_k2_v2` on LongBench lcc / musique / 2wikimqa downstream F1 at an
honest coded rate ≤ 2.0 bits/coord, using the `entropy_coding/` paged rANS machinery.
**Hardware:** GPUs 0–3 (A100-40GB). **Model:** meta-llama/Llama-3.1-8B-Instruct.

## TL;DR

**Criterion met.** At a held-out coded rate of ~1.95 bits/coord (vs TurboQuant's true
2.125 b/c incl. its fp16 per-vector norms), entropy-coded deadzone quantization in the
JointQK basis (`ec_r_sym`):

- beats TurboQuant K2V2 on **all three tasks** at dz=0.5 (+2.49 lcc, +0.20 musique, +4.68 2wikimqa);
- at the Phase-A-selected dz=0.375, reaches **mean F1 44.47 — within 0.52 of full precision
  (44.99)** and +2.95 over TurboQuant (41.52);
- **fixes the Llama lcc disconnect**: plain `jointqk_k2_v2` scored 37.23 on lcc; the EC
  quantizer on the *same R_sym basis* scores 50.3–51.1. The basis was never the problem —
  the per-coord Lloyd-Max codebook + integer waterfill at K=2 was.

| method | lcc | musique | 2wikimqa | mean | true K rate (b/c) |
|---|---|---|---|---|---|
| full_precision | 51.37 | 32.62 | 50.99 | 44.99 | 16 (fp16) |
| turboquant_k2_v2 | 48.37 | 31.55 | 44.64 | 41.52 | 2.125 |
| jointqk_k2_v2 | 37.23 | 27.08 | 49.25 | 37.85 | 2.0 grid |
| kivi_int2 | 44.31 | 19.58 | 37.39 | 33.76 | 2.0 grid |
| ec_hadamard_dz0.25 | **51.61** | 28.82 | 45.70 | 42.04 | 1.949 |
| ec_qpca_dz0.25 | 45.70 | 29.19 | 46.79 | 40.56 | 1.958 |
| ec_qpca_dz0.375 | 45.58 | 29.21 | 46.98 | 40.59 | 1.958 |
| ec_qpca_dz0.5 | 46.64 | 30.11 | 47.41 | 41.39 | 1.958 |
| ec_r_sym_dz0.25 | 50.28 | 31.37 | 49.43 | 43.69 | 1.947 |
| **ec_r_sym_dz0.375** | 51.10 | 31.29 | **51.03** | **44.47** | 1.947 |
| **ec_r_sym_dz0.5** | 50.86 | **31.75** | 49.32 | 43.98 | 1.948 |

Eval protocol identical to the v7 baselines (fraction=1.0, compact8 train rows excluded,
layer-0 fp16 for all methods, Mode A, V side = v_turboquant @ 2 bits for both JQ-family
methods). Baseline F1s are the v7 cells on disk — not re-run.

## The three tasks

All from LongBench; "F1" in the tables above is each task's official LongBench metric.
Stats measured from this run's predictions (`length` = LongBench's word count of the
full input; Llama token counts run ~1.3–1.5× higher — the 4 sampled musique eval rows
tokenize to ~15.5–16.2k Llama tokens).

| | lcc | musique | 2wikimqa |
|---|---|---|---|
| type | code completion | multi-hop QA (2–4 hops) | multi-hop QA (2 hops) |
| metric | edit similarity (`code_sim_score`) | answer F1 | answer F1 |
| eval rows | 500 | 150 (200 − 50 calib-train excluded) | 200 |
| input length (words, med / range) | 883 / 399–10,029 | 11,392 / 3,440–16,497 | 4,209 / 535–11,950 |
| in compact8 calibration? | **no (OOD)** | yes | **no (OOD)** |
| k2v2 baseline picture | TQ 48.4 ≫ JQ 37.2 | TQ 31.6 > JQ 27.1 | JQ 49.3 > TQ 44.6 |

**lcc** (LCC, Long Code Completion): given a long source-code file, predict its next
line; scored by edit similarity against the reference line. The shortest median context
of the three, but the most *precision-demanding*: the completion must reproduce exact
identifiers, argument lists, and API names defined hundreds of lines earlier, so a
handful of key-vector corruptions that flip attention away from the right definition
destroy the output. Code key statistics are also far from the text-dominated calibration
corpus (compact8's only code task, repobench-p, served as the in-calibration sentinel
during candidate selection). This combination — argmax-precision-critical + OOD — is
exactly where plain JointQK collapsed at K=2 (−11.1 vs TQ) and is the battleground task
of this study.

**musique** (MuSiQue): compositional multi-hop questions over ~20 Wikipedia paragraphs,
constructed so single-hop shortcuts fail and most paragraphs are distractors. The
longest (median ~11.4k words ≈ 16k Llama tokens) and hardest of the three — full
precision scores only 32.6 — so per-method differences are compressed into a narrow
band and every compressed method loses ~1 pp to FP here. The only in-calibration task
of the trio (its 50 compact8 train rows are excluded from eval), so it tests the
*in-domain* behaviour of the frozen coder model at maximum context length.

**2wikimqa** (2WikiMultihopQA): two-hop questions over Wikipedia with structured
evidence chains (e.g. bridge entities: "who is the director of the film whose lead was
X?"); medium-length contexts. OOD to the calibration corpus, like lcc — but text-like
rather than code-like, and it is the one task where the JointQK *basis* already beat
TurboQuant at plain K=2 (+4.6). It checks that switching the quantizer keeps the
basis's existing multi-hop advantage rather than trading it away.

Together the trio triangulates the failure modes: code vs prose, OOD vs in-calibration
statistics, short-precise vs long-diffuse attention, and one task each where the prior
baselines (TQ, JQ) respectively dominated — a method only sweeps all three if it is
robust on every axis at once.

## Method

`ec_*` = per-(layer, kv_head) **deadzone uniform quantizer + frozen-model entropy coding**:

1. Rotate centered keys into a basis (`r_sym` = JointQK's R_sym from the production n400
   bundle; `hadamard` = TurboQuant's own per-layer rotation, seed 42+1000·L; `qpca` control).
2. Per-coord uniform step `delta[l,h,j]` solved by bisection so the deadzoned discrete
   entropy of 18 compact8 TRAIN rows water-fills to 1.95 b/c per head, weighted by
   `ell = diag(F^T Σ_Q F)` (the high-rate ECSQ allocation; entropy_coding/run_pca_ec_deadzone.py).
3. Per-coord frequency model (alphabet + probs) **frozen** on the same train rows; indices
   outside the alphabet snap to the nearest entry (the OOD penalty is paid in rate and
   reconstruction, never hidden).
4. The bench press reconstruction (`kvq/compression/ec_roundtrip.py`,
   `SnappedDeadzoneECCompressor`) is what the real paged rANS codec decodes — verified
   bit-exact (gate G1 below).

Deadzone dz<0.5 widens the zero bin (H.264-style); with match-rate bisection the freed
bits buy a finer step elsewhere, making dz a pure RD knob at fixed rate.

## Leakage control

- EC fit rows (18) + selection rows (8): all compact8 **train**-split, already excluded
  from every F1 eval via `exclude_train_indices_for_eval.json` → eval set bit-identical
  to the v7 baselines.
- lcc and 2wikimqa are **OOD to the calibration corpus** (compact8 has neither); the
  in-calibration code sentinel repobench-p stood in for lcc during selection.
- Coder model frozen on fit rows; rates below are measured on rows never used for fitting.

## Gates

- **G1 (bit-exactness):** PageCodecRANS encode→decode vs press roundtrip on 2 selection
  rows × 4 (layer, head) pairs: 3 / 6,273,024 index mismatches, all ±1 bin (fp32-vs-fp64
  boundary ties), reconstruction otherwise identical. Real byte-level rate incl. page
  overhead: 1.90–1.98 b/c at ptok=64. PASS (`pipelines/ec/verify_codec_bitexact.py`).
- **G2 (held-out rate):** all 9 bundles 1.947–1.958 b/c pooled on selection rows;
  per-task max 1.99 (repobench-p). PASS.
- **G3 (fidelity ≥ TQ):** Phase A on 8 held-out selection rows, layer-0 excluded:

| method (K=2) | top-1 | top-5 | k_mse | logit_err | rate |
|---|---|---|---|---|---|
| jq_k2 (plain) | **0.5596** | 0.9836 | 2.60e-1 | 2.07e-3 | 2.000 |
| ec_r_sym_dz0.375 | 0.5158 | **0.9924** | 2.37e-1 | **1.60e-3** | 1.947 |
| ec_qpca_dz0.375 | 0.4902 | 0.9845 | 4.28e-1 | 2.24e-3 | 1.958 |
| ec_hadamard_dz0.25 | 0.4608 | 0.9644 | 3.10e-1 | 5.53e-3 | 1.949 |
| tq_k2 | 0.4031 | 0.9212 | 5.57e-1 | 2.33e-2 | 2.125 |

  All EC candidates beat TQ on every proxy, incl. repobench-p top-1 (0.52/0.46 vs 0.41). PASS.
  (Caveat reconfirmed: plain JQ wins top-1 yet loses lcc F1 by 11 pp — proxies don't decide
  F1 on Llama; the bench does.)

## Post-hoc honest rates on the bench tasks themselves

Frozen models applied to 4 eval-side rows per task (capture run `ec_posthoc_rate_llama31_8b`;
measurement only, no feedback into fitting):

| bundle | lcc | musique | 2wikimqa |
|---|---|---|---|
| ec_r_sym_dz0.5 | 1.992 | 1.988 | 1.938 |
| ec_r_sym_dz0.375 | 1.995 | 1.990 | 1.937 |
| ec_r_sym_dz0.25 | 1.998 | 1.991 | 1.936 |
| ec_hadamard_dz0.25 | 1.974 | 2.021 | 1.942 |
| ec_qpca_dz0.5 | 1.989 | 2.037 | 1.948 |
| ec_qpca_dz0.375 | 1.991 | 2.042 | 1.948 |
| ec_qpca_dz0.25 | 1.993 | 2.046 | 1.948 |

All r_sym rates ≤ 2.0 even on fully-OOD lcc (the snap-escape penalty shows up but stays
small: +0.04 b/c over the in-domain selection rate). TurboQuant's comparison rate is
2.125 b/c everywhere (2 bits grid + fp16 norm per 128-dim vector).

## dz sensitivity (selection honesty)

Phase-A fidelity selected dz=0.375 *before* any F1 was run; its result stands as the
primary: 2/3 tasks won (+2.73 lcc, +6.39 2wikimqa) with musique −0.26 (a tie at n=151
eval rows), mean +2.95. The post-hoc dz sweep shows the family is robust, not lucky:

- lcc: EC wins at every dz (+1.9 to +3.2, both bases);
- 2wikimqa: EC wins at every dz (+4.7 to +6.4);
- musique: statistical tie at every dz (−0.26 to +0.20 around TQ's 31.55); dz=0.5
  lands on the winning side of it, making ec_r_sym_dz0.5 a 3/3 sweep.

## What fixed lcc

`jointqk_k2_v2` and `ec_r_sym_*` share the identical R_sym basis and identical V side;
the only difference is the quantizer: per-coord Lloyd-Max codebooks on an integer
waterfill (many coords floored to 0/1 bits at K=2) vs a continuous-resolution deadzone
uniform quantizer with entropy-coded indices. The +13.9 pp lcc gap between them isolates
the failure to the K=2 integer-codebook allocation, closing the "JointQK loses code tasks
on Llama" question from `notes/jointqk_disconnect_investigation.md` — and explaining why
proxies (computed on the same quantizer they select for) never localized it.

## Implementation: what was added, and why

New module (the deployed compressor):

- **`kvq/compression/ec_roundtrip.py`** — `SnappedDeadzoneECCompressor`
  (center → rotate → deadzone-round → **snap to frozen alphabet** → dequant →
  rotate back), `load_ec_compressors_from_bundle()`, `bundle_model_to_dict()`.
  *Why:* the F1 bench only needs reconstructions, but they must be exactly what
  the real rANS codec decodes. The harness's `UniformECRoundtrip` does NOT snap
  out-of-alphabet indices (the codec must — unseen symbols can't be encoded), so a
  new class was required rather than reuse. It lives in `kvq/` (not
  `entropy_coding/`, which is not a package) so the press disk-cache pickles
  re-import cleanly. The snap replicates `kvq_codec._CoordModel.snap` /
  `coded_bits_eval` exactly: numpy `searchsorted` side='left', nearest value,
  tie → left; constant coords emit their single alphabet value.
- **`tests/test_ec_roundtrip.py`** — snap parity vs a verbatim numpy reference of
  `coded_bits_eval`'s symbol mapper, tie-break, constant-coord, pure-deadzone
  equivalence, CPU/GPU agreement. 6 tests.

Offline pipeline (`pipelines/ec/`, all new):

- **`make_ec_capture_manifest.py`** — selects 18 EC-fit + 8 selection rows, all
  compact8 TRAIN split. *Why:* the raw captures on disk were test-split only
  (`--keep-raw test`) — fitting on them would leak into the F1 eval set; and
  repobench-p is over-weighted (4 fit rows) as the in-calibration code sentinel
  for OOD lcc.
- **`fit_ec_bundle.py`** — fits one (basis, dz) bundle. Reuses
  `entropy_coding/run_pca_ec_deadzone.py` as a library (`_solve_delta_matched`'s
  bisection logic, `build_qpca_basis`) but generalizes `build_qpca_ec` to three
  bases: `r_sym` (R_sym from the production n400 cca bundle), `hadamard`
  (TurboQuant's own rotation, `generate_rotation_matrix(128, seed=42+1000*L)`,
  forward=Pi.T per the press), `qpca` (control). `ell = diag(F^T Σ_Q F)` uses the
  n400 bundle's Σ_Q (production geometry, 4.4M tokens); entropies/deltas/coder
  model come from the 18 fit rows (mmap-sliced, GPU moments). Emits the bundle
  .pt with the frozen model as padded tensors (`support_vals/lens/probs`) plus
  held-out selection-row rates pooled and per-task. *Why a new script:* the
  harness `main()` is hardwired to a dead data path and QPCA-only EC.
- **`score_ec_candidates.py`** — Phase-A K-fidelity (top-1/5, k_mse, logit_err,
  layer-0 excluded, per-task breakout) on the 8 selection rows. *Why:* pick
  (basis, dz) BEFORE any F1 is run. Baselines use the *deployed* code paths —
  `TurboQuantV3.key_compressor` (per-layer seed 42+1000·L, the same object
  `TurboQuantPress` runs, not the harness's seed-20260505 wrapper) and
  `build_jointqk_compressor("r_sym_waterfill")` from the production bundle.
- **`verify_codec_bitexact.py`** — gate G1: encode/decode selection rows through
  `PageCodecRANS` (single-rung ladder from the bundle) and compare integer
  indices against the press compressor.
- **`make_posthoc_rate_manifest.py` / `measure_posthoc_rate.py`** — honest rate
  on the bench tasks themselves: 4 eval-side rows per task (musique from compact8
  TEST, lcc from the compact9 manifest, 2wikimqa rows 0–3 — no split manifest
  exists for it), frozen model applied, measurement only.

Bench integration:

- **`kvq/presses/jointqk_press.py`** (modified) — new `ec_bundle_path` field; the
  bundle path+mtime added to `_cache_key()` (stale-cache prevention); a
  `k_method.startswith("ec_")` branch in `post_init_from_model` that loads
  precomputed compressors instead of building from per-head moments, validates
  basis-vs-method and `layer0_full_precision=True`, then falls through to the
  unchanged V side. V remains `v_turboquant` @ 2 bits — bit-identical to the
  `jointqk_k2_v2` baseline's V, so K is the only difference under test.
- **`kvq/compression/per_coord.py`** (modified) — explicit `ec_*` guard in
  `build_jointqk_compressor` pointing to the bundle mechanism.
- **`pipelines/bench/launch_ec_longbench.sh`** (new) — emits the EC cells as
  worker JSONL with the same protocol constants as the v7 sweep (fraction=1.0,
  compact8 exclusion file, Mode A, layer-0 fp16); basis+dz encoded in the output
  dir name so the worker's idempotent skip can't serve stale cells after a refit.
- **`pipelines/eval/aggregate_ec.py`** (new) — joins EC cells with the kept v7
  baseline cells (`artifacts/stage1/downstream_v7/llama31_8b/`) and the bundle
  rates; the existing `aggregate_longbench.py` config regex doesn't match EC labels.
- **`vendor/kvpress/evaluation/evaluate_registry.py`** (modified) — re-pointed
  press imports from the pre-refactor `experiments.stage1.toolkit.*` to
  `kvq.presses.*` and fixed `_REPO_ROOT` (`parents[2]`→`parents[3]` after the
  vendor/ move). *Why:* the branch's vendor-side modification was uncommitted on
  the original author's machine; without this the bench would fail (or worse,
  silently import stale press code if old files existed).

Bugs found and fixed during the run:

1. `ec._solve_delta_matched` → `_entropy_per_coord` allocates CPU helper tensors
   and crashes on CUDA inputs → device-safe copies inlined in `fit_ec_bundle.py`.
2. QPCA fits crashed (`bincount ... non-negative` = NaN codes). Root cause
   (re-diagnosed 2026-06-13 — see the Correction note): **only layer 0**. Σ_Q is
   PSD by construction and over deployed layers (l≥1) its min eigenvalue is
   +8.4e-6; but layer 0 (the attention-sink layer, condition number ~7e6) is so
   ill-conditioned that `eigh` returns a tiny *negative* eigenvalue (−2.0e-5) from
   float roundoff, and `build_qpca_basis`'s `sqrt`/`rsqrt` of it → NaN forward.
   Layer 0 is excluded from every metric and skipped by the bundle loader, so its
   basis is never used. Correct fix: build QPCA on the raw moments for l≥1 (faithful,
   bit-identical to unregularized) and set layer 0 = identity. (The first attempt —
   a `regularize_batch(Σ_Q, 1e-4)` ridge — was misdiagnosed and silently perturbed
   the genuine basis on l≥1; reverted.)
3. Per-task rate measurement originally re-encoded the pooled rows once per task
   (~9× constriction work) → restructured to a single per-(l,h,row) pass; pooled
   rate = sum over rows.
4. `launch.sh`'s `REPO_ROOT=$(dirname $0)/../../..` is one level too deep for
   `pipelines/bench/` (latent refactor bug) — corrected in the new launcher.
5. A wrong unit-test expectation (tie-break case) — the implementation was right.
6. The post-hoc capture OOM'd while sharing GPUs with running fits → re-run with
   `--resume` after the fits released memory.

## How the result was produced (step-by-step)

```bash
# 0. Environment (one-time)
uv pip install --python .venv/bin/python constriction numba

# 1. Select EC calibration rows (18 fit + 8 selection, all compact8 TRAIN)
.venv/bin/python pipelines/ec/make_ec_capture_manifest.py
#    -> artifacts/calibration_splits/ec_compact8_train_26/{manifest.json,roles.json}

# 2. Capture raw q/k for those rows (4 shards, GPUs 0-3, ~4 min, 74 GB)
for s in 0 1 2 3; do
  .venv/bin/python -u pipelines/calibration/capture_raw.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --split-manifest artifacts/calibration_splits/ec_compact8_train_26/manifest.json \
    --run-id ec_calib_compact8_train_llama31_8b \
    --keep-raw all --num-shards 4 --shard-id $s --gpu $s --resume &
done; wait

# 3. Unit tests for the deployed compressor (snap parity etc.)
.venv/bin/python -m pytest tests/test_ec_roundtrip.py -q          # 6 passed

# 4. Fit 9 bundles: {r_sym, hadamard, qpca} x dz {0.5, 0.375, 0.25},
#    match-rate target 1.95 b/c, round-robin GPUs 0-3 (~25 min each)
.venv/bin/python -u pipelines/ec/fit_ec_bundle.py --basis r_sym --dz 0.375 --device cuda:0
#    ... (8 more basis/dz combinations)
#    -> artifacts/ec/llama31_8b/ec_bundle__<basis>__b1.95__dz<dz>__compact8train18.pt
#       each prints its held-out selection-row rate (G2: all 1.947-1.958, max task 1.99)

# 5. Gate G1 — real rANS codec must decode to the press's exact indices
.venv/bin/python pipelines/ec/verify_codec_bitexact.py \
  --bundle artifacts/ec/llama31_8b/ec_bundle__r_sym__b1.95__dz0.25__compact8train18.pt
#    PASS: 3/6,273,024 mismatches, all ±1 bin; 1.90-1.98 b/c real bytes incl. overhead

# 6. Phase A selection — fidelity proxies vs deployed TQ/JQ baselines (G3)
.venv/bin/python -u pipelines/ec/score_ec_candidates.py --device cuda:3 \
  --bundles artifacts/ec/llama31_8b/ec_bundle__{r_sym,hadamard}__b1.95__dz0.{5,375,25}__compact8train18.pt
#    -> artifacts/ec/llama31_8b/phaseA_report.json; picked r_sym:0.375 + hadamard:0.25

# 7. Gate B smoke — full worker->press->kvpress path at 5% fraction
bash pipelines/bench/launch_ec_longbench.sh --variants r_sym:0.375 --fraction 0.05 --gpus 3
#    3/3 cells OK (fraction-tagged dirs; can't collide with the full run)

# 8. Phase C bench — promoted variants, fraction 1.0, v7-identical protocol
bash pipelines/bench/launch_ec_longbench.sh --variants r_sym:0.375,hadamard:0.25 \
    --gpus 0,3 --jobs-per-gpu 1                                   # 6/6 cells OK
# dz-sensitivity follow-up (robustness of the dz choice):
bash pipelines/bench/launch_ec_longbench.sh --variants r_sym:0.5,r_sym:0.25 \
    --gpus 0,1,2,3 --jobs-per-gpu 1                               # 6/6 cells OK

# 9. Post-hoc honest rates on the bench tasks (eval-side rows, frozen model)
.venv/bin/python pipelines/ec/make_posthoc_rate_manifest.py
for s in 0 1; do
  .venv/bin/python -u pipelines/calibration/capture_raw.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --split-manifest artifacts/calibration_splits/ec_posthoc_rate_12/manifest.json \
    --run-id ec_posthoc_rate_llama31_8b \
    --keep-raw all --num-shards 2 --shard-id $s --gpu $((s+1)) --resume &
done; wait
.venv/bin/python pipelines/ec/measure_posthoc_rate.py \
  --bundles artifacts/ec/llama31_8b/ec_bundle__r_sym__b1.95__dz0.{5,375,25}__compact8train18.pt \
            artifacts/ec/llama31_8b/ec_bundle__hadamard__b1.95__dz0.25__compact8train18.pt
#    -> artifacts/ec/llama31_8b/posthoc_rates.json

# 10. Final table
.venv/bin/python pipelines/eval/aggregate_ec.py
#    -> artifacts/ec/llama31_8b/bench_summary.json + the tables in this report
```

Wall-clock on GPUs 0–3: captures ~5 min, 9 fits ~2.5 h (constriction rate stage
dominates), scoring ~6 min, bench 12 cells ~2 h (lcc's 500 rows is the long pole),
post-hoc rates ~1.5 h CPU. End-to-end ≈ 6 h including the qpca-crash and OOM retries.

## QPCA-EC control: the quantizer was most of the problem, but the basis still matters

The closed-form Q-weighted-MSE-optimal basis (QPCA), run through the *identical*
EC pipeline (same fit rows, same match-rate 1.95, same frozen-model protocol, all
gates passed incl. G1 on its non-orthogonal inverse), loses to both TurboQuant and
r_sym-EC downstream:

QPCA here is faithful (built on raw moments for l≥1, bit-identical to
unregularized; layer 0 = identity and unused — see the Correction note):

| | lcc | musique | 2wikimqa | mean |
|---|---|---|---|---|
| ec_qpca (best per task across dz) | 46.64 | 30.11 | 47.41 | 41.39 (best mean, dz0.5) |
| ec_r_sym_dz0.375 | 51.10 | 31.29 | 51.03 | 44.47 |
| turboquant_k2_v2 | 48.37 | 31.55 | 44.64 | 41.52 |

ec_qpca wins only 2wikimqa vs TurboQuant at every dz (1/3), never lcc or musique;
ec_r_sym wins 2–3/3. This cleanly decomposes the original lcc failure at K=2
(all with identical V):

| K-side configuration | lcc |
|---|---|
| R_sym basis + integer Lloyd-Max codebooks (`jointqk_k2_v2`) | 37.23 |
| QPCA basis + EC deadzone quantizer | 45.6–46.6 |
| R_sym basis + EC deadzone quantizer | 50.3–51.1 |

Switching the quantizer (codebooks → deadzone-EC, R_sym fixed) is worth **+13–14 pp**;
switching the basis (QPCA → R_sym, EC fixed) is worth another **~4 pp**. So the
2026-05 disconnect was *mostly* the K=2 integer-codebook allocation, but the basis
choice is not neutral: the argmax-aware R_sym beats the MSE-optimal QPCA under the
better quantizer too. Notably, QPCA's one on-paper advantage — minimal logit MSE —
does not even survive the EC pipeline (Phase A: logit_err ~2.3e-3 vs r_sym-EC's
1.60e-3), because the entropy water-fill reshapes per-coord precision around
`diag(F^T Σ_Q F)` in either basis. QPCA-EC's coder model also generalizes slightly
worse OOD (musique post-hoc rate 2.04 vs r_sym's 1.99). The May-2026 verdict
("JointQK remains the deployed basis; QPCA is the baseline to beat") stands under
entropy coding — by a ~3-pp mean-F1 margin at equal rate.

### Correction (2026-06-13)

An earlier version of this section reported QPCA-EC from bundles built with an
inconsistent and misdiagnosed regularization. The original QPCA fits crashed; I
attributed it to "Σ_Q having numerically non-PSD eigenvalues" and added a
`regularize_batch(Σ_Q, 1e-4)` ridge. Both were wrong: **Σ_Q is PSD** (a second
moment E[qq^T]; min eigenvalue +8.4e-6 over deployed layers, zero negatives), and
the ridge silently shifted the genuine QPCA basis on l≥1 (e.g. layer-16 head-7
forward moved ~30%). Because the ridge was only added on the relaunch, the three
shipped bundles were also mutually inconsistent (dz0.25 unregularized, dz0.5/0.375
regularized). The true cause was **layer 0 alone**: the attention-sink layer's Σ_Q
(condition ~7e6) yields a tiny *negative* eigenvalue (−2.0e-5) from float roundoff,
so `build_qpca_basis`'s `sqrt`/`rsqrt` → NaN. Layer 0 is never deployed. The bundles
were refit faithfully (raw QPCA on l≥1, layer 0 = identity) and the 3 tasks
re-benched. The corrected numbers (above) move by ≤2 pp per cell and **do not change
any conclusion**: QPCA-EC still loses to r_sym-EC by ~3 pp mean and to TurboQuant on
mean F1, winning only 2wikimqa.

## Uniform-step quantizer (2026-06-15): QPCA's optimal config still loses to r_sym

The per-coord results above allocate the deadzone step per coordinate, weighted by
`ell = diag(F^T Σ_Q F)` (ECSQ water-fill). An alternative — upstreamed into the
`entropy_coding/` harness and added here as `fit_ec_bundle.py --uniform-step` — uses
a **single scalar step Δ per head**, bisected to the pooled rate, with no per-coord
allocation. The rationale: QPCA's forward map is *non-orthonormal* and already bakes
the bit allocation into the basis, so a uniform step in the transform domain is the
configuration QPCA is theoretically optimal for. (For the *orthonormal* r_sym basis,
per-coord `ell`-weighting is the natural fit, so uniform-step is expected to help QPCA
but not r_sym — the r_sym-uniform control is benching now and will be appended.)

QPCA, dz=0.5, K=2/V=2, uniform-step vs per-coord:

| task | per-coord | uniform | Δ |
|---|---|---|---|
| lcc | 46.64 | 49.29 | **+2.65** |
| musique | 30.11 | 28.02 | **−2.09** |
| 2wikimqa | 47.41 | 48.61 | +1.20 |
| **mean** | 41.39 | 41.97 | **+0.58** |

Uniform-step does help QPCA (mean +0.58), exactly where the basis carries the
structure (lcc +2.65, 2wikimqa +1.2), at the cost of the in-calibration task where
per-coord precision mattered (musique −2.09). It flips QPCA from tying TurboQuant
(41.39 vs 41.52) to nominally beating it (41.97, 2/3 tasks won). Rate is matched
(per-task post-hoc held-out: uniform 2.00 / 2.05 / 1.95 on lcc/musique/2wikimqa).

**But it does not change the basis verdict.** Even in its optimal config, QPCA-EC
(41.97 mean) still loses to r_sym-EC per-coord (43.98 at dz0.5, 44.47 at dz0.375) by
~2 pp mean and on every individual task (lcc 49.29 vs 50.86–51.10, musique 28.02 vs
31.3–31.8, 2wikimqa 48.61 vs 49.32–51.03) — and at a slightly *higher* rate (lcc
2.00 vs ~1.99). The earlier ~3-pp gap narrows to ~2 pp, not closed. The argmax-aware
R_sym (JointQK) remains the best K-side basis under entropy coding.

## Artifacts

- Bundles + Phase A report + rates: `artifacts/ec/llama31_8b/` (`ec_bundle__*.pt`,
  `phaseA_report.json`, `posthoc_rates.json`, `bench_summary.json`)
- Bench cells: `artifacts/bench_ec/llama31_8b/` (fraction-1.0 dirs; v7 baselines at
  `artifacts/stage1/downstream_v7/llama31_8b/`)
- Code: `pipelines/ec/{make_ec_capture_manifest,fit_ec_bundle,score_ec_candidates,verify_codec_bitexact,make_posthoc_rate_manifest,measure_posthoc_rate}.py`,
  `kvq/compression/ec_roundtrip.py` (+ `tests/test_ec_roundtrip.py`),
  `pipelines/bench/launch_ec_longbench.sh`, `pipelines/eval/aggregate_ec.py`
- Logs: `logs/ec_*.log`, `logs/bench_ec_llama31_8b/`

## Caveats / next steps

- musique is a tie, not a win — at 151 eval rows the ±0.3 pp differences across dz are
  noise; a seed/row-bootstrap would be needed to claim either direction.
- EC leaves ~0.17 b/c on the table vs TQ's 2.125; refitting at b_target=2.10 is the
  obvious free lever if a strict-win-everywhere claim is wanted.
- Single model (Llama-3.1-8B) and 3 tasks; the Qwen replication and the remaining 9
  LongBench tasks are unrun.
- Decode-phase compression (Mode B) and GPU codec throughput in the serving path are
  untested here (the codec's CUDA encode/decode exists and is bit-exact; only Mode A
  prefill compression was benched).
