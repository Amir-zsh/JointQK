# plan5 (pre-registration): pgq5 — Qwen3-8B port + long-generation (thinking-mode) survival study

**Registered:** 2026-07-10, branch `pgq5` (cut from `pgq4` @ 329078b).
**Status: REGISTERED.** Format constants frozen verbatim from pgq4
(`frozen_choices_pgq4.json`); Qwen statistics re-derived; freezes F-B / F-C
before each F1 wave. Calibration data pulled from lambda6 (10.137.32.78,
pull-only — its /vault is 100% full); stats source =
`qpca_qwen3_8b_longbench_compact8_n400.pt` (verified sigma_q/sigma_k
[36,8,128,128]; compact9_n450 fallback pulled too).

## Context

pgq4 established on Llama-3.1-8B that `pgq_proflmrw` (water-filled per-coord width
profiles + LM grids + qpca_unc + sink escape + recency window) ties EC at 2 b/c and
beats rvq SIG — but ALL evidence is prefill-side, single-model. The user's two
directives: move to Qwen3-8B, and confront the long-generation concern (MATH-style
tasks with thinking enabled: the cache becomes dominated by the model's OWN generated
tokens, compressed closed-loop — the regime where quantizers die; OSCAR's token-spam
comment is the documented failure class). pgq5 answers both:

1. **Transfer:** is the format model-generic (per-model statistics refit only), or
   was pgq4 a Llama fit?
2. **Long generation (new science):** does the codec hold when decode tokens are
   compressed on-line via Mode-B'? C0 separates DOMAIN (open-loop OOD diagnostics on
   trace keys) from C1's MECHANISM (closed-loop compounding), so failures are
   attributable.

Exploration facts this rests on: the Qwen calibration data survives on
**lambda6 (10.137.32.78, key-based ssh verified; NOT the stale /etc/hosts IP)**:
512 GB 01_raw (16 TRAIN rows / 102 test), 42 GB 02_stats, and the three
"deleted" basis bundles (jointqk_qwen3_8b_compact9_n450.pt,
qpca_qwen3_8b_longbench_compact8_n400.pt, v_stats_longbench_compact8_n400.pt).
lambda6's /vault is 100% FULL — strictly pull-only rsync, never write there.
math500/aime25 already registered with boxed scorers (schema unverified —
pre-flight); `enable_thinking` exists in the kvpress pipeline but is unreachable
from the runner (3-line vendored wiring); Mode-B' has a latent `_qlen` staleness
bug (kvpress `_remove_answer_from_cache` crops between questions → stale
bookkeeping silently leaves tokens fp16); math cells cost 1.5-4+ GPU-h each
(decode-dominated) so C1 is capped and timing-smoked. Existing Qwen bench cells
(FP/TQ, 230 cells) are reusable references — no rvq/ecu exists on Qwen, so the
admission gate becomes ratio-transfer vs a TQ proxy measured identically on both
models.

## Step 0 — Branch + registration

`git checkout -b pgq5` from pgq4 (clean). Write `notes/page_quant/studies/plan5.md`
(content = this design). Study log `logs/pgq5_study.log` + heartbeat. Commits at
study end need separate explicit approval.

## What transfers frozen vs re-derived (registered)

- **Frozen format constants (verbatim):** ptok=64; NBLK=4 (32-coord blocks); ladder
  {0,2,3,4,6}; ≤8 profiles / 3-bit ids; targets [0,128,192,224,256,288,320,768];
  LM unit-Gaussian cents (model-independent); gain=false; qpca_unc per-head basis;
  sink escape positions 0-3 8-bit; rw = last 4 prompt pages top rung; layer-0 fp16;
  V=TurboQuant 2b; `_paged_lambda_assign` + snap-aware D.
- **Re-derived Qwen statistics (never choices):** μ/μ_q, Σ_Q/Σ_K → forward/inverse
  [36,8,128,128], code_std, alphas, sink_scale, profile tuples (A2 procedure
  re-executed).
- **Re-tested rules (rule frozen, outcome model-dependent):** per-layer sharing iff
  penalty <3%; Qwen sink-position check (sink mass at positions 0-3 — if NOT,
  stop-and-amend: the position-derivable escape is a format assumption).
- **Dropped for Qwen (registered):** ω arms (tied rdo on Llama F1; needs omega refit
  with no question attached); 3-basis ablation (closed by pgq4); ecu-on-Qwen moved
  to conditional B2.

## Pre-flight P-1 (CPU, blocks everything)

1. math500/aime25 schema: load HF datasets, confirm
   context/question/answer_prefix/max_new_tokens/answer; render one row through
   `pipeline.preprocess` with enable_thinking=True (no empty <think></think>);
   hand-check the boxed scorer (known-naive nested-brace truncation — kept, uniform
   across arms; robust secondary extraction computed in analysis outside vendor).
2. Qwen reference numbers from existing cells → `pgq5_refs.json` BEFORE any F1:
   FP 5-task mean, turboquant_k2_v2 (matched-V incumbent) + k2_v3 secondary.
   Verify v7-compat (fraction 1.0, layer-0 fp16, exclude-train).
3. math500 registered split (seed 20260710): diag=60 (C0/C2), eval=200 (C1 only),
   spare=240. aime25 (30 rows) smoke-grade.

## P-1 findings (2026-07-10, pre-data amendments)

- **Refs extracted** (`artifacts/page_quant2/pgq5_refs.json`, from the May
  downstream_v7 Qwen sweep — fraction 1.0, layer-0 fp16, compact8 exclude-train,
  500 rows/task): FP mean5 = **50.70**, turboquant_k2_v2 = **42.13**
  (FP-retention **0.831** — far below TQ-on-Llama's 0.945; Qwen keys are harder
  for the incumbent), k2_v3 = 43.66.
- **math500 schema**: 500 rows, empty `context`, question in `question`,
  max_new_tokens=4096 native. aime25: 30 rows, max_new_tokens=32000 (smoke only).
  Empty context ⇒ Mode-A control compresses ~nothing (pure harness anchor);
  ALL compression signal in C1 is decode-side — as designed.
- **Thinking template verified**: enable_thinking=False injects empty
  `<think></think>`; True leaves it open. evaluate.py wiring needed as planned.
- **AMENDMENT P1-a (scorer, pre-data):** vendor boxed extractor truncates at the
  first `}`; 20.6% of math500 answers contain `}` ⇒ naive accuracy is capped at
  ~79% even for a perfect model, below the 85% FP-health bar. Registered fix:
  **primary metric = robust balanced-brace boxed extraction computed in analysis
  (outside vendor); vendor-naive number reported secondary.** The 85% FP-health
  gate applies to the robust metric. Uniform across arms either way.
- **Split registered**: `artifacts/page_quant2/pgq5_math500_split.json`
  (seed 20260710; diag=60 / eval=200 / spare=240).

## Stage A — pull from lambda6 + fit (~1-2 GPU-h, ~115 GB transfer)

1. **Pull (rsync over ssh from 10.137.32.78, read-only on the remote):** the 16
   `*__train.pt` raw rows (73 GB) into
   `artifacts/calibration/longbench_compact8_qkv_qwen3_8b/01_raw/` locally,
   the 42 GB `02_stats`, and the three bundles
   (`jointqk_qwen3_8b_compact9_n450.pt`, `qpca_qwen3_8b_longbench_compact8_n400.pt`
   → artifacts/bases/, `v_stats_longbench_compact8_n400.pt` → artifacts/v_bases/).
   Verify sizes + `torch.load` key check on each bundle (need sigma_q/sigma_k with
   shape [36,8,128,128]; prefer the compact8_n400 bundle for comparability with
   the existing Qwen bench cells, fall back to compact9_n450 — record choice).
2. `pipelines/scripts/make_qwen_fit_split.py` (~60 lines): split the 16 train rows
   → 12 fit + 4 selection (stratified: fit gets 2 rows for 4 tasks + 1 for the
   rest; selection 1 row each of 4 spread tasks incl. repobench-p for the OOD
   slice) → roles.json mirroring `ec_compact8_train_26` format. Selection n=4 is
   thinner than Llama's 8 — noted; gates use medians, and the CONTINGENCY below
   covers under-power.
3. **Model-tag parametrization (~100 lines, Llama defaults byte-identical):**
   `--model-tag {llama31_8b,qwen3_8b}` → path dict in `fit_ec_bundle.py`
   (RUN_ID/RAW_ROOT/ROLES/CCA_STATS/OUT_DIR), `fit_pgq2_bundle.py` (BIGPOOL_*),
   `fit_pgq4_bundle.py`, `phase1_empirics.load_mu_from_fit_stats`,
   `launch_pgq_longbench.sh` (MODEL/CCA/VST/OUT_BASE/LOG_DIR).
4. Fit `pgq5_bundle__qpca_unc__qwen3_8b_*` (qpca_unc only) + sharing-rule test;
   `tests/test_pgq4.py` shape-parametrized to L=36.
5. Held-out gates on selection rows → `pgq5_heldout_report.json` before F1.
6. **Contingency (only if pulled data is unusable or selection under-powered):**
   fresh `capture_raw.py --model Qwen/Qwen3-8B --keep-raw` of up to 32 extra TRAIN
   rows (~4-6 GPU-h, ~150 GB) — the original Stage A, demoted.

## Stage A findings (2026-07-10)

- Pull complete: 16 train raws (73 GB) + 02_stats (42 GB) + 3 bundles; stats
  source = compact8_n400 (verified). 10 stats duplicates across shards
  (re-capture pass) verified content-identical and deduped locally.
- Roles: `pgq5_qwen_compact8_16/roles.json` — 12 fit + 4 selection
  (musique 89, qasper 127, qmsum 154, repobench-p 223).
- μ denominator resolved tokens*group at trace ratio 1.0002 (Llama: 0.9827).
- Bundle `pgq5_bundle__qpca_unc__qwen3_8b_compact8train12.pt` sha8 cd6a3d41.
- **Sharing rule outcome: prof_share = HEAD** (penalty 6.19% > 3%; Llama was
  1.9% → layer). Qwen kv-heads are more profile-heterogeneous; loader
  supports per-head natively. Registered rule, model-dependent outcome — no
  amendment.
- **Sink-position check PASS**: median 11.7% attention mass at positions 0-3
  on selection rows; 1/160 probes with >5% mass beyond position 3 (a content
  token on one musique row, not a structural sink).
- **Bug pgq5-1 fixed** (tracker): Mode-B' `_qlen` staleness across kvpress
  cache crops — clamp to `cache_position[0]`; regression test green.
- Held-out gates (pgq5_heldout_report.json): all PASS. proflm@2.0 rate
  2.0000 / ovf 0 / sinkCE 0.0034 / normR 0.976; proflmrw@2.0 rate 2.0636;
  proflm@2.5 rate 2.4974 / normR 0.988.
- **G-P4' ratio-transfer: strict FAIL, but on the denominator**
  (pgq5_gp4prime_gate.json; identical harness both models, q_cap 8192):
  logit_err proflm@2.0 = 0.002716 (Llama) vs 0.002615 (Qwen) — the codec's
  absolute proxy quality FULLY transferred (Qwen slightly better). TQ-K2 =
  0.026233 (Llama) vs 0.016295 (Qwen) — the PROXY BASELINE improved on Qwen.
  ρ_llama 0.1035, ρ_qwen 0.1605, ratio 1.55 > 1.25 bar. The gate was designed
  to catch numerator degradation; it tripped on denominator improvement.
  Meanwhile TQ's F1 retention is far WORSE on Qwen (0.831 vs 0.945) — the
  logit_err↔F1 exchange rate differs per model. DECISION PENDING user
  discussion before any Stage B cell (user-requested stop-point).
- Fingerprint 262/262 PASS after all edits; pytest 93 passed
  (+ test_pgq4 11/11 incl. new crop-clamp and single-basis-bundle tests).

## Amendments A5-1 / A5-2 and scope reduction (2026-07-10, pre-F1, user-approved)

Project-direction discussion (recorded in plan6.md context when created; new
outline = merged-page codec, phase-symmetric signal policy, phase-asymmetric
signals deferred) reduced pgq5 to the Stage-B transfer question. **C0/C1/C2
(thinking-mode closed-loop) are PARKED as a deferred direction** — the design
below is preserved for a future run; the _qlen fix (pgq5-1) stays landed.

- **A5-1 (G-P4' rescope):** the gate tripped on its DENOMINATOR (TQ proxy
  improved on Qwen: 0.0163 vs 0.0262) while the numerator — the codec itself —
  transferred fully (0.002615 Qwen vs 0.002716 Llama, −3.7%). The gate's
  design intent was to catch numerator degradation. Amended admission rule:
  absolute numerator transfer within 10% ⇒ PASS (measured −3.7%). Both ratio
  numbers reported in report5; denominator sensitivity of ratio gates noted
  as a methods lesson.
- **A5-2 (T1 demoted):** FP-retention ≥0.90 becomes descriptive, not gating —
  the incumbent TQ-K2V2 itself retains only 0.831 on Qwen (0.945 on Llama);
  the logit_err↔F1 exchange rate is model-dependent (TQ has BETTER proxy on
  Qwen yet WORSE retention). **Hard gate = T2: Δ(ours − turboquant_k2_v2)
  row-paired CI-lower > −1.0.**

## Stage B — LongBench transfer (10 cells, ~9-13 GPU-h)

Arms ×5 v7 tasks: `pgq_proflm_rdo@2.0` (no window — does Qwen have an lcc-like
recency gap?) + `pgq_proflmrw_rdo@2.0` (winner). Refs: existing FP/TQ cells (0 GPU-h).
**Freeze F-B:** `frozen_choices_pgq5.json` (bundle sha8, constants, arms, bars)
before the first cell.

**B2 (conditional, ≤10 cells ≤8 GPU-h, only if T1 passes AND spend <45 GPU-h):**
ecu@2.0 ×5 (needs Qwen EC bundle fit) + proflmrw@2.5 ×5.

## Stage C0 — thinking-trace diagnostics (~4-5 GPU-h, ~60-80 GB)

1. `pipelines/thinking/gen_thinking_traces.py` (~120 lines): 50 FP greedy traces
   (enable_thinking=True, max_new_tokens 4096) on diag rows; save text + token ids.
   FP health: cap-hit fraction, 4-gram repetition, boxed rate, accuracy. C0 may
   revise max_new_tokens for C1 BEFORE the F-C freeze (cap-hit evidence).
2. **Concat trick** (decode keys ≡ prefill keys of prompt+trace): raw-token adapter
   for capture_raw (~50 lines, bypasses kvq/data's hardcoded enable_thinking=False)
   → trace-key pools via the existing capture pipeline.
3. `c0_diagnostics.py` (~150 lines), all open-loop on the frozen bundle:
   (a) OOD screen: per-(l,h) top-width clip rate + code_std ratio, trace vs
   LongBench selection keys; (b) codec logit_err on trace vs selection pools;
   (c) **W rule (frozen): W\* = smallest W ∈ {32,128,256} with logit_err ≤ 1.10× of
   W=256** (newest W trace keys held fp16).

## Stage C1 — closed-loop math500 (≤8 cells, capped 38 GPU-h)

**Freeze F-C** (arms, W\*, bars, detectors) before first cell. **Mandatory timing
smoke** (2 arms × 10 rows); registered rule: any arm projecting >6 GPU-h → eval N
200→120 (registered row-id prefix), noted.

Arms (math500 eval rows, greedy, enable_thinking=True):
1. FP (no_press) · 2. Mode-A control (proflmrw@2.0, compress_decode=False —
expected ≈FP by construction; harness sanity + attribution anchor) ·
3. **Mode-B' W\*** (decode_chunk=8) · 4. Mode-B' W=0 (adversarial) ·
5. TQ-K2V2 Mode-B (per-token decode compression baseline) ·
6. KIVI Mode-B (only if ≤4 GPU-h remain).
aime25 smoke: FP + Mode-B' W\* (report-grade, n=30).
Honest rates: full final cache incl. decode fp16-ring charged 16 b/c; decode-token
rate reported separately.

## Stage C2 — conditional calibration-domain ablation (≤2 cells, ≤8 GPU-h)

**Trigger (frozen):** C0 OOD (top-width clip >3× selection on >10% of (l,h), OR
trace logit_err >1.5× selection) OR (C1 arm-3 fails while arm-2 passes). Action:
STATISTICS-ONLY refit (code_std/alphas/profiles) on fit pool + 50 diag traces
(capture-free via concat trick); rerun the failing arm. One iteration.

## Gates & bars (frozen)

- **A gates:** G1 rate ±2%/ovf<1%; G2' sinkCE ≤0.01 (Llama 0.002, 5× margin) + sink
  positions 0-3 confirmed; G3' |normR−1| ≤0.05 (Llama ≤0.023);
  **G-P4' ratio-transfer:** ρ = logit_err(proflm@2.0)/logit_err(TQ-K2 proxy),
  measured identically on both models' selection rows; pass iff ρ_qwen ≤ 1.25·ρ_llama.
  G0-perf ≤1.5× existing Qwen jointqk per-sample time; Mode-B' ≤1.3× own Mode-A.
- **B bars:** T1 FP-retention ≥0.90 (Llama 0.948); T2 Δ vs turboquant_k2_v2
  CI-lower >−1.0; T3 window effect per-task (descriptive). Gate B→C: T1 ≥0.90 go;
  0.85-0.90 go with "degraded transfer" annotation; <0.85 stop (C uninterpretable).
- **C0→C1 gate (FP health):** diag accuracy ≥85% [calibrate against Qwen3-8B public
  math500 numbers at pre-flight], cap-hit ≤15%, boxed ≥95%. Fail → registered
  amendment: seeded sampling (T=0.6/top_p=0.95, shared per-row seed across arms)
  behind a pipeline flag.
- **C1 Tier-G (the "holds up" claim):** Δacc(B'W\* − FP) CI-lower ≥ −4.0 pts AND
  stability pass: cap-hit ≤ FP+10 pts; spam (any 4-gram ≥50 repeats) ≤ FP+5 pts;
  median trace length ≤1.5× FP. Mode-A must be CI ∋ 0 vs FP.

## Decision tree

1. G-P4' fail → no Qwen F1; "format does not transfer at matched proxy quality."
2. T1 + Tier-G pass → headline: model-generic format + decode survival; kernel-port
   evidence upgraded on both axes.
3. Tier-G fail + C0 OOD trigger → C2; C2 pass → "calibration domain must include
   generated text" (actionable positive); C2 fail → mechanism failure (closed-loop
   compounding), telemetry reported, no rescue.
4. W\* ≈ W=0 → failure is codec-level; W\* ≫ W=0 → fp16 ring is load-bearing.
5. TQ-Mode-B beats B' SIG → adaptivity gradient extends to decode; per-token grid
   scale becomes pgq6's registered priority.
6. Mode-A fails vs FP → halt, debug harness.

## Pre-registered contrasts (row-paired 10k bootstraps)

B: proflmrw−FP, proflmrw−tq_k2_v2, rw−plain; per-task + mean + non-lcc.
C1 (0/1 correctness): B'(W\*)−FP; B'(W\*)−ModeA; B'(W\*)−B'(0); B'(W\*)−TQ-ModeB;
ModeA−FP. Secondary robust-boxed accuracy. aime25 point estimates only.

## New code inventory

| item | files | size |
|---|---|---|
| model-tag parametrization | fit_ec_bundle / fit_pgq2_bundle / fit_pgq4_bundle / phase1_empirics / launch_pgq_longbench.sh | ~100 lines |
| Qwen roles split | pipelines/scripts/make_qwen_fit_split.py | ~60 |
| **_qlen bug fix** (→ fixes_to_apply.md) | jointqk_press.py decode branch: clamp qlen to cache_position (kvpress crops cache between questions; stale _qlen silently leaves tokens fp16 = quality cheating) | ~4 + ~60 tests |
| **vendored edit (explicit)** | vendor/kvpress/evaluation/evaluate.py: EvaluationConfig.enable_thinking + thread into both _run_inference pipeline calls | 3 lines |
| vendored edit (amendment-only) | pipeline.py generate_answer seeded-sampling flag — only if C0→C1 gate fails | ~15 |
| TQ open-loop proxy scorer | fit_pgq4_bundle --eval-modes tq:2 via vendor MSECompressor | ~80 |
| C0 tooling | gen_thinking_traces.py + raw-token capture adapter + c0_diagnostics.py | ~320 |
| C1 glue | work-item JSONLs, decode-flag plumbing, accuracy adapter for bootstrap tool | ~100 |

## Budget

A 1-2 (pull, was 7-9 capture) + B 9-13 + B2 ≤8 + C0 4-5 + C1 ≤38 + C2 ≤8 →
expected 35-60 GPU-h, **hard cap 80**, GPUs 0-3, disk ~200 GB local
(37 TB free — pass). lambda6 is pull-only (its /vault is at 100%).

## Top risks

- **R1 greedy × thinking instability** (Qwen card warns of loops): C0 health-checks
  FP before any compressed arm; frozen amendment path to seeded sampling; worst
  case C closes as unmeasurable at ≤5 GPU-h spent.
- **R2 C1 cost (4× spread):** eager per-token decode loop never timed at scale →
  mandatory timing smoke + 6 GPU-h/cell cap + registered N-reduction rule.
- **R3 OOD trace keys vs static calibration (the science risk):** strongest shift
  yet + closed-loop compounding; the Mode-A/W=0/TQ arm matrix + C0 diagnostics +
  C2 stats-only refit guarantee attributability — a clean negative is acceptable,
  an unattributable one is not.

## Verification

- Pre-flight P-1 gates Stage C; Stage A gates B; T1 gates C; F-B/F-C freezes before
  each F1 wave; timing smoke before C1.
- pytest (incl. new _qlen/crop tests + L=36 parametrization) + fingerprint 262/262
  before GPU spend and before commits.
- All claims bootstrap-backed; report5.md before commit proposal (separate approval).
