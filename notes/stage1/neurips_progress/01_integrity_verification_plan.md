# Integrity Verification Plan — TurboQuant + JointQK side by side

Plan only. No existing files modified. Files listed under "to be created"
will live under `experiments/stage1/{tests,scripts,eval}/...` once we agree.

---

## Why this exists

Phase 7's headline is "JointQK ≥ TurboQuant under matched configuration."
That claim is only meaningful if **(a)** our TurboQuant baseline is faithful
to upstream and **(b)** both wrappers are tested under the same protocol.

This plan defines a verification ladder that runs **TurboQuant and JointQK
side-by-side** at every tier. We don't trust either wrapper until both have
passed parity, reproduction-of-published-numbers, and pipeline-integration
checks.

---

## Calibration data — explicit decision

| Method | Calibration data |
|---|---|
| **TurboQuant** | None. Random orthogonal rotation + Lloyd–Max on unit-normalized vectors. Distribution-agnostic by construction. |
| **JointQK** | `artifacts/stage1/query_stats_longbench_under4k/` — 24 LongBench-E examples, sequences ≤ 4K tokens. From this we have `cca_stats.pt` (Σ_Q, Σ_K, R_sym, V_h) and `v_stats.pt` (Σ_V) per (layer, kv_head). Same calibration backing all Phase 7 numbers. |

**For Tier 1.3 (out-of-distribution stress test on upstream's needle prompt):**
we **do not** rebuild calibration. We use the existing LongBench-E `cca_stats`
and `v_stats` and report results with an explicit OOD caveat. This is cheap
(~1 GPU-h) and tells us whether JointQK's gains transfer or depend on
in-distribution calibration. Three possible outcomes (covered in the pass/fail
tree below) all yield publishable framings.

**For Tier 3 (head-to-head):** existing LongBench-E calibration, evaluated on
*held-out* LongBench tasks. Need to spot-check that the 24 calibration examples
are not accidentally a subset of the eval examples — verifiable from
`run_cca_vs_waterfill_study.py` example IDs vs `kvpress` LongBench loader
indices. Add this check to Tier 0.4 below.

**Out of scope for now:** option (iii) — calibrate on a generic corpus (C4 /
OpenWebText). Would cost ~4 GPU-h per model. Only pursued if Tier 1.3 shows
JointQK losing OOD (outcome 3 below).

---

## Upstream's published numbers (the targets to reproduce)

From `turboquant_pytorch/README.md` (corrected 2026-03-30 after their own
`rw=0` bug):

### Synthetic (no model) — paper's theoretical bound

| Bits | Measured MSE | Paper bound | Ratio |
|---|---|---|---|
| 1 | 0.362 | 0.680 | 0.53× |
| 2 | 0.116 | 0.170 | 0.68× |
| 3 | 0.034 | 0.043 | 0.81× |
| 4 | 0.009 | 0.011 | 0.87× |

Source: `turboquant_pytorch/test_turboquant.py::test_mse_quantizer`.

### Attention-score validation (Qwen2.5-3B-Instruct, 8K ctx, 36 layers)

| Config | Compr | cos sim | top-1 | top-5 |
|---|---|---|---|---|
| V3 K4/V2 (rw=0, prot=0) | 5.1× | **0.9996** | **94%** | **97%** |
| V3 K4/V2 + protected_layers=4 | 3.6× | 0.9997 | 99% | 100% |
| V2 3-bit (MSE+QJL) | 5.0× | 0.9945 | 86% | 94% |

Source: `turboquant_pytorch/validate_v3.py`.

### Generation needle ("AURORA-7749", Qwen2.5-3B-Instruct)

| Config | 2K ctx | 4K ctx |
|---|---|---|
| FP16 | EXACT | EXACT |
| K6/V4 rw=128 | EXACT | EXACT |
| K4/V2 rw=0 | MISS | MISS |

Source: `turboquant_pytorch/generation_test_v2.py`.

**Critical reading:** at K4/V2 rw=0 — the bare quantizer config that lines up
with our Phase 7 — upstream gets cos=0.9996 but **MISSES the needle**. So
cosine ≈ 1.0 does not imply task accuracy. This explains why our Phase 7
partial numbers show TurboQuant retaining ~85–95% of fp16 (not 99.96%) even
though upstream's attention scores look near-perfect.

---

## Risk inventory: where each wrapper might drift from upstream

### TurboQuantPress (`experiments/stage1/toolkit/turboquant_press.py`)

| # | Risk | Status |
|---|---|---|
| T1 | **Chunking** at `CHUNK_TOKENS=2048`. Per-vector ops are chunk-invariant; should be byte-exact at `rw=0`. | Tier-0.2 confirms. |
| T2 | **`residual_window` inside chunks** would keep last `rw` of each chunk fp16 instead of last `rw` globally. Currently dormant (we hardcode `rw=0`). | Document; add assert. |
| T3 | **Lazy CPU→GPU move** of `Pi` and `centroids` on first compress. We mutate `tq.{key,val}_compressor.{Pi,centroids,device}` in-place. | Tier-0.2 confirms. |
| T4 | **`protected_layers=0` default.** Upstream's strongest published number uses prot=4. Phase 7 strips this for apples-to-apples. | Documented in paper; not a wrapper bug. |
| T5 | **`compress_decode=True` path** untested — but Phase 6 → Mode A. | Out of scope. |

### JointQKPress (`experiments/stage1/toolkit/jointqk_press.py`)

| # | Risk | Status |
|---|---|---|
| J1 | **Per-(layer, kv_head) compressors** built in `post_init_from_model`. Risk: indexing mismatch between calibration bundle and model layer order. | Existing parity test covers L=5/h=3 only — broaden in Tier-0.3. |
| J2 | **Layer-0 carve-out** (`layer0_full_precision=True`) skips K compression on L=0. TurboQuant compresses all layers. Apples-to-apples comparison must run JointQK with `layer0_full=False` too. | Tier-3 ablates. |
| J3 | **Bit cap at 8 in waterfill**. Internal; should not affect equivalence to offline. | Existing test passes (max-abs-diff = 0.00). |
| J4 | **`compress_decode=True` path** untested. Phase 6 → Mode A. | Out of scope. |
| J5 | **Bundle re-shuffle hazard** (V_h post-rotation in Stage-1E may have reordered heads). | Tier-0.3 confirms across multiple (L, h). |

---

## Verification ladder — both methods, every tier

### Tier 0 — Parity (no model, < 5 min, 0 GPUs)

**Goal:** wrapper outputs match offline reference exactly.

**0.1 Synthetic upstream sanity.** Run `turboquant_pytorch.test_turboquant`
unchanged. Pass: matches upstream's MSE table within ±10%. Confirms vendored
package is intact.

**0.2 TurboQuantPress parity.** New test
`experiments/stage1/tests/test_turboquant_press_parity.py`. For seeded random
`(B, H, S, D)` tensors, build `TurboQuantV3` directly (with
`device=target_device`) and call `compress_kv → decompress_kv`. Build
`TurboQuantPress` and call `press.compress(fake_module, ..., K, V, ...)`.
Assert byte-exact equality.

Cases:
- (1, 8, 512, 128) on CPU — short, no chunking
- (1, 8, 512, 128) on CUDA — short, lazy GPU move triggered
- (1, 8, 4096, 128) on CUDA — triggers chunking (2 chunks)
- (1, 8, 8193, 128) on CUDA — chunking with non-divisible boundary
- (1, 8, 512, 128) with `protected_layers=4`, layer_idx=2 — protected path

Pass: `max_abs_diff == 0` for all cases.

**0.3 JointQKPress parity (broaden).** Existing test covers (L=5, h=3) only.
Add a sweep over (L, h) ∈ {0, 5, 15, 31} × {0, 3, 7} and assert
`max_abs_diff < 1e-5` vs offline `build_method_compressor` and
`build_v_compressor`. Catches J1, J5. New file:
`tests/test_jointqk_press_parity_broad.py`.

**0.4 Calibration / eval split check.** New script
`experiments/stage1/scripts/check_calibration_holdout.py`. Loads the example
IDs from the calibration bundle (`query_stats_longbench_under4k/*.pt`) and
checks that they are **not** present in the LongBench eval datasets used by
kvpress (qasper, narrativeqa, multifieldqa, hotpotqa, 2wikimqa). Reports
overlap count per task. Pass: 0 overlaps; if non-zero, document the affected
examples or filter eval set.

**Output:** `experiments/stage1/logs/integrity/tier0/`.

---

### Tier 1 — Reproduce upstream + first JointQK OOD test (~2 GPU-h)

**Goal:** running upstream's exact validation protocol gives back the README's
numbers (TurboQuant only). Then run our wrappers through the same protocol.

**1.1 Upstream direct reproduction (TurboQuant only).** Run upstream's
`turboquant_pytorch.validate_v3` on Qwen2.5-3B-Instruct unchanged. Capture cos
sim, top-1, top-5 at 2K / 4K / 8K context lengths.

Pass: `V3 K4/V2 rw=0 prot=0` row reports cos ≥ 0.999 and top-1 ≥ 90% at 8K.

**1.2 TurboQuantPress through the same protocol (TurboQuant only).** New
script `experiments/stage1/scripts/run_attention_parity.py`. Loads Qwen2.5-3B
same way, captures KV the same way, but compresses each layer's KV via
`TurboQuantPress.compress` instead of direct `TurboQuantV3.compress_kv`.
Computes the same metrics.

Pass: numbers match Test 1.1 within (cos: ±1e-3, top-1: ±2pp, top-5: ±2pp).

**1.3 OOD attention-score test on Qwen3-8B — both methods.** Same script
parameterized: load Qwen3-8B, run upstream's needle prompt at 2K/4K/8K,
capture KV, compress per layer with both methods. JointQK uses our existing
LongBench-E calibration (out-of-distribution for this prompt). TurboQuant
uses no calibration (matches upstream's protocol).

Configs: K∈{2,3,4} × V=2 × rw=0 × prot=0 × {TurboQuant, JointQK with
layer0_full ∈ {True, False}}. ~9 configs × 3 contexts = 27 metric rows.

Output: a CSV with columns `(method, K, V, layer0_full, ctx, cos, top1, top5)`.
This is the **first apples-to-apples attention-score comparison** of JointQK
vs TurboQuant, evaluated OOD for JointQK.

Pass criteria — three pre-registered outcomes:
- **Outcome 1 (clean win):** at K=2, JointQK cos > TurboQuant cos by ≥ 1e-3
  AND top-1 > by ≥ 5pp on at least 2 of 3 contexts. Headline strengthens.
- **Outcome 2 (parity):** within (1e-3, 5pp). Headline stays "wins on
  in-distribution; ties OOD" — a nuanced but honest framing.
- **Outcome 3 (loss):** JointQK loses to TurboQuant by ≥ 1pp top-1.
  → Trigger option (iii): build a generic-corpus calibration before Tier 3.

**Output:** `experiments/stage1/logs/integrity/tier1/`.

---

### Tier 2 — kvpress pipeline integration (~1 hour, 1 GPU)

**Goal:** when `TurboQuantPress` and `JointQKPress` are invoked through the
**full kvpress `evaluate.py` pipeline** (not direct call), behavior still
matches Tier 1 numbers.

This catches integration bugs that only surface in production: the
`_setup_press` instantiation path, model-loading dtype, BasePress
`forward_hook` semantics, kvpress's metric extraction.

**2.1 kvpress on the upstream needle prompt — both methods.** New script
`experiments/stage1/scripts/run_kvpress_needle.py`. Loads model via kvpress's
own loader, injects a press, runs the needle prompt, reports
generation-correctness (EXACT / PARTIAL / MISS) plus a hook-extracted cosine
similarity vs fp16 ground-truth attention.

Configs on Qwen2.5-3B-Instruct (matches upstream Table):
- TurboQuant K4/V2 rw=0 → expect MISS (matches upstream)
- JointQK K4/V2 rw=0 layer0_full ∈ {True, False} → first data on this
  combination

**2.2 kvpress LongBench-tiny — both methods.** 8 jobs (full + 3×TurboQuant +
3×JointQK + KIVI baseline) on qasper at fraction=0.05, V=2 fixed, rw=0,
prot=0. Compare result.json metrics across methods. Expected: Tier 1.3
attention-score ordering predicts Tier 2.2 F1 ordering.

**Output:** `experiments/stage1/logs/integrity/tier2/`.

---

### Tier 3 — Head-to-head ablation subset (~2 hours, 6 GPUs)

**Goal:** the paper's actual head-to-head with all knobs explicit and a
layer-0-carve-out ablation. Tells us whether JointQK's win comes from the
quantizer or from the layer-0 carve-out.

**3.1 Apples-to-apples sweep at V=2.** New launcher
`experiments/stage1/scripts/launch_tier3_head_to_head.sh`. Builds command file
for `parallel_launcher.py`:

| Method | K | V | rw | layer0_full | protected_layers |
|---|---|---|---|---|---|
| full_precision | — | — | — | — | — |
| TurboQuant | 2/3/4 | 2 | 0 | — | 0 |
| JointQK | 2/3/4 | 2 | 0 | False | — |
| JointQK | 2/3/4 | 2 | 0 | True | — |

Tasks: qasper + narrativeqa + multifieldqa at fraction=0.1.
Models: Qwen3-8B + Llama-3.1-8B.
Total: 10 configs × 3 tasks × 2 models = 60 jobs.

Pass criteria:
- **Strong claim:** at K=2, JointQK with `layer0_full=False` beats TurboQuant
  by ≥3pp F1 on at least 4 of 6 (model × task) cells.
- **Stronger claim:** at K=2, JointQK with `layer0_full=True` beats TurboQuant
  by ≥5pp F1 on all 6 cells. (Already partially observed in Phase 7: JointQK
  +7.23pp on qasper and +11.47pp on narrativeqa for Qwen3-8B.)

If only the second holds, the paper headline becomes "JointQK + layer-0
carve-out beats TurboQuant"; that's still a real result but reframes the
attribution.

**3.2 Same sweep at V=3** to confirm the ordering doesn't depend on V_lock
choice.

**Output:** `experiments/stage1/logs/integrity/tier3/`. Two CSVs:
`v2_summary.csv`, `v3_summary.csv`.

---

### Tier 4 — Generation needle (optional, ~3 hours, 1 GPU)

Run `generation_test_v2.py`-style needle for both methods at TurboQuant's
published configs. Direct text comparison. High-fidelity but doesn't bear on
quantitative LongBench/RULER claims. Skip unless time permits.

---

## Order of operations

| When | Tier | Rationale |
|---|---|---|
| Now (Phase 7 unaffected) | 0.1, 0.2, 0.3, 0.4 | < 5 min, no GPU; if 0.2/0.3 fail, Phase 7 numbers suspect |
| After 0 passes | 1.1 | 30 min on 1 GPU; doesn't disturb Phase 7 if a GPU has just freed up between jobs |
| After 1.1 | 1.2 | TurboQuant wrapper sanity, blocking |
| After 1.2 | 1.3 | first apples-to-apples + OOD generalization signal |
| After 1.3 | 2.1 + 2.2 | kvpress integration sanity for both methods |
| After Phase 7 finishes | 3.1 + 3.2 | needs all 6 GPUs |
| Optional | 4 | generation needle |

Tier 0 is non-blocking on Phase 7 (CPU-only).
Tier 1 needs ≤1 GPU briefly — schedule between Phase 7 jobs (each ~50 min).
Tier 3 waits for Phase 7 to free up the pool.

---

## Pass / fail decision tree

```
Tier 0.1 fail
    → vendored upstream broken → re-vendor or debug Lloyd-Max codebook

Tier 0.2 fail
    → TurboQuantPress drift → check chunking, GPU move, in-place mutations
    BLOCKER for any TurboQuant claim

Tier 0.3 fail
    → JointQKPress drift on some (L, h) → check bundle indexing, V_h ordering
    BLOCKER for any JointQK claim

Tier 0.4 fail (calibration overlap with eval)
    → filter eval set or re-document scope of calibration
    Not a method bug, but affects how Phase 7 numbers are framed

Tier 1.1 fail
    → can't reproduce upstream → environment issue (model, transformers
       version, bnb-4bit) before any other comparison is meaningful

Tier 1.2 fail (1.1 passes)
    → wrapper-direct drift → narrows to T1–T3 risks
    BLOCKER

Tier 1.3 outcome 1 (JointQK wins OOD)
    → strongest possible result; headline strengthens

Tier 1.3 outcome 2 (parity OOD)
    → publishable as "wins in-distribution; ties OOD" with explicit framing

Tier 1.3 outcome 3 (JointQK loses OOD)
    → trigger option (iii): build C4-based calibration → re-run Tier 1.3 + 3
    → DO NOT publish before resolving

Tier 2 shows pipeline-vs-direct drift
    → kvpress integration bug → debug evaluate.py / model loader / dtype

Tier 3 shows JointQK > TurboQuant ONLY with layer0_full=True
    → reframe headline to "JointQK + layer-0 carve-out as a system"

Tier 3 shows JointQK > TurboQuant with layer0_full=False
    → headline holds: the quantizer alone wins → publish
```

---

## Files to add (planned, not yet created)

All new — no edits to existing files.

**Tests:**
- `experiments/stage1/tests/test_turboquant_press_parity.py` — Tier 0.2
- `experiments/stage1/tests/test_jointqk_press_parity_broad.py` — Tier 0.3

**Scripts:**
- `experiments/stage1/scripts/check_calibration_holdout.py` — Tier 0.4
- `experiments/stage1/scripts/run_attention_parity.py` — Tier 1.2 / 1.3
- `experiments/stage1/scripts/run_kvpress_needle.py` — Tier 2.1
- `experiments/stage1/scripts/launch_tier2_longbench_tiny.sh` — Tier 2.2
- `experiments/stage1/scripts/launch_tier3_head_to_head.sh` — Tier 3

**Aggregator:**
- `experiments/stage1/eval/aggregate_integrity.py` — joint summary across
  tiers

**Logs:**
- `experiments/stage1/logs/integrity/tier{0,1,2,3,4}/`

---

## How Tier 1.3 will be written up in the paper

> "JointQK is calibrated on a 24-example LongBench-E subset (sequences ≤ 4K
> tokens). To stress-test generalization, we evaluate both methods on the
> synthetic needle-in-haystack prompt from [Tonbi 2026], which is
> out-of-distribution for our calibration. At K=2 / V=2 / rw=0 / prot=0,
> JointQK retains [X]% top-1 attention agreement with the fp16 oracle vs
> TurboQuant's [Y]% (averaged over 32 layers, 8K context). This [confirms /
> qualifies] the in-distribution LongBench result by showing that our
> calibrated allocation [transfers / is partially distribution-dependent]."

The paragraph writes itself once we have the numbers. We don't yet know
which version we'll write — that's why we run the test.
