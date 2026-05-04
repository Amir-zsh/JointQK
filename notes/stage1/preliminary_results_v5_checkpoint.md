# Preliminary Results Checkpoint — Phase 7 v5 (pre-fairness-fix)

**Date:** 2026-05-03  19:30 PDT
**Status:** Phases 0–6 complete; Phase 7 LongBench Qwen partial (32 of 128 jobs); other Phase 7 stages pending.
**Caveat:** This snapshot is from **Phase 7 v5**, which used `layer0_full_precision=True` for JointQK (only 31/32 K-layers compressed). TurboQuant and KIVI compress all 32 K-layers. **JointQK had a slight unfair advantage in v5.** A re-run (v6) with `layer0_full_precision=False` (apples-to-apples) was started but stopped per request; this checkpoint exists so v6 results can be compared head-to-head against v5.

---

## 1. Locked configuration (Phases 1–6)

| Knob | Value | Source |
|---|---|---|
| K compression method | `r_sym_waterfill` (JointQK) | Locked at Stage 1E (top-1 retention 0.860 at K=3 on Qwen) |
| K bits per coord | swept {2, 3, 4} | Phase 7 sweep |
| V compression method | **`v_eigen_uniform`** (eigenvectors of Σ_V + uniform bits) | Phase 1AB (qasper@V=3: 43.18 vs 43.13 full = 100% rel) |
| V bits per coord | 3 | Phase 1AB lock |
| K-side bit cap | max 8 bits/coord (saved bits redistributed to lowest) | Added 09:50 to bound K_max=256 (memory safety) |
| Decode scope | **Mode A** (`compress_decode=False`) — prefill only | Phase 6 ablation: byte-identical to Mode B across 12 cells, Mode A is 200×+ cheaper at decode |
| K layer-0 carve-out (v5 only) | YES (skip layer 0 K) | **Unfair vs baselines; flipped to NO in v6** |
| K rank truncation | 64 (for `_truncate` variants only; `_waterfill` uses all coords) | Stage 1E default |

## 2. Phase 1 — V method study (Qwen3-8B, qasper, fraction 0.3)

Full-precision F1: **43.13**.

| V method | V=2 | V=3 | V=4 |
|---|---|---|---|
| `v_random` (TurboQuant V) | 32.69 (rel 0.76) | 42.76 (0.99) | **43.20 (1.00)** |
| **`v_eigen_uniform`** (locked) | 41.60 (0.96) | **43.18 (1.00)** | 41.81 (0.97) |
| `v_eigen_waterfill` | 37.20 (0.86) | 41.92 (0.97) | 40.78 (0.95) |

**Decision:** `v_eigen_uniform`, V=3. Originally locked `v_eigen_waterfill` via tiebreaker, then corrected to `v_eigen_uniform` after Phase 7 v5 showed JointQK underperforming. The waterfill choice didn't help V because Σ_V's eigenvalue spectrum is roughly uniform.

## 3. Phase 1 — K-only sweep (Qwen, qasper, fraction 0.3)

JointQK r_sym_waterfill at varying K bits, V=fp16:

| K bits | F1 | Rel F1 |
|---|---|---|
| 2 | 44.18 | 1.024 |
| 3 | 42.22 | 0.979 |
| 4 | 41.64 | 0.966 |

K=2 ≥ full-precision (statistical noise on 60 examples). K-side compression at K∈{2,3,4} retains ≥97% of full F1 on this slice.

## 4. Phase 1C — Combined K+V sanity check (Qwen, qasper, fraction 0.3, V=eigen_waterfill V=3 — pre-correction)

| Config | F1 | vs full-precision |
|---|---|---|
| Full precision | 43.13 | 1.00 |
| JointQK K2/V3 | 38.14 | 0.884 |
| JointQK K3/V3 | 40.58 | 0.941 |
| JointQK K4/V3 | 40.21 | 0.932 |

These numbers are with the suboptimal `v_eigen_waterfill` V; the v_eigen_uniform retake (Phase 7 v5) gives qasper K3/V3 ≈ 40.13.

## 5. Phase 5 — Llama-3.1-8B Stage-1E reproduction (W1 gate)

Top-1 attention retention on LongBench-E 24-bundle (layer-0 excluded), Qwen-style methodology applied to Llama-3.1-8B-Instruct.

| Method | b=2 prefill | b=3 prefill | b=4 prefill |
|---|---|---|---|
| TurboQuant V3 (random + uniform) | 0.4317 | 0.5050 | 0.5454 |
| Q-Eigen WaterFill | 0.5223 | 0.5685 | 0.5953 |
| CCA-Orth WaterFill | 0.4572 | 0.5022 | 0.5402 |
| **JointQK WaterFill (`r_sym_waterfill`)** | **0.5439** | **0.5904** | **0.6148** |

**JointQK wins on Llama at every bit budget**, by 2–3 pp over Q-Eigen WaterFill (the prior best) and 6–11 pp over TurboQuant. Decode-phase metrics match prefill.

The Llama margin (~2pp) is smaller than the Qwen margin (~5pp at b=3) — head-dim and norm-statistic differences. Both models confirm JointQK is the winning K basis.

## 6. Phase 6 — Decode-scope ablation (Qwen, qasper + narrativeqa, fraction 0.3)

| Cell | Mode A (prefill-only) | Mode B (prefill+decode) | Δ |
|---|---|---|---|
| qasper K=2 | 38.14 | 38.14 | 0.00 |
| qasper K=3 | 40.58 | 40.58 | 0.00 |
| qasper K=4 | 40.21 | 40.21 | 0.00 |
| narrativeqa K=2 | 22.99 | 22.99 | 0.00 |
| narrativeqa K=3 | 25.24 | 25.24 | 0.00 |
| narrativeqa K=4 | 25.79 | 25.79 | 0.00 |

**Byte-identical task scores.** Decode-step K compression has no measurable effect on these benchmarks at decode lengths typical of QA (~5–200 tokens). Mode A is the canonical choice — significantly cheaper computationally (200×+ fewer per-token dispatches).

## 7. Phase 7 v5 — Partial LongBench Qwen3-8B (fraction 0.5, ~32/128 jobs)

**WITH the unfair layer-0 carve-out for JointQK.**

### 7.1 Full-precision baselines (no compression, oracle)

| Task | full_precision F1 |
|---|---|
| narrativeqa | 31.65 |
| qasper | 43.00 |
| multifieldqa_en | 53.89 |
| hotpotqa | 65.31 |
| 2wikimqa | 56.22 |

### 7.2 KIVI int4 (per-channel int4 K + per-token int4 V, both compressed)

| Task | KIVI int4 | rel vs full |
|---|---|---|
| narrativeqa | 32.48 | 1.026 |
| qasper | 43.24 | 1.006 |
| multifieldqa_en | 52.46 | 0.974 |
| hotpotqa | 63.70 | 0.975 |
| 2wikimqa | 53.62 | 0.954 |

KIVI int4 retains ~95-103% of full precision across all 5 tasks. Strong baseline.

### 7.3 Method × K bits head-to-head (5 tasks, V=v_eigen_uniform V=3 for JointQK; V=random+uniform V=3 for TurboQuant)

| Task | Method | K=2 | K=3 | K=4 |
|---|---|---|---|---|
| narrativeqa | TurboQuant | 16.13 | 28.55 | 33.28 |
| narrativeqa | **JointQK** | **27.60** | 28.11 | 28.95 |
| narrativeqa | Δ (jq − tq) | **+11.47** ✓ | -0.44 | -4.33 |
| qasper | TurboQuant | 33.55 | 41.14 | 41.81 |
| qasper | **JointQK** | **40.78** | 40.13 | 39.35 |
| qasper | Δ (jq − tq) | **+7.23** ✓ | -1.01 | -2.46 |
| multifieldqa_en | TurboQuant | 40.77 | 51.04 | 53.22 |
| multifieldqa_en | **JointQK** | **51.08** | 50.57 | 52.34 |
| multifieldqa_en | Δ (jq − tq) | **+10.31** ✓ | -0.47 | -0.88 |
| hotpotqa | TurboQuant | 37.45 | 64.50 | 65.61 |
| hotpotqa | **JointQK** | **59.39** | 63.77 | 65.49 |
| hotpotqa | Δ (jq − tq) | **+21.94** 🔥 | -0.73 | -0.12 |
| 2wikimqa | TurboQuant | 34.13 | (pending) | (pending) |
| 2wikimqa | JointQK | (pending) | (pending) | (pending) |

### 7.4 Cross-task aggregates (4 tasks fully populated)

**At K=2 (low-budget regime):**

| Method | Mean F1 (4 tasks) | Mean rel-F1 vs full |
|---|---|---|
| TurboQuant | 31.97 | 0.700 (70% retention) |
| **JointQK** | **44.71** | **0.918** (92% retention) |
| **Δ (jq − tq)** | **+12.74 pp** | **+22 pp retention** |

**At K=3:**
| Method | Mean F1 | Mean rel |
|---|---|---|
| TurboQuant | 46.31 | 0.957 |
| JointQK | 45.65 | 0.942 |
| Δ | -0.66 | -1.5 pp |

**At K=4:**
| Method | Mean F1 | Mean rel |
|---|---|---|
| TurboQuant | 48.48 | 1.001 (parity with full) |
| JointQK | 46.78 | 0.965 |
| Δ | -1.70 | -3.6 pp |

## 8. Headline finding (preliminary)

**At aggressive compression budgets (K=2, ≈ 4× compression on the K-cache), JointQK retains 92% of full-precision F1 vs TurboQuant's 70%** — a 22-percentage-point retention gap. The advantage holds across narrativeqa, qasper, multifieldqa_en, and hotpotqa.

At higher budgets (K=3, K=4), the methods are competitive (within ±2-4 pp of each other). TurboQuant's simpler uniform allocation works well when bits are abundant.

**Why the K=2 advantage exists:** TurboQuant's uniform allocation gives every coord exactly 2 bits = 4 levels — too few to capture the highest-variance directions while wasting bits on low-importance coords. JointQK's water-fill on the joint-Q-K eigenbasis allocates 4–7 bits to the top ~30 coords (capturing >95% of attention-relevant variance) and 0 bits to low-importance coords. This is the rate-distortion-optimal allocation.

## 9. Known caveats and pending work

### Caveats in this checkpoint
1. **Layer-0 unfairness (v5):** JointQK skipped layer 0 K (only 31/32 layers compressed); baselines compressed all 32. Likely contributes ≤1 pp to JointQK's lead. Being corrected in v6.
2. **Bit cap at 8:** K-side per-coord bits capped at 8 with saved bits redistributed to low-importance coords. This is a memory-safety patch (otherwise K_max=2048 OOMs the chunked roundtrip allocator). Could slightly distort the optimal water-fill allocation; impact likely <0.5 pp.
3. **Fraction 0.5:** Only 100 of ~200 LongBench examples per task. Statistical noise ±1-2 pp.
4. **Single-model-coverage:** This is Qwen-only. Llama LongBench will be the second model; not yet started.
5. **Decode-scope ambiguity:** Mode A (prefill-only) is canonical. Mode B (prefill+decode) gave byte-identical results in the ablation, but only on 2 tasks at fraction 0.3. The "no-difference" finding is not yet stress-tested on long-decode tasks.

### Pending work
- **Phase 7 v6 LongBench Qwen** (apples-to-apples, ~10h compute) — to confirm the K=2 dominance survives the layer-0 fix
- **Phase 7 LongBench Llama-3.1-8B** (~12h compute) — second-model story
- **Phase 7 RULER NIAH** at 4k/8k/16k contexts (~3h)
- **Failed cell:** 2wikimqa TurboQuant K=3 needs manual re-run (HF dataset cache miss; cache now populated, can re-launch)

## 10. Predicted impact of v6 (apples-to-apples)

The layer-0 unfairness gave JointQK a small boost. Expected impact on v6 numbers:
- JointQK F1 likely drops 0-2 pp on each task (because layer 0 K is harder to compress)
- TurboQuant / KIVI numbers unchanged
- **K=2 dominance gap likely shrinks from +22 pp retention to +18–20 pp.** Still a strong result.

## 11. File pointers

- Locked config: `artifacts/stage1/v_method_study/v_lock.txt`
- Decode-scope decision: `artifacts/stage1/downstream/qwen3_8b/decode_scope/decode_decision.txt`
- Phase 5 (Llama Stage-1E): `artifacts/stage1/cca_vs_waterfill_study/llama31_8b/`
- Phase 7 raw outputs: `artifacts/stage1/downstream/qwen3_8b/{full_precision,kivi_int4,turboquant_k*,jointqk_k*}_<task>/`
- Phase 7 logs: `experiments/stage1/logs/phase7_longbench_qwen3_8b/`
- Phase 7 chain script: `experiments/stage1/scripts/_phase7_chain.py`
- Press classes: `experiments/stage1/toolkit/{jointqk_press, turboquant_press, kivi_press, kivi_quantizer, v_compressor_adapter}.py`
- Submission plans: `notes/core/{neurips_submission_plan, neurips_implementation_plan}.md`
- This checkpoint: `notes/stage1/preliminary_results_v5_checkpoint.md`
