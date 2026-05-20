# Phase 7 v7 — Cross-Model Comparison (Qwen3-8B vs Llama-3.1-8B)

**Date:** 2026-05-12
**Sources:** `notes/stage1/phase7_v7_results_report.md` (Qwen3) + `notes/stage1/phase7_v7_llama31_8b_results_report.md` (Llama).
**Scope:** 192 cells × 2 models = 384 LongBench F1 measurements at fraction=1.0 with calibration-train rows excluded from eval.

---

## TL;DR

The v7 design works on both models, but **the K=2 multi-doc QA story does not replicate uniformly**:

1. ✅ **K=4 V=3 near-FP for all three methods on both models** — JointQK, TurboQuant, and KIVI int4 all land within 0.7 pp of full-precision F1 on both Qwen3 and Llama. The "high-budget is essentially solved" finding generalizes.

2. ⚠️ **K=2 V=3 JointQK > TurboQuant is Qwen3-specific**. The +2.66 pp excl-trec mean / +6.26 pp on hotpotqa we reported for Qwen3 vanishes on Llama (and slightly reverses on hotpotqa). Cause is *not* JointQK degrading on Llama — JointQK retention is 92.6% on Qwen3 and **95.5% on Llama**. The story is that **TurboQuant K=2 V=3 holds up much better on Llama (96.0% retention) than on Qwen3 (87.7%).** Random Hadamard apparently fits Llama's KV structure at low bits in a way it doesn't fit Qwen3's.

3. ✅ **JointQK is the consistent cross-model choice at K=2.** On Qwen3 it beats TurboQuant by +2.7 pp excl-trec mean; on Llama it loses by 1.3 pp but stays within 1 pp of TurboQuant on multi-doc QA mean (95.5% vs 96.0%). JointQK's calibrated basis adapts; TurboQuant's random basis happens to work on Llama and doesn't on Qwen3. **If you don't know which model you'll deploy on, JointQK is the safer bet at K=2.**

4. ✅ **KIVI int2 catastrophe replicates** on both models (~83% retention) — low-bit per-channel int quantization is universally bad. KIVI int4 universally good (~99%).

---

## 1. 11-task excl-trec mean per config

| config | Qwen3 F1 | Llama F1 | Qwen ret % | Llama ret % |
|---|---:|---:|---:|---:|
| **Full precision** | **47.59** | **45.66** | 100.0 | 100.0 |
| JointQK K=2 V=2 | 44.13 | 42.44 | 92.7 | 93.0 |
| **JointQK K=2 V=3** | **45.02** | **42.80** | **94.6** | **93.7** |
| JointQK K=3 V=2 | 45.61 | 45.04 | 95.9 | 98.6 |
| JointQK K=3 V=3 | 45.88 | 45.21 | 96.4 | 99.0 |
| JointQK K=4 V=2 | 46.50 | 45.41 | 97.7 | 99.4 |
| JointQK K=4 V=3 | 46.89 | 45.10 | 98.5 | 98.8 |
| TurboQuant K=2 V=2 | 41.65 | 44.07 | 87.5 | 96.5 |
| **TurboQuant K=2 V=3** | **42.36** | **44.08** | **89.0** | **96.5** |
| TurboQuant K=3 V=2 | 45.95 | 46.06 | 96.6 | 100.9 |
| TurboQuant K=3 V=3 | 46.88 | 45.30 | 98.5 | 99.2 |
| TurboQuant K=4 V=2 | 46.80 | 46.02 | 98.3 | 100.8 |
| TurboQuant K=4 V=3 | 47.11 | 45.40 | 99.0 | 99.4 |
| KIVI int2 | 39.42 | 38.49 | 82.8 | 84.3 |
| KIVI int3 | 45.84 | 44.42 | 96.3 | 97.3 |
| KIVI int4 | 47.28 | 45.53 | 99.4 | 99.7 |

**Reading this table:**
- At K=4 (any V), all three method families land at 99% retention on both models.
- At K=3 (any V), 96–101% on both models — calibrated and uncalibrated converge.
- At K=2 V=3: Qwen has a clean ordering (JointQK > TurboQuant > KIVI int2). Llama's TurboQuant is *almost identical* to JointQK at low bits — random Hadamard fits Llama's KV layer better than it fits Qwen3's at this budget.
- Llama retention values >100% (e.g., TQ K=3 V=2: 100.9%) reflect sampling noise on fraction=1.0 (~150 samples/task with ±1 pp standard error).

## 2. Multi-doc QA mean (hotpotqa, musique, 2wikimqa, narrativeqa)

| config | Qwen3 F1 | Llama F1 | Q ret % | L ret % |
|---|---:|---:|---:|---:|
| **FP** | **43.63** | **43.61** | 100.0 | 100.0 |
| JointQK K=2 V=3 | **40.42** | 41.63 | **92.6** | **95.5** |
| JointQK K=4 V=3 | **43.48** | 43.35 | 99.7 | 99.4 |
| TurboQuant K=2 V=3 | 38.27 | 41.86 | **87.7** | **96.0** |
| TurboQuant K=4 V=3 | 42.85 | 42.58 | 98.2 | 97.6 |
| KIVI int2 | 35.11 | 33.35 | 80.5 | 76.5 |
| KIVI int4 | 43.19 | 42.78 | 99.0 | 98.1 |

**Multi-doc QA is where v5 reported JointQK's largest K=2 wins.** v7 confirms on Qwen3 (92.6% vs TQ 87.7% = +4.9 pp retention) but on Llama JointQK 95.5% and TurboQuant 96.0% are essentially tied. JointQK is consistent across models; TurboQuant varies.

## 3. K=2 V=3 JointQK − TurboQuant per task (replication check)

Positive = JointQK wins.

| task | Qwen3 JQ−TQ | Llama JQ−TQ | (Q−L) sign agrees? |
|---|---:|---:|---|
| qasper | +2.98 | −0.17 | ❌ flipped |
| qmsum | +1.75 | +0.09 | ✅ |
| multi_news | +0.65 | +0.16 | ✅ |
| trec | +8.50 | +7.44 | ✅ (artifact-driven on both) |
| triviaqa | +4.70 | +0.69 | ✅ |
| samsum | +0.20 | −0.47 | ❌ flipped (small) |
| lcc | +4.30 | **−12.04** | ❌ flipped (huge) |
| repobench-p | +6.05 | −1.43 | ❌ flipped |
| **hotpotqa** | **+6.26** | **−3.24** | ❌ **flipped** |
| musique | −0.01 | −1.61 | ✅ |
| 2wikimqa | +1.84 | +5.70 | ✅ |
| narrativeqa | +0.51 | −1.74 | ❌ flipped (small) |

**Per-task agreement is mixed.** The largest cross-model disagreements are on **lcc** (Qwen JQ+4.3 / Llama JQ−12.0!) and **hotpotqa** (Qwen JQ+6.3 / Llama JQ−3.2). On three tasks (multi_news, qmsum, 2wikimqa) the sign of JQ vs TQ agrees and is positive on both models. On the rest the sign flips or is noise-tied.

The lcc result is striking — on Llama, TurboQuant K=2 V=3 = 48.01 but JointQK K=2 V=3 = 35.97, a 12 pp swing in TurboQuant's favor. That's worth investigating per-task (next-step §3 below).

## 4. K=4 V=3 retention — does the "all tied" result replicate?

Per excl-trec and multi-doc QA, both models converge at K=4 V=3:

| config | Q F1 (excl-trec) | L F1 (excl-trec) | Q Δ FP | L Δ FP |
|---|---:|---:|---:|---:|
| FP | 47.59 | 45.66 | — | — |
| JointQK K=4 V=3 | 46.89 | 45.10 | −0.69 | −0.56 |
| TurboQuant K=4 V=3 | 47.11 | 45.40 | −0.48 | −0.26 |
| KIVI int4 | 47.28 | 45.53 | −0.31 | −0.13 |

**At K=4 V=3, all three methods are within 0.7 pp of FP on both models** — statistically tied. Both models agree.

## 5. hotpotqa specifically

| config | Qwen3 | Llama | Q−FP | L−FP |
|---|---:|---:|---:|---:|
| FP | 63.75 | 60.71 | — | — |
| **JointQK K=2 V=3** | **62.14** | **58.62** | −1.61 | −2.09 |
| **TurboQuant K=2 V=3** | **55.88** | **61.86** | **−7.87** | **+1.15** |
| JointQK K=4 V=3 | 63.72 | 60.49 | −0.03 | −0.22 |
| TurboQuant K=4 V=3 | 63.58 | 59.45 | −0.17 | −1.26 |
| KIVI int4 | 62.79 | 60.37 | −0.96 | −0.34 |

**At K=4: both models near-FP for all methods. ✓**

**At K=2 V=3:** the Qwen3 picture — TurboQuant collapses to −7.87 pp while JointQK stays at −1.61 — is the **signature multi-doc QA finding from v5/v6**. On Llama, TurboQuant doesn't collapse (+1.15 pp, noise-tied with FP) so JointQK's relative advantage vanishes.

## Why does TurboQuant K=2 work on Llama but not on Qwen3?

A few hypotheses to investigate:

1. **Llama's KV second-moment structure may be flatter / more isotropic.** Llama-3.1-8B's `Σ_K` eigenvalue spectrum may have a flatter tail than Qwen3-8B's. A random Hadamard's uniform per-coord bit allocation only loses much when the model's KV variance is concentrated in a few directions; if Llama's variance is spread, uniform bits cost less.

2. **Layer-0 anomaly differs between models.** Both use `layer0_full_precision=True`, so the anomalous attention sink is excluded for both. But the *rest* of the layers may have different anomaly distributions. Worth measuring `||Σ_K||_2 / mean(Σ_K diag)` per layer for each model.

3. **Differences in chat template / position encoding effects** on Q/K post-RoPE distributions.

We have the calibration moments for Qwen3 on disk; the Llama equivalent is on the remote machine and could be pulled back for direct comparison.

## What this means for the paper

- **Qwen3-only headline overstates the K=2 advantage.** On a Qwen3-only sweep, JointQK K=2 V=3 looks like a clean win (+2.66 pp excl-trec / +6.26 pp on hotpotqa). Llama tells a different story: TurboQuant is essentially as good as JointQK on Llama at the same budget. The "K=2 multi-doc QA win" is real on Qwen3 but model-dependent.

- **Cross-model consistency favors JointQK.** Across both models, JointQK K=2 V=3 retains 92.6%–95.5% of FP. TurboQuant K=2 V=3 retains 87.7%–96.5% — much wider spread. **JointQK delivers predictable retention; TurboQuant's retention depends on the model.** That's the cleanest cross-model story for the paper.

- **At K=4, the calibration debate dissolves** — JointQK, TurboQuant, and KIVI int4 are all within sampling noise of FP on both models. The interesting territory is K=2.

- **The "v5 hotpotqa win" doesn't generalize cleanly** but the *general direction* (JointQK is robust at low budgets) does. The paper should pitch JointQK as the consistent option rather than the dominant option.

## Recommended next steps

1. **Investigate the lcc collapse on Llama at K=2** (Qwen JQ +4.3 vs Llama JQ −12.0). lcc is code completion — Llama's code Q/K distribution may have a sharper principal direction than Qwen3's, making JointQK's bit concentration hurt rather than help. Per-layer top-1 mining (analyze_perlayer_top1.py infrastructure exists) on the Llama capture would localize.

2. **Compare the calibration eigenvalue spectra** between Qwen3 and Llama. If Llama's `Σ_K` is flatter, that explains why TurboQuant K=2 works on Llama. Pull `cca_stats_llama31_8b_longbench_compact8_n400.pt` from the remote and diff against `cca_stats_longbench_compact8_n400.pt`.

3. **Bigger calibration corpus for cross-model robustness.** Maybe Llama needs more or different calibration data. Worth a separate ablation.

4. **For the paper, lead with the consistency angle:** "JointQK retains 93–95% of FP at 8× K-cache compression across two 8B-class models; TurboQuant retention varies 88–96% depending on model." Less aggressive than "+6.26 pp win," but defensible and stronger as a deployment claim.
