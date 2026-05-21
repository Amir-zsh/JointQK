# Phase 7 v7 Results — Llama-3.1-8B-Instruct

End-to-end run executed 2026-05-06 on 4× A100-40GB (GPUs 0,1,2,3).

## Pipeline summary

- **§1 Capture:** 480 LongBench prompts (8 tasks × 60, 50 train / 10 test, 2k–32k tokens) on 4 GPUs. Wall: ~23 min/shard. Cross-model split manifest reused via `messages_sha256` (tokenizer-independent).
- **§2 Stats:** aggregated to `02_stats/aggregate.pt` (2.1 GB), 480 files validated.
- **§3 Build:** pooled 400 train examples, computed `R_sym = eigvec((Σ_QΣ_K + Σ_KΣ_Q)/2)` per (layer, kv_head). Wrote `jointqk_llama31_8b_longbench_compact8_n400.pt` (80 MB) and `v_stats_llama31_8b_longbench_compact8_n400.pt` (32 MB). Wall: 1.5 min.
- **§4 Sweep:** 192 cells (12 tasks × 16 configs). Two runs:
  - run1 (2 jobs/GPU) hit cascading multi_news OOMs; salvaged 23 cells.
  - run2 (1 job/GPU, idempotent skip) completed the remaining 169 cells with 0 hard failures and 3 OOM auto-retries. Wall: ~7 h.

## F1 grid (LongBench task F1, fraction=1.0, calibration train rows excluded from eval)

```
config                 |     qasper |      qmsum | multi_news |       trec |   triviaqa |     samsum |        lcc | repobench- |   hotpotqa |    musique |   2wikimqa | narrativeq |   mean
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
full_precision         |      46.30 |      25.31 |      27.18 |      29.50 |      91.71 |      40.95 |      51.37 |      45.01 |      60.71 |      32.62 |      50.99 |      30.13 |  44.31
jointqk_k2_v2          |      42.64 |      24.62 |      26.78 |      44.00 |      91.68 |      38.42 |      37.23 |      42.26 |      58.89 |      27.08 |      49.25 |      28.03 |  42.57
jointqk_k2_v3          |      43.63 |      25.25 |      27.08 |      27.50 |      91.52 |      39.48 |      35.97 |      41.32 |      58.62 |      28.57 |      51.55 |      27.80 |  41.52
jointqk_k3_v2          |      45.83 |      24.94 |      27.65 |      30.50 |      92.19 |      40.28 |      47.88 |      47.11 |      61.08 |      31.21 |      47.46 |      29.85 |  43.83
jointqk_k3_v3          |      45.78 |      25.60 |      27.15 |      25.00 |      92.36 |      40.79 |      46.06 |      45.50 |      61.44 |      32.95 |      49.48 |      30.15 |  43.52
jointqk_k4_v2          |      45.22 |      25.35 |      27.83 |      33.50 |      92.21 |      40.60 |      49.58 |      47.80 |      61.15 |      33.31 |      46.94 |      29.48 |  44.41
jointqk_k4_v3          |      45.92 |      25.29 |      27.24 |      23.00 |      92.56 |      40.32 |      46.83 |      44.55 |      60.49 |      33.16 |      49.73 |      30.02 |  43.26
turboquant_k2_v2       |      42.79 |      24.33 |      26.67 |      27.67 |      89.93 |      40.36 |      48.37 |      46.58 |      61.29 |      31.55 |      44.64 |      28.24 |  42.70
turboquant_k2_v3       |      43.80 |      25.16 |      26.92 |      20.06 |      90.83 |      39.95 |      48.01 |      42.75 |      61.86 |      30.18 |      45.85 |      29.54 |  42.08
turboquant_k3_v2       |      47.50 |      25.12 |      27.58 |      38.50 |      92.05 |      41.02 |      52.69 |      47.94 |      61.18 |      30.43 |      50.55 |      30.64 |  45.43
turboquant_k3_v3       |      45.05 |      24.86 |      27.70 |      26.00 |      91.59 |      39.78 |      51.85 |      45.28 |      60.84 |      31.41 |      48.88 |      31.08 |  43.69
turboquant_k4_v2       |      48.24 |      25.19 |      27.25 |      33.00 |      92.32 |      41.13 |      52.92 |      47.61 |      60.33 |      32.33 |      48.79 |      30.16 |  44.94
turboquant_k4_v3       |      46.69 |      25.37 |      27.89 |      22.00 |      92.50 |      39.84 |      51.44 |      45.32 |      59.45 |      32.26 |      48.14 |      30.48 |  43.45
kivi_int2              |      36.84 |      24.43 |      27.55 |      25.00 |      85.33 |      38.91 |      44.31 |      32.67 |      47.11 |      19.58 |      37.39 |      29.32 |  37.37
kivi_int3              |      45.25 |      25.44 |      27.96 |      20.00 |      91.78 |      36.62 |      49.96 |      41.18 |      57.67 |      33.54 |      49.19 |      29.99 |  42.38
kivi_int4              |      45.76 |      25.32 |      27.50 |      24.50 |      92.34 |      41.64 |      51.49 |      45.66 |      60.37 |      30.71 |      49.22 |      30.84 |  43.78
```

## Headline mean F1 (ranked)

| Rank | Config | Mean F1 | Δ vs FP |
|---|---|---|---|
| 1 | turboquant_k3_v2 | **45.43** | **+1.12** |
| 2 | turboquant_k4_v2 | 44.94 | +0.63 |
| 3 | jointqk_k4_v2 | 44.41 | +0.10 |
| 4 | full_precision | 44.31 | — |
| 5 | jointqk_k3_v2 | 43.83 | −0.48 |
| 6 | kivi_int4 | 43.78 | −0.53 |
| 7 | turboquant_k3_v3 | 43.69 | −0.62 |
| 8 | jointqk_k3_v3 | 43.52 | −0.79 |
| 9 | turboquant_k4_v3 | 43.45 | −0.86 |
| 10 | jointqk_k4_v3 | 43.26 | −1.05 |
| 11 | turboquant_k2_v2 | 42.70 | −1.61 |
| 12 | jointqk_k2_v2 | 42.57 | −1.74 |
| 13 | kivi_int3 | 42.38 | −1.93 |
| 14 | turboquant_k2_v3 | 42.08 | −2.23 |
| 15 | jointqk_k2_v3 | 41.52 | −2.79 |
| 16 | kivi_int2 | 37.37 | −6.94 |

## Cross-model comparison: JointQK vs TurboQuant

For each (K, V) grid point, JointQK − TurboQuant on Llama-3.1-8B:

| K | V | JointQK | TurboQuant | Δ (J − T) |
|---|---|---|---|---|
| 2 | 2 | 42.57 | 42.70 | **−0.13** |
| 2 | 3 | 41.52 | 42.08 | **−0.56** |
| 3 | 2 | 43.83 | 45.43 | **−1.60** |
| 3 | 3 | 43.52 | 43.69 | −0.17 |
| 4 | 2 | 44.41 | 44.94 | **−0.53** |
| 4 | 3 | 43.26 | 43.45 | −0.19 |

**JointQK loses or ties at every grid point on Llama.** The Qwen3 v7 finding that JointQK beats TurboQuant — particularly at K=2 by several points and on multi-doc QA by 10–25 pp on hotpotqa — does **not** reproduce. The cross-model agreement predicted in the runbook is contradicted by these numbers.

Per-task at K=2 V=3 (the regime where JointQK was supposed to dominate):

| Task | JointQK k2v3 | TurboQuant k2v3 | Δ |
|---|---|---|---|
| hotpotqa | 58.62 | 61.86 | −3.24 |
| musique | 28.57 | 30.18 | −1.61 |
| 2wikimqa | 51.55 | 45.85 | **+5.70** |
| narrativeqa | 27.80 | 29.54 | −1.74 |

JointQK only wins on 2wikimqa (and by ~6 pp). On hotpotqa specifically — the headline Qwen3 task — TurboQuant wins by ~3 pp on Llama, the *opposite* sign of the Qwen3 finding.

## KIVI

- int4 (43.78) is competitive with FP (44.31) and most compressed methods.
- int3 (42.38) is acceptable but loses ~2 pp.
- int2 (37.37) collapses, losing ~7 pp; particularly bad on hotpotqa (−13.6), musique (−13.0), repobench-p (−12.3).

## Per-task observations

- **trec**: high variance across configs (range 20.06 to 44.00). 200 samples with classification labels means a few label flips swing F1 a lot. jointqk_k2_v2 = 44.0 and turboquant_k3_v2 = 38.5 are outliers; treat with caution.
- **multi_news**: essentially flat (~27 across all configs including FP). Task is at a ceiling for the model + context — not method-distinguishing.
- **triviaqa**: tightly clustered ~91–92 except kivi_int2 (85.3). Suggests this task is "easy" for any reasonable compression.
- **lcc**: jointqk at K=2 takes a steep hit (37.23 / 35.97) while TurboQuant K=2 (~48), KIVI int2 (44.3), and FP (51.4) hold up. This is the inverse of the multi-doc-QA story — JointQK hurts most on code.

## Wall-clock

- Capture: 23 min/shard (4 GPUs)
- Stats: 4 min
- Build: 1.5 min
- Sweep run2: ~7 h (1 job/GPU on 4 GPUs)
- **Total: ~8 h**

## Caveats

- 3 lcc cells were retried once (turboquant_k3_v2_lcc, jointqk_k3_v3_lcc, jointqk_k4_v2_lcc) due to OOM in attempt 0 of run2. All passed on attempt 1.
- One run1 attempt at 2 jobs/GPU produced a cascade of OOMs; abandoned in favor of run2 at 1 job/GPU. The 23 cells from run1 that did succeed are preserved (idempotent skip kept them).
- The kvpress evaluator wrote run2 results into a `/1/` subdir under each pre-existing run1 output dir; the F1 grid above picks the most recent metrics.json per cell.
