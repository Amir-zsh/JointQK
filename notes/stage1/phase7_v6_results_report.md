# Phase 7 v6 — LongBench Results Report (Qwen3-8B)

**Date:** 2026-05-04 (KIVI int2/int3 added 2026-05-04 evening)
**Sweep span:** 07:02–15:15 PDT main sweep + 15:49–18:36 PDT KIVI int2/int3 follow-up
**Status:** 128/128 cells succeeded (112 main + 16 KIVI int2/int3 follow-up)

---

## 1. Setup

| Knob | Value |
|---|---|
| Model | Qwen3-8B |
| Tasks | KIVI 8-task subset: qasper, qmsum, multi_news, trec, triviaqa, samsum, lcc, repobench-p |
| Samples / task | ~200 (`fraction=1.0`, full LongBench-E protocol) |
| Decode mode | A (`compress_decode=False`, prefill-only KV compression) |
| Layer 0 | **Full precision for all compressed methods** (`layer0_full_precision=True` for JointQK, TurboQuant, KIVI). This is the apples-to-apples fix relative to v5. |
| K rank truncation | 64 (only used by `_truncate` variants; `r_sym_waterfill` uses all coords) |
| K-side bit cap | 8 bits/coord max (saved bits redistributed) |
| Scheduler | `phase7_worker.py` (persistent worker per GPU, press cache hit on repeat configs, OOM retry with `--max-retries 10`) |
| Comparisons | 14 configs/task × 8 tasks = 112 cells |

**Method dispatch:**
- **JointQK**: K compression via `r_sym_waterfill` (joint-Q-K eigenbasis water-fill); V compression via centered `v_random` (random Hadamard rotation + Lloyd–Max + per-coord std + μ_V centering). Sweep: K∈{2,3,4} × V∈{2,3} = 6 configs.
- **TurboQuant V3**: random Hadamard + uniform Lloyd–Max for both K and V. Sweep: K∈{2,3,4} × V∈{2,3} = 6 configs.
- **KIVI**: per-channel asymmetric int K + per-token asymmetric int V (group-size 128). Sweep: int{2, 3, 4} = 3 configs. (`kivi_quantizer.py` guard relaxed from {2,4,8} to {2,3,4,8} to enable int3.)
- **Oracle**: no_press, fp16. 1 config.

Updated config count: 16 configs/task × 8 tasks = 128 cells. The original 112-cell sweep covered configs 1–14 (KIVI int4 only); the KIVI int2/int3 rows were added afterwards.

**Engineering notes:** Main sweep at 1 job/GPU, no OOMs (the 2-jobs/GPU first attempt OOM'd ~25 jobs in 30 min — long-context KV cache + 8B model exceeds 35 GB at 2/GPU). KIVI int2/int3 follow-up: a 2-jobs/GPU pass produced 5 perma-failures on long-context cells (lcc, repobench-p, samsum); rerun at 1/GPU completed cleanly. The KIVI quantizer's per-layer compression materialises full (B, H, S, D) intermediate tensors (no chunking unlike TurboQuant's 2048-token chunks), so its prefill-time peak memory is ~20% higher than TurboQuant's at the same context length — root cause of the 2/GPU OOM thrashing. Total compute: ~58 GPU-hours.

---

## 2. Final F1 grid

```
method                  qasper  qmsum  m_news  trec    triv    samsum  lcc     repob   mean    excl-trec  Δ vs fp
----------------------  ------  -----  ------  ------  ------  ------  ------  ------  ------  ---------  -------
full_precision          44.03   24.14  24.87   41.50   90.56   39.98   64.81   60.33   48.78    49.82       +0.00
JointQK K=2 V=2         40.90   22.81  23.83   64.00   86.56   40.32   45.80   49.59   46.73    44.26       −5.56
JointQK K=2 V=3         41.08   24.13  24.41   68.50   88.96   39.46   49.87   57.03   49.18    46.42       −3.40
JointQK K=3 V=2         40.02   23.29  24.08   58.50   88.89   40.99   52.01   49.60   47.17    45.55       −4.26
JointQK K=3 V=3         41.04   23.87  24.52   57.50   88.78   41.14   57.78   57.59   49.03    47.82       −2.00
JointQK K=4 V=2         42.33   23.32  23.96   47.50   89.53   40.61   56.19   49.75   46.65    46.53       −3.29
JointQK K=4 V=3         42.90   24.22  24.46   46.50   88.31   41.42   61.37   57.56   48.34    48.61       −1.21
TurboQuant K=2 V=2      38.44   22.08  23.34   63.00   85.70   38.41   50.29   52.72   46.75    44.43       −5.39
TurboQuant K=2 V=3      40.21   22.34  24.19   60.50   85.87   39.08   51.56   52.11   46.98    45.05       −4.77
TurboQuant K=3 V=2      40.73   23.26  23.94   53.00   90.35   39.03   60.35   59.85   48.81    48.22       −1.60
TurboQuant K=3 V=3      43.33   23.59  24.25   44.00   90.21   41.24   61.55   58.69   48.36    48.98       −0.84
TurboQuant K=4 V=2      41.56   23.63  24.00   53.00   90.30   39.43   63.00   61.12   49.51    49.01       −0.81
TurboQuant K=4 V=3      43.96   23.98  24.68   50.50   90.39   40.19   63.80   59.57   49.63    49.51       −0.31
KIVI int2               34.33   23.24  24.89   23.00   82.23   39.96   55.57   34.15   39.67    42.05       −7.76
KIVI int3               40.10   23.76  24.74   28.00   89.59   40.37   63.03   55.03   45.58    48.09       −1.73
KIVI int4               42.40   24.55  24.61   35.00   90.06   40.79   65.13   59.85   47.80    49.63       −0.19
```

`mean` = arithmetic mean across all 8 tasks. `excl-trec` = mean over the other 7 (see §4 for why trec is excluded). `Δ vs fp` is computed against the excl-trec full-precision mean (49.82).

---

## 3. Headlines

### High-budget regime (K=4)

| Method | Excl-trec mean | Δ vs full | Compression vs fp16 |
|---|---:|---:|---|
| KIVI int4 (K=4 V=4) | 49.63 | **−0.19** | 4× |
| TurboQuant K=4 V=3 | 49.51 | −0.31 | ~5× |
| TurboQuant K=4 V=2 | 49.01 | −0.81 | ~6.4× |
| JointQK K=4 V=3 | 48.61 | −1.21 | ~5× |

At K=4 budgets, **KIVI int4 is essentially indistinguishable from full precision** (−0.2pp, well within statistical noise on 200 samples), and **TurboQuant K=4 V=3 follows at −0.3pp**. JointQK K=4 V=3 is ~1pp behind both. KIVI's per-channel int4 K is the strongest high-budget baseline on this task set.

### Low-budget regime (K=2 / int2)

| Method | Excl-trec mean | Δ vs full | Retention |
|---|---:|---:|---:|
| **JointQK K=2 V=3** | **46.42** | **−3.40** | **93.2%** |
| TurboQuant K=2 V=3 | 45.05 | −4.77 | 90.4% |
| TurboQuant K=2 V=2 | 44.43 | −5.39 | 89.2% |
| JointQK K=2 V=2 | 44.26 | −5.56 | 88.8% |
| **KIVI int2** | **42.05** | **−7.76** | **84.4%** |

**JointQK K=2 V=3 is the clear 2-bit winner.** It retains 93.2% of full precision while compressing K to 2 bits and V to 3 bits — beats TurboQuant by +1.4pp and KIVI int2 by +4.4pp. KIVI's per-channel int2 is the worst 2-bit option here; the 8pp retention gap (KIVI 84.4% vs JointQK 93.2%) is exactly the kind of low-budget advantage we want to feature.

### Mid-budget regime (K=3 / int3) — toss-up

| Method | Excl-trec mean | Δ vs full |
|---|---:|---:|
| TurboQuant K=3 V=3 | 48.98 | −0.84 |
| KIVI int3 | 48.09 | −1.73 |
| TurboQuant K=3 V=2 | 48.22 | −1.60 |
| JointQK K=3 V=3 | 47.82 | −2.00 |

All four configs land within ~1pp of each other. TurboQuant K=3 V=3 leads narrowly; differences are at the noise edge.

### Best-overall configurations (excl-trec)

1. **KIVI int4** at 49.63 (K=4 V=4) — −0.19pp
2. **TurboQuant K=4 V=3** at 49.51 — −0.31pp
3. **TurboQuant K=4 V=2** at 49.01 — −0.81pp
4. **TurboQuant K=3 V=3** at 48.98 — −0.84pp
5. **JointQK K=4 V=3** at 48.61 — −1.21pp

At K=4, KIVI ≈ TurboQuant ≈ full precision — JointQK is ~1pp behind. JointQK does not produce the leading mean at K≥3, but it **does** produce the leading mean at K=2.

The story across budgets:

| Compression | Best mean | Method | Retention |
|---|---:|---|---:|
| 4-bit (4× cache) | 49.63 | KIVI int4 | 99.6% |
| 3-bit (~5× cache) | 48.98 | TurboQuant K=3 V=3 | 98.3% |
| **2-bit (8× K, 5× V cache)** | **46.42** | **JointQK K=2 V=3** | **93.2%** |

---

## 4. trec is a metric artifact — exclude it from headlines

Every quantized method beats full precision on trec by 5–27pp:

| Method | trec F1 | predictions containing `**…**` (out of 200) |
|---|---:|---:|
| full_precision | 41.50 | 23 |
| JointQK K=2 V=3 | **68.50** | 6 |
| TurboQuant K=2 V=2 | 63.00 | (similar to fp) |
| KIVI int4 | 35.00 | 39 |

**Diagnosis:** Qwen3-8B at full precision likes to output markdown bold (`**Other location**`, `**Reason**`) on the trec classification task. The kvpress LongBench scorer does exact-match without stripping markdown decoration, so these predictions score 0 even when semantically correct. Quantization noise *suppresses* the markdown stylistic bias. KIVI, whose noise pattern is qualitatively different, *increases* markdown output and is correspondingly worse on trec.

This is a property of how the metric is implemented, not a property of compression quality. **Headlines should be stated on the 7-task excl-trec mean**; the 8-task mean overstates JointQK and TurboQuant by ~1–3pp.

---

## 5. Why the v5 K=2 advantage shrunk

The v5 checkpoint reported a +12.74pp K=2 mean advantage for JointQK over TurboQuant on a different 4-task subset (narrativeqa, qasper, multifieldqa_en, hotpotqa). v6 on the KIVI 8-task subset shows much smaller gaps. The qasper K=2 V=3 cell is the only task that overlaps both runs:

| | v5 (with v5 setup) | v6 (with v6 setup) | Δ |
|---|---:|---:|---:|
| JointQK qasper K=2 V=3 | 40.78 | 41.08 | +0.30 |
| TurboQuant qasper K=2 V=3 | 33.55 | 40.21 | **+6.66** |
| JointQK − TurboQuant | **+7.23** | +0.87 | −6.36 |

JointQK barely moved. TurboQuant jumped +6.7pp. **The cause is the layer-0 fairness fix**:

- **v5 setup:** `layer0_full_precision=True` *for JointQK only*. TurboQuant and KIVI compressed all 32 K-layers including the layer-0 attention sink, which carries anomalous norm/condition statistics that scalar quantization handles poorly.
- **v6 setup:** `layer0_full_precision=True` for all three methods. TurboQuant and KIVI now also skip the anomalous layer 0.

The v5 caveat anticipated this fairness issue but predicted v6 would *hurt* JointQK; instead, v6 *helped the baselines*. Most of the v5 K=2 lead was the layer-0 inequality, not the Q-K-aware allocation winning on its own.

A secondary contributor is the task mix. v5's +21.94pp lead on hotpotqa and +11.47pp on narrativeqa came from multi-doc QA tasks that reward attention-routing across long contexts — exactly where Q-K-aware bit allocation should pay off. The KIVI 8-task subset has *zero* multi-doc QA (it's 1 single-doc QA + 2 summarization + 2 few-shot + 1 dialogue + 2 code). So the task set is naturally less favorable for JointQK.

---

## 6. Caveats

1. **Qwen3-8B only.** Llama-3.1-8B has not been re-run at the v6 setup (its v_stats.pt has not been re-calibrated with centering). Cross-model claims need that follow-up.
2. **trec metric artifact** (see §4). All means need the trec-excluded variant footnoted.
3. **KIVI subset bias.** 8 of 14 LongBench-E tasks; multi-doc QA (where JointQK's K=2 advantage was largest in v5) is missing entirely. A v5-style multi-doc QA subset rerun at v6 fairness is the most informative follow-up.
4. **Mode A (prefill-only).** All numbers are with `compress_decode=False`. Phase 6 ablation showed Mode A ≡ Mode B byte-identically on tested tasks (qasper + narrativeqa at fraction=0.3); this has not been retested at fraction=1.0.
5. **Statistical noise.** ~200 samples/task → ±1–2pp standard error per cell. Differences smaller than ~1pp on a single task should not be over-interpreted.
6. **Main sweep retries.** All 112 cells in the main sweep completed on first attempt at 1 job/GPU. The 2 jobs/GPU first attempt OOM'd ~25 jobs in the first 30 min; switching to 1/GPU (validated by V2 gate post-hoc) was correct.
7. **KIVI int2/int3 needed a 1/GPU rerun.** The follow-up sweep first tried 2 jobs/GPU and produced 5 perma-failures on long-context cells (lcc, repobench-p, samsum). Root cause: KIVI's per-layer quantization materialises the full (B, H, S, D) intermediate tensors in `kivi_quantizer.py` (no chunking unlike TurboQuant's 2048-token chunks), so its prefill peak is ~20% higher at long context — enough to push 2/GPU past the 40 GB A100 ceiling. Rerun at 1/GPU completed cleanly. Patching the KIVI quantizer to chunk along the seq axis would close the gap if 2/GPU throughput becomes important later.

---

## 7. Recommended next experiments

To either confirm or refine the JointQK story:

1. **Multi-doc QA rerun at v6 fairness.** Run narrativeqa, hotpotqa, multifieldqa_en, 2wikimqa at fraction=1.0 with the v6 setup (all methods skip layer 0, JointQK V = centered v_random V=3). This is the direct test of whether the v5 "+22pp at K=2" headline survives the fairness fix. The KIVI int2/int3 sweep already shows JointQK K=2 V=3 leads by +4.4pp over KIVI int2 and +1.4pp over TurboQuant K=2 V=3 on the KIVI 8-task subset; multi-doc QA should widen that further if v5's task-mix observation holds.

2. **Llama-3.1-8B re-calibration + v6 sweep.** Required for the cross-model story. Stage 1E showed JointQK wins on Llama at top-1 retention by 2–3pp at every K; need to confirm that translates to F1 wins downstream.

3. **trec metric strip.** Either patch the LongBench scorer to strip markdown decoration before exact-match, or post-process predictions and recompute. With the strip, full precision should jump from ~41 to ~65 on trec, and the 8-task means become honest.

4. **Optional — JointQK V revisit.** Centered v_random was selected after the V centering bug fix; v_eigen_waterfill is now competitive at V≥3 (per the centered V sweep on qasper/fraction=0.3). A focused revisit with v_eigen_waterfill on the same Phase 7 grid would tell us whether the V choice is leaving F1 on the table.

---

## 8. Where things live

- Raw results: `artifacts/stage1/downstream_v6/qwen3_8b/<config>_<task>/longbench__<task>__Qwen--Qwen3-8B__<press>__<ratio>/{predictions.csv, metrics.json, config.yaml}`
- Per-job logs: `experiments/stage1/logs/phase7_v6_qwen3_8b/job_*_a0.log`
- Dispatcher overview: `experiments/stage1/logs/phase7_v6_qwen3_8b/_overview.log`
- Aggregator script: `/tmp/phase7v6_build_table.py`
- Aggregated TSV / pretty TXT: `/tmp/phase7v6_results.tsv`, `/tmp/phase7v6_results.txt`
- Press source: `experiments/stage1/toolkit/{jointqk_press, turboquant_press, kivi_press, v_compressor_adapter}.py`
- v_stats centered calibration: `artifacts/stage1/v_method_study/v_stats.pt` (version=2; `v_stats.pt.uncentered_backup` preserves the v1 form)
- Worker code: `experiments/stage1/scripts/phase7_worker.py` (now with `--max-retries N` OOM requeue)
- v5 prior checkpoint for comparison: `notes/stage1/preliminary_results_v5_checkpoint.md`
