# NeurIPS 2026 Submission Plan — JointQK KV-Cache Compression

## Context

Stage 1E is complete. The headline result — **JointQK WaterFill** (`r_sym_waterfill`: orthogonal eigenbasis of $(\Sigma_Q \Sigma_K + \Sigma_K \Sigma_Q)/2$ + reverse water-fill bit allocation) — achieves 0.860 prefill top-1 / 0.944 decode top-1 at 3–4 bits/coord on Qwen3-8B / LongBench-E (24-example bundle), beating prior bases by 8–14 pp top-1 across every bit budget. Generalization (cross-task spread 0.2 pp; LOO std ≤ 0.009) and decode-phase transfer are both validated.

This plan converts that into a **NeurIPS 2026 main-track submission**. Deadline ~2 weeks (target May 14–22, 2026); user is solo, full-time, 4 GPUs (0–3). Paper title and method name are locked: *"Spend Bits Where Attention Reads: A Joint Q-K Energy Basis for KV-Cache Quantization"*, method = **JointQK**. NeurIPS template lives at `paper/Formatting_Instructions_For_NeurIPS_2026/neurips_2026.tex`.

Three claim-defining gaps must be closed for a competitive submission:

1. **Cross-architecture transfer** — Qwen3-8B alone is too narrow. We add Llama-3.1-8B.
2. **End-to-end downstream task scores** — current metrics are attention-internal (top-1, geo). Reviewers need LongBench task accuracy + RULER (∞Bench likely descoped to future work).
3. **Published baseline comparison** — current baselines are in-house (V3/CCA variants). Add KIVI as the canonical published K-cache quantizer.

Memory-savings claims are kept analytical (bytes saved per key); wall-clock latency is deferred to a follow-up systems paper, with a brief discussion in the main paper.

## Scope decisions

**In:** Qwen3-8B + Llama-3.1-8B; LongBench (full, not 24-example bundle) + RULER subset; baselines = full-precision + TurboQuant + KIVI; method = **JointQK WaterFill** (ours). Paper draft in `paper/main/`.

**Comparison set (locked, 2026-05-03):** the paper compares exactly four configurations — `full`, `KIVI`, `TurboQuant`, and `JointQK` (ours). CCA-Orth WaterFill and Q-Eigen WaterFill were intermediate basis-design ablations during Stage 1E; CCA was empirically dominated by JointQK and Q-Eigen is an in-house non-published baseline. They stay in the methodology study (`run_cca_vs_waterfill_study.py`) as documentation of the path to the final basis, but **do not appear in paper figures or tables**.

**Out (descoped to future work, called out explicitly in the paper):** ∞Bench, larger Llama (70B), Mistral / Mixtral, wall-clock latency benchmarks, end-to-end systems integration. Theoretical optimality proof is appendix-only (sketch + empirical validation), not main-text claim.

## Workstreams

The plan is structured as **four parallel workstreams** (W1–W4) to compress wall-clock duration. Each has its own gate; nothing depends on Stage 2/3 of the original research roadmap.

### W1 — Llama-3.1-8B reproduction (Days 1–6)

Confirm JointQK WaterFill wins on a second model family. Risk: if it doesn't, the paper scope shrinks dramatically.

**Steps:**
1. **Day 1–2: Adapt calibration to Llama-3.1-8B.** The driver `experiments/stage1/run_cca_vs_waterfill_study.py` already factors model loading from compression logic. Verify it runs unmodified on Llama-3.1-8B (same GQA shape, head dim 128, just different `model_id`); add a `--model-id` flag if not already present.
2. **Day 2–3: Run E3 on Llama** at b ∈ {2, 3, 4} with the 9 methods from Stage 1E. Use the same LongBench-E 24-example bundle for direct comparison.
3. **Day 3–4: Run E5 (decode-phase) on Llama.** Reuse `make_e5_charts.py` to regenerate charts.
4. **Day 4–5: Skip E2/E4 on Llama** unless W1 result surprises. E4 generalization is a Qwen3-8B claim; we can argue it transfers without re-running the 3×3 cross-task LOO.
5. **Day 5–6: Comparative chart** — overlay Qwen3-8B and Llama-3.1-8B top-1 bit-budget curves. New chart: `report_charts/cross_model_b_sensitivity.png`.

**Gate:** JointQK WaterFill wins at every b ∈ {2, 3, 4} on Llama, with a margin ≥ 5 pp top-1 over the next-best method at b=3. If it doesn't win, escalate to user before continuing — this is a paper-defining outcome.

**New files:**
- `experiments/stage1/scripts/launch_llama_e3_e5.sh` — orchestrator for E3+E5 on Llama
- `experiments/stage1/scripts/make_cross_model_chart.py` — overlay chart
- `artifacts/stage1/cca_vs_waterfill_study/llama31_8b/` — separate output directory for Llama runs

**Existing reused:** `run_cca_vs_waterfill_study.py`, `build_method_compressor` (no new methods needed), `gates/gate_e3.py` (auto-discovers methods).

### W2 — Downstream task evaluation (Days 3–10)

This is the highest-leverage missing experiment. Reviewers will reject a 0.860 attention-top-1 paper without task-level accuracy.

**Approach:** Hook the prefill key-quantization path into the model's actual forward pass during generation, then run LongBench + RULER tasks. We have two paths; we'll attempt path A first.

**Path A (preferred): Vendor `kvpress` adapter.** kvpress is already in this repo (`kvpress/`); it's NVIDIA's library for KV-cache compression with a press-style hook into HuggingFace generate. Wrap JointQK as a kvpress press; this gives us `transformers.AutoModel.generate()` with the compressed cache for free.

**Path B (fallback): Custom monkey-patch.** Wrap the attention forward pass to compress K after prefill, store compressed cache, decompress on each decode step. More fragile but doesn't depend on kvpress integrating cleanly.

**Steps:**
1. **Day 3–4: kvpress integration spike.** Read kvpress's press base class. Wrap `r_sym_waterfill` (and TurboQuant, KIVI) as kvpress presses. Validate by reproducing the Stage 1E top-1 numbers via this path on the 24-example bundle.
2. **Day 4–6: LongBench full evaluation.** All 12 LongBench tasks × 2 models × {full-precision, TurboQuant b=3, KIVI b=3, JointQK b=3} = 96 runs. Use kvpress's existing LongBench evaluator if it ships one; otherwise write a thin runner that reuses the LongBench-provided scorers.
3. **Day 6–8: RULER subset.** RULER's NIAH (single needle, multikey, multivalue) at lengths 4k, 8k, 16k. Likely descope to NIAH-single-needle if time pressed.
4. **Day 8–10: Bit-budget sweep on best tasks.** For 2–3 representative tasks, run b ∈ {2, 3, 4, 8} to show the quality–bits tradeoff.

**Gate:** JointQK at b=3 retains ≥ 90% of full-precision LongBench task score on both models, vs ≥ 85% for TurboQuant and KIVI at the same budget.

**New files:**
- `experiments/stage1/eval/longbench_runner.py` — generates predictions with each compression method, scores against gold
- `experiments/stage1/eval/ruler_runner.py` — same for RULER NIAH subset
- `experiments/stage1/toolkit/jointqk_press.py` — kvpress press wrapping JointQK
- `experiments/stage1/toolkit/kivi_press.py` — KIVI baseline wrapped as kvpress press
- `artifacts/stage1/downstream/{qwen3_8b,llama31_8b}/longbench_summary.json`
- `artifacts/stage1/downstream/{qwen3_8b,llama31_8b}/ruler_summary.json`

**Existing reused:** Calibration bundles already saved per (model, layer, head) in `artifacts/stage1/cca_vs_waterfill_study/calibration/` — feed the same bases into the press wrappers, no recomputation needed.

### W3 — KIVI baseline (Days 5–8, parallel with W2)

KIVI is the canonical comparison: per-channel K + per-token V int4 quantization. Implementation is straightforward (it's a 200-line algorithm); we use K-only since this paper is K-only.

**Steps:**
1. **Day 5: Vendor or reimplement KIVI** as a per-channel int4 K-only quantizer. Produces compressed K and a per-channel scale.
2. **Day 5–6: Wrap as kvpress press** (`kivi_press.py`) for compatibility with W2.
3. **Day 6–8: Run on 24-example bundle for top-1 / geo numbers**, then plug into W2 downstream evals. Match bit budgets carefully: KIVI's int4 = b=4; we run JointQK at the *same effective bytes/token* not just nominal b.

**Gate:** KIVI matches its published top-1 numbers on Qwen3-8B (~0.85 at int4 per the original paper, allowing for our different evaluation harness).

**Existing reused:** kvpress base class; same calibration / metric paths as JointQK.

### W4 — Paper writing (Days 1–17, parallel with all experiments)

The paper draft starts on Day 1, before experiments finish. Sections that are factually stable (intro, related work, method) get drafted first; experiment results plug in last.

**Day 1–2: Skeleton + abstract + intro.** Use `paper/Formatting_Instructions_For_NeurIPS_2026/neurips_2026.tex` as the starting point. New file `paper/main/main.tex`. Draft the abstract using the locked title and the headline numbers from the existing two-pager. Intro: motivation (KV cache dominates serving cost), the per-layer-head calibration pipeline, JointQK as the answer, three contributions (basis derivation; water-fill on basis-diag weights; cross-model + downstream validation).

**Day 3–5: Method section (§3).** Pipeline figure (compression triplet); Lloyd–Max codebook + per-coord scaling; basis families table (Q-Eigen / CCA-Orth / JointQK with their forward maps); allocation families (uniform / truncate / water-fill); the CCA objective-mismatch derivation that motivates JointQK (already in `notes/stage1/stage1e_summary_to_share.md` — port to LaTeX). Pipeline diagram is the only piece of original art; use TikZ.

**Day 6–8: Related work (§2).** KV-cache quantization (KIVI, KVQuant, Atom), KV-cache compaction (H2O, FastGen, expected attention), low-rank / projection methods (Eigen attention, ALISA), random-rotation methods (TurboQuant). One paragraph per category.

**Day 8–13: Experiments (§4).** Plug in W1, W2, W3 results as they finish. Charts: bit-budget sensitivity (single-model + cross-model overlay); cross-task generalization heatmap (Qwen only); LongBench task table; RULER NIAH table. Drop most of E1/E1_2 to the appendix (diagnostic, not load-bearing for the main story).

**Day 13–15: Discussion + limitations + conclusion.** Limitations: K-only (V is open), 2 models (8B-class only), no wall-clock numbers, theoretical optimality is empirically motivated. Future work: V-side compression, larger scales, latency-aware integration.

**Day 15–17: Polish.** Checklist (`paper/Formatting_Instructions_For_NeurIPS_2026/checklist.tex`); broader-impact statement; bibliography pass; figure quality / consistent fonts; ensure 9-page limit.

**New files:**
- `paper/main/main.tex` (the manuscript)
- `paper/main/refs.bib`
- `paper/main/figs/` — each figure is either copied from `artifacts/stage1/.../report_charts/` or a TikZ/pgfplots replot for consistent typography
- `paper/main/sections/` — split sections (`intro.tex`, `method.tex`, `experiments.tex`, etc.) so changes are localized

**Existing reused:** The two-pager, full report (`stage1e_summary_to_share.md`), and per-experiment review notes (`stage1e_cca_vs_waterfill/e*.md`) are the source of all narrative, math, and numbers. Charts in `artifacts/stage1/cca_vs_waterfill_study/report_charts/` are publication-grade for the most part — only the headline figure (bit-budget sensitivity) needs a polished re-export with consistent fonts.

## Critical files

**To create or substantially extend:**
- `experiments/stage1/scripts/launch_llama_e3_e5.sh`
- `experiments/stage1/scripts/make_cross_model_chart.py`
- `experiments/stage1/eval/longbench_runner.py`
- `experiments/stage1/eval/ruler_runner.py`
- `experiments/stage1/toolkit/jointqk_press.py`
- `experiments/stage1/toolkit/kivi_press.py`
- `paper/main/main.tex` and supporting files
- `paper/main/figs/` (new figures + ports of existing artifacts)

**To extend (small changes):**
- `experiments/stage1/run_cca_vs_waterfill_study.py` — accept `--model-id` if not already there
- `experiments/stage1/scripts/make_e3_charts.py`, `make_e5_charts.py` — multi-model overlay support
- `experiments/stage1/gates/gate_e3.py` — Llama-specific thresholds if needed (likely not — auto-discovery handles it)

**To reuse unchanged:**
- `experiments/stage1/toolkit/per_coord_quantization.py` (`build_method_compressor` already supports JointQK)
- `experiments/stage1/toolkit/metric_transform.py` (water-fill solver, basis builders)
- `experiments/stage1/toolkit/quantization.py` (Lloyd–Max)
- `kvpress/` (vendored library — wrap, don't fork)

## Decision points & risks

1. **Llama-3.1-8B doesn't reproduce the JointQK win (W1 gate).** Escalate immediately. Likely root cause: head-dim or norm-statistics differ enough to need recalibration of `r=64` choice or a per-model layer-0 exclusion rule. Falling back to a Qwen-only paper is feasible but weakens the claim.
2. **kvpress integration is harder than expected (W2 path A).** Switch to monkey-patch (path B) — budget 1 extra day. If neither works, descope downstream metrics to perplexity-only on a long-context corpus and explicitly flag this in the paper limitations. This is the worst-case scenario; it weakens but does not kill the submission.
3. **RULER is too compute-heavy.** Drop to NIAH-single-needle at one length (16k) with all bit budgets. Cite the reduced subset in the paper as "due to compute budget."
4. **Page-limit overflow.** NeurIPS 9-page main-text limit is tight. The full E1/E1_2/E4 deep-dives go to the appendix. Main-text experiments section is bit-budget chart + cross-model chart + LongBench table + RULER table + ablation table.
5. **Memory-only systems claim is weak.** Acceptable per scope decision; the paper's abstract should not lead with "X× faster"; lead with quality-at-bits.

## Verification (per workstream)

**W1:** Run `gate_e3` and `gate_e5` against the Llama output directory; confirm JointQK_waterfill is the top-ranked method at every (b, layer-0-excluded) cell. Cross-model overlay chart should show two qualitatively similar curves.

**W2:** For each model × method × bit budget, full-precision and JointQK b=4 task scores within 2 percentage points (sanity); JointQK at b=3 strictly dominates TurboQuant and KIVI at b=3 on ≥ 80% of tasks. Smoke-test one task with `max_new_tokens=8` before running the full sweep.

**W3:** KIVI's reproduced top-1 within 3 pp of its original paper's reported number on Qwen3-8B at int4. If the gap is larger, audit the per-channel scale tracking.

**W4:** Full pdflatex build with no errors; all citations resolve; checklist filled; under 9 pages of main text. Run `chktex` for LaTeX warnings.

**Final pre-submit:** Two-day buffer (Days 17–19) for the inevitable issues — figure rerendering, late number corrections, advisor feedback loop.

## Timeline at a glance

| Days | Workstream | Deliverable |
|---|---|---|
| 1–2 | W1 + W4 | Llama calibration runs + paper skeleton |
| 3–4 | W1 + W2 + W4 | Llama E3 done + kvpress integration spike + intro draft |
| 5–7 | W1 + W2 + W3 + W4 | Llama E5 done + LongBench full-precision baselines + KIVI implemented + method section |
| 8–10 | W2 + W3 + W4 | LongBench downstream done + RULER NIAH started + experiments section |
| 11–13 | W2 + W4 | RULER done + experiments section locked + figures finalized |
| 14–16 | W4 | Polish, related work, limitations, broader impact, checklist |
| 17–19 | W4 | Buffer for advisor feedback, final figure passes, submission |
