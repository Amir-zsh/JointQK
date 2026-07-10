# Approach 5 — Page selection / eviction

**Verdict: the calibrated signal is real and beats every zero-signal baseline;
in-context statistics win when hidden states are available. Niche: offloaded/cold
pages and any setting without runtime hidden states.**

## Idea
Instead of (or in addition to) smaller pages, keep FEWER pages: score 64-token pages
by predicted relevance, evict the rest. Static variant uses only calibration stats
(μ̄_q); query-aware variant would score per decode step.

## Offline probe (contested-mass recall@25% of pages, selection rows)
oracle 53.0 · **incontext_mu 50.0** · omega_mean 44.3 (frozen as press mode) ·
quest_true 42.9 · omega_max 40.7 · quest_mu 40.6 · random 23.8.
Two pre-registered rules fired: score_mode = omega_mean; **quest gate NOT fired** —
true-query Quest boxes LOSE to the static calibrated prior at page granularity, so the
query-aware press was dropped by rule. Oracle at 53% shows contested mass is dispersed:
selection headroom is inherently narrow.

## Bench (35 cells, ratio 0.50, 5-task, row-paired bootstraps)
FP 48.40 · expected_attention 46.98 · snapkv 45.66 · **omega_page 43.33** ·
omega_random_page control 41.04 · streaming_llm 38.99 · knorm 38.45 · random 6.46.
| contrast (omega_page −) | Δ | 95% CI | verdict |
|---|---|---|---|
| random-page control | +2.52 | [+0.58, +4.52] | SIG — signal beats structure |
| streaming_llm | +4.54 | [+2.42, +6.68] | SIG |
| snapkv | −1.38 | [−3.14, +0.33] | tie |
| expected_attention | −2.39 | [−4.47, −0.31] | SIG against |
Primary bar (≥ EA) failed → claim ladder resolved to the pre-agreed fallback:
**zero-runtime-statistics selection is competitive and is the only option where
hidden states are gone.** On OOD code (lcc) random pages ≈ ω pages (known weakness).

## Artifacts
`kvq/presses/omega_page_press.py` (+ registry line in gitignored vendor kvpress),
probe `pipelines/page_quant/probe_page_selection.py` →
`artifacts/page_quant2/page_selection_probe.json`, cells in `artifacts/bench_evict/`.

## Open directions
- Hybrid: in-context scorer for resident pages + ω for offloaded pages in one budget.
- Selection × compression joint budget (evict vs coarsen trade — untried).
- Serving-side: ω page skip in the sglang engine (Quest framework in vendor/OSCAR
  accepts new algorithms; ω needs no per-page min/max metadata, cheaper than Quest).
