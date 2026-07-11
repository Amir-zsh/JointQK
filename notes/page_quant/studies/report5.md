# report5 — pgq5: Qwen3-8B transfer of the pgq4 format (closeout)

**Study:** plan5.md (registered 2026-07-10, branch `pgq5`), reduced by the
project-direction pivot (see plan6.md) to the Stage-B transfer question.
The thinking-mode closed-loop stages (C0/C1/C2) are **parked** — full design
preserved in plan5.md; the Mode-B' `_qlen` crop fix (tracker pgq5-1) landed.

## Verdict

**The pgq4 format is model-generic.** With statistics refit on 12 Qwen
calibration rows (format constants byte-frozen from the Llama study), the
frozen winner `pgq_proflmrw_rdo@2.0`:

- **beats the TurboQuant incumbent by +6.30 F1 pooled** (row-paired 10k
  bootstraps, 95% CI [+4.21, +8.49], SIG on the same-V k2_v2; +5.18 SIG on
  the V-advantaged k2_v3) — a LARGER margin than on Llama (+0.34 tie);
- **retains 0.952 of full-precision** (48.29 vs 50.70 mean5; −1.73
  [−3.07, −0.38]) — above the original 0.90 T1 bar even though A5-2 demoted
  it to descriptive (the incumbent retains only 0.831 on Qwen);
- **reproduces the recency-window effect**: rw − plain = +1.66 [+0.81,
  +2.50] SIG pooled, driven by lcc (+6.99: 56.42 → 63.41), exactly the pgq4
  W3 structure.

| task | proflm@2.0 | proflmrw@2.0 | FP | TQ-K2V2 |
|---|---|---|---|---|
| lcc | 56.42 | 63.41 | 64.81 | 50.29 |
| musique | 31.89 | 32.49 | 33.47 | 27.51 |
| 2wikimqa | 42.85 | 44.17 | 48.43 | 40.72 |
| qasper | 37.12 | 37.76 | 43.06 | 37.65 |
| hotpotqa | 63.97 | 63.60 | 63.75 | 54.47 |
| **mean5** | 46.45 | **48.29** | 50.70 | 42.13 |

Protocol: v7-compatible (fraction 1.0, layer-0 fp16, V = TurboQuant 2b,
compact8 exclude-train, 500-row tasks, honest all-in rates; Qwen held-out
rates 2.0000 plain / 2.0636 rw). References from the May downstream_v7 Qwen
sweep, row-paired on shared `_id`s. Cells carry bundle sha8 cd6a3d41
(freeze F-B `frozen_choices_pgq5.json`).

## Stage A (calibration + gates)

- Data: lambda6 pull (16 surviving compact8 train raws, 73 GB + 42 GB
  02_stats + 3 bundles; 10 duplicate stats files verified content-identical
  and deduped). Roles: 12 fit + 4 selection
  (`pgq5_qwen_compact8_16/roles.json`).
- Stats source `qpca_qwen3_8b_longbench_compact8_n400.pt` (sigma_q/sigma_k
  [36,8,128,128] verified); μ denominator resolved tokens·group at trace
  ratio 1.0002.
- **Sharing rule fired the other way on Qwen: prof_share = HEAD** (penalty
  6.19% > 3%; Llama 1.9% → layer). Qwen kv-heads are more
  profile-heterogeneous; the loader supports per-head natively.
- Sink-position check PASS (median 11.7% attention mass at positions 0-3;
  no displaced structural sink in 160 probes) — the position-derivable
  escape assumption holds.
- Held-out gates PASS at every arm (ovf 0, sinkCE 0.003, normR 0.976-0.988).

## Amendments (both pre-F1, recorded in plan5.md)

- **A5-1 (G-P4′):** the ratio gate tripped on its DENOMINATOR — the codec's
  absolute logit_err transferred fully (0.002615 Qwen vs 0.002716 Llama,
  −3.7%) while the TQ proxy improved on Qwen (0.0163 vs 0.0262), inflating
  ρ_qwen/ρ_llama to 1.55 > 1.25. Amended admission: numerator transfer
  within 10%. **Methods lesson: ratio gates are denominator-sensitive; on
  cross-model transfer, gate on the numerator and report the ratio.**
- **A5-2 (T1):** FP-retention bar demoted to descriptive; T2 (Δ vs
  turboquant_k2_v2 CI-lower > −1.0) is the hard gate. Justification: TQ has
  BETTER proxy error yet WORSE F1 retention on Qwen — the logit_err↔F1
  exchange rate is model-dependent (`pgq5_gp4prime_gate.json`).

## Artifacts

`artifacts/page_quant2/`: frozen_choices_pgq5.json, pgq5_refs.json,
pgq5_heldout_report.json, pgq5_gp4prime_gate.json (+ gp4prime_{llama,qwen}),
pgq5_bootstraps.json, pgq5_math500_split.json (parked C-stage split),
pgq5_bundle__qpca_unc__qwen3_8b_compact8train12.pt (sha8 cd6a3d41);
`artifacts/bench_pgq/qwen3_8b/` 10 cells. Tools:
pipelines/page_quant/pgq5_bootstrap.py, score_pgq_candidates
--model-tag/--with-tq, model-tag parametrization across
fit_ec_bundle/phase1_empirics/fit_pgq4_bundle/launch_pgq_longbench.sh
(Llama defaults byte-identical, frozen sha8 reproduced in dry-run).

## Parked (deferred directions)

C0/C1/C2 thinking-mode closed-loop study (math500 split registered, scorer
amendment P1-a documented — vendor boxed extractor caps a perfect model at
~79%, robust balanced-brace metric specified; enable_thinking template
verified; 3-line evaluate.py wiring identified). Phase-asymmetric allocation
signals (see plan6 signal policy).
