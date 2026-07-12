# report6 — pgq6: merged-page codec (n→m token merging as an RDO rung)

**Study:** plan6.md (registered 2026-07-11, branch `pgq6`), protocol v7.
Format: `MergedPageCompressor` (`pgq_mrg*`) — contiguous-run Ward merging in
the qpca_unc code space as a per-page rung jointly RDO'd with the pgq4 width
profiles, fixed-byte pages, replication-form eval (exactly the bias form
k̄/v̄/log c the deployment stores; d+1 fold keeps FlashAttention unmodified).
No refit — runs off the frozen pgq4 Llama bundle (ff390304).

## Verdicts (registered tree, plan6 §4)

1. **Sub-wall gate: FAIL (clean negative).** At 1.5 b/c per original token,
   `mrglm@1.5` logit_err 0.00482 does not beat the pgq3 closure bars
   (ecu 0.00318, rvq 0.00405). At 1.0: 0.00894 vs ecu 0.00646. **The ~1.5 b/c
   wall extends to the token axis at this ladder** — merging does not open a
   sub-wall rate band vs entropy coding.
2. **Weak dominance: CONFIRMED.** Merge-as-rung never hurts and rescues the
   width codec's low-rate collapse: vs `proflm` at matched rates —
   1.0: 0.00894 vs 0.01233 (−28%); 1.25: 0.00679 vs 0.00728 (−7%);
   1.5/2.0: exact ties. The "worst case = unmerged" design argument holds
   empirically at every rate.
3. **Parity at 2.0: F1 TIE on every task — merging is free.** W1 (5 v7
   tasks, `mrglmrw@2.0` vs the pgq4 incumbent `proflmrw@2.0`, row-paired 10k
   bootstraps, pgq6-3-fixed parser): pooled −0.23 [−0.78, +0.25]; every task
   individually a tie (qasper −0.74 [−2.29, +0.23]); hotpotqa is
   prediction-identical. Mechanism (telemetry): at 2.0 the RDO disengages
   merging entirely (100% M64, 0% row reduction) — the codec reduces to
   proflmrw when bits are ample. **Adopting mrg as the format default costs
   nothing and buys the low-rate regime.**
4. **Redundancy-adaptivity: NOT SUPPORTED.** At 1.25 the merge engagement is
   near-uniform across all 8 calibration tasks (M32 fraction 30–39%; code
   [repobench-p] no higher than prose; max is passage_retrieval_en) —
   merging responds to RATE PRESSURE, not content type, under the
   adjacent-Ward objective. The per-task F1 structure of pgq4 (lcc recency)
   is NOT mirrored by a per-task merge structure.
5. **Oracle ablation: observed attention has NEGATIVE headroom** (−15% at
   both 1.25 and 1.5): realized-attention weighting + top-decile protection
   makes allocation strictly worse than the phase-symmetric RDO. The
   deployable signal set (calibration + position) is not leaving measurable
   quality on the table — empirical support for the plan6 signal policy and
   the user's long-generation objection to observed-attention methods.
6. **Value-coupling: PASS (exactness argument verified).** Direct
   attention-output error of mrg ≤ proflm at matched rate (ratio 0.95 @1.5,
   1.00 @2.0) — no hidden value-side penalty from merging
   (`pgq6_output_check.json`).

## W1 table (Llama, v7, honest rates; mrglmrw@2.0 heldout rate 2.1253)

| task | mrglmrw@2.0 | proflmrw@2.0 (pgq4) | Δ row-paired |
|---|---|---|---|
| lcc | 50.10 | 50.05 | +0.05 [−0.41, +0.53] |
| musique | 30.49 | 30.97 | −0.48 [−1.32, +0.05] |
| 2wikimqa | 45.11 | 45.11 | +0.00 [−2.00, +2.00] |
| qasper | 42.18 | 42.92 | −0.74 [−2.29, +0.23] |
| hotpotqa | 60.34 | 60.34 | identical predictions |
| **mean5** | **45.64** | 45.88 | −0.23 [−0.78, +0.25] tie |

Side discovery of this wave (pgq6-3, fixes_to_apply.md): the paired
bootstrap tool had been concatenating multi-answer golds since the pgq4
era; fixed and validated against every cell's metrics.json. Corrected
pgq4/pgq5 claims are appended to report4 (headline Tier-E tie HOLDS;
"SIG over rvq" downgrades to a tie) and folded into report5.

## Format facts learned

- **M16 (4:1) is never chosen** at 1.25–2.0 — two page size classes {S, S/2}
  suffice; the deployment story simplifies.
- Row reduction at 1.25: 15–20% across tasks (attention-FLOPs/page-count
  saving in the regime where merging engages).
- Held-out gates all pass on mrg arms (rate 1.4960/1.9963 exact, ovf 0,
  sinkCE 0.002, normR 0.959/0.978).
- Perf: hierarchy steps must stay sync-free (pgq6-2; 25× bench slowdown from
  per-step `nonzero` syncs — fixed, 71 ms/(l,h) warmed). Further speedup is
  kernel-port work (per-(l,h) launch overhead dominates).

## Interpretation

The merged-page format is the right DEFAULT shape for the codec: it
subsumes the pgq4 winner exactly where the pgq4 winner is optimal, extends
the operating range downward (−28% proxy error at 1.0), and carries a
serving win (fewer rows) when rate pressure engages it. What it does NOT do
is break the entropy-coding wall below 1.5 b/c — the effective-dimension
closure survives its strongest challenger yet, now tested on BOTH axes
(per-coord rate and token count). Together with the oracle result, pgq6
closes two doors with data: sub-wall rates (again, on a new axis) and
observed-attention allocation.

## Artifacts

`artifacts/page_quant2/`: frozen_choices_pgq6.json (F-6), phaseA_pgq6.json,
pgq6_heldout_report.json, pgq6_output_check.json, pgq6_oracle.json,
pgq6_merge_telemetry.json; `artifacts/bench_pgq/llama31_8b/
pgq__pgq_mrglmrw_rdo__b2__ff390304__*` (5 cells). Code:
kvq/compression/pgq6_merge.py, loader routing in page_quant.py,
pipelines/page_quant/pgq6_{output_check,oracle}.py, tests/test_pgq6.py (6),
launcher pgq_mrg routing. Bugs: pgq6-1/pgq6-2 in fixes_to_apply.md.

## Open directions (deferred, recorded)

Cross-page merging + general clusters; merged-tie preference (prefer fewer
rows at equal D for the FLOPs win — currently conservative); kernel port of
the two-size-class format; Qwen transfer of mrg (machinery ready via
--model-tag); pgq5 C0/C1 thinking-mode study on the mrg default.
