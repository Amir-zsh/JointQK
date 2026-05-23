# Research Notes

This directory organises the project notes by role.

## Core

- [kv_cache_rate_distortion_proposal.md](core/kv_cache_rate_distortion_proposal.md)
  - high-level research framing and thesis
- [research_plan.md](core/research_plan.md)
  - execution roadmap
- [neurips_submission_plan.md](core/neurips_submission_plan.md)
  - target submission scope and scaffold
- [neurips_implementation_plan.md](core/neurips_implementation_plan.md)
  - complement to the submission plan — code paths and checkpoints

## Bench results

- [bench_results_report.md](bench_results_report.md)
  - canonical downstream F1 results (Qwen3-8B)
- [bench_llama31_8b_results_report.md](bench_llama31_8b_results_report.md)
  - downstream F1 on Llama-3.1-8B
- [bench_cross_model_comparison.md](bench_cross_model_comparison.md)
  - cross-model Qwen3 vs Llama comparison
- [bench_llama_runbook.md](bench_llama_runbook.md)
  - runbook for the Llama bench sweep

## JointQK F1-inversion investigation (Llama)

- [jointqk_disconnect_investigation.md](jointqk_disconnect_investigation.md)
  - mechanism investigation: K-fidelity wins but F1 inverts
- [jointqk_investigation_report.html](jointqk_investigation_report.html)
  - self-contained HTML write-up with embedded charts
- [q_distribution_shift.md](q_distribution_shift.md)
  - Σ_Q top-16 subspace drift analysis

## Synthesis

- [experiments_and_findings.md](experiments_and_findings.md)
  - current overall takeaway across calibration, bench, and analysis
- [pooled_n50_report.md](pooled_n50_report.md)
  - preview of the pooled-N50 calibration corpus
- [qpca_basis_report.md](qpca_basis_report.md)
  - QPCA basis test: closed-form optimum for logit MSE. Wins logit_err by 16–36 % proxy and loses top-1 by 3–5 pp; downstream LongBench F1 confirms (−5.36 pp mean F1 at k=2). 3-experiment investigation (pooled K-fidelity → per-task K-fidelity → F1 sweep), production basis unchanged.

## Reference

- [reference/](reference/)
  - standalone derivations
