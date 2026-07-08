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

## Page-unit quantization (paged_quant branch)

- [page_quant/final_report.md](page_quant/final_report.md)
  - presentation-ready consolidated write-up (clear prose; start here)
- [page_quant/final_report.html](page_quant/final_report.html)
  - interactive HTML version (self-contained; rate-vs-F1 chart, mechanism stack)
- [page_quant/plan.md](page_quant/plan.md)
  - plan assessment → recon → 3-critic adversarial review → design v3
- [page_quant/report.md](page_quant/report.md)
  - per-token-precision fixed-byte pages: ω decomposition on F1, EC serving verdict
    (measured), fixed-width bug chain, kernel prototype
- [page_quant/report2.md](page_quant/report2.md)
  - pgq2: no-EC codec frontier mapped (EC fundamental at <=1.5 b/c; omega regime law)
- [page_quant/fixes_to_apply.md](page_quant/fixes_to_apply.md)
  - bug tracker (pgq_fixed three-stage root cause chain)

## Reference

- [reference/](reference/)
  - standalone derivations
