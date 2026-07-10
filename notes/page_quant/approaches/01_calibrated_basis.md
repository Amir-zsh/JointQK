# Approach 1 — Calibrated query-aware basis (Σ_Q / qpca_unc)

**Verdict: foundation of everything; independently validated.**

## Idea
Key quantization error matters only through attention logits: 𝔼_q|q·(k−k̂)|² =
(k−k̂)ᵀΣ_Q(k−k̂). So estimate Σ_Q = 𝔼[qqᵀ] per (layer, kv-head) from a small
calibration corpus and code keys in a basis where plain squared error equals that
quadratic form: forward F, decode G = F⁻¹ with G Σ_Q Gᵀ = I (qpca_unc).

## What was built
Calibration capture + pooling (`pipelines/calibration/`, `kvq/capture/`); basis
builder `build_basis("qpca_unc", …)` in `pipelines/ec/fit_ec_bundle.py`; consumed by
every codec family via the `(Dmat, rate_bits)` contract.

## Results
- All EC/no-EC results below are on this basis; earlier basis sweeps settled it
  (data-matched qpca_unc ties r_sym; see ec_k2v2 line of work).
- **External validation:** OSCAR (arXiv 2605.17757) proves U_Q = eig(Σ_Q) optimal
  under a frozen-error surrogate; their ablation: attention-aware target 70.0 vs raw
  KᵀK target 31.1 mean accuracy. Their eigenbases of Σ_Q and Σ_K are empirically
  near-orthogonal — Q-side and K-side statistics are genuinely different objects.
- **Negative control (pgq3 family d):** a learned (F, G) pair trained end-to-end
  (STE, 248/248 heads, no overfit) made the codec WORSE (logit_err 0.0174→0.0298,
  sinkCE 0.015→0.112). The basis is not the binding constraint.
- Calibration robustness: ~8k tokens suffice; corpus choice second-order.

## Artifacts
`artifacts/bases/`, `artifacts/calibration/…`, `artifacts/page_quant2/linear_pair_*`
(the learned-pair control), OSCAR reference `vendor/OSCAR/rotation/compute_kv_rotation.py`.

## Open directions
- **V-side analogue (registered as Thrust C):** score-weighted value covariance
  C_S = VᵀSᵀSV (estimable as diag(K·Σ_Q·Kᵀ)-weighted VᵀV on existing captures) —
  could lift the entire stack; separate phase because it breaks v7 comparability.
- Qwen3-8B replication of the basis pipeline (needs re-capture).
