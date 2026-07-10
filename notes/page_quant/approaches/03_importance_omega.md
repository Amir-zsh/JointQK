# Approach 3 — ω importance weighting (calibrated mean-logit signal)

**Verdict: works, as a regime law. The correction RDO needs, not a bonus.**

## Idea
Predict per-token attention relevance from the key alone: m_t = μ̄_qᵀk_t/√d;
ω_t = exp(τ·clip(m_t − median, ±c·ln2)) multiplies distortion in the page RDO.
Costs zero bits (decoder sees only rung ids). Frozen τ=0.5, clamp 4 bits.

## Key discovery — the predictor inversion
Q-weighted energy kᵀΣ_Qk ANTI-predicts attention (ρ ≈ −0.78): high-energy keys are
logit repellers parked deep in softmax's dead zone. Textbook Expected Attention ≈ 0.
The mean-logit signal predicts at +0.73…+0.82 and transfers OOD. Corollary: unweighted
RDO actively taxes attended tokens (they look expensive) — ω is the counter-steer.

## Results (regime law; three quantizer families)
| context | ω effect | verdict |
|---|---|---|
| EC rungs @1.0 (coarse regime) | **+2.9** mean, musique +5.8 | SIG |
| scalar dirs @1.0 | **+3.7** mean, hotpotqa +8.6 | SIG |
| fine VQ ladder @1.0 | −1.0 | tie (ladder self-protects) |
| EC @1.5 (rate abundant) | −1.6 (lcc −2.4) | lcc SIG harm (OOD weakness) |
Gating fix for lcc was tried and cleanly failed (pgq2). Rule: apply ω when the ladder
is coarse AND rate binds; skip otherwise.

## Artifacts
Frozen choices: `artifacts/page_quant2/frozen_choices.json`; decomposition cells in
`artifacts/bench_pgq/`; signal analysis in `pipelines/page_quant/phase1_empirics.py`.

## Open directions
- Better OOD behavior (lcc): domain-adaptive τ or per-head signal reliability
  weighting — untried.
- The same signal drives approach 5 (selection); combining rate-ω and selection-ω in
  one budget has not been studied.
