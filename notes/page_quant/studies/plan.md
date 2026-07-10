# Page-unit KV quantization — plan assessment & revised design

**Date started:** 2026-07-07 (overnight autonomous study)
**Branch:** `paged_quant` (off `entropy_coding` @ 0b1c582)
**Status:** v2 — revised after recon (5-agent inventory of data, EC machinery, bench
protocol, research lessons, kernel surface). v1 assessment retained below; the concrete
design is in "Design v2".
**Model:** meta-llama/Llama-3.1-8B-Instruct (Qwen replication impossible tonight — recon
found no Qwen raw captures or basis bundle on disk).

## The proposal (as given)

Quantize the K cache in **page units** (paged-attention blocks / FlashAttention tiles):

1. Pages are the natural serving granularity — compatible with PagedAttention and
   FlashAttention.
2. Within a page, tokens differ in importance; prior methods give every token the same
   bits, which is wasteful.
3. Use the calibrated Q distribution to drive transform coding: find good bases for
   pages, then an efficient quantization method.

Final deliverable: efficient implementation + FlashAttention integration.

## Assessment

### What is right (and why it has real headroom)

- **Page granularity is the correct serving unit.** vLLM blocks are 16–32 tokens;
  FlashAttention processes K in tiles of 64–128. A codec whose unit matches the
  attention tile can fuse dequantization into the tile load and never materialize an
  fp16 K cache. Our existing rANS codec already pages at ptok=64 — the infrastructure
  concept is proven in-repo.
- **Within-page token-importance skew is real and large.** Attention mass is extremely
  skewed (sinks, delimiters, a few content tokens carry most of the probability; a large
  fraction of tokens are never attended by any future query). The eviction literature
  (H2O, SnapKV, Quest) exploits exactly this signal to drop tokens entirely; using it to
  *modulate precision* instead of evicting is a natural, less lossy intermediate that
  none of our current baselines (JQ, TQ, KIVI, EC) exploit: they all spend the same bits
  per token at fixed rate.
- **Q-calibrated importance closes the loop with the project's machinery.** The expected
  attention-logit second moment of a cached key k under the calibrated query
  distribution is exactly `E_q[(qᵀk)²] = kᵀ Σ_Q k` — computable at compress time from
  the key alone, with the Σ_Q we already calibrate per (layer, kv_head). Better: in the
  rotated basis we already use, `kᵀ Σ_Q k ≈ Σ_j ℓ_j y_j²` (ℓ = the existing allocation
  weights), i.e. token importance is a *free byproduct* of the transform we already
  apply. No new calibration is needed.

### Revisions needed (the honest critique)

1. **"Same bits per token" is only half-wasteful under EC.** Our entropy-coded methods
   already spend *fewer bits* on small-magnitude tokens automatically (that is what the
   frequency model does) — the ~1.95 b/c EC results are, in rate terms, already
   token-adaptive. What EC does **not** do is allocate *distortion* non-uniformly: every
   token is coded with the same step size Δ, so every token gets ~the same reconstruction
   error regardless of whether it can ever win the attention argmax. The genuinely new
   lever is **token-adaptive distortion**: scale the quantization step per token by a
   Q-calibrated importance (fine steps for high-`kᵀΣ_Qk` tokens, coarse for
   never-attended ones). This is JPEG's perceptual quantization matrix, transposed to
   attention. The plan should be framed this way — it sharpens both the theory and the
   implementation.
2. **Importance must be decoder-recoverable.** The per-token step modifier must be known
   at dequant time. Two clean options: (a) transmit a per-token scale exponent as a tiny
   sideband (e.g. 2–4 bits/token ≈ 0.016–0.03 b/c at d=128 — cheap, and we account for
   it honestly); (b) derive it from page-level metadata (tier masks). Deriving it from
   the *reconstructed* key is circular; don't.
3. **Cross-token transforms inside a page need an empirical gate before we commit.**
   Keys in the cache are post-RoPE: adjacent positions are rotated by different angles,
   which actively decorrelates high-frequency dimensions across neighboring tokens.
   The cheap wins — per-page mean subtraction and a rank-r per-page sideband — should be
   measured first (Phase 1); a full 2D transform (e.g. DCT along the token axis) only if
   the residual correlation justifies its sideband + complexity. Any per-page sideband
   (means, low-rank factors, scales, tier masks) is charged to the rate. Prior: the
   per-channel structure KIVI exploits is *global*, and our Σ_K rotation already removes
   it; the open question is how much *local* (per-page) structure remains on top.
4. **Fixed-size pages are the serving story; variable-rate EC is the ceiling.** rANS
   pages have data-dependent size — awkward for block allocators and for a fused kernel
   (pointer chasing, no coalescing guarantees). The deployable variant should hold
   **every page to the same byte budget** and reallocate bits *within* the page by token
   importance (e.g. top-m tokens per page at b_hi bits, rest at b_lo, m fixed; mask in
   page metadata). Fixed page bytes ⇒ trivial paged layout, coalesced fused dequant,
   PagedAttention-compatible. EC token-adaptive deadzone remains in the study as the
   rate–distortion upper bound and to measure how much the fixed-rate constraint costs.
5. **FlashAttention integration scope must be realistic for one night.** Forking FA's
   CUDA is out. The right deliverable is a **Triton fused kernel** for the decode path
   (paged compressed K + V, dequant inside the attention tile loop, GQA-aware,
   A100/sm80, head_dim=128), validated bit-wise against a reference dequant→SDPA path,
   plus a throughput/memory microbench at 8k–64k context. That *is* "FlashAttention
   integration" in the sense that matters: same tiling scheme, online softmax, and it
   demonstrates the end-to-end serving claim. Full vLLM integration is documented as the
   follow-on path, not built tonight.
6. **Respect the project's hard-won lessons** (all from the EC study and the JQ
   inversion investigation):
   - Proxies (top-1, k_mse, logit_err) *prune* candidates; only downstream F1 *decides*.
     Plain JQ won top-1 and lost lcc by 11 pp.
   - Honest rates: every sideband bit (scales, masks, page means, low-rank factors,
     rANS page headers) counts in b/c; frequency models frozen on calibration; rates
     reported held-out. TQ's comparison rate stays 2.125 (its fp16 norms count too).
   - Calibration-data fairness: any new basis/statistic uses the same 400-prompt
     uncentered n400 bundle moments as r_sym/qpca_unc, or the confound is called out.
   - Layer 0 = attention sink: fp16, excluded from metrics, never fit (its Σ_Q has a
     numerically negative eigenvalue that NaNs QPCA-style constructions).
   - Leakage: fit/selection rows from compact8 TRAIN only; eval uses the existing
     exclusion file; lcc/2wikimqa stay fully OOD.
   - Eval noise: at 150–500 rows, differences ≲0.5 pp mean are noise; musique is a
     tie-band ±0.3.
   - Σ_Q drifts across tasks and into decode (q_distribution_shift note) — the
     importance signal must be checked for robustness under that drift (Phase 1
     measures importance transfer across tasks).
7. **One more improvement worth taking (free synergy):** the per-page metadata this
   codec needs (importance tiers / maxima) is exactly what query-aware page-skipping
   methods (Quest) use to skip whole pages at decode. We get the *option* of sparse
   attention for free later; tonight we only note the layout leaves room for it.

### Suggested framing

> Attention-weighted transform coding at page granularity: a Q-calibrated,
> per-token-distortion-allocated, fixed-page-byte K-cache codec whose dequantization
> fuses into the attention tile loop.

Two axes are being unified: the project's existing *coordinate-wise* allocation (which
basis, which per-coordinate bits) and the new *token-wise* allocation (which tokens
inside a page deserve fidelity), under one Q-weighted distortion objective.

## Revised plan (phases + gates)

- **P0 Recon** (done): data inventory, code surfaces, kernel constraints.
- **P1 Empirics** (GPU-light, existing captures; selection rows + posthoc rows
  read-only): (a) within-page importance skew — distribution of log ẑ_t and `kᵀΣ_Qk`
  inside 64-token pages; (b) predictiveness: does calibrated ω_t rank the tokens that
  *realized* future queries (last-10%-of-row queries as decode proxy) actually attend?
  Spearman + top-decile capture, per task incl. OOD lcc/2wikimqa; picks ω functional
  form (ẑ vs kᵀΣ_Qk), τ, clamp c; (c) oracle RD headroom at matched rate:
  token-uniform step vs per-page rung vs per-token weighted RDO, at b ∈ {2.0,1.5,1.0}
  — Q-weighted distortion + realized top-1 proxy; (d) quick within-page structure
  check (page-mean energy) to close the cross-token-transform question.
  **Gate G-P1:** per-token weighted allocation shows material oracle gain at b ≤ 1.5
  (≥15% Q-weighted-MSE or clear top-1-proxy gain) AND ω_t predicts realized importance
  above chance on OOD tasks. If not, report honestly and fall back to studying the
  unweighted-RDO + fixed-page serving story only.
- **P2 Codec implementation** (`kvq/compression/page_quant.py`,
  `pipelines/page_quant/fit_pgq_bundle.py` + tests): pgq_ea / pgq_plain / pgq_fixed +
  multi-rung bundle format (version 2), press branch `k_method="pgq_*"`, bit-exactness
  vs an ω-extended PageCodecRDO reference.
- **P3 Fidelity screening** (selection rows, layer-0 excluded): all candidates × rates
  vs TQ/JQ/EC baselines. **Gate G-P3:** candidate ≥ TQ on all proxies at 2.0; at
  1.5/1.0 candidates ranked vs ec_uniform controls. Promote ≤6 cells-worth. Selection
  BEFORE any F1 (selection-honesty rule).
- **P4 F1 bench** (lcc/musique/2wikimqa, fraction 1.0, v7 protocol, GPUs 0–3): smoke
  at fraction 0.05 first. Success bars: (i) at 2.0 b/c, pgq_* in the 44.5–45 ≈FP band
  (fixed-byte pages costing ~nothing vs variable-rate EC); (ii) at 1.5/1.0, pgq_ea >
  ec_uniform by more than noise (>0.5 pp mean) — the premise's direct test; (iii)
  pgq_fixed within ~0.5 pp of pgq_ea at matched rate (serving variant costs little).
- **P5 Fused kernel prototype** (Triton, decode path, pgq_fixed layout): correctness
  vs reference dequant→SDPA, throughput/memory vs bf16 SDPA (sm80 flash backend) at
  8k/32k/64k, GQA 4:1, head_dim 128, one GPU. rANS decode kernel reuse optional.
- **P6 Report** (`notes/page_quant/report.md`) + morning summary; regression
  fingerprint check; no commits (await explicit approval).

## Design v2 (post-recon): importance-weighted per-token RDO in fixed-byte pages

### What recon changed

1. **The premise "prior methods give every token the same bits" is already half-solved
   in-repo, unbenched.** `entropy_coding/notebooks/kvq_codec_rdo.py` (`PageCodecRDO`,
   MAGIC 'KVC5') implements per-token rung selection by Lagrangian RDO inside a fixed
   `page_bits` budget: each token independently picks a quantization step Δ·m_r from a
   rung ladder, λ bisected per page so the page fits its byte budget; per-token rung ids
   ride in the page header; a matching GPU decode kernel (`rans_decode_rdo.cu`) + test
   fixtures exist. It was never importance-weighted (its D_t is plain transform-domain
   SE), never wired into the F1 bench, and never benched.
2. **The user's insight lands as a reweighting of that RDO objective.** Choose per-token
   rung r_t = argmin_r [ ω_t·D_t(r) + λ·R_t(r) ]. ω_t is computed by the ENCODER from
   the full-precision key; the decoder only reads rung ids — **the importance signal
   costs zero extra sideband** and cannot leak precision decisions it can't reproduce.
3. **With the `qpca_unc` basis, transform-domain SE *is* Q-weighted key MSE**
   (G Σ_Q Gᵀ = I — the 2026-06-16 audit result), so plain RDO already optimizes the
   right *MSE*; ω_t adds the softmax/argmax sensitivity on top. Primary basis:
   `qpca_unc` (proven ≈FP at 1.95 b/c, uniform per-coord weight ⇒ one Δ per head ⇒
   cleanest rung ladder semantics).
4. **ω_t = Expected Attention** (`notes/core/kv_cache_rate_distortion_proposal.md`
   §4.2/§6, never implemented): log ẑ_t = μ̄_qᵀk_t/√d + k_tᵀΣ̄_Q k_t/(2d), with Σ̄_Q
   from the n400 bundle (calibration-fairness rule) and μ̄_q pooled from the 18 EC fit
   rows' 02_stats `sum_q` (the bundle stores no mean; a mean needs far fewer samples
   than a covariance — noted as a caveat). ω_t = clamp(ẑ_t^τ, 2^-c, 2^c); τ and c
   selected in P1 from realized-attention data, frozen before any F1. Only *relative*
   ω within a page matters (λ is per-page).
5. **At K=2 there is no F1 headroom left** (EC 44.90 vs FP 44.99): the winnable claims
   move to (a) **the rate axis** — hold ≈FP at b_target ∈ {1.5, 1.0} where token-uniform
   EC should degrade; (b) **the serving axis** — fixed bytes per page (paged-attention
   block-allocator compatible, unlike variable-size rANS streams) and a no-rANS
   fixed-rate variant whose decode is shift/mask, fused into a Triton attention kernel.
6. **Correlated-error lesson respected** (`q_distribution_shift.md`: the suspected F1
   killer is error *correlated across tokens*, not error magnitude): per-token rungs
   *decorrelate* quantization error relative to per-page or per-head steps; no
   cross-token transform in the primary design (page-mean/low-rank sidebands demoted to
   a P1 measurement + optional variant — RoPE mixing makes local structure doubtful).
7. **Per-token scale factors are forbidden, bits are not** (Stage-1D lesson #36):
   importance enters through the rung/rate channel, exactly as that lesson demands.

### Candidate methods (P2)

| id | coder | token adaptivity | sideband (counted in b/c) | role |
|---|---|---|---|---|
| `pgq_ea` | deadzone + frozen-model rANS, fixed page_bits = b·d·ptok | per-token rung, ω_t = Expected Attention | rung ids: R≤8 ⇒ 3 bits/tok = 0.023 b/c + page header | headline |
| `pgq_plain` | same | per-token rung, ω_t = 1 | same | isolates ω's contribution from RDO's |
| `pgq_fixed` | int-packed per-token bit-width b_t ∈ {0?,1,2,3,4}, no rANS | same weighted RDO, R_t = b_t·d | 3 bits/tok + per-head scales | serving headline, fused-kernel target |
| `ec_uniform@b` | existing fit_ec_bundle --uniform-step at b ∈ {1.5, 1.0} | none (token-uniform) | — | the control the premise is tested against |

All at page ptok=64 (matches existing G1-verified codec pages + FA tile), fixed
page_bits ⇒ held-out rate is guaranteed by construction; what's measured held-out is
overflow behaviour + F1. Rate grid: b_target ∈ {2.0, 1.5, 1.0}. m-grid: geometric,
m = 2^{j/2}, j = −2..5 (R=8; finer-than-base rungs let important tokens *gain*
precision, coarser rungs pay for them; exact grid revisited after P1's importance
dynamic-range measurement).

### Baselines & protocol
v7-identical (Mode A, layer-0 fp16, V = v_turboquant@2b, fraction 1.0, compact8
exclusions). On-disk baselines reused (never re-run): FP 44.99, TQ 41.52 @2.125,
ec_qpca_unc_dz0.5 44.90 @1.947. New controls fit tonight: ec_uniform @ 1.5 / 1.0
(token-uniform EC at the lower rates — the direct test of "per-token adaptivity is
where the win comes from"). dz: 0.5 everywhere (best data-matched config; dz is not
tonight's knob).

### Fit/eval data (leakage-clean, per recon)
- Fit: 18 compact8-TRAIN rows (`roles.json` fit set) — deltas, frozen models per rung,
  μ̄_q. Selection: the 8 compact8-TRAIN selection rows — P1 empirics, ω/τ/m-grid
  choice, Phase-A screening. Never eval rows.
- Predictiveness transfer check may additionally *read* the 12 eval-side posthoc rows
  (measurement-only, no parameter feeds back — same status as the EC study's post-hoc
  rate measurement).
- Big pool available if needed: compact8 400-train raw rows (356 GB) for a beefier μ̄_q
  or importance-statistics check.

## Design v3 revisions (post adversarial review + Phase-1A empirics, 2026-07-07 ~02:30)

Three-critic adversarial review (theory / protocol / systems) + corrected Part-A
empirics forced these changes. Full critique JSON in the session workflow logs.

1. **ω functional form is settled empirically: the mean-logit signal `m_t = μ̄_qᵀk_t/√d`.**
   Part A (selection rows): ρ(m, realized mass) = +0.73…+0.82, top-decile capture
   0.79–0.87. Both energy forms anti-predict (ρ(kᵀΣ_Qk, mass) ≈ −0.78: high-Q-energy
   keys are logit *repellers*), and centered EA (`m + kᵀΣ_c k/2d`) is washed out by its
   variance term (ρ ≈ 0). All three critics independently flagged the uncentered-Σ_Q
   double-count bug in the textbook EA form; empirics confirm even the corrected form
   loses to plain `m`. **ω_t = exp(τ·clamp(m_t − median_row(m), ±c·ln2))**, log-domain,
   continuous inside the clamp (rail saturation would staircase the per-page bisection).
   τ, c chosen on Part-C RD outcomes (rank metrics are provably invariant to them) from
   SELECTION rows only; posthoc-row predictiveness is report-only, computed after the
   freeze (leakage rule tightened per protocol critic).
2. **ω is never judged on Q-weighted MSE.** Under qpca_unc, unweighted per-token RDO is
   the Q-MSE optimum by construction; ω can only move away from it. The ω contrast
   (pgq_ea vs pgq_plain) is scored exclusively on realized-attention-aligned metrics
   (mass-weighted logit error, top-1) and, decisively, F1.
3. **Mandatory decomposition chain in P4, exempt from proxy screening:**
   `ec_uniform` (variable-rate, token-uniform) → `ec_pagerung` (fixed pages, per-page
   rung — isolates the fixed-page-byte constraint) → `pgq_plain` (+per-token RDO) →
   `pgq_ea` (+ω). Success bar (ii') pgq_ea > pgq_plain is the ω attribution; bar (iii)
   reworded to "no detectable difference at the noise floor"; row-paired bootstrap CIs
   from predictions.csv replace the flat 0.5-pp rule.
4. **Battleground discovery front-loaded:** ec_uniform fits at b ∈ {1.5, 1.0, 0.75}
   launched first (existing fit_ec_bundle, qpca_unc/dz0.5/uniform-step) and F1-benched
   before the pgq grid is fixed; primary rate = the highest rate where ec_uniform is
   distinguishably below FP (pre-registered before pgq F1 cells run).
5. **Codec format fixes:** rung ids packed at ceil(log2 R) = 3 bits (u16 would be a 5×
   sideband understatement — 0.125 b/c — enough to rig the b=1.0 comparison); 12-byte
   page header + lane overhead counted in b/c; snap-aware D (allocation sees the
   frozen-alphabet clipping the decoder actually produces — critical for OOD lcc at low
   b); guaranteed-fit terminal rung (no page ever exceeds page_bits); overflow gate
   (<1% pages, else the fixed-rate claim fails at that rate); allocation in fp64 with
   deterministic tie-break (lowest rung wins) in both press sim and reference; per-cell
   telemetry (rung histograms, overflow counts, ω dynamic range) as first-class output.
6. **One rung ladder shared across rates** (base Δ solved at b=1.5; page_bits alone
   sets the rate) — kills a fit-vs-fit confound and two fits' wall-clock. Coarse rungs
   whose fit entropy < 0.25 b/c are dropped for pgq_ea (silent k_mean-eviction =
   correlated error, the suspected F1 killer); pgq_fixed instead gets an explicit
   b_t = 0 width (reconstruct μ_k) as a first-class, RDO-chosen action (also the
   Quest-style page-skip hook).
7. **pgq_fixed quantizer redesigned** (systems critic): saturating uniform grids per
   width (b=1: sign·scale; b≥2: symmetric 2^b-level mid-tread), step per (l,h,width)
   fit to Q-SE incl. overload on fit rows; NOT the EC deadzone ladder int-packed
   (undefined overload, degenerate 1-bit deadzone). Exact R_t = b_t·d ⇒ greedy
   hull allocation, guaranteed page fit, no overflow loop.
8. **Rotated-domain attention scoring** is a named design element for P5: score in code
   space via q′ = q @ invᵀ once per (layer, q-head) per decode step; logits =
   Δ·(idx · q′)·m_rt + qᵀμ_k (the qᵀμ term added for compressed tokens in mixed
   caches). Never inverse-transform keys in the kernel — the per-tile 128×128 GEMM
   would cost more than the entire fp16 attention read at 64k ctx.
9. **Microbench honesty:** three traffic numbers (both-fp16 / K-comp+V-fp16 ≈ 1.8×
   ceiling / K-only ≈ 8–16×), `enable_gqa` flash SDPA baseline (never .expand()ed KV),
   optional in-tile int2 V dequant to show the ~7.8× combined case; rANS decode stays
   out of the serving hot path (symbol-serial ceiling ~10× off), measured via
   bench_decode.py if time permits.
10. **Toolchain guards:** pgq aggregator filters out fraction-tagged smoke subdirs and
    hard-fails on missing full-fraction cells; bundle sha8 embedded in every cell label
    so refits mint new cell dirs (worker's idempotent skip is mtime-blind).

## Risks

- Importance mis-prediction under Σ_Q drift → P1 measures transfer; tier scheme is
  conservative (b_lo still ≥ 1–2 bits, nothing is dropped to zero).
- Sideband overhead eats the win at d=128 → all sidebands costed in b/c from the start;
  designs with >0.1 b/c sideband need to earn it in P3.
- Fixed-rate tiering underperforms EC's continuous adaptivity → quantified by the
  pq_ec_tok vs pq_tier gap in P3/P4; both are results.
- Kernel prototype eats the night → time-boxed; the F1 result stands alone if so.

## Log / heartbeat

`logs/page_quant_overnight.log` (+ `.heartbeat`); per-run logs under `logs/page_quant_*`.
