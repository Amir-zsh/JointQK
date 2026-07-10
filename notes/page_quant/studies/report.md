# Page-unit K-cache quantization: per-token precision inside fixed-byte pages

**Date:** 2026-07-07 (overnight autonomous study) · **Branch:** `paged_quant`
**Model:** meta-llama/Llama-3.1-8B-Instruct · **GPUs 0–3** · **Status: COMPLETE**
**Plan & design history:** `notes/page_quant/plan.md` (assessment → recon → 3-critic
adversarial review → v3).

## TL;DR

1. **The page-unit idea works, precisely where rate binds.** Importance-weighted
   per-token rung selection (ω from the calibrated mean-logit signal, zero extra
   sideband) inside fixed-byte 64-token pages lifts b=1.0 F1 by +2.9 over unweighted
   RDO (musique +5.8, 95% CI [+1.3, +10.6] SIG; 3-task mean CI [−0.0, +4.8]) and
   beats even the variable-rate EC stream on 2wikimqa. At abundant rate (1.5) ω is
   unnecessary and hurts OOD code (lcc −2.4 SIG); at starvation (0.75) nothing saves
   fixed pages. **ω belongs on a rate-dependent switch.**
2. **Fixed-byte pages — the paged-attention-compatible format — are statistically
   free at 1.5 b/c**: pgq_plain 43.98 vs variable-rate ecu 43.02 (every task a tie),
   within ~1.0 of full precision (44.99) at ~10.7× K compression.
3. **New rate curve for token-uniform EC** (qpca_unc, dz 0.5): 43.02 / 41.25 / 40.01
   at 1.5 / 1.0 / 0.75 b/c — at 1.0 it already matches TurboQuant (41.52) at **half**
   TQ's true rate.
4. **Predictor discovery that rewrites the importance playbook:** Q-weighted key
   energy `kᵀΣ_Qk` ANTI-predicts realized attention (ρ ≈ −0.78; high-energy keys are
   repellers); the calibrated mean-logit `μ̄_qᵀk/√d` predicts it strongly (ρ ≈ +0.8,
   top-decile capture ~0.85). Textbook Expected Attention (variance term included) is
   ~uncorrelated. Also: unweighted RDO actively *strips* attended tokens (weighted
   logit error worse than token-uniform) — ω is the repair, not a bonus.
5. **Entropy coding is measured out of the decode hot path**: the real CUDA rANS
   decoder does 1.2–1.5 Gsym/s (kernel ~10% of wall time) → ~1.8 s per generated
   token for whole-context re-decode at 64k vs ~5 ms fp16 read. EC stays for
   offload/prefix/storage tiers and as the RD frontier.
6. **No-EC scalar fixed-width serving codec: clean, well-documented negative.** Four
   iterations (zero-level fix → sink-range fix → forced-sink-w8 → Lloyd-Max bulk
   codebooks) took it from degenerate (F1 ~0) to well-formed but 12–21 pp below the
   EC arms at 1.0–1.5 b/c. Scalar per-coord widths cannot buy back what entropy
   coding saves at these rates; a deployable hot path needs TQ-style vector codes
   (the ω width-selection mechanism ports unchanged). En route: two failure modes
   that ranking proxies provably cannot see (norm inflation; sink clipping), each
   diagnosed with attention-mass probes.
7. Fused-kernel prototype: correct, bandwidth-advantaged (1.78× e2e / 8× K-only),
   currently instruction-bound at 0.2–0.4× SDPA; width-planar layout designed as the
   fix. Regression fingerprint: 262/262 canonical values intact.

## The idea and what it became

User's proposal: quantize K in **page units** (paged-attention/FlashAttention
compatible), exploiting that tokens inside a page differ in importance — prior methods
spend the same bits per token; use the calibrated Q distribution + transform coding.

Post-analysis refinement: under entropy coding the *rate* is already token-adaptive;
the genuinely new lever is **token-adaptive distortion** — a per-token quantization
step chosen by importance. The deployed form: **per-token rung selection by Lagrangian
RDO inside a fixed page-byte budget**, with the distortion term weighted by an
encoder-side importance ω_t. The decoder needs only the packed per-token rung ids
(3 bits/token), so the importance signal costs **zero extra sideband** and no
decoder-side circularity. Builds directly on `entropy_coding/notebooks/kvq_codec_rdo.py`
(Samuel's per-token RDO prototype — previously unweighted, unwired, unbenched).

## Phase 1 empirics (selection rows; posthoc rows report-only)

1. **What predicts realized attention** (Spearman ρ vs softmax mass of the row's own
   last-10% queries over first-90% keys, layers 1–31):
   - mean-logit `m = μ̄_qᵀk/√d`: **ρ = +0.73…+0.82, top-decile capture 0.79–0.87**
   - Q-energy `kᵀΣ_Qk`: **ρ ≈ −0.78** — high-energy keys are logit *repellers*
   - textbook Expected Attention (even with correctly centered Σ): ρ ≈ 0 — its
     variance term (which anti-predicts) washes out its mean term (which predicts).
   OOD transfer (report-only, after freeze): lcc ρ_m ≈ 0.67–0.72, musique ≈ 0.68.
2. **Attention skew:** first 4 tokens carry ~74–87% of realized mass (sink); within a
   typical 64-token page the top-8 tokens carry ~43% of the page's mass.
3. **Oracle allocation headroom at matched rate** (qpca_unc codes; code-SE = Q-weighted
   key MSE exactly):
   - Fixed-page constraint costs ~1% distortion vs global allocation (serving is
     nearly free).
   - Flat per-token RDO ≈ token-uniform on Q-MSE (EC's uniform step is already
     near-optimal for total MSE) — **and is *worse* than token-uniform on
     attention-mass-weighted logit error** (0.206 vs 0.150 at b=1.0): unweighted RDO
     takes bits from high-energy tokens, which include the attended ones.
   - ω-weighted RDO (ω = exp(τ·clamp(m−median, ±c·ln2)), τ=0.5, c=4 — frozen from
     selection rows only, logged before posthoc reads) recovers it: weighted logit
     error 0.110 (−47% vs flat RDO, −27% vs uniform) at +9% unweighted MSE;
     top-1 +0.3 pp of the realized-ω oracle's +0.6 pp ceiling.

## Methods benched (decomposition chain)

All: qpca_unc basis (n400 uncentered moments), dz=0.5, V = v_turboquant@2b,
Mode A, layer-0 fp16, ptok=64, one shared rung ladder (base Δ solved at 1.5 b/c,
m-grid 2^{j/2} j=−2..4 after the 0.25-b/c coarse-rung floor; 7 rungs, 3-bit ids).

| arm | pages | token allocation | ω | coder |
|---|---|---|---|---|
| `ecu` | none (variable-rate stream) | uniform step | — | frozen-model EC |
| `pgq_pagerung` | fixed bytes | one rung per page | — | frozen-model EC |
| `pgq_plain` | fixed bytes | per-token RDO | 1 | frozen-model EC |
| `pgq_ea` | fixed bytes | per-token RDO | exp(τ·m̃) | frozen-model EC |
| `pgq_fixed` | fixed bytes | per-token width {0,1,2,4} | 1 | none (int pack) |
| `pgq_fixed_ea` | fixed bytes | same | exp(τ·m̃) | none |

Consecutive differences isolate: fixed pages (ecu→pagerung), per-token RDO
(pagerung→plain), Q-calibrated importance (plain→ea), no-entropy-coding serving cost
(ea→fixed_ea / plain→fixed).

## Gates

- **G2 rate (held-out selection rows, sideband charged):** ecu fits landed on target
  (1.497 / 0.998 / 0.749 b/c). pgq_ea/plain: 1.496 / 0.9965 / 0.747 with overflow
  0.11–0.15% (< 1% gate, PASS). pgq_fixed/fixed_ea: exact rate, **0.00% overflow**.
- **Rate-fill finding:** the page-uniform control can only fill fixed pages to
  0.82 b/c at a 1.0 budget (1.21 at 1.5) — rung granularity strands budget; per-token
  RDO fills to 0.997 / 1.496. Per-token allocation is what makes fixed-byte pages
  rate-efficient at all.
- **Smoke (worker→press→codec, fraction 0.05): PASS.**
- **Unit tests:** 8/8 (`tests/test_page_quant.py`) — snap parity vs numpy reference,
  budget respect, ω=0 determinism, ω sensitivity, pagerung invariant, fixed-width
  exact fit + w0→μ_k, CPU/GPU allocation parity.
- Phase-A fidelity screening (G-P3): *(pending — scorer running)*
- Codec-honesty caveat: the F1 press path reproduces the coded symbols/rates
  analytically (frozen-model nlp accounting incl. 3-bit rung ids, header, 4-lane rANS
  overhead); a full byte-level rANS stream gate (EC-study G1 style) was not re-run for
  the new page format tonight — flagged as follow-up. The EC family reuses the
  bit-exact-verified snap/dequant primitives from the EC study.

## Downstream F1 (fraction 1.0, v7 protocol, compact8 exclusions)

Baselines on disk (never re-run): FP 44.99; TQ k2v2 41.52 @2.125; jointqk_k2_v2 37.85;
ec_qpca_unc_dz0.5 44.90 @1.947 (the EC study's best).

**ec_uniform battleground (new tonight):**

| rate | lcc | musique | 2wikimqa | mean | Δ vs FP |
|---|---|---|---|---|---|
| ecu@1.5 | 47.59 | 32.17 | 49.31 | 43.02 | −1.97 |
| ecu@1.0 | 45.60 | 32.10 | 46.05 | 41.25 | −3.74 |
| ecu@0.75 | 48.76 | 30.14 | 41.12 | 40.01 | −4.98 |

Token-uniform EC at 1.0 b/c already ≈ TurboQuant's 41.52 **at half TQ's true rate**.
Degradation is real by 1.0 (−3.7) → primary rate b=1.0 (1.5 secondary), as
pre-registered. 2wikimqa degrades monotonically (49.3→46.1→41.1) and is the clean
rate-sensitivity signal; lcc is non-monotonic across rates (edit-sim noise band).

**Phase-A screening (G-P3), selection rows, 8192-query cap:**

| method | top1 | top5 | k_mse | logit_err | rate |
|---|---|---|---|---|---|
| tq_k2 (ref) | 0.4031 | 0.9212 | 0.557 | 0.0233 | 2.125 |
| ecu@1.0 | 0.4603 | 0.9632 | 0.740 | 0.0065 | 0.998 |
| pgq_plain@1.0 | 0.4535 | 0.9524 | 0.780 | 0.0070 | 0.996 |
| pgq_ea@1.0 | **0.4704** | 0.9616 | 0.823 | 0.0077 | 0.996 |
| pgq_fixed@1.0 | 0.4758 | 0.8802 | 1.036 | 0.0299 | 0.996 |

G-P3: EC-family beats TQ on top1/top5/logit_err at **half TQ's rate** (raw k_mse is
unweighted and expectedly above TQ at 1.0 b/c) — PASS. The ω ordering replicates on
held-out proxies: ea > ecu > plain on top1 (flat RDO hurts argmax; ω more than repairs
it). pgq_fixed is marginal (top5/logit_err below TQ; odd top1 inversion across rates —
flagged); it is benched anyway as the pre-registered serving arm.

**Full decomposition (3-task mean F1; per-task in `bench_summary.json`):**

| arm | b=1.5 | b=1.0 | b=0.75 |
|---|---|---|---|
| full_precision | 44.99 | 44.99 | 44.99 |
| ecu (variable-rate stream, token-uniform) | 43.02 | 41.25 | 40.01 |
| pgq_pagerung (fixed pages, page-uniform) | — | 38.24 | — |
| pgq_plain (fixed pages, per-token RDO) | **43.98** | 37.25 | — |
| pgq_ea (fixed pages, RDO + ω) | 41.97 | **40.12** | 35.14 |
| pgq_fixed (no-EC serving, Lloyd-Max v4) | 19.7 / 28.1 (musique/2wiki) | — | — |
| pgq_fixed_ea (no-EC serving, Lloyd-Max v4) | — | 17.4 / 25.4 (musique/2wiki) | — |

**No-EC serving verdict at b=1.0: a clean negative.** Four design iterations
(MSE-fit mid-rise → q999 mid-tread → sink-forced → Lloyd-Max bulk codebooks; full
chain in `fixes_to_apply.md`) took pgq_fixed from degenerate (F1 ~0–2, whitespace
generations) to well-formed but weak (musique 17.4, 2wiki 25.4) — still 15–20 pp
below the EC arms at the same 0.996 b/c. At 1.0 b/c the width mix is
eviction-dominated (~50% of tokens at w=0) and scalar fixed-width coding cannot
buy back what entropy coding saves; the sink-mass overshoot from competitor-logit
shrinkage persists structurally. Deployment implication: the no-EC hot path should
run at ≥1.5 b/c (still ~10.7× vs fp16), or adopt TQ-style vector representations —
not scalar widths — below that.

Row-paired bootstrap (10k resamples, per-row LongBench scores, 95% CI):

- **ω attribution (ea − plain @1.0): musique +5.80 [+1.29, +10.58] SIG**; 2wiki +2.87
  (tie); lcc −1.44 (tie); 3-task mean **+2.41 [−0.00, +4.81]** — the boundary case;
  the musique rescue is the solid claim.
- **Fixed pages are free at 1.5**: plain − ecu = +0.60 mean, every task a tie — a
  fixed-byte-page codec matches the unconstrained variable-rate stream.
- **ω harms OOD code when bits are abundant**: ea − plain @1.5 lcc −2.44
  [−4.31, −0.55] SIG.
- **The family's persistent deficit is lcc**: ea − ecu @1.0 lcc −4.88 SIG (mean −1.35
  tie); at 0.75 the fixed-page arm genuinely fails (mean −4.94 SIG).

**Reading: importance-weighted per-token allocation has a Goldilocks zone.** At 1.5
b/c bits are abundant — plain RDO in fixed pages is the best config (43.98 ≈ FP−1.0,
beating even variable-rate EC pointwise) and ω's skew wastes bits exactly where its
predictor is weakest (OOD code). At 1.0 b/c precision is the binding constraint and
ω is what keeps attended tokens alive (+2.9 mean over plain, musique +5.8 SIG,
2wiki above even ecu). At 0.75 fixed pages starve regardless — variable-rate's
cross-page freedom wins. The lcc axis is the standing failure mode: the m-predictor's
OOD transfer is weakest on code (ρ 0.67–0.72 vs 0.73–0.82 in-domain), and
importance-*mis*direction costs more than uniformity. Obvious follow-up: ω-confidence
gating (fall back to ω=1 when the within-page m-spread is small or the head's
predictor quality is low), or a page-level code/text discriminator.

## Entropy coding in the serving path: measured verdict

`pipelines/page_quant/probe_rans_throughput.py` ran the real `PageCodecRANSCUDA`
decoder (rans_decode.cu, 16 lanes, ptok=64, bit-exact output) on 67M synthetic
symbols: **1.2–1.5 Gsym/s kernel-only, with the entropy kernel just ~10% of wall
time** (host-side page parsing dominates). One full-model whole-context decode at
64k ctx = 2.08G symbols → **~1.8 s per generated token kernel-only** vs the ~5 ms
fp16 attention read. Parallelism is not the blocker (4M independent lane-chains
exist); the per-symbol cost and the whole-context-re-decode-per-token architecture
are. Even a DietGPU-class decoder (~100×) would only break even with reading an
*uncompressed* fp16 cache — while fixed-width dequant fuses into the attention
kernel for near-zero extra traffic. **Conclusion: entropy coding never enters the
per-token path. It remains correct for offload/prefix/storage tiers (one-shot
decode against PCIe/NVMe bandwidth) and as the rate–distortion frontier; the
importance mechanism (ω → rung/width selection) is coder-independent.**

## Serving prototype (P5)

Triton fused paged-dequant decode attention (`pipelines/page_quant/bench_fused_kernel.py`):
fixed-byte pages (2-bit width ids + power-of-2-width token rows), **rotated-domain
scoring** (q′ = q@invᵀ once per head per step; the 128×128 inverse GEMM never runs
in-kernel), split-K flash-decoding, GQA group padded to M=16.

v1 numbers (A100, b=2.0, bf16 SDPA `enable_gqa` baseline, correctness rel_err ≤ 2e-2
(TF32-class)):

| ctx | fused | SDPA bf16 | speedup | traffic e2e | traffic K-only |
|---|---|---|---|---|---|
| 8k | 0.240 ms | 0.095 ms | 0.40× | 1.78× | 8.0× |
| 32k | 0.569 ms | 0.144 ms | 0.25× | 1.78× | 8.0× |
| 64k | 1.036 ms | 0.240 ms | 0.23× | 1.78× | 8.0× |

**Honest verdict: correct but not yet competitive.** v1 (monolithic (64,128) fp32
dequant block): 0.23–0.40×. v2 (16-token fp16 sub-tiles, 8 warps, adaptive split-K):
0.19–0.20× — *worse*; register pressure was not the binding constraint. The bottleneck
is instruction-boundedness: ~25k scalar masked byte-loads + address computations per
page (per-(token,coord) pointer arithmetic across 3 width classes) against SDPA's
clean 128-bit vector loads. The byte-traffic advantage is real (1.78× end-to-end with
V fp16; 8× K-only) but unrealized.

**Identified fix (designed, not built tonight): width-planar pages.** Stable-sort
tokens by width inside each page so each width class is one dense contiguous block →
coalesced u32 vector loads + vectorized field extraction, no per-element gathers. The
permutation is recomputed by the decoder from the width-id header (deterministic
stable sort ⇒ **zero extra sideband**), and online softmax is order-invariant, so
processing key blocks in permuted order is exact; V rows are gathered by permuted
index (contiguous 256 B rows). Same bundle, same allocation — only `pack_pages` and
the kernel inner loop change.

## Caveats / next steps

- Single model (Llama-3.1-8B), 3 tasks, one run per cell; the decisive ω contrasts
  carry bootstrap CIs but musique (n=150) stays wide. Qwen replication requires
  re-capture (no Qwen raw pools on disk).
- ω hyperparameters (τ=0.5, c=4) frozen from one oracle grid on selection rows;
  a τ-schedule by rate and an ω-confidence gate (fall back to ω=1 on low
  within-page m-spread — targets the lcc failure) are the obvious next knobs.
- pgq_fixed's negative is 4-iterations deep with probe-verified fixes, but a
  residual subtle bug can't be fully excluded; the eviction-regime sink-mass
  overshoot is structural evidence the representation (not the tuning) is the limit.
- Kernel: width-planar page layout (header-derived permutation, zero sideband)
  designed but unbuilt; rANS host-side parsing dominates the codec path and was
  not optimized (kernel-only numbers reported).
- Full byte-level G1 stream gate for the new page format (3-bit ids) not re-run;
  press accounting is analytic (frozen-model nlp + charged sideband).
- Remaining LongBench tasks (9) and Mode B unbenched, as in the EC study.

## Artifacts

- Bundle: `artifacts/page_quant/pgq_bundle__qpca_unc__dz0.5__base1.5__compact8train18.pt`
- Phase 1: `artifacts/page_quant/phase1/{predictiveness,headroom,headroom_v1_negctrl}.json`
- Held-out gates: `artifacts/page_quant/pgq_heldout_report.json`
- Phase A: `artifacts/page_quant/phaseA_pgq.json`
- Bench cells: `artifacts/bench_pgq/llama31_8b/` · Summary: `artifacts/page_quant/bench_summary.json`
- Code: `kvq/compression/page_quant.py`, `pipelines/page_quant/{phase1_empirics,fit_pgq_bundle,score_pgq_candidates,bench_fused_kernel}.py`,
  `pipelines/bench/launch_pgq_longbench.sh`, `pipelines/eval/aggregate_pgq.py`,
  `tests/test_page_quant.py`
- Logs: `logs/page_quant_*.log`, `logs/bench_pgq_llama31_8b/`
