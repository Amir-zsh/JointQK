# plan3 (pre-registration): ω page selection + reopening the no-EC door

**Registered:** 2026-07-08, branch `paged_quant` @ c7c8358. **Status: ACTIVE**
(hold lifted 2026-07-08; code implemented remotely via Ultraplan, integrated as two
`git am` commits; amendments below registered before any GPU run). All tunables to be frozen on compact8-TRAIN selection rows
before any F1; row-paired 10k bootstraps on all claims; v7 protocol for Thrust B;
kvpress eviction conventions for Thrust A (NOT cross-comparable — different protocols).

## Thrust A — ω-driven page selection

**A1 probe** (`pipelines/page_quant/probe_page_selection.py`, ~2 GPU-h): recall of
realized attention mass vs fraction of 64-token pages kept (X ∈ {5,10,25,50}%), sink +
recent pages always kept, on selection rows (+ posthoc report-only). Scorers:
`omega_max`, `omega_mean` (static, calibrated μ̄_q), `quest_true` (per-page k_min/k_max
box scored with the TRUE query — the query-aware ceiling), `quest_mu` (box scored with
μ̄_q), `incontext_mu` (row's own prefill-query mean — the ExpectedAttention-style
in-context signal phase-1 never tested), `oracle`, `random_page`/positional controls.
*Frozen rules:* press score_mode = argmax recall at X=25%; A3 (query-aware press) built
only if `quest_true` beats best-static by >10 recall pts at X=25% (layers 8–31 mean).

**A2 static press** (`kvq/presses/omega_page_press.py`, ScorerPress subclass with
page-granular compress; sink page + recent page(s) protected; per-head keeps; registered
as class `omega_page`; new `launch_evict_longbench.sh` + `aggregate_evict.py`;
bootstrap_pairs gains `--bench-root/--cell-glob`). Cells: wave A2.1 = {omega_page,
expected_attention, snapkv, knorm, streaming_llm, random, omega_page[random_page]} ×
ratio 0.50 × 5 tasks (35 cells); A2.2 = top-4 × {0.25, 0.75} × 5 (conditional on
omega_page ≥ expected_attention − 0.5).
*Bars:* primary omega_page − expected_attention ≥ 0 at 0.50 (CI reported; CI-lower >
−1.0 = minimum claim, SIG > 0 = headline); secondary > streaming_llm AND > random_page
SIG; non-lcc mean reported separately.
*Claim ladder (pre-agreed):* if `incontext_mu` dominates calibrated scorers offline, the
headline becomes "page-granular selection with zero runtime query statistics is
competitive" (valuable for offloaded pages), not "ω beats ExpectedAttention".

**A3 (gated):** decode-time forward_pre_hook press setting `masked_key_indices` per step
from the Quest bound (quality-only emulation; memory wins live in the sglang engine,
out of scope). 5 cells at 0.50.

## Thrust B — no-EC reopening (bar: random-access pages, no entropy coding, honest rates)

**Two-tier bar at 1.496 b/c:** tier 1 = SIG > 42.62 (rvq_rdo record); tier 2 (headline)
= ≥ 44.7 (half the gap to ecu 46.80). Tier-2 success is the trigger to resume kernel
work. Expectation set honestly: theory caps lattice+trellis gains ~0.1–0.2 bits/dim;
EC's edge is mostly near-free zero-coordinate coding — family (e) exists for exactly
that reason.

**Families** (all on the `(Dmat, rate_bits)` contract; norms + forced-sink absolute-grid
rung inherited; zero level mandatory in every rung):
- (c) **Page-reset TCQ** (`kvq/compression/tcq.py`): 8-state trellis over 128 coords,
  warped empirical Lloyd–Max level sets (collapsed to per-coord tables), Viterbi encode,
  table-walk decode, token-granular random access. **1-state control rung = family (a)
  compander, free.** Trellis states {4,8} + warp p ∈ {0.5,1,1.5} frozen on selection rows.
- (b) **E8 Voronoi product codes** (`kvq/compression/e8.py`): 16 subvectors × 8 dims,
  Conway–Sloane Voronoi coding (closed-form encode/decode, zero trained tables), rungs =
  water-filled per-subvector bit profiles {0,1,2} bits/dim, 8-bit/token ladder steps;
  overload priced in snap-aware Dmat via TRAIN-q999 scaling.
- (e) **Fixed-K sparse significance coding** (inside tcq.py's ladder): top-K warped
  coords, K 7-bit indices + magnitudes, fixed width — the fixed-rate analogue of EC's
  deadzone advantage. ADOPTED per design review.
- (d) **Learned linear pair** (`pipelines/page_quant/train_linear_pair.py`): first
  autograd loop in repo; per-head (F,G)+scales, straight-through quantization, loss =
  code-domain SE on the 500k-token bigfit pool; overfit gate 1.3 with per-head fallback
  to qpca_unc. Sequenced last (user: all four in scope).

**Wiring:** routing prefixes `pgq_tcq_ / pgq_e8_` (+ sparse rung inside tcq) in
`page_quant.py:_load_pgq2`; `fit_pgq3_bundle.py` sibling (BigPool reuse, bigfit pool,
G1–G3 gates → `pgq3_heldout_report.json`); freeze file `frozen_choices_pgq3.json`
strictly before F1 (sha8 labels); scorer gains `--bundle`; launcher gains PGQ3_BUNDLE
branch; `aggregate_pgq.py` gains HELDOUT3.

**Gates (blocking):** G1 rate ±2%, overflow <1%; G2 sink code-relerr ≤0.05 AND
mass shift ≤ incumbent(pgq_rvq_rdo at matched rate) + 0.02; G3 |norm ratio − 1| ≤
|incumbent − 1| + 0.02; G-P3 screen: beat pgq_rvq_rdo on top1 AND logit_err at same
rate on selection rows or no F1 cells; G0-perf: ≤2× rvq press wall-clock on
fraction-0.05 smoke.
*G2/G3 recalibration (2026-07-08, before any pgq3 F1):* the absolute thresholds first
transcribed here (shift ≤0.01, normR ∈[0.98,1.02]) are stricter than what the pgq2
record-holder itself achieved (rvq_rdo@1.5: shift 0.054, normR 0.963 —
`pgq2_heldout_report.json`); no method at these rates has ever met them. Gates are
therefore incumbent-relative (do no worse than the arm behind the 42.62 bar); the F1
bar itself is unchanged and remains the arbiter.

**Cells:** B1 = gate-passing families @1.5 × 5 tasks (≤15 cells incl. sparse rung
variant); B2 = winner @1.0 × 5 (if ≥44.0); B3 = family (d) @1.5 × 5 (sequenced last).
ω variants excluded from wave 1 (fine ladders were ω-neutral).

## Sequencing & budget

Day-1 information blocks (GPU 0 alone OK): A1 probe → B fits/select/gates → B screening
(~9 GPU-h). Then B1 (decisive, 3 GPU-h) → A2.1 (~8 GPU-h) → conditionals. Ceiling
~38–45 GPU-h. Launchers default GPUS=0 until the GPU 1–3 orphans are cleared (34–36 GB
residents would OOM workers).

## OSCAR amendments (registered 2026-07-08, after reading raw/OSCAR.pdf + vendor/OSCAR)

1. **Code-domain Hadamard mixing** (frozen per-family choice): orthogonal
   Hadamard(+bit-reversal) composed after qpca_unc whitening preserves code-SE ≡
   Q-weighted MSE exactly ((QG)Σ_Q(QG)ᵀ = I). ON for E8 (isotropic subvectors, one
   shared β) and TCQ shared tables; OFF for sparse-K (needs compacted energy).
   Realized per-rung inside the TCQ ladder; parity vs
   `vendor/OSCAR/rotation/compute_kv_rotation.py` helpers to be spot-checked.
2. **Family (f) `pgq_oscar_*`** — OSCAR-emulation baseline: per-token quantile clip
   (ρ≈0.96) + per-(token,group) asymmetric min–max INT-w, fp16 scale+zero charged
   (32 b/group/token); faithfully zero-less and sink-agnostic (per-token scales).
   Published-SOTA anchor on our protocol (~2.25 b/c INT2 point; second width point
   chosen at fit time). ~5 F1 cells.
3. **Recent-window protection ablation** (~3 cells): force last 1–4 pages to max rung
   on the best arm + ecu, rate charged honestly (OSCAR's (S,R)=(64,256) knee).
4. **Thrust C (deferred, registered):** V-side score-weighted rotation; C_S estimated
   as diag(K·C_Q·Kᵀ)-weighted VᵀV on existing captures. Separate phase — changes the
   V protocol and breaks v7 comparability.
5. **Kernel plan on tier-2 success:** port OSCAR's Triton INT2 kernels and its
   disjoint-split-range stage-1-per-tier + unified online-softmax stage-2 merge
   (`vendor/OSCAR/sglang-research/.../decode_attention.py`); width-planar = one
   stage-1 kernel per rung width. Report3 frames OSCAR as complementary (channel-wise
   rotation-flattening vs our token-wise allocation; their limitations §D names
   per-token allocation as open).

## Risk flags carried from design review

R1 tier-2 bar likely needs family (e); R2 ExpectedAttention baseline uses in-context
stats (claim ladder above); R3 page-granularity confound (random_page control
mandatory); R4 thrusts not cross-comparable; R5 TCQ prefill cost (G0-perf gate);
R6 orphan GPUs.
