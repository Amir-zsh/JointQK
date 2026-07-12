# plan6 (pre-registration): pgq6 — merged-page codec (pruning ∪ quantization)

**Registered:** 2026-07-11, branch `pgq6` (cut from `pgq5` @ a8b4e86).
**Status: REGISTERED.** P0/P1 ran ahead of the branch cut on user-approved
plan sequencing (all selection-rows-only; F-6 frozen before any F1; the sole
F1 wave — the 2.0-parity cells — launched after the P1 gate verdict per the
registered tree). Protocol v7 (Llama-3.1-8B-Instruct, 5-task LongBench, FP
48.40, layer-0 fp16, V = TurboQuant 2b, honest all-in rates, compact8-TRAIN
fitting, row-paired 10k bootstraps). Qwen transfer later via the pgq5 Stage-A
machinery (--model-tag).

## 0. Thesis

Project outline (user, 2026-07-10): next-generation OSCAR — page-unit
quantization assigning bits non-uniformly across tokens, so eviction (0 bits)
and OSCAR (uniform bits) are interior points of one format. pgq4 validated the
per-token-bits half. pgq6 adds the missing axis: **n→m token merging inside
pages** — the page stores m < 64 rows (contiguous-run centroids + 6-bit run
counts), reconstructed exactly as the bias form (k̄, v̄, log c) via centroid
replication. Two registered questions:

1. **Sub-wall rates:** the pgq3 ~1.5 b/c closure is a *per-token* statement;
   merging cuts token count, not per-coord rate. Can (merge × width) RDO beat
   the closure bars at 1.0-1.5 b/c per ORIGINAL token?
2. **Parity behavior:** at 2.0 b/c does merging cost anything (RDO should
   simply not merge when bits are ample)?

**Signal policy (frozen):** phase-symmetric only — calibration ω, key-intrinsic,
position. Phase-asymmetric signals (observed attention) are a DEFERRED
DIRECTION; they may appear only as a screening oracle / post-hoc diagnostic.

## 1. Format (implemented, P0 complete 2026-07-10)

`kvq/compression/pgq6_merge.py::MergedPageCompressor` (loader family
`pgq_mrg*`, e.g. `pgq_mrglm_rdo`, `pgq_mrglmrw_rdo`) — v1 contract:

- Fixed-byte pages (b_page·d·64 bits each). Merge levels M ∈ {64, 32, 16}
  (power-of-2 size classes {S, S/2, S/4} for deployment). Per page, one
  adjacent-Ward hierarchy (nested snapshots) on the rotated codes r;
  in qpca_unc, unweighted r-distance IS Σ_Q-weighted raw distance.
- Per (page, M): per-row width RDO (`_paged_lambda_assign`, pgq4 machinery,
  LM grids, width profiles reused from the pgq4 bundle — NO REFIT; clustering
  is runtime). Page picks the M with min achieved distortion under the SAME
  budget; ties → less merged (conservative; merged-tie preference = ablation).
- Exact distortion decomposition (cross term vanishes per cluster):
  Σ‖r_i−Q(r̄)‖² = spread + c·‖r̄−Q(r̄)‖².
- Rate: rows charge profile bits + 6-bit count (M<64); side = header + 2-bit
  M-class + 3-bit width ids per row. Cluster maps are NOT charged (counts ARE
  the map for contiguous runs).
- Protection (phase-symmetric): sink rows 0-3 absolute escape, page 0 forced
  unmerged; force_recent_pages forced unmerged at top width. ω/gain
  unsupported in v1 (asserted).
- Eval form = centroid replication (identical to bias form; softmax over c
  identical keys ≡ +log c logit — test-pinned). K-only merging in eval keeps
  V per-token, which is EXACTLY output-equivalent to deployment storing
  v̄ = mean (equal within-cluster weights), so V stays v7-constant
  (TurboQuant 2b on n rows) and F1 comparability holds. Deployment V-row
  reduction is reported as a separate storage number, not an eval variable.

Tests: `tests/test_pgq6.py` 6/6 green (replication↔bias identity, rate
accounting incl. counts, redundancy-driven merging, ample-budget no-merge,
sink/recency protection, loader routing). Sanity on real data (selection row,
(l=8,h=3), b=1.25): rate 1.2455, ovf 0, 37% of tokens at 2:1 merge, rows
−19%, single-threaded CPU 17.6 s → screening/bench run on GPU.

## 2. P1 — screening (selection rows, ~4 GPU-h, freeze F-6 before any F1)

Harness: `score_pgq_candidates.py` unchanged (replicated roundtrip is
same-shape). Llama bundle ff390304, 8 selection rows, q_cap 8192.

- Arms: `pgq_mrglm_rdo` at per-original-token rates {1.0, 1.25, 1.5, 2.0};
  `pgq_proflm_rdo` at the same rates (paired no-merge control).
- Bars (registered): at 1.5 — beat BOTH pgq3-closure references
  (ecu 0.00318, rvq 0.00405 logit_err); at 2.0 — within 5% of proflm
  (0.00272). Monotone in rate.
- **Value-coupling verification** (one-off): direct attention-output error
  (softmax·V vs fp16 reference) on 2 selection rows × G2 layers confirms the
  exactness argument above; if replicated-K logit_err misranks vs output
  error, stop-and-amend.
- **Oracle ablation (screening only):** allocation driven by realized
  attention (prompt queries) as an upper bound — quantifies phase-symmetric
  headroom. Never promoted to the codec.
- Telemetry: merge_hist by task (code vs prose redundancy), rows_total.
- Freeze F-6 (`frozen_choices_pgq6.json`): winning rates, tie-break rule,
  merge levels, all constants — selection rows only.

## 3. P2 — F1 wave W1 (v7 5-task, ~12-18 GPU-h)

Cells (launch_pgq_longbench.sh, Mode A, labels carry bundle sha8):
1. `pgq_mrglmrw_rdo` at the lowest admitted rate (target 1.25 or 1.5) — the
   sub-wall headline.
2. `pgq_mrglmrw_rdo:2.0` — parity arm.
3. `pgq_proflmrw_rdo:2.0` — pgq4 incumbent control (existing cells reusable).
4. OSCAR arm at its native INT2-class all-in rate (existing pgq3/4 cells).
Refs: FP, TQ, ecu, rvq — all existing. Row-paired 10k bootstraps; per-task +
pooled + non-lcc; report6.md.

## 4. Gates & decision tree

- P0→P1: tests green (done); heldout G1 rate ±2% / ovf <1%, G2' sinkCE ≤0.01,
  G3' |normR−1| ≤0.05 on the mrg arms at 1.5 and 2.0.
- P1→P2: some merge rate ≤1.5 beats both closure bars → sub-wall F1 wave;
  else wall stands (negative registered, only the 2.0-parity cell runs).
- W1: (a) sub-1.5 arm F1 within CI of proflmrw@2.0 → wall broken (headline);
  (b) parity tie at 2.0 → merging free, format default; (c) per-task merge
  histograms: lcc/code merges more → redundancy-adaptivity demonstrated.
- Failure attribution: if mrg loses at ALL rates incl. 2.0 → clustering
  metric wrong (r-space Ward) — one registered retry with raw-k-space
  clustering; if it loses only sub-1.5 → dimension wall extends to the token
  axis (clean negative, valuable).

## 5. Budget

P1 ≈ 4 GPU-h + P2 ≈ 12-18 GPU-h → ≈ 16-22 GPU-h, cap 30 (Part-2 share of the
approved 50 total). GPUs 0-3, yield to user jobs. Logs logs/pgq6_*.log +
heartbeat. Commits only with per-commit approval; fingerprint 262/262 before
each.

## 6. Deferred (recorded)

Phase-asymmetric signals (prompt-window / ring-lifetime attention);
ω-protected merging arm; merged-tie preference; general (non-contiguous)
clusters; cross-page merging; V-row-reduction serving numbers; kernel port
(d+1 bias fold); Qwen transfer of the winner; pgq5 C0/C1 thinking study.
