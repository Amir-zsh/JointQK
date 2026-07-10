# Approach 2 — Entropy-coded fixed-byte pages + Mode-A serving

**Verdict: the deployed method. Owns quality at ≤2 b/c.**

**Lineage — credit where due:** the codec itself (qpca_unc basis + deadzone scalar
quantizer + frozen-model paged rANS) and its rate–quality curve are the **ec_k2v2
work** (`notes/ec_k2v2_report.md`), which predates this arc. page_quant did not
improve that curve. What this arc added on top: the fixed-byte page FORMAT (proved
free), per-token RDO allocation (what makes fixed pages fillable), the ω regime law
(approach 3), sink hardening, and the measured serving verdict (approach 6). If Mode-A
quality without paged serving had been the only goal, ec_qpca was already sufficient.

## Idea
In the qpca_unc code domain, quantize per-coordinate with a deadzone quantizer, choose
a per-token precision rung inside fixed-byte 64-token pages (per-page Lagrangian RDO),
and entropy-code the symbols with frozen-model paged rANS. Compress ONCE at prefill;
the resident cache holds the reconstructions (Mode A) so decode adds zero cost.

## What was built
`kvq/compression/page_quant.py` (PagedRDOCompressor + `_paged_lambda_assign`),
EC bundles + `ec_`/`pgq_` press routing (`kvq/presses/jointqk_press.py`), launcher
`pipelines/bench/launch_pgq_longbench.sh`, aggregators, `bootstrap_pairs.py`.

## Results (5-task F1, FP = 48.40; trio in parens)
- **≈FP at 1.95 b/c codec rate:** 44.90 vs 44.99 trio.
- 1.5 b/c: **46.80** (43.0) · 1.0 b/c: 45.22 (41.3). TurboQuant ref: 45.73 @2.125.
- Fixed 64-token pages vs unconstrained bitstream @1.5: 43.98 vs 43.02 trio — ties on
  every task; oracle page-constraint cost ~1% of distortion. Per-token allocation
  fills 99.6% of page bytes (uniform strands 18%).
- Sink invariants (mandatory, measured failure physics): forced max rung positions
  0–3; exact zero level in every grid; absolute scales for sinks (q999 scales clip
  sinks: mass 0.84→0.007 → whitespace). OSCAR replicates via their BF16 windows.

## Memory honesty (see also approach 6)
Mode A does NOT shrink the hot cache (reconstructions are fp16). The 10–16×
compression is realized off-GPU: offload, prefix caches, stored sessions, transfers —
decoded once on reload, PCIe-bound. Resident-byte reduction requires approach 4's
fixed-width codecs (with their F1 price) or approach 5's eviction.

## Artifacts
`artifacts/page_quant/bench_summary.json`, cells under `artifacts/bench_pgq/`,
EC bundles under `artifacts/page_quant2/`.

## Open directions
- Recent-window protection ablation (registered, never run — OSCAR's (64,256) knee
  suggests forcing the last 1–4 prompt pages to max rung could lift all arms).
- Qwen3-8B replication; V-side (Thrust C) would compound with this approach.
