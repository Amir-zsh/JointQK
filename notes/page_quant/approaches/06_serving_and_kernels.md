# Approach 6 — Serving architecture & kernels

**Verdict: measured, not speculated. Mode A for quality + cold tiers; the fixed-width
resident path has a proven kernel blueprint (OSCAR) if its F1 price is ever accepted.**

## The decode-path physics (measured on our CUDA rANS)
- Throughput 1.2–1.5 Gsym/s; entropy kernel only ~10% of wall time (stream parsing
  dominates). A resident EC cache must re-decode the whole context per generated
  token: at 64k context ≈ 2 Gsym → **~1.8 s/token vs a ~5 ms attention budget.**
  Parallel lanes exist (~4M); the architecture is the obstacle: entropy streams have
  no random access mid-page → in-kernel EC decode is structurally impossible.
- Resolution: **Mode A** — encode once at prefill (<0.1% of prefill compute), cache
  holds reconstructions, decode path reads plain fp16. EC decode runs only on tier
  transitions (offload reload, prefix-cache load), where it is faster than the PCIe
  transfer it shortens.

## Where memory is and is not saved
| tier / mode | resident bytes | quality | notes |
|---|---|---|---|
| Mode A hot cache | **unchanged (fp16)** | ≈FP at 1.95 b/c codec rate | quality tool + zero decode cost |
| off-GPU EC pages | 1.0–1.5 b/c (10–16×) | same on reload | transfers/storage; decode-once |
| eviction (approach 5) | ÷(1−ratio) | EA −1.4 / ω −5.1 @0.50 | no codec at all |
| fixed-width resident | ~2.1 b/c (~7.5×) | 45.7 @2.125 (TQ class) | kernel dequant; the OSCAR path |

## Kernel blueprints (for the fixed-width resident path)
- **OSCAR (vendored, mapped):** segment-partitioned decode — one stage-1 attention
  kernel per precision tier writing disjoint split ranges of shared scratch, one
  tier-agnostic online-softmax stage-2 merge; INT2 quant/dequant Triton kernels are
  torch+triton-only and lift out; rotation builder is pure torch. R_V absorbed into
  the output projection. Measured by them: ~3× batch-1 decode at 100k (bandwidth-bound),
  ~8× resident. Port cost = glue, not kernels.
- **Width-planar layout (ours, prototyped pgq1):** rotated-domain scoring
  (logits = idx·q′·Δ·m) hit 0.2–0.4× SDPA instruction-bound; width-planar storage was
  designed as the fix but only earns investment if a no-EC codec deploys — per
  approach 4, that means the ≥2 b/c operating point, where OSCAR's design is the
  better starting skeleton (one stage-1 kernel per rung width).

## Artifacts
`pipelines/page_quant/{probe_rans_throughput,bench_fused_kernel}.py`, pgq1 kernel
prototype notes in `studies/report.md`, OSCAR map in `studies/report3.md` + session
audit; vendored kernels under `vendor/OSCAR/sglang-research/`.

## Open directions
- If a ~2 b/c resident tier is wanted: port OSCAR kernels, swap their basis for ours
  (approach 1) and add per-token rungs (approach 3's regime) — the combination neither
  project has benchmarked end-to-end.
- Engine-level ω page skip (see approach 5) shares the same segment machinery.
