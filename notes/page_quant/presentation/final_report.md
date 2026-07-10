# Page-Unit KV-Cache Quantization: Findings and Deployment Recommendation

**Study window:** 2026-07-07 → 07-08 · **Branch:** `paged_quant` · **Model:** Llama-3.1-8B-Instruct
**Presentation version.** The full technical records are `report.md` (study 1), `report2.md`
(study 2), and `fixes_to_apply.md` (bug post-mortems). Every number below has a per-cell
artifact on disk and, where a comparison is claimed, a row-paired bootstrap confidence
interval behind it.

---

## The three takeaways

**1. What to deploy.** Compress keys with the entropy-coded codec **once, at prefill**, and
let the cache hold the reconstructed keys (the "Mode A" pattern — the same pattern every
benchmark in this project already runs). This delivers essentially full-precision accuracy
at a 1.95 bits-per-coordinate codec rate (44.90 vs 44.99 mean F1), degrades gracefully down
to 1.0 bits (41.25), and requires **no decompression machinery in the generation loop at
all**. Be precise about what is saved where: the hot GPU tensors stay fp16 (Mode A does
**not** shrink the resident cache — see §8), while the 8–16× compression is realized
wherever the coded pages are stored or shipped: CPU offload, prefix caches, stored
sessions, cross-machine transfer. Shrinking the *resident* cache instead requires eviction
or a fixed-width codec, at the measured F1 prices in §7–8. For cache tiers that live off-GPU — CPU offload, prefix caches,
stored sessions — use entropy-coded pages too: they are decoded once on reload, where
decode speed is irrelevant and the extra compression directly cuts transfer time.

**2. What we proved works.** The core idea of this study — *give more precision to the
tokens that attention will actually use* — works, and we now know exactly when. It is worth
**+3 to +4 F1 points (statistically significant)** whenever the quantizer's rate ladder is
coarse enough that ordinary allocation starves the important tokens. It costs nothing extra
to transmit (the choice is encoded in 2–3 bits per token that the format needs anyway). And
fixed-size 64-token pages — the format that paged attention serving needs — cost
**statistically nothing** at 1.5 bits per coordinate.

**3. What we ruled out.** A quantizer that can be decoded by pure array indexing (no entropy
coding), which is what you would want for decompressing inside the attention kernel, does
**not** reach entropy-coded quality at these compression rates. We built the two strongest
candidates, fixed every failure the probes found, retrained on 5× data, and the gap
remained: **−4.2 F1 at 1.5 bits (CI [−5.8, −2.6]), −12 at 1.0.** The gap is the entropy
model itself, not a fixable defect. This door is now closed with data, not opinion.

---

## 1. The question we set out to answer

The proposal: quantize the key cache in **page units** — blocks of 64 consecutive tokens,
matching how paged-attention servers and FlashAttention tiles already organize memory —
and, inside each page, **spend unequal bits on unequal tokens**, using the calibrated query
distribution to predict which tokens matter.

A provenance note before the findings: the underlying codec — the qpca_unc basis with
deadzone quantization and paged rANS — and its rate–quality curve come from the earlier
**ec_k2v2 study** (`../../ec_k2v2_report.md`); this study did not improve that curve and
does not claim to. What is new here is everything around the codec: the page format
(proved free), per-token allocation (what makes fixed pages fillable), the importance
regime law, sink hardening, the measured serving verdict, and the mapped-and-closed
fixed-width frontier.

Two separable claims live inside that idea:

- a **format** claim — fixed-byte pages are the right container (serving-friendly,
  random-access, block-allocator compatible);
- an **allocation** claim — per-token precision, driven by predicted attention, beats
  giving every token the same treatment.

Both turned out true. The surprise was in a third, hidden question — *what representation
goes inside the page* — and that is where the decisive economics live.

## 2. How we tested it

All experiments: Llama-3.1-8B-Instruct, the project's standard bench protocol (layer 0 kept
at full precision, values quantized identically across all methods so keys are the only
variable, calibration rows excluded from evaluation). Five LongBench tasks: the established
trio (lcc = code completion, musique = hard multi-hop QA, 2wikimqa = two-hop QA) plus
qasper and hotpotqa for generalization. Full precision scores 48.40 on the five-task mean.

Discipline, because this project has been burned by proxies before:

- **Pre-registration.** The experiment grid, success bars, and interpretation rules were
  written down before any downstream result existed; hyperparameters were frozen (with a
  git-stamped audit file) on calibration-side rows only.
- **Blocking gates.** Every codec had to pass rate honesty (every side-band bit charged),
  a sink-preservation probe, and a norm-preservation probe *before* it was allowed to spend
  GPU-hours on accuracy benchmarks. These gates caught six real bugs at probe cost.
- **Paired statistics.** Every claimed difference is a row-paired bootstrap over per-row
  scores, not a comparison of two averages.

## 3. Finding: which tokens actually matter (not the ones theory said)

To allocate precision by importance you need an importance signal computable from the key
alone. The literature's answer (Expected Attention) and the intuitive answer (Q-weighted
key energy, `kᵀΣ_Q k`) both fail on Llama — energy doesn't just fail, it points the wrong
way:

| candidate signal | rank correlation with realized attention |
|---|---|
| Q-weighted energy `kᵀΣ_Q k` | **−0.78** (anti-predicts!) |
| textbook Expected Attention | ≈ 0 |
| calibrated mean logit `μ̄_qᵀk/√d` | **+0.73 … +0.82** |

High-energy keys are logit *repellers* — tokens the model pushes far into negative
attention scores. They tolerate huge quantization error (softmax gradients vanish there),
while the tokens that receive attention are found reliably by the humble mean-logit signal
(it captures ~85% of the top-decile attended tokens, and transfers to out-of-distribution
tasks). Two structural facts frame everything else: the first four tokens of a sequence
(the "attention sinks") carry 19–84% of attention mass per head, and inside a typical
64-token page, the top 8 tokens carry ~43% of the page's mass.

**A second, subtler discovery:** plain rate-distortion-optimal allocation actively *hurts*
attended tokens — they tend to be expensive (high energy), so an unweighted optimizer takes
bits *away* from them. Importance weighting is not a bonus on top of RDO; it is the
correction RDO needs.

## 4. Finding: the allocation idea works — as a regime law

We implemented per-token precision as a Lagrangian choice among quantizer "rungs" inside
each fixed-byte page, with the distortion term multiplied by the importance weight ω
(derived from the mean-logit signal). The decoder only ever reads the 2–3-bit rung labels,
so ω costs nothing to transmit and cannot leak.

Measured across **three different quantizer families**, the same pattern appeared:

| where | effect of ω on F1 | significance |
|---|---|---|
| entropy-coded rungs @1.0 bits | **+2.9** mean (musique **+5.8**) | SIG |
| scalar direction codes @1.0 bits | **+3.7** mean (hotpotqa **+8.6**) | SIG |
| fine-grained vector quantizer @1.0 bits | −1.0 | tie |
| entropy-coded rungs @1.5 bits | −1.6 (lcc **−2.4**) | lcc SIG |

The law: **ω pays exactly when the rate ladder is coarse enough that unweighted allocation
starves the attended tokens.** Give the ladder fine granularity (the vector quantizer's
0.5-bit steps) and it protects important tokens on its own; give the system abundant rate
(1.5 bits) and there is nothing to rescue — and on out-of-distribution code (lcc), where
the importance predictor is weakest, misdirected ω does mild harm (a gating fix was tried
and cleanly failed; this is the one open edge).

## 5. Finding: fixed pages — the serving format — are free

At 1.5 bits per coordinate, the entropy-coded quantizer confined to fixed-byte 64-token
pages scores **43.98** (3-task mean) versus **43.02** for the same quantizer with an
unconstrained bitstream — every task a statistical tie, about one point below full
precision, at ~10.7× compression. Two supporting facts: the oracle cost of the page
constraint is ~1% of distortion, and per-token allocation is precisely what makes fixed
pages *fillable* — a page-uniform scheme strands 18% of the byte budget, per-token
allocation fills 99.6% of it.

## 6. Finding: entropy coding is too slow for decode — measured, and it doesn't matter

The concern was that entropy coding is serial and slow. Measured on the project's own CUDA
rANS decoder: 1.2–1.5 billion symbols/second (and the entropy kernel is only ~10% of wall
time — stream parsing dominates). A resident compressed cache must re-decode the *entire
context for every generated token*: at 64k context that is ~2 billion symbols, i.e. **~1.8
seconds per token against a ~5 ms attention budget**. Parallelism is not the obstacle —
four million independent decode lanes exist; the architecture is. And entropy streams
cannot be randomly accessed mid-page, so fusing decode into the attention kernel is
structurally impossible.

The resolution is that decode speed never needs to matter: in the Mode-A pattern the cache
stores *reconstructed* keys — compression happens once at prefill (encode cost is <0.1% of
prefill compute), and generation reads ordinary tensors. Entropy decode only ever runs
one-shot, on tier transitions (offload restore, prefix-cache load), where it is faster than
the PCIe transfer it shortens.

## 7. Finding: the hunt for an entropy-free codec — a closed door, with a map

If you instead want the *resident* cache itself compressed (maximum context per GPU), you
need a random-access codec. We pursued this through nine design iterations and closed it.

**The graveyard, briefly** (each failure caught by probes, each invisible to accuracy
proxies): scalar grids without a zero level inflate every key's norm coherently → F1
collapses to ~0 while top-1 metrics look *fine*; "tail-safe" 99.9th-percentile scales still
clip the attention sinks (which sit 2.7–5.4× beyond them) → sink attention mass 0.84→0.007
→ whitespace generation; even covering 4-bit grids are too coarse for sinks unless
positions 0–3 are explicitly forced to 8 bits. And one level deeper in study 2: sink
*directions* are outliers too, and only an absolute ±1 grid (legal because directions are
unit vectors) is outlier-proof.

**The two strongest candidates**, built clean on top of all those lessons — per-token fp16
norms (which dissolve the sink-scale problem) plus either width-adaptive scalar direction
codes or a four-stage residual vector quantizer with per-token depth:

| bits/coord (all-in) | scalar dirs | + ω | residual VQ | entropy-coded | full precision |
|---|---|---|---|---|---|
| ~1.14 | 12.9 | 23.1 | **33.0** | ~45.2 | 48.4 |
| 1.5 | 27.6 | — | **42.6** | **46.8** | 48.4 |

Each mechanism earns its keep in order — norms, then per-token allocation (+5), then ω
(+5), then vector quantization (+13, significant) — and the last 4-to-12-point gap belongs
to entropy coding alone. The final pre-registered test: retrain the VQ codebooks on **5×
the calibration tokens** (a fresh 40-row capture). Result: 42.62 → 42.06 — a wash. **The
gap is the entropy model over coordinate values. It is fundamental at these rates, not a
data or tuning problem.** Anyone who wants to reopen this door needs a different idea class
(learned transforms with lattice/trellis codes), and now has the exact bar to clear.

## 8. The deployment recommendation

| tier | codec | rate | quality (5-task / trio) | decode cost |
|---|---|---|---|---|
| **GPU-resident (hot)** | EC at prefill, cache holds reconstructions (Mode A) | 1.95 b/c | ≈FP (— / 44.90) | zero (reads plain tensors) |
| memory-pressure mode | same, lower budget | 1.5 / 1.0 | 46.8 / 45.2 (43.0 / 41.3) | zero |
| **off-GPU (cold)** | EC fixed-byte pages | 1.0–1.5 | same as above on reload | one-shot, PCIe-bound |

Reference points: TurboQuant 45.7 at 2.125 b/c (we match it at *half* the rate);
uncompressed fp16 is 16 b/c.

## 9. What we would do next

1. **Use ω for page *selection*, not just precision** — a Quest-style decode-time page
   skip driven by the mean-logit signal; attacks latency and memory together, works with
   any page contents, and reuses the study's most robust finding.
2. **Replicate on Qwen3-8B** (requires a calibration re-capture; no raw pools survive).
3. **If the compressed-resident-cache goal returns:** learned transforms + lattice codes,
   measured against this study's grid; and only then the width-planar fused kernel.

## Appendix: where everything lives

- Full grids & CIs: `artifacts/page_quant/bench_summary.json`, bootstrap tool
  `pipelines/page_quant/bootstrap_pairs.py`
- Gates & audit: `artifacts/page_quant2/{pgq2_heldout_report,frozen_choices}.json`
- Code: `kvq/compression/{page_quant,norm_direction,rvq}.py`, fit/select/launch/aggregate
  under `pipelines/page_quant/` and `pipelines/bench/launch_pgq_longbench.sh`
- Captures: `artifacts/calibration/pgq2_bigfit_llama31_8b/` (40 train rows, reusable)
- Technical reports: `report.md`, `report2.md`; bug post-mortems: `fixes_to_apply.md`
