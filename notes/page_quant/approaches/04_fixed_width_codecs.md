# Approach 4 — Fixed-width random-access codecs (the closed door)

**Verdict: closed with data at ≤1.5 b/c, against the full idea class. Effective
vector dimension is the binding constraint; EC's zero-coding edge is not replicable
fixed-width.**

## Idea
A codec decodable by pure array indexing (no entropy streams) could live *resident*
in GPU memory and be dequantized inside the attention kernel — real hot-tier savings.
Question: how close can it get to EC quality at matched honest rate?

## The frontier (best of each family; 5-task F1 @ ~1.5 b/c unless noted)
| family | proxy logit_err @1.5 | F1 | status |
|---|---|---|---|
| EC (approach 2, reference) | 0.0032 | **46.80** | — |
| **RVQ 4×(16-dim) + per-token depth (pgq2)** | 0.0041 | **42.62** | incumbent; −4.16 SIG vs EC |
| plain scalar norm+direction (pgq2 nd) | 0.0171 | 27.6 | far behind |
| page-reset TCQ + warp + sparse-K (pgq3) | 0.0174 | no F1 (screened) | ≈ scalar; trellis bought nothing |
| E8 Voronoi product codes (pgq3) | 0.0400 | no F1 (screened) | worst honest metric* |
| learned linear pair (pgq3, family d) | 0.0298 | no F1 (screened) | worse than frozen basis |
| OSCAR-emulation INT2 + windows (anchor) | — | no F1 (gate) | needs BF16 windows to survive sinks |
(*E8's top1 = 0.75 is proxy pathology — eviction protects sink argmaxes; third sighting.)

- RVQ retrained on 5× tokens: 42.62 → 42.06, a wash → the VQ→EC gap is the entropy
  model over coordinate values, not data/tuning. At 1.0 b/c the gap is ~12 F1.
- Fixed-K sparse significance coding (the fixed-width analogue of EC's deadzone) was
  embedded in TCQ's ladder and did not help — the zero-coding edge needs entropy.
- Mechanism, measured 3 ways (F1, gate physics, proxies):
  **EC > 64-dim VQ > 8-dim lattice ≈ per-coord scalar.**

## Durable design lessons (from three gate sweeps, `fixes_to_apply.md` pgq3-3..5)
1. Transmit the RAW-domain norm and renormalize after the inverse map
   (r̂ = n16·û/‖û@G‖): raw normR = 1 by construction — better than the incumbent's
   code-domain norms (0.963).
2. LS gains are shrinkage estimators — wrong tool wherever softmax scale matters.
3. Gates must be incumbent-relative; absolute sink/norm thresholds are unpassable at
   these rates (plan3 §gates recalibration).

## Artifacts
`kvq/compression/{norm_direction,rvq,tcq,e8,oscar_arm}.py`, bundles + gate reports in
`artifacts/page_quant2/`, screenings in `artifacts/page_quant/phaseA_*.json`.

## What would genuinely reopen this door
Codes with effective vector dimension ≫ 8 that stay table-free at decode: trellis-coded
VQ over long blocks, or learned implicit codebooks evaluated in-kernel. A different
engineering class from everything tried; the bar to beat is recorded (42.62 @1.496).
If only ~2 b/c is needed: the door is already open there (TurboQuant 45.73 @2.125;
OSCAR INT2+windows class) — see approach 6 for the kernel path.
