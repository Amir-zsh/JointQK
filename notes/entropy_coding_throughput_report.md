# KV-Cache K-Compression: Methods

Report summarizing findings about throughput/accuracy. We also put forward VQ as an alternative to EC.

## Shared setup

All calibrated methods operate per **(layer, kv-head)** at a budget of **2 bits per
coordinate** (8× smaller than BF16), on **Qwen3-8B** / **Llama-3.1-8B** (36×8 and
32×8 kv-heads, head-dim 128).

- **QPCA basis.** Keys are centered and rotated into a per-head *Q-weighted PCA*
  basis. The rotation is fitted on pooled second moments (Σ_Q, Σ_K) captured from a
  400-prompt calibration corpus. The scalar,
  entropy-coding, and VQ methods all quantize *in this basis*. Inverse is folded into RoPE and applied to the queries.
- **V-side / layers.** Values are quantized with TurboQuant @ 2-bit; layer 0 is kept
  in fp16 (its attention-sink statistics are anomalous). This is done to keep the comparison to TurboQuant fair.

## Methods

### BF16 — reference
Uncompressed keys. The full-precision upper bound.

### TurboQuant (2-bit)
Rotates each key by a fixed **random Walsh–Hadamard** matrix (seeded per layer) to
Gaussianize the coordinates, then per-vector scalar-quantizes. The rotation is
data-free (no calibration).

### INT2 / QPCA-Fixed — scalar
A **deadzone-uniform scalar quantizer** on the QPCA basis. Bits are
allocated **per coordinate by reverse water-filling the QPCA energy Λ** — important
coordinates get finer steps — with a widened zero-bin (deadzone) and an extra-widened
top coordinate to absorb heavy tails. Each coordinate is quantized independently.

### OSCAR
Rotates keys by the **query-covariance eigenbasis composed with a Hadamard** (U_Q·H),
clips per-token outliers, then applies a **per-token dynamic Lloyd-Max INT2**:
standardize each token's rotated row by its own mean/std and bucket on fixed
Lloyd-Max boundaries. The first 64 (sink) and last 256 (recent) tokens are retained
in **bf16** as a protection band, so its effective rate exceeds 2 b/coord at short
context.

### VQ — group vector quantization (G=4, stratified)
Quantizes **groups of coordinates jointly** against a learned codebook rather than
each coordinate alone.
- Split the QPCA coordinates into **groups of 4** by a **rank-grouping** (coordinate of rank *r* → group *r* mod NG), folded into the basis so every group spans the full energy spectrum instead of one group hoarding the high-variance coordinates.
- Allocate bits per group (hence codebook size *K* = 2^bits):
  **water-fill by importance** (variable *K*), average held at 2.0 b/coord.
- Learn a **per-group k-means codebook** on the calibration keys. Encoding replaces
  each group of 4 coordinates with its **nearest-centroid index**; decoding is an
  index **gather**. Centroids are stored in **fp8**.
- OSCAR-style **fp16 sink/recent band**: keep the first few + last few prompt tokens
  unquantized (e.g. 4/32). Overhead: ~0.12 extra b/coord.

### Entropy coding — rANS and Exp-Golomb
Quantizes the QPCA coordinates with a **deadzone-uniform step**, then **losslessly
entropy-codes the integer indices**. Two coders over the *same* indices — so they
give the *same* reconstruction — but at **different rates**:
- **rANS** (range Asymmetric Numeral System): arithmetic-style coding against a
  **frozen per-coordinate frequency model**; codes at fractional bits/symbol, so it
  tracks the true index entropy (~1.94 b/coord here).
- **Exp-Golomb**: a parameterized universal integer code with **no per-symbol
  frequency model**. Its code lengths are integers ≥1 bit/symbol, so it cannot go
  below ~1 bit on the many low-entropy (mostly-zero) deadzone coordinates and costs
  **~40% more** (~2.71 b/coord at the same reconstruction). Its only advantage is a
  faster serial decode.
Same reconstruction (identical F1), different rate.

### KIVI — baseline
Keys quantized per-channel and values per-token to INT2 in fixed groups, with no
learned basis.

## Results

### Rate & decode latency

True bits/coord (incl. metadata) and single-request decode+attend latency
(A100-SXM4-40GB, T=65,536, fused decode kernel).

| Method | bits/coord | decode+attend (ms) | vs BF16 |
|---|---|---|---|
| BF16 | 16 | 6.76 | 1.00× |
| INT2 / QPCA-Fixed | 2.00 | 3.95 | 1.71× |
| TurboQuant | 2.125 | ≈3.95 | ~1.7× |
| OSCAR | 3.47 (short ctx) / 2.28 (128K) | 4.50 | 1.50× |
| VQ (G=4, fp8) | 2.00 | 4.74 | 1.43× |
| Entropy coding — rANS | 1.94 | ~3655 (serial) | 0.002× |
| Entropy coding — Exp-Golomb | 2.71 | ~560 (serial) | 0.012× |
| KIVI | 2.00 | — | — |

Latencies measure the decode format (bits/coord, codebook size, gather pattern).

### Downstream F1

LongBench downstream F1. Keys @ 2 b/coord, V = TurboQuant @ 2-bit, layer-0 fp16,
400-prompt calibration. `mean` is over the three tasks. Best config per method
class: EC on the uncentered QPCA basis; VQ = G=4; INT2 =
best per-model deadzone (dz=0.375 on Qwen, dz=0.625 on Llama); OSCAR = Lloyd-Max.
VQ uses a 4/32-token fp16 sink/recent band on **both** models (+~0.12 b/coord); the no-band
Qwen/Llama rows are kept for reference.

### Qwen3-8B (fraction 1.0, full sample)

lcc n=500, 2wikimqa n=200, musique n=150. `mean ± ` = bootstrap 95% CI half-width (10k resamples).
`b/coord` = true rate incl. metadata (see footnote).

| Method | b/coord | lcc | 2wikimqa | musique | mean |
|---|---|---|---|---|---|
| BF16 (reference) | 16 | 66.58 | 48.41 | 33.42 | 49.47 ±3.2 |
| Entropy coding (rANS / Exp-Golomb) | 1.94 | 63.39 | 45.93 | 33.20 | 47.51 ±3.1 |
| VQ (G=4, fp8, +4/32 band) | 2.12 | 64.34 | 45.24 | 30.42 | 46.67 ±3.1 |
| VQ (G=4, fp8, +16/64 band) | 2.27 | 64.34 | 45.55 | 29.99 | 46.63 ±3.1 |
| VQ (G=4, fp8, no band) | 2.00 | 58.49 | 47.38 | 33.52 | 46.46 ±3.1 |
| OSCAR | 3.47 | 61.86 | 44.42 | 27.15 | 44.48 ±3.0 |
| INT2 / QPCA-Fixed | 2.00 | 57.05 | 42.74 | 25.15 | 41.65 ±2.9 |
| KIVI | 2.00 | 56.54 | 39.49 | 26.05 | 40.69 ±3.0 |
| TurboQuant | 2.125 | 52.99 | 38.73 | 27.09 | 39.60 ±2.8 |

`b/coord`: EC is the rANS realization (1.94; Exp-Golomb is 2.71). OSCAR 3.47 at ≤16K ctx
(2.28 at 128K). VQ band overhead (fp16 sink/recent) is shown at ~4K ctx and shrinks ∝1/seqlen.

### Llama-3.1-8B (fraction 1.0, full sample)

lcc n=500, 2wikimqa n=200, musique n=150. `mean ± ` = bootstrap 95% CI half-width.
`b/coord` as in the Qwen table.

| Method | b/coord | lcc | 2wikimqa | musique | mean |
|---|---|---|---|---|---|
| BF16 (reference) | 16 | 53.46 | 50.99 | 32.62 | 45.69 ±3.1 |
| VQ (G=4, fp8, +4/32 band) | 2.12 | 53.51 | 51.17 | 31.49 | 45.39 ±3.0 |
| Entropy coding (rANS / Exp-Golomb) | 1.94 | 53.63 | 47.92 | 30.66 | 44.07 ±3.1 |
| INT2 / QPCA-Fixed | 2.00 | 47.13 | 51.88 | 30.51 | 43.17 ±3.0 |
| TurboQuant | 2.125 | 51.05 | 44.35 | 31.80 | 42.40 ±3.1 |
| OSCAR | 3.47 | 51.55 | 43.20 | 29.00 | 41.25 ±3.0 |
| VQ (G=4 stratified, no band) | 2.00 | 43.41 | 45.29 | 26.38 | 38.36 ±2.9 |

## Next steps

Toward a paper, in rough priority order.

**H100 throughput (required).** All latency/rate numbers are A100-SXM4-40GB. HBM3
(~2× bandwidth) and 50 MB L2 shift the balance between bandwidth-bound (BF16/INT2) and
gather-bound (VQ) methods, and H100 runs **native e4m3 fp8** in-kernel. Re-measure the entire rate & latency table on H100.

**Entropy-constrained VQ (ECVQ) — tested, within noise of k-means; not shipped.** Rate-aware
k-means (Chou–Lookabaugh–Gray, λ=0.5; decoder unchanged), 2 b/coord. Qwen mean +1.1, Llama −0.8;
all differences within the 95% CIs above. Codebooks: `entropy_coding/vqa_G4_strat_flat_*ecvq05*.pt`.

**Broader evaluation.** (a) More models/families and sizes (MHA vs GQA, 32B/70B);
(b) full LongBench (16 tasks) + **RULER at 128 K** — long-context is where KV
compression matters most and where the current ≤16 K prompts under-test it;
(c) rate–distortion curves at 1/2/3/4 b/coord, not just 2; (d) multi-seed / bootstrap
CIs (musique is noisy at these fractions).

**Scope extensions.** (a) **V-side compression** — V is TurboQuant@2b fixed here and
left fp16 in the timing. (b) **Decode-phase (Mode B)**
compression, not just prefill (Mode A). (c) End-to-end **serving throughput**
(batched, tokens/s) beyond the single-request decode microbenchmark, plus the offline
**calibration cost**.