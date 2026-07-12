# Hand-off: Entropy-Constrained VQ (ECVQ) for the K-cache codebooks

> **RESOLVED (completed):** implemented and F1-tested at λ=0.5. Result: within noise of plain
> k-means (Qwen mean +1.1, Llama −0.8), does not close the VQ→BF16 gap, **not shipped**. See the
> ECVQ line under "Next steps" in `notes/entropy_coding_throughput_report.md`. Codebooks:
> `entropy_coding/vqa_G4_strat_flat_*ecvq05*.pt`. The task spec below is kept for context.


**Baseline to beat:** Milestone 1 (`handoffs/milestone1_reference.md`). The number to
improve is the **VQ row**: Qwen **50.30**, Llama **45.39** (G=4 stratified k-means codebook,
fp8 centroids, +4/32 band on Llama). Read `milestone1_reference.md` first — it has all
artifact paths, the environment, and the leakage rules. Read `notes/entropy_coding_throughput_report.md`
for the method context.

## Goal

Replace the **k-means** codebook training with **ECVQ** (Chou–Lookabaugh–Gray 1989) and
see if it improves downstream F1 at the same 2 b/coord — **without** changing the decoder
(keep the fast fixed-length int32/fp8 gather).

## What ECVQ is (and is NOT)

It is a **codebook-training** method — a rate-aware modification of the Lloyd/k-means loop.
It is **not** "put an entropy coder on the VQ indices at decode time." Same encode
(nearest under a penalty), same decode (index gather) as ordinary VQ.

The Lloyd loop changes in the **assignment step** only:

```
standard k-means:  assign x -> argmin_i ||x - c_i||^2
ECVQ:              assign x -> argmin_i [ ||x - c_i||^2  +  lambda * (-log2 p_i) ]
```

then update centroids as the mean of assigned points, and update `p_i = count_i / N`
(the empirical codeword probabilities). Iterate. `lambda = 0` recovers k-means. Larger
`lambda` biases assignments toward already-popular (cheap) centroids, so the effective
index entropy drops and rarely-used centroids die off.

## Why it's worth trying here (given our findings)

- It's a **drop-in for `_kmeans`** — the decode path, kernel, fp8 packing, and band all
  stay identical, so throughput is unchanged (this is the key advantage over anything that
  entropy-codes the indices).
- Our study repeatedly found **MSE-optimal ≠ F1-optimal**. k-means minimizes distortion;
  ECVQ minimizes a *rate-distortion* objective, which places centroids differently. Whether
  that helps F1 is exactly the open question — it's a principled, cheap alternative to the
  attention-KL fine-tune (which was verified but flat).

## Two regimes to test (report both)

1. **Fixed-K, fixed-length deploy (throughput-preserving, primary).** Keep K=256 per group,
   decode with the existing fixed-length gather. ECVQ here just gives a *different* codebook
   (rate-aware). Test whether the rate-aware placement helps F1 despite (nominally) higher
   distortion. Zero throughput cost.
2. **Larger-K, entropy-realized rate (secondary, optional).** Train with a large K (e.g.
   1024–4096) and a `lambda` that drives the *entropy* to ~2 b/coord; the nominal index is
   longer but the achieved rate is 2 b/coord. This needs variable-length index coding to
   realize (reintroduces a decode cost — measure it). This is the classic ECVQ rate regime;
   only pursue if regime 1 is promising.

## Concrete steps

1. **Implement ECVQ.** Add an ECVQ variant of `_kmeans` in
   `entropy_coding/group_vq_codec.py` (keep the chunked-assignment structure to stay within
   GPU memory — the current `_kmeans` chunks over N; add the `+ lambda * (-log2 p_i)` term to
   the per-block distance before `argmin`, and track `counts` → `p_i` across the epoch). Wire
   a `--ecvq-lambda` flag through `train_group_vq_alloc.py` (0 = current k-means).
2. **Sweep lambda** on a couple of (layer,head) via the fidelity proxy first
   (`entropy_coding/proxy_score.py`: top-1/top-5/relMSE on held-out q/k) — cheap, no F1 runs.
   Pick 2–3 promising lambdas. Also log the achieved index entropy per group to confirm ECVQ
   is doing something (entropy should drop with lambda).
3. **Retrain codebooks** (regime 1, fixed K=256) on the fair pools:
   Qwen `--basis-moments <qwen400> --data-root /vault/samuel/data/query_stats_qwen3_8b_compact8train80 --code-idx 0..79 --G 4 --grouping stratified --allocation flat --ecvq-lambda <L>`;
   same for Llama (`…_llama31_8b_compact8train80`, llama basis).
4. **fp8 + band** to match the milestone: fp8-quantize the ECVQ centroids (same recipe used
   for `*_fp8.pt`), and use the 4/32 band on Llama (`vq_sink=4 vq_recent=32`).
5. **F1** via `pipelines/bench/worker.py` (fresh `output_dir`!), aggregate with
   `entropy_coding/aggregate_f1.py`, compare to the milestone VQ (Qwen 50.30 / Llama 45.39).
   Qwen fraction 0.3, Llama fraction 1.0, musique excluded (`logs/bench_allmethods/exclude_musique_train.json`).
6. **Throughput**: regime 1 is unchanged (same gather) — state that; only re-measure
   (`fused_decode_all.py --G 4`) if you tried regime 2.

## Gates / honesty
- Confirm ECVQ actually changes the codebook (index entropy drops vs k-means at lambda>0).
- Beat the milestone on **F1 mean** on at least one model to call it a win; if it only lowers
  MSE but not F1, that's the (expected, by our findings) negative result — report it plainly.
- Leakage: train on compact8-train only; lcc/2wiki aren't in compact8; musique-train excluded.
- Don't ship a codebook that regresses F1 (we didn't ship the attention-KL ones).

## Deliverable
Extend `notes/entropy_coding_throughput_report.md` (or a sibling) with an ECVQ row vs the
milestone VQ, and a one-paragraph verdict (did rate-aware training beat MSE k-means on F1?).
