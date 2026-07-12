# Handoff: can a properly-allocated VQ beat scalar INT2 on accuracy?

**Goal for you (the next engineer):** the current group-VQ is handicapped vs the
scalar baseline and loses to it at every decode-viable group size. Find out
whether a *properly designed* VQ — with cross-coord bit allocation and/or better
grouping — actually beats scalar INT2 at 2 bits/coord, or whether VQ is
genuinely dominated. This is an **accuracy** task; the VQ *decode kernel* is
already done and fast (see §5), so your job is the codebook/allocation side.

---

## 1. The finding that motivates this (why the current VQ is unfair to itself)

At 2 bits/coord, same centered QPCA basis, held-out eval (calib `[0,5,6]` =
8234 tok; eval `[4,8,16,20]` = 14420 tok; layer-0 excluded), top-1/top-5 of
Q·K argmax vs full precision:

| method | top-1 | top-5 | k_mse |
|---|---|---|---|
| VQ G=2 | 0.192 | 0.386 | 0.294 |
| VQ G=3 | 0.394 | 0.772 | 0.279 |
| VQ G=4 | 0.479 | 0.863 | 0.274 |
| VQ G=6 | 0.684 | 0.893 | 0.269 |
| VQ G=7 | 0.695 | 0.881 | 0.323 |
| **scalar INT2 (QPCA-Fixed)** | **0.647** | **0.910** | 0.350 |
| rANS (entropy coding) | 0.752 | 0.954 | 0.341 |

Two red flags a user caught:
1. **Non-monotonic / worse than scalar.** VQ with group size G is a superset of
   scalar quantization of those G coords, so VQ should be ≥ scalar and monotone
   in G. Instead small-G VQ is *far* below scalar (G=2 = 0.19 vs 0.65) and only
   catches up around G=6.
2. **k_mse is nearly flat across G (0.29→0.27) but top-1 swings 0.19→0.68.**
   Same total reconstruction error, very different attention accuracy → it's
   about *where* the bits go (allocation), not how many.

**Root cause (confirmed in code).** The scalar baseline is *not* plain uniform
quantization:
- `run_pca_ec_deadzone.build_qpca_fixed_deadzone` does **per-coord waterfill bit
  allocation** — `allocate_bits(qpca_cen["score"], b=2, max_coord_bits=8)` gives
  each coord **0–8 bits** (2 avg), concentrating bits on high-variance QPCA
  directions — plus **per-coord std scaling** (`std_per_coord`) and a coord-0
  widen hack.
- Group-VQ (`group_vq_codec.py`) uses a **flat 2 bits/coord per group**
  (`K = 4^G`, no cross-group allocation), **k-means on raw coefficients**, and
  **consecutive** coord grouping.

So "VQ G=1 = scalar" is false as implemented: VQ G=1 is uniform-2-bit-per-coord
(no waterfill), strictly weaker than the allocated scalar. VQ is losing on
**allocation**, not on the vector-quantization idea. Its reported numbers are a
lower bound on a handicapped VQ.

---

## 2. Your task

Determine whether a properly-built VQ at **2.0 bits/coord** beats scalar INT2
(**0.647 / 0.910**) on top-1/top-5 on the same eval set. Approaches, roughly in
priority order:

1. **Cross-group bit allocation.** Give each group a bit budget by waterfilling
   the QPCA `score` (= `k_diag`, the Λ spectrum) over groups instead of a flat
   `2*G` bits/group — i.e. variable `K` per group, high-importance groups get
   more centroids, tail groups fewer (even K=1 → drop). Keep the *average* at
   2.0 bits/coord. This directly imports scalar's advantage into VQ.
2. **Stratified / interleaved grouping.** The current grouping is consecutive
   (`group_boundaries`), so each group is a narrow variance band. Try grouping
   coords *across* the Λ spectrum (e.g. coord i → group `i % NG`) so every group
   has balanced high+low-variance content and the k-means has to resolve all of
   them. Compare vs consecutive.
3. **Per-coord whitening within a group** before k-means (divide each coord by
   its std, unquantize by multiplying back) so the highest-variance coord in a
   group doesn't dominate the centroids and starve the others.
4. **Combine** the best of the above; sweep G ∈ {2,3,4,6}.

**Guard rails:** stay at a true 2.0 bits/coord average (report the real
bits/coord); watch samples-per-centroid (K=4^G at G=6 is only ~2 samples/centroid
on 8234 tokens — allocation that inflates K somewhere will undertrain it); keep
layer-0 excluded.

---

## 3. Environment (CLAUDE.md is stale on this)

```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate kv   # NOT .venv (doesn't exist)
cd <repo>/entropy_coding
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8   # REQUIRED: 255-core box, torch.linalg.eigh thread-thrashes/hangs otherwise
```
- **GPUs 0–3 only** (never 4–7; enforced). Heavily shared — often all busy with
  other users. Check `nvidia-smi`; pick the least-loaded of 0–3.
- Python 3.12, torch 2.11+cu128. A100-SXM4-40GB.

---

## 4. Files you need

- **`group_vq_codec.py`** — `GroupVQCompressor`, `train_group_vq_compressors`,
  `group_boundaries`, `_kmeans`. **This is where you add allocation / stratified
  grouping / whitening.** `roundtrip(k)` is the eval interface.
- **`train_group_vq.py`** — trainer CLI (`--calib-idx 0 5 6 --G <G> --iters 25
  --out <path>`). Trains one `GroupVQCompressor` per (layer, kv_head) and saves
  codebooks + the QPCA basis it used.
- **`vq_sweep_score.py`** — the fast accuracy scorer (top-1/top-5/k_mse vs
  full-precision Q·K argmax, same metric as the full harness). Run it on your new
  codebooks: `python vq_sweep_score.py --codebooks <paths...>`. **Use this to
  iterate.** (~1-2 min/codebook.)
- **`run_pca_ec_deadzone.py`** — `build_qpca_basis(sq, kc)` (the centered QPCA
  basis VQ sits on), `build_qpca_fixed_deadzone` (the scalar baseline — read its
  waterfill+std logic and mirror it), `allocate_bits(score, b, max_coord_bits)`
  (the waterfill you can reuse for per-group allocation), `calib_moments`,
  `_codes_for_idx`.
- **`test_codec_on_data.py`** — full harness (all 7 methods) for the *final*
  apples-to-apples comparison once you have a winner: `python
  test_codec_on_data.py --calib-idx 0 5 6 --eval-idx 4 8 16 20 --bits 2
  --vq-codebook <your.pt>` (~15 min; paged encode is slow, that's fine).
- Existing codebooks: `group_vq_b2_calib056_G{2,3,4,7}.pt`,
  `group_vq_b2_calib056.pt` (G=6) — baselines to beat / diff against.
- **`notes/entropy_coding_throughput_report.md`** — the report your result feeds.
- **`logs/entropy_coding_faithful_report.log`** — full running log (read the tail
  for the accuracy numbers and the whole VQ decode-kernel story).

---

## 5. The decode side is DONE — respect the L1 constraint

Don't re-derive the throughput story. A fused tensor-core decode+attend kernel
(`fused_decode_all.py`) already measures every method. Key result: VQ **G=4** was
made to decode in **5.70 ms — faster than BF16** — via an **int64 wide-gather**
trick (`_vq` `VEC` path), because G=4's 64 KB codebook is **L1-resident**. G=6's
1 MB codebook overflows L1 and is stuck at ~27 ms (7× INT2), and no kernel trick
fixes that (characterized in the log).

**Implication for you:** the decode-viable, accuracy-*and*-speed sweet spot is a
**G=4-class, L1-resident, fixed-K codebook**. If your allocation scheme inflates
K or makes it variable-per-group, it breaks the int64 decode kernel. So the
*most valuable* outcome is **a G=4 (or smaller-footprint) VQ that clears INT2's
0.647/0.910** — that would be both faster than BF16 *and* more accurate than
scalar. A G=6 win is interesting for the accuracy frontier but is decode-dead.
Flag the accuracy-vs-footprint tension explicitly in your result.

---

## 6. Success criterion

A VQ at a true 2.0 bits/coord that **beats scalar INT2 (0.647 top-1 / 0.910
top-5)** on the eval set, ideally at **G ≤ 4** (L1-resident → fast decode). If it
clears INT2, VQ is not dominated and the report's verdict flips. If even a
properly-allocated VQ can't beat scalar at G ≤ 4, that's a real, publishable
negative result — report it with the allocation/grouping you tried.

Update `logs/entropy_coding_faithful_report.log` and, if VQ wins, revise the VQ
rows + verdict in `notes/entropy_coding_throughput_report.md`.
