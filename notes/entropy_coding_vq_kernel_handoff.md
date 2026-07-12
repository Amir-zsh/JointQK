# Handoff: well-optimized VQ decode+attention kernel

> **RESOLVED 2026-07-09.** `_vq` in `fused_decode_all.py` rewritten (group-major
> coalesced indices + reconstruct-and-dot with tensor-core `tl.dot`, replacing the
> scattered per-query LUT). Correctness gate PASS (VQ err 8.4e-3). VQ read+qk/full
> went 448/439ms → **32.2/36.0ms** at T=65536 (14×/12×, and now correct + tensor-core).
> It did **not** reach the hoped 1.5–3× INT2 — it lands ~9× INT2 — and a codebook-size
> sweep (`vq_tune.py`) proves **why**: cost is set by codebook *residency*, not bytes.
> K=256 (64KB, L1) = 5.5ms; K=1024 = 19ms; K=4096 (1MB, the accuracy config) = 27ms,
> all gathering the same 4.8GB. The accuracy-config 1MB/head codebook overflows the
> 192KB L1 so its *random* gathers run at L2 speed (INT2's read is *sequential*). Fusing
> softmax forces all NG groups (full 1MB) as the per-key working set; the fast
> L1-resident regime needs grid-over-(head,group), which can't fuse without
> materializing K to HBM (which defeats VQ's read-time compression). So VQ is a real
> architectural loss on fused decode throughput, not a kernel-quality artifact. See
> `logs/entropy_coding_faithful_report.log` (2026-07-09 entry) for the full write-up.
> The sections below are the original (pre-resolution) task spec, kept for context.


**Goal for you (the next engineer):** write a genuinely well-optimized Triton
kernel for **VQ decode + attention** (group vector-quantized K-cache), so its
real fused decode cost can be measured head-to-head against BF16 / INT2 / OSCAR.
The current VQ kernel is correct but ~100× too slow (a bad gather), which makes
the throughput comparison meaningless for VQ. Everything else is done; VQ is the
one loose end.

This is a self-contained kernel task. You do **not** need to touch the accuracy
pipeline or the rest of the study — but read §6 for the broader context so your
numbers land in the right place.

---

## 1. Environment (important — CLAUDE.md is stale on this)

- **Use the `kv` conda env**, NOT the `.venv` CLAUDE.md mentions (it doesn't exist):
  ```bash
  source /opt/miniconda3/etc/profile.d/conda.sh && conda activate kv
  ```
  Python 3.12, torch 2.11+cu128, Triton. A100-SXM4-40GB.
- **GPUs 0–3 are the user's allocation; do NOT use 4–7.** GPU 0 is usually free.
  Check `nvidia-smi`. The machine is heavily shared (load avg has spiked to 300+),
  so compiles/runs can be slow through no fault of the code.
- Run everything from `entropy_coding/` with `CUDA_VISIBLE_DEVICES=0`.
- **Kernels take minutes to compile** (Triton autotune + several depth variants).
  Launch with `nohup ... > log 2>&1 &` and watch the log; do **not** pipe through
  `tail -N` (it buffers until EOF and hides progress). If a run hangs >5 min it's
  usually a real bug (see §4 gotchas), not slow compile.

---

## 2. What exists and works

`entropy_coding/fused_decode_all.py` — the real fused decode+attend harness, all
4 bandwidth-bound methods, 3 depths, split-K, tensor-core `tl.dot`, correctness-
gated vs a torch SDPA reference. **BF16 / INT2 / OSCAR kernels are good; the VQ
kernel is the problem.**

Key structure:
- `grid = (n_heads, n_splits)` split-K. Each program does one head + a key CHUNK.
- `BLOCK_M = 16` (GQA queries sharing each K read), `BLOCK_T = 32`, `d = 128`.
  (`BLOCK_T=64` OOMs shared memory on the VQ kernel — see §4.)
- Three depths via `DEPTH` constexpr: `0` read-only (checksum), `1` read+qk
  (scores), `2` full (softmax + `tl.dot(p, V)`). Depths 0/1 accumulate a checksum
  into `sink`; depth 2 writes per-(head,split) softmax partials `(m, l, acc)`
  combined in torch by `combine()`.
- **The inverse rotation is fused into the query** (`q_rot` precomputed once
  outside), so keys stay in coded space: `scores = q_rot · k_code`. No per-key
  rotation, no full-precision HBM write-back. Keep this invariant.
- V is left uncompressed (fp16) for all methods — the V read + softmax are shared,
  method-independent costs.

Validated: `python fused_decode_all.py --Ts 16384 65536` prints correctness
(`INT2 err≈9e-3, OSCAR err≈2e-2, VQ err≈6e-5`, all PASS) then the timing table.

The LUT-VQ *method* is validated correct in isolation too:
`/tmp/claude-1015/.../scratchpad/vq_lut_test.py` (may not persist — the logic is
inlined in `fused_decode_all.py`'s `_vq` kernel and VQ correctness block).

---

## 3. The problem: VQ is ~100× too slow

Current results (ms, full model, 288 heads × 16 q):

| method | read | read+qk | full | vs BF16(full) |
|---|---|---|---|---|
| BF16 | 3.25 | 3.28 | 3.52 | 1.0× |
| INT2 | 0.49 | 2.29 | 3.96 | 0.89× |
| OSCAR | 0.55 | 3.09 | 4.50 | 0.78× |
| **VQ** | **3.14** | **448** | **439** | **0.01×** |
(T=65536; T=16384 shows the same VQ pathology: read+qk=112ms.)

**The read floor is the robust, hardware-validated result** (INT2/OSCAR ~7× fewer
bytes → ~7× faster read at ~90% peak HBM; OSCAR ≈ INT2). VQ's 448ms is a bad
kernel, not VQ's real cost. For reference, an *earlier* crude measurement
(`vq_throughput_fused.py`, `vq_fused.cu` — real codebook gather but instruction-
bound reads) found VQ only ~1.5× slower than INT2 at long context once the
codebook staging amortizes. So a well-optimized VQ should land **within a few ×
of INT2**, not 100×.

### Why the current `_vq` kernel is slow
Look at `_vq` in `fused_decode_all.py`:
1. **Index read is uncoalesced.** It loops `for g in static_range(NG)` and does
   `tl.load(idx_ptr + h*(T*NG) + o*NG + g, ...)` — a stride-`NG` (=21) gather per
   group. Even depth-0 read is 3.14ms (vs INT2's 0.49) because of this.
2. **The LUT gather is random + uncoalesced.** Depth 1 does, per group,
   `tl.load(lh + qm[:,None]*(NG*K) + g*K + ig[None,:])` where `ig` is the
   per-key codebook index (random in `[0,K=4096)`). That's a `(16,32)` fully
   random L2/global gather, ×NG=21 groups × many tiles × 288 heads. This
   dominates (448ms).

### The LUT method (what `_vq` implements)
`LUT[m,g,e] = q_g · cb[g,e]` is precomputed once per query (T-independent,
amortized — currently a torch `einsum`, not timed as part of per-token cost).
Then each key's score is `score[m,t] = Σ_g LUT[m, g, idx[t,g]]` — NG table
lookups, **no per-key key reconstruction, no full-D dot**. The method is sound;
the *gather implementation* is what's slow. `LUT` shape `(H, BLOCK_M, NG, K)`.
VQ config: `NG=21` groups, `G=6` coords/group, `K=4096` (2 bits/coord). Coded
key dim = `NG*G = 126` (last 2 of 128 coords dropped — matches this repo's
`vq_throughput_fused.py` `d_eff` convention; fine).

---

## 4. Directions to try (kernel engineering)

The core tension: to compute `q·k` for VQ you need either the full reconstructed
key (needs all NG group codebooks = ~1MB/head, doesn't fit 167KB SMEM) or the
LUT (`(BM,NG,K)` = 256KB/head, also doesn't fit SMEM). Either path hits L2 with
some random access. The job is to make that access as coalesced/staged as
possible. Ideas, roughly in priority order:

1. **Coalesce the index read.** Store indices `(T, NG)` row-major and load a
   `(BLOCK_T, NG)` tile in ONE coalesced load, then index per group from
   registers (reshape / `tl.split` tricks), instead of NG strided loads. This
   alone should fix depth-0 read (target: ≈ INT2's read, since NG int16 ≈
   0.33 B/coord vs INT2's 0.25). Consider packing indices to 12-bit (0.25 B/coord).
2. **Codebook-gather path instead of LUT, with SMEM-staged per-group codebook.**
   Each group's codebook `cb[g]` is `K×G = 4096×6` = 48KB fp16 → fits SMEM. Stage
   it once per (head, group), reused across all T keys (amortized). Then
   accumulate the qk **incrementally per group**: `score += q_g · gather_g` where
   `gather_g = cb[g, idx[:,g], :]` is `(BLOCK_T, G)`. Avoids materializing the
   full key. Caveat: `G=6 < 16` so `tl.dot` needs padding to 16, or do the
   `q_g·gather_g` as an explicit `(BLOCK_T,) = sum over G` reduction (G=6 is tiny,
   a manual FMA loop may beat a padded tl.dot). This is likely the most promising
   route — it turns the random K-gather into an SMEM lookup.
3. **LUT staging per group.** `LUT[h,:,g,:]` is `(BM=16, K=4096)` = 256KB, too
   big. But for a single query `(K,)` = 16KB fits. Restructure so a warp/query
   handles the gather from an SMEM-resident per-(query,group) LUT slice.
4. **Tune `num_warps` / `num_stages` / `BLOCK_T`.** The current kernel wobbles
   with block size (a symptom of poor tuning). A well-tuned kernel should be
   stable. Note `BLOCK_T=64` OOMs SMEM for VQ at the current structure.
5. **Reference literature.** Vector-quantized KV-cache / VQ-attention decode
   kernels exist (e.g. GPTVQ-style, VPTQ, "coupled" VQ KV papers). The
   SMEM-staged-codebook + incremental-dot pattern (idea 2) is the standard trick.

**Success criterion:** VQ `read+qk` and `full` become trustworthy and stable
(not tuning-wobbly), landing within a small multiple of INT2 (expect VQ a bit
slower than INT2 due to the gather, ~1.5–3× INT2, NOT 100×). Read floor should
approach INT2's.

### Gotchas already hit (don't re-discover)
- **int64 pointer offsets are required.** `h*T*d` overflows int32 at T≥65536
  (2.4e9 > 2^31) → illegal memory access. Cast `program_id`, `arange`, strides
  to `tl.int64` for base-pointer math. See how `_bf16`/`_int2` do it.
- **`tl.arange` bounds must be powers of 2.** Coded key dim 126 is not — that's
  why the kernel's `arange` is only over the V/output dim `D=128`, and the score
  comes purely from NG group lookups. Keep that separation.
- **`x[:, 0]` slicing of a 2D load is unsupported** — load 1D directly.
- **Setup OOM:** at T=100000, H=288, building fp32 `randn(H,T,d)` then `.half()`
  peaks ~30GB. Allocate fp16 directly (`torch.randn(..., dtype=torch.float16)`)
  and free intermediates. Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **BLOCK_T=64 OOMs SMEM** for the VQ kernel (NG=21 unrolled live tensors).
  Currently BLOCK_T=32. A better-structured kernel may allow 64.

---

## 5. How to validate + measure

```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate kv
cd <repo>/entropy_coding
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup python3 -u fused_decode_all.py --Ts 16384 65536 > logs/vq_kernel.log 2>&1 &
# watch logs/vq_kernel.log
```
The correctness block (small T, Hc=8) must print `VQ err < 5e-2` before you trust
any timing. The reference: reconstruct `krec` from the codebook, then
`softmax(q·krec)·V`. Keep that gate; if you change the VQ storage/kernel, make the
torch reference match exactly.

Compare your optimized VQ's `read / read+qk / full` against BF16/INT2/OSCAR in the
same table. Report at T ∈ {16384, 65536} minimum; add 100000 if memory allows
(fix the setup OOM first).

---

## 6. Broader context (so your numbers land right)

This kernel is one piece of a larger "faithful report" study rebuilding
`notes/entropy_coding_throughput_report.md` to be an apples-to-apples
accuracy+throughput comparison of 7 K-compression methods (BF16, TurboQuant,
INT2, VQ, rANS, Exp-Golomb, OSCAR), audited against OSCAR (arXiv:2605.17757).

- **Full running log & findings:** `logs/entropy_coding_faithful_report.log`
  (read the latest checkpoints — has every result and correction).
- **Plan:** `/home/samuel/.claude/plans/proud-imagining-hippo.md`.
- **Prior throughput mistakes to NOT repeat** (both caught by the user):
  1. `decode_timing_fullmodel.py` measured the UN-fused path (dense fp64 inverse-
     rotation matmul + full-precision HBM write-back) → compute-bound, BF16
     trivially "won." WRONG. It's marked SUPERSEDED in its docstring.
  2. A hand-rolled elementwise `sum(k*q)` qk (instead of tensor-core `tl.dot`)
     masked the bandwidth win. Fixed by `tl.dot`. Your VQ kernel must likewise
     use tensor cores / proper memory access, not naive loops.
- **The read floor is the defensible bandwidth result** (INT2/OSCAR/VQ read ~7×
  fewer bytes → ~7× faster read at ~90% peak HBM, `bw_clean.py` / `bw_fullmodel.py`).
  Decode attention is memory-bound (arithmetic intensity ~0.5 madd/byte), so a
  production fused kernel keeps qk/softmax hidden under the read and preserves
  this win. Your job is to show VQ can too, with a real kernel.
- **Accuracy side (already done, for context):** real trained group-VQ codebook
  (`group_vq_b2_calib056.pt`, G=6) gives top-1/top-5 ≈ 0.68/0.89 — does NOT match
  rANS/Exp-Golomb (0.75/0.96) as the old report wrongly claimed. G-sweep concluded
  G=6 is the best VQ operating point. (Not your concern, but explains why G=6.)

Files you care about: `fused_decode_all.py` (main), `group_vq_codec.py`
(GroupVQCompressor + real codebook structure), `vq_fused.cu` /
`vq_throughput_fused.py` (the older crude gather kernel — a reference point that
got VQ ~1.5× INT2), `bw_fullmodel.py` (read-floor bandwidth numbers).
