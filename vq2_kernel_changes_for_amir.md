# vq2 decode: kernel changes + how to reproduce the batched-throughput figure

2026-07-27, Samuel. Everything below is in the vendored fork
`oscar_vq2/vendor/OSCAR-vq/sglang-research/python/sglang/srt/`. Ready-to-apply diffs are in
`oscar_vq2/notes/amir_patch/*.patch` (against the pre-change files, which are kept as `*.pre_bscap`).

Target figure: `JointQK/entropy_coding/figures/decode_throughput_bs.png` -- decode throughput
(tokens/s, aggregate over the batch), one panel per context, one cluster per batch size, three
methods.

**All three changes are behind env flags that default OFF.** With the flags unset the engine is
byte-identical to before (verified: vq2 bs=4 measured 25.082 with flags off vs 25.067/25.070 on the
unmodified build, three independent server loads within 0.06%). So applying the patches cannot
change any existing result until you opt in.

---

## 1. What changed

### `SGLANG_VQ2_BS_ADAPTIVE_SPLITS` -- batch-aware split cap (`triton_backend.py`)

The quant-tier split cap (`--triton-attention-num-kv-splits`, 48 for vq2) was tuned at bs=1, where
the gather kernel is latency-bound and wants many programs. At bs>1 the batch itself supplies that
parallelism and 48 is past the optimum. New `quant_split_cap(bs)` returns `clamp(cap // bs)`, i.e. it
holds the launched block count at what the bs=1 sweep tuned, and is wired into all FOUR quant-tier
sites (three `get_num_kv_splits` fills -- eager, cuda-graph capture, cuda-graph replay -- plus the
stage-1 grid bound).

Measured at bs=4 / ctx=24k, splits/seq vs ITL: 8 -> 26.35 ms, **12 -> 24.11**, 16 -> 24.51,
24 -> 24.53, 29 (shipped) -> 25.07, 48 -> 25.10. At bs=8 the optimum is 6, again `cap // bs`.

**Trap worth knowing if you touch this:** the stage-2 partials arena is SHARED between the HP and
quant tiers -- HP owns the head, quant owns the tail, and the callee asserts
`attn_logits.shape[2] == hp_max + quant_max`. Reducing the quant cap therefore requires slicing the
buffer (`[:bs, :, :n_split]`) as well. Miss that and stage-2 reduces over uninitialised tail
entries: silently wrong output at bs>=2 only. The assert caught it at cuda-graph capture.

### `SGLANG_VQ2_FUSED_MAP` -- per-head map as one Triton kernel (`vq_codebook.py`)

`vq_map_k` / `vq_map_q` are per-head maps: 8 separate `[T,128] x [128,128]` GEMMs. torch runs them as
a batch-8 bmm with M=4 rows -- 524K MACs in ~11-14 us, i.e. ~1 GFLOP/s. It is not arithmetic- or
bandwidth-bound; a batch-8 GEMM with M=4 simply occupies a handful of blocks on 108 SMs. The new
kernel splits the OUTPUT COLUMNS as well as the heads (grid `(H, D/32)` -> 32 blocks) and folds in
the mean subtraction and the output cast.

Bit-identical to torch (`rel = 0.000e+00`), 3.56x on `vq_map_k` and 4.06x on `vq_map_q`, ~0.58 ms/step
saved at bs=4. One kernel serves both call sites: `GRP` folds the GQA group into the row axis, so
`GRP=1` + mean is the K map and `GRP=group_size` without mean is the query map. Shapes outside the
decode range (T*GRP > 32, i.e. prefill) return `None` and fall back to torch.

### `SGLANG_VQ2_FUSED_ENCODE` -- flush encode without the score tensor (`vq_codebook.py`, `unified_kv_pool.py`)

The flush encode materialised `sco = [L, n, H, NG, K]` fp32 -- **36 MB to encode four tokens** on
Qwen3-8B -- then rewrote it (`-= cb_sq`) and read it a third time to argmax. The new kernel fuses
per-token RMS norm + nearest-centroid search, never writing a score to memory. Codes are bit-exact
(0 mismatches out of 589,824 at n=64).

**The subtlety that matters most here:** `n = bs * flush_interval` (see
`QuantKernel/gpu_flush_int2.py:47`, `src_hp_slot: int64 [bs * flush_interval]`), so the token count
GROWS with batch -- 8 at bs=1, 64 at bs=8. My first version declined anything above n=8, i.e. it was
inactive at *every* batch size above one, which is exactly where it was needed. Tiling the token axis
is what moved bs=8 from +27.3% to +13.5% vs int2.

`BLOCK_N` is batch-adaptive (8 at n<=8, else 16): measured 178/553/1067 us at n=8/32/64 for
BLOCK_N=8 vs 291/542/931 for 16. Overridable via `SGLANG_VQ2_ENCODE_BLOCK_N`.

### Not changed

`decode_attention.py` is untouched (the `.pre_bscap` diff is empty). The stage-1 attention kernel
itself is exactly as you shipped it -- see "what we could not fix" below.

---

## 2. Applying

```bash
cd <fork>/sglang-research/python/sglang/srt
patch -p0 layers/attention/triton_backend.py < .../amir_patch/triton_backend.py.patch
patch -p0 mem_cache/vq_codebook.py           < .../amir_patch/vq_codebook.py.patch
patch -p0 mem_cache/unified_kv_pool.py       < .../amir_patch/unified_kv_pool.py.patch
```

Correctness gate (no server needed, ~1 min), run with the flags ON:

```bash
SGLANG_VQ2_FUSED_MAP=1 SGLANG_VQ2_FUSED_ENCODE=1 python logs/gate_landed_fixes.py
```

It compares the fused paths against the torch paths through the real engine signatures and must
print `GATE: PASS` -- exact at T = 1/2/4/8 for both maps, 0 code mismatches at n = 8/16/32/64, and
prefill-sized input correctly declined.

Rollback: `bash logs/revert_bscap.sh` (restores from `*.pre_bscap`, md5-verified, refuses if a
backup drifted). Or just unset the flags.

---

## 3. Reproducing the figure

Serve with, per arm:

```
vq2 :  SGLANG_VQ2_BS_ADAPTIVE_SPLITS=1 SGLANG_VQ2_FUSED_MAP=1 SGLANG_VQ2_FUSED_ENCODE=1
int2:  (nothing -- OSCAR runs in its released configuration, deliberately; see note below)
bf16:  (nothing)
```

Scripts (all in `oscar_vq2/logs/`):

| script | what it produces |
|---|---|
| `decode_matrix.sh <gpu> <shard>` | the cells. shard 0 = bs 1,2 @24k; 1 = bs=4 @24k + 60k pair; 2 = bs=8 @24k + 100k pair; 3/4 = long-context re-runs; 5/6 = post-fix re-measures |
| `decode_stream_bench_bs.py` | the client: B concurrent streams, steady-state window |
| `plot_decode_throughput.py` | the figure (plain = batch figure; `ALIGNED=1` = the OSCAR-protocol bs=1 figure) |
| `agg_decode_bs.py [--matrix]` | the numeric tables, guards enforced |

Each shard starts with the same anchor cell (int2, bs=4, ctx=24k) so per-card offset is measured,
not assumed -- eight anchors across three A100s agreed to 0.09%, which is what makes it safe to
shard across GPUs. Same-card repeats reproduce to 0.04-0.06%.

**Measurement guards -- a cell is only valid if all hold.** These are not paranoia; every one of them
caught a wrong number today:

* Throughput is measured only over `[max_i first_token_i, min_i last_token_i]`, the window in which
  ALL streams are decoding. A naive window put other streams' prefills inside it and reported bf16
  at bs=4 as 31.7 tok/s when the true figure was 168.
* `#running-req == bs`, read from the SERVER log. The client cannot tell a sustained batch from a
  serialized one.
* `>= 2` reps and window `>= 5 s`.
* `ngen` scales with context (`ctx/25`, min 1200) AND is inside the served `context_length` and the
  KV budget. Otherwise the server rejects every request as over-length and the failure looks like a
  batching failure.

## 4. Gotchas that cost me hours

1. **EOS.** Random-token prompts can emit EOS on the first step; that stream contributes one token,
   the all-streams window collapses to ~0.03 s, and the cell looks like a batching failure. Seeds are
   fixed, so it reproduces per cell and reads as method-specific. Pass `"ignore_eos": true`.
2. **sglang's HTTP frontend holds no GPU memory**, so `nvidia-smi`-based cleanup never sees it. It
   survives, keeps the port, and answers `/get_model_info` 200 for an already-dead scheduler -- one
   cell reported "served" 60 s after the previous one. Kill by port too, and gate readiness on
   `"The server is fired up and ready to roll"` appearing in the freshly-truncated log.
3. **`free_gpu` with an empty uuid.** `nvidia-smi ... | grep "$U"` with `U=""` matches every line and
   kills your sglang processes on ALL cards. This box does throw transient
   `NVRM: GPU ... Device is currently unavailable`, and it killed a job on another GPU mid-run. Bail
   if the uuid read comes back empty.
4. **Do not abort a server wait on "CUDA error".** sglang's benign startup warning reads
   "...may lead to incorrect model outputs or CUDA errors", so that substring aborts healthy servers.
   Match `Traceback|CUDA out of memory|torch.OutOfMemoryError|Received sigquit|AssertionError`.
5. **`MEM_FRAC` is ignored** -- `serve_oscar.sh:18` assigns `MEM_FRAC=0.78` unconditionally instead of
   defaulting. Everything measured today is at 0.78.
6. **The quantized path gets ZERO radix-cache hits** (`#cached-token: 0` vs 399,996 for BF16 on the
   identical cell). So prefix-warm-up protocols are not reproducible for int2/vq2 on this stack, and
   any radix-on comparison flatters BF16. Benchmark with sharing OFF for all arms.

---

## 5. Results these produce (A100-SXM4-40GB, Qwen3-8B, ctx=24k unless noted)

| bs | bf16 | int2 | vq2 (all fixes) | vq2 vs int2 ITL |
|----|------|------|-----------------|-----------------|
| 1  | 50.7 | 58.0 | **59.9** | -3.8% |
| 2  | 97.4 | 108.5| **108.7**| -0.6% |
| 4  | 168.2| 185.8| 181.5 | +2.1% |
| 8  | does not fit | 310.2 | 272.8 | +13.5% |

Long context, bs=4: 60k 122.3 vs 122.3 tok/s (parity); 100k 87.2 vs 88.6 (+1.5%).
BS=1 across contexts (OSCAR-aligned protocol, 1024 output tokens): vq2 is 1.25x / 1.52x / 1.75x over
BF16 at 30/60/100k, against OSCAR's 1.18x / 1.34x / 1.48x.

## 6. What we could NOT fix, and the one idea left

The residual is a flat ~24% premium in the vq2 stage-1 attention kernel, present at every batch size
once splits are equalised. ncu (bs=8): both kernels are register-bound at 254/250 regs -> 2 blocks/SM
-> 12.5% occupancy; int2 tolerates it (`long_scoreboard` 0.04 cycles/issue) while vq2 does not
(**0.94**), with the LSU issue port idle (`lg_throttle` 0.04) -- unhidden gather latency, not
saturated throughput. vq2's L1/TEX pipe runs at 62.6% vs int2's 31.9% while moving LESS DRAM
traffic: ~8x the load instructions for the same bytes.

Measured and closed: splits (u-shaped, optimum `cap//bs`), BLOCK_N {64,128,256}, warps {2,4,8},
stages {2,3,4}, BLOCK_H {4,8}, cache-eviction hints, codebook footprint (flat 32 KB -> 0.1 KB, so
latency not capacity), wider-index product LUT (0.43x -- L1 residency is why the current kernel is
fast), Triton software prefetch (0.86x -- the staged tile spills past the register wall), a CUDA
shared-memory codebook LUT (0.53x), occupancy 1..16 blocks/SM (flat), and CUDA `cp.async` prefetch of
the index chain (1.02x / 0.94x).

**The one untested idea, and the reason your CUDA kernel may be the answer:** the warp
lane -> (token, group) mapping. Each warp issues 32 codeword gathers; a group's sub-table is
1 KB = 8 cache lines. If a warp's 32 lanes read *32 tokens at one group* they touch ~8 lines; if they
read *one token across 32 groups* they touch ~32. That is a 4x L1-traffic difference on the busiest
pipe, and Triton chooses the mapping from the `[BLOCK_N, NG]` tile shape without exposing it. An
isolated CUDA gather using the good mapping sustains 402 G gathers/s against production's 127 G/s.

So the question about your shared-memory kernel: **do a warp's 32 lanes read 32 different tokens at
the same group, or one token across 32 groups?** If the former, that alone may be where your gains
come from -- and it would explain why every Triton-side lever above came up empty.

Standalone harness for A/B-ing your kernel against ours, no serving stack needed:
`logs/ncu_stage1_driver.py <vq2|int2> <bs> [splits]` -- the production stage-1 kernel isolated at
serving-accurate shapes and split counts, timed, and ncu-ready.
