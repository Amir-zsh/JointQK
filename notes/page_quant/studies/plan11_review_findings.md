# plan11 mid-run implementation review — findings to revisit

Reviewed 2026-07-17 (while C1 waves were running): the vq2 engine edits in
`vendor/OSCAR-vq` (branch `vq2-longhorizon`), the eval plumbing
(exporter/client/scorers/aggregator), and the three-arm symmetry. Nothing
here invalidates the running comparison; items are reporting obligations,
caveats, and hardening TODOs.

## Verified correct (checked, not assumed)

- **GQA q-head mapping**: `vq_map_q` assumes q heads contiguous per kv head
  (`view(T, 8, 4, D)`); the engine's own grouped HP kernel makes the same
  assumption, and NIAH-served 93.7 with Mode-A-matching subtask pattern
  confirms empirically.
- **Mean-shift softmax invariance** holds through sm_scale and split-K LSE
  combine; requires `logit_cap == 0` (true for Qwen3) — see constraint C1
  below.
- **Flush**: torch VQ K-encode sits at the same stream position as the
  proven int2 quant kernel; trash row cannot enter `req_to_token`
  (allocator never issues it); duplicate `index_copy_` targets collide only
  on trash.
- **Kernel lanes**: masked slots gather codeword 0 then get −inf'd before
  softmax; int16 idx holds 0–255 (sign-extension moot); fp8 byte order
  matches little-endian `view(int32)` packing (G3 gate).
- **Encoder/decoder consistency**: assignment is nearest-neighbor against
  e5m2-*snapped* centroids = exactly what decode reconstructs.
- **Arm symmetry**: same clone/stack, byte-identical row files, same
  sampler backend (pytorch, verified in all three server logs), same
  chunked-prefill size, same 64/256 window for both quant arms, shared
  V-clip + absorbed-R_v path; arms differ only in the K quant tier.
- **vq2 wave codebook**: server log confirms gpqacc64k loaded 03:38, before
  the wave's first request (not our flat_ptn bundle from the earlier boot).
- **Aggregator**: complete-or-absent cells (client raises on failure), rid
  alignment by construction, bootstrap shapes correct.

## Findings

### F1 (most important — reporting) vq2 K memory ≠ information rate
As built, K indices are int16 (Samuel's measured speed choice; uint8 was
slower): 64 B/token/head codes + 8 B scale slots = **4.5 b/coord memory**
vs int2's 2.5. Information content is **2.25 b/coord** (2.0 codes + fp32
ptn scale; no zero-point) — slightly BELOW int2's 2.5. Accuracy comparisons
unaffected; any "same compression ratio" framing must say: equal code
length, ~1.8× int2's K memory in this implementation, packable to parity at
a small measured speed cost. → comparison page + report11 honest-rates.

### F2 (fairness — disclosed, resolved symmetric) calibration domain
vq2's codebook is calibrated on GPQA-concat traces (gpqacc64k) — same
domain as the GPQA eval. Checked: the OSCAR authors' rotations are ALSO
calibrated from GPQA prompt dumps (their `dump_gpqa_prompts.py` is the
rotation-calibration producer). Both arms in-domain on GPQA, both equally
out-of-domain on math500/aime25. Disclose, no correction needed.

### F3 (eval validity) subset selection
math500 cell = FIRST 200 rows in dataset order (not random): paired deltas
fine (identical rows per arm), absolute accuracy not an unbiased estimate
of full math500. aime25 n=30: descriptive only, CIs ~±15 pts
(pre-registered).

### F4 (protocol asymmetry, negligible) AIME concurrency
bf16 ran at 6 client threads; int2/vq2 reran at 4 (mixed-KV pool budget:
6×32K generations > 140K-token pool). Concurrency changes throughput and
batching numerics (~1e-3 logits), not the sampling distribution; not
directionally biased. Footnote.

### F5 (scorer deviation, deliberate) GPQA extraction = LAST `Answer: X`
Upstream simple-evals uses the FIRST regex match. Last-match is correct for
thinking traces (format gets restated mid-reasoning) and identical across
arms, but our absolute GPQA numbers are not bit-comparable to
simple-evals-scored runs elsewhere.

### F6 (hardening TODOs, engine clone)
- The non-quantized extend fallback (`use_quantized_dense_prefill ==
  False`) would read the raw K arena as values — garbage for int2 AND vq2.
  Unreachable in our config (no sliding window; custom masks only from
  speculative decoding, which is off) but deserves a loud assert.
- Mixed-KV pool HARD-CRASHES on quant-slot exhaustion where vanilla sglang
  retracts (`RuntimeError: Mixed KV windows failed to allocate quant flush
  slots`) — this killed int2's first AIME attempt. Proper fix: retraction
  support in `_alloc_for_decode_mixed`; workaround: bound concurrency ×
  max_new_tokens ≤ pool.

### F7 (reproducibility, known) no per-request sampling seeds
sglang's native /generate takes no seed; K samples not bit-reproducible.
Server seeds recorded in each cell's `server_info.json`. Statistically
irrelevant to the paired design (sampling noise independent across arms
regardless).

### F8 (cosmetic) `SGLANG_OSCAR_K_CLIP_RATIO` exported in vq2 mode but
correctly unused for K (VQ doesn't clip); V clip shared. Commented in
serve_oscar.sh; no action.

### C1 (latent design constraint) mean-shift trick requires no logit softcap
A constant per-query score shift is softmax-invariant but NOT invariant
under tanh soft-capping. Fine for Qwen3 (`logit_cap=0`); porting vq2 to a
Gemma-class model requires folding the mean differently (e.g., qm bias
added pre-cap inside the kernel).
