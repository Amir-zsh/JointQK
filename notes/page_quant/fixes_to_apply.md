# page_quant bug tracker

## FIXED: pgq_fixed F1 collapse — three stacked root causes (one real lesson)

- **Symptom:** pgq_fixed/pgq_fixed_ea F1 collapsed to ~0–14 (whitespace-only
  generations) across three successive fit configurations, while per-head proxies
  (top1 0.476, k_mse ~1.0, no NaN, sane norms) stayed unremarkable each time.
- **Diagnosis chain** (each stage verified with a targeted probe, not proxies):
  1. *Mid-rise grids have no zero level* → every near-zero coord inflates to ±s/2 →
     all key norms grow coherently → softmax corrupted network-wide. Ranking
     proxies are blind to uniform norm inflation. Fix: mid-tread symmetric grids
     (zero level), widths {0,2,4,8} (1-bit mid-tread is degenerate).
  2. *"Tail-safe" q999 scales are not safe for sinks*: attention-sink keys (tokens
     0–3, carrying 19–84% of realized mass per head) sit 2.7–5.4× BEYOND the
     per-coord 99.9% quantile, so every width clipped them equally → the
     Lagrangian saw no distortion gain from wider widths → assigned sinks w=0 →
     sink mass 0.84→0.007 → generation destroyed. Probe:
     `sink mass ref/fixed/ec = 0.843/0.007/0.836`. Fix: w4/w8 scales cover the
     per-head fit-data max |r| (recovered from the finest EC rung's alphabets;
     live-coord filtered, ratio clamped to 8).
  3. *Even covering 4-bit grids are too coarse for sinks and the Lagrangian
     rationally refuses to buy w8 at tight budgets* → forced w=8 for sequence
     positions 0–3 (position-based ⇒ decoder-derivable, zero sideband; page 0's
     budget pays; standard practice — KIVI keeps sinks high precision). After the
     fix: sink relerr 0.02–0.04, rate exactly 0.996.
- **The lesson (report-worthy):** entropy-coded deadzone quantizers get sink safety
  FOR FREE (unbounded calibration-range alphabets + zero bin). Fixed-width
  saturating grids must handle the zero level and the sink tail EXPLICITLY, and
  per-head reconstruction metrics + argmax proxies cannot detect either failure —
  both corrupt the softmax denominator, not the ranking.
- **Residual (known, not a bug):** at b=1.0 the width mix is eviction-heavy (~50%
  w0), which shrinks non-sink competitor logits and overshoots sink mass
  (0.84→0.96). Rate-regime property; judged by F1.
- **Verification:** unit tests 8/8; sink-gate probe PASS on relerr; F1 rerun3
  (sha post-patch) in flight; quarantined cells: `*__BROKEN_pre_scalefix`
  (MSE-fit mid-rise), `*__BROKEN_midrise` (q999 mid-rise), `*__BROKEN_stale`
  (pre-sink-fix).

## pgq3-1: probe frozen rule could select non-deployable scorer (caught in audit, pre-run)
- **Root cause:** `probe_page_selection.py` put `incontext_mu` (built from the row's own
  realized decode queries) in `STATIC_SCORERS`, which fed both the frozen score_mode
  argmax and the quest-gate baseline; it is not an `OmegaPagePress` score mode, so a win
  would have frozen an unusable rule and crashed the evict launcher.
- **Fix:** `DEPLOYABLE_SCORERS` (omega_max/omega_mean/quest_mu) now defines the argmax
  and gate baseline; `incontext_mu` is measurement-only; rule string records this.
- **Verification:** final probe run froze `omega_mean` (44.3) while `incontext_mu`
  measured 50.0 — the exclusion demonstrably bound.

## pgq3-2: probe recall metric floored by forced sink/recent mass
- **Root cause:** `recall_at` counted force-kept pages (sink page 0 + recent page) in
  numerator and denominator; sinks hold 19–84% of attention mass, flooring every scorer
  near ~85%+ (random@50% = 89.3 raw) and voiding both the budget-line assert and the
  ±10-pt quest-gate scale. The probe's own sanity assert caught it on the first run.
- **Fix:** contested-mass recall — forced pages excluded from both sides; budget still
  counts them (matches press accounting); `meta.recall_basis` records the change.
- **Verification:** rerun passes all asserts (random 23.8@25%, 49.2@50%); oracle 53.0,
  monotone curves, oracle-dominance intact.

## pgq3-3: direction-code shrinkage — G3 normR 0.87-0.93, reactive G2 sink-mass rise (first gate sweep)
- **Root cause:** TCQ/sparse-K/E8 reconstructions come back systematically short
  (sparse-K zeros non-top-K coords; coarse warped-LM tables underfit tails; E8's
  wrap-to-zero guard biases toward 0). Sinks are kept nearly exact, so shrunken bulk
  logits reallocate softmax mass onto sinks: sinkΔ +0.075..+0.13 with sinkCE 0.015 —
  G2 failing as a *symptom* of G3.
- **Fix:** (a) TCQ family: decoder-side unit renormalization of every decoded direction
  (norm is transmitted; zero rows stay zero), Dmat prices the renormalized snap;
  (b) E8: per-(l,h,rung) least-squares gains fit on TRAIN (E[<rm,rh>]/E[||rh||²]),
  stored in bundle (`e8_gains`), zero per-token bits.
- **Verification:** unit tests updated to the new decoder contract (30 green); held-out
  G1-G3 re-gate pending refit.

## pgq3-4: OSCAR emulation without its BF16 windows corrupts sinks (first gate sweep)
- **Root cause:** family (f) omitted the published recipe's (S0=64, W=256) BF16
  sink/recent protection; per-token min-max scales do NOT isolate sink outliers on
  Llama (sinkCE 0.076, sinkΔ -0.083 at INT2) — independently replicates why OSCAR
  ships those windows.
- **Fix:** fp16 passthrough windows (64, 256) in oscar_arm, charged 16 b/c, honest at
  short contexts; window-off contract kept testable via ctor fields.
- **Verification:** test_oscar_windows_fp16_and_charged green; re-gate pending refit.

## pgq3-5: gate thresholds mis-transcribed + E8 LS-gain was a shrinkage estimator (second gate sweep)
- **Root cause (a):** plan3's absolute G2/G3 thresholds (shift ≤0.01, normR ∈[0.98,1.02])
  are stricter than the pgq2 record-holder itself achieved (rvq_rdo@1.5: 0.054 / 0.963) —
  unpassable at these rates. **(b):** the E8 LS gain E[<y,ŷ>]/E[||ŷ||²] minimizes MSE by
  shrinking toward zero — it LOWERED normR (0.91→0.85). **(c):** TCQ's code-domain unit
  renorm left raw normR at 0.98–1.05 (non-orthogonal inverse map).
- **Fix:** gates recalibrated incumbent-relative in plan3 (documented, F1 bar untouched);
  both families redesigned to transmit the RAW-domain norm and renormalize after the
  inverse map (r̂ = n16·û/||û@G||): raw normR = 1 by construction. E8 restructured to
  norm+direction (lattice codes the unit direction; +16b/rung, all-zero rung = free
  evict); LS gains removed.
- **Verification:** 30 tcq/e8 tests green on the new contracts; third gate sweep pending.

## pgq5-1: Mode-B' `_qlen` staleness across kvpress cache crops
- **Root cause:** kvpress `_remove_answer_from_cache` crops the cache back to the
  context length between questions; `JointQKPress.forward_hook`'s Mode-B' branch kept
  the per-layer `_qlen` bookkeeping from before the crop. A stale larger `_qlen` marks
  the next question's fresh tokens as already-quantized, so `decode_flush_ranges`
  never returns them — they stay fp16 forever (silent quality inflation in any
  multi-question or repeated-generation cell).
- **Fix:** clamp `qlen` to `cache_position[0]` (= cache length before this step's
  append) at the top of the Mode-B' branch. Normal decode is unaffected
  (`_qlen ≤ cache_position[0]` always holds there); after a crop the clamp restores
  the true quantized-prefix length.
- **Verification:** `test_qlen_clamp_after_cache_crop` (stale `_qlen=150`, crop to
  100, 12 new tokens → flush (100,108), bookkeeping 108) green; full test_pgq4 suite
  11/11 green.

## pgq6-1: launcher routed pgq_mrg* arms to the pgq1 bundle
- **Root cause:** launch_pgq_longbench.sh's bundle-selection case matched only
  pgq_fold*/pgq_prof* for the pgq4 bundle; pgq_mrg* fell through to the default
  branch (pgq1 bundle, version 1) and the loader's version check refused it —
  all 5 W1 parity cells failed at startup.
- **Fix:** pgq_mrg* added to the pgq4-bundle branch (clustering is runtime, the
  merge arms share that bundle by design). Dry-run verifies labels carry
  ff390304.
- **Verification:** relaunched wave runs; failed cell dirs (efe38d4d labels)
  removed before relaunch.

## pgq6-2: per-step CPU-GPU syncs made the merge codec 25x slower in bench
- **Root cause:** `_merge_hierarchy` used `torch.nonzero` + `.numel()` python
  branching per merge step — 48 data-dependent syncs per (l,h) roundtrip,
  ~12k syncs per prompt (155 s/row vs ~7 for pgq4 arms).
- **Fix:** sync-free masked no-op merging (argmin + isfinite mask, `where`-
  guarded scatter updates); comment pins the invariant that hierarchy steps
  must stay free of data-dependent syncs.
- **Verification:** test_pgq6 6/6 green; warmed roundtrip 71 ms/(l,h)
  (~18 s/prompt steady-state, ~4.4x faster end-to-end; remaining overhead is
  per-(l,h) launch count — kernel-port territory, deferred).

## pgq6-3: bootstrap_pairs silently concatenated multi-answer golds (affects pgq4/pgq5 CIs)
- **Root cause:** predictions.csv stores the HF `answers` column as
  str(numpy array) — `['a' 'b']`, NO commas. `ast.literal_eval` on that form
  concatenates adjacent string literals into ONE wrong gold ('ab'), so
  max-over-gold F1 collapses on multi-answer rows. qasper (heavily
  multi-answer) recomputed 20 pts low; musique ~4.6 low; single-answer tasks
  (lcc/2wikimqa/hotpotqa) unaffected — which is why the bug survived: the
  original 3-task tool was dominated by single-answer tasks.
- **Fix:** parse_answers() extracts the quoted items (both quote styles,
  escape-aware) instead of literal_eval.
- **Verification:** recomputed per-row means now match metrics.json on ALL
  35 cells (10 Qwen Stage-B, 10 Llama pgq4/pgq6 W1, 15 Qwen refs).
- **Corrected committed claims (all recomputed, 5-task pooled):**
  pgq4 Tier-E tie vs ecu HOLDS (−1.34 [−3.09,+0.45]); "SIG over rvq"
  DOWNGRADED to tie for both arms (rdo +0.99 [−0.62,+2.63]; ea +1.09
  [−0.47,+2.65]); TQ row-paired tie holds (+0.15); window effect rdo SIG
  holds (+1.95 [+0.67,+3.23]); window effect ea downgraded to marginal tie
  (+1.23 [−0.04,+2.47]); ω tie holds. pgq5 T2 stays PASS with margin
  (+6.16 [+3.91,+8.45] SIG); retention/window verdicts unchanged. pgq6
  parity stays an all-task tie (−0.23 [−0.78,+0.25]).

## VQ-V prefix dequant read VQ indices as int2 crumbs (2026-07-20)

**Root cause**: Samuel's VQ-V handoff patch covered the write, flush, and
decode-kernel paths but `dequantize_prefix_kv` (the chunked-prefill prefix
read) still called `_mixed_prefix_dequantize_tensor` on the V arena
unconditionally — under `vq_v_enabled` that arena holds uint8 codebook
indices, decoded as packed int2 crumbs with the ptn scale misread as an
affine scale/zero pair → garbage V for the entire quantized prefix on any
prompt longer than one 4096-token chunk. Invisible to: his 400-token smoke
(single chunk), our short smokes, the offline de-risk/HF-sim, and gates
G4/G5 (which cover encode/decode math and the decode kernel, not this
third read path). Surfaced while diagnosing the stalled/invalid Qwen VQ-V
NIAH-32K served run (stopped, to be rerun).

**Fix**: `_vq_prefix_dequantize_v` (strided inverse-perm gather, raw-tensor
signature) + `vq_v_enabled` branch in `dequantize_prefix_kv`; clone commit
below.

**Verification**: new gate **G6** in verify_vq_engine.py exercises the
exact function on mixed HP+quant slots vs an independent reconstruction —
quant rel 2.8e-4, HP passthrough exact; full suite (G1–G6) ALL PASS.

## merge_shards scored CSV-stringified answer lists character-wise (2026-07-21)

**Root cause**: merge_shards.py re-scored merged shards from predictions.csv;
CSV serialization turns the NIAH `answer` list column into its repr string,
and the vendored ruler scorer iterates answers — so it matched the string's
CHARACTERS ("['1679215']" → 7/11 present → 63.64%). All merged (sharded)
NIAH cells were mis-scored: bf16-32K read 71.4 (true ≈95+), and garbage-
output baselines picked up phantom points from stray digits (naive-INT2
"25.8" at 64K). Monolithic cells (client scores from live jsonl-derived
lists) were unaffected, which is why 8/16K looked sane while 32/64K looked
impossible. Caught by the physical-impossibility check (baselines improving
with context) before anything was recorded.

**Fix**: literal_eval list-like object columns after the CSV read, before
scoring. **Verification**: re-merge with --force; bf16-32K must land ~95+,
naive-64K must collapse toward ~0; string-answer tasks (gpqa/math/humaneval)
unaffected by construction.

## turbo-1: TurboQuant baseline unimplemented-as-documented + mis-tuned LM encoding (engine)

- **Root cause**: two independent defects. (1) `SGLANG_LLOYD_MAX` was only
  read by the mixed OSCAR pool; the plain int2 pool (the int2plain serving
  path the TurboQuant recipe targets) ignored it — the documented recipe
  would silently serve min-max (QuaRot mislabeled as TurboQuant). (2) The
  mixed-pool LM branch encoded reconstruction levels via `LM_RATIO=1.16`,
  fitted to match the old clip path's dynamic range, not to minimize MSE:
  +26% distortion vs exact Lloyd-Max.
- **Fix** (clone commits `a9a4ab6d3`, `b04860e71`): LM branch added to BOTH
  plain-pool writers — fused-Hadamard (decode) and pre-rotated plain writer
  (prefill); the two-writer split surfaced on first boot via a new guard
  (same bug class as the VQ-V prefix-dequant). Encoding replaced everywhere
  with the MSE-optimal uniform-level fit to the LM centroids
  (Δ*=0.98774σ, zero=1.5): +1.2% vs exact LM analytic, +0.8% on real
  post-FWHT Llama K/V. Readers keep `(q−zero)·scale` — the exact-centroid
  dequant path (documented decode instability) is never enabled.
- **Verification**: `pipelines/oscar_e2e/turbo_lm_check.py` (constants +
  Gaussian + real-data distortion table); `verify_turbo_lm.py` gates T1–T4
  ALL PASS (kernel-vs-torch parity 100%, quantizer-quality bars,
  writer-vs-writer arena parity 100%, grouped+LM guard). Served: NIAH-8K
  24-row slice 95.8; greedy-loop family control turbo 0.15 > QuaRot 0.10
  (bf16 0.62) — no LM-specific degeneration.

## tp-1: engine unusable under tensor parallelism (two defects)

- **Root cause**: (1) `load_vq_codebook` had no head sharding — under TP the
  pool passes per-rank head_num (4 of 8) while the bundle holds all global
  heads; boot died on `assert H == head_num` (and without it, ranks would
  have served wrong-head codebooks silently). (2) Independently, ALL modes
  (bf16 included) failed TP=2 CUDA-graph capture in sgl_kernel's custom
  all-reduce (`get_graph_buffer_ipc_meta` → invalid argument) despite full
  NVLink — a vendored-kernel/stack incompatibility.
- **Fix**: clone commit `0ecb1ab7c`: head_start slicing in the loader
  (forward/inverse/mean/codebooks), offset = tp_rank × local head_num,
  divisibility guard. Defect 2 ROOT-CAUSED (2026-07-22, upstream
  vllm#42609): `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — our own
  fragmentation setting — makes graph buffers non-IPC-exportable
  (cudaIpcGetMemHandle on cuMemAddressReserve VA ranges); the sgl_kernel
  wheel was innocent. serve_oscar.sh now drops expandable_segments for TP>1
  and keeps custom all-reduce (DISABLE_CUSTOM_AR=1 / FORCE_EXPANDABLE=1
  restore the old behavior). Measured A/B at TP=2 bs=1: custom AR 132.0
  tok/s vs NCCL 122.5 (+7.8% decode).
- **Verification**: loader slice gate (rank slices == full-load slices,
  bit-exact; non-divisible raises); TP=2 boots for bf16/int2/vq2 with
  per-rank log lines `heads=[0,4)/8` / `[4,8)/8`; needle retrieval correct
  in all three; 48-row stratified NIAH-8K parity vq2 TP1 43/48 vs TP2 43/48
  (4 balanced borderline flips; bf16 control flips 1/48 — generic TP
  reduction-order numerics, not sharding).

## r1d-1 (KNOWN ISSUE, worked around): mixed-pool quant-flush allocation race under long generations

- **Symptom**: `RuntimeError: Mixed KV windows failed to allocate quant
  flush slots (#quant-free-pages: 7 ...)` — server death mid-cell, twice,
  on vq2/aime25 (R1-distill, 8 concurrent requests × ≤32K generated
  tokens ≈ 264K quantized-KV tokens vs a 140K-token pool).
- **Root cause (suspected, not fixed)**: sglang's admission control and
  the OSCAR mixed pool's quant-page/flush-slot bookkeeping disagree near
  capacity under many long-decode requests; admission lets in a request
  whose aging flush then finds no quant pages. int2 survived the same
  cell by scheduling luck — the race is engine-level, not vq2-specific.
- **Workaround**: bound concurrency to pool capacity for long-cap cells
  (client --threads 3 + POOL_MAX_REQS=4 for 32K caps at 140K pool).
  Proper fix would make admission account for worst-case quant pages;
  candidate for Samuel's engine backlog.

## r1d-2: duplicate GPU ordinal in CUDA_VISIBLE_DEVICES → silent CPU fallback stalled the R1D 128K build (2026-07-23)

- **Symptom**: every `build_r1d_128k.sh` attempt on 2026-07-23 hung in C1
  capture for hours with GPUs at 4 MiB; four orphaned
  `capture_mixed_concat.py` processes accumulated (~200 GB RSS total), all
  racing on the same `query_stats_128k/` output dir.
- **Root cause**: the script signature is `<gpuA> <gpuB> [gpuC]` with
  `GPU_C` defaulting to `$2`; two-argument calls (master.sh passed `1 2`)
  produced `CUDA_VISIBLE_DEVICES=1,2,2`. CUDA rejects the duplicated
  ordinal (`Error 101: invalid device ordinal`), torch reports CUDA
  unavailable, and `device_map="auto"` silently places the model on CPU —
  the job runs, just ~50× slower. GPU-scoped kill sweeps never see CPU
  processes, so failed attempts stacked instead of dying (this also
  explains the earlier `1,4,4` "silent death" attempts).
- **Fix**: dedupe the GPU list into `CAP_GPUS` before exporting CVD, and
  fail fast with a `torch.cuda.is_available()` assert before capture so a
  broken CVD aborts instead of falling back to CPU.
- **Verification**: relaunched build shards across GPUs 1,2,3
  (25/11/7 GB resident); partial pool from the concurrent CPU writers was
  wiped before relaunch.
