# page_quant bug tracker

## FIXED: RunPod queue coordinator relaunches in-flight items after a restart

- **Symptom:** appending new items to a live `unified_queue.sh` (coordinator-
  wrapper-only restart, to avoid killing in-flight `run_cell.sh` jobs) caused
  the SAME cell to be computed twice on two different GPUs, sometimes
  minutes to tens of minutes after the restart rather than immediately.
- **Root cause:** the new coordinator generation's `JOB_GPUS`/`RESERVED`
  bookkeeping is in-memory and starts empty. An item launched by the PRIOR
  generation and still running at restart time survives as an orphan (it's
  not a child of the new coordinator), but the new generation has no record
  of it. Its `nvidia-smi`-busy GPU correctly blocks an immediate relaunch,
  but once *any* GPU frees later while that item is still the front pending
  entry (no `metrics.json` yet, since the orphan hasn't finished), the new
  generation launches it again — the duplicate can appear well after the
  restart, whenever that GPU-free event happens to occur.
- **Fix:** startup reconciliation in `unified_queue.sh` — before entering
  the main loop, `ps -eo cmd` is scanned for `run_cell.sh --protocol ...`
  invocations, and any `QUEUE_ITEMS` entry matching a currently-running
  orphan (`--arm X --task Y --gpus`, matched through the literal `--gpus`
  that always follows `--task`'s value in `run_cell.sh`'s own invocation —
  needed to avoid `niah_131072` false-matching inside
  `niah_131072_bf16_remainder`) is dropped before the loop starts.
- **Verification:** unit-tested the reconciliation snippet standalone
  against a synthetic `QUEUE_ITEMS`/`ps` pair covering the exact prefix-
  collision case (`niah_131072` unsharded vs. `niah_131072_bf16_remainder`
  shard 6/7) — confirmed the true duplicate drops and the unsharded item
  survives. Then deployed and confirmed via live restart logs on both pods:
  `RECONCILE: dropping ...` fired for every genuinely in-flight orphan, and
  no further duplicate launches occurred across several subsequent restarts.
- **Separate, related trap (not a code bug):** each pod is an independent
  filesystem — a cell finished on one pod is invisible to another pod's own
  `metrics.json`-existence check. Running the same task set across multiple
  pods (e.g. pulling backlog items forward onto an idle pod) requires
  syncing `metrics.json`/`provenance.json` (small files only, not the large
  `predictions.csv`/`io_log.jsonl`) back across pods afterward, or the other
  pods will legitimately recompute cells they don't know are already done.

## FIXED: HumanEval scorer treated gpt-oss's raw analysis channel as the solution

- **Symptom:** gpt-oss-20B HumanEval smoke cell (16 rows, greedy) scored 1/16
  (6.25%) — implausibly low for a 20B code-capable model. `predicted_answer`
  for every row was the full raw completion starting with `"analysisWe need
  to implement ..."`, i.e. the chain-of-thought preamble, not code.
- **Root cause:** `pipelines/eval/code_scorers.py:extract_code()` strips
  `<think>...</think>` (Qwen3) and falls back to slicing from the first
  line-initial `def `/`class `/`import `/`from `/`@` when there's no fenced
  code block. gpt-oss's harmony format switches from the analysis channel to
  the final channel via control tokens that decode (special tokens stripped)
  to the literal, delimiter-less substring `"assistantfinal"` glued onto the
  end of the analysis prose — e.g. `"...produce code.assistantfinalfrom
  typing import List\n..."`. Because `"from"` isn't at a line start there,
  the fallback regex never matches, and the entire analysis-plus-code blob is
  returned as "code" (a syntax error when executed as Python). Confirmed
  100% of the 16-row sample carried the marker with no fence.
  LiveCodeBench is unaffected: its completions are consistently fenced, so
  the existing regex already finds the right block regardless of the
  preceding "assistantfinal" text (verified against the existing
  gpt-oss/bf16/lcb result, 1268/1280 rows carry the marker and score
  correctly at 76.2%).
- **Fix:** `extract_code()` now splits on `"assistantfinal"` (last
  occurrence) before applying the fence/heuristic logic, isolating the final
  channel's content.
- **Verification:** re-scoring the same 16-row prediction file with the
  patched extractor: 16/16 compile as valid Python, 15/16 (93.75%) pass —
  consistent with a working code-capable 20B model at greedy decoding on this
  small a sample.

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

## p2-1: premature mixed-KV flush double-assigns a quant slot (latent, 2026-07-24)

- **Symptom**: in the offline gate
  (`pipelines/oscar_e2e/verify_gptoss_mixed_pool.py`), an 8K prefill
  followed by decode steps whose per-request flush counter starts at 0
  instead of the layout-derived `counter_init` ends with one quant slot
  referenced by two prefill positions (e.g. slot 7887 held by positions
  7936 and 7944, exactly `N_Q` apart, straddling the
  quant/HP-recent boundary at `seq - hp_recent`).
- **Root cause**: with the counter seeded low, the first flush fires
  before the demote window `[seq - hp_recent - flush_overflow, seq -
  hp_recent)` has moved fully inside HP-recent, so the 8-position window
  straddles the boundary; part of the page is written into
  `req_to_token` and part is returned to the free pool, and the page is
  later handed out again. This is the same failure the `_mixed_extend_
  layout_counts` docstring predicts for chunked prefill.
- **Status**: NOT the cause of the gpt-oss int2 zero-retrieval — with the
  production seed (`counter_init` from `_mixed_extend_layout_counts`) the
  gate is clean at 8192/8192 unique slots and 100% self-retrieval over
  every position after 256 decode steps. Filed because any path that
  admits a request without seeding the counter (retract/resume,
  chunk continuation) reaches the bad state, and it is a plausible source
  of the observed ~2 slot/req leak.
- **Verification**: `verify_gptoss_mixed_pool.py --decode-steps 256`
  reproduces on demand by seeding the counter at 0.


## p2-2: SWAKVPool missing `release_req_slab` delegate (hybrid models only)

- **Symptom**: on hybrid-SWA models the per-request HP-recent ring cursor and
  flush counter are never reset — neither on request completion nor on
  retract.
- **Root cause**: `release_kv_cache` (mem_cache/common.py) resets them via
  `getattr(kvcache, "release_req_slab", None)`. For hybrid models
  `get_kvcache()` returns the `SWAKVPool` wrapper, which forwards
  `hp_prefix_tokens` / `hp_recent_tokens` / `hp_global_offset` /
  `flush_interval` / `N_Q` to the inner pool but **not** `release_req_slab`,
  so the `getattr` returns None and the reset is silently skipped.
- **Impact**: NOT the cause of the gpt-oss INT2 failure (that is a method
  limitation — see `studies/report14.md`). The cursor advance is a rotation of
  a per-request ring, so correctness appears unaffected, but it is a plausible
  source of the observed ~2 slot/req leak and it is silently wrong.
- **Fix**: add a `release_req_slab` delegate to `SWAKVPool` forwarding to
  `full_kv_pool`.
- **Status**: applied with the p2-3 ring-lifecycle fix. A second strict
  24-request pass reused request slots without leaking.

## p2-3: hybrid-SWA mapping overwritten on the first HP-recent ring wrap

- **Symptom**: after concurrent long generations return to idle, the full
  allocator balances exactly while the SWA allocator reports approximately
  one leaked slot per request that crosses the first HP-recent ring wrap
  (`full_leaked=0, swa_leaked=21` in one 24-row reproduction).
- **Root cause**: the ring held `hp_recent + N_Q - 1` slots, matching the
  maximum steady-state live occupancy but not the operation order. Decode
  allocates and installs the new full→SWA mapping before the flush plan
  releases the oldest `N_Q` mappings. On the flush step that first wraps the
  ring, allocation overwrote slot 0's mapping; cleanup then freed the new SWA
  slot and lost the old one.
- **Fix**: reserve one transient slot by sizing the ring as
  `hp_recent + N_Q`. The flush-step allocation now lands in the final spare
  slot, the flush releases slots 0 through `N_Q-1`, and only the following
  decode step reuses slot 0.
- **Verification**: a focused lifecycle regression fails with the old geometry.
  With the fix and strict idle checking enabled, two consecutive 24-row VQ2
  runs completed with scores 75.0 and 91.67. The fixed-length pass forced all
  24 requests across the wrap and reused request slots; all 126 prefills across
  both passes were graphed and the server returned to idle with no leak. The
  previously failing split-16 configuration also completed 24/24, scored
  95.83, and graphed all 57 prefills under strict checking.

## p2-4: chunked prefill attends to the prompt through the quantized cache

- **Symptom**: gpt-oss-20B vq2 loses ~24 NIAH points whenever the prompt
  spans 2+ prefill chunks (healthy300: 67.33 at chunk 4096 vs 91.00 single
  chunk); bf16 is chunk-insensitive (95.67 vs 94.33). Reproduces at bs=1;
  per-task losses concentrate in whatever the trailing tokens must retrieve.
- **Root cause**: with one chunk, the prompt's forward pass attends to the
  exact in-pass k3/v3 and quantization only ever affects decode. With 2+
  chunks, chunk N+1 reads chunk N through `dequantize_prefix_kv`, i.e.
  through VQ-K reconstructions and int2-V crumbs, and those errors compound
  through every remaining layer of the prompt's own forward pass — corrupting
  the hidden states of the trailing tokens, where the question lives. Control
  that pinned it: forcing the whole prefix into the bf16 HP tier
  (PREFIX_TOKENS=9216) recovered full accuracy, so the chunked attention /
  concat / causal / sink logic was all correct and only the prefix VALUES
  mattered.
- **Fix**: `SGLANG_MIXED_KV_EXACT_CHUNKED_PREFILL` (default ON). While a
  request is mid chunked prefill, the unified pool keeps a bf16 shadow of the
  stored-space rows its non-final chunks write to quant slots
  (`shadow_register` in `_alloc_for_extend_mixed`; capture in
  `set_kv_buffer` from the already-computed HP-tier tensors);
  `dequantize_prefix_kv` swaps shadow rows in after reconstruction, so the
  swap covers K and V in all quant flavors (VQ or int2). Release is deferred
  to the alloc step after the final chunk (whose forward still reads the
  shadow), with owner-absence at extend allocs as the abort fallback. Quant
  codes are still written at chunk time, so the FINAL cache is bit-identical
  with or without the flag and decode behavior is unchanged. Cost: the
  in-flight chunked prompt's K/V in bf16 (quant layers only) during its
  prefill. Skipped under CUDA-graph capture (falls back to dequantization).
- **Verification**: engine gates G1–G6 all pass. healthy300 vq2 chunk 4096:
  67.33 -> 89.00 (twice, identical), vs single-chunk 91.67/92.33 on the same
  code — the chunking penalty shrinks from -23.67 to ~-3 with mixed per-task
  signs, comparable to bf16's own +-1.3 chunk sensitivity. Single-chunk
  scores are unchanged (91.00 pre vs 91.67/92.33 post, shadow provably
  inert there since nothing registers). Artifacts:
  `artifacts/oscar_gptoss20b/healthy/vq2_c{4096,16384}_shadowfix{,_r2}`.
- **Attribution (root-cause depth 2)**: with the shadow's side switch
  (`SGLANG_MIXED_KV_SHADOW_SIDE=k|v|kv`), healthy300 at chunk 4096:
  no shadow 67.33, V-only exact 71.33 (+4.0), K-only exact 83.67 (+16.3),
  both 89.00 (+21.7). VQ-K reconstruction error causes ~3/4 of the loss,
  int2-V ~1/4, additive. The int2-arm control is uninformative (gpt-oss
  INT2 is already at 0.00 from its known method failure, report14).
- **Not a calibration artifact**: the MXFP4-calibrated 128k bundle improves
  decode-side quality (single-chunk 91.7 -> 93.67) but does NOT reduce the
  native chunked loss (no-shadow c4096: 62.33, still ~31 pts below its own
  single chunk). The same codes that cost ~1 pt when read at decode cost
  ~30 pts when the trailing prompt tokens read them during prefill — the
  fragility is intrinsic to prefill-time reads at this rate, so no codebook
  upgrade closes it; the shadow (or unchunked prefill) is the fix.
- **Best configuration**: MXFP4 bundle + shadow: chunked 93.00 vs
  single-chunk 93.67 — parity within noise.
  Artifacts: `artifacts/oscar_gptoss20b/healthy/vq2_c4096_shadow{K,V}`,
  `vq2mx_c4096_{noshadow,shadow}`, `vq2mx_c16384`, `int2_c4096`.
- **Default changed to OFF** (user decision): the shadow is opt-in via
  `SGLANG_MIXED_KV_EXACT_CHUNKED_PREFILL=1`. Accuracy-critical chunked
  serving MUST set it explicitly; without it, chunked prefill carries the
  quantization penalty measured above.
- **Universality (answers "why not other models")**: Qwen3-8B is NOT immune —
  it was only ever tested at 8K where it scores 100 (ceiling). At 64K under
  the exact published protocol (gpqacc64k bundle, chunk 8192): chunked 78.00
  (reproduces the 77.69 record), single-chunk 81.12, chunked+shadow 81.00;
  bf16 chunk sensitivity only -0.50 (84.25 vs 84.75). Same mechanism, same
  full recovery by the shadow, smaller magnitude than gpt-oss because Qwen's
  K quantization error is smaller (QK-norm-bounded rows, half the row norm,
  lower bundle distortion). Consequence: every chunked-served quant accuracy
  number, Qwen included, was quietly depressed by this effect; the published
  Qwen vq2-64K record understates vq2 by ~3 points. Artifacts:
  `artifacts/oscar_gptoss20b/healthy/qwen64k_vq2gpqacc_*`, `qwen64k_bf16_*`.

## gpt-oss NIAH rows missing a harmony channel tag → self-recovery truncation (2026-07-28)

**Root cause**: `export_prompt_rows.py`'s `build_prompt()` glues `answer_prefix`
directly onto `apply_chat_template(add_generation_prompt=True)`'s output. For
gpt-oss that output is a bare `<|end|><|start|>assistant` — no `<|channel|>`
tag — even though gpt-oss's own system prompt states "Channel must be
included for every message." The model doesn't treat the injected
answer_prefix as a valid continuation; it burns tokens re-deriving the whole
harmony structure (channel tag, reasoning, channel switch, final tag) before
it can answer. On NIAH's 128-token budget this frequently truncates before
ever reaching a formatted answer — verified one `niah_multivalue` row where
the model's own reasoning found all 4 correct numbers but ran out of budget
before restating them in the final channel. Checkpoint-independent (confirmed
on both `unsloth/gpt-oss-20b-BF16` and `openai/gpt-oss-20b`) and
quantization-independent (bf16 arm, zero KV compression, equally affected) —
purely a prompt-construction bug. Task-difficulty correlated: harder-to-locate
needles (multivalue/multiquery/single_1/2/multikey_1) fail 40-100% of the
time; easy ones (multikey_2/3, single_3) stay under 10%. Almost certainly the
true root cause behind the task-13 "vq2 NIAH loss concentrated in
single_1/multivalue/multiquery" investigation, which had attributed the same
pattern to vq2 specifically. math500 (200/200 clean) shares the identical
template bug but never manifests it — empty `answer_prefix` + 32768-token
budget avoids both trigger conditions.

**Fix**: `build_prompt()` takes a `model=""` kwarg (default no-op, so
Qwen/Llama callers are unaffected); when `"gpt-oss" in model.lower()`, insert
`<|channel|>final<|message|>` before `answer_prefix`.

**Verification**: direct `transformers.generate()` A/B on 3 rows (rid 600
niah_multivalue, rid 100 niah_single_2, rid 400 niah_multikey_2 control),
greedy, same 128-token budget as production. Fixed prompts answered correctly
with near-zero preamble on all 3 (down from ~25-80 wasted tokens); the
control row was unaffected. **Not yet applied to the row files** —
`artifacts/prompt_rows/niah_*_gptoss*.jsonl` still use the unfixed prompts;
regenerating them invalidates every prior gpt-oss NIAH number in the project
(today's RunPod smoke cells, `niah8k_postmerge`, `healthy/*`).

## RULER scorer breaks on comma-formatted numeric answers (2026-07-28)

**Root cause**: `vendor/kvpress/evaluation/benchmarks/ruler/calculate_metrics.py`
scores by exact substring match (`ref.lower() in pred.lower()`). Reference
answers are plain digit strings ("4484859"); models frequently write large
numbers with thousands-separators ("4,484,859") in prose, which breaks the
substring match even though the answer is correct. Surfaced while verifying
the harmony-channel-tag fix above: the fixed prompt for niah_multivalue (rid
600) got all 4 numbers exactly right but scored 0.0 because of comma
formatting alone.

**Fix (LOCAL PATCH to vendored code — not upstream kvpress)**: added
`polish()`, stripping a comma only when it sits directly between two digits
(`(?<=\d),(?=\d)`), applied to `predicted_answer` in `calculate_metrics()`'s
existing per-row cleanup step (alongside the control-character strip already
there). Deliberately narrow: a two-item list written as "12, 34" (space after
comma) is untouched; only comma-with-no-space between digits merges, which no
observed model output does. `vendor/kvpress` is its own separate git
checkout (not part of this repo's history) — this patch is NOT committed
anywhere and will be silently lost on any future re-clone/pull of that
directory unless re-applied from this note.

**Verification**: `calculate_metrics()` end-to-end on the captured
comma-formatted niah_multivalue prediction: 0.0 → 100.0. Regression check —
an already-correct plain-digit prediction (niah_single_2) still scores 100.0.
No inference rerun needed to re-score existing `predictions.csv`/
`io_log.jsonl` data with the patched scorer.

**Addendum (same day)**: the actual production rerun of gpt-oss mxfp4 bf16
(niah_smoke, with the channel-tag fix from the entry above applied) landed
7/8 subtasks at 100.0 but `niah_multiquery` at 33.33 — reading the raw
predictions, 2 of 3 rows had all 4 numbers exactly right but formatted with a
narrow no-break space (U+202F) inside a markdown table ("1 681 570") instead
of a comma; the model used a plain bullet list (clean digits, no separator)
for the row that scored correctly. Same bug class, different character.
Extended the pattern to `[,  ]` (comma, no-break space, narrow
no-break space) between digits. Re-verified against a fresh kvpress clone
(still applies), re-scored the same predictions.csv with zero inference
rerun: 33.33 → 100.0. Full bf16 niah_smoke now 8/8 subtasks at 100.0.

## vq2 pool double-count survives on the hybrid-SWA branch (gpt-oss only, 2026-07-28)

**Root cause**: `pool_configurator.py` charges vq2 an extra `head_dim // 4`
bytes per (token, kv-head) on top of
`_get_unified_mixed_kv_bytes_per_quant_token`, whose
`(head_dim + v_head_dim) * hp_bytes // n_q` term already covers K *and* V.
`unified_kv_pool._create_arenas` allocates the VQ index arena with
`if self.vq_enabled: ... else: <int2 K arena>` — in *place* of the int2 K
arena, never in addition — so the extra charge is pure double-count. This was
found and deleted once (see the NOTE at `:266-275`), but only on the
non-hybrid branch; the `enable_mixed_kv` hybrid-SWA branch kept its own copy
at `:421-422`. gpt-oss-20b is the only hybrid-SWA model in the throughput
grid, so the surviving copy was never exercised until the native-mxfp4 run.

**Fix**: delete the `bytes_per_head += model_config.head_dim // 4` block on
the hybrid branch, leaving a comment pointing at the non-hybrid NOTE.

**Verification**: gpt-oss-20b at `mem_frac 0.85`, boot-reported pools —
vq2 2,166,760 → **2,303,896**, exact parity with oscar_int2 (identical real
bytes/token), and derived `b_max` 69 → 74. Before the fix the arm also left
3.04 GB of HBM unused (`Memory pool end. avail mem=14.77` vs int2's `11.73`).
Qwen3/Llama numbers are unaffected — they take the non-hybrid branch, which
was already correct. Prior gpt-oss vq2 artifacts no longer reproduce
bit-for-bit; the affected throughput run was relaunched from scratch.

## Hybrid-SWA: quant-tier tokens leak their SWA slots (gpt-oss, 2026-07-28)

**Symptom**: `RuntimeError: Hybrid mixed KV: SWA pool exhausted during extend
(need 8192, free 8150)` on every quant arm at 60k/bs>=29, blocking the 60k and
90k cells of `throughput_gptoss20b_fp32`. Diagnostic: the numbers are
*identical* at bs=30 and bs=29 -- the tier fills to a fixed absolute point
(~1.835 M of 1,843,112 tokens) independent of batch, so it is a leak, not a
sizing margin. Three rounds of b_max headroom tuning could not fix it.

**Root cause**: `mem_cache/common.py:659` allocates one SWA slot for *every*
token in `out_cache_loc` -- `[hp-prefix][quant-middle][hp-recent]`. The only
free path (`:827-834`, inside the flush plan) releases SWA slots solely for
tokens returned from the HP tier (`returned >= hp_global_offset`). Tokens
written straight to the **quant tier** during prefill never transit the HP tier,
so their SWA slots are never freed and leak for the life of the request. The
non-quantized hybrid path does not hit this: it runs the normal
SWARadixCache/SWAChunkCache lifecycle, which owns `sliding_window_size`.

It is also over-allocation on principle: sliding-window layers only ever read
the last `window` tokens (128 on gpt-oss), so a 60k-token request allocates
~470x more SWA slots than those layers can address.

**Proposed fix (NOT implemented -- needs verification before trust)**: allocate
SWA slots only for the HP portions of `out_cache_loc` (`hp_prefix` +
`hp_recent`), not the quant middle. The SWA window (128) is always <=
`hp_recent_tokens` (256), so every token an SWA layer can read still has a slot,
and the existing HP-return free path at `:827-834` then covers the whole
lifecycle with no new bookkeeping. `full_to_swa_mapping == 0` is already the
established "no SWA slot" state used for freed HP tokens.
**Verification required before use**: NIAH accuracy parity on gpt-oss with
quantized KV (a wrong mapping makes SWA layers attend the wrong slots silently,
which throughput gates would not catch), plus the 60k/90k throughput cells
completing without exhaustion.

---

## vq2 offline tuning table is tuned at a hardcoded kv_group_num=4, then applied at any group size

**Symptom**: on gpt-oss-20b the `vq2_triton` arm is competitive at bs=1 (189
tok/s at 30k, fastest vq2 arm) but collapses at batch -- 460 tok/s at bs=4 where
int2/vq2_cuda/bf16 all sit at 610-630, and 1,601 at bs=210 against int2's 2,820
(-43%) and vq2_cuda's 2,642 (-39%). `vq2_cuda_fp32` sets the same
`SGL_VQ2_CONFIG_JSON` and is unaffected, because the table only feeds the Triton
kernel `_decode_grouped_att_m_fwd_quant_vq2`.

**Root cause, two halves.**

1. `pipelines/throughput/kernel_study/tune_vq2_config.py:63` derived the query
   head count as `g["kv_heads"] * 4` -- a hardcoded `kv_group_num=4`, which is
   Qwen3-8B's ratio (32 q / 8 kv). gpt-oss-20b is 64 q / 8 kv = **8**. So every
   config in `artifacts/throughput/vq2_tuned_gptoss20b.json` was benchmarked at
   `h_q=32`, half the real query heads and therefore half the work per kv head.
   The table records the geometry it used (`"h_q": 32`), so the error is visible
   in the artifact.

2. `decode_attention.py:_vq2_tuned_config` gated the table on
   `(head_dim, ng, kc)` only. Those match (64, 16, 256) at both group sizes, so a
   table tuned at group 4 was applied at group 8 with no warning. Also note
   `buckets = [b for b in sorted(cfgs) if b <= batch]` clamps, so bs=210 runs the
   bs=58 entry -- extrapolation off the end of a table whose largest key is 58.

**Fix applied**: `capacity.model_geometry` now returns `q_heads` from
`num_attention_heads`; the tuner uses it instead of `kv_heads * 4`.
`_vq2_tuned_config` additionally rejects a table whose recorded `h_q / h_kv`
disagrees with the caller's `kv_group_num`, falling back to the built-in ladder.

**Verification outstanding**: the existing `vq2_tuned_gptoss20b.json` is now
rejected by the new guard (it records group 4), so vq2_triton falls back to the
ladder until the table is regenerated at `h_q=64`. Both need measuring: (a)
re-tune and confirm the new table beats the ladder at gpt-oss geometry, (b)
re-run the vq2_triton arm -- its bs>=4 numbers in
`artifacts/throughput/throughput_gptoss20b_fp32` (2026-07-28) were produced with
the mis-tuned table and understate the arm. int2, vq2_cuda and bf16 rows are
unaffected.

---

## REGRESSION (mine): the hybrid-SWA slot fix unbalanced alloc against free

**Status: open. Introduced by the fix recorded above. Present in every post-fix
gpt-oss run, including the published `throughput_gptoss20b_fp32` grid.**

**Symptom.** The scheduler logs a *negative* SWA occupancy, thousands of times:

```
Decode batch, #running-req: 32, full token usage: 0.28,
              #swa token: -6248697, swa token usage: -5.96
```

and, once enough cells run on one server, dies:

```
RuntimeError: Mixed KV windows failed to allocate quant flush slots.
  #quant-free-pages: 31, #quant-release-pages: 0, size: 6568360,
  hp_offset=6541488, recent_ring=264, max_req_slots=210
```

at 28% full-tier usage.

**Evidence it is the fix and not pre-existing.** `swa token usage: -` appears
**0** times across four pre-fix logs (`gptoss_int2_tp1`, `math_vq2`,
`niah_chunk16k_vq2`, `vq2_c8192_r256`) and **938 / 3,018 / 1,228** times in the
three post-fix `throughput_gptoss20b_fp32` int2 server logs.

**Why it hid.** The headline grid runs 3 cells per length per server (bs=1, 4,
b_max). The imbalance needs ~10 cells on one server to exhaust the quant arena,
so it only surfaced under the batch-size ladder
(`configs/gptoss20b_batch_curve.json`), which was written for an unrelated
question. Every gate passed in the grid run.

**What is established.**
- `common.py:661-685` (the fix) allocates SWA slots for the HP parts only.
- `common.py:795-803` (**untouched**) still allocates one SWA slot *per token*
  on the decode path.
- `common.py:846-854` frees only `full_to_swa_mapping[hp_returned]` where `> 0`,
  then zeroes the mapping — so it is not a naive double-free of the same ids.

**What is NOT established.** The exact path that drives the counter negative.
Leading hypothesis, unverified: `batch.maybe_evict_swa()` (`:933`, `:1027`)
accounts evicted SWA tokens by token count rather than by actually-held slots.
Before the fix every full-tier token owned a slot so the two agreed; after it,
quant-tier tokens hold none, so eviction decrements for slots that were never
allocated. A second candidate is the HP-recent ring recycling a slot id whose
previous `full_to_swa_mapping` entry is overwritten rather than freed
(`:803`, `:676`) — that leaks rather than over-frees, so it does not explain a
negative count on its own, but it may compound.

**Do not fix by reverting alone.** The original leak is what blocked the 60k/90k
cells; reverting restores that. The alloc sites must be made consistent with each
other *and* with whatever the evictor counts.

**Verification required before the gpt-oss numbers are final.**
1. An invariant assert that SWA occupancy never goes negative, run across a long
   cell ladder on one server (the batch curve is the reproducer).
2. Confirm no SWA slot aliasing — two full ids mapping to one live slot would be
   silent wrong attention on the sliding layers, which no throughput gate catches.
3. Re-run `throughput_gptoss20b_fp32` and re-issue the report/artifact.

### Root cause located, and the shape of a robust fix

`scheduler_runtime_checker_mixin.py:104` computes

```
swa_num_used = swa_tokens_per_layer - (swa_available_size + swa_evictable_size)
```

from **two independent sources of truth**:

* `swa_available_size` — physical free **slots** in the SWA allocator.
* `swa_evictable_size` — a **token** count maintained by the radix tree
  (`swa_radix_cache.py:688-689`, `self.swa_evictable_size_ -= len(node.value)`).

These agree only while the invariant **"one full-tier token owns one SWA slot"**
holds. Stock hybrid-SWA satisfies it. The mixed-quant path does not — after the
fix, only HP-tier tokens own slots — so the tree reports the full tier's ~6.5M
tokens as SWA-evictable against a 1.05M SWA tier, and the sum exceeds capacity.
Magnitude matches the observed `-6,248,697`.

This is one root cause with two symptoms: the negative counter *and* the
`alloc_quant` exhaustion, because eviction is driven by these same sizes — the
tree believes it has room, stops reclaiming, and the quant arena fills. The
reported "28% full token usage" at the crash is itself computed from the broken
accounting and cannot be trusted.

**Proposed fix (design, not yet implemented).** In the mixed-quant path the SWA
tier is entirely **request-owned**, never **tree-owned**: cached prefixes live in
the quant tier and hold no SWA slots, while live requests' HP prefix/recent parts
hold slots that the tree does not own. Therefore `swa_evictable_size_` should be
identically **0** under mixed KV, and tree-driven SWA eviction a no-op — there is
nothing tree-owned to reclaim. That removes the second source of truth rather
than trying to keep two counters agreeing, which is what makes it robust instead
of a patch.

Rejected alternatives:
* *Have the tree count only tokens with `full_to_swa_mapping > 0`* — needs a
  device read per node on every accounting query; expensive and racy.
* *Revert to one SWA slot per token* — restores the invariant but reinstates the
  original leak that blocked the 60k/90k cells, and re-inflates the SWA tier.

**Landed now (detection, independent of the fix):** the invariant is asserted at
the point of computation, so this class of corruption fails loudly instead of
running a whole benchmark grid with every gate green. Any code that breaks the
token/slot correspondence now raises immediately.

**Still required:** implement the accounting change, re-verify with the batch
ladder (`configs/gptoss20b_batch_curve.json`, the reproducer), rule out SWA slot
aliasing, then re-run the grid and re-issue the report.

### CORRECTION: upstream already solves this; the fix is to adopt its contract

Checked against the stock hybrid-SWA path in the same tree. Upstream's design is a
**two-part contract**, and the mixed path implements only the first half.

1. **Allocation is 1:1 and structural.** `SWATokenToKVPoolAllocator.alloc()` and
   `.alloc_extend()` allocate full and SWA slots in lockstep in a single call and
   record `full_to_swa_index_mapping`; `available_size()` returns
   `min(full, swa)`. The "one full token owns one SWA slot" invariant is
   *enforced by the allocator*, not assumed by its consumers — which is exactly
   why the radix tree may account SWA occupancy in tokens.
2. **SWA slots are released EARLY, independently of the full tier**, via
   `SWATokenToKVPoolAllocator.free_swa()` (`swa_memory_pool.py:569-573`) driven by
   SWA-specific eviction in `SWARadixCache` (`swa_radix_cache.py:622`). Nodes
   carry two lock refs, `full_lock_ref` and `swa_lock_ref`, with the documented
   invariant `full_lock_ref >= swa_lock_ref` (`:76-78`): once a node falls out of
   the sliding window its `swa_lock_ref` drops to 0 and its SWA slots become
   evictable **while the full-tier prefix stays cached**.

**This is precisely the capability the SWA pin was reaching for** — a small SWA
tier alongside a large full tier — obtained without breaking any invariant.

**The mixed path never calls `free_swa`** (zero occurrences in `common.py` and
`unified_kv_allocator.py`). It hand-rolls part 1 and omits part 2. That omission
*is* the original leak. The earlier fix then attacked part 1 by declining to
allocate, which broke the invariant the tree, the scheduler's admission control
and the eviction logic all rely on. `common.py:846-854` is additionally a partial
inline reimplementation of `free_swa` (same `>0` filter, same mapping zeroing)
rather than a call to it.

**Revised fix, superseding the "make swa_evictable_size_ zero" proposal above:**

1. **Restore 1:1 allocation** in the mixed path (revert the HP-only change). The
   invariant holds again, the tree's token-based accounting becomes valid, and
   the new assertion passes.
2. **Wire the mixed path into SWA release**: call `free_swa()` for tokens leaving
   the sliding window, replacing the inline block at `:846-854`. The flush plan
   already identifies returned HP slots; the same treatment must extend to
   quant-tier tokens, whose slots are what currently leak.
3. Prefer routing mixed allocation through the allocator's own `alloc`/`alloc_extend`
   rather than hand-rolled `swa_allocator.alloc()` calls in `common.py`, so the
   invariant cannot be broken again from the outside.

The earlier proposal (zeroing `swa_evictable_size_` under mixed KV) is **rejected**:
it would have made the accounting self-consistent while permanently diverging from
upstream's model and forfeiting SWA-side eviction, which is the very mechanism that
makes a small SWA tier viable.

### Follow-on: the mixed path never *asked* for SWA eviction (found by the batch ladder)

Applying the revised fix above (1:1 alloc + `free_swa`) removed the leak — 0
negative-SWA lines across bf16/int2/vq2 in the coherence check — but the batch
ladder then failed all three quant arms within a minute:

```
RuntimeError: Hybrid mixed KV: SWA pool exhausted during extend (need 8192, free 7968)
```

**Root cause.** With allocation restored to 1:1, the SWA tier charges one slot per
*raw* token, but the mixed extend/decode paths sized every eviction request in
*pooled* units — `pooled_need = total_quant_alloc + total_hp_prefix_alloc`
(`common.py:489`), a compressed count, and a retry loop that checked only
`free_quant_slots`. `evict_from_tree_cache` derives both tiers' requests from that
one number, so the SWA tier was asked to free a few hundred tokens when it needed
8,192. Nothing else fills the gap: `maybe_evict_swa`'s extend branch fires only for
`is_chunk_cache()`, and gpt-oss runs `SWARadixCache`.

The symptom that identifies it: the serve log reports `swa token usage: 0.01`
while the allocator has 7,968 of 1,048,576 slots free. The tree's ~1.04M evictable
SWA tokens were real and reclaimable — no one requested them.

**Fix.** `evict_swa_for(tree_cache, swa_allocator, need)` in `common.py`, called
immediately before `alloc_swa_for` on both the extend and decode paths, sized in
raw tokens. It issues `evict(EvictParams(num_tokens=0, swa_num_tokens=short))`, so
`SWARadixCache` takes only its SWA branch (`swa_radix_cache.py:610`): internal
nodes are tombstoned via `free_swa` and the full-tier prefix stays cached. Bounded
to 4 passes, matching the adjacent quant-tier loop.

**Verification:** batch ladder rerun on the three quant arms
(`configs/gptoss20b_batch_curve_quant.json`) — pending.

### Still open: SWA accounting drifts under a radix tree (mixed-KV)

**Symptom.** `available + evictable > capacity` on the SWA tier, caught by the
invariant assertion. Requires a radix tree: the radix-cache-OFF arm never trips it
(a `ChunkCache` owns nothing), the radix-cache-ON arm always does.

**Measured overshoot by code state** (MATH-500, vq2, gpt-oss-20b, n=100):

| state | overshoot | accuracy |
|---|---|---|
| flush-path `free_swa` | 32 | 0.58 (contaminated run) |
| release-on-reuse in `alloc_swa_for` | 11 | — (crashed) |
| + `move_swa` (src→dst re-key) | **218** | — (crashed) |
| radix cache OFF (any of the above) | 0 | **0.91**, answered 75 |

**Established.** The 1:1 SWA allocation is NOT the cause: radix-off reproduces the
reference exactly (0.91 / answered 75 vs 0.91 / 75). The fault is confined to the
tree/eviction interaction. bf16 is unaffected (0.93, does not use the mixed path).

**Cause 1 (fixed).** `free_swa` may only be called from `SWARadixCache.evict`,
which decrements `swa_evictable_size_` in the same step. The flush path called it
directly, so the allocator reclaimed slots the tree still counted. Replaced with
release-on-reuse inside `alloc_swa_for`. Overshoot 32 -> 11: real but partial.

**Cause 2 (NOT the mapping re-key — that hypothesis is refuted).** Re-keying a
flushed token's SWA mapping from its HP slot to its quant slot made the overshoot
*worse* (11 -> 218). Reason: quant slots are pooled (`N_Q=8`), so several flushed
tokens share one `dst_quant_slots` id. `mapping[dst] = mapping[src]` is then
last-writer-wins, orphaning the other SWA slots, and `free_swa(node.value)` later
frees the same SWA index repeatedly — inflating the free list past capacity. The
change is reverted; do not retry it without deduplicating `dst`.

**The unresolved residual (overshoot ~11)** is a per-token/per-slot unit mismatch:
the tree accounts SWA in TOKENS, the mapping is keyed by full-tier SLOT id, and the
mixed path's quant tier packs `N_Q` tokens per slot. Any fix must reconcile those
units rather than move mappings around. Next diagnostic: instrument
`swa_evictable_size_` against a direct count of non-zero `full_to_swa_index_mapping`
entries per tree node, to find which operation diverges first.

### ROOT CAUSE FOUND: the tree over-counts; the allocator was never wrong

Instrumented the assertion to recount from allocator state at the moment of
failure (MATH-500, vq2, radix cache on):

```
free_list=3280618  unique=3280618  dups=0
mapped_held=790    held+avail=3281408  capacity=3281408  delta=0
tree_evictable=800
```

**`delta=0`**: every SWA slot is either in the free list or reachable through
`full_to_swa_index_mapping`. No double-free (`dups=0`), no leak, no orphan. The
allocator is exactly self-consistent.

**`tree_evictable=800` > `mapped_held=790`**: the radix tree claims 800 evictable
SWA *tokens* while only 790 slots exist in total. The 10-token gap is precisely the
reported overshoot. **The tree counts tokens that own no SWA slot.**

This refutes every earlier hypothesis, all of which targeted the allocator side:
flush-path `free_swa` (partial, real, but not the cause), and the `move_swa`
re-key (wrong, reverted). It also invalidates ranking the three states by overshoot
magnitude: that number is a snapshot of when the check ran, not a severity measure.

**Why the tree over-counts.** Upstream keeps this consistent with a per-request
watermark, not with paired free/decrement: `_evict_swa` frees SWA below
`seqlen - window - page_size` and advances `req.swa_evicted_seqlen`; the tree learns
of it at insert, where `_insert_helper` marks that region `swa_tombstone=True` and
`swa_evictable_size_ += len(value)` runs **only `if not swa_tombstone`**
(`swa_radix_cache.py:1071`). Tokens whose SWA is gone are simply never counted.

In the mixed-KV path that watermark does not describe reality. SWA layers only ever
read the last `sliding_window=128` tokens, and the HP-recent ring (256) always
covers them, so **every flushed/tree-cached token legitimately has no SWA KV and
never needs any** -- but the tree still counts them as SWA-evictable.

**The fix is to correct the watermark, not the allocator.** Maintain
`req.swa_evicted_seqlen` in the mixed path so it reflects what is actually true --
everything outside the HP-recent window is SWA-evicted -- and the existing
tombstone machinery then does the rest: nodes insert tombstoned, `dec_lock_ref`
never re-adds them (it asserts `not swa_tombstone`), and the invariant holds by
construction. No new accounting path, and no divergence from upstream: tombstone is
upstream's own representation for "this node's SWA is gone".

**This also reverses the SWA sizing conclusion.** With cached tokens correctly
holding no SWA, the tier only ever needs the live windows, so the ~33-request
ceiling on warm-cached prefixes disappears and `evict_swa_for` becomes unnecessary.
The pre-existing HP-only allocation was closer to right than it looked; its actual
defect was never telling the tree, which is what the original leak was.

### CORRECTION: the watermark is maintained; SWARadixCache had no mixed-KV awareness at all

The section above is **wrong about the cause and wrong about the fix**. Two
premises fail on inspection:

1. *"The mixed path never advances `req.swa_evicted_seqlen`."* It does.
   `alloc_for_decode` calls `batch.maybe_evict_swa()` at `common.py:1068`, **before**
   the mixed dispatch at `:1070`, and its decode branch fires every
   `sliding_window_size` steps (`schedule_batch.py:2519`). The watermark is
   maintained on the mixed path exactly as upstream intends.
2. *"The tree is told via the watermark."* Only for the request that owns it.
   `_evict_swa` floors the watermark at `req.cache_protected_len`
   (`schedule_batch.py:2545`), which is precisely what stops a request from freeing
   SWA the tree owns. That guard is per-request and cannot see a **different**
   request's stale node — which is the actual failure.

**Actual root cause.** `SWARadixCache` has zero mixed-KV awareness --
`grep -n "mixed" swa_radix_cache.py` returns nothing. `RadixCache` carries a full
apparatus for it, built because mixed-KV slot ids are not tree-safe:
`_mixed_kv_tail_to_drop` (`radix_cache.py:923`, comment: *"HP-recent slot ids are
per-request and must not enter the tree"*), `_mixed_kv_slack_insert_limit`,
`_with_mixed_quant_slack`, the `match_prefix` cap, and a `cache_finished_req`
early-return that never extends the tree. gpt-oss is hybrid-SWA so
`scheduler.py:841-844` routes it to `SWARadixCache`, which had none of that.

The unsafe ids are the HP-recent ring: `unified_kv_allocator.py:165-173` computes
`req_pool_idx * hp_recent_ring_size + _hp_recent_offset`, so those ids are
**deterministic per req_pool_idx** and are re-issued to the next request that takes
the slot. Chain: A finishes -> `cache_finished_req` inserts A's HP-recent tail into
the tree -> A's req slot is released (`req_to_token` not cleared) -> B takes the same
`req_pool_idx` and is handed the same ids -> B's `_evict_swa` frees their SWA (its
`cache_protected_len` floor protects B's tree region, not A's node) -> mapping
zeroed while the tree still counts them -> `available + evictable > capacity`.

Consistent with the measurement above: allocator exactly self-consistent
(`delta=0`, `dups=0`), tree over-counting by 10.

**Second consequence, worse than the assertion:** A's node aliases B's *live* KV, so
a later request matching A's prefix reads B's data. This is the failure class
`radix_cache.py:585-598` documents.

**Fix.** Port the apparatus: five `mixed_kv_*` helpers extracted to `radix_cache.py`
module scope (`RadixCache`/`ChunkCache` methods become delegators, removing a
pre-existing duplicate), then detection + match cap + `cache_finished_req`
early-return + `cache_unfinished_req` trims added to `SWARadixCache`, every branch
gated on `_mixed_kv_enabled` so the stock hybrid-SWA path is untouched.

Two things deliberately **not** ported, both verified against the source:
- `radix_cache.py`'s post-insert dup free. SWA's `_insert_helper` already frees
  overlaps internally (`:981`, `:991`, `:1002`, `:1005`); `RadixCache`'s does not.
  Porting it is a straight double free. Clamping the *insert key* bounds the frees
  instead, since `_insert_helper` only walks as far as the key it is given.
- `assert full_match_len >= new_prefix_len`. False under SWA: `_match_prefix_helper`
  truncates at `best_value_len` when a tombstone leaves fewer than
  `sliding_window_size` non-tombstone tokens behind it.

Also: `dec_lock_ref` needs `DecLockRefParams(swa_uuid_for_lock=...)` here, unlike
`RadixCache`'s bare call, which would unlock SWA to the root and trip
`swa_lock_ref > 0`.

**Correction to the sizing claim above:** it does *not* follow that the SWA tier only
needs live windows. Tree-cached tokens below `cache_protected_len` do own SWA slots
and are released by `SWARadixCache.evict`, so `evict_swa_for` remains necessary.

**Evidence hygiene.** The 0.58-vs-0.91 MATH-500 gap quoted earlier as proof of silent
corruption is **not** a clean A/B: the radix-on server crashed mid-run on this very
assertion (`logs/math500_swafix.log`: `Connection refused` on rids 84/96/99) and the
recorded 0.58 resumed 97 of 100 rows from that crashed run. The low `answered=54`
partly reflects a dead server. A clean fresh A/B is what establishes the accuracy
impact; the structural aliasing argument above does not depend on it.

## KNOWN ISSUE (not fixed, out of scope): `self_check_during_busy` cannot pass on a mixed-KV pool

Found while trying to use `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY=2` as the
carrier for a temporary diagnostic. It aborts on the **first scheduler pass**, with
an empty tree and before any request has run:

```
[Mem Check (BUSY)] available_size=608184, evictable_size=0, protected_size=0, uncached_size=8
AssertionError: Mem Leak Detected! total_tokens=608192 vs self.max_total_num_tokens=600000
```

**Root cause: the two sides count different tiers.** From the same boot's log --
`num_quant_pages=75001`, `N_Q=8`, `hp_prefix_pool_tokens=8192`:

```
quant tier            75001 * 8 = 600008
+ HP-prefix arena               =   8192
- page 0 (reserved, free_pages = arange(1, num_quant_pages))
                                =     -8
                          total = 608192   == available_size + uncached_size
max_total_num_tokens            = 600000   (quant tier only)
```

`available_size()` on `UnifiedInt2HPKVAllocator` pools quant + HP-prefix slots,
while `max_total_num_tokens` is the profiled quant budget; the HP arena is
allocated *on top of* it by design (`unified_kv_pool.py` HP arena reservation is
not in `cell_size`). So `available + evictable + protected + uncached` structurally
exceeds `max_total_num_tokens` by the HP-prefix pool size, always, on every
mixed-KV boot.

**Why it was never seen:** `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY` is
`EnvInt(0)` (`environ.py:335`) and has zero occurrences in any prior log in this
repo. The check has simply never been exercised on this stack.

**Not fixed here** -- it is orthogonal to the SWARadixCache work and touching
pool-accounting semantics mid-campaign would invalidate the throughput baselines.
Recorded so the next person to enable the flag does not mistake it for a
regression. The fix, when wanted, is to compare against the allocator's own
capacity (quant + HP-prefix) rather than `max_total_num_tokens`, or to exclude the
HP-prefix arena from `available_size()`.

**Consequence for diagnostics:** anything needing a per-batch hook on mixed-KV must
use its own flag rather than riding on this one.

### FURTHER CORRECTION: there are TWO independent defects, not one

The section above ("SWARadixCache had no mixed-KV awareness") identifies a real
defect and its fix is necessary -- but it is **not sufficient**, and the claim that
it explains the SWA accounting assertion is **wrong**. Measured directly.

**The experiment.** With the HP-recent trim in place and `cache_finished_req`
inserting the trimmed tail, on gpt-oss / 30k / bs1:

```
[HP-recent probe] scanned 6 nodes / 30760 tokens, 0 in HP-recent range
AssertionError: SWA accounting invariant violated:
  available + evictable (1018696 + 30760) exceeds capacity (1048576) by 880
```

Zero HP-recent slot ids in the tree, and the accounting still breaks:
held = 1048576 - 1018696 = 29880 slots; the tree claims 30760 tokens; gap = 880.
Aliasing cannot explain a gap that appears when no aliased id is present.

**Defect (a) -- correctness.** HP-recent slot ids are per-`req_pool_idx` and are
re-issued to the next occupant, so a tree node holding one aliases another
request's live KV. Fixed by `mixed_kv_tail_to_drop` on both insert paths; verified
by the DFS probe over ~1M tree tokens with zero hits.

**Defect (b) -- accounting.** The tree counts one SWA token per tree token, but in
the mixed path a flushed quant-tier token owns **no SWA slot at all** -- SWA layers
read only the last `sliding_window` tokens and the HP-recent ring always covers
them. So `swa_evictable_size_` counts SWA that was never allocated. This is
independent of (a) and is NOT fixed by the trim.

**Why (b) hid for so long.** The `cache_finished_req` early-return kept the tree
small enough to stay under capacity by margin, not by correctness: the ladder run
reached 951,552 tree tokens against a 1,048,576 SWA capacity without tripping.
Inserting the trailing partial chunk pushed it over. Every "0 assertions" result
from the early-return build is therefore evidence about (a), not about (b).

**Consequence: the earlier watermark/tombstone analysis was half right.** Passing
`swa_evicted_seqlen` so `_insert_helper` tombstones the region that owns no SWA is
exactly the mechanism (b) needs (`swa_radix_cache.py:1069-1071` -- the
`if not swa_tombstone` guard). It was dismissed on the grounds that the watermark
is already maintained; that is true and irrelevant, because the watermark being
correct does not make the tree's per-token SWA assumption correct.

**Neither current variant is shippable:**
- `cache_finished_req` early-return: tree stays small, (b) does not trip, but the
  final partial chunk never reaches the tree -- warm-pass TTFT ratio 0.068 -> 0.252
  at 30k/bs1, failing the harness `warm_took` gate (a timing gate, not a
  cached-token threshold).
- `cache_finished_req` trimmed insert: reuse restored, but (b) trips at 880 over.

The complete fix needs both halves: the HP-recent trim for (a), and the
tombstone/watermark path for (b).

### GATE 5 RESULT: the HP-recent fix does NOT restore radix-on accuracy

Clean A/B, same 100 rows x K=4, gpt-oss-20b vq2, identical except RADIX_CACHE.
Run on the early-return build -- the one with ZERO assertions and a DFS probe
confirming ZERO HP-recent ids across ~1M tree tokens.

```
radix OFF   acc_mean=0.9225  per_k=[0.92,0.93,0.91,0.93]  answered=[73,82,74,76]  mean_tok=1669
radix ON    acc_mean=0.3975  per_k=[0.41,0.41,0.45,0.32]  answered=[57,61,53,41]  mean_tok=3407
delta = -0.525
```

`mean_completion_tokens` doubling (1669 -> 3407) is the tell: generations stop
terminating. That is corrupted attention, not sampling variance.

**Two conclusions, both against my earlier claims.**

1. **Defect (a) (HP-recent aliasing) is NOT the accuracy driver.** It is a real
   defect and the trim is worth keeping, but this run had zero aliased ids in the
   tree by direct measurement and still lost 53 accuracy points. The earlier
   framing -- "the radix cache hands requests other requests' KV, and that is the
   vq2 accuracy regression" -- is not supported.

2. **The SWA assertion is not a usable proxy for corruption.** Zero assertions
   fired across a run whose accuracy collapsed. Any past "0 assertions -> healthy"
   inference in this campaign is unsound, including several of mine today.

**Prime suspect is defect (b), and not merely as bookkeeping.** `SWARadixCache.evict`
drives SWA eviction off `swa_evictable_size_` (the swa branch, :610-651). If the
tree believes it holds SWA that was never allocated, eviction is mis-driven and
`free_swa` releases slots live requests still need -- corrupting precisely the
sliding-window layers, which is the observed failure. So (b) plausibly causes both
the assertion and the accuracy loss, and (a) causes neither.

Consistent with the 25-row high-churn run on the same build: acc_mean 0.54.

**Operational consequence: RADIX_CACHE=0 remains the only safe configuration for
gpt-oss + quantized KV.** Unchanged by this work. No throughput or accuracy result
should be produced with radix on until (b) is fixed.

---

### ROOT CAUSE OF DEFECT (b), PROVEN: the flush demotes a token but not its SWA slot

Defect (b) is not in `SWARadixCache` at all. Both earlier localisations were wrong.

`_alloc_for_decode_mixed` (`mem_cache/common.py`) pairs an SWA slot to the
**HP-recent** slot it allocates each decode step (`alloc_swa_for(out_cache_loc)`).
The flush then demotes tokens HP-recent -> quant: `gpu_flush_int2_apply` rewrites
`req_to_token` to `dst_quant_slots`, and `allocator.free(plan.returned_slot_ids)`
returns the vacated HP ids. Nothing moves the SWA slot. `grep -rn move_swa`
returned nothing -- the transfer never existed.

So the token's SWA stays stranded on the vacated HP id, while the quant id -- the
one `cache_finished_req` inserts into the radix tree -- owns no SWA slot.

Measured at the flush site (`full_to_swa_index_mapping[dst_quant_slots] == 0`):

```
[flush dst] 8 quant slots, 8 unmapped     <- 100%, every step
[flush dst] 32 quant slots, 32 unmapped
```

and at the tree node built from them:

```
[SWA phantom @_add_new_node] node=9 len=152 unbacked=152 |
  update_kv_after_len=64 swa_evicted_seqlen=104 total_prefix=72 keylen=184 |
  first_vals=[320..327]        <- exactly the flush-destination range
```

The tombstone split was already correct (32 tombstoned at watermark 104); all 152
tokens *above* the watermark were unbacked. That is why the residual over-count
survived the tail-reserve fix, and why `delta=0` in every RECOUNT: the stranded HP
mappings inflate `mapped_held` by exactly what the quant ids lack, so the allocator
looks self-consistent while the tree counts slots that do not exist.

This also explains the accuracy collapse without invoking mis-driven eviction:
`match_prefix` serves those nodes as SWA-valid, so the next request's sliding
layers read `full_to_swa_index_mapping == 0` -> slot 0 -> garbage. `RADIX_CACHE=0`
is unaffected because a ChunkCache never shares a prefix; that is the whole
radix-on/radix-off asymmetry.

It subsumes the open item in *"Hybrid-SWA: quant-tier tokens leak their SWA slots"*
above: the leak there is the same stranded mapping seen from the allocator side.
The fix proposed in that entry -- stop allocating SWA for the quant middle -- would
have made the phantom permanent rather than fixing it.

**Fix**, three parts:

1. `swa_memory_pool.py: SWATokenToKVPoolAllocator.move_swa(src, dst)` -- retarget
   the mapping, releasing any SWA still held by a destination id (quant slots
   recycle through the full tier's free list, which does not clear the mapping).
   Pure relabel: no alloc, no net occupancy change.
2. `common.py` flush path -- call it with `plan.returned_slot_ids[valid]` ->
   `plan.dst_quant_slots[valid]`, before the `allocator.free`. Invalid rows carry
   `src == dst` and must be excluded or the relabel zeroes a live mapping.
3. `schedule_batch.py: _evict_swa` -- subtract `tree_cache.swa_evict_tail_reserve()`
   so the eviction frontier stays below the tree's insert boundary. Under mixed-KV
   that boundary is `page_floor(seq_len) - tail_trim`, not `page_floor(seq_len)`,
   so the stock margin left the frontier *above* it. Plus `_insert_helper` case 3
   `==` -> `>=`: a shortened insert key can push the watermark past the key, and
   the equality test then fell through to a non-tombstone `_add_new_node`.

Parts 1 and 3 are independent defects; 3 alone cut the over-count 880 -> 144.

**Verification (structural): PASSED.** 148,516 flush events, every one reporting
0 still-unmapped destinations (100% unmapped before). Tree reached 222,056 tokens
/ 156 nodes with 0 phantom-SWA, 0 HP-recent ids, 0 assertions of any kind; the
pre-fix build died at a 152-token tree.

**Verification (accuracy): FAILED. The phantom-SWA defect is NOT the accuracy
driver.** Clean MATH-500 A/B, 100 rows x K=4, identical except RADIX_CACHE, no
crash on either arm (`assertions=0 sigquit=0 conn_refused=0`):

```
ON  (radix, move_swa)   acc_mean=0.4100  per_k=[0.43,0.37,0.41,0.43]  mean_tok=2814
OFF (control)           acc_mean=0.8975  per_k=[0.90,0.90,0.88,0.91]  mean_tok=1615
pre-fix ON              acc_mean=0.3975                               mean_tok=3407
```

0.3975 -> 0.4100 is noise. So the three fixes above are correct and necessary --
they are proven by direct measurement of the invariant they target -- but a
FOURTH, independent defect owns the radix-on accuracy loss. `mean_completion_tokens`
is still ~1.7x the control, i.e. generations still fail to terminate.

Method note, third time in this campaign: a structural invariant that is proven
to hold still did not predict accuracy. Do not accept "the invariant now holds"
as evidence that the accuracy bug is fixed; only the A/B is.

Next step is a deterministic bisect (temperature 0, same prompt twice against one
server, so the second call is served from the tree) sweeping prompt length,
because the mixed-KV match cap is `max(hp_prefix, n-263)` -- short prompts reuse
only the 64-token HP-prefix, long ones reuse deep into the quant tier. K=4 at
temperature 1.0 can say "worse" but not "where".

**Operational status unchanged: RADIX_CACHE=0 remains the only safe configuration
for gpt-oss + quantized KV.**
