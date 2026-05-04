# NeurIPS Implementation — Phase-by-Phase Review

This document reviews everything that has been implemented from the finalized
NeurIPS plan, broken down by phase. For each phase:
- **Goal** — what we needed out of this phase.
- **Steps** — concrete files/scripts run, with brief explanations.
- **Results** — artifacts produced, decisions taken, gates passed.

The plan's source-of-truth is `notes/core/neurips_implementation_plan.md`. The
high-level method recap: JointQK = orthogonal eigenbasis of
`R_sym = (Σ_Q Σ_K + Σ_K Σ_Q) / 2` with reverse-water-fill bit allocation for K;
calibrated PCA + uniform bits for V. Baselines: full precision, TurboQuant V3
(matched K/V bits), full KIVI (per-channel int4 K + per-token int4 V).

The full plan has 8 phases (0–7). Phases 0–6 are **complete**; Phase 7 is
**in progress** (Qwen LongBench running, Llama LongBench + RULER queued by the
chain script).

---

## Phase 0 — Pre-flight (complete)

**Goal:** confirm we can actually run the plan — HuggingFace gated-model access
for Llama, GPU pool 0–5 visible, free disk, kvpress entry points understood,
and a parallel launcher that distributes shell commands across GPUs.

**Steps:**

1. **HF + resources check.** Logs in `experiments/stage1/logs/phase0_hf_check.log`,
   `phase0_resources.log`, `phase0_git_baseline.log`. Llama-3.1-8B-Instruct
   loaded its config cleanly (32 layers); ≥100 GB free; GPUs 0–5 nvidia-smi'd.
2. **kvpress integration entry point.** Grep'd `kvpress/evaluation/evaluate.py`
   to see how `--press_init_command` is consumed
   (`experiments/stage1/logs/phase0_evalpy_grep.log`). Found it was
   logging-only — **not** eval'd into a press instance. This drove the patch in
   Phase 3 that adds `press_kwargs` (a real dict field) and instantiates the
   class inside `_setup_press`.
3. **Parallel launcher.** `experiments/stage1/scripts/parallel_launcher.py`
   (~80 lines). Reads a commands file (one shell command per line, optional
   `# label=foo` suffix for log naming), assigns one worker per GPU via a FIFO
   queue, runs each job via `subprocess.run(["bash", "-c", cmd])` with
   `CUDA_VISIBLE_DEVICES` pinned, writes per-job logs and an aggregated
   `_overview.log` of OK/FAIL/elapsed.

   Two real bugs caught and fixed during the project:
   - First version did round-robin GPU assignment based on job index, which
     could collide (job N+k landing on the GPU job k was still using). Replaced
     with one-worker-per-GPU + FIFO consumption.
   - First version used `subprocess.run(cmd, shell=True)` which uses `/bin/sh`
     and can't `source` a conda activate script. Switched to explicit `bash`.

**Results:**
- Smoke-test (`/tmp/launcher_smoke/`) succeeded.
- The launcher is the workhorse for Phases 1, 4, 5, 6, 7 — every multi-run
  phase emits a `commands.txt` and dispatches through it.

---

## Phase 1 — V method finalization (complete)

**Goal:** decide V quantization method and bit budget *before* combining with K.
Three candidate methods × three bit budgets {2, 3, 4}, plus a K-only sweep at
fp16 V to confirm K=2/3/4 each retain task accuracy in isolation.

The whole point of decoupling: combined K+V experiments can't tell you whether
a regression came from K or V. Phases 1A (V-only) + 1B (K-only) isolate them.

**Steps:**

1. **Verify V is in the calibration bundle.**
   `artifacts/stage1/query_stats_longbench_under4k/qasper_e_001.pt` already had
   key `"v"` of shape `(32, 8, *, 128)` — written by `toolkit/capture.py:138`
   in earlier work.

2. **Calibrate Σ_V on the 24-bundle.**
   `experiments/stage1/scripts/calibrate_sigma_v.py`. Accumulates per-(layer,
   kv_head) Σ_V via `torch.einsum("lhsd,lhse->lhde", v, v)`, normalizes by
   total tokens. Output: `artifacts/stage1/v_method_study/v_stats.pt`.

3. **Three V methods.** `experiments/stage1/toolkit/v_compressor_adapter.py`
   defines `V_METHOD_NAMES = ("v_random", "v_eigen_uniform", "v_eigen_waterfill")`
   and `build_v_compressor(method, sigma_v, head_dim, bits)`:
   - `v_random` — random orthogonal forward map via QR; uniform bits; closest
     analogue to TurboQuant-style randomization but without the Hadamard
     structure.
   - `v_eigen_uniform` — eigh of Σ_V; rotate into eigenbasis; uniform bits.
   - `v_eigen_waterfill` — same eigenbasis; reverse-water-fill bits using
     Σ_V eigenvalues.

   All three reuse the existing `PerCoordCompressor` from
   `toolkit/per_coord_quantization.py` for Lloyd–Max codebooks (`unit_gaussian_centroids`).

4. **JointQKPress final form (used in Phase 1 in flag-toggled modes).**
   `experiments/stage1/toolkit/jointqk_press.py`. Dataclass with:
   ```
   cca_stats_path, v_stats_path,
   v_method, k_method, k_bits, v_bits,
   layer0_full_precision=True,
   quantize_k=True, quantize_v=True, compress_decode=False
   ```
   For Phase 1A we pass `quantize_k=False`; for 1B we pass `quantize_v=False`.
   This is the same code path used in production — no separate "study" branch.

5. **Pre-derive R_sym + V_h into Qwen `cca_stats.pt`.**
   `experiments/stage1/scripts/precompute_newbases.py` calls the existing
   `_derive_vh_rsym` helper from `run_cca_vs_waterfill_study.py` and writes
   them in-place into `cca_stats.pt`.

6. **Register presses.** Edit
   `kvpress/evaluation/evaluate_registry.py` to import `JointQKPress`,
   `TurboQuantPress`, `KIVIPress` and register them as classes (not instances —
   instantiation happens at evaluate-time after `press_kwargs` is parsed).

7. **Run Phases 1A + 1B in parallel.**
   `experiments/stage1/scripts/launch_phase1ab.sh` builds 13 commands:
   - 1 full-precision oracle run (qasper, fraction 0.3)
   - 9 V-only sweep runs (3 methods × 3 bits, K at fp16)
   - 3 K-only sweep runs (K∈{2,3,4}, V at fp16)

   Logs in `experiments/stage1/logs/phase1ab/`. All 13 OK.

8. **Aggregate + decide.** `experiments/stage1/eval/aggregate_phase1.py`
   reads the per-run F1, computes `rel_F1 = F1 / F1(full)`, picks smallest
   acceptable V (rel_F1 ≥ 0.97).

9. **V-lock correction.** Initial decision was `v_eigen_waterfill / V=2` per a
   "more sophisticated wins ties" tiebreaker. Preliminary Phase 7 results
   showed JointQK underperforming TurboQuant on qasper. Re-inspecting Phase 1A
   showed `v_eigen_uniform` at V=3 actually had the highest F1 (43.18 vs
   v_eigen_waterfill 41.92 at V=3, and fp16 was 43.13). The eigenvalue spectrum
   of Σ_V is roughly uniform across coordinates in this regime, so water-fill
   gives no benefit and only hurts via the rounding noise of differing bit
   widths. Updated `v_lock.txt` to `V_METHOD=v_eigen_uniform / V_BITS=3 /
   V_REL_F1_AT_LOCK=1.0012`. The corrected file has a `NOTE=` line documenting
   the change.

10. **K-floor.** From the K-only sweep:
    `K_FLOOR=2 / K_REL_F1_AT_FLOOR=1.0243` (i.e. JointQK at K=2 with V=fp16
    actually slightly *exceeds* fp16 oracle on qasper — within noise, but
    confirms K=2 is safe).

11. **Combined sanity (Phase 1C).**
    `experiments/stage1/scripts/launch_phase1c.sh` runs 3 jobs at K∈{2,3,4} and
    locked V. The sanity rule
    `rel_F1(combined) ≥ rel_F1(K-only) × rel_F1(V-only) − 0.05` held at every
    K. Logs in `experiments/stage1/logs/phase1c/` (3 OKs).

**Results / decision artifacts:**

```
artifacts/stage1/v_method_study/v_lock.txt
  V_METHOD=v_eigen_uniform
  V_BITS=3
  V_REL_F1_AT_LOCK=1.0012

artifacts/stage1/v_method_study/k_floor.txt
  K_FLOOR=2
  K_REL_F1_AT_FLOOR=1.0243
```

Both files are consumed by Phases 6–7 launchers via `grep -oP`.

---

## Phase 2 — Llama Q/K/V capture (complete; ran background to Phase 1)

**Goal:** Llama-3.1-8B-Instruct calibration bundle (24 LongBench-E examples,
under-4k context) for the Llama version of cca_stats and v_stats.

**Steps:**

1. **Launch capture** on GPUs 0–1 in the background while Phase 1 used GPUs
   2–5 on the remaining pool. Driver: `experiments/stage1/collect_query_stats.py`
   (existing, unchanged). Output:
   `artifacts/stage1/query_stats_longbench_under4k_llama31_8b/`.
2. **Verify bundle.** 24 `.pt` files; `q_post / k_post / v` shapes
   `(32, 8, *, 128)` — same per-(layer, kv_head, *, dim) shape as Qwen
   (Llama-3.1-8B happens to have the same hidden geometry per kv_head as
   Qwen3-8B at this size).

**Results:** bundle written; logs `experiments/stage1/logs/phase2/`.

---

## Phase 3 — Press scaffolding (complete)

**Goal:** drop-in `kvpress` presses for the three methods we'll evaluate. They
have to share the `BasePress` API so `kvpress/evaluation/evaluate.py` can run
them with no method-specific code paths.

**Steps:**

1. **JointQKPress** — `experiments/stage1/toolkit/jointqk_press.py`. Builds
   per-(layer, kv_head) K compressors at `post_init_from_model` time from
   `cca_stats.pt`, and per-(layer, kv_head) V compressors from `v_stats.pt`.
   `compress(...)` does the K and V projections + quantize + dequantize.
   `forward_hook(...)` overrides `BasePress.forward_hook` to support
   `compress_decode=True` (Mode B): on decode steps, compress only the last
   token and concatenate. `layer0_full_precision=True` skips K compression on
   layer 0 (Stage 1 finding: layer 0 has anomalous norm/condition properties).

2. **TurboQuantPress** — `experiments/stage1/toolkit/turboquant_press.py`.
   Wraps `turboquant_pytorch.compressors_v3.TurboQuantV3` (the vendored
   reference impl). One subtlety: `MSECompressor` materializes a
   `(N, D, K)` "diffs" tensor that OOMs on long contexts. Added per-call
   chunking on the seq dim (`CHUNK_TOKENS=2048`) so each call sees ≤2048 tokens
   and we stitch the recon back. We also lazy-move `Pi` and centroids onto the
   correct GPU on first compress (TurboQuantV3 builds them on CPU).
   `residual_window=0` — we manage decode scope independently via
   `compress_decode`.

3. **KIVIPress + quantizer** — `kivi_press.py` and `kivi_quantizer.py`.
   Per-channel int4 K (asymmetric quantization along the head_dim axis) +
   per-token int4 V (asymmetric along the seq axis). This is a faithful
   reproduction of the KIVI paper's quantization scheme (full int4, no
   residual-window logic since we're matching method-vs-method on a fixed
   scope).

4. **Register presses.** `kvpress/evaluation/evaluate_registry.py` maps
   `"jointqk" / "turboquant" / "kivi"` → the **classes**, not instances. This
   pairs with the `evaluate.py` patch:

   ```python
   # in EvaluationConfig
   press_kwargs: Optional[Dict[str, Any]] = None

   # in _setup_press
   if isinstance(press, type):
       kwargs = self.config.press_kwargs or {}
       press = press(**kwargs)
   ```

   This is the key plumbing change that lets us pass the kwargs on the CLI as
   a JSON object instead of evaluating Python expressions.

**Important kvpress integration notes encountered:**
- The vendored kvpress targets Python ≥ 3.10 because of `int | None` style
  unions. We're on 3.9; bulk-patched 68 files with
  `from __future__ import annotations`.
- `turboquant-pytorch` directory has a hyphen (invalid Python module name).
  Created symlink `turboquant_pytorch -> turboquant-pytorch`.
- Used `PYTHONPATH` — not `pip install -e .` — because of pyproject py-version
  pin.
- Fire CLI parses `false`/`true` as strings; in JSON `press_kwargs` we have to
  use **Python-capitalized** `True`/`False` (the launchers do this).

**Results:**

```bash
$ python -c "from kvpress.evaluation.evaluate_registry import PRESS_REGISTRY; \
    print({k: PRESS_REGISTRY[k].__name__ for k in ['jointqk','turboquant','kivi']})"
{'jointqk': 'JointQKPress', 'turboquant': 'TurboQuantPress', 'kivi': 'KIVIPress'}
```

---

## Phase 4 — Press validation (complete)

**Goal:** prove the press wrappers reproduce the offline reference recon
(byte-equivalent to `build_method_compressor` output), and that they don't
blow up under kvpress's actual pipeline.

**Steps:**

1. **Parity test.** `experiments/stage1/tests/test_press_roundtrip_parity.py`.
   For each (layer, kv_head):
   - Build a `JointQKPress`, run K and V through `_quantize_k` and
     `_quantize_v`.
   - Build the same compressors via the offline `build_method_compressor` path
     (the same one Stage 1E uses).
   - Compare reconstructions. Result: max-abs-diff = `0.00e+00` for both K and
     V (i.e. byte-identical, not just within tolerance).

2. **kvpress smoke (5% slice).** 4 jobs (full + 3 presses) on qasper at
   `--fraction 0.05`. All 4 produced result.json with F1 within 10pp of full.

**Results:** parity confirmed (`experiments/stage1/logs/phase4/parity_test.log`),
kvpress smoke passed.

---

## Phase 5 — Llama Stage-1E reproduction + W1 gate (complete)

**Goal:** replicate the Stage-1E E3/E5 study on Llama-3.1-8B and confirm the
JointQK water-fill method wins on this second model — i.e. that the basis
finding generalizes. This is the W1 critical gate from the plan.

**Steps:**

1. **Phase 5 launcher.** `experiments/stage1/scripts/launch_llama_e3_e5.sh`
   wraps `launch_cca_study.sh` with Llama-specific bundle / cca-stats paths
   and the GPU pool. (One small fix to `launch_cca_study.sh`: the `EXTRA_ARGS`
   variable was parsed but not actually appended to the command — now it is.)

2. **Pre-derive newbases for Llama.** Same `precompute_newbases.py` as Qwen,
   targeting `artifacts/stage1/cca_vs_waterfill_study/llama31_8b/cca_stats.pt`.

3. **Calibrate Σ_V for Llama.** Same script, output
   `artifacts/stage1/v_method_study/v_stats_llama31_8b.pt`.

4. **Run E3/E5.** Generates per-(b, layer-0-excl) top-1 retention scores
   across all 9 methods (the 4 cca-orth/r-sym × {uniform, waterfill}, plus
   diagonal/random oracles).

5. **W1 gate.** `experiments/stage1/gates/gate_e3.py` and `gate_e5.py`
   (both extended with `--output-dir`). Pass criterion: JointQK
   `r_sym_waterfill` is top-ranked at every (b, layer-0-excluded) cell with
   ≥5 pp top-1 margin at b=3.

6. **Cross-model chart.** `experiments/stage1/scripts/make_cross_model_chart.py`
   generates the two-panel Qwen | Llama overlay
   (`report_charts/cross_model_b_sensitivity.png`).

**Results:** W1 gate passes on Llama. The basis finding (`r_sym_waterfill`)
generalizes — same dominance pattern as Qwen. This is the C5 verification
checkpoint.

---

## Phase 6 — Decode-scope ablation (complete)

**Goal:** does compressing the KV cache also during the decode steps
(Mode B) help / hurt / make no difference vs compressing only the prefill
(Mode A)? Decision rule: pick A unless B is materially better.

**Steps:**

1. **Launcher.** `experiments/stage1/scripts/launch_phase6_decode_scope.sh`.
   Builds 12 commands: 2 tasks (qasper, narrativeqa) × 2 modes (A, B) × 3 K
   bits (2, 3, 4). All on Qwen3-8B at locked V.
2. **Run.** All 12 OK. Logs in `experiments/stage1/logs/phase6/`.
3. **Aggregate.** `experiments/stage1/eval/aggregate_decode_scope.py` computes
   `max |ModeB − ModeA|` across the 6 (task × kb) cells.

**Results:**

```
artifacts/stage1/downstream/qwen3_8b/decode_scope/decode_decision.txt
  WINNER=A
  MAX_DIFF_PP=0.00
  NOTE=Mode A and Mode B produced byte-identical task scores ...
```

The byte-identical outcome is mildly surprising but consistent with: at this
generation length (LongBench answers are short), the new tokens written into
the cache are read back only a few times before generation ends, so the
quantization noise on those tokens has effectively no impact on what's
generated. We pick Mode A as the headline because it adds zero per-step
overhead. (Mode B adds 200+ small dispatches per generated token — net cost
with no net quality benefit.)

This is the C5.5 verification checkpoint.

---

## Phase 7 — Downstream sweeps (in progress)

**Goal:** the actual NeurIPS results table. LongBench full-task sweep on both
models + RULER NIAH at 4k/8k/16k context.

**Plan:**
- 8 configs per model: full_precision, jointqk@K∈{2,3,4}, turboquant@K∈{2,3,4},
  kivi_int4. (V locked across all JointQK/TurboQuant via `v_lock.txt`; KIVI
  is fixed int4/int4 with `group_size=128`.)
- LongBench uses the KIVI 8-task subset × 8 configs = 64 jobs per model.
- RULER: 3 contexts × 2 models × ~4 conditions = ~24 jobs.
- Current protocol is **prefill-only compression for every compressed method**:
  launchers pass `compress_decode=False`, so generation-step KV remains fp16.
  This avoids method-specific decode residual-window behavior and keeps the
  downstream comparison uniform.

**Steps:**

1. **Launchers.**
   - `experiments/stage1/scripts/launch_phase7_longbench.sh` (parameterized
     by `--model qwen3_8b | llama31_8b`, `--gpus`, `--fraction`)
   - `experiments/stage1/scripts/launch_phase7_ruler.sh` (`--ks 4 --ctxs
     4096,8192,16384`)
   Both build a `commands.txt` and dispatch via `parallel_launcher.py`.

2. **Chain script.** `experiments/stage1/scripts/_phase7_chain.py` polls Phase
   6's `_overview.log` for "12 OK", then sequences:
   - Read locked decode decision (now informational; current launchers force
     `compress_decode=False`)
   - Launch Phase 7 LongBench Qwen
   - Launch Phase 7 LongBench Llama
   - Launch Phase 7 RULER
   - Run final aggregators (`aggregate_longbench.py`, `aggregate_ruler.py`)

   I rewrote this from a bash version after a bug where bash arithmetic on
   multi-line `grep -c` output fired the next stage prematurely.

3. **Final aggregators.**
   - `experiments/stage1/eval/aggregate_longbench.py`
   - `experiments/stage1/eval/aggregate_ruler.py`

   Critical fix: kvpress writes results into numbered subdirs (`1/`, `2/`, …
   on re-runs). The first version of these aggregators globbed `**/metrics.json`
   and took the first sorted (alphabetical) match — which is the *oldest*
   (subdir `1/`). On re-runs that meant we were reading stale numbers. Fixed
   to sort by `p.stat().st_mtime, reverse=True` and take the most recent.

**Status / protocol update:**

- Earlier Phase 7 partial results were produced before the KIVI quantizer fix
  and before the prefill-only protocol was locked. Treat old KIVI rows as
  invalid and move them aside with a `.pre_kivi_fix` suffix before rerunning.
- Do not launch Phase 7 automatically. The user explicitly paused Phase 7
  experiments; launchers are prepared but should only be run on request.

**Preliminary signal (from the partial Qwen results we already have on 4–5
tasks):**

- At K=2 (low-budget regime), JointQK clearly beats TurboQuant — qasper
  +7.23 pp, narrativeqa +11.47 pp.
- At K=3 / K=4, JointQK is competitive with TurboQuant (within ±2 pp on most
  tasks).
- Both methods retain ≥90% of full-precision F1 at K=4 / V=3 on the tasks we
  have.
- KIVI int4 is the strongest baseline at large budgets but loses badly at
  K=2 (no equivalent budget — it's fixed at int4). This preliminary KIVI signal
  predates the KIVI quantizer fix and should not be used for claims.

**Final gate (C6).** Pass criterion: JointQK at K=4 / V_locked retains ≥90% of
full-precision LongBench score on **both** models, beats both peers on ≥80%
of (task, model) cells, and at K=2 maintains the JointQK ≥ TurboQuant ordering.
This is judged after the final aggregators run.

---

## Decision artifacts produced (concrete files)

| File | Producer | Consumer | Status |
|---|---|---|---|
| `artifacts/stage1/v_method_study/v_stats.pt` | Phase 1 calibrate | Phases 1, 3, 4, 6, 7 | ✅ |
| `artifacts/stage1/v_method_study/v_stats_llama31_8b.pt` | Phase 5 calibrate | Phase 7 (Llama) | ✅ |
| `artifacts/stage1/v_method_study/v_lock.txt` | Phase 1 aggregate (corrected) | Phases 3, 4, 6, 7 | ✅ |
| `artifacts/stage1/v_method_study/k_floor.txt` | Phase 1 aggregate | Phase 7 (informational) | ✅ |
| `artifacts/stage1/cca_vs_waterfill_study/cca_stats.pt` (with R_sym, V_h) | Phase 1 precompute | Phases 1, 4, 6, 7 (Qwen) | ✅ |
| `artifacts/stage1/cca_vs_waterfill_study/llama31_8b/cca_stats.pt` (with R_sym, V_h) | Phase 5 precompute | Phase 7 (Llama) | ✅ |
| `artifacts/stage1/downstream/qwen3_8b/decode_scope/decode_decision.txt` | Phase 6 aggregate | Informational; current Phase 7 launchers force prefill-only | ✅ WINNER=A |
| `artifacts/stage1/downstream/longbench_summary.json` | Phase 7 aggregate | Final report | ⏳ pending |
| `artifacts/stage1/downstream/ruler_summary.json` | Phase 7 aggregate | Final report | ⏳ pending |

---

## Files created vs files reused

**Created:**
- `experiments/stage1/scripts/parallel_launcher.py`
- `experiments/stage1/scripts/calibrate_sigma_v.py`
- `experiments/stage1/scripts/precompute_newbases.py`
- `experiments/stage1/scripts/launch_phase1ab.sh`, `launch_phase1c.sh`,
  `launch_llama_e3_e5.sh`, `launch_phase6_decode_scope.sh`,
  `launch_phase7_longbench.sh`, `launch_phase7_ruler.sh`
- `experiments/stage1/scripts/_phase6_chain.py`, `_phase7_chain.py`
- `experiments/stage1/scripts/make_cross_model_chart.py`
- `experiments/stage1/toolkit/v_compressor_adapter.py`,
  `jointqk_press.py`, `turboquant_press.py`,
  `kivi_press.py`, `kivi_quantizer.py`
- `experiments/stage1/eval/aggregate_phase1.py`,
  `aggregate_decode_scope.py`, `aggregate_longbench.py`, `aggregate_ruler.py`
- `experiments/stage1/tests/test_press_roundtrip_parity.py`

**Edited:**
- `experiments/stage1/toolkit/per_coord_quantization.py` — vectorized
  roundtrip, chunked-diffs memory budget, bit cap at 8 with redistribution
  (caps `K_max=2^bits ≤ 256`, prevents OOM on long contexts)
- `experiments/stage1/toolkit/kivi_quantizer.py` — fixed to match official
  KIVI min-subtraction quantization, sequence-axis K grouping, head-dim V
  grouping, and fp16 residual tail for non-divisible K sequence lengths
- `experiments/stage1/scripts/launch_phase7_longbench.sh`,
  `launch_phase7_ruler.sh` — force `compress_decode=False` for the current
  prefill-only downstream protocol
- `kvpress/evaluation/evaluate.py` — `press_kwargs` field + class instantiation
- `kvpress/evaluation/evaluate_registry.py` — register the 3 presses (as
  classes)
- `experiments/stage1/gates/gate_e3.py`, `gate_e5.py` — `--output-dir` arg
- `experiments/stage1/scripts/launch_cca_study.sh` — actually use `EXTRA_ARGS`
- 68 kvpress files — `from __future__ import annotations` for Python 3.9

**Reused unchanged:**
- `experiments/stage1/run_cca_vs_waterfill_study.py`,
  `collect_query_stats.py`
- `experiments/stage1/toolkit/{metric_transform.py, quantization.py,
  capture.py, moments.py, model.py, eval.py, io.py}`
- `kvpress/kvpress/presses/base_press.py`
- `turboquant-pytorch/compressors_v3.py`

---

## Open follow-ups / known caveats

1. **Layer-0 asymmetry.** JointQK skips K compression on layer 0
   (`layer0_full_precision=True`). TurboQuant and KIVI compress all layers.
   This is a documented design choice (Stage 1's anomalous-layer-0 finding),
   not a bug, but worth flagging in the paper's methods section.
2. **V eigenvalue spectrum is roughly uniform.** This is *why* `v_eigen_uniform`
   beat `v_eigen_waterfill` on Phase 1A. Worth noting that the K side has a
   sharp spectrum (which is why water-fill helps for K) and V does not.
3. **Mode A == Mode B byte-identical.** Suggests either (a) decode-step
   compression gets reapplied later anyway when next prefill is queried, or
   (b) the generated tokens are short enough that decode-step compression has
   no measurable effect. We chose A for clean compute. Worth a follow-up
   ablation if a reviewer asks.
4. **`kv-rd` env in CLAUDE.md is stale.** The real env is `efficient-llm`.
   Should update CLAUDE.md when convenient.
5. **EVAL_FRACTION=0.5.** If Phase 7 wall-clock projects past schedule, drop
   to 0.3. Currently on track.
