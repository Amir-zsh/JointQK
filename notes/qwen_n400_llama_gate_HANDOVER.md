# HANDOVER — Qwen3-8B 400-prompt-basis F1 study + Llama-3.1-8B replication gate

**Branch:** `entropy_coding` (all changes below are UNCOMMITTED). **Env:** conda `kv`
(NOT `.venv`), `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`, GPUs **0-3 only** (shared box —
never touch 4-7, never mass-kill GPU PIDs; kill only our own procs by command pattern).
Run everything from the repo root `/vault/samuel/efficient-llm/JointQK`.
`source /opt/miniconda3/etc/profile.d/conda.sh && conda activate kv`.

## Goal

Run downstream LongBench F1 (lcc / 2wikimqa / musique) for ALL of the report's
K-compression methods on **Qwen3-8B**, calibrated "**do-as-the-Llama-study**": the basis
second moments (Σ_Q/Σ_K/k_mean/k_cov) come from a **400-prompt pooled corpus** (~4.5M
tokens, matching the Llama EC study's n400 basis), while the entropy-coder / VQ-codebook
fits use a small retained per-example code pool. Then a **hard validation gate**: run the
SAME harness on **Llama-3.1-8B** and confirm we recover the study's published table
(FP≈44.99 / TurboQuant≈41.52 / EC≈44.9) before trusting the Qwen numbers.

Methods (11): bf16(FP), turboquant2, kivi2, scalar_int2 (QPCA-Fixed), oscar,
ec_deadzone (rANS/Exp-Golomb reconstruction — one bundle stands for both, lossless),
and 5 VQ configs (G4 cons_flat / strat_flat / strat_wf / cons_wf, G2 strat_flat).
V-side = v_turboquant@2b, layer-0 fp16 for all. musique-train rows excluded from musique eval.

## STATUS (as of handover)

- ✅ **Qwen table COMPLETE** (`artifacts/bench_allmethods_n400/`, fraction 0.3). Numbers below.
- ✅ Both 400-prompt bases captured & validated (Qwen 4,545,518 tok; Llama 4,429,650 tok).
- ✅ All Qwen n400 bundles + Llama bundles built. Starved-VQ codebooks (`*_n400.pt`) done.
- 🔄 **RUNNING NOW: fair VQ retrain** — 5 codebooks `vqa_*_fair.pt` on the 80-prompt /
  888k-tok / 8-task code pool (background wait `bw8jl2xp7`; G2 queued behind cons_flat).
- ⏳ **TODO: re-run VQ F1 (fair) + Llama gate**, then aggregate + verdict + report.

### Qwen n400 result (starved VQ — to be superseded by fair VQ)
| method | lcc | 2wikimqa | musique | mean |
|---|---|---|---|---|
| bf16 (FP) | 66.39 | 59.85 | 29.03 | 51.76 |
| ec_deadzone | 63.95 | 56.28 | 25.29 | 48.51 |
| oscar | 60.87 | 52.63 | 31.73 | 48.41 |
| scalar_int2 | 56.82 | 58.01 | 28.71 | 47.85 |
| vqG4_strat_wf (starved) | 55.37 | 52.31 | 28.40 | 45.36 |
| vqG4_strat_flat (starved) | 55.88 | 53.79 | 24.50 | 44.72 |
| kivi2 | 53.80 | 48.56 | 25.90 | 42.75 |
| vqG4_cons_wf (starved) | 53.12 | 47.83 | 24.43 | 41.79 |
| turboquant2 | 50.91 | 45.26 | 26.50 | 40.89 |
| vqG4_cons_flat (starved) | 50.63 | 42.23 | 8.26 | 33.71 |
| vqG2_strat_flat (starved) | 32.29 | 35.80 | 7.04 | 25.04 |

**Structure reproduces the study: EC ≈ FP ≫ TurboQuant, EC leads on the code task (lcc).**
VQ underperforms — DIAGNOSED as a training-data handicap (see below), being re-tested fair.

## Why the VQ retrain (the current in-flight work)

scalar/OSCAR are pure-Σ (get the full 400-prompt 8-task basis); EC's grid also generalizes.
But VQ's *centroids ARE the codebook*, and they were k-means'd on the leftover EC pool =
**24 prompts / 81k tok / 3 short non-code tasks** (`query_stats_longbench_under4k`), which
doesn't cover the eval distribution (code/long-QA). Fair fix: retrain on a matched
**80-prompt / 888k-tok / 8-task (incl. repobench-p code)** pool = `query_stats_qwen3_8b_compact8train80`.
Same leakage control (musique-train already excluded from eval).

## KEY ARTIFACTS (all on disk)

Bases (pooled Σ, `calib_moments` convention, drop-in via `--basis-moments`):
- Qwen: `/vault/samuel/data/basis_moments_qwen3_8b_compact8train400/basis_moments.pt`
- Llama: `/vault/samuel/data/basis_moments_llama31_8b_compact8train400/basis_moments.pt`

Code pools (per-example k_post, query_stats format, read via `--data-root`):
- Qwen EC (old 3-task, module default FULL_DIR): `/vault/samuel/data/query_stats_longbench_under4k` (24 ex)
- Qwen VQ-fair (8-task): `/vault/samuel/data/query_stats_qwen3_8b_compact8train80` (80 ex, 888,731 tok)
- Llama EC (study's exact 26 rows): `/vault/samuel/data/query_stats_llama31_8b_eccalib26`

Bundles (`entropy_coding/`): `mb_{scalar_int2,oscar,ec_deadzone}_n400.pt` (Qwen),
`mb_{scalar_int2,oscar,ec_deadzone}_llama.pt` (Llama).

VQ codebooks (`entropy_coding/`): `vqa_*_n400.pt` (starved, done),
`vqa_*_fair.pt` (fair, RETRAINING: G4 cons_flat/strat_flat/strat_wf/cons_wf + G2 strat_flat).

F1 command JSONLs + output dirs:
- Qwen all-methods: `logs/bench_allmethods_n400/commands.jsonl` → `artifacts/bench_allmethods_n400/` (DONE)
- Qwen VQ-fair: `logs/bench_vqfair/commands.jsonl` → `artifacts/bench_vqfair/` (READY, not yet run)
- Llama gate: `logs/bench_llama_gate/commands.jsonl` → `artifacts/bench_llama_gate/` (READY; model_kwargs has local_files_only:true)
- musique exclusion: `logs/bench_allmethods/exclude_musique_train.json` (50 rows)

Llama model (gated — load from local snapshot):
`/vault/ultraz/llm_models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659`

## CODE CHANGES made this session (uncommitted, branch `entropy_coding`)

- `entropy_coding/capture_basis_moments.py` (NEW) — subcommands: `capture`/`merge`
  (400-prompt pooled Σ, calib_moments convention, sharded), `codepool`/`codepool_merge`
  (per-example k_post, query_stats format, sharded by global index).
- `entropy_coding/build_method_bundles.py` — added `--basis-moments`, `--code-idx`,
  `--data-root`, `--suffix`. scalar/oscar pure-Σ; only EC reads codes.
- `entropy_coding/train_group_vq_alloc.py` — added `--basis-moments`, `--code-idx`, `--data-root`.
- `entropy_coding/group_vq_codec.py` — `_kmeans` now **chunks the Lloyd assignment over
  samples** (fixes OOM: 888k samples × K=8192 → 29GB cdist). Behaviour identical, bounded mem.
- `entropy_coding/oscar_codec.py` — `OSCARCompressor.roundtrip` computes in float32 &
  returns input dtype (fix: bf16 K × float32 rotation mismatch; OSCAR slices K without the
  float re-centering that auto-promotes in the QPCA/EC paths).
- `vendor/kvpress/evaluation/evaluate.py` — applies `exclude_indices_file` (drops calib-train
  rows by positional row_index before sampling). PRESS_REGISTRY already has jointqk/turboquant/
  kivi/vq/kbundle + `press_kwargs`/`exclude_indices_file` config fields (from earlier).
- `entropy_coding/aggregate_f1.py` (NEW) — F1 table aggregator (walks `<root>/<method>/**/metrics.json`).
- Presses: `kvq/presses/kbundle_press.py`, `vq_press.py` (KBundlePress loads mb_*.pt; VQPress loads vqa_*.pt).

## IMMEDIATE NEXT STEPS

1. **Wait for 5 fair VQ codebooks** `entropy_coding/vqa_*_fair.pt` (background wait `bw8jl2xp7`).
   If a waterfill config OOMs again, it already has `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   + chunked kmeans; if still OOM, lower `--max-k-bits` or subsample code_idx.
2. **Run fair VQ F1** (Qwen, online dataset):
   `python3 pipelines/bench/worker.py --model Qwen/Qwen3-8B --commands-file logs/bench_vqfair/commands.jsonl --log-dir logs/bench_vqfair --gpus 0,1 --jobs-per-gpu 1 --max-retries 2`
3. **Run Llama gate** (local model, online dataset — do NOT set HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE;
   model_kwargs already has local_files_only:true):
   `python3 pipelines/bench/worker.py --model <LLAMA_SNAPSHOT_PATH> --commands-file logs/bench_llama_gate/commands.jsonl --log-dir logs/bench_llama_gate --gpus 2,3 --jobs-per-gpu 1 --max-retries 2`
   (steps 2 and 3 can run concurrently on split GPUs.)
4. **Aggregate**:
   `python3 entropy_coding/aggregate_f1.py artifacts/bench_allmethods_n400 --order bf16 turboquant2 kivi2 scalar_int2 oscar ec_deadzone`
   `python3 entropy_coding/aggregate_f1.py artifacts/bench_vqfair`
   `python3 entropy_coding/aggregate_f1.py artifacts/bench_llama_gate`
5. **Llama-gate verdict**: does EC≈FP≫TQ hold and land near 44.9/44.99/41.52? If yes → harness
   validated. If EC ≪ FP → investigate before trusting Qwen (our EC = UniformECRoundtrip ≈ study's
   ec_qpca_unc; scalar = QPCA-Fixed ≠ study's Lloyd-Max, so compare FP/TQ/EC cleanly, scalar loosely).
6. **Report**: fold results into `notes/entropy_coding_throughput_report.md` (currently proxy-based;
   add the downstream-F1 section + the calibration-data-amount finding + the fair-VQ result).

## GOTCHAS (bit us this session)

- **Qwen loads online; Llama is gated.** Load Llama from the local snapshot PATH with
  `model_kwargs.local_files_only=true` and NO offline env vars — `HF_HUB_OFFLINE=1` also forces
  *datasets* offline where the LongBench cache config ('default-data_dir=lcc') doesn't match → all cells fail.
- **Worker forks children from the parent's imported modules.** Editing a press/codec module does
  NOT affect an already-running worker — launch a FRESH worker to pick up code fixes.
- **Worker skips cells with an existing `metrics.json`.** To force a re-run, point at a fresh output_dir
  (that's why fair VQ uses `artifacts/bench_vqfair/`, not the original dirs).
- **Shared box.** Never `nvidia-smi ... | xargs kill` (hits other users). Kill only our procs by
  command pattern (`pkill -f "bench_.../commands.jsonl"`) or specific owned PIDs. Never GPUs 4-7.
- Cell timing: ~450-740s each (VQ slowest). 12-15 cells/GPU-hour.
