# Milestone 1 — Reference implementation & baseline (freeze)

**This is the fixed baseline. Every subsequent experiment compares against these numbers.**
Full write-up: `notes/entropy_coding_throughput_report.md`. Branch `entropy_coding`
(all artifacts below are on disk; changes uncommitted as of this milestone).

All F1 numbers are **`fraction 1.0` (full sample)**; `mean ± ` is the bootstrap 95% CI
half-width. (Earlier `fraction 0.3` Qwen numbers were inflated ~+11 by an easy 2wikimqa
head-subset — these supersede them. Per-task CIs in `notes/entropy_coding_throughput_report.md`.)

## Reference results

**Qwen3-8B — LongBench F1, fraction 1.0** (V=TurboQuant@2b, layer-0 fp16, 400-prompt calib;
VQ uses a 4/32 fp16 sink/recent band):

| method | b/coord | lcc | 2wiki | musique | mean |
|---|---|---|---|---|---|
| BF16 | 16 | 66.58 | 48.41 | 33.42 | 49.47 ±3.2 |
| EC (rANS/EG, uncentered basis) | 1.94 | 63.39 | 45.93 | 33.20 | 47.51 ±3.1 |
| **VQ (G4 stratified, 4/32 band, fp8)** | 2.12 | 64.34 | 45.24 | 30.42 | **46.67 ±3.1** |
| VQ (no band) | 2.00 | 58.49 | 47.38 | 33.52 | 46.46 ±3.1 |
| OSCAR (Lloyd-Max) | 3.47 | 61.86 | 44.42 | 27.15 | 44.48 ±3.0 |
| INT2/QPCA-Fixed (dz=0.375) | 2.00 | 57.05 | 42.74 | 25.15 | 41.65 ±2.9 |
| KIVI | 2.00 | 56.54 | 39.49 | 26.05 | 40.69 ±3.0 |
| TurboQuant | 2.125 | 52.99 | 38.73 | 27.09 | 39.60 ±2.8 |

**Llama-3.1-8B — LongBench F1, fraction 1.0:**

| method | b/coord | lcc | 2wiki | musique | mean |
|---|---|---|---|---|---|
| BF16 | 16 | 53.46 | 50.99 | 32.62 | 45.69 ±3.1 |
| **VQ (G4 stratified, 4/32 band, fp8)** | 2.12 | 53.51 | 51.17 | 31.49 | **45.39 ±3.0** |
| EC (uncentered) | 1.94 | 53.63 | 47.92 | 30.66 | 44.07 ±3.1 |
| INT2 (dz=0.625) | 2.00 | 47.13 | 51.88 | 30.51 | 43.17 ±3.0 |
| TurboQuant | 2.125 | 51.05 | 44.35 | 31.80 | 42.40 ±3.1 |
| OSCAR | 3.47 | 51.55 | 43.20 | 29.00 | 41.25 ±3.0 |
| VQ (no band) | 2.00 | 43.41 | 45.29 | 26.38 | 38.36 ±2.9 |

**Rate & decode latency (A100-SXM4-40GB, T=65,536):** BF16 6.76ms/1.00× · INT2 3.95/1.71× ·
OSCAR 4.50/1.50× · **VQ (fp8) 4.74/1.43×** · rANS 1.94 b/c (~3655ms serial) · Exp-Golomb 2.71 b/c
(~560ms serial). VQ reconstruct stage fp16→fp8: 3.85→2.14ms.

**Fidelity proxy (top-1/top-5/relMSE, layer-0 excl, held-out q/k):** EC-uncentered 0.81/0.98/0.040 ·
INT2 0.67/0.92/0.089 · OSCAR-LM 0.53/0.72/0.126. (`entropy_coding/proxy_score.py`)

## Headline findings (context for future work)
1. **Calibration data amount was decisive** — 3-prompt → 400-prompt basis fixed everything
   (replicated the Llama EC study's own finding). All methods now use the 400-prompt pooled Σ.
2. **EC bug fixed**: EC was on the *centered* k_cov basis; switching to *uncentered* Σ_K
   (matching scalar) recovered it (musique +6). 
3. **VQ was starved, not bad**: its k-means codebook trained on 24 short 3-task prompts; on the
   fair 80-prompt 8-task pool + a fp16 sink/recent band, VQ leads the compressed field on Llama
   (45.39) and ties EC at full sample on Qwen (VQ+band 46.67 vs EC 47.51, CIs overlap). **fp8
   centroids** make it faster too (int32 codeword, −44% reconstruct) at no F1 cost.
4. **MSE ≠ F1** (recurring): the INT2 dz-sweep to MSE *backfired* (47.9→37.8); attention-KL
   codebook fine-tuning was verified-but-flat. Optimize the downstream objective, not MSE.

## Reference implementation (key code + artifacts)

Bases (400-prompt pooled Σ, `calib_moments` convention):
- Qwen: `/vault/samuel/data/basis_moments_qwen3_8b_compact8train400/basis_moments.pt`
- Llama: `/vault/samuel/data/basis_moments_llama31_8b_compact8train400/basis_moments.pt`

VQ codebooks (fp16 + fp8): `entropy_coding/vqa_G4_strat_flat_fair.pt` (Qwen),
`vqa_G4_strat_flat_llama.pt` (Llama), `*_fp8.pt` (e4m3), `*_fp8e5.pt` (e5m2).

Per-example code pools (query_stats format): k_post only —
`/vault/samuel/data/query_stats_qwen3_8b_compact8train80` (80 prompts, 8 tasks),
`…_llama31_8b_compact8train80`; q_post+k_post — `…_qwen3_8b_qk8` (8 tasks, capped 8k tok).

Code:
- **VQ trainer** `entropy_coding/train_group_vq_alloc.py` — `--basis-moments`, `--data-root`, `--code-idx`, `--G/--grouping/--allocation`.
- **k-means** `entropy_coding/group_vq_codec.py::_kmeans` (chunked; **this is what ECVQ replaces**) + `GroupVQCompressor`, `SinkRecentWrap`.
- **VQ press** `kvq/presses/vq_press.py` (`vq_sink`/`vq_recent` band).
- **Bundle builder (scalar/OSCAR/EC)** `entropy_coding/build_method_bundles.py` (`--basis-moments`/`--data-root`/`--ec-basis`).
- **Decode/throughput kernel** `entropy_coding/fused_decode_all.py` (`--G 4`; fp8 `VQ8` int32 path; inverse rotation folded into the query, keys stay coded).
- **Fidelity proxy** `entropy_coding/proxy_score.py`; **attention-KL fine-tune** `entropy_coding/attnkl_finetune.py`.
- **F1**: `pipelines/bench/worker.py` + aggregator `entropy_coding/aggregate_f1.py`.
- musique-train exclusion for eval: `logs/bench_allmethods/exclude_musique_train.json`.

## Environment & rules (do not deviate)
- conda env **`kv`** (NOT `.venv`): `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate kv`. `OMP_NUM_THREADS=8`.
- **GPUs 0–3 only** (shared box). Never mass-kill GPU PIDs; kill only own procs by pattern. Never touch GPUs 4–7.
- Run from repo root `/vault/samuel/efficient-llm/JointQK`.
- **Llama is gated** → load local snapshot `/vault/ultraz/llm_models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659` with `model_kwargs.local_files_only=true` and **no** `HF_HUB_OFFLINE` (that breaks the dataset; datasets load online, no token needed).
- **Leakage**: lcc & 2wikimqa are NOT in compact8; musique-train rows are excluded from eval. Train/fine-tune on compact8-train only.
- Worker **skips cells with an existing `metrics.json`** → use a fresh `output_dir` to force a re-run. On OOM the worker's retry can **leak GPU memory** — start from clean GPUs, kill only your own leaked children.
