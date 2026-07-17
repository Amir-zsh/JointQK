# `entropy_coding/` — KV-cache **Key** compression: methods, reproduction, kernels

This directory holds the K-cache key-compression methods compared in
`../notes/entropy_coding_throughput_report.md` (the deliverable) and frozen as
`../handoffs/milestone1_reference.md` (the baseline every experiment compares to).
Open experiments are in `../handoffs/`.

Everything operates **per (layer, kv-head)** at **2 bits/coord**, in a per-head
**QPCA basis** (a Q-weighted PCA fit on pooled `(Σ_Q, Σ_K)`). The inverse rotation is
folded into the query (RoPE), so keys stay in the coded space at decode.

---

## File map

**Codecs (the methods)**
| file | method / role |
|---|---|
| `run_pca_ec_deadzone.py` | **core lib** — `calib_moments`, `build_qpca_basis`, `build_qpca_fixed_deadzone` (INT2), `build_qpca_ec` (EC delta+model), `UniformECRoundtrip`, `_dz_round/_dz_dequant` |
| `group_vq_codec.py` | **VQ** — `GroupVQCompressor` (with optional `pertoken_norm=True` for OSCAR-style per-token RMS normalization), `_kmeans` (chunked Lloyd, optional ECVQ rate penalty), `SinkRecentWrap` (fp16 band), `OutlierProtectWrap` (content-based outlier band), `group_boundaries` |
| `oscar_codec.py` | **OSCAR** — `build_oscar_rotation`, `OSCARCompressor` (Lloyd-Max INT2 + sink/recent) |
| `expgolomb_codec.py` | **Exp-Golomb** coder (`estimate_bits_per_coord`, `choose_k_per_coord`, GPU decode) |
| `kvq_codec.py` (+`rans_interleaved.py`) | **rANS** paged codec (`PageCodecRANS[CUDA]`, `BatchRANS{En,De}coder`) |

**Build / train**
| file | what it builds |
|---|---|
| `capture_basis_moments.py` | 400-prompt pooled Σ (`capture`/`merge`); per-example q/k pools (`codepool`) |
| `train_group_vq_alloc.py` | VQ codebooks (`vqa_*.pt`) — `--G` (default 4, deployment config), `--grouping`, `--allocation`, `--pertoken-norm` (train on per-token-normalized residuals; codebook decodes with the same normalization at inference — matches OSCAR/KIVI's per-token dynamic scale approach), `--ecvq-lambda`, `--qtau` |
| `build_method_bundles.py` | scalar-INT2 / OSCAR / EC compressor bundles (`mb_*.pt`) |
| `make_fp8.py` | fp8-quantize a VQ codebook (`--fmt e4m3|e5m2`) |

**Measure / analyse**
| file | output |
|---|---|
| `aggregate_f1.py` | LongBench F1 table from a bench dir |
| `proxy_score.py` | fidelity proxy: top-1/top-5/relMSE on held-out q/k |
| `int2_sweep.py` | INT2 dz/load/widen sweep (proxy) |
| `attnkl_finetune.py` | attention-KL codebook fine-tune (VQ / INT2) |
| `eg_vs_rans_matched.py`, `expgolomb_rate_match.py` | EG-vs-rANS bits & throughput |
| `diag_ec_basis.py` | EC centered-vs-uncentered basis diagnostic |
| `ecvq_sweep.py` | ECVQ λ sweep (open experiment, `../handoffs/ecvq_handoff.md`) |

**Fused decode kernels** (Triton + CUDA; keep for the **H100** extension)
`fused_decode_all.py` (BF16/INT2/OSCAR/VQ + fp8 `VQ8` path, correctness-gated,
**`--G 4` is the deployment default** — L1-resident codebook, int32-gather VEC
path; `--G 6` exceeds L1 and runs ~5× slower at L2 speed),
`fused_decode_vark.py` (variable-K), `vq_fused.cu`, `vq_decode.cu`,
`rans_{decode,encode,decode_lut}.cu`, `expgolomb_decode.cu`.

`_archive/` — superseded prototypes/micro-benchmarks (safe to delete).

---

## Reproduce the report (end to end)

Activate env first: `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate kv`
(NOT `.venv`). Run from the repo root. `OMP_NUM_THREADS=8`. GPUs **0–3 only**.

```bash
Q=Qwen/Qwen3-8B
MAN=artifacts/calibration/longbench_compact8_qkv_qwen3_8b/00_split/manifest.json
B=/vault/samuel/data/basis_moments_qwen3_8b_compact8train400   # already on disk

# 1) 400-prompt pooled basis (Σ_Q, Σ_K, k_mean, k_cov) — sharded over GPUs, then merge
for s in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$s python3 -u entropy_coding/capture_basis_moments.py \
    capture --model $Q --split-manifest $MAN --num-shards 4 --shard-id $s --out $B/shard_$s.pt & done; wait
python3 entropy_coding/capture_basis_moments.py merge --shards $B/shard_{0,1,2,3}.pt --out $B/basis_moments.pt

# 2a) VQ codebook (G=4 stratified, k-means on the fair 8-task code pool), then fp8
CP=/vault/samuel/data/query_stats_qwen3_8b_compact8train80
python3 entropy_coding/train_group_vq_alloc.py --basis-moments $B/basis_moments.pt \
    --data-root $CP --code-idx $(seq -s' ' 0 79) --G 4 --grouping stratified --allocation flat \
    --out entropy_coding/vqa_G4_strat_flat.pt
python3 entropy_coding/make_fp8.py --in entropy_coding/vqa_G4_strat_flat.pt \
    --out entropy_coding/vqa_G4_strat_flat_fp8.pt --fmt e4m3

# 2b) scalar-INT2 / OSCAR / EC bundles (EC on the UNCENTERED basis — important)
python3 entropy_coding/build_method_bundles.py --basis-moments $B/basis_moments.pt \
    --data-root $CP --code-idx 0 10 20 30 40 50 60 70 --ec-basis uncentered --suffix _n400

# 3) downstream F1 (worker runs one method×task cell per line of a JSONL; see logs/bench_*/commands.jsonl)
python3 pipelines/bench/worker.py --model $Q --commands-file <cells.jsonl> \
    --log-dir logs/<run> --gpus 0,1,2,3 --jobs-per-gpu 1 --max-retries 1
python3 entropy_coding/aggregate_f1.py artifacts/<run>          # -> the F1 table

# 4) decode throughput (A100; correctness-gated; prints BF16/INT2/OSCAR/VQ/VQ8)
# `--G 4` is now the default (deployment config); safe to omit.
CUDA_VISIBLE_DEVICES=0 python3 entropy_coding/fused_decode_all.py --Ts 65536

# 4b) fair-conditions timing: VQ8 vs OSCAR with BOTH K and V at INT2 (matches OSCAR's own
#     benchmark setup; V=fp16 in step 4 above dilutes K-compression ratios toward 1×).
CUDA_VISIBLE_DEVICES=0 python3 entropy_coding/bench_vint2.py --Ts 131072
#     Expected on A100, bs=1, T=128K, 288 heads: VQ8 ≈ 10.9 ms, OSCAR ≈ 11.4 ms
#     (VQ8 slightly faster; both around 0.30–0.32 ms/layer). Under matched Triton
#     kernels the two are within 5% — the ~4× gap Amir reported is OSCAR's native
#     CUDA kernel vs a non-VEC/L2-resident VQ (G=6, K=4096), NOT VQ8 (G=4, fp8, L1).

# 5) fidelity proxy + EG rate
python3 entropy_coding/proxy_score.py --eval-idx 20 21 22 23 \
    --bundles scalar=entropy_coding/mb_scalar_int2_n400.pt oscar=entropy_coding/mb_oscar_n400.pt \
              ec=entropy_coding/mb_ec_deadzone_n400.pt --vq VQ=entropy_coding/vqa_G4_strat_flat.pt
```

For **Llama-3.1-8B**: use the local snapshot as `--model`
(`/vault/ultraz/llm_models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659`),
add `"local_files_only": true` to the bench cell's `model_kwargs`, and do **not** set
`HF_HUB_OFFLINE` (it breaks the dataset load; the LongBench dataset loads online, no
token needed). Llama VQ additionally uses a 4/32 fp16 band (`vq_sink=4, vq_recent=32`
in the press kwargs).

## Calling the pieces from Python

```python
import run_pca_ec_deadzone as base
sq, sk, km, kc, meta = base.calib_moments(root, manifest, idx)   # or load basis_moments.pt
qpca = base.build_qpca_basis(sq, sk)                              # uncentered QPCA
scalar = base.build_qpca_fixed_deadzone(qpca, km, b=2, ...)      # INT2/QPCA-Fixed -> {(l,h): comp}
from group_vq_codec import GroupVQCompressor, SinkRecentWrap      # VQ codec + fp16 band
from oscar_codec import build_oscar_rotation, OSCARCompressor     # OSCAR
```
Every compressor exposes `.roundtrip(k)` (K → reconstructed K) and `.to(device)`; the
kvpress side (`kvq/presses/vq_press.py::VQPress`, `kbundle_press.py::KBundlePress`)
loads a codebook/bundle and wires it into the prefill hook.

## Key artifacts (already on disk)
- Bases: `/vault/samuel/data/basis_moments_{qwen3_8b,llama31_8b}_compact8train400/basis_moments.pt`
- Code pools (query_stats format): `…/query_stats_{qwen3_8b,llama31_8b}_compact8train80` (k_post),
  `…/query_stats_qwen3_8b_qk8` (q+k, for attention-KL).
- Trained codebooks/bundles: `entropy_coding/{vqa_*,mb_*}.pt`.

## Samples & error bars (statistical rigor)

Each LongBench task has a fixed size — **lcc 500, 2wikimqa 200, musique 200** (musique
drops to **150** after excluding the 50 compact8-train rows). The bench `fraction`
knob subsamples; **`fraction: 1.0` uses the full task** (the most samples available —
there is no "more" without a different dataset, e.g. RULER). Always run final numbers
at `fraction: 1.0`.

**musique is intrinsically noisy** — its per-example F1 is bimodal (~53% score 0, ~16%
score 100; per-example std ≈ 38), so the sample mean has a wide CI (±11 at n=45, ±5 at
the full n=150). Rank methods on **lcc / 2wikimqa** (tight); treat musique as directional.

**Error bars, no GPU.** The per-example scores are already in each cell's
`predictions.csv`, so bootstrap CIs are free (no re-running the model):
```bash
python3 entropy_coding/bootstrap_ci.py artifacts/<bench_dir> --order bf16 vq ec_deadzone int2 oscar kivi turboquant
```
It reuses the repo's exact per-task metric (`dataset2metric`: code_sim for lcc, qa_f1
for musique/2wikimqa, …) and prints `mean ± 95% CI` per method×task plus the sample
count. Its point estimates **match `aggregate_f1.py` / the worker's `metrics.json`
exactly** on all three tasks (verified full-sample).

**Gotcha (fixed in `bootstrap_ci.py`):** the worker's `predictions.csv` mangles the gold
`answers` column — a multi-element numpy array is written with no separator, so e.g.
`['Knowsley', 'Metropolitan Borough of Knowsley']` becomes one glued string. This hits
~27% of musique rows (multiple acceptable answers) and ~0% of lcc, so re-scoring the CSV
`answers` column undercounts musique by ~3–5 pts. `bootstrap_ci` therefore re-fetches the
true golds from the LongBench dataset by `_id` (predicted_answer itself round-trips fine).
Any custom re-scorer must do the same. (Also: never call `calculate_metrics(df)` on a
CSV-loaded df — `answers` reloads as a string, so it iterates characters and returns ≈2.)

## Rules
GPUs 0–3 only (shared box; never mass-kill GPU PIDs, never touch 4–7). Bench worker
**skips a cell if its `metrics.json` already exists** → use a fresh `output_dir` to
force a rerun. Train/calibrate on **compact8-train only**: lcc & 2wikimqa are not in
compact8; musique-train rows are excluded from eval via
`logs/bench_allmethods/exclude_musique_train.json`.
