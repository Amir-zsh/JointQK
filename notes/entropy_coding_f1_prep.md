# F1 test prep: downstream LongBench/RULER for the compression methods

**Goal:** run *real downstream F1* (LongBench/RULER), not the attention-retention
proxy, for the K-compression methods — especially the properly-allocated VQ winner
(G4 stratified+waterfill, 0.747/0.964 proxy) vs scalar INT2 (jointqk) and
full-precision.

## STATUS (2026-07-10): env + integration DONE; smoke run validates the path

- **Env:** bench deps installed into the **`kv` conda env** (torch 2.11+cu128/triton 3.6
  verified unchanged). transformers 5.2.0, pandas, datasets, accelerate, nltk, jieba,
  fuzzywuzzy, rouge, rouge_score, bert_score, cachetools, fire. **flash-attn skipped →
  pass `model_kwargs.attn_implementation: "sdpa"`.**
- **Integration layer reconstructed** in `vendor/kvpress/evaluation/evaluate.py` (it was
  lost with the `.venv`): registers `jointqk/turboquant/kivi/vq` presses; adds
  `press_kwargs` + `exclude_indices_file` config fields; `_setup_press` rebuilds the press
  dataclass from `press_kwargs`. Verified: `press_name="vq"` + `press_kwargs` → VQPress.
- **`kvq/presses/vq_press.py` (`VQPress`)** loads a trained VQ codebook into the K side,
  V side = `v_turboquant@2b` (no v_stats needed). `GroupVQCompressor.roundtrip` accepts
  `(B,S,d)`.

### Run recipe
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate kv
cd <repo>
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python3 -u pipelines/bench/worker.py \
  --model "Qwen/Qwen3-8B" --commands-file <jsonl> --log-dir <dir> \
  --gpus 0,1,2,3 --jobs-per-gpu 1
```
Commands JSONL (one cell/line): see `logs/bench_vq_smoke/commands.jsonl` for the working
`no_press` (oracle) and `vq` (press_kwargs.vq_codebook_path) rows — copy and swap
`data_dir` (task) / `vq_codebook_path`. **Every cell needs**
`model_kwargs:{attn_implementation:"sdpa",dtype:"auto",device_map:"auto"}` (no flash-attn).

### Still to decide / watch
- **Calibration fairness:** VQ codebooks are calibrated on `longbench_under4k [0,5,6]`; the
  stock `jointqk` scalar baseline bundle is `compact8_n400`. For an honest VQ-vs-scalar F1
  delta, run the scalar (jointqk) baseline on the SAME calibration as VQ, or retrain VQ on
  the bench calib.
- Results land under each cell's `output_dir` as `predictions.csv` + `metrics.json`.

---

## (original prep notes below)

## What's ready (integration — done)

- **`kvq/presses/vq_press.py` — `VQPress`.** Subclasses `JointQKPress`, reuses all
  its machinery (layer-0 skip, V-side `v_turboquant@v_bits` identical to
  `jointqk_k2_v2`, prefill/decode hooks) and only swaps the K compressors: loads a
  trained group-VQ codebook bundle and builds one `GroupVQCompressor` per
  (layer, kv_head). `GroupVQCompressor.roundtrip` already matches the interface
  `_quantize_layer` calls; it now flattens leading dims so it accepts the
  `(B, S, d)` tensors the kvpress hook passes. Registered in
  `kvq/presses/__init__.py`. **Untested** — the bench env doesn't exist yet (below).
- press_kwargs: `{"vq_codebook_path": <.pt>, "v_stats_path": <v_stats>, "v_method":
  "v_turboquant", "v_bits": 2, "layer0_full_precision": true, "quantize_k": true,
  "quantize_v": true}`. No `cca_stats_path` — the basis is baked into the codebook.

## The blocker: no bench environment

Nothing on the box can run the bench today:
- **`kv` conda env** (what all the kernel/accuracy work used): has torch 2.11+cu128
  and triton 3.6 (matches the lock), but **no pandas / transformers / datasets /
  accelerate / flash-attn**.
- **`base`**: transformers 4.26.1 (2023) — far too old for Qwen3-8B; Python 3.8.
- **`.venv/bin/python`** that every `pipelines/bench/launch_*.sh` calls: **does not
  exist**.

`requirements.lock.txt` (Python 3.12) pins: torch==2.11.0+cu128, transformers==5.2.0,
flash-attn==2.8.3, pandas==2.3.3, datasets==4.8.4, accelerate==1.13.0, triton==3.6.0.

**Two options to unblock (needs a decision — big/slow op):**
1. **Rebuild `.venv` from the lock via `uv`** (canonical, what the scripts expect):
   `uv venv --python 3.12 .venv && uv pip install -r requirements.lock.txt`.
   ~5 GB; flash-attn builds ~10–30 min. Isolated, reproducible.
2. **Add the bench-layer deps into `kv`** (fastest — kv already has the torch/triton/
   cuda base the lock needs): `pip install transformers==5.2.0 datasets accelerate
   pandas flash-attn==2.8.3 nltk jieba fuzzywuzzy bert-score rouge`. Risk: numpy/
   transformers churn against kv's existing pins; could disturb the kernel env.
   Then point the launchers at `python` instead of `.venv/bin/python`.

Recommendation: option 1 (isolated, matches the scripts). Confirm before running —
it's a multi-GB install.

## Two more things to settle before the run

- **Registry registration.** The bench dispatches `press_name` through
  `PRESS_REGISTRY`, but the *vendored* `vendor/kvpress/evaluation/evaluate_registry.py`
  contains **no `jointqk`/`turboquant`** — those must have been registered by the
  now-missing `.venv`'s kvpress (a patched site-packages or local registry). When the
  env is rebuilt, locate that mechanism and add `"vq": VQPress(...)` (and confirm
  `"jointqk"` is present) so `press_name="vq"` resolves. `_setup_press` sets attrs on
  the registry instance, so the registry entry can be a bare `VQPress()` with
  press_kwargs supplying the codebook path (verify press_kwargs plumbing in the
  actual evaluate.py the env ships).
- **Calibration alignment (fairness).** The VQ codebooks
  (`entropy_coding/vqa_*.pt`, `group_vq_b2_calib056*.pt`) are calibrated on
  `longbench_under4k` idx [0,5,6] (Qwen3-8B). The bench's stock `jointqk` scalar
  baseline bundle is `compact8_n400` (different calibration). For an honest
  VQ-vs-scalar F1 delta, both must use the **same** calibration — either (a) retrain
  the winning VQ config on the bench's calib corpus, or (b) also run the scalar
  (jointqk) baseline built on the entropy_coding [0,5,6] calibration. (b) is less
  work if a jointqk bundle can be built from the same moments the VQ used.

## Suggested first F1 run (once unblocked)

Model Qwen3-8B, a few LongBench tasks (e.g. `lcc`, `2wikimqa`, `hotpotqa`),
`fraction` small first (smoke), then 1.0. Cells:
- `no_press` (full-precision oracle)
- `jointqk` scalar INT2 @ k=2,v=2 (the baseline to beat)
- `vq` with `vqa_G4_strat_flat.pt` (decode-ready fixed-K, proxy 0.594/0.946)
- `vq` with `vqa_G4_strat_wf.pt` (accuracy winner, proxy 0.747/0.964)
Drive with `pipelines/bench/worker.py --commands-file <jsonl> --gpus 0,1,2,3`
(GPUs 0–3 only). See `launch_ec_longbench.sh` for the emit/JSONL pattern.

## Open question for the proxy→F1 check

The proxy (top-1/top-5 attention retention) ranked VQ-waterfill > scalar > VQ-flat.
F1 may re-order (it depends on generation, not just argmax fidelity). The point of
this run is to see whether the proxy win survives to task accuracy.

State/log: `logs/entropy_coding_faithful_report.log`; report:
`notes/entropy_coding_throughput_report.md`; VQ trainer:
`entropy_coding/train_group_vq_alloc.py`.
