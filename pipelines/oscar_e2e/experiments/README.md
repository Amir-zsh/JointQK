# Declarative served-eval experiments

One JSON spec per study. The spec is the single place every argument lives:
model, per-method serve flags + env + calibration artifacts, cells with
rows/shards/sampling. Nothing method- or cell-specific is hardcoded in the
runner.

## Replicating a study

```bash
cd /vault/amir/efficient-llm/teamily-project     # identical on lambda6/7

# 1. Materialize: manifest (git SHAs, artifact sha256s, row counts),
#    per-method arm scripts, sharded queue. Idempotent — completed cells
#    are skipped, so re-running resumes.
.venv/bin/python pipelines/oscar_e2e/run_experiment.py \
    --spec pipelines/oscar_e2e/experiments/llama31_8b_grid_v2.json \
    --queue logs/pool_queue_llama_v2.tsv

# 2. Workers (one per GPU; ARMS_DIR switches the pool runner to spec mode):
for g in 1 2 3 4 5 6; do
  ARMS_DIR=$PWD/artifacts/oscar_llama31_8b/grid_v2/arms \
    nohup bash pipelines/oscar_e2e/gpu_pool_runner.sh \
      logs/pool_queue_llama_v2.tsv $g > /dev/null 2>&1 &
done

# 3. When the queue drains (no TODO/RUN rows), merge shards + score:
.venv/bin/python pipelines/oscar_e2e/merge_shards.py \
    --root artifacts/oscar_llama31_8b/grid_v2
```

Provenance of a finished run lives in `<out_root>/experiment_manifest.json`
(spec snapshot, repo + engine-clone commits, sha256 of every calibration
artifact and rotation file, queue size) and `<out_root>/arms/*.sh` (the
exact env + serve flags each method booted with). Per-cell results are
`<out_root>/<method>/<cell>/metrics.json` with `predictions.csv` beside.

## Calibration policy (v2 studies)

All calibration-based methods derive from ONE corpus per model — the 198
GPQA-Diamond prompts (`artifacts/prompt_rows/gpqa_diamond.csv`), the same
text that trains the gpqacc64k codebooks. Delivery differs per artifact
type, matching each pipeline's own recipe:

- **Rotations**: per-prompt short capture (198 prompts ≈ 48K tokens; the
  authors' own recipe used 83 ≈ 20K) → `compute_kv_rotation qqt_sst
  r_h_pbr`. Built by `pipelines/oscar_e2e/recalibrate_rotations.sh`.
- **K codebooks**: the same 198 prompts concatenated + cycled into 8×64K
  sequences (RoPE-position coverage) → Samuel's trainer. Rotation-
  independent (the bundle carries its own forward map), so codebooks are
  NOT retrained when rotations change.
- **Calibration-free methods** (bf16, TurboQuant, QuaRot, Naive) have
  nothing to match and are reused from v1 (`reused_methods` in the spec).

v1 provenance, for the record: Llama rotations = 50 GPQA prompts (ours),
Qwen rotations = authors' released (83 prompts, their capture). The Qwen
v1 int2 column stays reported as "production OSCAR as released".
