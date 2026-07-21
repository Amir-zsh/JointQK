# Provenance — Samuel's group-VQ snapshot

Source repo: /vault/samuel/efficient-llm/JointQK (branch entropy_coding, HEAD 52e4875; all files below were
UNCOMMITTED working-tree state at snapshot time — no source commit SHA exists).
Snapshot date: 2026-07-15T01:05:42Z. Copied read-only with author's
permission (user-authorized copy-with-attribution); the source repo was not modified.

| file | source path | source mtime | sha256 |
|---|---|---|---|
| group_vq_codec.py | /vault/samuel/efficient-llm/JointQK/entropy_coding/group_vq_codec.py | 2026-07-14 14:49:19 | 4ffd14f0e0dd94fed27da7abf1eb64950ab7f341acdbd93a06d066b8375590ca |
| train_group_vq_alloc.py | /vault/samuel/efficient-llm/JointQK/entropy_coding/train_group_vq_alloc.py | 2026-07-14 14:49:43 | 1ff5ddd268abf3868c5a8c2b93e0ae3fcf23facfe1744cada93ea0e9df5f10c4 |
| run_pca_ec_deadzone.py | /vault/samuel/efficient-llm/JointQK/entropy_coding/run_pca_ec_deadzone.py | 2026-07-14 13:05:28 | a21c9db615499370928742d2013fc0a45855b91df25cf42fe63736e0abd4512a |
| vq_press.py | /vault/samuel/efficient-llm/JointQK/kvq/presses/vq_press.py | 2026-07-14 14:49:50 | 9f74f379a307a4f052db49c6dbdc66cd38364e8addf0fd747fe98cebf79e3ac1 |
| vqa_G4_strat_flat_fair_fp8.pt | /vault/samuel/efficient-llm/JointQK/entropy_coding/vqa_G4_strat_flat_fair_fp8.pt | 2026-07-11 14:09:11 | 8e216e60f2a404ba07fece60311fc090d500edc546906037eb950fb14ad105c1 |
| vqa_G4_strat_flat_llama_fp8.pt | /vault/samuel/efficient-llm/JointQK/entropy_coding/vqa_G4_strat_flat_llama_fp8.pt | 2026-07-11 14:09:13 | 72bc94492f087c5932493fc97a8a1dde977feecdef00484790c24b16a63408a6 |

## NIAH benchmark data (added 2026-07-15)

`artifacts/niah_bench/<ctx>/test.parquet` (gitignored) are parquet conversions
of Samuel's regenerated RULER-NIAH datasets (`gen_niah.py`, uncommitted) read
from `/vault/samuel/efficient-llm/JointQK/artifacts/niah/{8192,16384,32768,65536}`
(HF save_to_disk format, 800 rows = 8 subtasks x 100 each). Same schema as
simonjegou/ruler (which caps at 16k); scorer = vendored ruler scorer.

## Snapshot refresh (2026-07-17) — commit 3c65507 (PINNED)

Source tree now committed: entropy_coding commit 3c65507 ("per-token-norm VQ,
G=4 as deployment default, V=INT2 benchmark"). Files above refreshed to the
committed versions; added bench_vint2.py and the per-token-norm long-context
codebook:

| file | sha256 |
|---|---|
| bench_vint2.py | 3c49584c401ca1a9c85ce03dda313977dac60b6e6895a0f6bd343f289cb5b60d |
| vqa_G4_strat_flat_ptn_gpqacc64k_fp8.pt | b277bddf16f753c5fc31d8e8f1db05c0aaca1007d217ead7b7467214297e6b63 |

## Handoff integration (2026-07-20)

Applied Samuel's HANDOFF_engine.patch + HANDOFF_pipeline.patch from
/vault/samuel/efficient-llm/oscar_vq2/ (his clone of vq2-longhorizon; see
HANDOFF_AMIR.md there). Copied tooling: pipelines/eval/{code_scorers,
code_exec,export_code_rows,archive_raw_generations}.py, pipelines/oscar_e2e/
{run_baseline_plain.sh,proto_grid_worker.sh,score_running.py,
decode_stream_bench.py} (ROOT parameterized, otherwise as-is), and code/proto
prompt rows (gitignored dirs artifacts/prompt_rows_{code,proto}/).

| file | sha256 |
|---|---|
| vqv_G4_strided_gpqa_engine.pt (V codebook, engine basis: forward=identity, mean=0, STRIDED groups, K=256 G=4 ptn) | dc2db8e35edcaede1b34cd066019b8910aecd7d576b78f446e5e6eab26ecc4c7 |
