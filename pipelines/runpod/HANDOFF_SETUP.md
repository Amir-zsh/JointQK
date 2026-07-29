# RunPod setup handoff — QuaRot + TurboQuant grid

*What to transfer to a pod, in what order, and how to prove it landed before
spending GPU time. Written 2026-07-28.*

Covers five protocols (`pipelines/runpod/protocols/*.json`) across three model
families, each with six arms: `bf16`, `oscar_int2`, `vq2`, `quarot`,
`turboquant_int2`, `turboquant_k3v3`.

---

## 0. The one-paragraph version

A pod needs **three things**: the two git repos, the two Python envs, and the
gitignored artifact payload (rows, rotations, codebooks, sim-quant bundles).
The payload does **not** travel with `git clone` — `artifact_manifest.json` is
the contract, and `manifest.py verify` is the proof. Run `bootstrap.sh` last;
it fails loudly on anything missing, including the LiveCodeBench test cases
that would otherwise only fail *after* generation.

---

## 1. Repos — note they are TWO, on DIFFERENT branches

```bash
git clone git@github.com:Amir-zsh/JointQK.git teamily-project
cd teamily-project
git checkout h100-vq2-opt-and-fig4          # protocols, run_cell, builders, gates

# the engine is a SEPARATE repo, NOT a submodule, and is gitignored here
git clone git@github.com:Amir-zsh/VQ-SGLang.git vendor/OSCAR-vq
git -C vendor/OSCAR-vq checkout oscar-gptoss-vq-cleanup   # HEAD 671ce4fe9
```

> The branch names deliberately do **not** match. `sim_quant.py` and the
> hybrid-SWA fix live only in **VQ-SGLang / oscar-gptoss-vq-cleanup**; they will
> never appear in JointQK on any branch, because `.gitignore:4` excludes
> `vendor/OSCAR-vq/`. If you go looking for the engine change on the h100
> branch you will not find it.

Also clone/keep `vendor/kvpress` — `kvq/benchmarks/evaluate_registry.py` is a
shim over it and imports the NIAH/GPQA/math scorers directly from the vendored
copy.

## 2. Python environments

Two, deliberately different; do not merge them. No lockfile exists, so pin by hand:

| env | used for | python | torch | triton | transformers |
|---|---|---|---|---|---|
| `.venv` | client, exporters, scorers | 3.12 | 2.11.0+cu128 | 3.6.0 | 5.2.0 |
| `.venv-oscar` | the SGLang server only | 3.12 | 2.9.1+cu128 | 3.5.1 | 5.3.0 |

`.venv` also needs `datasets`, `math-verify`, `pandas`, `numpy`.
`.venv-oscar` installs sglang **editable from the vendored tree**, and
`serve_oscar.sh` additionally prepends
`vendor/OSCAR-vq/sglang-research/python` to `PYTHONPATH` so the source shadows
the install — both mechanisms are load-bearing.

CUDA 12.8 + gcc-12 are required for the Triton JIT. `serve_oscar.sh` hardcodes
two conda paths (`CUDA128`, `GCC12`) — **edit these for the pod** or the JIT
dies with `ninja exited with status 1`.

## 3. Artifact payload — what to rsync

`git clone` gives code only. Everything below is gitignored.

```bash
# on the SOURCE host (must hold the complete payload):
rsync -av --files-from=<(python - <<'PY'
import json
m=json.load(open("pipelines/runpod/artifact_manifest.json"))
for g in m["groups"].values():
    for i in g: print(i["path"])
PY
) ./ pod:/workspace/teamily-project/

# on the POD, before any GPU work:
python pipelines/runpod/manifest.py verify --group all    # or qwen3_8b | gptoss | gptoss_mxfp4 | llama
```

Per-group payload size:

| group | files | size |
|---|---:|---:|
| `qwen3_8b` | 20 | 750 MB |
| `gptoss` | 20 | 1068 MB |
| `gptoss_mxfp4` | 4 | 10 MB |
| `llama` | 6 | 92 MB |

> **`manifest.py make` must only ever run on a host holding the COMPLETE
> payload.** It regenerates from local files, so running it on a partial host
> silently deletes entries — the contract shrinks to whatever happened to be
> there. This was hit on 2026-07-28: a `make` on a host missing the gpt-oss
> NIAH sets and the mxfp4 artifacts dropped 21 entries (~1 GB) before it was
> caught by diffing against `HEAD`. To ADD paths, edit `GROUPS` in
> `manifest.py` and run `make` on the complete host only.

### Sim-quant bundles: build, don't transfer

The `turboquant_k3v3` arm needs two `.pt` bundles. They are gitignored and
cheap to regenerate, so build them on the pod rather than shipping them:

```bash
python pipelines/oscar_e2e/build_simquant_turboquant.py --k-bits 3 --v-bits 3 \
    --head-dim 128 --layers 36 --out artifacts/simquant/simquant_turboquant_d128_L36_k3v3.pt
python pipelines/oscar_e2e/build_simquant_turboquant.py --k-bits 3 --v-bits 3 \
    --head-dim 64  --layers 12 --out artifacts/simquant/simquant_turboquant_d64_L12_k3v3.pt

python pipelines/oscar_e2e/gate_simquant.py        # MUST print GATE PASS
```

The gate compares the served quantizer against `TurboQuantPress` on identical
input; it must read `0.00e+00`. Anything larger means the rotations/centroids
or the normalisation convention drifted, and the baseline row is measuring a
quantizer nobody else ran. `head_dim 128 × 36L` serves Qwen and Llama;
`head_dim 64 × 12L` serves gpt-oss (12 = its full-attention layers).

## 4. Bootstrap

```bash
bash pipelines/runpod/bootstrap.sh --group <qwen3_8b|gptoss|gptoss_mxfp4|llama>
```

Checks both pythons, sglang import, the artifact manifest, the models, and —
new — the **LiveCodeBench test cases**. LCB tests are *not* embedded in the
prompt rows (28k cases would balloon `predictions.csv`), so
`code_scorers.load_lcb_tests()` pulls six shards from the Hub at **scoring**
time, i.e. after the GPU work is paid for. The bootstrap warms that cache so a
network/auth failure surfaces during setup instead of destroying a finished
cell's score. HumanEval needs nothing — its asserts travel inside the row.

Models: `huggingface-cli download` (or `download_models.sh --token <HF_TOKEN>`)
for `Qwen/Qwen3-8B`, `meta-llama/Llama-3.1-8B-Instruct`,
`unsloth/gpt-oss-20b-BF16` and/or `openai/gpt-oss-20b`.

## 5. Running cells

```bash
bash pipelines/runpod/run_cell.sh \
    --protocol pipelines/runpod/protocols/qwen3_8b_v1.json \
    --arm turboquant_k3v3 --task humaneval --gpus 0 [--shard 0/4] [--port 30901]
```

One cell = one (arm × task). Parallelism is invoking it N times with disjoint
`--gpus`. Resume-safe: a cell whose `metrics.json` exists is skipped.

**TP=2 is mandatory for gpt-oss** (`--gpus 0,1`). Qwen and Llama run TP=1.

### Arm semantics — read before quoting anything

| arm | what it is |
|---|---|
| `quarot` | `--int2plain` + full Hadamard over head_dim, unclipped. `hadamard_order` **must equal head_dim**: 64 for gpt-oss, 128 for Qwen/Llama. |
| `turboquant_int2` | TurboQuant at **2 bits** (Lloyd-Max levels). Collapses in practice. Keep as a datapoint; it is **not** the paper row. |
| `turboquant_k3v3` | TurboQuant **K=3/V=3**, simulated on a BF16 pool. **This is the Table-2 comparison.** |

Label the last one **"TurboQuant, MSE stage only (V3 community variant; QJL
omitted)"** — the vendored implementation drops the QJL stage, which is
precisely what debiases inner products, and attention logits are inner products.

### Sampling is per-model and must never be cross-copied

Each model's own `generation_config.json`:

| model | temperature | top_p | top_k |
|---|---:|---:|---:|
| Qwen3-8B / 4B-Thinking | 0.6 | 0.95 | 20 |
| Llama-3.1-8B-Instruct | 0.6 | 0.9 | — (`-1`) |
| gpt-oss-20B | 1.0 | 1.0 | −1 (off) |

gpt-oss's config specifies only `do_sample`; 1.0/1.0 is the vendor
recommendation. Applying one model's values to another is the same class of
error as the legacy `T=1.0 / top_k=40` GLM-branch protocol that invalidated an
earlier Qwen grid. The protocols already encode this — do not "harmonise" them.

## 6. Verifying an arm actually did what it claims

**sim-quant** — the serve log must show, per TP rank:

```
[sim_quant] ACTIVE: K=3b V=3b head_dim=<D> layers=<L> from <bundle>
[sim_quant] eager_writes=N layers_seen=L/L running NMSE: K(3b) 0.03xx V(3b) 0.03xx  <- nonzero => not BF16
```

Non-zero NMSE proves it is not silently serving BF16 under a quantized arm's
name. `layers_seen=L/L` proves full coverage. On gpt-oss expect **12/12**, not
24/24 — the sliding-window layers are deliberately excluded (see §7).

The serve log is at `logs/runpod_<arm>_<task>[_s<shard>]_serve.log`, **not**
under the cell directory.

**Everything else** — `resolved_server_info.json` in each cell records the
server config as actually launched; `provenance.json` records the protocol name
and its sha256, so two cells are comparable iff their protocol hashes match.

## 7. Traps

1. **`CTX` and `MEM_FRAC` are not env-readable** in `serve_oscar.sh` — they are
   assigned unconditionally. Passing `CTX=131072` via env silently serves a
   73728 window. Wrong answer, no error. Pass `--ctx` / `--mem-frac` as flags.
2. **`--out` is relative to the repo root**, and if `metrics.json` already
   exists the client opens `io_log.jsonl` in `"w"` and destroys the finished
   run. Keep every driver's skip-if-metrics guard.
3. **`limit_rows` is task-balanced only when rows carry a `task` field.** NIAH
   has 8 subtasks so it balances correctly; math500 / HumanEval / LCB have no
   `task` field, so `limit_rows` degenerates to a **plain prefix**. Fine for a
   smoke, never quotable.
4. **`lcb_v6_sub256_*` is a contiguous chronological prefix** of LCB's 1055. On
   Qwen that subset flipped the int2-vs-bf16 ordering by 11 points versus the
   full set. It is a fixed internal comparison — every arm sees the same rows —
   but not a LiveCodeBench score.
5. **gpt-oss BPE is 3.25, not 3.125**, for K3V3: head_dim 64 doubles the
   fp16-norm overhead (16/64 vs 16/128). Do not quote one BPE across models.
6. **`--kv-cache-quant-group-size` raises on hybrid-SWA models**, so gpt-oss
   protocols set `quant_group_size: 0` (omits the flag) and
   `absorb_v_rotation: 0`. Both are mandatory, not tuning.
7. **Split-K must be matched when comparing latency** — 48 for vq2, 8
   otherwise is the per-arm default. int2 at 64K is 24.6 ms/tok at splits=8 but
   20.9 at 48; an unmatched comparison invents a vq2 win. Irrelevant for
   accuracy.

## 8. The hybrid-SWA fix (why gpt-oss needed engine work)

`SWAKVPool.set_kv_buffer` forwards a **pool-local** layer id to each inner
pool, so on gpt-oss the SWA and full-attention pools both count `0..11` and
would index the same sim-quant bundle rows — applying one layer's constants to
a different layer, silently. The engine marks the SWA inner pool
(`_is_swa_inner_pool`) and skips it by default
(`SGLANG_SIMQUANT_SKIP_SWA=1`), which also matches `oscar_int2`/`vq2`, where
only the 12 full-attention layers are quantized.

This collision cannot occur on a flat model, so no upstream gate covers it.
`gate_simquant.py` checks both geometries and asserts per-layer rotations
differ; the live proof is `layers_seen=12/12` in a gpt-oss serve log.

Everything is env-gated: **unsetting `SGLANG_SIMQUANT_PATH` is a complete
functional revert**, so no existing number depends on the change.

## 9. Known-good reference numbers

For sanity-checking a fresh pod. Qwen3-8B, from `metrics.json` on disk:

| arm | GPQA | AIME25 | MATH500 | HumanEval |
|---|---:|---:|---:|---:|
| bf16 | 57.17 | 70.00 | 97.00 | 91.71 |
| oscar_int2 | 53.54 | 66.25 | 96.72 | 90.85 |
| vq2 | 55.05 | 64.58 | 96.92 | 92.07 |

gpt-oss-20B (`unsloth/gpt-oss-20b-BF16`, 200-row math500, 4 seeds):
bf16 91.50 · vq2 90.25 · oscar_int2 41.88.

Smoke-level, single sample — treat as "works / collapsed", not measurements:
Qwen3-8B `turboquant_k3v3` NIAH-8K 24 rows greedy = 95.83;
gpt-oss `turboquant_k3v3` math500 rows 0–23 = 58.33 (vs bf16 91.67, int2 35.42
on the same rows).

**There is no RNG seed anywhere in the harness.** `--samples K` fires K
unseeded passes; you can match the distribution, never individual numbers.
