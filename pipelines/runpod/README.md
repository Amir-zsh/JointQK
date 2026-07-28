# Running experiments on RunPod

Layout: **image = dependencies** (immutable, versioned tag) · **volume =
code + data** (git clone + rsynced artifacts + HF cache at `/workspace`) ·
**results pulled back per cell**. Code fixes are a `git pull` on the pod,
never an image rebuild.

What lives where:

| Layer | Contents | Size |
|---|---|---|
| image `kvq-runpod:v1` | cu130 torch base + nvcc 13, `/opt/venv-oscar` (engine deps, no sglang — the vendored fork shadows via PYTHONPATH), `/opt/venv-client` (lockfile), sshd, `pod_init.sh` | 25.7 GB |
| volume `/workspace` | both repos (git), artifact payload (rsync), HF cache | qwen3_8b payload **546 MB** (codebook 97 MB, NIAH-64K rows 211 MB); gptoss 854 MB; llama 92 MB; Qwen3-8B weights ~16 GB |
| synced back | `artifacts/runpod/<protocol>/` — metrics + provenance per cell | KBs per cell |

Every command below is labeled with WHERE it runs: **[host]** = the source
machine holding the artifacts (the H100 box), **[console]** = the RunPod web
UI, **[pod]** = ssh'd into the pod.

## 1. [host] Build & push the image — once

```bash
# from the repo root (the .dockerignore keeps artifacts/ out of the context)
docker build -f docker/Dockerfile.runpod -t <registry>/kvq-runpod:v1 .
docker push <registry>/kvq-runpod:v1
```

## 2. [console] Template + pod — once per template

Template: the pushed image, container disk ≥ 40 GB, volume ≥ 100 GB mounted
at `/workspace`, expose TCP 22. Template env: `POD_INIT=1`,
`GIT_TOKEN=<PAT with read on both repos>`, `IMAGE_TAG=<the tag>`, optionally
`MAIN_REF`/`ENGINE_REF` to pin SHAs. Deploy a pod, copy the ssh command from
its Connect tab.

## 3. Provision — once per volume, strictly in this order

```bash
# [pod] code (skip if POD_INIT=1 already cloned; rerun any time to update)
/opt/runpod/pod_init.sh

# [host] artifacts — ~546 MB for the Qwen group, a few minutes
RSYNC_SSH="ssh -p <port> -i <key>" \
bash pipelines/runpod/sync_results.sh push-artifacts root@<pod-ip> qwen3_8b

# [pod] models — ~16 GB into /workspace/hf, survives pod restarts
bash pipelines/runpod/download_models.sh --token <HF_TOKEN> Qwen/Qwen3-8B

# [pod] gate — MUST print BOOTSTRAP READY before any cell runs
bash pipelines/runpod/bootstrap.sh --group qwen3_8b
```

## 4. [pod] Run cells

One invocation = one (arm, task) cell or one shard of it, running one server
on the GPUs you hand it. **Sequential per GPU, parallel across GPUs** — there
is deliberately no scheduler; you decide what runs where. TP is always
explicit (protocol `tp` field), never inferred.

```bash
cd /workspace/teamily-project
P=pipelines/runpod/protocols/qwen3_8b_v1.json

# smoke first on any new pod (24 NIAH-8K rows, greedy, ~5 min for both)
bash pipelines/runpod/run_cell.sh --protocol $P --arm bf16 --task niah_smoke --gpus 0
bash pipelines/runpod/run_cell.sh --protocol $P --arm vq2  --task niah_smoke --gpus 0

# e.g. four cells at once on a 4-GPU pod (tmux recommended)
bash pipelines/runpod/run_cell.sh --protocol $P --arm bf16       --task gpqa --gpus 0 &
bash pipelines/runpod/run_cell.sh --protocol $P --arm oscar_int2 --task gpqa --gpus 1 &
bash pipelines/runpod/run_cell.sh --protocol $P --arm vq2        --task gpqa --gpus 2 &
bash pipelines/runpod/run_cell.sh --protocol $P --arm vq2        --task math500 --gpus 3 &
wait

# shard a long cell across 4 GPUs (disjoint rids; merge afterwards)
for i in 0 1 2 3; do
  bash pipelines/runpod/run_cell.sh --protocol $P --arm vq2 --task niah_65536 \
      --gpus $i --shard $i/4 &
done; wait
/opt/venv-client/bin/python pipelines/oscar_e2e/merge_shards.py \
    --root artifacts/runpod/qwen3_8b_v1/vq2
```

Every cell writes `metrics.json` + `provenance.json` (protocol hash, both
repo SHAs + dirty flags, artifact hashes, resolved server config, GPU name)
+ `resolved_server_info.json`. A cell whose `metrics.json` exists is skipped
(resume guard) — if a pod dies mid-cell, redeploy and rerun the same
commands; finished cells skip, the interrupted one resumes from its row log.
The echo gate hard-fails the cell if the server's resolved config disagrees
with the protocol — that is intentional; fix the cause, not the gate.

## 5. [host] Pull results — any time, including mid-run

```bash
RSYNC_SSH="ssh -p <port> -i <key>" \
bash pipelines/runpod/sync_results.sh pull-results root@<pod-ip> qwen3_8b_v1
```

## Local testing on the H100 host

The whole chain runs locally in the same image before touching a pod:

```bash
docker run --rm --gpus '"device=1"' --shm-size 16g \
  -v $PWD:/workspace/teamily-project -v /raid/amir/.cache/huggingface:/workspace/hf \
  -e PYTHONDONTWRITEBYTECODE=1 -e IMAGE_TAG=kvq-runpod:v1 \
  -w /workspace/teamily-project kvq-runpod:v1 bash -c '
    bash pipelines/runpod/bootstrap.sh --group qwen3_8b &&
    bash pipelines/runpod/run_cell.sh --protocol pipelines/runpod/protocols/qwen3_8b_v1.json \
        --arm vq2 --task niah_smoke --gpus 0'
```

Caveat: the container runs as root, so cell outputs under `artifacts/runpod/`
come back root-owned — clean them up via a container `rm`, not from the host.

## Protocol discipline

`protocols/qwen3_8b_v1.json` is the single source of truth for every knob;
its sha256 is stamped into each cell. Two cells are comparable iff their
protocol hashes match. Changing any knob = a new protocol version file, not
an edit-in-place. Open decisions deliberately surfaced as fields:
`exact_chunked_prefill` (chunked-prefill error confound; false = published
serving default) and the NIAH sampling regime (currently T=0.6/3-seed
matching the existing 8–64K numbers).

Known data gap: `artifacts/prompt_rows_proto/math500_think32k_qwen.jsonl` is
absent from this host (the similarly-named file in `prompt_rows/` is the
8K-cap variant, different hash) — sync it from the origin tree or decide on
regeneration before the math500 cell can run.
