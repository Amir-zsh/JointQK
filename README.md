# KV-Cache Compression Research

A research codebase studying how to **compress the key half of the KV cache** in transformer LLMs without losing the attention behaviour the model relies on. We test methods that calibrate offline from a small prompt corpus, transform keys into a basis chosen by those statistics, scalar-quantize each coordinate independently, and decompress on the fly when future queries need to compute attention.

## Background

Modern transformer LLMs cache the keys (`K`) and values (`V`) from prefill so subsequent generated tokens can attend to the prompt without re-running the forward pass. This **KV cache** grows linearly with context length and dominates GPU memory at long contexts — gigabytes per active sequence on 8B-class models with 30k-token prompts. Cheap, high-quality KV-cache compression is one of the most consequential levers for serving long-context LLMs.

This project specifically investigates compression of the **K** half: per-(layer, kv_head) linear bases learned from second-moment statistics, paired with per-coordinate bit allocations. Compressing V is studied separately and is largely orthogonal.

## Approach

Every method we evaluate fits a common three-phase template:

1. **Calibration (offline, once).** Observe queries and keys on a small representative corpus; compute per-(layer, kv_head) second moments such as $\Sigma_Q$, $\Sigma_K$, and the cross-moment $C_{QK}$.
2. **Compression (online, per prefill).** Linearly transform each prefill key into a basis chosen by the calibration. Independently scalar-quantize each coordinate of the transformed key to a small Lloyd–Max codebook scaled to that coordinate's standard deviation. Compressed keys replace raw keys in the cache.
3. **Reconstruction (online, per generated token).** When a future query arrives, dequantize the cached keys and compute attention scores against them in the place of the originals.

Two design knobs govern this template:

- **The basis.** Which linear transform encodes "the directions that matter for attention"? Random rotation, Q-eigenbasis, CCA, joint Q-K eigenbases, …
- **The bit allocation.** How are bits distributed across the coordinates of that basis? Uniform, hard top-r truncation, continuous water-fill on a per-coordinate weight, …

Different (basis × allocation) choices produce dramatically different attention top-1 retention at the same average bit budget. The codebase systematically evaluates them on real LLMs and real long-context tasks.

## What's evaluated

- **Models.** Public open-weight LLMs (e.g. Qwen3-class). The compression methods apply per (layer, kv_head); architectural specifics matter for calibration but not for the methodology.
- **Tasks.** Long-context benchmark suites (LongBench-E and similar). Calibration and evaluation use prefill activations from the bundled examples.
- **Metrics.**
  - *Attention top-1 / top-5 retention* — fraction of queries whose argmax (or top-5) over the compressed keys matches the uncompressed baseline. The production metric.
  - *Q-weighted geometry distortion* — $\mathbb{E}\!\left[(k - \hat{k})^\top M_q (k - \hat{k})\right] / d$, the natural rate-distortion target with $M_q = \mathbb{E}[q\, q^\top]$.
  - *Logit MSE / cosine* and decode-phase top-1 against the compressed prefill cache (production scenario).

The full catalogue of methods, headline numbers, and per-experiment analyses live in the **stage notes** under `notes/stage1/`. The latest line of work has its own deep-dive folder and shareable summaries (one-pager, two-pager, full report).

## Repository layout

```
.
├── experiments/             # Drivers, gates, and toolkit code per research stage
│   ├── stage1/              # Active research: calibration, real quantization, generalization, decode
│   │   ├── toolkit/         # Reusable building blocks (Lloyd-Max, water-fill, per-coord compressor, …)
│   │   ├── scripts/         # Launchers, mergers, chart generators, gate runners
│   │   ├── gates/           # Sanity-check assertions on per-experiment outputs
│   │   ├── notebooks/       # Exploratory notebooks
│   │   └── logs/            # Run logs and heartbeats (gitignored)
│   └── stage2/              # Reserved for the next research stage
├── artifacts/               # Per-stage outputs (calibrated stats, run summaries, charts, gates)
│   └── stage1/
├── notes/                   # Research notes, organised by role
│   ├── core/                #   Project framing and execution roadmap
│   ├── reference/           #   Standalone derivations and references
│   ├── stage1/              #   Per-stage findings, deep-dives, shareable summaries
│   └── README.md            #   Index of the notes directory
├── kvpress/                 # Vendored Apache-2.0 KV-compression library (see kvpress/LICENSE)
├── turboquant-pytorch/      # Vendored TurboQuant baseline (see turboquant-pytorch/LICENSE)
├── environment.yml          # Conda environment specification
└── README.md
```

## Setup

The repository ships a conda environment specification. Create and activate it once:

```bash
conda env create -f environment.yml
conda activate kv-rd
```

This installs PyTorch (with CUDA), the scientific Python stack, and the dependencies needed by `kvpress`. The vendored `kvpress/` package is installed in editable mode by the same spec.

## Where to start reading

For a guided tour of the project:

1. **`notes/core/kv_cache_rate_distortion_proposal.md`** — high-level research framing and thesis.
2. **`notes/core/research_plan.md`** — stage-by-stage execution roadmap.
3. **`notes/README.md`** — index of all research notes, grouped by role.
4. **`notes/stage1/stage1_experiments_and_findings.md`** — synthesis of the current stage's results.

Each individual stage has its own per-experiment review notes plus a shareable summary set (one-pager / two-pager / full report) for colleagues outside the project. The latest stage's summary is the recommended on-ramp once you've read the framing.

## Reproducing experiments

Experiments are organised under `experiments/<stage>/`. Each stage has:

- A driver script that consumes a calibration bundle and emits per-(example, layer, kv_head) metrics.
- Launcher scripts under `scripts/` that parallelise runs across GPUs.
- Gate scripts under `gates/` that assert basic sanity on the outputs.
- A merge/aggregation step that lifts per-run output into canonical summary JSONs and chart-ready row PTs.

See `notes/stage1/stage1e_cca_vs_waterfill_note.md` (or the most recent stage note) for the exact command sequence used to regenerate the canonical artifacts in `artifacts/stage1/`.

## Vendored dependencies

| Path | Source | License |
|---|---|---|
| `kvpress/` | NVIDIA's [kvpress](https://github.com/NVIDIA/kvpress) library, used as a thin benchmarking adapter | Apache-2.0 (`kvpress/LICENSE`) |
| `turboquant-pytorch/` | TurboQuant baseline implementation | see `turboquant-pytorch/LICENSE` |

These are vendored rather than installed from PyPI to pin specific versions; their original licenses apply unchanged.

## Citation

A paper distilling this line of research is in preparation. Once published, citation details will appear here. In the meantime, please cite the relevant per-stage notes under `notes/stage1/` if you build on this work.

## License

The project's first-party code is provided as-is for research use. Vendored dependencies retain their own licenses (see the table above). A formal project-wide license will be added with the public release.
