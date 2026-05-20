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

The full catalogue of methods, headline numbers, and per-experiment analyses live under `notes/`.

## Repository layout

```
.
├── experiments/
│   ├── calibration/         # K/V capture + pooled second-moment computation
│   ├── bench/               # Downstream LongBench/RULER F1 sweep (worker + launchers)
│   ├── analysis/            # JointQK F1-inversion probes (logit-KL, Σ_Q drift, HTML report)
│   ├── calibration_stability/  # Basis-stability ablation sweeps
│   ├── toolkit/             # Reusable building blocks (Lloyd-Max, water-fill, JointQK press, …)
│   ├── scripts/             # Calibration-build consumer utilities
│   ├── eval/                # Aggregators (LongBench, RULER, decode-scope, integrity)
│   ├── benchmarks/, data/   # Vendored kvpress data adapters
│   ├── tests/               # Press parity, regression fingerprint
│   └── logs/                # Run logs (gitignored)
├── artifacts/
│   ├── calibration/         # K/V captures + pooled second moments
│   ├── bases/               # Joint Q-K basis files (cca_stats_*.pt)
│   ├── v_bases/             # V-side basis files
│   ├── bench/, bench_llama_*/  # Downstream F1 results
│   ├── decode_q_captures_llama/, q_distribution_shift/  # Llama analysis outputs
│   └── query_stats_longbench_under4k*  # Captured query stats (large, protected)
├── notes/                   # Research notes
│   ├── core/                #   Project framing and execution roadmap
│   ├── reference/           #   Standalone derivations and references
│   └── README.md            #   Index of the notes directory
├── vendor/                  # Vendored third-party libraries
│   ├── kvpress/             #   Apache-2.0 KV-compression library (NVIDIA)
│   ├── turboquant-pytorch/  #   TurboQuant baseline
│   ├── turboquant_pytorch   #   symlink → turboquant-pytorch (so `import` works)
│   └── kivi/                #   KIVI baseline
└── README.md
```

## Setup

The repository ships a conda environment specification. Create and activate it once:

```bash
conda env create -f environment.yml
conda activate kv-rd
```

This installs PyTorch (with CUDA), the scientific Python stack, and the dependencies needed by `kvpress`. The vendored `vendor/kvpress/` package is installed in editable mode by the same spec.

The vendored `vendor/turboquant-pytorch/` directory uses a hyphen, which is not a valid Python module name. The repo ships a symlink `vendor/turboquant_pytorch → turboquant-pytorch` so `import turboquant_pytorch` resolves; recreate it after every fresh clone if missing:

```bash
ln -sfn turboquant-pytorch vendor/turboquant_pytorch
```

Calibration artifacts (`artifacts/{v_bases,bases/llama31_8b,query_stats_longbench_under4k*}/` and any `*.pt`) are gitignored. Regenerate with the scripts under `experiments/scripts/` after capturing a fresh Q/K/V bundle.

## Where to start reading

For a guided tour of the project:

1. **`notes/core/kv_cache_rate_distortion_proposal.md`** — high-level research framing and thesis.
2. **`notes/core/research_plan.md`** — stage-by-stage execution roadmap.
3. **`notes/README.md`** — index of all research notes, grouped by role.
4. **`notes/experiments_and_findings.md`** — synthesis of the current stage's results.

Each individual stage has its own per-experiment review notes plus a shareable summary set (one-pager / two-pager / full report) for colleagues outside the project. The latest stage's summary is the recommended on-ramp once you've read the framing.

## Reproducing experiments

The pipeline:

1. **Calibrate** with `experiments/calibration/` — capture K/V from a prompt corpus and pool per-(layer, kv_head) second moments.
2. **Build bases** with `experiments/scripts/build_calibration_artifacts_from_pool.py` — produce a joint Q-K basis file under `artifacts/bases/`.
3. **Bench** with `experiments/bench/launch.sh` — parallelise the LongBench/RULER worker across GPUs; each cell emits `metrics.json`.
4. **Aggregate** with `experiments/eval/aggregate_*.py` — lift per-cell JSONs into canonical summary tables.
5. **Analyse** with `experiments/analysis/` — fidelity probes (K-MSE, top-1, attention-KL, logit-KL, Σ_Q drift) against the same bases.

See `notes/bench_results_report.md` and `notes/jointqk_disconnect_investigation.md` for the most recent runs and findings.

## Vendored dependencies

| Path | Source | License |
|---|---|---|
| `vendor/kvpress/` | NVIDIA's [kvpress](https://github.com/NVIDIA/kvpress) library, used as a thin benchmarking adapter | Apache-2.0 (`vendor/kvpress/LICENSE`) |
| `vendor/turboquant-pytorch/` | TurboQuant baseline implementation | see `vendor/turboquant-pytorch/LICENSE` |
| `vendor/kivi/` | KIVI baseline | see `vendor/kivi/LICENSE` |

These are vendored rather than installed from PyPI to pin specific versions; their original licenses apply unchanged.

## Citation

A paper distilling this line of research is in preparation. Once published, citation details will appear here. In the meantime, please cite the relevant per-stage notes under `notes/` if you build on this work.

## License

The project's first-party code is provided as-is for research use. Vendored dependencies retain their own licenses (see the table above). A formal project-wide license will be added with the public release.
