# Vendored dependencies

Third-party libraries the project uses. All three live here as separate
git clones (each has its own `.git`) and are **gitignored from the parent
repo** so we don't accidentally commit their files. On a fresh clone of
this project, recreate the three repos:

```bash
git clone https://github.com/NVIDIA/kvpress.git           vendor/kvpress
git clone https://github.com/IBM/turboquant-pytorch.git   vendor/turboquant-pytorch
git clone https://github.com/jy-yuan/KIVI.git             vendor/kivi

# Underscore-name symlink so `import turboquant_pytorch` resolves (the
# hyphen in the source dir isn't a valid Python module name):
ln -sfn turboquant-pytorch vendor/turboquant_pytorch
```

## Pinned commits (last verified)

| repo | URL | pinned commit | role |
|---|---|---|---|
| `kvpress` | `github.com/NVIDIA/kvpress` | `d8349a9` | Press base classes + `evaluation/evaluate.py` harness. Our `evaluate.py` and `evaluate_registry.py` have local extensions (see `vendor/kvpress/git diff`). |
| `turboquant-pytorch` | `github.com/IBM/turboquant-pytorch` | `03e6112` | TurboQuant V3 K/V compressors. Used as a baseline and as the `v_turboquant` V-side compressor inside JointQK. |
| `kivi` | `github.com/jy-yuan/KIVI` | `876b4d2` | KIVI int2/int3/int4 group quantization. Used only as a baseline in the bench sweep (no local modifications). |

If you need exact reproducibility, check out the pinned commits:

```bash
git -C vendor/kvpress         checkout d8349a9
git -C vendor/turboquant-pytorch checkout 03e6112
git -C vendor/kivi            checkout 876b4d2
```

## Local modifications

Only `vendor/kvpress/` has uncommitted modifications relative to upstream
(see `git -C vendor/kvpress diff --stat`). They are concentrated in three
files:

- `pyproject.toml` — adds the `pytorch-cu128` uv index.
- `evaluation/evaluate.py` — adds `EvaluationConfig.exclude_indices_file`
  (calibration-leak prevention), `EvaluationConfig.press_kwargs`,
  class-instantiation in `_setup_press`, and the dataset-loader exclude
  logic.
- `evaluation/evaluate_registry.py` — registers `JointQKPress`,
  `TurboQuantPress`, `KIVIPress` (from `experiments.toolkit.*`) as
  classes in `PRESS_REGISTRY`.

`vendor/turboquant-pytorch/` and `vendor/kivi/` are unmodified.
