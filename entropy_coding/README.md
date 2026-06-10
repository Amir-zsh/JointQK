# Entropy coding paged codec

Self-contained slice of the paged KV-cache codec, fused down to three files.

## In this folder
- `kvq_codec.py` — fused codec stack:
  - `PageCodec` (constriction, CPU) — reference.
  - `PageCodecRANS` (interleaved rANS, CPU) — same pipeline, swapped per-page coder; reconstructs the SAME snapped indices, so bit-exact `K_hat` vs `PageCodec` proves the rANS stream format.
  - `PageCodecRANSCUDA` (GPU encode + decode) — decode-identical to `PageCodecRANS`.
  - factories + `BatchRANSDecoder` / `BatchRANSEncoder`.
- `run_pca_ec_deadzone.py` — calibration moments, QPCA/JointQK/TurboQuant bases, QPCA-EC delta + frozen coder model, scoring helpers.
- `test_codec_on_data.py` — the harness.

## Drop in before running (not included — your files / repo / data)
- repo: `_bootstrap`, `kvq.compression.{lloyd_max,per_coord}`, `pipelines.calibration.analyze_bases`.
- the dataset at the path `run_pca_ec_deadzone.data_root()` resolves.

## Run
```
python test_codec_on_data.py \
    --calib-idx 0 1 2 --eval-idx 4 --bits 2 --ptok 16 --dz 0.375 --lanes 1
```
Default path = GPU encode (`BatchRANSEncoder`) + GPU decode (`BatchRANSDecoder`), needs CUDA.