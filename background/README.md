# Background reading

Reference papers and in-house derivations that motivate the project's
approach to KV-cache key compression. PDFs are gitignored (the `*.pdf`
rule at the repo root); this README is the durable index.

## Reference papers

### `TurboQuant.pdf`
TurboQuant — random-rotation + Lloyd-Max scalar quantization for KV
caches. The TurboQuant V3 baseline we benchmark against (also vendored
as code under `vendor/turboquant-pytorch/`). Defines the random-Hadamard
+ uniform-bits comparison baseline used throughout the bench sweep.

### `Expected Attention.pdf`
Expected Attention — proposes approximating the future query distribution
to guide KV pruning. One of the key prior works our rate-distortion view
builds on; cited in `notes/core/kv_cache_rate_distortion_proposal.md` as
evidence that future-query approximation is tractable enough to guide
compression.

### `attention_matching.pdf`
Attention Matching — variational/expectation-based distortion measure
internal to attention. Provides the conceptual basis for using
attention-level distortion rather than reconstruction-MSE as the
compression objective.

### `QPTQ.pdf`
QPTQ — post-training quantization for large transformers. Standard
reference for weight + KV quantization tradeoffs; orients our work
within the broader quantization literature.

## In-house derivations

### `derivation_qpca_for_queries.pdf`
Derivation note for Q-PCA (a basis variant we explored before settling
on JointQK). Records the closed-form Q-weighted reconstruction-MSE
analysis that motivated the trace-formula metric used in
`experiments/toolkit/per_coord_quantization.py`. Useful when re-deriving
the JointQK objective or comparing against the V3/CCA baselines.
