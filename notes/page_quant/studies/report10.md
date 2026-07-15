# pgq10 — Samuel's VQ optimized vs production OSCAR + VQ×DCT (plan10)

Branch `pgq10`. Two user-requested directions: **D1** — port Samuel's group-VQ
(latest, uncommitted tree) and optimize it to beat OSCAR in quality AND speed;
**D2** — compose his VQ with our page-DCT transform. User-selected framing:
quality judged on LongBench + 32–64K long-context; OSCAR baseline = the
authors' actual SGLang stack end-to-end; codebooks retrained from our pool;
math tasks scored with HF math-verify.

## Pre-registered bars → outcomes

| bar | statement | outcome |
|---|---|---|
| B-Q1 | ported VQ reproduces Samuel's LongBench margin over OSCAR, row-paired | **MET** — +2.37 [+0.46, +4.31] SIG vs authors'-rotation OSCAR emulation |
| B-Q2 | optimized VQ ≥ production OSCAR at 32K AND 64K NIAH, ≤2.3 b/c | PENDING (Q3 wave + e2e controls in flight) |
| B-M | math500/aime25 deltas (descriptive) | production OSCAR INT2: math500 72.4% (math-verify) / 58.6% (regex), aime25 16.7%; arms pending |
| B-S1 | VQ-K fused step ≤ vendored OSCAR @128k bs=1 | **NOT MET** — best VQ-K 0.971 ms vs OSCAR 0.261 (3.7×) |
| B-S2 | ≥0.8× OSCAR @32k | **NOT MET** — 0.434 vs 0.101 ms |
| B-D2 | VQ×DCT proxy ≤ scalar-DCT proxy at ≥1 rate | **NOT MET (NO-GO)** — 2.2–2.6× worse at every rate; F1 wave cancelled per gate |

## D1 quality (Qwen3-8B, LongBench 5-task, fraction 1.0)

| arm | b/c (base+overhead) | lcc | musique | 2wiki | qasper | hotpotqa | mean |
|---|---|---|---|---|---|---|---|
| pgq_vqg_flat (no band) | 2.00 | 54.14 | 30.19 | 46.06 | 43.40 | 62.83 | 47.32 |
| pgq_vqgb_flat (s4r32 band) | ~2.12 | 62.46 | 29.34 | 45.19 | 41.97 | 63.47 | 48.49 |
| pgq_vqgbo05 (band + 5% outlier→fp8) | ~2.4 | 62.65 | 29.73 | 45.83 | 42.00 | 63.51 | 48.74 |
| pgq_dctlmrw_rdo@2.0 (incumbent) | 2.13 honest | 63.43 | 31.77 | 46.00 | 40.00 | 62.82 | 48.80 |
| pgq_oscar_uni (authors' rotation, 64/256, clips .96/.92) | ~2.25+win | 60.39 | 27.94 | 42.77 | 39.46 | 58.37 | 45.79 |

Row-paired bootstraps (10k, per-task resample, equal-weight pool over
lcc/musique/2wikimqa):
- vqgbo05 − oscar_uni(rotzoo): **+2.37 [+0.46, +4.31] SIG** → **B-Q1 MET**.
  Samuel's +2.2 margin reproduces against a *stronger* baseline (authors'
  calibrated rotation, not a Hadamard reimplementation).
- vqgbo05 − dctlmrw@2.0: −1.00 [−2.70, +0.65] tie (consistent with the
  cross-repo row-paired result from the comparison study).
- vqgbo05 − vqgb: +0.40 [−0.23, +1.06] tie (outlier wrap helps, not SIG at
  LongBench lengths; its target regime is long-context needles).
- No-band lcc collapse (54.1 vs 62.5) replicates Samuel's band finding.

Port fidelity: bit-exact roundtrip parity vs his snapshot codec on his fp8
Qwen codebook; retrained-from-our-pool codebooks pass the held-out gate at
ratio 1.042 (Qwen, tol 1.05) and 1.006 (Llama). Provenance (his tree was
uncommitted): `third_party/samuel_vq/PROVENANCE.md`.

## Production OSCAR end-to-end (O4)

The authors' vendored sglang-research stack serving Qwen3-8B with INT2 KV on
our A100-40GB — their RotationZoo rotation (`seq20000_prompt83_group128`),
their driver env (64/256 windows, clips 0.96/0.92), triton/triton backends
(fa3 prefill is Hopper-only; triton/triton is a supported path — recorded
caveat vs the paper's H100 numbers). Port hardships worth recording:
`kernels==0.14.0` pin (transformers 5.3.0 LayerRepository break), pool
sizing for 40 GB (`--max-total-tokens`, `--max-running-requests` — the HP
arenas land on top of mem-fraction), conda gcc-12 + `NVCC_PREPEND_FLAGS=
-ccbin` + `lib64` symlink + `LD_LIBRARY_PATH` for the C++20 JIT kernels,
`--disable-radix-cache` + bigger HP-prefix pool (exhaustion after ~3200
unique rows — eval workloads have no prefix reuse).

Scoring: identical rows + identical final prompt strings as our harness
(pipelines/eval/export_prompt_rows.py), native /generate, temperature 0,
scored by our registry scorers.

**INT2 results:** NIAH means **97.6 / 95.8 / 74.2 / 23.4** at 8/16/32/64K
(multikey subtasks collapse first: 0% at 64K). LongBench 5-task mean
**47.39** — production OSCAR scores ~1.6 F1 above our in-harness emulation
(45.79); the BF16-served control (in flight) isolates the serving-stack
contribution. Math: math500 72.4%, aime25 16.7%. Caveats: 64K exceeds
Qwen3-8B's native 40960 window (RoPE extrapolation affects every method;
BF16 control quantifies it); serving-stack prompt handling differs from our
harness's max-context truncation.

## Speed (S5) — honest negative

New phase-A kernels (parity 20/20, CUDA-graph bench, A100, bs=1):

| arm @128k | ms | vs OSCAR 0.261 |
|---|---|---|
| pgq adaptive scalar + fp16 V (v2g) | 0.836 | 0.31× |
| pgq adaptive scalar + INT2 V (vI2) | 1.115 | 0.23× |
| VQ-K naive gather + INT2 V (vqk) | 1.437 | 0.18× |
| **VQ-K int64-codeword + INT2 V (vqk64)** | **0.971** | 0.27× |

The registered hypothesis (fixed-rate VQ row ≈ OSCAR parity by removing the
per-row adaptive-format constants) is **falsified**: the constants are real
(vqk64 beats the scalar path at matched INT2-V by 13%) but the codebook
gather dominates — one gather instruction per 4-coord codeword is still
~3.7× behind OSCAR's sequential 32 B/token unpack. This reproduces Samuel's
L1-residency/gather diagnosis in a second, independent harness. Venue for
closure remains batched/H100 (both repos' next-steps agree).

## D2 / D6 — VQ×DCT: closed NO-GO at the proxy gate

`PageDCTVQCompressor` (DCT coefficient rows → group-VQ rungs 0/1/2/3 b/c,
per-rung (G,K) = –/[4,16]/[4,256]/[2,64], dct_std-normalized, same paged
λ-RDO; codebooks trained per (l,h) on transform-eligible pages):

| rate | dvq logit_err | scalar dct logit_err | ratio |
|---|---|---|---|
| 1.5 | 0.00716 | 0.00293 | 2.4× |
| 1.75 | 0.00601 | 0.00231 | 2.6× |
| 2.0 | 0.00516 | 0.00188 | 2.7× |

Per the pre-registered gate, no F1 wave. Reading: after QPCA (coord axis)
and DCT (token axis) the coefficients are doubly decorrelated — VQ's joint
structure has nothing left to exploit, and one shared codebook per (l,h)
cannot match per-coordinate scalar grids with per-row σ. This is precisely
the falsifiable condition Samuel wrote into his own trainer notes
(G>1 must beat G=1 on decorrelated coords, else VQ is dominated).

## Math-verify (T0)

`kvq/benchmarks/math_verify_scorer.py` registered for aime25/math500
(vendored regex kept as `accuracy_boxed_exact`). On production OSCAR
math500 the symbolic scorer credits 72.4% vs 58.6% regex — 14 points of
formatting-equivalent answers (1/2 vs 0.5 etc.), confirming the directive's
value. Registry note: the bench worker resolves the VENDORED
evaluate_registry, not the kvq shim — the vendored module (local, untracked
vendor tree) now bridges the shim so both paths share one source of truth.

## PENDING (waves in flight)

- **B-Q2 table:** our 5 arms (fp16 / oscar-rotzoo / vqgb / vqgbo05 /
  dctlmrw) × NIAH 32K/64K at fraction 0.25 vs production-OSCAR e2e + BF16
  controls.
- BF16-served control (all 11 sets) → serving-stack delta + RoPE-extrapolation
  attribution at 64K.
- L7 Llama transfer: codebooks gated (1.006); LongBench cells queued behind
  the Q3 wave; vs existing pgq_oscar_uni Llama cells.
- R8: comparison-page update + republish.
