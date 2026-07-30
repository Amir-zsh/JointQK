# Paper Figure 4: provenance audit, missing-cell fill, and the OSCAR baseline retune

**Date** 2026-07-29 · **Hardware** 1× H100 80GB HBM3, **TP=1, single GPU, every
cell** · **Engine** vendored `OSCAR-vq/sglang-research` · **Figure scripts**
`pipelines/eval/plot_paper_fig4.py` (v7), `pipelines/eval/plot_paper_fig4_retuned.py` (v8)

Audit of the Throughput section and Figure 4 of *Spend Bits Where Queries Look*
(6-panel revision: GPT-OSS-20B, Qwen3-8B, Qwen3-4B-Thinking; row 1 batch 1 vs
input length, row 2 batch scaling at 90K). Three questions: does the figure trace
to real measurements, are the quoted numbers right, and is the OSCAR baseline
tuned as hard as ours.

Not to be confused with `pipelines/eval/plot_fig4.py`, which reproduces the
*OSCAR* paper's Figure 4 from `artifacts/oscar_e2e/fig4_*` (100K, bs 1/8/32,
shared prefix). That is a different figure and unrelated to this audit.

## 1. Provenance

| panel | arm | study |
|---|---|---|
| a, d | BF16 | `throughput_gptoss20b_fp32` (+ `bf16_gptoss20b_90k_bs8`, §2) |
| a, d | OSCAR, NOVA-KV | `vq2_cuda_thr_sweep_onserver` |
| b, e | BF16 | `throughput_qwen3_8b` |
| b, e | OSCAR, NOVA-KV | `vq2_cuda_thr_sweep_qwen3_8b` (+ `vq2_cuda_thr512_qwen3_8b_90k_rerun`) |
| c, f | BF16 | `throughput_qwen3_4b_thinking` |
| c, f | OSCAR, NOVA-KV | `vq2_cuda_thr_sweep_qwen3_4b_thinking` |

NOVA-KV is the `vq2_cuda` arm at its best THR per cell (512 at batch 1, 256 at
batch 4, 128 at batch 8/16 — each a separately compiled kernel).

Two composition facts worth stating in the paper's methods:

- **BF16 comes from a different study than the 2-bit arms**, because the THR
  sweeps have no BF16 arm. The `oscar_int2` arm is present in both and agrees
  across them to <1.5% (e.g. Qwen3-8B 90K/bs16: 312.9 vs 313.0), which is what
  licenses the splice.
- **Qwen3-8B's 90K NOVA cells come from a rerun.** THR=512 is missing from the
  main sweep at 90K; without `vq2_cuda_thr512_qwen3_8b_90k_rerun` the peak
  speedup reads 2.68× instead of 3.05×.

`TP=1` on a single GPU for every panel, from the per-arm `provenance.json`.
GPT-OSS here is the native **MXFP4** checkpoint (`openai/gpt-oss-20b`, ~12 GB).
The MATH-500 accuracy runs use `unsloth/gpt-oss-20b-BF16` at TP=2 — different
weights *and* different parallelism, so "gpt-oss" does not denote one setup
across the paper.

## 2. The two BF16 gaps were not the same kind of gap

Row 2 originally had no BF16 bar in three places. They had different causes, and
rendering both as silent absences invited the reader to assume a capacity limit
in both:

- **Qwen 90K / bs16** — BF16 genuinely does not fit; `b_max=4` at 90K. This is a
  *result*, and arguably the strongest capacity claim in the figure. Now drawn as
  an explicit `n/a`.
- **GPT-OSS 90K / bs8** — never measured. BF16 admits `b_max=10` there, so batch
  8 fits fine; it was simply absent from the fp32 study's ladder
  `[1, 4, 10, 16, 32, …]`. **Now measured**: `artifacts/throughput/bf16_gptoss20b_90k_bs8`,
  serve settings copied verbatim from `throughput_gptoss20b_fp32`.

```
GPT-OSS-20B 90K, BF16 decode_tok_s:  bs1 103.1 → bs4 370.0 → bs8 644.0 → bs10 704.7
```

Monotone, all six gates pass (`full_length, no_retract, warm_took,
batch_coexisted, no_nan, server_agrees`), `max_running_req=8`.

**This changes panel (d).** At GPT-OSS 90K batch 8, NOVA-KV (649.9) is **1.01×
BF16** and 4.9% *below* OSCAR (683.5). The missing bar was concealing that
compression buys essentially nothing at that point on this model — consistent
with the paper's own stated mechanism (sliding-window shrinks the compressed
share of the cache).

## 3. Metric: `decode_tok_s` vs `decode_tok_s_total`

The harness records two aggregate throughputs and they are **not**
interchangeable at large batch:

| metric | definition | note |
|---|---|---|
| `decode_tok_s_total` | `bs × out_tokens / decode_window` | matches the paper's wording "aggregate tokens per second across the batch" |
| `decode_tok_s` = `decode_tok_s_stagger_free` | sum of each request's own decode rate | unbiased under staggered decode entry; the harness's headline, used for all arm ranking and speedups |

`estimator_gap_frac` = 0.000 at bs1, ~0.02 at bs4, **0.058 at Qwen bs16**, ~0.00
for gpt-oss at every batch. So the two agree everywhere except the Qwen bs16
bars.

The earlier revision of the figure used `decode_tok_s_total`; `fig4_v7`/`v8` use
`decode_tok_s`. **This was an undocumented metric switch, not a correction** —
an earlier version of this audit wrongly reported the bs16 values as ~6-7%
transcription drift. They were the other metric, exactly:

```
Qwen3-8B 90K bs16   OSCAR 332.14 (_total) vs 312.93 (decode_tok_s)
                    NOVA  295.85 (_total) vs 276.09 (decode_tok_s)
Qwen3-4B 90K bs16   OSCAR 349.57 (_total) vs 329.39 (decode_tok_s)
```

Either is defensible; `gptoss_throughput_report.md` reports `decode_tok_s_total`.
**Whichever is chosen must be stated and used consistently** — conclusions do not
move (NOVA/OSCAR at Qwen3-8B bs16 is 1.123 on `_total`, 1.133 on `decode_tok_s`).

## 4. Throughput-section claims, verified

Against `fig4_v7` (untuned OSCAR), computed on **both** metrics — the metric
choice changes no verdict.

| claim | measured | |
|---|---|---|
| 30K/60K/90K, warm-up request, decode window only | — | ok |
| "Cross-request prefix sharing is disabled" | harness uses a distinct random prompt per request; `RADIX_CACHE=1` serves only each request's own warm pass | ok |
| 1.7–3.4× BF16, Qwen3-4B | 1.68–3.44 | ok |
| 1.1–1.5× BF16, GPT-OSS-20B | 1.05–1.49 (1.052 rounds to 1.1) | ok |
| "larger factors at the longer inputs" | 1.15 → 1.34 → 1.49 (gpt-oss bs1) | ok |
| BF16 admits at most b4 at 90K on either Qwen, no batch-16 config | bf16 90K cells are bs1, bs4 only, both models | ok |
| **1.6–3.1× BF16, Qwen3-8B** | **1.54–3.05** | **low end wrong → 1.5–3.1×** |
| **"Against OSCAR … within a few percent of each other throughout"** | **−12.6% … +17.6%** | **not supported** |

The 1.54 sits at 30K/batch 4 — a cell Figure 4 does not plot, which is plausibly
how it slipped. The OSCAR sentence fails in both directions at once: NOVA-KV is
11–13% *below* OSCAR at batch 16 on both Qwen models and 14–18% *above* at batch
1/90K. The figure's own bars are correct; only the sentence is wrong.

## 5. OSCAR baseline retune (closes task #18)

Every prior vq2-vs-int2 comparison was **tuned NOVA-KV against untuned OSCAR**:
NOVA-KV selects its best THR per cell, OSCAR ran one heuristic tile config.

OSCAR-INT2 stage-1 exposes `SGL_INT2_BLOCK_N`, `SGL_INT2_BLOCK_H`,
`SGL_INT2_NUM_WARPS`, `SGL_INT2_NUM_STAGES` over a ladder in
`_decode_grouped_att_m_fwd_quant_int2`. That ladder's own docstring records it as
tuned at **bs=1, seq=80k**, later hand-extended for narrow heads — so the
batch>1 branches were the untuned part.

**Method.** Eight configurations × 9 cells × 3 models = 216 cells
(`pipelines/throughput/configs/int2_retune_*.json` → `artifacts/throughput/int2_retune_*`).
Serve settings and the `oscar_int2` arm copied verbatim from the sweeps that feed
Figure 4, so winners are directly substitutable. `oscar_default` is included as a
control: any cell it wins was already tuned. It reproduces the original sweeps to
within 1.5%, which validates the comparison.

**Where OSCAR was actually losing performance:**

| model | batch 1 | batch 4 | largest batch |
|---|---|---|---|
| GPT-OSS-20B | +1.3…+3.5% | ~0% | **+5.2…+8.3%** (bs8) |
| Qwen3-8B | 0.0% | **+6.1%** | 0.0% (bs16) |
| Qwen3-4B-Thinking | 0.0% | **+7.0%** | 0.0% (bs16) |

Winners: `BLOCK_N=128, BLOCK_H=8, num_warps=2` (gpt-oss bs8 — the heuristic picks
4 warps for batch<16 on narrow heads and over-subscribes registers);
`…num_warps=4, num_stages=2` (Qwen3-8B bs4); `BLOCK_H=16` (Qwen3-4B bs4).

**Effect — NOVA/OSCAR, v7 → v8:**

| cell | v7 | v8 |
|---|---|---|
| GPT-OSS 90K bs1 | 1.140 | 1.092 |
| GPT-OSS 90K bs4 | 1.016 | 1.004 |
| **GPT-OSS 90K bs8** | 0.951 | **0.898** |
| Qwen3-8B 90K bs4 | 1.073 | 1.023 |
| Qwen3-4B 90K bs4 | 1.096 | 1.018 |
| Qwen bs16 | 0.882 / 0.874 | unchanged |

NOVA-KV leads at batch 1 (1.09–1.18×), is level at batch 4 (1.00–1.02×), and
trails at the largest batch on all three models (−10.2% gpt-oss, −11.8% / −12.4%
Qwen). **NOVA/BF16 is unchanged**, since only the OSCAR series moved — so the
BF16 speedup claims in §4 stand either way.

The high-batch deficit is a kernel-selection problem, not a method problem: the
sweep shows THR=128 is correct there, and task #22 (per-batch THR dispatch) is
the lever.

## 6. Caption

For `fig4_v8` (both arms tuned):

> **Figure 4: Decode throughput, prefill excluded.** Single H100, TP = 1. Top
> row: batch 1 against input length. Bottom row: batch scaling at 90K input;
> batch axes differ by model (GPT-OSS-20B 1/4/8, Qwen 1/4/16). BF16 has no
> batch-16 bar on either Qwen model — at 90K its KV cache admits at most batch 4,
> whereas both 2-bit arms serve batch 16. Both 2-bit arms use the best
> per-setting kernel configuration.

For `fig4_v7` (OSCAR untuned) the last sentence must instead read: *"NOVA-KV uses
the best per-setting kernel configuration; OSCAR-INT2 uses a single tile
configuration."* Do not ship v7 with the v8 sentence.

## 7. Open risk: panels (a) and (d)

GPT-OSS throughput is measured with `RADIX_CACHE=1` (`run_throughput.py:132`),
and the warm pass re-sends each request's own prompt, so it is tree-served. That
is the path proven to serve corrupted KV on gpt-oss + mixed KV (see the flush
SWA-transfer entry in `fixes_to_apply.md`). The defect is a correctness one and
need not move tok/s, but that is **unverified** — task #28. Qwen is structurally
immune (no sliding-window layers → plain `RadixCache`).

## 8. Reproduction

```bash
# missing BF16 cell
.venv/bin/python pipelines/throughput/run_throughput.py \
  --config pipelines/throughput/configs/bf16_gptoss20b_90k_bs8.json --gpus 4,5

# OSCAR baseline retune (8 configs x 9 cells, per model)
.venv/bin/python pipelines/throughput/run_throughput.py \
  --config pipelines/throughput/configs/int2_retune_gptoss20b.json --gpus 0,1,2,3,4,5,6,7

# figures
.venv/bin/python pipelines/eval/plot_paper_fig4.py          # fig4_v7 (OSCAR untuned)
.venv/bin/python pipelines/eval/plot_paper_fig4_retuned.py  # fig4_v8 (both tuned)
```

`plot_paper_fig4_retuned.py` reads every value from the study JSONs at run time
and refuses to render if any model's retune is incomplete, so a half-tuned OSCAR
series cannot reach the figure by accident.
