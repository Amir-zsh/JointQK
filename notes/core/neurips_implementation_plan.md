# NeurIPS 2026 Implementation Plan

Complement to `notes/core/neurips_submission_plan.md`. The submission plan is *what* we ship; this plan is *how* we build it. Code paths are absolute relative to the repo root. Every section ends in a verifiable checkpoint.

## Topology and key insight

The fastest path to all downstream metrics is **kvpress's existing harness**:

- `kvpress/evaluation/evaluate.py` already drives both LongBench and RULER evaluations end-to-end (`Fire`-style CLI, default model `meta-llama/Meta-Llama-3.1-8B-Instruct`, dataset registry, scorer registry, press registry).
- We do **not** write `longbench_runner.py` / `ruler_runner.py`. Instead we register three new presses in `kvpress/evaluation/evaluate_registry.py` and invoke the existing harness.
- kvpress's `BasePress.compress(module, hidden_states, keys, values, attentions, kwargs) -> (keys', values')` returns same-shape recon tensors. Our `PerCoordCompressor.roundtrip(keys)` is a perfect drop-in (line `kvpress/kvpress/presses/base_press.py:144`).
- Compression hook fires per-layer after attention forward and is **auto-skipped during decode** (line 139). That is the prefill-only semantics Stage 1E used and the kvpress default — but **not** what KIVI/KVQuant/TurboQuant do in their published evaluations, which compress every key including decode-step keys. We need to evaluate both scopes (see "Decode-scope ablation" below).
- kvpress also ships `PrefillDecodingPress` (`kvpress/kvpress/presses/prefill_decoding_press.py`) — a wrapper that delegates to a `prefilling_press` during prefill and a `decoding_press` during generation. We lean on this pattern for the prefill+decode mode.

The Stage-1E driver `experiments/run_bases.py` is bundle-driven (operates on pre-captured Q/K stats; does not load a model). For W1 we re-run `experiments/collect_query_stats.py --model meta-llama/Llama-3.1-8B` to produce a Llama bundle, then everything downstream is unchanged.

## Decode-scope ablation (decision blocker for the headline number)

The kvpress default press only compresses the prefill K cache; new keys generated during decode are stored at full precision. Published KV-quantization baselines (KIVI, KVQuant) instead compress *every* key including decode-step ones. We can't pick one a priori — the offline-calibrated R_sym basis was computed from prefill keys, and decode-step keys come from a slightly different distribution (model-generated continuation, not user prompt). So we evaluate **two scopes**:

- **Mode A (prefill-only):** compress the prefill K cache; decode-step keys stored at fp16. Aligns with kvpress's hook default and Stage 1E's evaluation regime.
- **Mode B (prefill + decode):** compress every K, including each newly-generated decode-step key. Aligns with how published quantizers operate; bigger memory savings; fair comparison to KIVI/TurboQuant.

**Decision rule (W2c gate):** run JointQK in Mode A vs Mode B on **2 representative LongBench tasks** at b ∈ {2, 3, 4}. If Mode B's task score is within **2 pp** of Mode A on both tasks at every b, the headline result uses Mode B. Otherwise the headline uses Mode A and Mode B becomes a "memory-vs-quality tradeoff" ablation table in the paper.

**Recommended task pair:** `qasper_e` (short answers, low decode load — stresses prefill quality) + `narrativeqa` (long answers, high decode load — stresses decode-key compression). They bracket the decode/prefill ratio space.

## Critical-path graph

```
[A: Llama Q/K capture] ─→ [B: Llama Stage-1E E3/E5] ─→ [C: cross-model chart]   W1
                                                                                  
[D: JointQKPress (Mode A + Mode B)] ┐                                            
[E: TurboQuantPress]                ├─→ [G: parity test on Qwen 24-bundle]      W2
[F: KIVIPress]                      ┘            │
                                                 ├─→ [H: W2c decode-scope ablation]
                                                 │     (2 tasks, picks Mode A or B)
                                                 ├─→ [I: LongBench full sweep]
                                                 └─→ [J: RULER NIAH]
                                                                                  
[K: paper scaffold]─→ [L: method §]→[M: experiments §]→[N: polish]                W4
```

Independent: A, D/E, F, K can all start day 1. The blocking edges are A→B, D/E/F→G, G→I/J, all-experiments→M.

---

## W1 — Llama-3.1-8B reproduction

### W1.1 Capture Q/K bundle for Llama (Day 1, ~4–8 h GPU)

`experiments/collect_query_stats.py` already has `--model` (default `Qwen/Qwen3-8B`, line 26). Re-run with Llama:

```bash
conda activate kv-rd
python -u -m experiments.collect_query_stats \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --output artifacts/query_stats_longbench_under4k_llama31_8b \
    --gpus 0,1 \
    2>&1 | tee logs/capture_llama31.log
```

**Verification:**
- Output dir contains `manifest.json` + 24 example `.pt` files (`qasper_e_*`, `hotpotqa_e_*`, `passage_retrieval_en_e_*`).
- Each `.pt` has `q_post`, `k_post` of shape `(n_layers=32, n_kv_heads=8, seq_len, head_dim=128)` (Llama-3.1-8B GQA: 32 Q-heads / 8 KV-heads). Note: layer count differs from Qwen (36).
- Total bundle size ≈ matches Qwen's bundle within ~50% (capture length is per-example, model-dim independent).

**Risks:** RoPE differences, model-specific attention sinks may shift second-moment shapes. Mitigation: `jointqk.pt` will have `(32, 8, 128, 128)` rather than `(36, 8, 128, 128)` — Stage 1E driver handles this generically (it reads `n_layers` from the calibration manifest, line 424).

### W1.2 Run Stage-1E E3/E5 on Llama bundle (Days 2–3)

```bash
# Calibration + E3 + E5 in one launcher (extends launch_cca_study.sh)
bash pipelines/scripts/launch_llama_e3_e5.sh \
    --bundle artifacts/query_stats_longbench_under4k_llama31_8b \
    --output-subdir llama31_8b \
    --gpus 0,1,2,3 \
    2>&1 | tee logs/llama31_e3_e5.log
```

**New file:** `pipelines/scripts/launch_llama_e3_e5.sh` (~30 lines). Models after `launch_cca_study.sh`. Hardcoded:
- bundle path → input
- output_subdir → `llama31_8b`
- methods → all 9 (TurboQuant + 8 calibrated variants)
- b_avg ∈ {2, 3, 4}, rank=64

**Verification (W1 gate):**
```bash
python -m experiments.gates.gate_e3 \
    --output-dir artifacts/bases/llama31_8b/e3
python -m experiments.gates.gate_e5 \
    --output-dir artifacts/bases/llama31_8b/e5
```
Gates auto-discover methods. Pass criterion: `r_sym_waterfill` is top-ranked at every (b, layer-0-excluded) cell with margin ≥ 5 pp top-1 over runner-up at b=3. **If fails → escalate to user before W1.3.**

### W1.3 Cross-model chart (Day 5)

**New file:** `pipelines/scripts/make_cross_model_chart.py` (~80 lines). Reads:
- `artifacts/bases/e3/{e3_b{2,3,4}_r64_summary.json}` (Qwen, post-newbases)
- `artifacts/bases/llama31_8b/e3/{e3_b{2,3,4}_r64_summary.json}` (new Llama)

Produces `report_charts/cross_model_b_sensitivity.png` — two-panel (Qwen | Llama) of top-1 vs b_avg, highlighting JointQK WaterFill.

---

## W2 — kvpress press wrappers (Days 3–6)

### W2.1 — `JointQKPress` (Days 3–4)

**New file:** `kvq/jointqk_press.py`.

**Class skeleton:**
```python
from dataclasses import dataclass, field
from pathlib import Path
import torch
from kvpress.presses.base_press import BasePress
from kvq.compression.per_coord import (
    PerCoordCompressor, build_method_compressor,
)

@dataclass
class JointQKPress(BasePress):
    cca_stats_path: str           # path to artifacts/.../jointqk.pt
    b_avg: float = 3.0
    rank: int = 64
    method: str = "r_sym_waterfill"   # or v_waterfill, cca_orth_waterfill
    layer0_full_precision: bool = True
    compress_decode: bool = False     # W2c: False=Mode A (prefill-only), True=Mode B (prefill+decode)

    # Set by post_init_from_model
    _compressors: dict = field(default_factory=dict, init=False)

    def post_init_from_model(self, model):
        stats = torch.load(self.cca_stats_path, map_location="cpu")
        n_layers, n_kv_heads, d, _ = stats["sigma_q"].shape
        self._compressors = {}
        for L in range(n_layers):
            if L == 0 and self.layer0_full_precision:
                continue   # signal: roundtrip is identity
            for h in range(n_kv_heads):
                self._compressors[(L, h)] = build_method_compressor(
                    method=self.method,
                    sigma_q_for_head=stats["sigma_q"][L, h],
                    sigma_k_for_head=stats["sigma_k"][L, h],
                    cca={"P_K": stats["P_K"][L, h], "P_K_inv": stats["P_K_inv"][L, h], "rho": stats["rho"][L, h]},
                    mq_eigvals=stats["mq_eigvals"][L, h],
                    mq_eigvecs=stats["mq_eigvecs"][L, h],
                    b_avg=self.b_avg,
                    r=self.rank,
                    V_h=stats.get("V_h", [None]*n_layers)[L][h] if stats.get("V_h") is not None else None,
                    R_sym=stats.get("R_sym", [None]*n_layers)[L][h] if stats.get("R_sym") is not None else None,
                )

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        L = module.layer_idx
        if L == 0 and self.layer0_full_precision:
            return keys, values
        # keys: (batch, n_kv_heads, seq_len, head_dim)
        recon = torch.empty_like(keys)
        for h in range(keys.shape[1]):
            comp = self._compressors[(L, h)].to(keys.device)
            recon[:, h] = comp.roundtrip(keys[:, h])
        return recon, values   # V untouched (K-only paper)

    def forward_hook(self, module, args, kwargs, output):
        """Override BasePress.forward_hook to optionally fire on decode steps."""
        if self.compress_decode:
            # Mode B: skip the prefill-only early-return. compress() will be called every step.
            # During decode, `keys` arrives shaped (batch, n_kv_heads, full_seq_len, head_dim).
            # We compress only the LAST token (the new key); previously-quantized keys are left alone.
            from kvpress.utils import extract_keys_and_values
            cache = kwargs["past_key_values"]
            cache_layer = cache.layers[module.layer_idx]
            q_len = kwargs["hidden_states"].shape[1]
            is_prefill = kwargs["cache_position"][-1] <= q_len
            keys, values = extract_keys_and_values(cache, module.layer_idx)
            if is_prefill:
                keys, values = self.compress(module, kwargs["hidden_states"], keys, values, output[1], kwargs)
            else:
                # decode: compress only the newly-appended key (last position)
                new_k = keys[:, :, -1:, :]
                new_k_recon, _ = self.compress(module, kwargs["hidden_states"], new_k, values[:, :, -1:, :], None, kwargs)
                keys = torch.cat([keys[:, :, :-1, :], new_k_recon], dim=2)
            cache_layer.keys = keys
            cache_layer.values = values
            return output
        # Mode A: defer to BasePress.forward_hook (prefill-only)
        return super().forward_hook(module, args, kwargs, output)
```

For Mode B we could alternatively wrap two presses with `kvpress.PrefillDecodingPress`, but the in-class flag keeps configuration simpler at the evaluate.py call site (single `--press_init_command`).

**Notes:**
1. Per-batch loop over heads is OK for evaluation (we're not optimizing throughput here).
2. `jointqk.pt` may need to have R_sym / V_h pre-computed and saved. Currently, `run_bases.py` derives them on-the-fly via `_derive_vh_rsym`. To keep press lean, **add a one-time pre-derivation step** that saves R_sym + V_h into jointqk.pt for both models.

**Pre-derive script** (~30 lines): `pipelines/scripts/precompute_newbases.py`. Loads jointqk.pt, runs `_derive_vh_rsym` per (layer, head), writes jointqk.pt back with two new keys. Idempotent.

### W2.2 — `TurboQuantPress` (Day 4, ~1h)

**New file:** `kvq/turboquant_press.py`. ~80 lines.

Same skeleton as JointQKPress but no calibration: per layer/head, build a random Hadamard matrix once (seeded), use uniform Lloyd-Max bits. Reuses `PerCoordCompressor` directly with `forward_map = H_hadamard`, `inverse_map = H_hadamard.T`.

### W2.3 — `KIVIPress` (Days 5–6) — see W3

### W2.4 — Register presses in kvpress (Day 6, ~10 min)

**Edit:** `kvpress/evaluation/evaluate_registry.py`.
```python
from kvq.presses.jointqk_press import JointQKPress
from kvq.presses.turboquant_press import TurboQuantPress
from kvq.presses.kivi_press import KIVIPress

PRESS_REGISTRY.update({
    "jointqk": JointQKPress,
    "qeigen_waterfill": JointQKPress,    # same class, method="v_waterfill"
    "ccaorth_waterfill": JointQKPress,   # same class, method="cca_orth_waterfill"
    "turboquant": TurboQuantPress,
    "kivi": KIVIPress,
})
```

**Smoke test:**
```bash
cd kvpress/evaluation
python evaluate.py \
    --press_name jointqk \
    --dataset longbench \
    --data_dir qasper \
    --fraction 0.05 \
    --model Qwen/Qwen3-8B \
    --press_init_command 'JointQKPress(cca_stats_path="artifacts/bases/jointqk.pt", b_avg=3.0)' \
    --output_dir results/smoke_jointqk
```

(`evaluate.py` accepts `--press_init_command` as a Python expression evaluated to instantiate the press; check current implementation.)

### W2.5 — Validate presses reproduce Stage-1E numbers (Day 6, ~2h)

**New file:** `tests/test_press_roundtrip_parity.py`.

Loads:
- One example from the 24-example bundle (e.g. `qasper_e_001.pt`)
- Cross-checks `JointQKPress.compress` output against the stage1 driver's offline `PerCoordCompressor.roundtrip` output for layer L, head h, b=3.

Pass criterion: max-abs-diff < 1e-5 (numerical equivalence).

If parity fails → bug in press wrapper, not in the underlying compressor.

---

## W2c — Decode-scope ablation (Day 7, ~6 GPU-h)

**Goal:** decide whether the headline result uses Mode A (prefill-only quantization) or Mode B (prefill+decode quantization). See "Decode-scope ablation" section above for context.

**Design:** 2 LongBench tasks × 2 modes × 3 bit budgets × 1 model = 12 runs. Qwen3-8B only at this stage (Llama covered later if Mode B wins).

```bash
MODEL=Qwen/Qwen3-8B
CCA=artifacts/bases/jointqk.pt
OUT=artifacts/downstream/qwen3_8b/w2c_decode_scope

for task in qasper narrativeqa; do
  for mode_flag in "False" "True"; do      # False=Mode A, True=Mode B
    for b in 2 3 4; do
      tag=$(if [ "$mode_flag" = "True" ]; then echo "modeB"; else echo "modeA"; fi)
      python kvpress/evaluation/evaluate.py \
        --press_name jointqk \
        --press_init_command "JointQKPress(cca_stats_path='$CCA', b_avg=$b.0, compress_decode=$mode_flag)" \
        --model "$MODEL" --dataset longbench --data_dir $task \
        --output_dir "$OUT/${task}_${tag}_b${b}"
    done
  done
done
```

**Aggregator:** `pipelines/eval/aggregate_decode_scope.py` (~80 lines). Reads the 12 result.json files, prints the comparison table:

```
              | qasper      | narrativeqa
              | b2  b3  b4  | b2  b3  b4
Mode A (pref) | ...
Mode B (both) | ...
Δ (B - A)     | ...
```

**Decision rule:** if `max(|Δ|) < 2.0 pp` across all 6 (task, b) cells → Mode B wins, set `JOINTQK_DEFAULT_MODE = B` for all subsequent W2b runs and the headline result. Otherwise Mode A wins, headline uses Mode A, Mode B numbers go into a paper appendix table titled "Decode-scope ablation".

**New file:** `pipelines/eval/aggregate_decode_scope.py`. Outputs both the comparison table (LaTeX) and a one-line "WINNER: Mode A/B" decision string consumed by W2b launchers.

**W2c gate (C5.5 — see checkpoints):** decision string written to `$OUT/decision.txt`. W2b.1 reads this file to set `compress_decode` for all JointQK runs; this prevents the full sweep from racing the ablation.

**Risk: Mode B is much worse than Mode A on `narrativeqa`.** Plausible if decode-step keys really do come from a different distribution than what offline calibration captured. In that case the paper has a clean negative result + ablation, and we proceed with Mode A as the headline. Either outcome is publishable; the goal is not to *prove* Mode B works but to *honestly characterize* the scope tradeoff.

---

## W2b — LongBench + RULER evaluation (Days 6–10)

### W2b.1 LongBench full sweep on Qwen3-8B (Days 6–8)

Reads the W2c decision file before launching:

```bash
MODEL=Qwen/Qwen3-8B
CCA=artifacts/bases/jointqk.pt
OUT=artifacts/downstream/qwen3_8b
DECISION_FILE=$OUT/w2c_decode_scope/decision.txt
COMPRESS_DECODE=$(grep -oE 'True|False' "$DECISION_FILE")  # set by W2c

# Full-precision baseline (oracle)
python kvpress/evaluation/evaluate.py \
    --press_name knorm --compression_ratio 0.0 \
    --model "$MODEL" --dataset longbench \
    --output_dir "$OUT/full_precision"

# JointQK at b=2,3,4 in the chosen mode
for b in 2 3 4; do
    python kvpress/evaluation/evaluate.py \
        --press_name jointqk \
        --press_init_command "JointQKPress(cca_stats_path='$CCA', b_avg=$b.0, compress_decode=$COMPRESS_DECODE)" \
        --model "$MODEL" --dataset longbench \
        --output_dir "$OUT/jointqk_b$b"
done

# TurboQuant at b=2,3,4
for b in 2 3 4; do
    python kvpress/evaluation/evaluate.py \
        --press_name turboquant \
        --press_init_command "TurboQuantPress(b_avg=$b.0)" \
        --model "$MODEL" --dataset longbench \
        --output_dir "$OUT/turboquant_b$b"
done

# KIVI at int4 (b=4 native)
python kvpress/evaluation/evaluate.py \
    --press_name kivi \
    --press_init_command "KIVIPress(bits=4, group_size=128)" \
    --model "$MODEL" --dataset longbench \
    --output_dir "$OUT/kivi_int4"
```

**Compute estimate:** LongBench has 12 tasks, ~200 examples each, ~4–32k token contexts. On Qwen3-8B at fp16, prefill ~6 ms/token, generation ~25 ms/token, max_new_tokens ~200. Per-config: ~12 * 200 * (16k * 6ms + 200*25ms) ≈ 4 hours wall. 8 configs (4 methods × b=2/3/4 ish) × 2 models = a lot. Plan: parallelize across GPUs 0–3. Estimated end-to-end ~24 GPU-hours = 6 wall-hours on 4 GPUs.

**Aggregator:** `pipelines/eval/aggregate_longbench.py` (~100 lines). Reads all `$OUT/*/results.json`, builds a single `longbench_summary.json` with per-task × per-method × per-b cells, plus mean/std.

### W2b.2 LongBench on Llama-3.1-8B (Day 8)

Same as W2b.1 with `MODEL=meta-llama/Llama-3.1-8B-Instruct` and `CCA=artifacts/bases/llama31_8b/jointqk.pt`. Note kvpress `evaluate.py` defaults to this model already.

### W2b.3 RULER NIAH on both models (Days 8–10)

RULER's NIAH-single-needle at lengths {4096, 8192, 16384}. Same harness, swap `--dataset ruler --data_dir 4096` (or 8192/16384). Use `kvpress/evaluation/evaluate.sh` if it has a RULER pre-set; otherwise script directly.

If time-pressed: drop to NIAH at length 16384 only; cite "compute budget" in the paper.

### W2b.4 Bit-budget sensitivity tables/charts (Day 10)

**New file:** `pipelines/scripts/make_downstream_charts.py`. Inputs the aggregated `longbench_summary.json` and `ruler_summary.json`; produces:
- `report_charts/downstream_longbench_table.tex` — LaTeX-ready table (12 tasks × 4 methods × b∈{2,3,4})
- `report_charts/downstream_ruler_table.tex`
- `report_charts/downstream_bit_budget_curve.png` — task score vs b_avg per method

**Verification (W2 gate):** JointQK at b=3 retains ≥ 90% of full-precision LongBench task score on both models. TurboQuant and KIVI at b=3 retain ≥ 85%. JointQK strictly beats both peers on ≥ 80% of (task, model) cells.

---

## W3 — KIVI baseline (Days 5–8, parallel with W2)

### W3.1 Implement KIVI K-quantizer (Day 5, ~6h)

**New file:** `kvq/kivi_quantizer.py` (~150 lines).

Per the KIVI paper (Liu et al. 2024): per-channel int4 quantization for K with group size G=128 along the channel axis (with G=head_dim=128, this collapses to per-channel asymmetric int4).

Key signatures:
```python
def kivi_quantize_keys(keys: Tensor, bits: int, group_size: int) -> tuple[Tensor, Tensor, Tensor]:
    """Return (q_keys: int8, scales: fp16, zeros: fp16). Shapes: keys (..., head_dim);
    scales/zeros (..., head_dim/group_size)."""

def kivi_dequantize_keys(q_keys, scales, zeros, group_size) -> Tensor:
    """Inverse — recon shape matches input."""
```

Per-channel asymmetric scheme: for each channel `c` and group `g`, `scale = (max - min) / (2^bits - 1)`, `zero = -round(min / scale)`, `q = clamp(round(x/scale) + zero, 0, 2^bits-1)`. Standard int-quant routine.

### W3.2 Wrap as KIVIPress (Day 6, ~1h)

**New file:** `kvq/kivi_press.py` (~60 lines). BasePress subclass. `compress()` calls `kivi_quantize_keys(keys, ...)` then `kivi_dequantize_keys(...)` and returns the recon. No state per layer.

### W3.3 Validate KIVI matches paper (Days 7–8)

**New file:** `pipelines/scripts/validate_kivi.py`. Runs KIVIPress at int4 on a subset of LongBench (qasper, narrative_qa) and compares to the KIVI paper's reported numbers (~0.85 retention vs full precision). Tolerance: ±3 pp.

If gap > 3 pp, audit:
- Is V quantized as well? KIVI's published numbers are with V quantized too (per-token int4). Our paper is K-only — so the comparison should be K-only KIVI, not full KIVI. Add a note in the paper: "we compare K-only KIVI for an apples-to-apples comparison of the K-cache quantization mechanism."
- Group size: KIVI uses 128; we should match.

---

## W4 — Paper LaTeX (Days 1–17, parallel)

### W4.1 Scaffold (Day 1, ~1h)

```bash
mkdir -p paper/main/sections paper/main/figs
cp paper/Formatting_Instructions_For_NeurIPS_2026/neurips_2026.tex paper/main/main.tex
cp paper/Formatting_Instructions_For_NeurIPS_2026/neurips_2026.sty paper/main/
cp paper/Formatting_Instructions_For_NeurIPS_2026/checklist.tex paper/main/
touch paper/main/refs.bib
for s in abstract intro related_work method experiments discussion conclusion; do
    touch "paper/main/sections/$s.tex"
done
```

Edit `main.tex` to:
- Set `\title{Spend Bits Where Attention Reads: A Joint Q-K Energy Basis for KV-Cache Quantization}`
- `\input{sections/abstract}`, `\input{sections/intro}`, etc.
- Add `\usepackage{tikz, pgfplots, booktabs, amsmath, amssymb}`

### W4.2 Section drafts (Days 2–14)

Each section starts as an outline with bullets, then prose. Source material cross-references:

| Section | Pages | Source | Days |
|---|---|---|---|
| Abstract | 0.25 | `notes/stage1e_one_pager.md` | 2 |
| Intro | 1 | One-pager + Stage 1E proposal `notes/core/kv_cache_rate_distortion_proposal.md` | 2–3 |
| Related work | 1 | New (KIVI, KVQuant, H2O, FastGen, TurboQuant, Atom, Eigen-attn) | 6–8 |
| Method | 2.5 | `notes/stage1e_summary_to_share.md` §2 — port math + pipeline figure | 3–6 |
| Experiments | 3 | Stage 1E E3/E4/E5 + W1 (Llama) + W2b (downstream) results | 9–13 |
| Discussion + limitations | 0.5 | New | 14 |
| Conclusion | 0.25 | One-paragraph wrap | 14 |
| Appendix | unlimited | E1/E1_2/E4 deep-dives + theoretical sketch | 14–16 |

### W4.3 Figures

Two categories:
- **Direct copy** — already publication-quality. Source PNGs live in `artifacts/bases/report_charts/`. Copy to `paper/main/figs/`. Reference via `\includegraphics[width=...]{figs/<name>.png}`.
- **Replot** — for typography consistency, replot the headline bit-budget chart in pgfplots inside the LaTeX source. Single chart, ~50 lines of pgfplots code.

Required figures (main text):
1. Pipeline diagram (TikZ, new) — compression triplet (basis → Lloyd-Max → inverse).
2. Cross-model bit-budget sensitivity (W1.3 output, copy).
3. Cross-task generalization heatmap (Qwen E4, copy).
4. LongBench downstream table (W2b.4 LaTeX output).
5. RULER NIAH curves (W2b.4 LaTeX output).

Appendix figures: E1 cumulative energy, E1_2 subspace overlap, E4 LOO trace, E5 decode-vs-prefill.

### W4.4 Polish (Days 15–17)

- Run `pdflatex -shell-escape main && bibtex main && pdflatex main && pdflatex main` cleanly.
- `chktex main.tex 2>&1 | grep -v "^$"` for warnings.
- Page count: `pdfinfo paper/main/main.pdf | grep Pages` ≤ 9.
- Fill `checklist.tex` (NeurIPS reproducibility checklist).
- Add broader-impact statement (1 paragraph): KV compression democratizes long-context inference; no obvious dual-use concerns.

---

## Daily schedule

| Day | Morning | Afternoon | Evening |
|---|---|---|---|
| 1 (Mon May 4)  | W1.1: launch Llama Q/K capture (background) | W4.1: paper scaffold + abstract draft  | W2.1 spec + skeleton of `jointqk_press.py` |
| 2 (Tue May 5)  | W2.1: finish JointQKPress + pre-derive script | W2.2: TurboQuantPress | W4.2: intro outline |
| 3 (Wed May 6)  | W1.2: Llama E3 (run + monitor) | W3.1: implement KIVI quantizer | W4.2: method §3.1–3.2 |
| 4 (Thu May 7)  | W1.2: Llama E5 | W3.2: KIVIPress + W2.4 register all 3 presses | W4.2: method §3.3 (allocations + CCA mismatch) |
| 5 (Fri May 8)  | W2.5: parity tests + smoke evaluate.py | W2c kickoff: 2-task ablation × 2 modes × 3 budgets | W4.2: method §3.4 (basis families) |
| 6 (Sat May 9)  | W2c finish + aggregate + decision.txt → W2b.1 LongBench Qwen kickoff | W3.3: KIVI vs paper validation | W4.2: related work outline |
| 7 (Sun May 10) | W2b.1 finish + aggregate | W1.3: cross-model chart + W2b.2 LongBench Llama kickoff | W4.2: related work prose |
| 8 (Mon May 11) | W2b.2 finish + aggregate | W2b.3 RULER NIAH both models | W4.2: experiments outline |
| 9 (Tue May 12) | W2b.3 finish + W2b.4 charts/tables | W4.3 figures into LaTeX | W4.2: experiments §4.1–4.2 (Stage 1E results) |
| 10 (Wed May 13)| Buffer for re-runs / fixes | W4.2: experiments §4.3 (cross-model + downstream) | W4.2: discussion |
| 11 (Thu May 14)| W4.2: limitations + conclusion | Appendix: port E1/E1_2/E4 sections | W4.4: figure typography pass |
| 12 (Fri May 15)| Full pdflatex build + chktex | Advisor review pass 1 | Address review notes |
| 13 (Sat May 16)| Page-budget pass — trim or expand | Bibliography polish | W4.4: NeurIPS checklist |
| 14 (Sun May 17)| Final figure consistency | Broader-impact statement | Final read-through |
| 15 (Mon May 18)| Buffer | Buffer | **Submission day** |

Days 16–19 are unallocated buffer; if everything is on rails, those days absorb advisor feedback and final polish.

---

## Validation checkpoints (must-pass before next workstream)

| Checkpoint | When | Pass criterion | If fails |
|---|---|---|---|
| C1: Llama bundle valid | End of Day 1 | manifest.json + 24 .pt files; one example loads cleanly | Re-capture; check VRAM, model auth |
| C2: Llama E3 W1 gate | End of Day 3 | JointQK_waterfill top at every (b, layer-0-excluded); ≥5 pp margin at b=3 | Escalate; consider per-model layer-0 logic |
| C3: Press parity | Day 5 | max-abs-diff < 1e-5 vs offline compressor | Audit press wrapper before W2b |
| C4: kvpress smoke | Day 5 | `evaluate.py --fraction 0.05` runs end-to-end and produces `results.json` | Fall back to monkey-patch (Path B in submission plan) |
| C5: KIVI sanity | Day 7 | KIVI int4 within 3 pp of published Qwen number | Audit per-channel scales; relax to "K-only KIVI" disclaimer |
| C5.5: W2c decode-scope decision | End of Day 7 | `decision.txt` written; max\|Mode B − Mode A\| < 2 pp on both ablation tasks → headline uses Mode B, else Mode A | Either decision is publishable; the paper reports both and is explicit about which is the headline |
| C6: W2 gate (downstream) | Day 9 | JointQK at b=3 ≥ 90% of full-precision LongBench task score (in winning mode); beats peers on ≥80% of cells | Add to limitations; reduce claim strength in abstract |
| C7: PDF build clean | Day 12 | `pdflatex` + `bibtex` no errors; ≤9 pages main; chktex no errors | Trim sections; fix bibliography |

---

## Files inventory

### New files (count: ~14 code + ~10 paper)

**Code:**
- `pipelines/scripts/launch_llama_e3_e5.sh`
- `pipelines/scripts/precompute_newbases.py`
- `pipelines/scripts/make_cross_model_chart.py`
- `pipelines/scripts/make_downstream_charts.py`
- `pipelines/scripts/validate_kivi.py`
- `pipelines/eval/aggregate_longbench.py`
- `pipelines/eval/aggregate_ruler.py`
- `pipelines/eval/aggregate_decode_scope.py` (W2c decision)
- `kvq/jointqk_press.py`
- `kvq/turboquant_press.py`
- `kvq/kivi_press.py`
- `kvq/kivi_quantizer.py`
- `tests/test_press_roundtrip_parity.py`
- `artifacts/bases/llama31_8b/` (output dir, populated by W1.2)
- `artifacts/downstream/{qwen3_8b,llama31_8b}/` (output dirs, populated by W2b)

**Paper:**
- `paper/main/main.tex` and `paper/main/refs.bib`
- `paper/main/sections/{abstract,intro,related_work,method,experiments,discussion,conclusion}.tex`
- `paper/main/figs/` (copied + new figures)

### Edited files (small changes)

- `kvpress/evaluation/evaluate_registry.py` — register 3 new presses (~6 lines added)
- (No edits to `run_bases.py`, `build_method_compressor`, or `PerCoordCompressor` — they already work for both models)

### Existing infrastructure reused unchanged

- `experiments/run_bases.py` — bundle-driven, model-agnostic
- `kvq/compression/per_coord.py` — `build_method_compressor` already supports JointQK
- `kvq/metric_transform.py` — water-fill, basis builders, polar orthogonalization
- `kvq/quantization.py` — Lloyd–Max codebooks
- `experiments/gates/gate_e{3,5}.py` — auto-discovery, no edits needed
- `kvpress/kvpress/presses/base_press.py` — BasePress, hook lifecycle (no edits)
- `kvpress/evaluation/evaluate.py` — harness for both LongBench and RULER

---

## Open questions / pre-flight items

1. **Llama-3.1-8B model auth.** Do we have HuggingFace token / model already cached locally? Check before Day 1: `python -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained('meta-llama/Llama-3.1-8B-Instruct'))"`. If not, request access on HF a day in advance (auto-approval is usually < 1h).

2. **Disk budget.** Llama bundle ≈ 24 GB (24 examples × ~1 GB Q/K @ fp16 × 32 layers). Verify under `/vault/amir/efficient-llm/teamily-project/artifacts/`.

3. **`evaluate.py --press_init_command` semantics.** Need to confirm it accepts a Python expression. Day 1 spike: read `kvpress/evaluation/evaluate.py:setup_press` to verify the init pathway and adjust the wrapper API if needed.

4. **kvpress press requires `compression_ratio` field?** The base config schema has `compression_ratio` and `key_channel_compression_ratio`. Quantization presses don't reduce sequence length; we need to confirm `compression_ratio=0.0` is a valid no-op or whether we need to add a quantization-flavored config field.

These three questions are answered in the first 2 hours of Day 1 (W1.1 background run + parallel kvpress code reading).
