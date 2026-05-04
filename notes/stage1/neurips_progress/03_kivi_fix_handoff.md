# KIVI Baseline Fix — Status

This note supersedes the handoff prompt preserved below. The KIVI baseline
mismatch has been diagnosed and the local implementation has been fixed without
replacing `KIVIPress` with direct calls into vendored upstream code.

## Current Status

`experiments/stage1/toolkit/kivi_quantizer.py` now mirrors official
`jy-yuan/KIVI` quantization math while preserving our local reconstruction-only
experiment API:

- K groups along sequence length (`T`) with `group_size` tokens per group.
  The divisible prefix is quantized and any `T % group_size` tail is left fp16,
  matching KIVI residual-cache semantics.
- V groups along head dimension (`D`).
- Both paths use min-subtraction asymmetric quantization:
  `q = round((x - mn) / scale)` and `x_hat = q * scale + mn`.
- `KIVIPress` did not require a code change because it already calls
  `kivi_quantize_keys` and `kivi_quantize_values`.

The verified command is:

```bash
.venv/bin/python experiments/stage1/tests/test_kivi_press_parity.py
```

Passing checks:

- V parity against upstream KIVI pack/dequant path: `max_abs_diff = 0`.
- K parity for divisible sequence lengths: `max_abs_diff = 0`.
- K residual-tail behavior for non-divisible sequence lengths:
  `max_abs_diff = 0`.
- `KIVIPress.compress()` parity against the direct local functions:
  `max_abs_diff = 0` for both K and V.

Phase 7 policy after the fix:

- KIVI uses `group_size=128`, matching official KIVI's default CLI setting.
- Generation-time KV quantization is disabled for all compressed methods:
  Phase 7 LongBench and RULER launchers pass `compress_decode=False`.
- Existing KIVI Phase 7 rows produced before this fix should be treated as
  invalid. Move them aside with a `.pre_kivi_fix` suffix or otherwise exclude
  them before any post-fix KIVI rerun; preserve the audit trail.

Why we did **not** use the original Option B below: keeping a local PyTorch
implementation maintains the same experiment surface as JointQK and TurboQuant.
All methods return reconstructed tensors through the same kvpress hook, and
packing/storage details are not part of the quality comparison.

Remaining follow-ups:

- Re-run KIVI rows only after the user explicitly asks to resume Phase 7.
- Update final paper wording to say KIVI uses official KIVI quantization rule
  and grouping, implemented locally as a reconstruction-only press.
- Add a separate stateful decode-mode KIVI press only if an apples-to-published
  rolling-residual-window comparison is needed.

---

# Archived Handoff Prompt — KIVI baseline fix

Use this as the opening prompt for a fresh chat to continue the KIVI work.
Self-contained — no prior context needed.

---

## Prompt to paste into new chat

> I'm continuing integrity verification for the JointQK KV-cache compression
> paper. The current task is to fix our KIVI baseline so it matches the
> official KIVI implementation (jy-yuan/KIVI) byte-for-byte (or as close as
> possible).
>
> ### What's already done
>
> 1. **Vendored upstream KIVI** at `vendor/kivi/` (cloned from
>    https://github.com/jy-yuan/KIVI, MIT license). The reference quantize
>    functions live in `vendor/kivi/quant/new_pack.py`:
>    - `quant_and_pack_kcache(k, group_size, bits)` → `(code, scale, mn)`
>    - `quant_and_pack_vcache(v, group_size, bits)` → `(code, scale, mn)`
>    - `unpack_and_dequant_kcache(code, scale, mn, group_size, bits)`
>    - `unpack_and_dequant_vcache(code, scale, mn, group_size, bits)`
>    Triton is required (already installed: `triton 3.6.0`).
>
> 2. **Parity test exists** at
>    `experiments/stage1/tests/test_kivi_press_parity.py`. Already runs and
>    reports the gaps. Last run logged at
>    `experiments/stage1/logs/integrity/tier0/05_kivi_parity.log`.
>
> 3. **Two real gaps discovered between our impl and upstream:**
>    - **Zero-point handling.** Upstream computes `q = round((x - mn)/scale)`
>      with implicit zero. Ours pre-rounds the zero-point and rounds the
>      shifted value, which introduces up to 1 LSB extra error per element
>      (V parity max_abs_diff ≈ 0.46 at int4, rmse ≈ 0.14).
>    - **K grouping scheme is fundamentally different.** Upstream groups
>      along T (sequence) with `group_size` tokens per group → multiple
>      quantizers per channel. Ours groups along D (channel) and reduces
>      over the full S → one global quantizer per channel. Result: ours has
>      ~40% higher K reconstruction error than upstream at T=4096 / GS=128.
>
> ### The task
>
> Implement **Option B from the prior chat**: replace our
> `KIVIPress.compress` to call upstream's `quant_and_pack_kcache` /
> `quant_and_pack_vcache` (and their unpack/dequant counterparts) directly,
> rather than going through our own `kivi_quantizer.py` math.
>
> Files involved:
> - `experiments/stage1/toolkit/kivi_press.py` — the BasePress wrapper
>   (modify the `compress` method to call upstream).
> - `experiments/stage1/toolkit/kivi_quantizer.py` — our standalone quant
>   functions. Either delete or keep as a deprecated reference; the press
>   should no longer call them.
>
> Key constraints from upstream's API:
> - `quant_and_pack_kcache` asserts `T % group_size == 0`. Our wrapper needs
>   to handle the case where the prefill length doesn't divide evenly — the
>   simplest fix is to leave a remainder of `T mod group_size` tokens in
>   fp16 and only quantize the divisible prefix. Match KIVI's published
>   "residual fp16 window" semantics if possible — see
>   `vendor/kivi/models/llama_kivi.py` for how upstream handles this in the
>   end-to-end path.
> - Inputs to upstream are `(B, nh, T, D)` fp16 tensors. Make sure
>   `keys.contiguous().to(torch.float16)` before calling.
> - `unpack_and_dequant_*` returns fp16; cast back to original dtype before
>   returning so the kvpress hook gets a tensor in the model's working
>   dtype.
>
> ### Acceptance criteria
>
> 1. **Parity test passes.** Re-run
>    `python experiments/stage1/tests/test_kivi_press_parity.py`. The "K
>    parity at T == group_size" and "V parity" sections should now show
>    `max_abs_diff < 1e-3` (effectively byte-exact modulo float dtype noise),
>    not 0.46. The "K demonstration" section is no longer interesting since
>    both paths are now upstream — remove or repurpose it.
>
> 2. **Tier 0.2 / 0.3 style byte-exactness.** Add a new test case that
>    confirms `KIVIPress.compress(K, V)` produces output byte-equivalent to
>    calling `unpack_and_dequant_*(quant_and_pack_*(K, V))` directly.
>
> 3. **Phase 7 KIVI rows must be re-run** since the prior ones used the
>    broken impl. Existing artifacts at
>    `artifacts/stage1/downstream/qwen3_8b/kivi_int4_*` should either be
>    deleted (so Phase 7 reruns them on next launch) or moved to
>    `artifacts/stage1/downstream/qwen3_8b/kivi_int4_*.pre_kivi_fix`. Don't
>    blow them away without saving — the user values audit trails.
>
> 4. **Update the integrity progress note**
>    `notes/stage1/neurips_progress/02_integrity_results_so_far.md` with a
>    "Tier 0.5 — KIVI parity" section noting that the original Phase 7 KIVI
>    numbers were biased low (weaker than published KIVI), the fix has been
>    applied, and any post-fix re-run results.
>
> ### Why option B (wrap upstream) over option A (rewrite ours to match)
>
> - One-time refactor instead of debugging quantization math.
> - We can cite "we use the official jy-yuan/KIVI implementation" in the
>   paper. Stronger than a from-scratch reimplementation.
> - Triton kernel path makes it production-fast on GPU; pure-PyTorch
>   reproduction would be slower.
> - If upstream fixes a bug in their kernel, we get the fix automatically by
>   updating the vendored copy.
>
> ### Reference: integrity context already on disk
>
> - `notes/stage1/neurips_progress/01_integrity_verification_plan.md` —
>   the overall plan.
> - `notes/stage1/neurips_progress/02_integrity_results_so_far.md` — Tiers
>   0.1 / 0.2 / 0.3 / 0.4 + Tier 1.1+1.2 + Tier 1.3 results.
> - `notes/stage1/neurips_progress/03_kivi_fix_handoff.md` — this file.
>
> ### Things to be careful about
>
> - **Don't modify the vendored KIVI code under `vendor/kivi/`.** It's a
>   reference; we treat it as read-only.
> - **The user prefers explicit confirmation before destructive ops** like
>   deleting Phase 7 result subdirs. When in doubt, move + suffix instead.
> - **Phase 7 may or may not be running.** Check `ps -ef | grep evaluate`
>   before launching anything new on GPUs 0–5; coexist if it is, queue if
>   it isn't. GPUs 0–5 are the only ones available; 6 and 7 belong to
>   another user.
> - **No commits without explicit "commit now" approval.**
>
> ### Optional follow-ups, after the core fix lands
>
> - Run the layer0_full=False ablation on Qwen3-8B with the *fixed* KIVI to
>   add a head-to-head row showing JointQK vs faithful-KIVI (separate from
>   TurboQuant comparison).
> - Sweep KIVI's `group_size ∈ {32, 64, 128}` once at K=2 / 4 to find
>   their best operating point — currently we lock to group_size=128.
> - Add a residual-fp16-window flag to `KIVIPress` so we can run KIVI in
>   its published configuration (residual window) for an apples-to-published
>   comparison alongside the apples-to-our-other-methods comparison.
>
> Please start by reading the parity test
> (`experiments/stage1/tests/test_kivi_press_parity.py`), confirming the
> upstream API in `vendor/kivi/quant/new_pack.py:8-83`, and proposing the
> minimal `KIVIPress.compress` rewrite before implementing.
