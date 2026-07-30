"""NOVA-KV paper, Figure 4 -- WITH a fairly-tuned OSCAR baseline.

Companion to ``plot_paper_fig4.py`` (which stays as-is and still renders
fig4_v7). This one reads every value from the study JSONs at run time and emits
``fig4_v8``, so the two can be compared side by side before deciding which ships.

The difference is the OSCAR series. Previously OSCAR-INT2 ran a single heuristic
tile configuration while NOVA-KV picked its best THR per cell -- tuned-vs-untuned.
Here OSCAR gets the same treatment: ``artifacts/throughput/int2_retune_*`` sweeps
eight (BLOCK_N, BLOCK_H, num_warps, num_stages) settings per cell and this script
takes the winner, exactly as it takes NOVA-KV's winning THR.

Sourcing note: all OSCAR values come from the retune study alone, including its
``oscar_default`` control, rather than being mixed with the older sweep. Run-to-
run noise between the two runs of the *same* default config is 1-2% (e.g. 668.0
vs 683.5 at gpt-oss 90K/bs8), so mixing them would inject that spread into the
comparison. NOVA-KV and BF16 still come from their original studies.
"""
import json
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

REPO = "/raid/amir/quantization/teamily-project"
ART = os.path.join(REPO, "artifacts/throughput")


def rows(study):
    p = os.path.join(ART, study, "throughput_summary.json")
    return json.load(open(p))["rows"] if os.path.exists(p) else {}


def tok(r, arm, cell):
    v = (r.get(arm) or {}).get(cell)
    return (v or {}).get("decode_tok_s") if v else None


# model -> (nova/oscar sweep, bf16 study, int2 retune study, batch ladder)
SPEC = {
    "GPT-OSS-20B":            ("vq2_cuda_thr_sweep_onserver", "throughput_gptoss20b_fp32",
                               "int2_retune_gptoss20b", [1, 4, 8]),
    "Qwen3-8B":               ("vq2_cuda_thr_sweep_qwen3_8b", "throughput_qwen3_8b",
                               "int2_retune_qwen3_8b", [1, 4, 16]),
    "Qwen3-4B-Thinking-2507": ("vq2_cuda_thr_sweep_qwen3_4b_thinking",
                               "throughput_qwen3_4b_thinking",
                               "int2_retune_qwen3_4b", [1, 4, 16]),
}
# BF16 at gpt-oss 90K/bs8 was never in the fp32 ladder; measured separately.
BF16_EXTRA = {("GPT-OSS-20B", "in90000_bs8"): "bf16_gptoss20b_90k_bs8"}
# THR=512 is absent from the Qwen3-8B sweep at 90K; the rerun supplies it.
NOVA_EXTRA = {"Qwen3-8B": "vq2_cuda_thr512_qwen3_8b_90k_rerun"}

MODELS = list(SPEC)
LENGTHS = [30000, 60000, 90000]


def bf16_at(model, cell):
    _, base, _, _ = SPEC[model]
    v = tok(rows(base), "bf16", cell)
    if v is None and (model, cell) in BF16_EXTRA:
        v = tok(rows(BF16_EXTRA[(model, cell)]), "bf16", cell)
    return v


def oscar_at(model, cell):
    """Best of the retuned int2 configs; None until that sweep exists."""
    r = rows(SPEC[model][2])
    cands = [tok(r, a, cell) for a in r]
    cands = [c for c in cands if c]
    return max(cands) if cands else None


def nova_at(model, cell):
    r = rows(SPEC[model][0])
    cands = [tok(r, f"vq2_cuda_thr{t}", cell) for t in (128, 256, 512)]
    if model in NOVA_EXTRA:
        cands.append(tok(rows(NOVA_EXTRA[model]), "vq2_cuda_thr512", cell))
    cands = [c for c in cands if c]
    return max(cands) if cands else None


GETTER = {"BF16": bf16_at, "OSCAR": oscar_at, "NOVA-KV (ours)": nova_at}

missing = [m for m in MODELS if not rows(SPEC[m][2])]
if missing:
    raise SystemExit(f"int2 retune not finished for: {missing}. "
                     "Refusing to plot a half-tuned OSCAR series.")

BRICK, SLATE, GRAY = "#9C3D3D", "#3B4B6B", "#888888"
ARMS = ["BF16", "OSCAR", "NOVA-KV (ours)"]
COLORS = [GRAY, SLATE, BRICK]
SHORT = {"GPT-OSS-20B": "GPT-OSS-20B", "Qwen3-8B": "Qwen3-8B",
         "Qwen3-4B-Thinking-2507": "Qwen3-4B-Thinking"}

mpl.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "cm", "text.usetex": False,
    "font.size": 36, "axes.titlesize": 37, "axes.labelsize": 36,
    "xtick.labelsize": 35, "ytick.labelsize": 35, "legend.fontsize": 36,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.35, "figure.dpi": 300, "savefig.dpi": 300,
})

W = 0.26
PANEL = "abcdef"
fig, axes = plt.subplots(2, 3, figsize=(22.5, 12.0))

for ci, model in enumerate(MODELS):
    bss = SPEC[model][3]

    ax = axes[0, ci]                                   # batch 1 vs input length
    x = np.arange(len(LENGTHS))
    for i, (arm, c) in enumerate(zip(ARMS, COLORS)):
        vals = [GETTER[arm](model, f"in{l}_bs1") for l in LENGTHS]
        ax.bar([x[j] + (i - 1) * W for j, v in enumerate(vals) if v is not None],
               [v for v in vals if v is not None], W, color=c, label=arm)
    ax.set_xticks(x); ax.set_xticklabels(["30K", "60K", "90K"])
    ax.set_xlabel("Input length"); ax.set_title(f"({PANEL[ci]}) {SHORT[model]}")
    ax.set_ylim(0, max(GETTER["NOVA-KV (ours)"](model, f"in{l}_bs1")
                       for l in LENGTHS) * 1.18)
    ax.set_axisbelow(True)
    if ci == 0:
        ax.set_ylabel("Decode tok/s")

    ax = axes[1, ci]                                   # batch scaling at 90K
    x = np.arange(len(bss))
    cells = [f"in90000_bs{b}" for b in bss]
    top = max(v for arm in ARMS for v in
              (GETTER[arm](model, c) for c in cells) if v is not None)
    for i, (arm, c) in enumerate(zip(ARMS, COLORS)):
        vals = [GETTER[arm](model, cell) for cell in cells]
        ax.bar([x[j] + (i - 1) * W for j, v in enumerate(vals) if v is not None],
               [v for v in vals if v is not None], W, color=c, label=arm)
        if arm == "BF16":
            for j, v in enumerate(vals):
                if v is None:      # a capacity result, not a missing run
                    ax.text(x[j] - W, top * 0.03, "n/a", ha="center", va="bottom",
                            fontsize=24, color=GRAY, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([str(b) for b in bss])
    ax.set_xlabel("Batch size"); ax.set_title(f"({PANEL[ci + 3]}) {SHORT[model]}")
    ax.set_ylim(0, top * 1.18); ax.set_axisbelow(True)
    if ci == 0:
        ax.set_ylabel("Decode tok/s")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.04))
fig.tight_layout()
fig.savefig("fig4_v8.pdf", dpi=300, bbox_inches="tight")
fig.savefig("fig4_v8.png", dpi=300, bbox_inches="tight")

print(f"{'model':>20} {'cell':>14} {'BF16':>8} {'OSCAR':>8} {'NOVA':>8} "
      f"{'N/BF16':>8} {'N/OSCAR':>9}")
for model in MODELS:
    cells = [f"in{l}_bs1" for l in LENGTHS] + \
            [f"in90000_bs{b}" for b in SPEC[model][3][1:]]
    for cell in cells:
        b, o, n = (bf16_at(model, cell), oscar_at(model, cell), nova_at(model, cell))
        if not (o and n):
            continue
        print(f"{SHORT[model]:>20} {cell:>14} "
              f"{(round(b, 1) if b else '-'):>8} {o:>8.1f} {n:>8.1f} "
              f"{(f'{n/b:.2f}x' if b else '-'):>8} {n/o:>8.3f}x")
