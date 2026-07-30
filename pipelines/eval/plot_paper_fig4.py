"""NOVA-KV paper, Figure 4 -- decode throughput: BF16, OSCAR, NOVA-KV.

Row 1: batch 1 against input length.  Row 2: batch scaling at 90K input.

(Not to be confused with ``plot_fig4.py``, which reproduces the *OSCAR* paper's
Figure 4 from ``artifacts/oscar_e2e/fig4_*``.)

Every number below is ``decode_tok_s`` read from the study JSONs named per block,
not recovered from a rendered chart. The previous revision was transcribed from a
figure and drifted: the batch-16 cells were +6-7% on both Qwen models -- both
arms, so the ratio survived but the absolute tok/s did not.

NOVA-KV is the vq2 CUDA arm at its best THR for that cell. The winning THR is
annotated per bar because it is NOT constant (512 at batch 1, 256 at batch 4, 128
at batch 8/16 -- each a separately compiled kernel), whereas OSCAR-INT2 is a
single untuned tile config. That asymmetry favours NOVA-KV and belongs in the
caption.

The two BF16 gaps in row 2 were different in kind, which is why only one remains:
  GPT-OSS 90K bs=8   -> never measured, though BF16 admits b_max=10 there so it
                        fits. Now measured: artifacts/throughput/
                        bf16_gptoss20b_90k_bs8 (all six gates pass; monotone with
                        its neighbours, 370.0 -> 644.0 -> 704.7 at bs4/8/10).
  Qwen    90K bs=16  -> genuinely does not fit; BF16 b_max=4 at 90K. Drawn as an
                        explicit "n/a" rather than a silent absence, so it reads
                        as the capacity result it is and not as a missing run.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "cm",
    "text.usetex": False,
    "font.size": 36,
    "axes.titlesize": 37,
    "axes.labelsize": 36,
    "xtick.labelsize": 35,
    "ytick.labelsize": 35,
    "legend.fontsize": 36,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

BRICK = "#9C3D3D"   # NOVA-KV
SLATE = "#3B4B6B"   # OSCAR
GRAY  = "#888888"   # BF16 reference

ARMS   = ["BF16", "OSCAR", "NOVA-KV (ours)"]
COLORS = [GRAY, SLATE, BRICK]

MODELS  = ["GPT-OSS-20B", "Qwen3-8B", "Qwen3-4B-Thinking-2507"]
SHORT   = {"GPT-OSS-20B": "GPT-OSS-20B",
           "Qwen3-8B": "Qwen3-8B",
           "Qwen3-4B-Thinking-2507": "Qwen3-4B-Thinking"}
LENGTHS = ["30K", "60K", "90K"]

# ---- row 1: batch 1, by input length ----------------------------------
#   BF16  <- throughput_gptoss20b_fp32 | throughput_qwen3_8b | throughput_qwen3_4b_thinking
#   OSCAR <- vq2_cuda_thr_sweep_{onserver,qwen3_8b,qwen3_4b_thinking}, arm oscar_int2
#   NOVA  <- same sweeps, best of vq2_cuda_thr{128,256,512}
BS1 = {
    "GPT-OSS-20B":            {"BF16":  [173.8, 129.6, 103.1],
                               "OSCAR": [190.9, 158.6, 134.8],
                               "NOVA-KV (ours)": [199.6, 173.8, 153.7]},
    "Qwen3-8B":               {"BF16":  [ 50.8,  30.8,  22.6],
                               "OSCAR": [ 94.5,  72.5,  59.2],
                               "NOVA-KV (ours)": [100.4,  81.8,  68.7]},
    "Qwen3-4B-Thinking-2507": {"BF16":  [ 59.0,  33.7,  23.6],
                               "OSCAR": [122.7,  88.4,  69.2],
                               "NOVA-KV (ours)": [131.8, 100.9,  81.4]},
}
THR1 = {m: [512, 512, 512] for m in MODELS}   # winning THR, batch 1

# ---- row 2: batch scaling at 90K input --------------------------------
BATCHES = {
    "GPT-OSS-20B":            ["1", "4", "8"],
    "Qwen3-8B":               ["1", "4", "16"],
    "Qwen3-4B-Thinking-2507": ["1", "4", "16"],
}
BS90K = {
    "GPT-OSS-20B":            {"BF16":  [103.1, 370.0, 644.0],   # bs8 newly measured
                               "OSCAR": [134.8, 445.1, 683.5],
                               "NOVA-KV (ours)": [153.7, 452.4, 649.9]},
    "Qwen3-8B":               {"BF16":  [ 22.6,  90.7,  None],   # b_max=4 at 90K
                               "OSCAR": [ 59.2, 174.5, 312.9],
                               "NOVA-KV (ours)": [ 68.7, 192.2, 276.1]},
    "Qwen3-4B-Thinking-2507": {"BF16":  [ 23.6,  95.2,  None],   # b_max=4 at 90K
                               "OSCAR": [ 69.2, 195.6, 329.4],
                               "NOVA-KV (ours)": [ 81.4, 214.3, 287.7]},
}
THR90K = {
    "GPT-OSS-20B":            [512, 256, 128],
    "Qwen3-8B":               [512, 512, 128],
    "Qwen3-4B-Thinking-2507": [512, 512, 128],
}

# Qwen3-8B's 90K NOVA cells come from vq2_cuda_thr512_qwen3_8b_90k_rerun: THR=512
# is missing from the main sweep at 90K, and without the rerun the peak speedup
# reads 2.68x instead of 3.05x.

ANNOTATE_THR = False   # per-bar THR labels; kept off for the camera-ready.
                       # The tuning asymmetry they documented (NOVA-KV tuned per
                       # cell, OSCAR-INT2 a single tile config) still belongs in
                       # the caption -- see THR1 / THR90K above for the values.

W = 0.26
PANEL = "abcdef"

fig, axes = plt.subplots(2, 3, figsize=(22.5, 12.0))

for ci, model in enumerate(MODELS):
    # ---- top row: batch 1 vs input length
    ax = axes[0, ci]
    x = np.arange(len(LENGTHS))
    for i, (arm, c) in enumerate(zip(ARMS, COLORS)):
        ax.bar(x + (i - 1) * W, BS1[model][arm], W, color=c, label=arm)
    ax.set_xticks(x)
    ax.set_xticklabels(LENGTHS)
    ax.set_xlabel("Input length")
    ax.set_title(f"({PANEL[ci]}) {SHORT[model]}")
    ax.set_ylim(0, max(BS1[model]["NOVA-KV (ours)"]) * 1.18)
    ax.set_axisbelow(True)
    if ANNOTATE_THR:
        for j, v in enumerate(BS1[model]["NOVA-KV (ours)"]):
            ax.text(x[j] + W, v, str(THR1[model][j]), ha="center", va="bottom",
                    fontsize=20, color=BRICK)
    if ci == 0:
        ax.set_ylabel("Decode tok/s")

    # ---- bottom row: batch scaling at 90K
    ax = axes[1, ci]
    labels = BATCHES[model]
    x = np.arange(len(labels))
    top = max(v for vals in BS90K[model].values() for v in vals if v is not None)
    for i, (arm, c) in enumerate(zip(ARMS, COLORS)):
        vals = BS90K[model][arm]
        xs = [x[j] + (i - 1) * W for j, v in enumerate(vals) if v is not None]
        ys = [v for v in vals if v is not None]
        ax.bar(xs, ys, W, color=c, label=arm)
        # A silently absent bar reads as "not run". Mark the cell BF16 cannot
        # reach so the gap reads as the capacity result it is.
        if arm == "BF16":
            for j, v in enumerate(vals):
                if v is None:
                    ax.text(x[j] - W, top * 0.03, "n/a", ha="center", va="bottom",
                            fontsize=24, color=GRAY, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Batch size")
    ax.set_title(f"({PANEL[ci + 3]}) {SHORT[model]}")
    ax.set_ylim(0, top * 1.18)
    ax.set_axisbelow(True)
    if ANNOTATE_THR:
        for j, v in enumerate(BS90K[model]["NOVA-KV (ours)"]):
            ax.text(x[j] + W, v, str(THR90K[model][j]), ha="center", va="bottom",
                    fontsize=20, color=BRICK)
    if ci == 0:
        ax.set_ylabel("Decode tok/s")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.04))
fig.tight_layout()
fig.savefig("fig4_v7.pdf", dpi=300, bbox_inches="tight")
fig.savefig("fig4_v7.png", dpi=300, bbox_inches="tight")

# ---- ratios over the cells THIS FIGURE plots. Not the same set as the text's
# "batch 1 and batch 4" claim, which also covers 30K/60K at batch 4 -- those are
# not plotted here, and they hold the low end (e.g. Qwen3-8B 30K bs4 = 1.54x).
print(f"{'model':>22} {'NOVA/BF16 (plotted)':>24} {'NOVA/OSCAR (plotted)':>26}")
for m in MODELS:
    rb = [BS1[m]["NOVA-KV (ours)"][k] / BS1[m]["BF16"][k] for k in range(3)]
    rb += [BS90K[m]["NOVA-KV (ours)"][1] / BS90K[m]["BF16"][1]]
    ro = ([BS1[m]["NOVA-KV (ours)"][k] / BS1[m]["OSCAR"][k] for k in range(3)]
          + [BS90K[m]["NOVA-KV (ours)"][k] / BS90K[m]["OSCAR"][k] for k in range(3)])
    print(f"{SHORT[m]:>22} {min(rb):>10.2f}x .. {max(rb):<10.2f} "
          f"{min(ro):>11.3f}x .. {max(ro):<11.3f}")
b = BS90K["GPT-OSS-20B"]
print(f"\nGPT-OSS 90K bs8 (new BF16 cell): BF16={b['BF16'][2]}  "
      f"OSCAR={b['OSCAR'][2]} ({b['OSCAR'][2]/b['BF16'][2]:.2f}x BF16)  "
      f"NOVA={b['NOVA-KV (ours)'][2]} ({b['NOVA-KV (ours)'][2]/b['BF16'][2]:.2f}x BF16)")
