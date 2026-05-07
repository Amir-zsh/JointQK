#!/usr/bin/env python3
"""Mine per-(layer, head) top-1 retention from preview_pooled_n50 shard JSONs.

Tests hypothesis A3: JointQK's top-1 lead may be concentrated in F1-irrelevant
layers (early/middle feature-mixing) rather than late "answer-extraction" layers.

Reads:
  artifacts/stage1/calibration/longbench_compact8_qkv/05_reports/
    preview_pooled_n50/shard_NNN.json

Each shard contains accumulators[method][bits][layer][stat] where stat ∈
{top1_num, top1_den, ...}. We sum across shards (already done in `merge_accumulators`),
then derive top1 ratios per (method, bits, layer).

Outputs:
  artifacts/stage1/calibration/longbench_compact8_qkv/05_reports/perlayer_top1.json
  Plus a printed table.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from experiments.stage1.calibration.analyze_bases import merge_accumulators


def main() -> None:
    shard_dir = REPO / "artifacts/stage1/calibration/longbench_compact8_qkv/05_reports/preview_pooled_n50"
    shard_files = sorted(shard_dir.glob("shard_*.json"))
    print(f"loading {len(shard_files)} shards from {shard_dir}")

    payloads = [json.loads(f.read_text()) for f in shard_files]
    methods = payloads[0]["methods"]
    k_bits = payloads[0]["k_bits"]
    n_layers = int(payloads[0]["n_layers_total"])

    # Merge per-method accumulators across shards.
    per_method_accums: dict[str, dict] = {}
    for method in methods:
        shards_for_method = [
            {int(b): {int(l): dict(stats) for l, stats in per_layer.items()}
             for b, per_layer in p["accumulators"][method].items()}
            for p in payloads
        ]
        per_method_accums[method] = merge_accumulators(shards_for_method)

    # Per-(method, bits, layer) top-1 = top1_num / top1_den.
    # Per-(method, bits, layer) k_mse and logit_err for completeness.
    per_layer_metrics: dict[str, dict[int, dict[int, dict[str, float]]]] = {}
    for method, by_bits in per_method_accums.items():
        per_layer_metrics[method] = {}
        for bits, per_layer in by_bits.items():
            per_layer_metrics[method][bits] = {}
            for L in sorted(per_layer.keys()):
                acc = per_layer[L]
                top1 = acc["top1_num"] / max(1, acc["top1_den"])
                top5 = acc["top5_num"] / max(1, acc["top5_den"])
                kmse = acc["mse_num"] / max(1, acc["mse_den"])
                logit = acc["logit_num"] / max(1, acc["logit_den"])
                per_layer_metrics[method][bits][L] = {
                    "top1": top1, "top5": top5, "k_mse": kmse, "logit_err": logit
                }

    # Print: at each bit width, layer-by-layer top-1 for each method, plus jointqk - v3 delta.
    for bits in k_bits:
        print(f"\n=== b={bits}: per-layer top-1 retention ===")
        # Header
        print(f"{'layer':>5} | " + " | ".join(f"{m:>8}" for m in methods) + " | jointqk-v3")
        print("-" * (5 + 3 + 11 * len(methods) + 12))
        for L in range(n_layers):
            cells = [per_layer_metrics[m][bits][L]["top1"] for m in methods]
            jq = per_layer_metrics["jointqk"][bits][L]["top1"]
            v3 = per_layer_metrics["v3"][bits][L]["top1"]
            delta = jq - v3
            print(f"{L:>5} | " + " | ".join(f"{c:>8.4f}" for c in cells) + f" | {delta:+.4f}")

    # Summary: split layers into bands [early, middle, late] and compare.
    print(f"\n=== Layer-band summary (top-1 retention, jointqk vs v3) ===")
    n = n_layers
    bands = [
        ("early (1-12)", list(range(1, 12))),
        ("middle (12-24)", list(range(12, 24))),
        ("late (24-35)", list(range(24, 35))),
    ]
    for bits in k_bits:
        print(f"\n  b={bits}:")
        for band_name, band_layers in bands:
            jq_vals = [per_layer_metrics["jointqk"][bits][L]["top1"] for L in band_layers]
            v3_vals = [per_layer_metrics["v3"][bits][L]["top1"] for L in band_layers]
            qo_vals = [per_layer_metrics["q_only"][bits][L]["top1"] for L in band_layers]
            ko_vals = [per_layer_metrics["k_only"][bits][L]["top1"] for L in band_layers]
            jq_mean = sum(jq_vals) / len(jq_vals)
            v3_mean = sum(v3_vals) / len(v3_vals)
            qo_mean = sum(qo_vals) / len(qo_vals)
            ko_mean = sum(ko_vals) / len(ko_vals)
            print(f"    {band_name:<16} jointqk={jq_mean:.4f}  v3={v3_mean:.4f}  q_only={qo_mean:.4f}  k_only={ko_mean:.4f}  | jq-v3={jq_mean - v3_mean:+.4f}")

    # Save the full layer-by-layer JSON.
    out_path = REPO / "artifacts/stage1/calibration/longbench_compact8_qkv/05_reports/perlayer_top1.json"
    out_path.write_text(json.dumps(per_layer_metrics, indent=2, default=str) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
