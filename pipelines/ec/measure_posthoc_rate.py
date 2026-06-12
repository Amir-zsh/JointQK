#!/usr/bin/env python3
"""Post-hoc honest coded rate of shipped EC bundles on the BENCH tasks.

Applies the FROZEN coder model (no refit, measurement only) to eval-side
captured rows of lcc / musique / 2wikimqa and reports bits/coord per task —
the rate number quoted next to each task's F1.

    python pipelines/ec/measure_posthoc_rate.py \
        --bundles artifacts/ec/llama31_8b/ec_bundle__{r_sym,hadamard}__*dz0.375*.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import _bootstrap  # noqa: E402,F401

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import constriction  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from kvq.compression.ec_roundtrip import bundle_model_to_dict, dz_round  # noqa: E402

RUN_ID = "ec_posthoc_rate_llama31_8b"
RAW_ROOT = REPO / "artifacts/calibration" / RUN_ID / "01_raw"
OUT_JSON = REPO / "artifacts/ec/llama31_8b/posthoc_rates.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", nargs="+", required=True)
    args = ap.parse_args()
    t0 = time.time()

    files = sorted(RAW_ROOT.glob("shard_*/*.pt"))
    if not files:
        sys.exit(f"no captured rows under {RAW_ROOT} — run the posthoc capture first")
    by_task: dict[str, list[Path]] = {}
    for f in files:
        task = f.name.split("__")[1]
        by_task.setdefault(task, []).append(f)
    print(f"[posthoc] rows: " + " ".join(f"{t}={len(v)}" for t, v in sorted(by_task.items())))

    out = {}
    for bp in args.bundles:
        blob = torch.load(bp, map_location="cpu", weights_only=False)
        model = bundle_model_to_dict(blob)
        L, Hkv, d = blob["delta"].shape
        dz = float(blob["dz"])
        label = f"ec_{blob['basis']}_dz{blob['dz']:g}"
        bits = {t: 0.0 for t in by_task}
        toks = {t: 0 for t in by_task}
        arts = {t: [torch.load(f, map_location="cpu", mmap=True, weights_only=False)
                    for f in fs] for t, fs in by_task.items()}
        for t, alist in arts.items():
            toks[t] = sum(int(a["prompt_length"]) for a in alist)
        for l in range(1, L):
            for h in range(Hkv):
                per = model[(l, h)]
                dlh = blob["delta"][l, h].double().clamp_min(1e-12)
                Fd = blob["forward"][l, h].double()
                mud = blob["mu"][l, h].double()
                for t, alist in arts.items():
                    for a in alist:
                        T = int(a["prompt_length"])
                        r = (a["k_post"][l, h, :T].double() - mud) @ Fd
                        idx = dz_round(r, dlh.unsqueeze(0), dz).long().numpy()
                        for j in range(d):
                            vals, p = per[j]
                            if vals.size <= 1:
                                continue
                            col = idx[:, j]
                            pos = np.clip(np.searchsorted(vals, col), 0, vals.size - 1)
                            left = np.clip(pos - 1, 0, vals.size - 1)
                            cl = np.abs(vals[left] - col) <= np.abs(vals[pos] - col)
                            sym = np.where(cl, left, pos).astype(np.int32)
                            enc = constriction.stream.queue.RangeEncoder()
                            enc.encode(sym, constriction.stream.model.Categorical(p, perfect=False))
                            bits[t] += len(enc.get_compressed()) * 32
            print(f"[posthoc] {label} layer {l+1}/{L} ({time.time()-t0:.0f}s)", flush=True)
        rates = {t: bits[t] / (toks[t] * (L - 1) * Hkv * d) for t in by_task}
        out[label] = rates
        print(f"[posthoc] {label}: " + " ".join(f"{t}={r:.4f}" for t, r in sorted(rates.items())))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    existing.update(out)
    OUT_JSON.write_text(json.dumps(existing, indent=1))
    print(f"[posthoc] wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
