#!/usr/bin/env python3
"""Score EC bundle candidates against TurboQuant/JointQK K=2 baselines on the
held-out selection rows (K-fidelity proxies: top-1, top-5, k_mse, logit_err).

Baselines use the *deployed* code paths:
  - tq_k2: TurboQuantV3.key_compressor per layer (seed 42+1000*L, unit-norm +
    LloydMax — the same object TurboQuantPress runs).
  - jq_k2: build_jointqk_compressor("r_sym_waterfill") from the production
    calibration bundle (the same object JointQKPress runs).
EC candidates are SnappedDeadzoneECCompressor (snap included — what the rANS
codec actually reconstructs), loaded from fit_ec_bundle.py bundles.

Headline metrics exclude layer 0 (project convention). Per-task breakout kept
(repobench-p = in-calibration code sentinel for OOD lcc).

    python pipelines/ec/score_ec_candidates.py --device cuda:0 \
        --bundles artifacts/ec/llama31_8b/ec_bundle__*.pt
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

import torch  # noqa: E402

from kvq.compression.ec_roundtrip import load_ec_compressors_from_bundle  # noqa: E402
from kvq.compression.per_coord import build_jointqk_compressor  # noqa: E402

sys.path.insert(0, str(REPO / "vendor"))
from turboquant_pytorch.compressors_v3 import TurboQuantV3  # noqa: E402

RUN_ID = "ec_calib_compact8_train_llama31_8b"
RAW_ROOT = REPO / "artifacts/calibration" / RUN_ID / "01_raw"
ROLES = REPO / "artifacts/calibration_splits/ec_compact8_train_26/roles.json"
CCA_STATS = REPO / "artifacts/bases/jointqk_llama31_8b_longbench_compact8_n400.pt"
OUT_JSON = REPO / "artifacts/ec/llama31_8b/phaseA_report.json"

Q_CHUNK = 8192  # query rows per attention chunk (bounds (Q_CHUNK, T) buffers)


def raw_path(config: str, row_index: int) -> Path:
    name = f"longbench__{config}__row{int(row_index):05d}__train.pt"
    hits = sorted(RAW_ROOT.glob(f"shard_*/{name}"))
    if len(hits) != 1:
        raise FileNotFoundError(name)
    return hits[0]


class TQKeyRoundtrip:
    """K-side of TurboQuantPress for one layer (chunked like the press)."""

    def __init__(self, head_dim: int, bits: int, layer: int, n_layers: int):
        self.tq = TurboQuantV3(head_dim=head_dim, key_bits=bits, value_bits=bits,
                               residual_window=0, layer_idx=layer, n_layers=n_layers,
                               protected_layers=0, seed=42, device="cpu")

    def to(self, device):
        kc = self.tq.key_compressor
        kc.Pi = kc.Pi.to(device)
        kc.centroids = kc.centroids.to(device)
        kc.device = str(device)
        return self

    @torch.no_grad()
    def roundtrip(self, k: torch.Tensor) -> torch.Tensor:  # (T, d)
        kc = self.tq.key_compressor
        out = torch.empty_like(k)
        for s in range(0, k.shape[0], 2048):  # press chunks at 2048 tokens
            chunk = k[s:s + 2048].unsqueeze(0).unsqueeze(0)
            out[s:s + 2048] = kc.decompress(kc.compress(chunk)).squeeze(0).squeeze(0)
        return out


def zero_acc():
    return dict(mse_num=0.0, mse_den=0, logit_num=0.0, logit_den=0,
                top1_num=0, top1_den=0, top5_num=0, top5_den=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", nargs="+", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--k-bits", type=int, default=2)
    args = ap.parse_args()
    dev = torch.device(args.device)
    t0 = time.time()

    roles = json.loads(ROLES.read_text())
    sel_rows = [(c, int(i)) for c, i in roles["selection"]]

    cca = torch.load(CCA_STATS, map_location="cpu", weights_only=False)
    L, Hkv, d, _ = cca["sigma_q"].shape

    methods: dict[str, dict] = {}
    rates: dict[str, dict] = {}
    for bp in args.bundles:
        comps, meta = load_ec_compressors_from_bundle(bp)
        label = f"ec_{meta['basis']}_dz{meta['dz']:g}"
        methods[label] = comps
        rates[label] = {"pooled": meta["achieved_rate_heldout_pooled"],
                        "per_task": meta["achieved_rate_per_task"]}
    methods["tq_k2"] = {(l, h): None for l in range(1, L) for h in range(Hkv)}
    tq_layers = {l: TQKeyRoundtrip(d, args.k_bits, l, L).to(dev) for l in range(1, L)}
    jq = {}
    for l in range(1, L):
        for h in range(Hkv):
            jq[(l, h)] = build_jointqk_compressor(
                method="r_sym_waterfill",
                sigma_q_for_head=cca["sigma_q"][l, h],
                sigma_k_for_head=cca["sigma_k"][l, h],
                R_sym=cca["R_sym"][l, h],
                b_avg=float(args.k_bits), r=64, head_dim=d)
    methods["jq_k2"] = jq
    print(f"[score] methods: {sorted(methods)} | rows={len(sel_rows)}")

    acc = {m: {} for m in methods}     # acc[m][task][layer] -> counters
    for ri, (task, idx) in enumerate(sel_rows):
        art = torch.load(raw_path(task, idx), map_location="cpu", mmap=True,
                         weights_only=False)
        T = int(art["prompt_length"])
        gs = art["q_post"].shape[1] // Hkv
        print(f"[score] row {ri+1}/{len(sel_rows)} {task} T={T} ({time.time()-t0:.0f}s)",
              flush=True)
        for l in range(1, L):
            for h in range(Hkv):
                k = art["k_post"][l, h, :T].to(dev).float()                      # (T, d)
                q = art["q_post"][l, h * gs:(h + 1) * gs, :T].to(dev).float().reshape(-1, d)
                qq = q.transpose(0, 1) @ q                                        # (d, d)
                n_q = q.shape[0]
                k5 = min(5, T)
                recons = {}
                for m, comps in methods.items():
                    if m == "tq_k2":
                        recons[m] = tq_layers[l].roundtrip(k)
                    else:
                        recons[m] = comps[(l, h)].to(dev).roundtrip(k).float()
                # one chunked pass over queries computes real argmax + per-method tops
                hits1 = {m: 0 for m in methods}
                hits5 = {m: 0 for m in methods}
                for s in range(0, n_q, Q_CHUNK):
                    qa = q[s:s + Q_CHUNK]
                    real_top = (qa @ k.transpose(0, 1)).argmax(-1)
                    for m in methods:
                        ap_l = qa @ recons[m].transpose(0, 1)
                        hits1[m] += int((real_top == ap_l.argmax(-1)).sum())
                        top5 = ap_l.topk(k5, dim=-1).indices
                        hits5[m] += int((top5 == real_top.unsqueeze(-1)).any(-1).sum())
                        del ap_l
                    del real_top
                for m in methods:
                    a = acc[m].setdefault(task, {}).setdefault(l, zero_acc())
                    err = k - recons[m]
                    a["mse_num"] += float(err.square().sum()); a["mse_den"] += err.numel()
                    ee = err.transpose(0, 1) @ err
                    a["logit_num"] += float((qq * ee).sum()); a["logit_den"] += n_q * T * T
                    a["top1_num"] += hits1[m]; a["top1_den"] += n_q
                    a["top5_num"] += hits5[m]; a["top5_den"] += n_q
                del recons

    def fold(counters: list[dict]) -> dict:
        s = {k: sum(c[k] for c in counters) for k in zero_acc()}
        return dict(top1=s["top1_num"] / max(1, s["top1_den"]),
                    top5=s["top5_num"] / max(1, s["top5_den"]),
                    k_mse=s["mse_num"] / max(1, s["mse_den"]),
                    logit_err=s["logit_num"] / max(1, s["logit_den"]))

    report = {"k_bits": args.k_bits, "selection_rows": roles["selection"],
              "rates": rates, "pooled": {}, "per_task": {}}
    for m in methods:
        report["pooled"][m] = fold([acc[m][t][l] for t in acc[m] for l in acc[m][t]])
        report["per_task"][m] = {t: fold(list(acc[m][t].values())) for t in acc[m]}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=1))
    print(f"\n{'method':<22} {'top1':>7} {'top5':>7} {'k_mse':>10} {'logit_err':>10} {'rate':>6}")
    for m in sorted(report["pooled"], key=lambda m: -report["pooled"][m]["top1"]):
        p = report["pooled"][m]
        rate = rates.get(m, {}).get("pooled", float(args.k_bits))
        print(f"{m:<22} {p['top1']:7.4f} {p['top5']:7.4f} {p['k_mse']:10.3e} "
              f"{p['logit_err']:10.3e} {rate:6.3f}")
    rb = {m: report['per_task'][m].get('repobench-p', {}).get('top1') for m in methods}
    print("\nrepobench-p (code sentinel) top1: "
          + " ".join(f"{m}={v:.4f}" for m, v in sorted(rb.items()) if v is not None))
    print(f"[score] wrote {OUT_JSON} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
