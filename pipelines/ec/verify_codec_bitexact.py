#!/usr/bin/env python3
"""G1 gate: the press-side EC roundtrip must reconstruct what the real paged
rANS codec decodes.

For each tested (layer, head) and selection row:
  1. Encode K through PageCodecRANS (single-rung ladder from the bundle's
     delta + frozen model; CPU, validates the actual bitstream format) and
     decode it back.
  2. Run SnappedDeadzoneECCompressor.roundtrip (what JointQKPress executes).
  3. Compare in the integer index domain.

The codec rotates in numpy float64; the press path uses torch fp32. Values
that land exactly on a deadzone-bin boundary can therefore flip by one bin at
~1e-6 frequency. Gate: >= 99.99% exact index equality, every mismatch within
±1 bin, and reconstruction max-abs-diff <= the local delta.

    python pipelines/ec/verify_codec_bitexact.py \
        --bundle artifacts/ec/llama31_8b/ec_bundle__r_sym__b1.95__dz0.5__compact8train18.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import _bootstrap  # noqa: E402,F401

sys.path.insert(0, str(REPO / "entropy_coding"))

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from kvq.compression.ec_roundtrip import (  # noqa: E402
    SnappedDeadzoneECCompressor, bundle_model_to_dict, dz_round,
)
from kvq_codec import PageCodecRANS  # noqa: E402

RUN_ID = "ec_calib_compact8_train_llama31_8b"
RAW_ROOT = REPO / "artifacts/calibration" / RUN_ID / "01_raw"
ROLES = REPO / "artifacts/calibration_splits/ec_compact8_train_26/roles.json"


def raw_path(config: str, row_index: int) -> Path:
    name = f"longbench__{config}__row{int(row_index):05d}__train.pt"
    hits = sorted(RAW_ROOT.glob(f"shard_*/{name}"))
    if len(hits) != 1:
        raise FileNotFoundError(name)
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--rows", type=int, default=2, help="selection rows to test")
    ap.add_argument("--heads", default="1:0,5:3,16:7,31:4",
                    help="layer:head pairs to test")
    ap.add_argument("--ptok", type=int, default=64)
    args = ap.parse_args()

    blob = torch.load(args.bundle, map_location="cpu", weights_only=False)
    model = bundle_model_to_dict(blob)
    dz = float(blob["dz"])
    sel = json.loads(ROLES.read_text())["selection"][: args.rows]
    pairs = [tuple(int(x) for x in p.split(":")) for p in args.heads.split(",")]

    worst_eq, n_total, n_mismatch, max_bin_diff, max_recon = 1.0, 0, 0, 0, 0.0
    for (l, h) in pairs:
        delta = blob["delta"][l, h]
        press = SnappedDeadzoneECCompressor(
            forward_map=blob["forward"][l, h], inverse_map=blob["inverse"][l, h],
            mu=blob["mu"][l, h], delta=delta, dz=dz,
            support_vals=blob["support_vals"][l, h],
            support_lens=blob["support_lens"][l, h])
        codec = PageCodecRANS(
            fwd=blob["forward"][l, h].numpy(), inv=blob["inverse"][l, h].numpy(),
            mu=blob["mu"][l, h].numpy(),
            rungs=[(delta, model[(l, h)])],
            page_bits=1 << 30, P_tok=args.ptok, dz=dz, lanes=1)
        for task, idx in sel:
            art = torch.load(raw_path(task, int(idx)), map_location="cpu",
                             mmap=True, weights_only=False)
            T = int(art["prompt_length"])
            k = art["k_post"][l, h, :T].float()

            buf = codec.encode(k.numpy())
            k_codec = torch.from_numpy(np.asarray(codec.decode(buf))).float()
            k_press = press.roundtrip(k)

            # index-domain comparison (invert dequant exactly: idx = snapped ints)
            d_row = delta.clamp_min(1e-12).unsqueeze(0)
            idx_codec = dz_round((k_codec - press.mu) @ press.fwd, d_row, dz).round()
            idx_press = dz_round((k_press - press.mu) @ press.fwd, d_row, dz).round()
            same = (idx_codec == idx_press)
            eq = float(same.float().mean())
            diff = (idx_codec - idx_press).abs()
            n_total += same.numel()
            n_mismatch += int((~same).sum())
            worst_eq = min(worst_eq, eq)
            max_bin_diff = max(max_bin_diff, int(diff.max()))
            rec = float((k_codec - k_press).abs().max())
            max_recon = max(max_recon, rec)
            print(f"  (l={l},h={h}) {task} T={T}: idx_eq={eq:.6f} "
                  f"max_bin_diff={int(diff.max())} recon_maxabs={rec:.2e} "
                  f"bytes={len(buf)} ({len(buf)*8/(T*press.d):.3f} b/c incl. overhead)")

    ok = worst_eq >= 0.9999 and max_bin_diff <= 1
    print(f"\nG1 {'PASS' if ok else 'FAIL'}: worst idx_eq={worst_eq:.6f} "
          f"mismatches={n_mismatch}/{n_total} max_bin_diff={max_bin_diff} "
          f"recon_maxabs={max_recon:.2e}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
