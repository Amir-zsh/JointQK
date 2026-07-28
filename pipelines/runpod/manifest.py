#!/usr/bin/env python3
"""sha256 manifest for the gitignored artifact payload.

The data artifacts (rotations, codebooks, prompt rows) do not travel with a
git clone — they are rsynced onto each pod. The manifest, which IS tracked in
git, is the contract: `verify` proves the sync landed intact before any GPU
time is spent on it.

    # on the source host, after changing the payload:
    python pipelines/runpod/manifest.py make

    # on a pod, after rsync (group = qwen3_8b | gptoss | llama | all):
    python pipelines/runpod/manifest.py verify --group qwen3_8b
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).resolve().parent / "artifact_manifest.json"

# Payload groups, repo-relative. Extend here when a new model joins the grid.
GROUPS: dict[str, list[str]] = {
    "qwen3_8b": [
        "artifacts/oscar_e2e/rotzoo/Qwen3-8B/seq20000_prompt83_group128/k_rotation_qqt_r_h_pbr.pt",
        "artifacts/oscar_e2e/rotzoo/Qwen3-8B/seq20000_prompt83_group128/v_rotation_sst_r_h_pbr.pt",
        "third_party/samuel_vq/codebooks/vqa_G4_strat_flat_ptn_gpqacc64k_fp8.pt",
        "third_party/samuel_vq/codebooks/vqv_G4_strided_gpqa_engine.pt",
        "artifacts/prompt_rows/gpqa_diamond_think_qwen.jsonl",
        "artifacts/prompt_rows/aime25_think_qwen.jsonl",
        "artifacts/prompt_rows_proto/math500_think32k_qwen.jsonl",
        "artifacts/prompt_rows_code/humaneval_qwen.jsonl",
        "artifacts/prompt_rows_code/lcb_v6_qwen.jsonl",
        "artifacts/prompt_rows_code/lcb_v6_sub256_qwen.jsonl",
        "artifacts/prompt_rows/niah_8192_qwen.jsonl",
        "artifacts/prompt_rows/niah_16384_qwen.jsonl",
        "artifacts/prompt_rows/niah_32768_qwen.jsonl",
        "artifacts/prompt_rows/niah_65536_qwen.jsonl",
        "artifacts/niah_corpus/essays.txt",
        "artifacts/niah_corpus/noise.txt",
    ],
    "gptoss": [
        "artifacts/oscar_gptoss20b/rotations_gpqa198/k_rotation_qqt_r_h_pbr.pt",
        "artifacts/oscar_gptoss20b/rotations_gpqa198/v_rotation_sst_r_h_pbr.pt",
        "artifacts/oscar_gptoss20b/rotations_gpqa198/layer_map.json",
        "artifacts/oscar_gptoss20b/vqa_gptoss20b_G4_strat_flat_ptn_gpqacc128k_fp8.pt",
        "artifacts/prompt_rows/niah_8192_gptoss.jsonl",
        "artifacts/prompt_rows/niah_16384_gptoss.jsonl",
        "artifacts/prompt_rows/niah_32768_gptoss.jsonl",
        "artifacts/prompt_rows/niah_65536_gptoss.jsonl",
        "artifacts/prompt_rows/niah_131072_gptoss.jsonl",
        "artifacts/prompt_rows/gpqa_diamond.csv",
    ],
    # Calibrated on the official openai MXFP4 checkpoint (task #15) — the
    # go-forward set for serving that checkpoint. Codebook is 128k-calibrated;
    # rotations are short-dump by design (long-concat rotations are a known
    # failure mode). NIAH rows come from the "gptoss" group.
    "gptoss_mxfp4": [
        "artifacts/oscar_gptoss20b_mxfp4/rotations_gpqa198/k_rotation_qqt_r_h_pbr.pt",
        "artifacts/oscar_gptoss20b_mxfp4/rotations_gpqa198/v_rotation_sst_r_h_pbr.pt",
        "artifacts/oscar_gptoss20b_mxfp4/rotations_gpqa198/layer_map.json",
        "artifacts/oscar_gptoss20b_mxfp4/vqa_gptoss20b_G4_strat_flat_ptn_gpqacc128k_fp8.pt",
    ],
    "llama": [
        "artifacts/oscar_llama31_8b/rotations_gpqa198/k_rotation_qqt_r_h_pbr.pt",
        "artifacts/oscar_llama31_8b/rotations_gpqa198/v_rotation_sst_r_h_pbr.pt",
        "artifacts/oscar_llama31_8b/vqa_llama31_8b_G4_strat_flat_ptn_gpqacc128k_fp8.pt",
        "artifacts/prompt_rows/gpqa_diamond_llama.jsonl",
        "artifacts/prompt_rows/math500_llama.jsonl",
        "artifacts/prompt_rows/aime25_llama.jsonl",
    ],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def make(args) -> int:
    out: dict[str, list[dict]] = {}
    missing = []
    for group, paths in GROUPS.items():
        rows = []
        for rel in paths:
            p = REPO / rel
            if not p.exists():
                missing.append(rel)
                continue
            rows.append({"path": rel, "bytes": p.stat().st_size, "sha256": sha256(p)})
            print(f"  {rows[-1]['sha256'][:16]}  {rel}")
        out[group] = rows
    MANIFEST.write_text(json.dumps({"groups": out}, indent=2) + "\n")
    print(f"wrote {MANIFEST} ({sum(len(v) for v in out.values())} files)")
    if missing:
        print("WARNING — listed in GROUPS but absent on this host (not manifested):")
        for m in missing:
            print(f"  {m}")
    return 0


def verify(args) -> int:
    data = json.loads(MANIFEST.read_text())["groups"]
    groups = list(data) if args.group == "all" else [args.group]
    bad = 0
    for g in groups:
        for row in data[g]:
            p = REPO / row["path"]
            if not p.exists():
                print(f"MISSING   {row['path']}")
                bad += 1
            elif p.stat().st_size != row["bytes"] or sha256(p) != row["sha256"]:
                print(f"MISMATCH  {row['path']}")
                bad += 1
            elif args.verbose:
                print(f"ok        {row['path']}")
    n = sum(len(data[g]) for g in groups)
    if bad:
        print(f"VERIFY FAILED — {bad}/{n} files missing or corrupt")
        return 1
    print(f"VERIFY OK — {n} files match ({', '.join(groups)})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("make")
    v = sub.add_parser("verify")
    v.add_argument("--group", default="all", choices=[*GROUPS, "all"])
    v.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    return make(args) if args.cmd == "make" else verify(args)


if __name__ == "__main__":
    sys.exit(main())
