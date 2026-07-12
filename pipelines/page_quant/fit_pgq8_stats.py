"""pgq8 P0 stats fit: per-(layer, head, coefficient-row, coord) std of the
token-axis DCT rows, pooled over the fit pool. Writes a pgq8 bundle = the
frozen pgq4/pgq5 bundle + `dct_std` (L,H,ptok,d) fp16. Everything else
(basis, mu, profiles, alphas, LM cents, ladder) is carried byte-identical —
stats refit only, per plan8 §1.

Usage: python -u pipelines/page_quant/fit_pgq8_stats.py --model-tag llama31_8b --gpu 0
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO))

from pipelines.page_quant.probe_token_axis import MODELS, dct_matrix  # noqa: E402

PTOK = 64
NSINK = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", required=True, choices=list(MODELS))
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    spec = MODELS[args.model_tag]
    dev = torch.device(f"cuda:{args.gpu}")

    blob = torch.load(REPO / spec["bundle"], map_location="cpu", weights_only=False)
    L, H, d = blob["n_layers"], blob["n_kv_heads"], blob["head_dim"]
    F = blob["bases"]["qpca_unc"]["forward"].to(dev).float()
    MU = blob["mu"].to(dev).float()
    sel = {tuple(x) for x in map(tuple, blob["selection_rows"])}

    raws = sorted((REPO / spec["raw_root"]).glob("shard_*/*.pt"))
    rows, seen = [], set()
    for p in raws:
        m = re.match(r"longbench__(.+)__row(\d+)__", p.name)
        key = m and (m.group(1), int(m.group(2)))
        if key and key not in sel and key not in seen:
            seen.add(key)
            rows.append((key[0], key[1], p))
    print(f"[pgq8fit] {args.model_tag}: {len(rows)} fit rows", flush=True)

    Dm = dct_matrix(PTOK).to(dev)
    V = torch.zeros(L, H, PTOK, d, dtype=torch.float64, device=dev)
    n = torch.zeros(L, H, dtype=torch.float64, device=dev)
    t0 = time.time()
    for ri, (cfg, ridx, path) in enumerate(rows):
        art = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        T = int(art["prompt_length"])
        k_all = art["k_post"][:, :, :T].to(dev).float()
        for l in range(1, L):
            for h in range(H):
                r = ((k_all[l, h] - MU[l, h]) @ F[l, h])[NSINK:]
                P = r.shape[0] // PTOK
                if P == 0:
                    continue
                pg = r[: P * PTOK].reshape(P, PTOK, d)
                y = torch.einsum("st,ptd->psd", Dm, pg)
                V[l, h] += y.square().sum(0).double()
                n[l, h] += P
        del k_all
        print(f"[pgq8fit] row {ri+1}/{len(rows)} ({cfg},{ridx}) "
              f"{time.time()-t0:.0f}s", flush=True)

    dct_std = (V / n.clamp_min(1).unsqueeze(-1).unsqueeze(-1)).sqrt().half().cpu()
    # layer 0 is fp16-served (never quantized); fill with token code_std so the
    # tensor is well-formed everywhere
    cs = blob["stats"]["qpca_unc"]["code_std"]                 # (L,H,d)
    dct_std[0] = cs[0].unsqueeze(1).expand(H, PTOK, d).half()
    blob["dct_std"] = dct_std
    blob["pgq8_fit_rows"] = [(c, r) for c, r, _ in rows]
    blob["pgq8_source_bundle"] = spec["bundle"]
    out = REPO / "artifacts/page_quant2" / f"pgq8_bundle__{args.model_tag}.pt"
    torch.save(blob, out)
    import hashlib
    sha8 = hashlib.sha256(out.read_bytes()).hexdigest()[:8]
    print(f"[pgq8fit] wrote {out} sha8={sha8} pages={int(n.sum())}", flush=True)


if __name__ == "__main__":
    main()
