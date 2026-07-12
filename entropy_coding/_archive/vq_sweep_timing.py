#!/usr/bin/env python3
"""VQ-only decode+attend timing across the G-sweep codebooks, same real-data,
correctness-gated, encode-once/decode-timed methodology as decode_timing_unified.py
(see that file's docstring), but scoped to VQ alone so we can sweep multiple
codebook files without re-running the other 6 methods each time.

Smaller G means more groups per head (each group is its own gather + argmin),
so decode cost is not free to shrink -- this measures the real tradeoff instead
of assuming it away.
"""
import argparse
import torch
import torch.nn.functional as Fnn
from torch.nn.attention import sdpa_kernel, SDPBackend

import run_pca_ec_deadzone as base
from group_vq_codec import GroupVQCompressor

dev = torch.device("cuda")
CALIB_IDX = [0, 5, 6]
EVAL_IDX = 4


def timeit_ev(f, reps=15, warmup=5):
    for _ in range(warmup):
        f()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(reps):
        f()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


def tile_to(x, T):
    reps = (T + x.shape[0] - 1) // x.shape[0]
    return x.repeat(reps, 1)[:T].contiguous()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--codebooks", type=str, nargs="+", required=True)
    ap.add_argument("--Ts", type=int, nargs="+", default=[4096, 16384, 65536, 100000])
    args = ap.parse_args()

    root = base.data_root(); man = base.load_manifest(root)
    _, _, _, _, meta = base.calib_moments(root, man, CALIB_IDX)
    L, Hkv, d = meta["n_layers"], meta["n_kv_heads"], meta["d_head"]
    l, h = 1, 0
    art = torch.load(root / man["examples"][EVAL_IDX]["file"], map_location="cpu", weights_only=False)
    T0 = int(art["prompt_length"]); gs = art["q_post"].shape[1] // Hkv
    k_real0 = art["k_post"][l, h, :T0, :].float()

    print(f"{'codebook':<34} {'T':>7} {'decode+attend ms':>18} {'n_groups':>9}")
    for path in args.codebooks:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cb = [c.to(dev) for c in payload["codebooks"][(l, h)]]
        comp = GroupVQCompressor(payload["forward"][l, h].to(dev), payload["inverse"][l, h].to(dev),
                                 payload["mean"][l, h].to(dev), cb, payload["bounds"])
        name = path.split("/")[-1]
        for T in args.Ts:
            kT = tile_to(k_real0, T).to(dev)
            q1 = torch.randn(gs, 1, d, device=dev, dtype=torch.float32)
            vT = tile_to(art["v"][l, h, :T0, :].float(), T).to(dev)

            state = comp.encode_idx(kT)
            k_hat = comp.decode_idx(state, dtype=torch.float32)
            k_ref = comp.roundtrip(kT).float()
            err = (k_hat - k_ref).abs().max().item()
            assert err < 1e-3, f"{name} T={T}: correctness FAIL err={err:.2e}"

            def run_decode_attend():
                kh = comp.decode_idx(state, dtype=torch.float32)
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                    Fnn.scaled_dot_product_attention(
                        q1.half().unsqueeze(0),
                        kh.half().unsqueeze(0).unsqueeze(0).expand(1, gs, T, d),
                        vT.half().unsqueeze(0).unsqueeze(0).expand(1, gs, T, d))

            ms = timeit_ev(run_decode_attend)
            print(f"{name:<34} {T:>7} {ms:>18.4f} {len(payload['bounds']):>9}", flush=True)
