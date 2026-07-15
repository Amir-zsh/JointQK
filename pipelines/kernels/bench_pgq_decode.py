"""pgq7 K3: decode-attention microbenchmark — pgq v1 kernel vs BF16 paths
vs the vendored OSCAR INT2 kernel (plan7/plan8 §4).

One decode step of one layer at Qwen3-8B geometry (8 KV heads x GQA group 4
= 32 q heads, d=128), bs=1, contexts {8k, 32k, 80k, 128k}. Buffers are
fabricated at the REAL pgq operating point: rung mix drawn from the
dctlmrw@2.0 held-out telemetry (pgq8_heldout_report.json), all interior
pages DCT, LM-style LUTs — byte volumes and code paths identical to a real
cell; contents random (timing is data-independent). Correctness is pinned
BEFORE timing by a real-compressor parity case (same gate as the unit
tests), mirroring OSCAR's check-then-time harness discipline.

OSCAR arm: their `decode_attention_fwd_grouped_quant_int2` loaded straight
from the vendored tree (2 module stubs), buffers fabricated per their own
_build_case shapes (INT2 crumbs (T, H, d/4) uint8 + fp32 scales/zeros
(T, H, 2) for K and V; note THEIR arm quantizes V too — ours keeps V fp16,
so their V-side reads ~4x less; reported per-arm bytes make this explicit).

Pre-registered bar (plan7 K3): >= 2.5x vs the BF16 dense path at 80k bs=1.

Usage: python -u pipelines/kernels/bench_pgq_decode.py --gpu 0
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import types
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import _bootstrap  # noqa: E402,F401

from kvq.compression.per_coord import unit_gaussian_centroids  # noqa: E402
from kvq.kernels.pgq_decode_attn import page_attention_v1  # noqa: E402
from kvq.kernels.pgq_pack import BLOCK  # noqa: E402

H_KV, GROUP, D, PTOK, HGP = 8, 4, 128, 64, 16
CONTEXTS = (8192, 32768, 81920, 131072)
PROFILES = [
    [0, 0, 0, 0], [2, 0, 0, 0], [2, 2, 0, 0], [3, 2, 0, 0],
    [3, 2, 2, 0], [4, 3, 2, 0], [4, 4, 3, 2], [6, 4, 4, 3],
]
RUNG_HIST = [29502, 895713, 4167124, 668206, 3904337, 2525971, 1000733,
             834798]                       # dctlmrw@2.0 heldout telemetry


def build_lut_bench(dev):
    lut = torch.zeros(91, device=dev)
    off = torch.zeros(7, dtype=torch.int32, device=dev)
    o = 0
    for w in (2, 3, 4):
        c = unit_gaussian_centroids(w).sort().values
        c[c.abs().argmin()] = 0.0
        lut[o:o + c.numel()] = c.to(dev)
        off[w] = o
        o += c.numel()
    lim = 31
    lut[o:o + 63] = (torch.arange(63, device=dev) - lim).float() * 0.28
    off[6] = o
    return lut, off


def fabricate(T, dev, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    P = T // PTOK
    bw = torch.tensor(PROFILES, dtype=torch.int32, device=dev)
    strides = (bw * BLOCK).sum(1) // 8
    probs = torch.tensor(RUNG_HIST, dtype=torch.float64)
    probs /= probs.sum()
    rung = torch.multinomial(probs.expand(H_KV * T, -1).float(), 1,
                             generator=g).reshape(H_KV, T).to(torch.uint8)
    row_len = strides.cpu()[rung.long()]
    row_ptr = torch.zeros(H_KV, T, dtype=torch.int32)
    row_ptr[:, 1:] = row_len.cumsum(1)[:, :-1].to(torch.int32)
    pay_off, total = [], 0
    for h in range(H_KV):
        pay_off.append(total)
        total += int(row_len[h].sum())
    payload = torch.randint(0, 256, (total + 1,), dtype=torch.uint8,
                            generator=g)
    # random payload bytes can index one past the width-6 table (code 63 of
    # a 63-level grid); a zero-padded LUT absorbs it without masking
    lut, lut_off = build_lut_bench(dev)
    lut_pad = torch.zeros(256 + 91, device=dev)
    lut_pad[:91] = lut

    kinds = torch.full((P,), 2, dtype=torch.int8)
    kinds[0] = 1
    kinds[-4:] = 1                                       # rw pages identity
    from kvq.compression.pgq8_dct import dct_matrix
    gm = {
        "H": H_KV, "T": T, "d": D, "ptok": PTOK, "hg": GROUP, "hgp": HGP,
        "payload": payload.to(dev),
        "pay_off": torch.tensor(pay_off, dtype=torch.int64, device=dev),
        "row_ptr": row_ptr.to(dev), "row_rung": rung.to(dev),
        "bw": bw, "lut": lut_pad.expand(H_KV, -1).contiguous(),
        "lut_off": lut_off,
        "sigma_id": (torch.rand(H_KV, PTOK, D, generator=g) * 0.6
                     + 0.7).to(dev),
        "sigma_dct": (torch.rand(H_KV, PTOK, D, generator=g) * 0.6
                      + 0.7).to(dev),
        "dct_t": dct_matrix(PTOK).to(dev),
        "pages": torch.arange(P, dtype=torch.int32, device=dev),
        "page_kind": kinds.to(dev),
        "qt": torch.randn(H_KV, HGP, D, generator=g).to(dev),
        "sm_scale": 1.0 / math.sqrt(D),
        "v": torch.randn(H_KV, T, D, generator=g).half().to(dev),
        "tier_idx": torch.empty(0, dtype=torch.int64, device=dev),
        "z_tier": torch.empty(H_KV, GROUP, 0, device=dev),
    }
    gm["qt"][:, GROUP:] = 0
    kb = (int(payload.numel()) + int(rung.numel() * 3 / 8)
          + H_KV * T * D * 2)                            # K packed + V fp16
    gm["bytes_step"] = kb
    return gm


def load_oscar():
    utils_stub = types.ModuleType("sglang.srt.utils")
    utils_stub.is_hip = lambda: False

    class _E:
        def __getattr__(self, k):
            class V:
                value = None

                def get(self):
                    return None
            return V()

    env_stub = types.ModuleType("sglang.srt.environ")
    env_stub.envs = _E()
    for name, mod in [("sglang", types.ModuleType("sglang")),
                      ("sglang.srt", types.ModuleType("sglang.srt")),
                      ("sglang.srt.utils", utils_stub),
                      ("sglang.srt.environ", env_stub)]:
        sys.modules.setdefault(name, mod)
    spec = importlib.util.spec_from_file_location(
        "oscar_decode_attention",
        REPO / "vendor/OSCAR/sglang-research/python/sglang/srt/layers/"
               "attention/triton_ops/decode_attention.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def oscar_case(T, dev, splits=16):
    B, Hq = 1, H_KV * GROUP
    k_buf = torch.randint(0, 256, (T + 8, H_KV, D // 4), dtype=torch.uint8,
                          device=dev)
    v_buf = torch.randint(0, 256, (T + 8, H_KV, D // 4), dtype=torch.uint8,
                          device=dev)
    k_sz = torch.rand(T + 8, H_KV, 2, device=dev) * 0.05
    v_sz = torch.rand(T + 8, H_KV, 2, device=dev) * 0.05
    q = torch.randn(B, Hq, D, dtype=torch.float16, device=dev)
    o = torch.empty_like(q)
    logits = torch.empty(B, Hq, splits, D, dtype=torch.float32, device=dev)
    lse = torch.empty(B, Hq, splits, dtype=torch.float32, device=dev)
    indptr = torch.tensor([0, T], dtype=torch.int32, device=dev)
    indices = torch.arange(T, dtype=torch.int64, device=dev)
    nsplits = torch.tensor([splits], dtype=torch.int32, device=dev)
    bytes_step = ((T * H_KV * D // 4) * 2 + T * H_KV * 2 * 4 * 2)
    return dict(q=q, k_buf=k_buf, v_buf=v_buf, k_sz=k_sz, v_sz=v_sz, o=o,
                logits=logits, lse=lse, indptr=indptr, indices=indices,
                nsplits=nsplits, splits=splits,
                sm=1.0 / math.sqrt(D), bytes_step=bytes_step)


def fabricate_segmented(gm, dev, seed=1):
    """Group fabricated rows by rung into the K2c' coalesced layout."""
    g = torch.Generator().manual_seed(seed)
    bw = gm["bw"].cpu()
    strides = (bw * BLOCK).sum(1) // 8
    R, H, T, ptok = bw.shape[0], gm["H"], gm["T"], gm["ptok"]
    rung = gm["row_rung"].cpu()
    pay, tok, pay_off, tok_off, seg_n = [], [], [], [], []
    op = ot = 0
    for r in range(R):
        for h in range(H):
            idx = torch.nonzero(rung[h] == r).squeeze(1).to(torch.int32)
            n, s = idx.numel(), int(strides[r])
            pay_off.append(op)
            tok_off.append(ot)
            seg_n.append(n)
            pay.append(torch.randint(0, 256, (n * s,), dtype=torch.uint8,
                                     generator=g))
            tok.append(idx)
            op += n * s
            ot += n
    from kvq.kernels.golden import planarize
    pay_pl, pl_off, off_pl = [], [], 0
    i = 0
    for r in range(R):
        ws = [int(w) for w in bw[r]]
        for h in range(H):
            n = seg_n[i]
            sbytes = int(strides[r])
            buf = pay[i].reshape(max(n, 0), sbytes) if sbytes else                 torch.zeros(n, 0, dtype=torch.uint8)
            pl = planarize(buf, ws)
            pl_off.append(off_pl)
            pay_pl.append(pl.reshape(-1))
            off_pl += int(pl.numel())
            i += 1
    gm = dict(gm)
    kd = torch.ones(T // ptok, dtype=torch.int8)
    kd[gm["pages"].long().cpu()] = gm["page_kind"].cpu()
    lut_off = gm["lut_off"].cpu()
    seg_w = [[int(w) for w in bw[r]] for r in range(R) for _ in range(H)]
    seg_lo = [[int(lut_off[w]) if w > 0 else 0 for w in row]
              for row in seg_w]
    seg_stride = [int(strides[r]) for r in range(R) for _ in range(H)]
    gm.update({
        "seg_pay": torch.cat(pay).to(dev),
        "seg_tok": torch.cat(tok).to(dev),
        "seg_pay_off": torch.tensor(pay_off, dtype=torch.int64,
                                    device=dev).reshape(R, H),
        "seg_tok_off": torch.tensor(tok_off, dtype=torch.int64,
                                    device=dev).reshape(R, H),
        "seg_n": torch.tensor(seg_n, dtype=torch.int32,
                              device=dev).reshape(R, H),
        "seg_stride": torch.tensor(seg_stride, dtype=torch.int32,
                                   device=dev).reshape(R, H),
        "seg_w": torch.tensor(seg_w, dtype=torch.int32,
                              device=dev).reshape(R, H, 4),
        "seg_lo": torch.tensor(seg_lo, dtype=torch.int32,
                               device=dev).reshape(R, H, 4),
        "kind_dense": kd.to(dev),
        "sig_cat": torch.stack([gm["sigma_id"], gm["sigma_dct"]],
                               dim=1).half().contiguous(),
        "lut16": gm["lut"].half().contiguous(),
        "seg_pay_pl": (torch.cat(pay_pl) if off_pl else
                       torch.zeros(0, dtype=torch.int32)).to(dev),
        "seg_pl_off": torch.tensor(pl_off, dtype=torch.int64,
                                   device=dev).reshape(R, H),
        "bw_host": bw, "strides_host": strides})
    return gm


def graphed(fn, warmup=3):
    """CUDA-graph capture (standard decode-serving practice; applied to
    every arm that tolerates capture, ours and baselines alike)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        fn()
    return lambda: gr.replay(), gr


def bench(fn, n=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1e3


def parity_gate(dev):
    """Real-compressor parity before any timing row (harness discipline)."""
    sys.path.insert(0, str(REPO / "tests"))
    from test_pgq_kernel import build_multi
    gm = build_multi(H=2, T=64 * 9 + 21, dct=True, rw=4)
    o = page_attention_v1(gm)
    err = float((o - gm["o_ref"]).norm() / gm["o_ref"].norm())
    assert err < 1e-2, f"parity gate failed: {err}"
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(dev)
    err = parity_gate(dev)
    print(f"[k3] parity gate: {err:.2e}", flush=True)

    oscar = None
    try:
        oscar = load_oscar()
        print("[k3] OSCAR kernel loaded", flush=True)
    except Exception as e:
        print(f"[k3] OSCAR arm unavailable: {e}", flush=True)

    from kvq.kernels.pgq_decode_attn import page_attention_v2
    rows = {}
    keep = []
    for T in CONTEXTS:
        gm = fabricate(T, dev)
        best = min(
            (bench(lambda p=p, w=w: page_attention_v1(gm, p, num_warps=w),
                   args.iters), p, w)
            for p in (4, 8, 16) for w in (4, 8))
        pgq_ms, pps, nw = best
        gms = fabricate_segmented(gm, dev)
        v2fn, v2g = graphed(lambda: page_attention_v2(gms, 16))
        keep.append((gms, v2g))
        v2_ms = bench(v2fn, args.iters)
        gms["vq"] = torch.randint(0, 256, (H_KV, T, D // 4),
                                  dtype=torch.uint8, device=dev)
        gms["vqs"] = torch.rand(H_KV, T, device=dev) * 0.05
        gms["vqz"] = torch.rand(H_KV, T, device=dev) * 0.02
        vqfn, vqg = graphed(lambda: page_attention_v2(gms, 16, v_int2=True))
        keep.append(vqg)
        vq_ms = bench(vqfn, args.iters)

        # pgq10 S5: fixed-rate group-VQ K (32 uint8 group indices/token ->
        # L1-resident codebook gather; no unpack/LUT/sigma). Random codes/cb
        # (values don't affect timing); B128 phase B; fp16-V and INT2-V tiers.
        gms["vq_codes"] = torch.randint(
            0, 256, (H_KV, T, D // 4), dtype=torch.uint8, device=dev)
        gms["vq_cb"] = (torch.randn(H_KV, D // 4, 256, 4, device=dev) * 0.05
                        ).half().contiguous()
        vkfn, vkg = graphed(lambda: page_attention_v2(
            gms, 16, phase_a="vqk", phase_b="kernel2"))
        keep.append(vkg)
        vqk_ms = bench(vkfn, args.iters)
        vkifn, vkig = graphed(lambda: page_attention_v2(
            gms, 16, phase_a="vqk", phase_b="kernel2", v_int2=True))
        keep.append(vkig)
        vqk_i2_ms = bench(vkifn, args.iters)
        gms["vq_cb64"] = gms["vq_cb"].view(torch.int64).squeeze(-1).contiguous()
        gms["qt_vq"] = gms["qt"]        # random qt: permutation timing-neutral
        vk64fn, vk64g = graphed(lambda: page_attention_v2(
            gms, 16, phase_a="vqk64", phase_b="kernel2", v_int2=True))
        keep.append(vk64g)
        vqk64_i2_ms = bench(vk64fn, args.iters)

        K16 = torch.randn(H_KV, T, D, device=dev).half()
        V16 = gm["v"]
        Q16 = gm["qt"][:, :GROUP].half()

        def dense():
            outs = []
            for h in range(H_KV):
                z = (Q16[h].float() @ K16[h].float().T) / math.sqrt(D)
                outs.append(torch.softmax(z, 1) @ V16[h].float())
            return outs

        dense_ms = bench(dense, max(5, args.iters // 3))

        qs = Q16.reshape(1, H_KV * GROUP, 1, D)
        ks = K16[:, None].expand(H_KV, GROUP, T, D).reshape(
            1, H_KV * GROUP, T, D)
        vs = V16[:, None].expand(H_KV, GROUP, T, D).reshape(
            1, H_KV * GROUP, T, D)

        def sdpa():
            return torch.nn.functional.scaled_dot_product_attention(
                qs, ks, vs)

        sdpa_fn, sdpa_g = graphed(sdpa)
        keep.append((qs, ks, vs, sdpa_g))
        sdpa_ms = bench(sdpa_fn, args.iters)

        row = {
            "pgq_v2_vint2_graph_ms": vq_ms,
            "pgq_vint2_MB": (gm["bytes_step"] - H_KV * T * D * 2
                             + H_KV * T * (D // 4 + 8)) / 1e6,
            "speedup_vint2_vs_dense": dense_ms / vq_ms,
            "pgq_v2_graph_ms": v2_ms,
            "pgq_v1_ms": pgq_ms, "pgq_cfg": {"pps": pps, "warps": nw},
            "dense_fp16_ms": dense_ms, "sdpa_fp16_graph_ms": sdpa_ms,
            "pgq_MB": gm["bytes_step"] / 1e6,
            "fp16_MB": 2 * H_KV * T * D * 2 / 1e6,
            "speedup_vs_dense": dense_ms / v2_ms,
            "speedup_vs_sdpa": sdpa_ms / v2_ms,
            "vqk_graph_ms": vqk_ms,
            "vqk_vint2_graph_ms": vqk_i2_ms,
            "vqk64_vint2_graph_ms": vqk64_i2_ms,
            "vqk_MB": (H_KV * T * (D // 4) + H_KV * T * D * 2) / 1e6,
            "vqk_vint2_MB": (H_KV * T * (D // 4) * 2 + H_KV * T * 8) / 1e6,
            "speedup_vqk_vs_dense": dense_ms / vqk_ms,
            "speedup_vqk_vs_sdpa": sdpa_ms / vqk_ms,
        }
        if oscar is not None:
            best_o = None
            for s in (8, 16, 32):
                c = oscar_case(T, dev, splits=s)
                fn = lambda c=c: oscar.decode_attention_fwd_grouped_quant_int2(
                    c["q"], c["k_buf"], c["v_buf"], c["k_sz"], c["v_sz"],
                    c["o"], c["indptr"], c["indices"], c["logits"],
                    c["lse"], c["nsplits"], c["splits"], c["sm"])
                ofn, og = graphed(fn)
                keep.append((c, og))
                ms = bench(ofn, args.iters)
                if best_o is None or ms < best_o[0]:
                    best_o = (ms, s, c["bytes_step"])
            row["oscar_int2_graph_ms"] = best_o[0]
            row["oscar_splits"] = best_o[1]
            row["oscar_MB"] = best_o[2] / 1e6
            row["pgq_vs_oscar"] = best_o[0] / v2_ms
            row["pgq_vint2_vs_oscar"] = best_o[0] / vq_ms
            row["vqk_vs_oscar"] = best_o[0] / vqk_ms
            row["vqk_vint2_vs_oscar"] = best_o[0] / vqk_i2_ms
            row["vqk64_vint2_vs_oscar"] = best_o[0] / vqk64_i2_ms
        rows[str(T)] = row
        print(f"[k3] T={T:6d}: v2g {v2_ms:7.3f} ms | vI2 {vq_ms:7.3f} | "
              f"vqk {vqk_ms:7.3f} | vqkI2 {vqk_i2_ms:7.3f} | "
              f"vqk64I2 {vqk64_i2_ms:7.3f} | "
              f"dense {dense_ms:7.3f} | sdpa {sdpa_ms:7.3f} | "
              f"oscar {row.get('oscar_int2_graph_ms', float('nan')):7.3f} | "
              f"x_dense {row['speedup_vs_dense']:.2f} "
              f"x_sdpa {row['speedup_vs_sdpa']:.2f} "
              f"x_oscar {row.get('pgq_vs_oscar', float('nan')):.2f} "
              f"x_oscar_vqkI2 {row.get('vqk_vint2_vs_oscar', float('nan')):.2f}",
              flush=True)
        del gm, K16, V16, ks, vs
        keep.clear()
        torch.cuda.empty_cache()

    out = {"geometry": {"H_kv": H_KV, "group": GROUP, "d": D, "ptok": PTOK,
                        "bs": 1},
           "note": ("one layer-step; pgq v2 = coalesced rung-segmented two-"
                    "phase kernel under CUDA graph (as are SDPA and OSCAR "
                    "arms; dense is a python loop, ungraphed); pgq K packed "
                    "@2.0 real-telemetry mix + V fp16; OSCAR = their INT2 "
                    "kernel, K AND V INT2 (reads ~4x less V by design); "
                    "bar >=2.5x vs dense fp16 at 80k on the primary arm"),
           "parity_gate_err": err, "rows": rows}
    dst = REPO / "artifacts/kernels/bench_pgq_decode.json"
    dst.parent.mkdir(exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print(f"[k3] wrote {dst}", flush=True)
    bar = rows["81920"]["speedup_vs_dense"]
    print(f"[k3] PRE-REGISTERED BAR (>=2.5x dense @80k bs=1): "
          f"{bar:.2f}x -> {'PASS' if bar >= 2.5 else 'MISS'}", flush=True)


if __name__ == "__main__":
    main()
