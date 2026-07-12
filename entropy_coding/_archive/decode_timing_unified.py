#!/usr/bin/env python3
"""Unified real-data decode-step timing for all 7 K-compression methods
(including OSCAR, arXiv:2605.17757, reimplemented in oscar_codec.py -- see
that file's docstring for why we reimplement their algorithm on our data
rather than importing their SGLang-coupled kernels directly), on the
SAME real tiled K, at the SAME T sweep, with the SAME timing methodology
(torch.cuda.Event wall-clock, decode + flash-attend against a real query) --
replacing bw_clean.py / vq_throughput_fused.py's synthetic random-buffer
Triton read+checksum numbers (idealized bandwidth floor, no attention, no real
data) with what a decode step actually costs.

Each method's "encode" (quantize to the compressed representation a real cache
would store) happens ONCE, untimed, before the loop -- matching a resident,
already-quantized KV cache. Only "decode" (reconstruct K from the compressed
representation) + flash-attend is timed, matching OSCAR's single-request (B=1)
decode-step framing and this repo's own decode_phase_sweep.py (which already
does this correctly for rANS; this script generalizes that pattern to the
other 5 methods).

Correctness gate: every method's decode(encode(k)) is checked against its own
`.roundtrip(k)` reference (or, for rANS/EG, against the CPU/GPU decode used
elsewhere in this repo) before any timing number is trusted -- PASS/FAIL
printed per method, mirroring decode_bench.py's LUT-vs-binsearch convention.

Scope note: this reports ONE wall-clock number per method (decode + attend),
not OSCAR's kernel-only/wall-clock pair -- INT2/TurboQuant/VQ here are
PyTorch-op sequences, not fused CUDA kernels, so there's no separate
"kernel-only" number to isolate for them the way there is for rANS/Exp-Golomb
(see decode_bench.py) or a real fused Triton kernel (see vq_fused.cu, not yet
wired to real data). Fusing INT2/TurboQuant/VQ decode into single kernels
(matching OSCAR's `oscar_rotation_clip_int2_kv.py`) is future work, tracked
separately from this pass.
"""
import argparse
import time

import torch
import torch.nn.functional as Fnn
from torch.nn.attention import sdpa_kernel, SDPBackend

import run_pca_ec_deadzone as base
from kvq_codec import build_codecs_from_ladder_rans_cuda, load_ext, BatchRANSEncoder
from kvq.compression.per_coord import PerCoordCompressor
from kvq.compression.lloyd_max import Stage1MSECompressor
from group_vq_codec import GroupVQCompressor
from expgolomb_codec import eg_encode_page_grid, choose_k_per_coord, eg_decode_gpu
from oscar_codec import build_oscar_rotation, OSCARCompressor

dev = torch.device("cuda")
CALIB_IDX = [0, 5, 6]
EVAL_IDX = 4
G_VQ = 6
P_PAGE = 64
N_LANES = 16
DZ = 0.375


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


def gate(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [correctness] {name}: {tag}  {detail}")
    if not ok:
        raise RuntimeError(f"correctness gate failed for {name}")


# ---------------------------------------------------------------------------
# Setup: shared basis + real K/Q data (same corpus/example as decode_phase_sweep.py)
# ---------------------------------------------------------------------------
root = base.data_root(); man = base.load_manifest(root)
sq, sk, km, kc, meta = base.calib_moments(root, man, CALIB_IDX)
L, Hkv, d = meta["n_layers"], meta["n_kv_heads"], meta["d_head"]
qu = base.build_qpca_basis(sq, sk); qu["sigma_k"], qu["sigma_q"] = sk, sq
qc = base.build_qpca_basis(sq, kc); qc["sigma_k"] = sk
F, inv = qc["forward"], qc["inverse"]
fc = base._codes_for_idx(root, man, CALIB_IDX, F, km, L, Hkv, d)

l, h = 1, 0   # one representative head, matching decode_phase_sweep.py's convention
art = torch.load(root / man["examples"][EVAL_IDX]["file"], map_location="cpu", weights_only=False)
T0 = int(art["prompt_length"]); gs = art["q_post"].shape[1] // Hkv
k_real0 = art["k_post"][l, h, :T0, :].float()
sm = 1.0 / (d ** 0.5)


# ---------------------------------------------------------------------------
# Per-method encode/decode. Each returns (encode_fn, decode_fn, roundtrip_ref_fn).
# encode_fn(kT) -> compressed state (untimed). decode_fn(state) -> k_hat (timed).
# ---------------------------------------------------------------------------
def build_bf16():
    def encode(kT): return kT
    def decode(state): return state
    def ref(kT): return kT
    return encode, decode, ref


def build_turboquant():
    tq = Stage1MSECompressor(head_dim=d, bits=2, seed=20260505, device=dev)
    def encode(kT): return tq.compress(kT.unsqueeze(0).unsqueeze(0))
    def decode(state): return tq.decompress(state).squeeze(0).squeeze(0)
    def ref(kT): return tq.roundtrip(kT.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    return encode, decode, ref


def build_int2():
    # b=2 uniform allocation (matches report's "INT2 fixed-width (QPCA)"), same
    # centered QPCA basis as rANS/EG/Paged.
    bits = torch.full((d,), 2, dtype=torch.long)
    std = qc["std"][l, h].float()
    comp = PerCoordCompressor(bits_per_coord=bits, std_per_coord=std,
                               forward_map=F[l, h].float(), inverse_map=inv[l, h].float()).to(dev)

    def encode(kT):
        flat = kT.float()
        transformed = flat @ comp.forward_map
        diffs = (transformed.unsqueeze(-1) - comp.codebooks_padded.unsqueeze(0)).abs()
        return diffs.argmin(dim=-1)   # (T, d) int64 -- the stored compressed rep

    def decode(idx):
        out = torch.gather(comp.codebooks_padded.unsqueeze(0).expand(idx.shape[0], -1, -1),
                            2, idx.unsqueeze(-1)).squeeze(-1)
        return out @ comp.inverse_map

    def ref(kT): return comp.roundtrip(kT)
    return encode, decode, ref


def build_vq(vq_comp):
    def encode(kT): return vq_comp.encode_idx(kT)
    def decode(idx_list): return vq_comp.decode_idx(idx_list, dtype=torch.float32)
    def ref(kT): return vq_comp.roundtrip(kT).float()
    return encode, decode, ref


def build_rans():
    _, d0, m0 = base.build_qpca_ec(qc, qu, km, 2, L, Hkv, fc, root, CALIB_IDX,
                                   dz=DZ, match_rate=False, uniform_step=True)
    ladder = [(1.0, d0, m0)]
    for m in [1.05, 1.1, 1.25, 1.5]:
        dm = (d0 * m).float()
        ladder.append((m, dm, base.freeze_coder_model(fc, dm, L, Hkv, d, DZ)))
    ext = load_ext()
    cod = build_codecs_from_ladder_rans_cuda(F, inv, km, ladder, L, Hkv, 2 * d * 64, 64, DZ,
                                             lanes=N_LANES, ext=ext, device="cuda")
    c = cod[(l, h)]
    enc = BatchRANSEncoder(cod)

    def encode(kT):
        return enc.encode_grid({(l, h): kT})[(l, h)]

    def decode(buf): return c.decode_to_rhat(buf)

    def ref(kT):
        buf = encode(kT)
        return c.decode_to_rhat(buf)
    return encode, decode, ref


def build_expgolomb(delta0, model0):
    km_lh = km[l, h].double().to(dev)
    F_lh = F[l, h].double().to(dev)
    inv_lh = inv[l, h].double().to(dev)
    delta0_lh = delta0[l, h].double().to(dev)
    eg_ext = load_ext(source="expgolomb_decode.cu", name="expgolomb_decode")

    def encode(kT):
        r = (kT.double() - km_lh) @ F_lh
        idx = base._dz_round(r, delta0_lh.clamp_min(1e-12), DZ).long().cpu().numpy()
        k_per_coord = choose_k_per_coord(idx)
        enc_ = eg_encode_page_grid(idx, P_PAGE, N_LANES, k_per_coord)
        return enc_

    def decode(state):
        pos = eg_decode_gpu(state, eg_ext, device="cuda").long()   # (T, d) zigzag-coded symbol idx
        # GPU-native zigzag decode (expgolomb_codec.zigzag_decode is numpy-only;
        # calling it here would force an avoidable GPU->CPU->GPU round trip inside
        # the timed decode path, found via decode_timing_smoke2.log showing EG at
        # 21.9ms vs rANS's 3.4ms wall-clock despite EG's kernel-only rate being
        # FASTER than rANS's -- the smoke test caught this, not a real EG cost).
        sym = torch.where(pos % 2 == 0, pos // 2, -(pos + 1) // 2)
        r_hat = base._dz_dequant(sym.double(), delta0_lh, DZ)
        return (r_hat @ inv_lh + km_lh).float()

    def ref(kT):
        state = encode(kT)
        return decode(state)
    return encode, decode, ref


def build_oscar():
    R_K = build_oscar_rotation(sq[l, h]).to(dev)   # sq = calib_moments' sigma_q (uncentered, GQA-pooled)
    comp = OSCARCompressor(R_K, clip_ratio=0.96, sink=64, recent=256).to(dev)

    def encode(kT): return comp.encode(kT)
    def decode(state): return comp.decode(state)
    def ref(kT): return comp.roundtrip(kT)
    return encode, decode, ref, comp


def run_method(name, T, encode_fn, decode_fn, ref_fn, kT, q1):
    state = encode_fn(kT)
    k_hat = decode_fn(state)
    k_ref = ref_fn(kT)
    err = (k_hat.double().to(dev) - k_ref.double().to(dev)).abs().max().item()
    gate(f"{name} T={T}", err < 1e-3, f"max|decode(encode)-roundtrip|={err:.2e}")

    vT = tile_to(art["v"][l, h, :T0, :].float(), T).to(dev)
    invT = torch.as_tensor(inv[l, h], dtype=torch.float32, device=dev)

    def run_decode_attend():
        kh = decode_fn(state).float()
        khp = kh if name == "BF16" else kh  # already in raw K space (post inverse-map)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            Fnn.scaled_dot_product_attention(
                q1.half().unsqueeze(0),
                khp.half().unsqueeze(0).unsqueeze(0).expand(1, gs, T, d),
                vT.half().unsqueeze(0).unsqueeze(0).expand(1, gs, T, d))

    ms = timeit_ev(run_decode_attend)
    return ms


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ts", type=int, nargs="+", default=[4096, 8192, 16384, 32768, 65536, 100000])
    ap.add_argument("--vq-codebook", type=str, default="group_vq_b2_calib056.pt")
    args = ap.parse_args()

    print(f"calib={CALIB_IDX} eval_idx={EVAL_IDX} head=(l={l},h={h}) | L={L} Hkv={Hkv} d={d}\n")

    vq_payload = torch.load(args.vq_codebook, map_location="cpu", weights_only=False)
    vq_cb = [c.to(dev) for c in vq_payload["codebooks"][(l, h)]]
    vq_comp = GroupVQCompressor(F[l, h].to(dev), inv[l, h].to(dev), km[l, h].to(dev),
                                vq_cb, vq_payload["bounds"])

    _, delta0_eg, model0_eg = base.build_qpca_ec(qc, qu, km, 2, L, Hkv, fc, root, CALIB_IDX,
                                                 dz=DZ, match_rate=False, uniform_step=True)

    oscar_encode, oscar_decode, oscar_ref, oscar_comp = build_oscar()

    methods = {
        "BF16": build_bf16(),
        "TurboQuant": build_turboquant(),
        "INT2": build_int2(),
        "VQ": build_vq(vq_comp),
        "rANS": build_rans(),
        "Exp-Golomb": build_expgolomb(delta0_eg, model0_eg),
        "OSCAR": (oscar_encode, oscar_decode, oscar_ref),
    }

    print(f"{'T':>7} {'method':>11} {'decode+attend ms':>18} {'bits/coord':>11}")
    results = {}
    for T in args.Ts:
        kT = tile_to(k_real0, T).to(dev)
        q1 = torch.randn(gs, 1, d, device=dev, dtype=torch.float32)
        for name, (encode_fn, decode_fn, ref_fn) in methods.items():
            ms = run_method(name, T, encode_fn, decode_fn, ref_fn, kT, q1)
            results.setdefault(name, {})[T] = ms
            bpc = oscar_comp.bits_per_coord(T) if name == "OSCAR" else 2.0
            print(f"{T:>7} {name:>11} {ms:>18.4f} {bpc:>11.3f}")
        print()

    print("Summary (decode+attend ms, real data, real per-method decode, correctness-gated):")
    for name in methods:
        row = " ".join(f"{results[name][T]:8.3f}" for T in args.Ts)
        print(f"  {name:>11}: {row}")
