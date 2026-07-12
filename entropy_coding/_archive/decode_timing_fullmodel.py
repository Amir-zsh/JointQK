#!/usr/bin/env python3
"""!!! SUPERSEDED / DO NOT USE FOR THE BANDWIDTH QUESTION -- see bw_fullmodel.py !!!

This script measures the UN-FUSED decode path: it applies the inverse rotation as
a dense fp64 (T,d)x(d,d) matmul per key and materializes a full-precision K back
to HBM. That is compute-bound, not bandwidth-bound, so BF16 (whose "decode" is a
no-op clone) trivially wins here -- which is BACKWARDS from reality. In a real
fused attention-decode kernel the compressed KV is read from HBM (few bytes),
reconstructed in SRAM, and the inverse rotation is fused into the query side
(never an O(T.d^2) per-key cost, never a full-precision HBM write-back). Under
that correct framing (bw_fullmodel.py / bw_clean.py) the 2-bit fixed-width methods
are ~7x FASTER than BF16 at long context. Kept only as a record of the un-fused
worst case; do not cite its numbers for decode throughput.

Full-model (36 layers x 8 kv-heads = 288 heads) DECODE-ONLY timing, real data,
same real per-method decode math as decode_timing_unified.py, aggregated to
model scale instead of a single head.

Why: decode_timing_unified.py measures ONE (layer, head) pair -- at that scale
BF16 beats every compressed method, because flash-attention's own compute floor
dominates over such a small read, so the "read fewer bytes -> faster" story
never gets a chance to show up (see chat 2026-07-08). The ORIGINAL report's
claim that INT2 is ~8x faster than BF16 was computed over the FULL model's
KV cache (36x8 heads read at once) -- a ~288x bigger read per call, which is
the regime where HBM bandwidth, not per-call overhead, should dominate. This
script re-does that comparison honestly: same real per-method decode paths
(not bw_clean.py's synthetic random-buffer read), aggregated to the same
full-model scale the original claim was actually about.

DECODE ONLY, no attend -- aggregating flash-attention across 288 (layer, head)
pairs at once isn't a well-defined single kernel call the way a KV-cache read
is, and conflating it back in would reintroduce the same per-head compute
floor this script exists to isolate away from.

Methodology to avoid OOM at T=100000 x 288 heads: per-head loop (not one giant
stacked batch tensor -- a naive stack blows up intermediate quantization
tensors to 50-100+ GB). rANS and Exp-Golomb use their NATIVE grid-batched
paths (BatchRANSDecoder.decode_grid / eg_decode_gpu_batch) since those are
literally built for one-shot full-grid decode. The whole per-head loop for
each method is bracketed by a single pair of CUDA events (not per-head
events) so per-event overhead doesn't skew the aggregate.
"""
import argparse
import torch

import run_pca_ec_deadzone as base
from kvq_codec import build_codecs_from_ladder_rans_cuda, load_ext, BatchRANSEncoder, BatchRANSDecoder
from kvq.compression.lloyd_max import Stage1MSECompressor
from group_vq_codec import GroupVQCompressor
from oscar_codec import build_oscar_rotation, OSCARCompressor
from expgolomb_codec import (eg_encode_page_grid, choose_k_per_coord, combine_encodings,
                             eg_decode_gpu_batch, zigzag_decode)

dev = torch.device("cuda")
CALIB_IDX = [0, 5, 6]
EVAL_IDX = 4
P_PAGE = 64
N_LANES = 16
DZ = 0.375


def timeit_ev(f, reps=8, warmup=2):
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
    ap.add_argument("--Ts", type=int, nargs="+", default=[65536, 100000])
    ap.add_argument("--vq-codebook", type=str, default="group_vq_b2_calib056.pt")
    args = ap.parse_args()

    root = base.data_root(); man = base.load_manifest(root)
    sq, sk, km, kc, meta = base.calib_moments(root, man, CALIB_IDX)
    L, Hkv, d = meta["n_layers"], meta["n_kv_heads"], meta["d_head"]
    qu = base.build_qpca_basis(sq, sk); qu["sigma_k"], qu["sigma_q"] = sk, sq
    qc = base.build_qpca_basis(sq, kc); qc["sigma_k"] = sk
    F, inv = qc["forward"], qc["inverse"]
    fc = base._codes_for_idx(root, man, CALIB_IDX, F, km, L, Hkv, d)

    art = torch.load(root / man["examples"][EVAL_IDX]["file"], map_location="cpu", weights_only=False)
    T0 = int(art["prompt_length"])
    heads = [(l, h) for l in range(L) for h in range(Hkv)]
    print(f"Full model: {len(heads)} heads (L={L}, Hkv={Hkv}), Ts={args.Ts}\n", flush=True)

    print("Building shared per-method state (rotations, codebooks, rANS/EG ladders)...", flush=True)
    vq_payload = torch.load(args.vq_codebook, map_location="cpu", weights_only=False)
    R_oscar_all = build_oscar_rotation(sq)                      # (L,Hkv,d,d), CPU double
    tq = Stage1MSECompressor(head_dim=d, bits=2, seed=20260505, device=dev)  # shared Pi, all heads

    _, delta0, model0 = base.build_qpca_ec(qc, qu, km, 2, L, Hkv, fc, root, CALIB_IDX,
                                           dz=DZ, match_rate=False, uniform_step=True)
    ladder = [(1.0, delta0, model0)]
    ext = load_ext()
    cod = build_codecs_from_ladder_rans_cuda(F, inv, km, ladder, L, Hkv, 2 * d * 64, 64, DZ,
                                             lanes=N_LANES, ext=ext, device="cuda")
    rans_enc = BatchRANSEncoder(cod); rans_dec = BatchRANSDecoder(cod)
    eg_ext = load_ext(source="expgolomb_decode.cu", name="expgolomb_decode")

    from kvq.compression.per_coord import unit_gaussian_centroids
    cent2 = unit_gaussian_centroids(2).to(dev)   # (4,) shared LM centroids for the uniform 2-bit alloc

    results = {}
    for T in args.Ts:
        print(f"--- T={T} ---", flush=True)
        kg = {(l, h): tile_to(art["k_post"][l, h, :T0, :].float(), T).to(dev) for (l, h) in heads}

        # ---------------- BF16: real HBM touch (clone every head), no compute ----------------
        kg_bf16 = {lh: k.half() for lh, k in kg.items()}
        def bf16_decode():
            return {lh: k.clone() for lh, k in kg_bf16.items()}
        ms_bf16 = timeit_ev(bf16_decode, reps=5, warmup=2)
        print(f"  BF16: {ms_bf16:.3f}ms", flush=True)
        del kg_bf16   # not used by any other method -- was leaking ~5-15GB across the T iteration
        torch.cuda.empty_cache()

        # ---------------- INT2: per-head uniform 2-bit affine (4-centroid, no giant diffs tensor) ----------------
        int2_state = {}
        for (l, h) in heads:
            std_h = qc["std"][l, h].to(dev).double()
            cb_h = cent2.double().view(1, 4) * std_h.view(d, 1)      # (d,4)
            r = kg[(l, h)].double() @ F[l, h].double().to(dev)        # (T,d)
            # nearest of 4 centroids per coord without materializing a (T,d,4) tensor
            best_idx = torch.zeros(T, d, dtype=torch.uint8, device=dev)
            best_dist = torch.full((T, d), float("inf"), dtype=torch.float64, device=dev)
            for ci in range(4):
                dist = (r - cb_h[:, ci].unsqueeze(0)).abs()
                upd = dist < best_dist
                best_idx = torch.where(upd, torch.full_like(best_idx, ci), best_idx)
                best_dist = torch.where(upd, dist, best_dist)
            int2_state[(l, h)] = (best_idx, cb_h, inv[l, h].double().to(dev))

        def int2_decode():
            out = {}
            for lh, (idx, cb_h, inv_h) in int2_state.items():
                vals = torch.gather(cb_h.unsqueeze(0).expand(T, -1, -1), 2, idx.long().unsqueeze(-1)).squeeze(-1)
                out[lh] = vals @ inv_h
            return out
        ms_int2 = timeit_ev(int2_decode, reps=5, warmup=2)
        print(f"  INT2: {ms_int2:.3f}ms", flush=True)
        del int2_state
        torch.cuda.empty_cache()

        # ---------------- TurboQuant: shared Pi across heads, encode once per head, chunked decode ----------------
        tq_state = {lh: tq.compress(kg[lh].unsqueeze(0).unsqueeze(0)) for lh in heads}
        def turbo_decode():
            return {lh: tq.decompress(st) for lh, st in tq_state.items()}
        ms_turbo = timeit_ev(turbo_decode, reps=5, warmup=2)
        print(f"  TurboQuant: {ms_turbo:.3f}ms", flush=True)
        del tq_state
        torch.cuda.empty_cache()

        # ---------------- OSCAR: per-head rotation + dynamic quant, sink/recent protected ----------------
        oscar_comps = {(l, h): OSCARCompressor(R_oscar_all[l, h].to(dev).float(), clip_ratio=0.96,
                                               sink=64, recent=256) for (l, h) in heads}
        oscar_state = {lh: oscar_comps[lh].encode(kg[lh]) for lh in heads}
        def oscar_decode():
            return {lh: oscar_comps[lh].decode(st) for lh, st in oscar_state.items()}
        ms_oscar = timeit_ev(oscar_decode, reps=5, warmup=2)
        print(f"  OSCAR: {ms_oscar:.3f}ms", flush=True)
        del oscar_comps, oscar_state
        torch.cuda.empty_cache()

        # ---------------- VQ: per-head group codebook gather (already the unfused/slow path) ----------------
        # encode_idx's per-group cdist materializes a (T, K) transient (up to ~2GB at T=100000,
        # K=4096) 21+ times per head x 288 heads -- periodic empty_cache() during this loop keeps
        # the allocator from fragmenting/bloating across ~6000+ such large transient allocations
        # (this was the actual OOM source, not the persistent per-head state).
        vq_comps = {}
        vq_state = {}
        for i, (l, h) in enumerate(heads):
            comp = GroupVQCompressor(F[l, h].to(dev), inv[l, h].to(dev), km[l, h].to(dev),
                                     [c.to(dev) for c in vq_payload["codebooks"][(l, h)]],
                                     vq_payload["bounds"])
            vq_comps[(l, h)] = comp
            vq_state[(l, h)] = comp.encode_idx(kg[(l, h)])
            if i % 32 == 0:
                torch.cuda.empty_cache()
        def vq_decode():
            return {lh: vq_comps[lh].decode_idx(st, dtype=torch.float32) for lh, st in vq_state.items()}
        ms_vq = timeit_ev(vq_decode, reps=3, warmup=1)
        print(f"  VQ: {ms_vq:.3f}ms", flush=True)
        del vq_comps, vq_state
        torch.cuda.empty_cache()
        print(f"  [mem before rANS] allocated={torch.cuda.memory_allocated()/1e9:.2f}GB "
              f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)

        # ---------------- rANS: NATIVE full-grid one-shot decode ----------------
        # encode_grid's PHASE 1 (_prepare_head) accumulates per-head job structures for
        # EVERY head in k_grid before PHASE 2 batches them -- at T=65536+ across 288 heads
        # this blew the allocator (39GB) even with 10GB free beforehand. Not a leak from
        # this script (memory was verified low right before the call) -- encode_grid's own
        # per-call accumulation scales with T x n_heads. Chunking the untimed ENCODE call
        # into head-batches bounds that peak without touching kvq_codec.py (production code)
        # or affecting the TIMED decode_grid call below (that still runs on the full grid).
        rans_bufs = {}
        chunk = 48
        for ci in range(0, len(heads), chunk):
            sub = {lh: kg[lh] for lh in heads[ci:ci + chunk]}
            rans_bufs.update(rans_enc.encode_grid(sub))
            torch.cuda.empty_cache()

        def rans_decode():
            return rans_dec.decode_grid(rans_bufs)
        ms_rans = timeit_ev(rans_decode, reps=5, warmup=2)
        print(f"  rANS: {ms_rans:.3f}ms", flush=True)
        del rans_bufs
        torch.cuda.empty_cache()

        # ---------------- Exp-Golomb: NATIVE full-grid one-shot decode ----------------
        eg_encs = []
        for (l, h) in heads:
            r = (kg[(l, h)].double() - km[l, h].double().to(dev)) @ F[l, h].double().to(dev)
            idxg = base._dz_round(r, delta0[l, h].double().to(dev).clamp_min(1e-12), DZ).long().cpu().numpy()
            kpc = choose_k_per_coord(idxg)
            eg_encs.append(eg_encode_page_grid(idxg, P_PAGE, N_LANES, kpc))
        eg_merged = combine_encodings(eg_encs)
        def eg_decode():
            return eg_decode_gpu_batch(eg_merged, eg_ext, device="cuda")
        ms_eg = timeit_ev(eg_decode, reps=5, warmup=2)
        print(f"  Exp-Golomb: {ms_eg:.3f}ms", flush=True)
        del eg_encs, eg_merged

        results[T] = dict(BF16=ms_bf16, INT2=ms_int2, TurboQuant=ms_turbo, OSCAR=ms_oscar,
                          VQ=ms_vq, ExpGolomb=ms_eg, rANS=ms_rans)
        del kg, kg_bf16
        torch.cuda.empty_cache()

    print("\nFull-model (288-head) DECODE-ONLY time, ms, real data:")
    print(f"{'method':>12} " + " ".join(f"{T:>10}" for T in args.Ts))
    for m in ["BF16", "INT2", "TurboQuant", "OSCAR", "VQ", "ExpGolomb", "rANS"]:
        print(f"{m:>12} " + " ".join(f"{results[T][m]:>10.3f}" for T in args.Ts))
    print("\nvs BF16 (>1x = faster than BF16):")
    for m in ["INT2", "TurboQuant", "OSCAR", "VQ", "ExpGolomb", "rANS"]:
        print(f"  {m:>12}: " + " ".join(f"{results[T]['BF16']/results[T][m]:>9.2f}x" for T in args.Ts))
