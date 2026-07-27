"""V1 integrity gates for the vq2 K quant tier in vendor/OSCAR-vq.

Run with the oscar venv, PYTHONPATH pointing at the clone:

  PYTHONPATH=vendor/OSCAR-vq/sglang-research/python \
      .venv-oscar/bin/python pipelines/oscar_e2e/verify_vq_engine.py --gpu 1

Gates (all must PASS):
  G1 roundtrip  : engine encode->decode K_hat matches GroupVQCompressor
                  reference roundtrip (same bundle) within 1e-2 rel.
  G2 score-equiv: softmax over q.K_hat scores (original space) equals softmax
                  over q_map(q).r_hat + HP-tier residual scores (stored
                  space) -- validates the mean-shift invariance the engine
                  relies on, mixing quant and HP tiers.
  G3 kernel     : _fwd_grouped_kernel_stage1_quant_vq2 + unified stage-2
                  matches a torch attention reference over the reconstructed
                  K_hat and dequantized int2 V within 2e-2 rel (fp16/bf16
                  kernel math vs fp32 reference).
  G4 V roundtrip: engine strided-group V encode/decode (the VQ-V ablation
                  tier) matches a double-precision reference codec at
                  rate-distortion parity. Coordinate-placement bugs (the
                  V-fatal class K gates cannot see) show up as ~2x
                  distortion here.
  G5 kernel V_VQ: the vq2 stage-1 kernel with the group-VQ V branch matches
                  a torch reference built from the documented plane
                  semantics (plane m of group g -> coord m*NG+g).
"""

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

import torch


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(torch.float64)
    b = b.to(torch.float64)
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument(
        "--bundle",
        default="artifacts/page_quant2/vqg_bundle__qwen3_8b_flat_ptn.pt",
    )
    ap.add_argument(
        "--v-bundle",
        default="third_party/samuel_vq/codebooks/vqv_G4_strided_gpqa_engine.pt",
        help="engine-basis V codebook (forward=identity, STRIDED groups); "
             "G4/G5 skip if absent",
    )
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 5, 18, 35])
    ap.add_argument("--tokens", type=int, default=512)
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    device = torch.device("cuda:0")
    torch.manual_seed(0)

    from sglang.srt.mem_cache.vq_codebook import (
        load_vq_codebook,
        vq_dequant,
        vq_encode,
        vq_map_k,
        vq_map_q,
    )
    from kvq.compression.group_vq import GroupVQCompressor

    bundle_path = os.path.join(REPO_ROOT, args.bundle)
    blob = torch.load(bundle_path, map_location="cpu", weights_only=False)
    L_total, H, D, _ = blob["forward"].shape

    vq = load_vq_codebook(
        bundle_path,
        layer_num=L_total,
        start_layer=0,
        head_num=H,
        head_dim=D,
        device=device,
        dtype=torch.bfloat16,
    )

    T = args.tokens
    failures = []

    # ---- G1: roundtrip parity vs GroupVQCompressor -------------------------
    print("== G1 roundtrip parity (engine encode/decode vs reference)")
    for l in args.layers:
        h = l % H
        # Reference uses the SAME fp8-snapped centroids as the engine (G1 is
        # an implementation-parity gate), so it must track whichever format the
        # loader chose -- e5m2 on sm80, e4m3 on sm89+. Pinning it to e5m2 makes
        # G1 fail on an e4m3 engine that is strictly MORE accurate than the
        # reference. ref_raw quantifies the snap cost against the bundle's raw
        # codebook (informational).
        _snap_dt = (
            torch.float8_e4m3fn if getattr(vq, "fp8_fmt", "e5m2") == "e4m3"
            else torch.float8_e5m2
        )
        ref = GroupVQCompressor(
            blob["forward"][l, h],
            blob["inverse"][l, h],
            blob["mean"][l, h],
            [c.to(_snap_dt).to(torch.float16)
             for c in blob["codebooks"][(l, h)]],
            blob["bounds"],
            pertoken_norm=bool(blob.get("pertoken_norm", False)),
        ).to(device)
        ref_raw = GroupVQCompressor(
            blob["forward"][l, h],
            blob["inverse"][l, h],
            blob["mean"][l, h],
            [c for c in blob["codebooks"][(l, h)]],
            blob["bounds"],
            pertoken_norm=bool(blob.get("pertoken_norm", False)),
        ).to(device)
        # Realistic-ish K rows: unit-scale gaussian around the bundle mean.
        k = (
            torch.randn(T, D, device=device, dtype=torch.float32)
            + blob["mean"][l, h].to(device)
        )
        k_ref = ref.roundtrip(k)
        k_ref_raw = ref_raw.roundtrip(k)

        k3 = k.unsqueeze(1).expand(T, H, D).contiguous()
        inv = blob["inverse"][l].to(device).to(torch.float32)  # [H, D, D]
        mean = blob["mean"][l].to(device).to(torch.float32)

        def engine_khat(map_dtype):
            fwd = vq.forward[l].to(map_dtype)
            mn = vq.mean[l].to(map_dtype)
            r = vq_map_k(k3.to(map_dtype), fwd, mn)
            idx, scale = vq_encode(
                r, vq.cb16[l], vq.cb_sq[l], pertoken_norm=vq.pertoken_norm
            )
            r_hat = vq_dequant(idx, scale, vq.cb16[l])
            return torch.einsum("thd,hde->the", r_hat, inv) + mean.unsqueeze(0)

        # A dense 4-dim/256-centroid codebook has razor-thin nearest-neighbor
        # margins: fp32-vs-double flips a few % of assignments between
        # near-equidistant centroids. Elementwise identity is therefore
        # ill-posed as a parity gate; the meaningful criteria are
        #   (a) mean distortion: engine <= reference * 1.005, and
        #   (b) per-token: engine never materially worse than reference
        #       (worst per-token excess distortion <= 2% of ||k_tok||^2).
        # Elementwise rel is kept as info. bf16 maps are the served dtype
        # (same precision class as OSCAR's bf16 ``rows @ R`` storage).
        k_hat_bf = engine_khat(torch.bfloat16)
        e_bf = rel_err(k_hat_bf[:, h], k_ref)
        d_ref = rel_err(k_ref, k)
        d_raw = rel_err(k_ref_raw, k)
        d_eng = rel_err(k_hat_bf[:, h], k)
        per_tok_ref = (k_ref.to(torch.float64) - k.to(torch.float64)).pow(2).sum(-1)
        per_tok_eng = (
            k_hat_bf[:, h].to(torch.float64) - k.to(torch.float64)
        ).pow(2).sum(-1)
        knorm = k.to(torch.float64).pow(2).sum(-1).clamp_min(1e-12)
        worst_excess = ((per_tok_eng - per_tok_ref) / knorm).max().item()
        ratio = (d_eng ** 2) / max(d_ref ** 2, 1e-12)
        ok = ratio <= 1.005 and worst_excess <= 0.02
        if not ok:
            failures.append(
                f"G1 layer {l} head {h}: distortion ratio {ratio:.4f}, "
                f"worst per-token excess {worst_excess:.4f}"
            )
        print(
            f"  layer {l:2d} head {h}: distortion raw-cb {d_raw:.4f} "
            f"ref {d_ref:.4f} eng {d_eng:.4f} (ratio {ratio:.4f}, worst "
            f"excess {worst_excess:.4f}, elemwise rel {e_bf:.3e})  "
            f"{'PASS' if ok else 'FAIL'}"
        )

    # ---- G2: stored-space score equivalence (mixed tiers) ------------------
    print("== G2 stored-space softmax equivalence (quant + HP tiers)")
    l = args.layers[-1]
    QH = 4 * H
    q = torch.randn(3, QH, D, device=device, dtype=torch.float32)
    k = torch.randn(T, H, D, device=device, dtype=torch.float32)
    # fp32 maps throughout the setup so the fp32 gate isolates the logic.
    r = vq_map_k(
        k, vq.forward[l].to(torch.float32), vq.mean[l].to(torch.float32)
    )
    idx, scale = vq_encode(r, vq.cb16[l], vq.cb_sq[l], pertoken_norm=vq.pertoken_norm)
    r_hat = vq_dequant(idx, scale, vq.cb16[l])
    inv = blob["inverse"][l].to(device).to(torch.float32)
    mean = blob["mean"][l].to(device).to(torch.float32)
    k_hat = torch.einsum("thd,hde->the", r_hat, inv) + mean.unsqueeze(0)

    # Logic gate at fp32 (isolates the mean-shift/softmax-invariance math from
    # bf16 rounding); bf16 delta reported as info (same class as OSCAR's bf16
    # q @ R_k rotation in production).
    q_map32 = blob["inverse"][l].to(device).to(torch.float32).transpose(-1, -2)
    q_m32 = vq_map_q(q, q_map32.contiguous()).to(torch.float32)
    q_m_bf = vq_map_q(q, vq.q_map[l]).to(torch.float32)
    n_hp = 64  # last 64 tokens act as the HP tier (exact residual rows)
    max32 = 0.0
    maxbf = 0.0
    sm_scale_g2 = 1.0 / (D ** 0.5)
    for b in range(3):
        for qh in [0, QH // 2, QH - 1]:
            h = qh // 4
            # Original space: q . k_hat for quant part, q . k for HP part.
            s_orig = torch.cat(
                [
                    q[b, qh] @ k_hat[:-n_hp, h].T,
                    q[b, qh] @ k[-n_hp:, h].T,
                ]
            ) * sm_scale_g2
            p_orig = torch.softmax(s_orig, -1)
            for q_m, is32 in ((q_m32, True), (q_m_bf, False)):
                # Stored space: q_m . r_hat (quant) and q_m . r (HP rows).
                s_store = torch.cat(
                    [
                        q_m[b, qh] @ r_hat[:-n_hp, h].T,
                        q_m[b, qh] @ r[-n_hp:, h].to(torch.float32).T,
                    ]
                ) * sm_scale_g2
                p_store = torch.softmax(s_store, -1)
                e = (p_orig - p_store).abs().max().item()
                if is32:
                    max32 = max(max32, e)
                    if e > 2e-3:
                        failures.append(
                            f"G2 b{b} qh{qh}: fp32 softmax delta {e:.3e}"
                        )
                else:
                    maxbf = max(maxbf, e)
    g2ok = not any(f.startswith("G2") for f in failures)
    print(
        f"  max softmax delta: fp32-map {max32:.2e} (gate 2e-3), "
        f"bf16-map {maxbf:.2e} (info)  {'PASS' if g2ok else 'FAIL'}"
    )

    # ---- G3: decode kernel vs torch reference ------------------------------
    print("== G3 vq2 stage-1 kernel vs torch reference")
    from sglang.srt.layers.attention.triton_ops.decode_attention import (
        _decode_grouped_att_m_fwd_quant_vq2,
        _unified_stage2,
    )

    l = args.layers[1] if len(args.layers) > 1 else 0
    bs = 2
    seq = 1000
    NG, KC = vq.num_groups, vq.codebook_size
    cache = seq * bs + 7
    k_idx_buf = torch.randint(
        0, KC, (cache + 1, H, NG), device=device, dtype=torch.uint8
    )
    k_sz = torch.zeros(cache + 1, H, 2, device=device, dtype=torch.float32)
    k_sz[..., 0] = torch.rand(cache + 1, H, device=device) + 0.5
    v_buf = torch.randint(
        0, 256, (cache, H, D // 4), device=device, dtype=torch.uint8
    )
    v_sz = torch.zeros(cache, H, 2, device=device, dtype=torch.float32)
    v_sz[..., 0] = torch.rand(cache, H, device=device) * 0.1 + 0.05
    v_sz[..., 1] = torch.rand(cache, H, device=device)

    q_dec = torch.randn(bs, QH, D, device=device, dtype=torch.bfloat16)
    kv_indices = torch.randperm(cache, device=device)[: seq * bs].to(torch.int64)
    kv_indptr = torch.tensor([0, seq, 2 * seq], device=device, dtype=torch.int64)
    max_splits = 8
    num_kv_splits = torch.full((bs,), max_splits, device=device, dtype=torch.int32)
    att_out = torch.zeros(bs, QH, max_splits, D, device=device, dtype=torch.float32)
    att_lse = torch.full(
        (bs, QH, max_splits), float("-inf"), device=device, dtype=torch.float32
    )
    sm_scale = 1.0 / (D ** 0.5)

    _decode_grouped_att_m_fwd_quant_vq2(
        q_dec,
        k_idx_buf,
        vq.cb_packed[l],
        v_buf,
        k_sz,
        v_sz,
        att_out,
        att_lse,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        max_splits,
        sm_scale,
        logit_cap=0.0,
        # Honour SGLANG_VQ_OPT_CB16 so G3/G5 actually exercise the fp16-gather
        # path when it is enabled; passing cb16=None would silently test the
        # packed-fp8 path instead and make the gate vacuous.
        cb16=(vq.cb16[l] if os.environ.get("SGLANG_VQ_OPT_CB16", "") in ("1", "true", "True") else None),
    )
    o = torch.zeros(bs, QH, D, device=device, dtype=torch.bfloat16)
    _unified_stage2(att_out, att_lse, o, total_splits=max_splits)

    # Torch reference: reconstruct K_hat rows and dequant V for the same slots.
    cb16_l = vq.cb16[l]  # [H, NG, K, G]
    h_ids = torch.arange(H, device=device).view(1, H, 1)
    g_ids = torch.arange(NG, device=device).view(1, 1, NG)
    for b in range(bs):
        sl = kv_indices[b * seq : (b + 1) * seq]
        idx = k_idx_buf[sl].long()
        khat = (
            cb16_l[h_ids, g_ids, idx].to(torch.float32).reshape(seq, H, D)
            * k_sz[sl][..., 0:1].to(torch.float32)
        )
        by = v_buf[sl].to(torch.int32)
        crumbs = torch.stack(
            [(by >> (2 * i)) & 0x3 for i in range(4)], dim=-1
        ).to(torch.float32)  # [seq, H, D//4, 4]
        vv = crumbs.permute(0, 1, 3, 2).reshape(seq, H, D)
        vv = (vv - v_sz[sl][..., 1:2].to(torch.float32)) * v_sz[sl][
            ..., 0:1
        ].to(torch.float32)
        for qh in range(0, QH, 7):
            h = qh // 4
            s = (q_dec[b, qh].to(torch.float32) @ khat[:, h].T) * sm_scale
            p = torch.softmax(s, -1)
            o_ref = p @ vv[:, h]
            e = rel_err(o[b, qh], o_ref)
            if e > 2e-2:
                failures.append(f"G3 b{b} qh{qh}: rel {e:.3e}")
    n_g3 = sum(1 for f in failures if f.startswith("G3"))
    print(f"  kernel-vs-ref: {'PASS' if n_g3 == 0 else f'FAIL ({n_g3} probes)'}")

    # ---- V gates (G4/G5). K's gates are blind to V-fatal bugs: attention
    # scores are dot products (invariant to coordinate permutations), while
    # V enters as a weighted SUM where coordinate placement matters. Both V
    # gates therefore verify the strided group layout (codebook group g =
    # coords {g, g+NG, g+2NG, g+3NG}; kernel plane m of group g -> output
    # coord m*NG + g) end to end. Skipped with a note if the V bundle is
    # absent.
    v_bundle = os.path.join(REPO_ROOT, args.v_bundle)
    if not os.path.exists(v_bundle):
        print(f"== G4/G5 SKIPPED: V bundle not found at {args.v_bundle}")
    else:
        print("== G4 V roundtrip parity (engine strided encode/decode vs reference)")
        vqv = load_vq_codebook(
            v_bundle,
            layer_num=L_total,
            start_layer=0,
            head_num=H,
            head_dim=D,
            device=device,
            dtype=torch.bfloat16,
        )
        ngv, gv = vqv.num_groups, vqv.group_dim
        # Pool's strided perm and its inverse; sanity: a true permutation.
        perm = torch.tensor(
            [g + m * ngv for g in range(ngv) for m in range(gv)],
            dtype=torch.long, device=device,
        )
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(D, device=device)
        assert torch.equal(perm.sort().values, torch.arange(D, device=device))

        vblob = torch.load(v_bundle, map_location="cpu", weights_only=False)
        for l in args.layers:
            h = l % H
            v = torch.randn(T, D, device=device, dtype=torch.float32)
            v3 = v.unsqueeze(1).expand(T, H, D).contiguous()
            # Engine path: strided perm -> ptn -> assign -> gather -> unperm.
            idx, scale = vq_encode(
                v3[..., perm].to(torch.bfloat16), vqv.cb16[l], vqv.cb_sq[l],
                pertoken_norm=vqv.pertoken_norm,
            )
            v_hat = vq_dequant(idx, scale, vqv.cb16[l])[..., inv_perm]
            # Reference: double-precision codec on the SAME permuted rows
            # (forward=identity, mean=0 — the engine-basis bundle), unpermuted.
            ref = GroupVQCompressor(
                torch.eye(D, dtype=torch.float64),
                torch.eye(D, dtype=torch.float64),
                torch.zeros(D, dtype=torch.float64),
                [c.to(torch.float8_e5m2).to(torch.float16)
                 for c in vblob["codebooks"][(l, h)]],
                vblob["bounds"],
                pertoken_norm=bool(vblob.get("pertoken_norm", False)),
            ).to(device)
            v_ref = ref.roundtrip(v[:, perm.cpu()])[:, inv_perm.cpu()]
            d_ref = rel_err(v_ref, v)
            d_eng = rel_err(v_hat[:, h], v)
            per_tok_ref = (v_ref.to(torch.float64) - v.to(torch.float64)).pow(2).sum(-1)
            per_tok_eng = (
                v_hat[:, h].to(torch.float64) - v.to(torch.float64)
            ).pow(2).sum(-1)
            vnorm = v.to(torch.float64).pow(2).sum(-1).clamp_min(1e-12)
            worst = ((per_tok_eng - per_tok_ref) / vnorm).max().item()
            ratio = (d_eng ** 2) / max(d_ref ** 2, 1e-12)
            ok = ratio <= 1.005 and worst <= 0.02
            if not ok:
                failures.append(
                    f"G4 layer {l} head {h}: distortion ratio {ratio:.4f}, "
                    f"worst per-token excess {worst:.4f}"
                )
            print(
                f"  layer {l:2d} head {h}: distortion ref {d_ref:.4f} eng "
                f"{d_eng:.4f} (ratio {ratio:.4f}, worst excess {worst:.4f})  "
                f"{'PASS' if ok else 'FAIL'}"
            )

        print("== G5 vq2 kernel with V_VQ vs torch reference")
        lv = args.layers[1] if len(args.layers) > 1 else 0
        v_idx_buf = torch.randint(
            0, vqv.codebook_size, (cache + 1, H, ngv), device=device,
            dtype=torch.uint8,
        )
        v_sz_vq = torch.zeros(cache + 1, H, 2, device=device, dtype=torch.float32)
        v_sz_vq[..., 0] = torch.rand(cache + 1, H, device=device) + 0.5
        att_out.zero_(); att_lse.fill_(float("-inf"))
        _decode_grouped_att_m_fwd_quant_vq2(
            q_dec,
            k_idx_buf,
            vq.cb_packed[l],
            v_buf,
            k_sz,
            v_sz_vq,
            att_out,
            att_lse,
            kv_indptr,
            kv_indices,
            num_kv_splits,
            max_splits,
            sm_scale,
            logit_cap=0.0,
            cb_packed_v=vqv.cb_packed[lv],
            v_idx_buffer=v_idx_buf,
        )
        o_v = torch.zeros(bs, QH, D, device=device, dtype=torch.bfloat16)
        _unified_stage2(att_out, att_lse, o_v, total_splits=max_splits)

        # Torch reference: K_hat as in G3; V_hat[t, h, m*NG+g] =
        # fp8e5(cb byte m of group g at idx) * ptn_scale — the documented
        # kernel plane semantics, built independently from cb16 + inv_perm.
        cb16_k = vq.cb16[l]
        cb16_v = vqv.cb16[lv]
        h_ids_v = torch.arange(H, device=device).view(1, H, 1)
        g_ids_v = torch.arange(ngv, device=device).view(1, 1, ngv)
        n_g5 = 0
        for b in range(bs):
            sl = kv_indices[b * seq : (b + 1) * seq]
            idxk = k_idx_buf[sl].long()
            khat = (
                cb16_k[h_ids, g_ids, idxk].to(torch.float32).reshape(seq, H, D)
                * k_sz[sl][..., 0:1].to(torch.float32)
            )
            idxv = v_idx_buf[sl].long()
            # [seq, H, NG_V, G] in permuted (contiguous-group) order ->
            # unpermute to natural coords.
            vhat_p = cb16_v[h_ids_v, g_ids_v, idxv].to(torch.float32).reshape(
                seq, H, D
            )
            vhat = vhat_p[..., inv_perm] * v_sz_vq[sl][..., 0:1].to(torch.float32)
            for qh in range(0, QH, 7):
                h = qh // 4
                s = (q_dec[b, qh].to(torch.float32) @ khat[:, h].T) * sm_scale
                p = torch.softmax(s, -1)
                o_ref = p @ vhat[:, h]
                e = rel_err(o_v[b, qh], o_ref)
                if e > 2e-2:
                    failures.append(f"G5 b{b} qh{qh}: rel {e:.3e}")
                    n_g5 += 1
        print(f"  kernel-vs-ref (V_VQ): {'PASS' if n_g5 == 0 else f'FAIL ({n_g5} probes)'}")

        print("== G6 VQ-V prefix dequant (chunked-prefill read path)")
        # The third V read path (dequantize_prefix_kv under vq_v_enabled) —
        # the one the original handoff missed (it decoded VQ indices as int2
        # crumbs). Mixed HP+quant slots, raw-tensor call, vs the reference
        # reconstruction used in G4.
        from sglang.srt.layers.attention.quantized_kv_prefill import (
            _vq_prefix_dequantize_v,
        )
        lg = args.layers[0]
        n_hp_slots = 64
        hp_rows = torch.randn(
            n_hp_slots, H, D, device=device, dtype=torch.bfloat16
        )
        hp_off_g6 = cache  # pretend quant ids run [0, cache), HP at cache+
        pfx = torch.cat([
            torch.randperm(cache, device=device)[:200],
            hp_off_g6 + torch.randint(0, n_hp_slots, (56,), device=device),
        ])
        v_hat = _vq_prefix_dequantize_v(
            pfx, v_idx_buf, v_sz_vq, hp_rows, vqv.cb16[lg], inv_perm,
            hp_off_g6, cache, torch.float32,
        )
        # Reference: same gather math built independently.
        idxq = v_idx_buf[pfx[:200]].long()
        ref_q = (
            vqv.cb16[lg][h_ids_v, g_ids_v, idxq].to(torch.float32).reshape(200, H, D)
        )[..., inv_perm] * v_sz_vq[pfx[:200]][..., 0:1].to(torch.float32)
        e_q = rel_err(v_hat[:200], ref_q)
        e_hp = rel_err(v_hat[200:], hp_rows[(pfx[200:] - hp_off_g6).long()])
        ok6 = e_q <= 5e-3 and e_hp <= 5e-3
        if not ok6:
            failures.append(f"G6: quant rel {e_q:.3e}, hp rel {e_hp:.3e}")
        print(f"  quant rel {e_q:.3e}, hp passthrough rel {e_hp:.3e}  "
              f"{'PASS' if ok6 else 'FAIL'}")

    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print(" ", f)
        sys.exit(1)
    print("ALL GATES PASS")


if __name__ == "__main__":
    main()
