#!/usr/bin/env python3
"""Step 2b: Triton causal flash-attention in the RESIDUAL DOMAIN.
Takes q' = q@invᵀ and r̂ (decoded residual), runs fused online-softmax causal
attention, outputs @v. Equivalent to standard causal attention on decoded K,
but never forms fp16 K. Validated against a torch causal reference.
Single (gs, T, d) head-group; this is the kernel, not the full grid."""
import argparse, numpy as np, torch, triton, triton.language as tl
import torch.nn.functional as F

import run_pca_ec_deadzone as base
from kvq_codec import build_codecs_from_ladder_rans_cuda, load_ext, BatchRANSEncoder, BatchRANSDecoder


@triton.jit
def _attn_fwd(Q, R, V, Out, sm_scale,
              stride_qh, stride_qm, stride_qd,
              stride_rm, stride_rd, stride_vm, stride_vd,
              stride_oh, stride_om, stride_od,
              H, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr):
    # program: one (head, query-block)
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    q_ptrs = Q + pid_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < M, other=0.0)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    # causal: keys only up to the max query index in this block
    end_n = (pid_m + 1) * BLOCK_M
    for start_n in range(0, end_n, BLOCK_N):
        n = start_n + offs_n
        r_ptrs = R + n[:, None] * stride_rm + offs_d[None, :] * stride_rd      # r̂ shared across head-group
        r = tl.load(r_ptrs, mask=n[:, None] < N, other=0.0)
        qk = tl.dot(q, tl.trans(r), allow_tf32=False) * sm_scale                                  # (BLOCK_M, BLOCK_N)
        # causal mask
        causal = offs_m[:, None] >= n[None, :]
        qk = tl.where(causal & (n[None, :] < N), qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v_ptrs = V + n[:, None] * stride_vm + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=n[:, None] < N, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v, allow_tf32=False)
        m_i = m_new
    acc = acc / l_i[:, None]
    o_ptrs = Out + pid_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc, mask=offs_m[:, None] < M)


def triton_resid_attn(q_proj, r_hat, v, sm_scale):
    # q_proj: (gs, T, d)  r_hat: (T, d)  v: (T, d)
    gs, T, d = q_proj.shape
    out = torch.empty_like(q_proj)
    BLOCK_M, BLOCK_N = 32, 32
    grid = (triton.cdiv(T, BLOCK_M), gs)
    _attn_fwd[grid](
        q_proj, r_hat, v, out, sm_scale,
        q_proj.stride(0), q_proj.stride(1), q_proj.stride(2),
        r_hat.stride(0), r_hat.stride(1), v.stride(0), v.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        gs, T, T, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=d,
    )
    return out


def torch_causal(q, k, v, sm):
    sc = torch.einsum("gtd,sd->gts", q, k) * sm
    mask = torch.tril(torch.ones(q.shape[1], k.shape[0], device=q.device, dtype=torch.bool))
    sc = sc.masked_fill(~mask[None], -float("inf"))
    return torch.softmax(sc, -1) @ v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-idx", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--eval-idx", type=int, default=4)
    ap.add_argument("--b", type=int, default=2)
    ap.add_argument("--ptok", type=int, default=64)
    ap.add_argument("--dz", type=float, default=0.375)
    ap.add_argument("--lanes", type=int, default=16)
    ap.add_argument("--m-grid", type=float, nargs="+", default=[1.0, 1.05, 1.1, 1.25, 1.5])
    args = ap.parse_args()
    dev = torch.device("cuda")

    root = base.data_root(); manifest = base.load_manifest(root)
    sigma_q, sigma_k, k_mean, k_cov, meta = base.calib_moments(root, manifest, args.calib_idx)
    L, Hkv, d = meta["n_layers"], meta["n_kv_heads"], meta["d_head"]
    qpca_unc = base.build_qpca_basis(sigma_q, sigma_k)
    qpca_unc["sigma_k"], qpca_unc["sigma_q"] = sigma_k, sigma_q
    qpca_cen = base.build_qpca_basis(sigma_q, k_cov); qpca_cen["sigma_k"] = sigma_k
    Ffwd, inv = qpca_cen["forward"], qpca_cen["inverse"]
    fetch_calib = base._codes_for_idx(root, manifest, args.calib_idx, Ffwd, k_mean, L, Hkv, d)
    _, delta0, model0 = base.build_qpca_ec(qpca_cen, qpca_unc, k_mean, args.b, L, Hkv,
                                           fetch_calib, root, args.calib_idx,
                                           dz=args.dz, match_rate=False, uniform_step=True)
    ladder = [(1.0, delta0, model0)]
    for m in sorted(set(args.m_grid) | {1.0}):
        if abs(m - 1.0) > 1e-9:
            dm = (delta0 * m).float()
            ladder.append((m, dm, base.freeze_coder_model(fetch_calib, dm, L, Hkv, d, args.dz)))
    page_bits = args.b * d * args.ptok
    ext = load_ext()
    codecs = build_codecs_from_ladder_rans_cuda(Ffwd, inv, k_mean, ladder, L, Hkv,
                                                page_bits, args.ptok, args.dz,
                                                lanes=args.lanes, ext=ext, device="cuda")
    enc = BatchRANSEncoder(codecs); decod = BatchRANSDecoder(codecs)

    art = torch.load(root / manifest["examples"][args.eval_idx]["file"],
                     map_location="cpu", weights_only=False)
    q_all, k_all, v_all = art["q_post"], art["k_post"], art["v"]
    T = int(art["prompt_length"]); gs = q_all.shape[1] // Hkv
    k_grid = {(l, h): k_all[l, h, :T, :].float() for l in range(L) for h in range(Hkv)}
    bufs = enc.encode_grid(k_grid); k_dec = decod.decode_grid(bufs)

    sm = 1.0 / np.sqrt(d)
    max_err = 0.0
    for l in range(1, L):
        for h in range(Hkv):
            c = decod.codecs[(l, h)]
            q  = q_all[l, h*gs:(h+1)*gs, :T, :].to(dev).float().contiguous()
            v  = v_all[l, h, :T, :].to(dev).float().contiguous()
            kh = k_dec[(l, h)][:T].to(dev).float()
            invT = torch.as_tensor(c.inv, dtype=torch.float32, device=dev)
            fwd  = torch.as_tensor(c.fwd, dtype=torch.float32, device=dev)
            mu   = torch.as_tensor(c.mu,  dtype=torch.float32, device=dev)
            r_hat  = ((kh - mu) @ fwd).contiguous()
            q_proj = (q @ invT.T).contiguous()

            # reference: causal attention on decoded K
            ref = torch_causal(q, kh, v, sm)
            # triton residual-domain causal
            out = triton_resid_attn(q_proj, r_hat, v, sm)
            e = float((ref - out).abs().max()); max_err = max(max_err, e)
            if l == 1 and h == 0:
                print(f"first head max|delta| = {e:.3e}")
    print(f"\nTriton residual causal vs torch causal-on-decoded-K  max|delta| = {max_err:.3e}")
    print("PASS if ~1e-3 or below.")

    

if __name__ == "__main__":
    main()