"""pgq7 K4: end-to-end quantized-resident decode demo on Qwen3-8B.

After a normal prefill, the prompt's K cache (layers >= 1) is compressed
with the shipping pgq8 codec (pgq_dctlmrw_rdo @ 2.0, bundle 2cf29a8a),
PACKED, and the fp16 K tensors are FREED — decode attention over the
prefix runs the fused two-phase kernel on packed bytes, merged (via
softmax partials) with an fp16 ring that holds layer 0, the sink page,
the partial tail page, and all generated tokens. This is the plan7 R3
demo form (standalone loop, not bench-stack integration).

Wiring: transformers 5.x attention registry — a "pgq" attention function
is registered and a RingCache (Cache subclass) returns only ring tokens
from update(); the packed prefix never materializes. Logits are mu-dropped
consistently on both tiers (kernel: (q G^T) . y_hat; ring: q.(k - mu)).

Gates (run in order; the script aborts on a failed gate):
  G1  plumbing: our loop with an UNCOMPRESSED fp16 ring (everything in the
      ring) reproduces HF generate's greedy tokens exactly.
  G2  numerics: packed-kernel arm vs a control loop whose prefix attention
      is exact torch over the DEQUANTIZED K-hat (Mode-A semantics) —
      greedy agreement reported, expected ~100% (kernel parity is 3e-4).
Metrics: greedy agreement vs full-fp16, decode tokens/s per arm, prefix
K bytes (packed vs fp16), page-seal (pack) time, peak GPU memory.

Usage: python -u pipelines/kernels/k4_decode_demo.py --gpu 0 --new-tokens 64
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import _bootstrap  # noqa: E402,F401

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.cache_utils import Cache  # noqa: E402
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402

from kvq.compression.page_quant import load_pgq_compressors_from_bundle  # noqa: E402
from kvq.kernels.golden import build_golden, segment_layout, stack_heads  # noqa: E402
from kvq.kernels.pgq_pack import pack_sequence  # noqa: E402
from kvq.kernels.pgq_decode_attn import page_attention_v2  # noqa: E402

MODEL = "Qwen/Qwen3-8B"
BUNDLE = REPO / "artifacts/page_quant2/pgq8_bundle__qwen3_8b.pt"
K_METHOD, K_BITS = "pgq_dctlmrw_rdo", 2.0
HGP = 16


class RingCache(Cache):
    """fp16 ring per layer; update() returns ring-only tensors. The packed
    prefix lives in DEMO.packs and is served inside pgq_attention."""

    def __init__(self, n_layers):
        self.rk = [None] * n_layers
        self.rv = [None] * n_layers
        self.prefix_len = [0] * n_layers

    def update(self, k, v, layer_idx, cache_kwargs=None):
        if self.rk[layer_idx] is None:
            self.rk[layer_idx], self.rv[layer_idx] = k, v
        else:
            self.rk[layer_idx] = torch.cat([self.rk[layer_idx], k], 2)
            self.rv[layer_idx] = torch.cat([self.rv[layer_idx], v], 2)
        return self.rk[layer_idx], self.rv[layer_idx]

    def get_seq_length(self, layer_idx=0):
        r = self.rk[layer_idx]
        return self.prefix_len[layer_idx] + (0 if r is None else r.shape[2])

    def get_max_cache_shape(self):
        return None


class Demo:
    """Holds per-layer packed state + the attention implementation mode."""

    def __init__(self):
        self.packs = {}          # layer -> gm dict (segment layout)
        self.mu = {}             # layer -> (H, d) key means
        self.gt = {}             # layer -> (H, d, d) G^T stacks
        self.khat = {}           # layer -> (H, T, d) dequantized control
        self.served = {}         # layer -> kernel-served row indices
        self.mode = "ring"       # ring | kernel | control


DEMO = Demo()


def pgq_attention(module, q, k, v, attention_mask, scaling=None,
                  dropout=0.0, **kwargs):
    li = module.layer_idx
    qlen = q.shape[2]
    if qlen > 1 or li not in DEMO.packs or DEMO.mode == "ring":
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=qlen > 1, scale=scaling, enable_gqa=True)
        return out.transpose(1, 2).contiguous(), None
    gm = DEMO.packs[li]
    H, d = gm["H"], gm["d"]
    hg = gm["hg"]
    qh = q[0, :, 0].reshape(H, hg, d).float()               # (H, hg, d)
    mu = DEMO.mu[li]
    if DEMO.mode == "kernel":
        gm["qt"][:, :hg] = torch.bmm(qh, DEMO.gt[li])
        m_p, l_p, acc_p = page_attention_v2(gm, 16, return_partials=True)
    else:                                                    # control
        rows = DEMO.served[li]
        kh = DEMO.khat[li][:, rows]                          # kernel rows only
        z = torch.bmm(qh, kh.transpose(1, 2).float())        # (H, hg, R)
        z = (z - (qh @ mu.unsqueeze(2))) * scaling
        mt = z.max(2).values
        pt = torch.exp(z - mt[..., None])
        lt = pt.sum(2)
        at = torch.bmm(pt, gm["v"][:, rows].float())
        m_p = torch.full((H, 1, HGP), float("-inf"), device=q.device)
        l_p = torch.zeros(H, 1, HGP, device=q.device)
        acc_p = torch.zeros(H, 1, HGP, d, device=q.device)
        m_p[:, 0, :hg] = mt
        l_p[:, 0, :hg] = lt
        acc_p[:, 0, :hg] = at
    # ring partials (mu-dropped): z = q.(k_ring - mu)
    rk, rv = k[0].float(), v[0].float()                      # (H, R, d)
    zr = (torch.bmm(qh, (rk - mu.unsqueeze(1)).transpose(1, 2))) * scaling
    mr = zr.max(2).values
    pr = torch.exp(zr - mr[..., None])
    lr = pr.sum(2)
    ar = torch.bmm(pr, rv)
    mrp = torch.full((H, 1, HGP), float("-inf"), device=q.device)
    lrp = torch.zeros(H, 1, HGP, device=q.device)
    arp = torch.zeros(H, 1, HGP, d, device=q.device)
    mrp[:, 0, :hg] = mr
    lrp[:, 0, :hg] = lr
    arp[:, 0, :hg] = ar
    m_all = torch.cat([m_p, mrp], 1)
    l_all = torch.cat([l_p, lrp], 1)
    a_all = torch.cat([acc_p, arp], 1)
    m_star = m_all.max(1).values
    w = torch.exp(m_all - m_star[:, None, :]).nan_to_num(0.0)
    l_star = (l_all * w).sum(1).clamp_min(1e-30)
    o = ((a_all * w[..., None]).sum(1) / l_star[..., None])[:, :hg]
    out = o.reshape(1, H * hg, 1, d).to(q.dtype)
    return out.transpose(1, 2).contiguous(), None


def greedy(model, cur, cache, n_new, pos0):
    """cur: (1, m) tokens not yet in cache; pos0 = absolute position AFTER
    cur (i.e. total sequence length including cur)."""
    out_ids = []
    torch.cuda.synchronize()
    t0 = time.time()
    pos = pos0
    for i in range(n_new):
        with torch.no_grad():
            r = model(input_ids=cur, past_key_values=cache, use_cache=True,
                      cache_position=torch.arange(pos - cur.shape[1], pos,
                                                  device=cur.device))
        nxt = r.logits[0, -1].argmax().reshape(1, 1)
        out_ids.append(int(nxt))
        cur = nxt
        pos += 1
    torch.cuda.synchronize()
    dt = time.time() - t0
    return out_ids, n_new / dt


def pack_prefix(model, cache, comps, ptok, dev):
    """Compress+pack each layer's prefix K (layers >= 1), move sink page /
    partial tail into the ring, free the fp16 prefix K."""
    L = model.config.num_hidden_layers
    t0 = time.time()
    kbytes_fp16 = kbytes_packed = 0
    for li in range(L):
        k_full = cache.rk[li][0]                             # (H, T, d)
        v_full = cache.rv[li][0]
        H, T, d = k_full.shape
        kbytes_fp16 += k_full.numel() * 2
        if li == 0:
            cache.prefix_len[li] = 0
            continue                                         # ring keeps L0
        nfull = T // ptok
        cut = max(ptok, (nfull - 0) * ptok)                  # tail split
        cut = nfull * ptok
        golds, packeds, mus, gts, khs = [], [], [], [], []
        for h in range(H):
            comp = comps[(li, h)].to(dev)
            em = {}
            khat = comp.roundtrip(k_full[h].unsqueeze(0), emit=em).squeeze(0)
            comp.to(torch.device("cpu"))
            em = {k2: (v2.cpu() if torch.is_tensor(v2) else v2)
                  for k2, v2 in em.items()}
            packed = pack_sequence(em["codes"], em["assign"], comp.profiles,
                                   comp.ptok, nsink=em["nsink"],
                                   sink_codes=em["sink_codes"])
            packeds.append(packed)
            golds.append(build_golden(comp, em, packed,
                                      torch.zeros(4, d), device=dev,
                                      v=v_full[h].float().cpu()))
            mus.append(comp.mu.clone())
            gts.append(comp.inverse_map.t().contiguous().clone())
            khs.append(khat)
        gm = segment_layout(stack_heads(golds), packeds)
        # ring keeps rows the kernel doesn't serve (sink page + tail)
        served = torch.zeros(T, dtype=torch.bool)
        for p in gm["pages"].tolist():
            served[p * ptok:(p + 1) * ptok] = True
        keep = torch.nonzero(~served).squeeze(1).to(dev)
        gm["tier_idx"] = torch.empty(0, dtype=torch.int64, device=dev)
        DEMO.served[li] = torch.nonzero(served).squeeze(1).to(dev)
        DEMO.packs[li] = gm
        DEMO.mu[li] = torch.stack(mus).to(dev).float()
        DEMO.gt[li] = torch.stack(gts).to(dev).float()
        DEMO.khat[li] = torch.stack(khs).to(dev)
        cache.rk[li] = k_full[:, keep][None].contiguous()
        cache.rv[li] = v_full[:, keep][None].contiguous()
        cache.prefix_len[li] = T - keep.numel()
        kbytes_packed += (int(gm["seg_pay"].numel())
                          + int(gm["row_rung"].numel() * 3 / 8)
                          + cache.rk[li].numel() * 2)
    torch.cuda.synchronize()
    return time.time() - t0, kbytes_fp16, kbytes_packed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--prompt-tokens", type=int, default=6000)
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(dev)

    ALL_ATTENTION_FUNCTIONS["pgq"] = pgq_attention
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, attn_implementation="pgq",
        device_map={"": dev}).eval()
    ptok = 64

    text = (REPO / "notes/page_quant/studies/report8.md").read_text() * 8
    ids = tok(text, return_tensors="pt").input_ids[:, :args.prompt_tokens]
    ids = torch.cat([ids, tok("\n\nSummarize the study above in one "
                              "paragraph:", return_tensors="pt").input_ids],
                    1).to(dev)
    T0 = ids.shape[1]
    print(f"[k4] prompt {T0} tokens", flush=True)

    def prefill(cache):
        DEMO.mode = "ring"
        with torch.no_grad():
            r = model(input_ids=ids, past_key_values=cache, use_cache=True,
                      cache_position=torch.arange(T0, device=dev))
        return r.logits[0, -1].argmax().reshape(1, 1)

    # --- arm A: full fp16 via our loop (G1 vs HF generate) ---------------
    cache_a = RingCache(model.config.num_hidden_layers)
    first_a = prefill(cache_a)
    toks_a, tps_a = greedy(model, first_a, cache_a, args.new_tokens,
                           pos0=T0 + 1)
    toks_a = [int(first_a)] + toks_a
    ref = model.generate(ids, max_new_tokens=args.new_tokens + 1,
                         do_sample=False, temperature=None, top_p=None,
                         top_k=None)
    toks_hf = ref[0, T0:].tolist()
    g1 = sum(a == b for a, b in zip(toks_a, toks_hf)) / len(toks_hf)
    print(f"[k4] G1 plumbing vs HF generate: agreement {g1:.3f}", flush=True)
    assert g1 > 0.98, "G1 failed: loop does not reproduce HF"

    # --- pack, then arm B (kernel) and arm C (control) --------------------
    comps, _ = load_pgq_compressors_from_bundle(str(BUNDLE), K_METHOD,
                                                K_BITS)
    torch.cuda.reset_peak_memory_stats()
    cache_b = RingCache(model.config.num_hidden_layers)
    first_b = prefill(cache_b)
    seal_s, kb_fp16, kb_packed = pack_prefix(model, cache_b, comps, ptok,
                                             dev)
    # numeric self-check (layer 1): kernel partial-reduce vs exact torch on
    # the served rows — catches wiring bugs before any generation
    li = 1
    gm = DEMO.packs[li]
    H, d, hg = gm["H"], gm["d"], gm["hg"]
    qtest = torch.randn(H, hg, d, device=dev)
    gm["qt"][:, :hg] = torch.bmm(qtest, DEMO.gt[li])
    m_p, l_p, acc_p = page_attention_v2(gm, 16, return_partials=True)
    m_s = m_p.max(1).values
    w = torch.exp(m_p - m_s[:, None, :]).nan_to_num(0.0)
    l_s = (l_p * w).sum(1).clamp_min(1e-30)
    o_k = ((acc_p * w[..., None]).sum(1) / l_s[..., None])[:, :hg]
    rows = DEMO.served[li]
    kh = DEMO.khat[li][:, rows].float()
    z = (torch.bmm(qtest, kh.transpose(1, 2))
         - qtest @ DEMO.mu[li].unsqueeze(2)) * gm["sm_scale"]
    o_r = torch.softmax(z, 2) @ gm["v"][:, rows].float()
    sc_err = float((o_k - o_r).norm() / o_r.norm())
    print(f"[k4] self-check layer1 kernel vs exact: rel {sc_err:.2e}",
          flush=True)
    assert sc_err < 2e-2, "self-check failed"
    print(f"[k4] page-seal {seal_s:.1f}s; prefix K fp16 {kb_fp16/1e6:.0f} MB"
          f" -> packed+ring {kb_packed/1e6:.0f} MB "
          f"({kb_fp16/max(kb_packed,1):.1f}x)", flush=True)

    DEMO.mode = "control"
    cache_c = RingCache(model.config.num_hidden_layers)
    for li in range(model.config.num_hidden_layers):
        cache_c.rk[li] = cache_b.rk[li].clone()
        cache_c.rv[li] = cache_b.rv[li].clone()
        cache_c.prefix_len[li] = cache_b.prefix_len[li]
    toks_c, tps_c = greedy(model, first_b, cache_c, args.new_tokens,
                           pos0=T0 + 1)

    DEMO.mode = "kernel"
    toks_b, tps_b = greedy(model, first_b, cache_b, args.new_tokens,
                           pos0=T0 + 1)
    g2 = sum(a == b for a, b in zip(toks_b, toks_c)) / len(toks_c)
    gq = sum(a == b for a, b in zip(toks_b, toks_hf[1:])) / len(toks_b)
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[k4] G2 kernel vs control agreement: {g2:.3f}", flush=True)
    print(f"[k4] quality: quantized vs fp16 tokens agreement {gq:.3f}",
          flush=True)
    print(f"[k4] tokens/s: fp16 {tps_a:.2f} | control {tps_c:.2f} | "
          f"kernel-resident {tps_b:.2f} | peak mem {peak:.1f} GB",
          flush=True)

    out = {"prompt_tokens": T0, "new_tokens": args.new_tokens,
           "G1_plumbing_vs_hf": g1, "G2_kernel_vs_control": g2,
           "quant_vs_fp16_agreement": gq,
           "tps": {"fp16_loop": tps_a, "control_dequant": tps_c,
                   "kernel_resident": tps_b},
           "prefix_k_bytes": {"fp16": kb_fp16, "packed_plus_ring": kb_packed},
           "page_seal_s": seal_s, "peak_mem_gb": peak,
           "bundle": "2cf29a8a", "method": f"{K_METHOD}@{K_BITS}"}
    dst = REPO / "artifacts/kernels/k4_decode_demo.json"
    dst.write_text(json.dumps(out, indent=1))
    print(f"[k4] wrote {dst}", flush=True)


if __name__ == "__main__":
    main()
