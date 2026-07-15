# Ported from Samuel's JointQK group-VQ implementation (uncommitted working
# tree of /vault/samuel/efficient-llm/JointQK, branch entropy_coding, HEAD
# 52e4875), with his permission. Source snapshot + sha256 hashes:
# third_party/samuel_vq/PROVENANCE.md. Quantization math (_kmeans, ECVQ
# penalty, GroupVQCompressor roundtrip, wraps, stratified perm, waterfill
# allocation) is kept verbatim so parity against his codebooks holds
# bit-for-bit in double precision; only the bundle loading / method-name
# plumbing below is ours.
"""Per-token group vector quantization of QPCA-basis key residuals.

Design (Samuel's): residual r = (k - mean) @ forward is split into NG
contiguous groups of G coords (stratified rank permutation folded into the
basis so each group spans the eigenvalue spectrum); each group is quantized
by nearest-centroid lookup into a per-(layer, head, group) k-means codebook
with K_g = 2^{b_g} entries. Flat allocation: b_g = bpc*G everywhere (2.0
b/coord exact). Waterfill: reverse water-filling of the QPCA score over
groups. Decode = index gather + inverse transform; no per-token metadata
(vs OSCAR's 32 bits/token of runtime scale/zero).

Protection wraps (compose): SinkRecentWrap keeps a small positional band
fp16; OutlierProtectWrap restores the worst-reconstructed frac of tokens to
fp8 (fixed-rate VQ gets variable-rate behaviour where it matters).
"""
from __future__ import annotations

import math

import torch


def group_boundaries(d: int, G: int, bpc: int = 2) -> list[tuple[int, int, int]]:
    """(start, end, bits_for_group) per group covering [0, d) at uniform bpc."""
    bounds = []
    n_full = d // G
    for i in range(n_full):
        bounds.append((i * G, (i + 1) * G, bpc * G))
    rem = d - n_full * G
    if rem > 0:
        bounds.append((n_full * G, d, bpc * rem))
    return bounds


def _kmeans(x: torch.Tensor, K: int, iters: int = 25, seed: int = 0,
            ecvq_lambda: float = 0.0) -> torch.Tensor:
    """Batched Lloyd. x: (N, g) -> centroids (K, g). ECVQ: ecvq_lambda>0 adds
    lambda*(-log2 p_i) (previous epoch's usage) to the squared-distance
    assignment objective; decoder unchanged. lambda=0 == plain k-means."""
    N, g = x.shape
    gen = torch.Generator(device=x.device).manual_seed(seed)
    if N <= K:
        idx = torch.randint(0, N, (K,), generator=gen, device=x.device)
        cent = x[idx].clone()
        cent[N:] += 1e-4 * torch.randn(cent[N:].shape, generator=gen, device=x.device)
        return cent
    idx = torch.randperm(N, generator=gen, device=x.device)[:K]
    cent = x[idx].clone()

    cost = torch.zeros(K, device=x.device, dtype=x.dtype)
    # Chunk assignment so the (block, K) distance matrix stays ~0.8 GB fp32.
    block = max(4096, min(N, int(2e8 // max(K, 1))))

    for _ in range(iters):
        new_cent = torch.zeros_like(cent)
        counts = torch.zeros(K, device=x.device, dtype=x.dtype)
        min_d = torch.empty(N, device=x.device, dtype=x.dtype)
        for s in range(0, N, block):
            xb = x[s:s + block]
            db = torch.cdist(xb, cent)
            if ecvq_lambda > 0.0:
                obj = db * db + ecvq_lambda * cost[None, :]
            else:
                obj = db
            ab = obj.argmin(dim=1)
            new_cent.index_add_(0, ab, xb)
            counts.index_add_(0, ab, torch.ones(xb.shape[0], device=x.device, dtype=x.dtype))
            min_d[s:s + block] = db.gather(1, ab.unsqueeze(1)).squeeze(1)
        empty = counts == 0
        nonempty = ~empty
        new_cent[nonempty] /= counts[nonempty].unsqueeze(-1)

        n_empty = int(empty.sum())
        if n_empty > 0:
            worst = torch.argsort(min_d, descending=True)[:n_empty]
            new_cent[empty] = x[worst]

        if ecvq_lambda > 0.0:
            p = counts / counts.sum().clamp_min(1.0)
            cost = -torch.log2(p.clamp_min(1.0 / (N * K)))
            cost[empty] = 0.0

        cent = new_cent
    return cent


def _assign(x: torch.Tensor, cent: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x, cent).argmin(dim=1)


def stratified_perm(d: int, G: int) -> torch.Tensor:
    """rank r -> group (r % NG), contiguous in the permuted basis."""
    NG = math.ceil(d / G)
    perm = []
    for g in range(NG):
        for j in range(G):
            r = g + j * NG
            if r < d:
                perm.append(r)
    assert sorted(perm) == list(range(d)), "not a permutation"
    return torch.tensor(perm, dtype=torch.long)


def waterfill_continuous(score: torch.Tensor, total: float) -> torch.Tensor:
    """reverse water-filling: bits_j = max(0, 0.5*log2(score_j/theta))."""
    s = score.clamp_min(1e-30).double()
    lo = torch.tensor(1e-40, dtype=torch.float64)
    hi = s.max()
    for _ in range(80):
        th = (lo * hi).sqrt()
        b = (0.5 * torch.log2(s / th)).clamp_min(0)
        if b.sum() > total:
            lo = th
        else:
            hi = th
    th = (lo * hi).sqrt()
    return (0.5 * torch.log2(s / th)).clamp_min(0)


def group_bit_alloc(score_perm, bounds, avg_bits, max_k_bits) -> list[int]:
    """Integer bits per group from waterfilling the permuted score; sums to
    avg_bits*d with a largest-remainder round + cap redistribution."""
    d = bounds[-1][1]
    total = avg_bits * d
    bc = waterfill_continuous(score_perm, total)
    gb_cont = torch.tensor([float(bc[s:e].sum()) for (s, e, _) in bounds])
    fl = torch.floor(gb_cont)
    rem = int(round(total - float(fl.sum())))
    frac = gb_cont - fl
    order = torch.argsort(frac, descending=True)
    gb = fl.clone()
    for i in range(max(0, rem)):
        gb[order[i % len(order)]] += 1
    gb = gb.long().clamp(0, max_k_bits)
    diff = int(total - int(gb.sum()))
    while diff > 0:
        cand = (gb < max_k_bits).nonzero(as_tuple=True)[0]
        if len(cand) == 0:
            break
        j = cand[gb[cand].argmin()]
        gb[j] += 1
        diff -= 1
    while diff < 0:
        cand = (gb > 0).nonzero(as_tuple=True)[0]
        if len(cand) == 0:
            break
        j = cand[gb[cand].argmax()]
        gb[j] -= 1
        diff += 1
    return gb.tolist()


class GroupVQCompressor:
    """Group VQ over a QPCA-basis residual. Row-vector convention:
    transformed = (k - mean) @ forward_map; recon = (r_hat @ inverse_map) + mean.
    pertoken_norm: OSCAR/KIVI-style per-token RMS scale before lookup (codebook
    must be trained with the same normalization)."""

    def __init__(self, forward_map, inverse_map, mean, codebooks, bounds,
                 pertoken_norm=False):
        self.forward_map = forward_map
        self.inverse_map = inverse_map
        self.mean = mean
        self.codebooks = codebooks   # list of (K_i, g_i)
        self.bounds = bounds         # list of (start, end, bits)
        self.pertoken_norm = pertoken_norm

    def to(self, device):
        self.forward_map = self.forward_map.to(device)
        self.inverse_map = self.inverse_map.to(device)
        self.mean = self.mean.to(device)
        self.codebooks = [c.to(device) for c in self.codebooks]
        return self

    def encode_idx(self, k: torch.Tensor) -> list[torch.Tensor]:
        r = (k.double() - self.mean.double()) @ self.forward_map.double()
        return [_assign(r[:, s:e], cb.double())
                for (s, e, _bits), cb in zip(self.bounds, self.codebooks)]

    def decode_idx(self, idx_list, dtype=torch.float32) -> torch.Tensor:
        d = self.bounds[-1][1]
        T = idx_list[0].shape[0]
        r_hat = torch.empty(T, d, dtype=torch.float64, device=idx_list[0].device)
        for (s, e, _bits), cb, idx in zip(self.bounds, self.codebooks, idx_list):
            r_hat[:, s:e] = cb.double()[idx]
        k_hat = r_hat @ self.inverse_map.double() + self.mean.double()
        return k_hat.to(dtype)

    def roundtrip(self, k: torch.Tensor) -> torch.Tensor:
        dtype = k.dtype
        lead = k.shape[:-1]
        kf = k.reshape(-1, k.shape[-1])
        r = (kf.double() - self.mean.double()) @ self.forward_map.double()
        if self.pertoken_norm:
            scale = r.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
            r = r / scale
        r_hat = torch.empty_like(r)
        for (s, e, _bits), cb in zip(self.bounds, self.codebooks):
            idx = _assign(r[:, s:e], cb.double())
            r_hat[:, s:e] = cb.double()[idx]
        if self.pertoken_norm:
            r_hat = r_hat * scale
        k_hat = r_hat @ self.inverse_map.double() + self.mean.double()
        return k_hat.reshape(*lead, k.shape[-1]).to(dtype)

    @property
    def bits_per_coord(self) -> float:
        d = self.bounds[-1][1]
        return sum(b for _, _, b in self.bounds) / d


class SinkRecentWrap:
    """Keep first `sink` / last `recent` positions fp16, quantize the middle.
    Sequence axis is dim -2 ((B, S, d) from the press hook)."""

    def __init__(self, inner, sink: int = 0, recent: int = 0):
        self.inner = inner
        self.sink = int(sink)
        self.recent = int(recent)
        self.forward_map = getattr(inner, "forward_map", None)

    def to(self, device):
        self.inner.to(device)
        return self

    def roundtrip(self, k):
        if k.dim() < 2 or (self.sink == 0 and self.recent == 0):
            return self.inner.roundtrip(k)
        S = k.shape[-2]
        s = min(self.sink, S)
        r = min(self.recent, max(S - s, 0))
        if s + r >= S:
            return k
        out = k.clone()
        out[..., s:S - r, :] = self.inner.roundtrip(k[..., s:S - r, :])
        return out


class OutlierProtectWrap:
    """Restore the worst-reconstructed `frac` of tokens to fp8 (content-based
    protection; composes on top of SinkRecentWrap). Rate cost:
    +frac*(8 - base) b/coord."""

    def __init__(self, inner, frac: float = 0.0):
        self.inner = inner
        self.frac = float(frac)
        self.forward_map = getattr(inner, "forward_map", None)

    def to(self, device):
        self.inner.to(device)
        return self

    def roundtrip(self, k):
        out = self.inner.roundtrip(k)
        if self.frac <= 0 or k.dim() < 2:
            return out
        S = k.shape[-2]
        n = int(self.frac * S)
        if n <= 0:
            return out
        err = ((k.double() - out.double()) ** 2).sum(-1)
        idx = err.topk(n, dim=-1).indices
        prot = k.gather(-2, idx.unsqueeze(-1).expand(*idx.shape, k.shape[-1]))
        prot = prot.to(torch.float8_e4m3fn).to(out.dtype)
        out.scatter_(-2, idx.unsqueeze(-1).expand(*idx.shape, k.shape[-1]), prot)
        return out


# --- bundle loading (ours) --------------------------------------------------

# pgq_vqg method grammar: pgq_vqg[b][oNN]_<flat|wf>
#   b   -> SinkRecentWrap(sink=4, recent=32)   (Samuel's s4r32 headline band)
#   oNN -> OutlierProtectWrap(frac=NN/100)     (e.g. o05 = 5%)
# The rate is baked into the codebook bundle; the press's k_bits is validated
# against the bundle's bits_per_coord as a label-integrity check.
_VQG_BAND_SINK, _VQG_BAND_RECENT = 4, 32


def parse_vqg_method(k_method: str):
    import re
    m = re.fullmatch(r"pgq_vqg(b?)(?:o(\d{2}))?_(flat|wf)", k_method)
    if m is None:
        raise ValueError(f"unrecognised vqg method {k_method!r}")
    band = bool(m.group(1))
    frac = int(m.group(2)) / 100.0 if m.group(2) else 0.0
    return band, frac, m.group(3)


def load_vqg_compressors(blob, path, k_method, b_page):
    band, frac, alloc = parse_vqg_method(k_method)
    want_alloc = {"flat": "flat", "wf": "waterfill"}[alloc]
    if blob.get("allocation") != want_alloc:
        raise ValueError(
            f"{k_method} expects allocation={want_alloc!r} but bundle {path} "
            f"was trained with {blob.get('allocation')!r}")
    base_bpc = float(blob["bits_per_coord"])
    if abs(base_bpc - float(b_page)) > 0.01:
        raise ValueError(
            f"k_bits={b_page} but bundle {path} carries {base_bpc:.4f} b/coord "
            f"(rate is fixed by the codebook; pass the bundle's rate)")
    F, inv, mean = blob["forward"], blob["inverse"], blob["mean"]
    bounds, cbs = blob["bounds"], blob["codebooks"]
    ptn = bool(blob.get("pertoken_norm", False))
    L, H = F.shape[0], F.shape[1]
    comps = {}
    for l in range(1, L):
        for h in range(H):
            c = GroupVQCompressor(F[l, h], inv[l, h], mean[l, h],
                                  list(cbs[(l, h)]), bounds, pertoken_norm=ptn)
            if band:
                c = SinkRecentWrap(c, _VQG_BAND_SINK, _VQG_BAND_RECENT)
            if frac > 0:
                c = OutlierProtectWrap(c, frac)
            comps[(l, h)] = c
    meta = {"n_layers": L, "n_kv_heads": H, "ptok": None,
            "bits_per_coord": base_bpc, "band": band, "outlier_frac": frac}
    return comps, meta
