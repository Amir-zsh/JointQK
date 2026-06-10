#!/usr/bin/env python3
"""Deployable QPCA-EC KV codec with a RUNG LADDER for fixed-size pages.

Everything is frozen on calibration: QPCA basis, the rung ladder (each rung =
a deadzone step delta + per-coord frequency model), all of it. Nothing is refit
on eval. The only eval-dependent step is (a) snapping an eval index to the
nearest calib alphabet bin (a held-out penalty) and (b) per-page rung SELECTION,
which is legitimate adaptive coding: the encoder always sees the data it codes,
the chosen rung id is written to the header, and decode is exact. Selection is
what buys the fixed page size -- each page is forced to fit page_bits.

Per (layer, kv-head), per page of P_tok tokens:
  - estimate coded length at each rung (cross-entropy -sum log2 p, cheap) to pick
    the finest rung whose estimate fits the payload budget;
  - actually range-encode at that rung; if the real payload overflows, step to a
    coarser rung and re-encode; the coarsest (terminal) rung is the guaranteed fit.
So every stored page satisfies  payload_bits + rung_id_bits <= page_bits.

decode() reverses it and is self-consistent (dequantizes the SNAPPED symbol it
stored). Pages are independently decodable (random access).

Timing: every stage accumulates into module timers. reset_timers() before a run,
timer_report(total_tokens, total_coord_symbols) after, for a stage breakdown +
analytic order.

Build from run_paged_split-style ladder (rungs finest-first):
    rungs_lh = [(delta_r[l,h], model_r[(l,h)]) for r in ladder]      # finest first
    codec = PageCodec(F[l,h], inv[l,h], k_mean[l,h], rungs_lh,
                      page_bits=b*d*P_tok, P_tok=P_tok, dz=dz)
    buf = codec.encode(K);  Khat = codec.decode(buf)
"""

import math
import struct
import time
from contextlib import contextmanager
from collections import defaultdict

import numpy as np
import constriction

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# --- timing -----------------------------------------------------------------
_T = defaultdict(float)
_N = defaultdict(int)
_ORDER = {
    "encode.transform":   "O(T*d^2)",
    "encode.quantize":    "O(U*T*d)  U=distinct rungs used (<=R)",
    "encode.snap":        "O(U*T*d) gather  [LUT, no searchsorted]",
    "encode.estimate":    "O(R*T) cheap  [step-shift, no snap]",
    "encode.rangecode":   "O(T*d), worst O(R*T*d) on overflow re-encode",
    "encode.assemble":    "O(T*d)",
    "decode.parse":       "O(npages)",
    "decode.rangedecode": "O(T*d)",
    "decode.dequant":     "O(T*d)",
    "decode.inverse":     "O(T*d^2)",
    "build.models":       "O(R*L*Hkv*d*A) one-time",
    "build.luts":         "O(R*L*Hkv*d*range) one-time",
}


@contextmanager
def _timed(name):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _T[name] += time.perf_counter() - t0
        _N[name] += 1


def reset_timers():
    _T.clear(); _N.clear()


def timer_report(total_tokens=0, total_coord_symbols=0):
    groups = [("BUILD (one-time, calibration)", "build"),
              ("ENCODE", "encode"), ("DECODE", "decode")]
    print("\n" + "=" * 78)
    print("COMPLEXITY / TIMING BREAKDOWN")
    print("=" * 78)
    enc_t = sum(v for k, v in _T.items() if k.startswith("encode"))
    dec_t = sum(v for k, v in _T.items() if k.startswith("decode"))
    for title, pre in groups:
        d = {k: v for k, v in _T.items() if k.startswith(pre)}
        if not d:
            continue
        tot = sum(d.values())
        print(f"\n{title} (total {tot:.4f}s)")
        print(f"  {'stage':<22} {'sec':>10} {'%':>6} {'ns/coord-sym':>13}  order")
        for k in sorted(d, key=lambda x: -d[x]):
            pct = 100 * d[k] / tot if tot else 0
            nspc = (d[k] / total_coord_symbols * 1e9) if total_coord_symbols else float("nan")
            print(f"  {k.split('.',1)[1]:<22} {d[k]:>10.4f} {pct:>5.1f}% {nspc:>13.1f}  {_ORDER.get(k,'')}")
    if enc_t and dec_t and total_tokens:
        print(f"\nthroughput: encode {total_tokens/enc_t:,.0f} tok/s | decode {total_tokens/dec_t:,.0f} tok/s")
    if total_coord_symbols:
        print(f"coord-symbols processed: {total_coord_symbols:,}  (= sum over heads of T*d)")
    print("=" * 78)


def _np(x, dtype=np.float64):
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().to("cpu").numpy().astype(dtype)
    return np.asarray(x, dtype=dtype)


def _dz_round(r, delta, dz):
    return np.sign(r) * np.floor(np.abs(r) / np.clip(delta, 1e-12, None) + dz)


def _dz_dequant(idx, delta, dz):
    return np.sign(idx) * (np.abs(idx) + (0.5 - dz)) * delta


# fixed-width bit packing for the per-page rung ids (block metadata) ----------
def _pack(ids, bits):
    out = bytearray(); acc = nacc = 0
    for v in ids:
        acc = (acc << bits) | int(v); nacc += bits
        while nacc >= 8:
            nacc -= 8; out.append((acc >> nacc) & 0xFF)
    if nacc:
        out.append((acc << (8 - nacc)) & 0xFF)
    return bytes(out)


def _unpack(buf, count, bits, off):
    out = np.zeros(count, dtype=np.int64)
    acc = nacc = 0; bi = off
    for i in range(count):
        while nacc < bits:
            acc = (acc << 8) | buf[bi]; bi += 1; nacc += 8
        nacc -= bits; out[i] = (acc >> nacc) & ((1 << bits) - 1)
    return out, off + (count * bits + 7) // 8


class _CoordModel:
    def __init__(self, vals, p):
        self.vals = np.asarray(vals, dtype=np.int64)
        self.constant = self.vals.size <= 1
        if self.constant:
            self.const_val = int(self.vals[0]) if self.vals.size else 0
            return
        p = np.asarray(p, dtype=np.float64); p = (p / p.sum()).clip(1e-12)
        self.p = p / p.sum()
        self.nlp = -np.log2(self.p)                       # bits per position
        self.model = constriction.stream.model.Categorical(self.p, perfect=False)

    def snap(self, q):
        v = self.vals
        pos = np.clip(np.searchsorted(v, q), 0, v.size - 1)
        left = np.clip(pos - 1, 0, v.size - 1)
        choose_left = np.abs(v[left] - q) <= np.abs(v[pos] - q)
        return np.where(choose_left, left, pos).astype(np.int32)


class PageCodec:
    """One (layer, kv-head). Rung ladder -> fixed-size pages. encode/decode."""
    MAGIC = 0x4B564333  # 'KVC3'

    def __init__(self, fwd, inv, mu, rungs, page_bits, P_tok, dz):
        # rungs: list of (delta_d, model_lh) finest-first
        self.fwd = _np(fwd); self.inv = _np(inv); self.mu = _np(mu).reshape(1, -1)
        self.d = self.fwd.shape[-1]
        self.dz = float(dz); self.P = int(P_tok)
        self.page_bits = float(page_bits)
        self.R = len(rungs)
        self.id_bits = max(1, int(math.ceil(math.log2(max(2, self.R)))))
        self.payload_budget_bits = self.page_bits - self.id_bits
        self.deltas = [_np(dl).reshape(-1) for (dl, _) in rungs]
        with _timed("build.models"):
            self.models = [[_CoordModel(v, p) for (v, p) in model] for (_, model) in rungs]
        # fix 3: per-rung dense nearest-bin LUTs, flattened across coords with offsets.
        # snap becomes a vectorized clip + gather (no per-coord searchsorted at runtime).
        with _timed("build.luts"):
            self.lut_lo, self.lut_hi, self.lut_off = [], [], []
            self.lut_pos, self.lut_nlp = [], []
            for ri in range(len(rungs)):
                lo, hi, off, fpos, fnlp = self._build_lut(self.models[ri])
                self.lut_lo.append(lo); self.lut_hi.append(hi); self.lut_off.append(off)
                self.lut_pos.append(fpos); self.lut_nlp.append(fnlp)
        # fix 2: reference rung = finest (index 0). Estimate coded bits at any rung
        # from the reference's snapped per-token bits via the high-rate step shift:
        #   bits_r(t) ~ bits_ref(t) - sum_j log2(delta_r[j]/delta_ref[j])  over non-const j.
        # Lets us pick a rung WITHOUT snapping every rung; the real-encode verify below
        # still guarantees the page fits.
        self.ref = 0
        dref = np.clip(self.deltas[self.ref], 1e-12, None)
        nonconst = np.array([not m.constant for m in self.models[self.ref]])
        self.shift = np.array([
            -np.sum(np.log2(np.clip(self.deltas[r], 1e-12, None)[nonconst] / dref[nonconst]))
            for r in range(self.R)])                  # (R,), shift[ref]=0
        self.rung_hist = np.zeros(self.R, dtype=np.int64)
        self.overflow = 0; self.nblocks = 0

    def _build_lut(self, models_r):
        """Dense nearest-position LUT per coord, concatenated with offsets. For
        index x in [lo_j, hi_j] (clamped outside), gives the nearest calib position
        and its -log2 p. Built once; searchsorted runs over the small dense range,
        not over tokens. Constant coords -> length-1 slot (pos 0, nlp 0)."""
        d = self.d
        lo = np.zeros(d, np.int64); hi = np.zeros(d, np.int64); off = np.zeros(d, np.int64)
        pos_parts, nlp_parts, cur = [], [], 0
        for j in range(d):
            m = models_r[j]
            if m.constant:
                lo[j] = hi[j] = m.const_val; off[j] = cur
                pos_parts.append(np.zeros(1, np.int32)); nlp_parts.append(np.zeros(1)); cur += 1
                continue
            v = m.vals; loj, hij = int(v[0]), int(v[-1])
            lo[j] = loj; hi[j] = hij; off[j] = cur
            x = np.arange(loj, hij + 1)
            ins = np.clip(np.searchsorted(v, x), 0, v.size - 1)
            left = np.clip(ins - 1, 0, v.size - 1)
            choose_left = np.abs(v[left] - x) <= np.abs(v[ins] - x)
            spos = np.where(choose_left, left, ins).astype(np.int32)
            pos_parts.append(spos); nlp_parts.append(m.nlp[spos]); cur += x.size
        return lo, hi, off, np.concatenate(pos_parts), np.concatenate(nlp_parts)

    def _snap_rung(self, r, ri, want_bits=False):
        """Quantize + snap the whole tensor at rung ri via the flat LUT (one gather).
        Returns pos (T,d) int32 and, if want_bits, per-token -log2 p estimate."""
        with _timed("encode.quantize"):
            q = _dz_round(r, self.deltas[ri], self.dz).astype(np.int64)
        with _timed("encode.snap"):
            lo, hi, off = self.lut_lo[ri], self.lut_hi[ri], self.lut_off[ri]
            fi = (np.clip(q, lo, hi) - lo) + off               # (T,d) into flat LUT
            pos = self.lut_pos[ri][fi]                          # (T,d) int32, vectorized
            b = self.lut_nlp[ri][fi].sum(1) if want_bits else None
        return (pos, b) if want_bits else pos

    def _encode_page(self, pos_ri, sl, ri):
        enc = constriction.stream.queue.RangeEncoder()
        for j in range(self.d):
            m = self.models[ri][j]
            if not m.constant:
                enc.encode(pos_ri[sl, j], m.model)
        return enc.get_compressed()

    def encode(self, k):
        k = _np(k)
        with _timed("encode.transform"):
            r = (k - self.mu) @ self.fwd
        T = r.shape[0]; P = self.P; nb = (T + P - 1) // P

        # fix 2: snap ONLY the reference rung; estimate every other rung by the shift.
        pos_cache = {}
        pos_cache[self.ref], b_ref = self._snap_rung(r, self.ref, want_bits=True)

        with _timed("encode.estimate"):
            est_tok = b_ref[None, :] + self.shift[:, None]      # (R,T) cheap
            page_est = np.empty((self.R, nb))
            for bi in range(nb):
                page_est[:, bi] = est_tok[:, bi * P:min((bi + 1) * P, T)].sum(1)
            fits = page_est <= self.payload_budget_bits
            chosen0 = np.where(fits.any(0), fits.argmax(0), self.R - 1)

        def _pos(ri):                                            # lazily snap a rung once
            if ri not in pos_cache:
                pos_cache[ri] = self._snap_rung(r, ri)
            return pos_cache[ri]

        rung_ids = np.empty(nb, dtype=np.int64); blobs = []
        for bi in range(nb):
            sl = slice(bi * P, min((bi + 1) * P, T))
            ri = int(chosen0[bi])
            while True:                                          # verify real fit, step coarser
                prc = _pos(ri)
                with _timed("encode.rangecode"):
                    blob = self._encode_page(prc, sl, ri)
                if self._blob_nbits(blob) <= self.payload_budget_bits or ri == self.R - 1:
                    break
                ri += 1
            if self._blob_nbits(blob) > self.payload_budget_bits:
                self.overflow += 1                               # even terminal didn't fit
            rung_ids[bi] = ri; blobs.append(blob)
            self.rung_hist[ri] += 1
        self.nblocks += nb

        with _timed("encode.assemble"):
            out = bytearray()
            out += struct.pack("<IIII", self.MAGIC, T, P, nb)
            out += _pack(rung_ids, self.id_bits)
            for blob in blobs:
                out += self._serialize_blob(blob)
        return bytes(out)

    # --- pluggable per-page coder (constriction here; rANS overrides) --------
    def _blob_nbits(self, words):
        return len(words) * 32                       # uint32 words

    def _serialize_blob(self, words):
        return struct.pack("<I", len(words)) + np.asarray(words, dtype="<u4").tobytes()

    def _read_blob(self, buf, off):
        (nw,) = struct.unpack_from("<I", buf, off); off += 4
        words = np.frombuffer(buf, dtype="<u4", count=nw, offset=off).copy()
        return words, off + nw * 4

    def _decode_page(self, blob, ri, n):
        """blob -> idx_page (n,d) of quantization-index VALUES."""
        idx = np.zeros((n, self.d), dtype=np.int64)
        dec = constriction.stream.queue.RangeDecoder(blob)
        for j in range(self.d):
            m = self.models[ri][j]
            if m.constant:
                idx[:, j] = m.const_val
            else:
                p = dec.decode(m.model, n)
                idx[:, j] = m.vals[np.asarray(p, dtype=np.int64)]
        return idx

    def decode(self, buf):
        with _timed("decode.parse"):
            magic, T, P, nb = struct.unpack_from("<IIII", buf, 0)
            assert magic == self.MAGIC, "bad codec stream"
            rung_ids, off = _unpack(buf, nb, self.id_bits, 16)
        idx = np.zeros((T, self.d), dtype=np.int64)
        with _timed("decode.rangedecode"):
            for bi in range(nb):
                blob, off = self._read_blob(buf, off)
                sl = slice(bi * P, min((bi + 1) * P, T)); n = sl.stop - sl.start
                idx[sl] = self._decode_page(blob, int(rung_ids[bi]), n)
        with _timed("decode.dequant"):
            r_hat = np.empty((T, self.d))
            for bi in range(nb):
                sl = slice(bi * P, min((bi + 1) * P, T))
                r_hat[sl] = _dz_dequant(idx[sl], self.deltas[int(rung_ids[bi])], self.dz)
        with _timed("decode.inverse"):
            k_hat = ((r_hat @ self.inv) + self.mu).astype(np.float32)
        return k_hat


# --- frozen config ----------------------------------------------------------
def build_codecs_from_ladder(F, inv, k_mean, ladder, n_layers, n_kv, page_bits, P_tok, dz):
    """ladder: list of (m, delta(L,Hkv,d), model_dict) finest-first."""
    codecs = {}
    for l in range(n_layers):
        for h in range(n_kv):
            rungs_lh = [(delta[l, h], model[(l, h)]) for (_, delta, model) in ladder]
            codecs[(l, h)] = PageCodec(F[l, h], inv[l, h], k_mean[l, h],
                                       rungs_lh, page_bits, P_tok, dz)
    return codecs


def save_config(path, F, inv, k_mean, ladder, meta):
    import pickle
    payload = dict(
        forward=_np(F, np.float32), inverse=_np(inv, np.float32),
        k_mean=_np(k_mean, np.float32),
        ladder=[(float(m), _np(delta, np.float32),
                 {k: [(np.asarray(v, np.int64), np.asarray(p, np.float64)) for (v, p) in per]
                  for k, per in model.items()})
                for (m, delta, model) in ladder],
        meta=meta)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_config(path, page_bits, P_tok, dz):
    import pickle
    with open(path, "rb") as f:
        c = pickle.load(f)
    L, Hkv = c["meta"]["n_layers"], c["meta"]["n_kv_heads"]
    return build_codecs_from_ladder(c["forward"], c["inverse"], c["k_mean"],
                                    c["ladder"], L, Hkv, page_bits, P_tok, dz)