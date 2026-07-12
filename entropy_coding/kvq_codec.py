#!/usr/bin/env python3
"""Fused KV codec stack for the QPCA-EC paged pipeline.

Three codecs in one file, finest-first rung ladder -> fixed-size pages:

  PageCodec          -- constriction range coder (reference; CPU).
  PageCodecRANS      -- same pipeline, per-page coder swapped to interleaved
                        rANS (rans_interleaved.py). CPU. Reconstructs the SAME
                        snapped indices -> bit-identical K_hat to PageCodec, so
                        it validates the rANS bitstream format with no CUDA.
  PageCodecRANSCUDA  -- GPU decode (rans_decode.cu) + GPU encode (rans_encode.cu).
                        Decode-identical to PageCodecRANS.

Drop-in for test_codec_on_data.py. External deps Amir must supply:
  rans_interleaved.py, rans_decode.cu, rans_encode.cu.

Timing: reset_timers() before a run, timer_report(total_tokens, total_coord_symbols)
after, for a stage breakdown + analytic order.
"""

import math
import struct
import time
from contextlib import contextmanager
from collections import defaultdict

import numpy as np
import constriction
import rans_interleaved as rans

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# rANS constants (must match rans_decode.cu / rans_encode.cu)
SCALE_BITS = 14
TOTAL = 1 << SCALE_BITS   # 16 384


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
    "build.rans_tables":  "O(R*L*Hkv*d*A) one-time",
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


_np_local = _np   # alias used by the CUDA encode paths below


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


# =========================================================================== #
# PageCodecRANS -- CPU interleaved-rANS per-page coder (was kvq_codec_rans.py) #
# =========================================================================== #
# Same external interface as PageCodec; reconstructs the SAME snapped indices ->
# identical K_hat. Use it to validate the exact GPU bitstream format on real data
# before any CUDA work: if PageCodecRANS matches PageCodec bit-for-bit on K_hat,
# the rANS format is correct and the kernel just has to reproduce decode_page.
# N = intra-page lanes. N=1 (default) = one rANS stream per page, lowest overhead,
# still fully parallel ACROSS pages. N>1 shortens the per-page serial chain at the
# cost of per-lane header/state (eats the fixed-page budget). Selection/fit/
# step-coarser are inherited unchanged; the real-byte fit check measures rANS bytes.

class PageCodecRANS(PageCodec):
    MAGIC = 0x4B564334  # 'KVC4' (distinct stream tag)

    def __init__(self, fwd, inv, mu, rungs, page_bits, P_tok, dz, lanes=1):
        super().__init__(fwd, inv, mu, rungs, page_bits, P_tok, dz)
        self.N = int(lanes)
        # rANS freq/cdf tables per (rung, coord), reusing the same frozen (vals,p).
        with _timed("build.rans_tables"):
            self.rtab = []
            for ri_ in range(self.R):
                row = []
                for m in self.models[ri_]:
                    if m.constant:
                        row.append(rans.FreqTable(m.vals, np.array([1.0])))
                    else:
                        row.append(rans.FreqTable(m.vals, m.p))
                self.rtab.append(row)

    # --- per-page coder overrides (bytes instead of uint32 words) -----------
    def _encode_page(self, pos_ri, sl, ridx):
        sub = pos_ri[sl]                                  # (n,d) positions
        n = sub.shape[0]
        return rans.encode_page(sub, self.rtab[ridx], n, self.d, self.N)

    def _blob_nbits(self, blob):
        return len(blob) * 8                              # raw bytes

    def _serialize_blob(self, blob):
        return struct.pack("<I", len(blob)) + blob

    def _read_blob(self, buf, off):
        (nb_,) = struct.unpack_from("<I", buf, off); off += 4
        return bytes(buf[off:off + nb_]), off + nb_

    def _decode_page(self, blob, ridx, n):
        positions = rans.decode_page(blob, self.rtab[ridx], n, self.d)
        return rans.positions_to_values(positions, self.rtab[ridx], n, self.d)


def build_codecs_from_ladder_rans(F, inv, k_mean, ladder, n_layers, n_kv,
                                  page_bits, P_tok, dz, lanes=1):
    codecs = {}
    for l in range(n_layers):
        for h in range(n_kv):
            rungs_lh = [(delta[l, h], model[(l, h)]) for (_, delta, model) in ladder]
            codecs[(l, h)] = PageCodecRANS(F[l, h], inv[l, h], k_mean[l, h],
                                           rungs_lh, page_bits, P_tok, dz, lanes=lanes)
    return codecs


# =========================================================================== #
# PageCodecRANSCUDA -- GPU decode + GPU encode (was kvq_codec_cuda.py)         #
# =========================================================================== #
# Encode + decode on GPU. K_hat is decode-identical to PageCodecRANS provided the
# CDF tables match the encoder's FreqTable. Needs rans_decode.cu / rans_encode.cu
# (compiled lazily by load_ext / load_enc_ext).

from torch.utils.cpp_extension import load as _cu_load  # noqa: E402  (torch required for CUDA path)


# --------------------------------------------------------------------------- #
# CDF helpers

def _quantize_freqs(p: np.ndarray) -> np.ndarray:
    """Round probability vector to integer frequencies summing to TOTAL.

    Uses largest-remainder (Hamilton) rounding -- the standard for rANS.
    Must produce the same result as rans_interleaved.FreqTable; if it doesn't,
    override _extract_freqs() below to read the pre-computed ints directly.
    """
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum()
    raw = p * TOTAL
    freqs = np.floor(raw).astype(np.int64)
    deficit = TOTAL - int(freqs.sum())
    if deficit:
        fracs = raw - np.floor(raw)
        freqs[np.argsort(-fracs)[:deficit]] += 1
    assert freqs.sum() == TOTAL
    assert (freqs > 0).all(), "zero-frequency symbol -- check your probability table"
    return freqs.astype(np.int32)


def _extract_freqs(ft) -> np.ndarray:
    """Pull integer frequencies (sum == TOTAL) from a FreqTable object."""
    for attr in ("freqs", "freq", "counts"):
        if hasattr(ft, attr):
            arr = np.asarray(getattr(ft, attr), dtype=np.int32)
            if arr.sum() == TOTAL:
                return arr
    if hasattr(ft, "cdf"):
        arr = np.asarray(ft.cdf, dtype=np.int32)
        diff = np.diff(arr)
        if diff.sum() == TOTAL:
            return diff.astype(np.int32)
    raise AttributeError(
        f"Cannot read integer freqs from FreqTable.  Attrs: "
        f"{[a for a in dir(ft) if not a.startswith('_')]}"
    )


# --------------------------------------------------------------------------- #

class PageCodecRANSCUDA(PageCodecRANS):
    MAGIC = 0x4B564334

    def __init__(self, fwd, inv, mu, rungs, page_bits, P_tok, dz,
                 lanes: int = 1, ext=None, enc_ext=None, device: str = "cuda"):
        # 1. Let the base class safely parse the raw ladder tuples into real objects first!
        super().__init__(fwd, inv, mu, rungs, page_bits, P_tok, dz, lanes)

        if ext is None:
            raise ValueError("Pass the compiled CUDA extension as ext=load_ext()")
        self.ext = ext
        self.enc_ext = enc_ext        # rans_encode extension (None until set)
        self.device = torch.device(device)

        self._cuda_cdf_tensors = []
        self._vals_gpu    = []
        self._nc_mask_gpu = []
        self._cv_gpu      = []
        self._deltas_gpu  = []
        self._bitcost     = []

        # 2. Vectorize table allocations per rung across all d coordinates at once
        for ri in range(self.R):
            models_r = self.models[ri]
            d = self.d

            # Find the maximum alphabet size for non-constant models in this rung
            A_max = 1
            for m in models_r:
                if not m.constant:
                    A_max = max(A_max, len(m.vals))

            nc_mask = np.zeros(d, dtype=bool)
            cv = np.zeros(d, dtype=np.int64)
            vals_padded = np.zeros((d, A_max), dtype=np.int64)

            # Collect non-constant data for batch quantization
            nc_indices = []
            p_list = []
            vals_list = []

            for j, m in enumerate(models_r):
                if m.constant:
                    cv[j] = m.vals[0] if (hasattr(m.vals, '__len__') and len(m.vals) > 0) else m.vals
                else:
                    nc_mask[j] = True
                    nc_indices.append(j)
                    p_list.append(m.p)
                    vals_list.append(m.vals)

            # Use the encoder's EXACT freq quantization so decode tables match.
            freqs_dict = {}
            for idx, j in enumerate(nc_indices):
                v_len = len(vals_list[idx])
                vals_padded[j, :v_len] = vals_list[idx]
                freqs_dict[j] = rans.normalize_freqs(p_list[idx]).astype(np.int32)

            # bit-cost LUT for rung selection proxy: cost[j, s] = SCALE_BITS - log2(freq)
            bitcost = np.zeros((d, A_max), dtype=np.float64)
            for j in range(d):
                if models_r[j].constant:
                    continue
                f = freqs_dict[j].astype(np.float64)
                bitcost[j, :len(f)] = SCALE_BITS - np.log2(f)   # f>=1 guaranteed by normalize_freqs
            self._bitcost.append(bitcost)

            # 3. Construct the flat CDF layouts
            nonconst = np.array(nc_indices, dtype=np.int32)
            parts = []
            cdf_off = np.zeros(d + 1, dtype=np.int32)

            for j in range(d):
                if models_r[j].constant:
                    cdf_off[j + 1] = cdf_off[j]
                    continue

                freqs = freqs_dict[j]
                cdf = np.zeros(len(freqs) + 1, dtype=np.int32)
                cdf[1:] = np.cumsum(freqs)
                parts.append(cdf)
                cdf_off[j + 1] = cdf_off[j] + len(cdf)

            cdf_flat = (np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)).astype(np.int32)

            # 4. Push directly to GPU tensors (Zero loop overhead)
            self._cuda_cdf_tensors.append((
                torch.as_tensor(nonconst, dtype=torch.int32, device=self.device),
                torch.as_tensor(cdf_flat, dtype=torch.int32, device=self.device),
                torch.as_tensor(cdf_off, dtype=torch.int32, device=self.device),
            ))

            self._vals_gpu.append(torch.as_tensor(vals_padded, dtype=torch.int64, device=self.device))
            self._nc_mask_gpu.append(torch.as_tensor(nc_mask, dtype=torch.bool, device=self.device))
            self._cv_gpu.append(torch.as_tensor(cv, dtype=torch.int64, device=self.device))
            self._deltas_gpu.append(torch.as_tensor(self.deltas[ri].reshape(1, -1), dtype=torch.float32, device=self.device))

        self._inv_gpu = torch.as_tensor(self.inv, dtype=torch.float32, device=self.device)
        self._mu_gpu  = torch.as_tensor(self.mu,  dtype=torch.float32, device=self.device)

        # cache GPU LUT tensors for the encoder (built once, reused per encode)
        self._enc_lut = {}

        # Re-map legacy fallback paths if validation requires them
        self._vals = [
            [None if m.constant else m.vals for m in self.models[ri]]
            for ri in range(self.R)
        ]
        self._const_val = [
            [m.vals[0] if m.constant else None for m in self.models[ri]]
            for ri in range(self.R)
        ]

    def _enc_Lt(self, ri_):
        """Lazily build + cache this rung's GPU LUT tensors. Cleared per head."""
        c = self._enc_lut.get(ri_)
        if c is None:
            c = dict(
                delta=torch.as_tensor(self.deltas[ri_], dtype=torch.float64, device=self.device),
                lo=torch.as_tensor(self.lut_lo[ri_], dtype=torch.int64, device=self.device),
                hi=torch.as_tensor(self.lut_hi[ri_], dtype=torch.int64, device=self.device),
                off=torch.as_tensor(self.lut_off[ri_], dtype=torch.int64, device=self.device),
                lut_pos=torch.as_tensor(self.lut_pos[ri_], dtype=torch.int64, device=self.device),
                lut_nlp=torch.as_tensor(self.lut_nlp[ri_], dtype=torch.float64, device=self.device),
                bc=torch.as_tensor(self._bitcost[ri_], dtype=torch.float64, device=self.device),
                nc=self._cuda_cdf_tensors[ri_][0].to(torch.int64),
                cdf=self._cuda_cdf_tensors[ri_][1].to(torch.int64),
                coff=self._cuda_cdf_tensors[ri_][2].to(torch.int64),
            )
            self._enc_lut[ri_] = c
        return c

    def _enc_clear(self):
        """Free this head's cached encoder LUTs."""
        self._enc_lut.clear()

    def _build_rung_cdf(self, ri: int):
        models_r = self.models[ri]
        meta = self.precomputed_data[ri]

        nonconst = [j for j, m in enumerate(models_r) if not m.constant]
        d = self.d
        parts = []
        cdf_off = np.zeros(d + 1, dtype=np.int32)

        for j in range(d):
            m = models_r[j]
            if m.constant:
                cdf_off[j + 1] = cdf_off[j]
                continue

            # Bypass _extract_freqs() and _quantize_freqs() entirely!
            freqs = meta['freqs'][j]

            cdf = np.zeros(len(freqs) + 1, dtype=np.int32)
            cdf[1:] = np.cumsum(freqs)
            parts.append(cdf)
            cdf_off[j + 1] = cdf_off[j] + len(cdf)

        cdf_flat = (np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)).astype(np.int32)
        return np.array(nonconst, dtype=np.int32), cdf_flat, cdf_off

    def decode_to_gpu(self, buf: bytes) -> torch.Tensor:
        """Decode buffer, keep k_hat on GPU. Logs kernel-only CUDA events into
        self._kernel_events (no sync here -- decode_grid syncs once at the end)."""
        buf = bytes(buf)

        if not hasattr(self, "_kernel_events"):
            self._kernel_events = []

        # 1. Parse stream header
        magic, T, Phdr, nb = struct.unpack_from("<IIII", buf, 0)
        rung_ids, off0 = _unpack(buf, nb, self.id_bits, 16)

        # 2. Read all page blobs
        blobs_meta = []
        off = off0
        for bi in range(nb):
            blob, off = self._read_blob(buf, off)
            n = min((bi + 1) * Phdr, T) - bi * Phdr
            blobs_meta.append((blob, int(rung_ids[bi]), n))

        idx_gpu = torch.zeros((T, self.d), dtype=torch.int64, device=self.device)
        rung_groups = defaultdict(list)
        for bi, (blob, ri, n) in enumerate(blobs_meta):
            rung_groups[ri].append((bi, blob, n))

        # 3. Group pages by rung and call kernel
        for ri, pages in rung_groups.items():
            nc_t, cf_t, co_t = self._cuda_cdf_tensors[ri]
            C = int(nc_t.shape[0])
            n_pg = len(pages)
            dev = self.device

            all_bytes = b"".join(blob for _, blob, _ in pages)
            blob_arr = np.frombuffer(all_bytes, dtype=np.uint8).copy()

            page_byte_off = np.zeros(n_pg, dtype=np.int64)
            lane_off_arr = np.zeros((n_pg, self.N), dtype=np.int64)
            k0a = np.zeros((n_pg, self.N), dtype=np.int32)
            k1a = np.zeros((n_pg, self.N), dtype=np.int32)
            ns_arr = np.zeros(n_pg, dtype=np.int32)

            cumoff = 0
            for pi, (bi, blob, n_toks) in enumerate(pages):
                S = n_toks * C
                Nh, Sb = struct.unpack_from("<II", blob, 0)
                lane_lens = [struct.unpack_from("<I", blob, 8 + 4 * l)[0] for l in range(self.N)]
                starts = np.cumsum([0] + lane_lens)
                bounds = [round(k * S / self.N) for k in range(self.N + 1)]
                page_byte_off[pi] = cumoff
                ns_arr[pi] = n_toks
                header_size = 8 + 4 * self.N
                for l in range(self.N):
                    lane_off_arr[pi, l] = header_size + int(starts[l])
                    k0a[pi, l] = bounds[l]
                    k1a[pi, l] = bounds[l + 1]
                cumoff += len(blob)

            def _T(a, dt): return torch.as_tensor(a, dtype=dt, device=dev)

            # --- time ONLY the entropy-decode kernel (CUDA events, no sync) ---
            _ks = torch.cuda.Event(enable_timing=True)
            _ke = torch.cuda.Event(enable_timing=True)
            _ks.record()
            pos_gpu = self.ext.decode_pages(
                _T(blob_arr, torch.uint8), _T(page_byte_off, torch.int64),
                _T(lane_off_arr.reshape(-1), torch.int64), _T(k0a.reshape(-1), torch.int32),
                _T(k1a.reshape(-1), torch.int32), _T(ns_arr, torch.int32),
                nc_t, cf_t, co_t, self.N, C, self.P, self.d,
            )
            _ke.record()
            self._kernel_events.append((_ks, _ke))

            vals_t = self._vals_gpu[ri]
            nc_mask = self._nc_mask_gpu[ri]
            cv_t = self._cv_gpu[ri]

            j_idx = torch.arange(self.d, device=dev)
            flat = pos_gpu.reshape(-1, self.d).long()        # (n_pg*P, d) all pages at once
            gathered = vals_t[j_idx.unsqueeze(0), flat]      # (n_pg*P, d) single gather
            gathered[:, ~nc_mask] = cv_t[~nc_mask]           # const coords -> const value
            gathered = gathered.reshape(n_pg, self.P, self.d)
            for pi, (bi, _, n_toks) in enumerate(pages):
                idx_gpu[bi * self.P: bi * self.P + n_toks] = gathered[pi, :n_toks]

        # 4. Dequantise + inverse transform -- float32 (cached device tensors)
        delta_tok = torch.empty((T, self.d), dtype=torch.float32, device=self.device)
        for bi in range(nb):
            ri = int(rung_ids[bi])
            sl = slice(bi * self.P, min((bi + 1) * self.P, T))
            delta_tok[sl] = self._deltas_gpu[ri]
        idx_f = idx_gpu.float()
        r_hat = idx_f.sign() * (idx_f.abs() + (0.5 - self.dz)) * delta_tok
        return r_hat @ self._inv_gpu + self._mu_gpu

    # ----------------------------------------------------------------------- #
    # Decode override

    def decode(self, buf: bytes) -> np.ndarray:
        buf = bytes(buf)

        # 1. Parse stream header ------------------------------------------------
        with _timed("decode.parse"):
            magic, T, Phdr, nb = struct.unpack_from("<IIII", buf, 0)
            assert magic == self.MAGIC, (
                f"stream magic {magic:#010x} != expected {self.MAGIC:#010x}")
            rung_ids, off0 = _unpack(buf, nb, self.id_bits, 16)

        # 2. Read all page blobs ------------------------------------------------
        blobs_meta = []   # list of (blob: bytes, ri: int, n_toks: int)
        off = off0
        for bi in range(nb):
            blob, off = self._read_blob(buf, off)
            n = min((bi + 1) * Phdr, T) - bi * Phdr
            blobs_meta.append((blob, int(rung_ids[bi]), n))

        # 3. Group pages by rung and call kernel once per group -----------------
        idx_gpu = torch.zeros((T, self.d), dtype=torch.int64, device=self.device)

        rung_groups: dict[int, list[tuple[int, bytes, int]]] = defaultdict(list)
        for bi, (blob, ri, n) in enumerate(blobs_meta):
            rung_groups[ri].append((bi, blob, n))

        with _timed("decode.rangedecode"):
            for ri, pages in rung_groups.items():
                nc_t, cf_t, co_t = self._cuda_cdf_tensors[ri]
                C    = int(nc_t.shape[0])
                n_pg = len(pages)
                dev  = self.device

                # -- concatenate raw blob bytes ---------------------------------
                all_bytes = b"".join(blob for _, blob, _ in pages)
                blob_arr  = np.frombuffer(all_bytes, dtype=np.uint8).copy()

                # -- per-page metadata arrays ----------------------------------
                page_byte_off = np.zeros(n_pg, dtype=np.int64)
                lane_off_arr  = np.zeros((n_pg, self.N), dtype=np.int64)
                k0a           = np.zeros((n_pg, self.N), dtype=np.int32)
                k1a           = np.zeros((n_pg, self.N), dtype=np.int32)
                ns_arr        = np.zeros(n_pg, dtype=np.int32)

                cumoff = 0
                for pi, (bi, blob, n_toks) in enumerate(pages):
                    S = n_toks * C
                    Nh, Sb = struct.unpack_from("<II", blob, 0)
                    if Nh != self.N:
                        raise ValueError(f"rung {ri} page {bi}: blob N={Nh}, codec N={self.N}")
                    if Sb != S:
                        raise ValueError(
                            f"rung {ri} page {bi}: blob S={Sb}, expected {S} "
                            f"(n_toks={n_toks}, C={C})")
                    lane_lens = [struct.unpack_from("<I", blob, 8 + 4 * l)[0]
                                 for l in range(self.N)]
                    starts = np.cumsum([0] + lane_lens)
                    bounds = [round(k * S / self.N) for k in range(self.N + 1)]
                    page_byte_off[pi] = cumoff
                    ns_arr[pi]        = n_toks
                    header_size = 8 + 4 * self.N
                    for l in range(self.N):
                        lane_off_arr[pi, l] = header_size + int(starts[l])
                        k0a[pi, l]          = bounds[l]
                        k1a[pi, l]          = bounds[l + 1]
                    cumoff += len(blob)

                def _T(a, dt):
                    return torch.as_tensor(a, dtype=dt, device=dev)

                # -- kernel call -- pos_gpu stays on GPU ------------------------
                pos_gpu = self.ext.decode_pages(
                    _T(blob_arr,                 torch.uint8),
                    _T(page_byte_off,            torch.int64),
                    _T(lane_off_arr.reshape(-1), torch.int64),
                    _T(k0a.reshape(-1),          torch.int32),
                    _T(k1a.reshape(-1),          torch.int32),
                    _T(ns_arr,                   torch.int32),
                    nc_t, cf_t, co_t,
                    self.N, C, self.P, self.d,
                )  # (n_pg, P, d) int32

                # -- position -> quantisation-index values (GPU gather) ---------
                vals_t   = self._vals_gpu[ri]    # (d, A_max) int64, pre-padded
                nc_mask  = self._nc_mask_gpu[ri] # (d,) bool -- True for non-const
                cv_t     = self._cv_gpu[ri]      # (d,) int64 -- const values

                for pi, (bi, _, n_toks) in enumerate(pages):
                    pos_page = pos_gpu[pi, :n_toks].long()  # (n_toks, d)
                    sl_start = bi * self.P
                    sl_end   = sl_start + n_toks

                    j_idx = torch.arange(self.d, device=dev)                  # (d,)
                    gathered = vals_t[j_idx.unsqueeze(1),                     # (d, n_toks)
                                      pos_page.t()]
                    gathered[~nc_mask] = cv_t[~nc_mask].unsqueeze(1).expand(
                        (~nc_mask).sum(), n_toks)
                    idx_gpu[sl_start:sl_end] = gathered.t()                   # (n_toks, d)

        # 4. Dequantise and inverse transform -- entirely on GPU ----------------
        with _timed("decode.dequant"):
            delta_tok = torch.empty((T, self.d), dtype=torch.float64, device=self.device)
            for bi in range(nb):
                ri  = int(rung_ids[bi])
                sl  = slice(bi * self.P, min((bi + 1) * self.P, T))
                delta_tok[sl] = self._deltas_gpu[ri]   # (d,) broadcast

            idx_f  = idx_gpu.double()
            sign   = idx_f.sign()
            r_hat  = sign * (idx_f.abs() + (0.5 - self.dz)) * delta_tok     # (T, d)

        with _timed("decode.inverse"):
            inv_t  = torch.as_tensor(self.inv, dtype=torch.float64, device=self.device)
            mu_t   = torch.as_tensor(self.mu,  dtype=torch.float64, device=self.device)
            k_hat  = (r_hat @ inv_t + mu_t).float().cpu().numpy()            # one transfer out

        return k_hat

    def decode_to_rhat(self, buf: bytes) -> torch.Tensor:
        """Decode to the residual domain r̂ — STOP before inverse+mu. Fused
        residual-domain attention applies inv implicitly via q' = q@invᵀ, so the
        per-key inverse never runs and fp16 K is never formed."""
        buf = bytes(buf)
        if not hasattr(self, "_kernel_events"):
            self._kernel_events = []
        magic, T, Phdr, nb = struct.unpack_from("<IIII", buf, 0)
        rung_ids, off0 = _unpack(buf, nb, self.id_bits, 16)
        blobs_meta = []
        off = off0
        for bi in range(nb):
            blob, off = self._read_blob(buf, off)
            n = min((bi + 1) * Phdr, T) - bi * Phdr
            blobs_meta.append((blob, int(rung_ids[bi]), n))

        idx_gpu = torch.zeros((T, self.d), dtype=torch.int64, device=self.device)
        rung_groups = defaultdict(list)
        for bi, (blob, ri, n) in enumerate(blobs_meta):
            rung_groups[ri].append((bi, blob, n))

        for ri, pages in rung_groups.items():
            nc_t, cf_t, co_t = self._cuda_cdf_tensors[ri]
            C = int(nc_t.shape[0]); n_pg = len(pages); dev = self.device
            all_bytes = b"".join(blob for _, blob, _ in pages)
            blob_arr = np.frombuffer(all_bytes, dtype=np.uint8).copy()
            page_byte_off = np.zeros(n_pg, dtype=np.int64)
            lane_off_arr = np.zeros((n_pg, self.N), dtype=np.int64)
            k0a = np.zeros((n_pg, self.N), dtype=np.int32)
            k1a = np.zeros((n_pg, self.N), dtype=np.int32)
            ns_arr = np.zeros(n_pg, dtype=np.int32)
            cumoff = 0
            for pi, (bi, blob, n_toks) in enumerate(pages):
                S = n_toks * C
                Nh, Sb = struct.unpack_from("<II", blob, 0)
                lane_lens = [struct.unpack_from("<I", blob, 8 + 4 * l)[0] for l in range(self.N)]
                starts = np.cumsum([0] + lane_lens)
                bounds = [round(k * S / self.N) for k in range(self.N + 1)]
                page_byte_off[pi] = cumoff; ns_arr[pi] = n_toks
                header_size = 8 + 4 * self.N
                for l in range(self.N):
                    lane_off_arr[pi, l] = header_size + int(starts[l])
                    k0a[pi, l] = bounds[l]; k1a[pi, l] = bounds[l + 1]
                cumoff += len(blob)
            def _T(a, dt): return torch.as_tensor(a, dtype=dt, device=dev)
            _ks = torch.cuda.Event(enable_timing=True); _ke = torch.cuda.Event(enable_timing=True)
            _ks.record()
            pos_gpu = self.ext.decode_pages(
                _T(blob_arr, torch.uint8), _T(page_byte_off, torch.int64),
                _T(lane_off_arr.reshape(-1), torch.int64), _T(k0a.reshape(-1), torch.int32),
                _T(k1a.reshape(-1), torch.int32), _T(ns_arr, torch.int32),
                nc_t, cf_t, co_t, self.N, C, self.P, self.d,
            )
            _ke.record(); self._kernel_events.append((_ks, _ke))
            vals_t = self._vals_gpu[ri]; nc_mask = self._nc_mask_gpu[ri]; cv_t = self._cv_gpu[ri]
            j_idx = torch.arange(self.d, device=dev)
            flat = pos_gpu.reshape(-1, self.d).long()
            gathered = vals_t[j_idx.unsqueeze(0), flat]
            gathered[:, ~nc_mask] = cv_t[~nc_mask]
            gathered = gathered.reshape(n_pg, self.P, self.d)
            for pi, (bi, _, n_toks) in enumerate(pages):
                idx_gpu[bi * self.P: bi * self.P + n_toks] = gathered[pi, :n_toks]

        # dequant only — STOP HERE, no inverse, no mu
        delta_tok = torch.empty((T, self.d), dtype=torch.float32, device=self.device)
        for bi in range(nb):
            ri = int(rung_ids[bi])
            sl = slice(bi * self.P, min((bi + 1) * self.P, T))
            delta_tok[sl] = self._deltas_gpu[ri]
        idx_f = idx_gpu.float()
        r_hat = idx_f.sign() * (idx_f.abs() + (0.5 - self.dz)) * delta_tok
        return r_hat          # (T, d) residual domain    # ----------------------------------------------------------------------- #
    # Optional sanity check

    def decode_to_idx(self, buf: bytes):
        """Decode to quantized indices (int8) + per-token scalar delta. STOP before
        dequant — the kernel dequantizes in-register: r̂ = sign(idx)·(|idx|+0.5−dz)·δ.
        Returns (idx_i8 (T,d) int8, delta_tok (T,) float32)."""
        buf = bytes(buf)
        if not hasattr(self, "_kernel_events"):
            self._kernel_events = []
        magic, T, Phdr, nb = struct.unpack_from("<IIII", buf, 0)
        rung_ids, off0 = _unpack(buf, nb, self.id_bits, 16)
        blobs_meta = []; off = off0
        for bi in range(nb):
            blob, off = self._read_blob(buf, off)
            n = min((bi + 1) * Phdr, T) - bi * Phdr
            blobs_meta.append((blob, int(rung_ids[bi]), n))
        idx_gpu = torch.zeros((T, self.d), dtype=torch.int64, device=self.device)
        rung_groups = defaultdict(list)
        for bi, (blob, ri, n) in enumerate(blobs_meta):
            rung_groups[ri].append((bi, blob, n))
        for ri, pages in rung_groups.items():
            nc_t, cf_t, co_t = self._cuda_cdf_tensors[ri]
            C = int(nc_t.shape[0]); n_pg = len(pages); dev = self.device
            all_bytes = b"".join(blob for _, blob, _ in pages)
            blob_arr = np.frombuffer(all_bytes, dtype=np.uint8).copy()
            page_byte_off = np.zeros(n_pg, dtype=np.int64)
            lane_off_arr = np.zeros((n_pg, self.N), dtype=np.int64)
            k0a = np.zeros((n_pg, self.N), dtype=np.int32)
            k1a = np.zeros((n_pg, self.N), dtype=np.int32)
            ns_arr = np.zeros(n_pg, dtype=np.int32)
            cumoff = 0
            for pi, (bi, blob, n_toks) in enumerate(pages):
                S = n_toks * C
                Nh, Sb = struct.unpack_from("<II", blob, 0)
                lane_lens = [struct.unpack_from("<I", blob, 8 + 4 * l)[0] for l in range(self.N)]
                starts = np.cumsum([0] + lane_lens)
                bounds = [round(k * S / self.N) for k in range(self.N + 1)]
                page_byte_off[pi] = cumoff; ns_arr[pi] = n_toks
                header_size = 8 + 4 * self.N
                for l in range(self.N):
                    lane_off_arr[pi, l] = header_size + int(starts[l])
                    k0a[pi, l] = bounds[l]; k1a[pi, l] = bounds[l + 1]
                cumoff += len(blob)
            def _T(a, dt): return torch.as_tensor(a, dtype=dt, device=dev)
            _ks = torch.cuda.Event(enable_timing=True); _ke = torch.cuda.Event(enable_timing=True)
            _ks.record()
            pos_gpu = self.ext.decode_pages(
                _T(blob_arr, torch.uint8), _T(page_byte_off, torch.int64),
                _T(lane_off_arr.reshape(-1), torch.int64), _T(k0a.reshape(-1), torch.int32),
                _T(k1a.reshape(-1), torch.int32), _T(ns_arr, torch.int32),
                nc_t, cf_t, co_t, self.N, C, self.P, self.d,
            )
            _ke.record(); self._kernel_events.append((_ks, _ke))
            vals_t = self._vals_gpu[ri]; nc_mask = self._nc_mask_gpu[ri]; cv_t = self._cv_gpu[ri]
            j_idx = torch.arange(self.d, device=dev)
            flat = pos_gpu.reshape(-1, self.d).long()
            gathered = vals_t[j_idx.unsqueeze(0), flat]
            gathered[:, ~nc_mask] = cv_t[~nc_mask]
            gathered = gathered.reshape(n_pg, self.P, self.d)
            for pi, (bi, _, n_toks) in enumerate(pages):
                idx_gpu[bi * self.P: bi * self.P + n_toks] = gathered[pi, :n_toks]
        # per-token scalar delta (delta is scalar per rung here)
        delta_tok = torch.empty((T,), dtype=torch.float32, device=self.device)
        for bi in range(nb):
            ri = int(rung_ids[bi])
            sl = slice(bi * self.P, min((bi + 1) * self.P, T))
            delta_tok[sl] = float(self._deltas_gpu[ri].reshape(-1)[0])
        # idx fits int8 (verified |idx|<=~53 at b=2); assert to be safe
        # assert disabled for timing (forces device sync)
        return idx_gpu.to(torch.int16), delta_tok
    
    def validate_one_page(self, buf: bytes, page_idx: int = 0) -> bool:
        """Decode a single page with both CPU and GPU paths and compare K_hat."""
        magic, T, Phdr, nb = struct.unpack_from("<IIII", buf, 0)
        rung_ids, off0 = _unpack(buf, nb, self.id_bits, 16)

        off = off0
        for bi in range(nb):
            blob, off = self._read_blob(buf, off)
            if bi == page_idx:
                ri    = int(rung_ids[bi])
                n     = min((bi + 1) * Phdr, T) - bi * Phdr
                break

        # CPU path (constriction)
        cpu_idx = self._decode_page(blob, ri, n)
        cpu_r   = _dz_dequant(cpu_idx, self.deltas[ri], self.dz)
        cpu_k   = ((cpu_r @ self.inv) + self.mu).astype(np.float32)

        # GPU path (kernel, single page)
        nc_t, cf_t, co_t = self._cuda_cdf_tensors[ri]
        C  = int(nc_t.shape[0])
        S  = n * C
        Nh, Sb = struct.unpack_from("<II", blob, 0)

        blob_arr = np.frombuffer(blob, dtype=np.uint8).copy()
        lane_lens = [struct.unpack_from("<I", blob, 8 + 4 * l)[0] for l in range(self.N)]
        starts = np.cumsum([0] + lane_lens)
        bounds = [round(k * S / self.N) for k in range(self.N + 1)]

        header_size  = 8 + 4 * self.N
        lane_off_arr = np.zeros((1, self.N), dtype=np.int64)
        k0a          = np.zeros((1, self.N), dtype=np.int32)
        k1a          = np.zeros((1, self.N), dtype=np.int32)
        for l in range(self.N):
            lane_off_arr[0, l] = header_size + int(starts[l])
            k0a[0, l]          = bounds[l]
            k1a[0, l]          = bounds[l + 1]

        dev = self.device
        def _T(a, dt): return torch.as_tensor(a, dtype=dt, device=dev)

        pos_gpu = self.ext.decode_pages(
            _T(blob_arr, torch.uint8),
            _T(np.zeros(1, np.int64), torch.int64),
            _T(lane_off_arr.reshape(-1), torch.int64),
            _T(k0a.reshape(-1), torch.int32),
            _T(k1a.reshape(-1), torch.int32),
            _T(np.array([n], np.int32), torch.int32),
            nc_t, cf_t, co_t,
            self.N, C, self.P, self.d,
        )
        pos_np = pos_gpu.cpu().numpy()[0, :n]

        gpu_idx = np.zeros((n, self.d), dtype=np.int64)
        for j in range(self.d):
            vals = self._vals[ri][j]
            if vals is None:
                gpu_idx[:, j] = self._const_val[ri][j]
            else:
                gpu_idx[:, j] = vals[pos_np[:, j]]

        gpu_r = _dz_dequant(gpu_idx, self.deltas[ri], self.dz)
        gpu_k = ((gpu_r @ self.inv) + self.mu).astype(np.float32)

        match = np.array_equal(cpu_k, gpu_k)
        if not match:
            diff = np.abs(cpu_k - gpu_k)
            print(f"validate_one_page FAIL: max diff={diff.max():.3e}, "
                  f"n_mismatch={(cpu_k != gpu_k).sum()}/{cpu_k.size}")
        else:
            print(f"validate_one_page OK: page {page_idx} rung {ri} n={n} -- CPU==GPU")
        return match

    def _proxy_bytes(self, pos, ri, n):
        bc = self._bitcost[ri]                      # (d, A_max)
        rows = np.arange(self.d)
        payload_bits = bc[rows[None, :], pos].sum() # ideal coded bits
        header = 8 + 4 * self.N                      # <N><S><lane_len[N]>
        overhead_bytes = header + 4 * self.N + self.N  # 4B state/lane + <=1B round/lane
        return overhead_bytes + payload_bits / 8.0

    def encode(self, k):
        k = _np_local(k)
        with _timed("encode.transform"):
            r = (k - self.mu) @ self.fwd
        T = r.shape[0]; P = self.P; nb = (T + P - 1) // P

        # snap only the reference rung; estimate others by the step shift
        pos_cache = {}
        pos_cache[self.ref], b_ref = self._snap_rung(r, self.ref, want_bits=True)

        with _timed("encode.estimate"):
            est_tok = b_ref[None, :] + self.shift[:, None]
            page_est = np.empty((self.R, nb))
            for bi in range(nb):
                page_est[:, bi] = est_tok[:, bi * P:min((bi + 1) * P, T)].sum(1)
            fits = page_est <= self.payload_budget_bits
            chosen0 = np.where(fits.any(0), fits.argmax(0), self.R - 1)

        def _pos(ridx):
            if ridx not in pos_cache:
                pos_cache[ridx] = self._snap_rung(r, ridx)
            return pos_cache[ridx]

        rung_ids = np.empty(nb, dtype=np.int64); blobs = []
        for bi in range(nb):
            sl = slice(bi * P, min((bi + 1) * P, T))
            ri_ = int(chosen0[bi])
            # PROXY fit check: climb the ladder using the analytic byte estimate,
            # NOT a real encode. Only the winner gets range-encoded (once).
            while ri_ < self.R - 1:
                prc = _pos(ri_)
                proxy_bits = self._proxy_bytes(prc[sl], ri_, sl.stop - sl.start) * 8.0
                if proxy_bits <= self.payload_budget_bits:
                    break
                ri_ += 1
            prc = _pos(ri_)
            with _timed("encode.rangecode"):
                blob = self._encode_page(prc, sl, ri_)
            # real backstop: if the proxy under-predicted and we overflowed, climb for real
            while self._blob_nbits(blob) > self.payload_budget_bits and ri_ < self.R - 1:
                ri_ += 1
                with _timed("encode.rangecode"):
                    blob = self._encode_page(_pos(ri_), sl, ri_)
            if self._blob_nbits(blob) > self.payload_budget_bits:
                self.overflow += 1
            rung_ids[bi] = ri_; blobs.append(blob)
            self.rung_hist[ri_] += 1
        self.nblocks += nb

        with _timed("encode.assemble"):
            out = bytearray()
            out += struct.pack("<IIII", self.MAGIC, T, P, nb)
            out += _pack(rung_ids, self.id_bits)
            for blob in blobs:
                out += self._serialize_blob(blob)
        return bytes(out)

    def encode_gpu(self, k):
        """Full GPU encode. Decode-identical to encode(). Requires self.enc_ext."""
        if self.enc_ext is None:
            raise ValueError("enc_ext not set; pass enc_ext=load_enc_ext() to the factory")
        dev = self.device; P = self.P; R = self.R; N = self.N; d = self.d; dz = self.dz
        kd = torch.as_tensor(_np_local(k), dtype=torch.float64, device=dev)
        T = kd.shape[0]; nb = (T + P - 1) // P; pad = nb * P - T
        mu = torch.as_tensor(self.mu, dtype=torch.float64, device=dev)
        fwd = torch.as_tensor(self.fwd, dtype=torch.float64, device=dev)
        r = (kd - mu) @ fwd
        budget = self.payload_budget_bits
        overhead_bits = 8 * (8 + 4 * N + 4 * N + N)

        def Lt(ri_):
            return self._enc_Lt(ri_)

        def pos_at(ri_):
            t = Lt(ri_)
            q = (torch.sign(r) * torch.floor(torch.abs(r) / t['delta'] + dz)).to(torch.int64)
            fi = (torch.clamp(q, t['lo'][None, :], t['hi'][None, :]) - t['lo'][None, :]) + t['off'][None, :]
            return t['lut_pos'][fi]

        # selection
        tref = Lt(self.ref)
        qr = (torch.sign(r) * torch.floor(torch.abs(r) / tref['delta'] + dz)).to(torch.int64)
        fir = (torch.clamp(qr, tref['lo'][None, :], tref['hi'][None, :]) - tref['lo'][None, :]) + tref['off'][None, :]
        b_ref = tref['lut_nlp'][fir].sum(1)
        shift = torch.as_tensor(self.shift, dtype=torch.float64, device=dev)
        est = b_ref[None, :] + shift[:, None]
        if pad: est = torch.cat([est, torch.zeros((R, pad), dtype=torch.float64, device=dev)], 1)
        page_est = est.reshape(R, nb, P).sum(2)
        fits = (page_est + overhead_bits) <= budget
        chosen0 = torch.where(fits.any(0), fits.float().argmax(0),
                              torch.full((nb,), R - 1, device=dev)).to(torch.int64)

        def proxy_pb(ri_):
            t = Lt(ri_); pos = pos_at(ri_); jidx = torch.arange(d, device=dev)
            tb = t['bc'][jidx[None, :], pos].sum(1)
            if pad: tb = torch.cat([tb, torch.zeros(pad, dtype=torch.float64, device=dev)])
            return tb.reshape(nb, P).sum(1)

        rung = chosen0.clone(); ppb = {}
        for _ in range(R):
            need = torch.zeros(nb, dtype=torch.bool, device=dev)
            for rr in torch.unique(rung).tolist():
                if rr not in ppb: ppb[rr] = proxy_pb(rr)
                need |= ((rung == rr) & ((ppb[rr] + overhead_bits) > budget) & (rung < R - 1))
            if not bool(need.any()): break
            rung = torch.where(need, torch.clamp(rung + 1, max=R - 1), rung)

        # encode + real-byte backstop
        page_blobs = [None] * nb
        pending = list(range(nb))
        for _it in range(R + 1):
            rung_np = rung.cpu().numpy()
            by_rung = {}
            for bi in pending:
                by_rung.setdefault(int(rung_np[bi]), []).append(bi)
            redo = []
            for ri_, pages in by_rung.items():
                t = Lt(ri_); pos = pos_at(ri_)
                nc = t['nc']; C = int(nc.numel()); cdf = t['cdf']; coff = t['coff']
                freq_parts = []; start_parts = []; sym_off = []; metas = []; cur = 0
                for bi in pages:
                    s0 = bi * P; n = min((bi + 1) * P, T) - s0
                    jj = nc.repeat(n); tt = (torch.arange(n, device=dev) + s0).repeat_interleave(C)
                    psym = pos[tt, jj]; basej = coff[nc.repeat(n)]
                    st = cdf[basej + psym]; fr = cdf[basej + psym + 1] - st
                    freq_parts.append(fr); start_parts.append(st)
                    sym_off.append(cur); metas.append((bi, n, C)); cur += n * C
                freq_cat = torch.cat(freq_parts).to(torch.int64)
                start_cat = torch.cat(start_parts).to(torch.int64)
                npg = len(pages)
                k0a = np.zeros((npg, N), np.int32); k1a = np.zeros((npg, N), np.int32)
                psoff = np.zeros(npg, np.int64)
                for pi, (bi, n, C_) in enumerate(metas):
                    S = n * C_; bnd = [round(x * S / N) for x in range(N + 1)]
                    k0a[pi] = bnd[:-1]; k1a[pi] = bnd[1:]; psoff[pi] = sym_off[pi]
                max_lane = int(2 * (max(n for _, n, _ in metas) * C) // N + 64)
                out_bytes = torch.zeros(npg * N * max_lane, dtype=torch.uint8, device=dev)
                out_len = torch.zeros(npg * N, dtype=torch.int32, device=dev)
                def _T(a, dt): return torch.as_tensor(a, dtype=dt, device=dev)
                self.enc_ext.encode_pages(freq_cat, start_cat,
                    _T(k0a.reshape(-1), torch.int32), _T(k1a.reshape(-1), torch.int32),
                    _T(psoff, torch.int64), N, npg, max_lane, out_bytes, out_len)
                out_len = out_len.cpu().numpy().reshape(npg, N); ob = out_bytes.cpu().numpy()
                for pi, (bi, n, C_) in enumerate(metas):
                    S = n * C_
                    lane_byte = [ob[(pi * N + l) * max_lane:(pi * N + l) * max_lane + int(out_len[pi, l])]
                                 for l in range(N)]
                    if 8 * sum(len(x) for x in lane_byte) > budget and int(rung_np[bi]) < R - 1:
                        redo.append(bi); continue
                    hdr = struct.pack("<II", N, S) + b"".join(struct.pack("<I", len(x)) for x in lane_byte)
                    page_blobs[bi] = hdr + b"".join(bytes(x) for x in lane_byte)
            if not redo: break
            for bi in redo: rung[bi] = min(int(rung[bi].item()) + 1, R - 1)
            pending = redo

        out = bytearray()
        out += struct.pack("<IIII", self.MAGIC, T, P, nb)
        out += _pack(rung.cpu().numpy().astype(np.int64), self.id_bits)
        for bi in range(nb):
            blob = page_blobs[bi]
            out += struct.pack("<I", len(blob)) + blob
        self._enc_clear()
        return bytes(out)

    def _prepare_head(self, k):
        """Selection + gather only (no encode kernel). Returns:
           rung_np (nb,), T, nb, jobs: per-page list of dict(freq, start, C, n, ri)
           with freq/start as GPU int64 tensors in lane symbol order."""
        dev = self.device; P = self.P; R = self.R; N = self.N; d = self.d; dz = self.dz
        kd = torch.as_tensor(_np_local(k), dtype=torch.float64, device=dev)
        T = kd.shape[0]; nb = (T + P - 1) // P; pad = nb * P - T
        mu = torch.as_tensor(self.mu, dtype=torch.float64, device=dev)
        fwd = torch.as_tensor(self.fwd, dtype=torch.float64, device=dev)
        r = (kd - mu) @ fwd
        budget = self.payload_budget_bits
        overhead_bits = 8 * (8 + 4 * N + 4 * N + N)

        posD = {}

        def Lt(ri_):
            return self._enc_Lt(ri_)

        def pos_at(ri_):
            if ri_ not in posD:
                t = Lt(ri_)
                q = (torch.sign(r) * torch.floor(torch.abs(r) / t['delta'] + dz)).to(torch.int64)
                fi = (torch.clamp(q, t['lo'][None, :], t['hi'][None, :]) - t['lo'][None, :]) + t['off'][None, :]
                posD[ri_] = t['lut_pos'][fi]
            return posD[ri_]

        # selection (identical to encode_gpu)
        tref = Lt(self.ref)
        qr = (torch.sign(r) * torch.floor(torch.abs(r) / tref['delta'] + dz)).to(torch.int64)
        fir = (torch.clamp(qr, tref['lo'][None, :], tref['hi'][None, :]) - tref['lo'][None, :]) + tref['off'][None, :]
        b_ref = tref['lut_nlp'][fir].sum(1)
        shift = torch.as_tensor(self.shift, dtype=torch.float64, device=dev)
        est = b_ref[None, :] + shift[:, None]
        if pad: est = torch.cat([est, torch.zeros((R, pad), dtype=torch.float64, device=dev)], 1)
        page_est = est.reshape(R, nb, P).sum(2)
        fits = (page_est + overhead_bits) <= budget
        chosen0 = torch.where(fits.any(0), fits.float().argmax(0),
                              torch.full((nb,), R - 1, device=dev)).to(torch.int64)
        def proxy_pb(ri_):
            t = Lt(ri_); pos = pos_at(ri_); jidx = torch.arange(d, device=dev)
            tb = t['bc'][jidx[None, :], pos].sum(1)
            if pad: tb = torch.cat([tb, torch.zeros(pad, dtype=torch.float64, device=dev)])
            return tb.reshape(nb, P).sum(1)
        rung = chosen0.clone(); ppb = {}
        for _ in range(R):
            need = torch.zeros(nb, dtype=torch.bool, device=dev)
            for rr in torch.unique(rung).tolist():
                if rr not in ppb: ppb[rr] = proxy_pb(rr)
                need |= ((rung == rr) & ((ppb[rr] + overhead_bits) > budget) & (rung < R - 1))
            if not bool(need.any()): break
            rung = torch.where(need, torch.clamp(rung + 1, max=R - 1), rung)
        rung_np = rung.cpu().numpy()

        # gather freq/start per page at its chosen rung
        jobs = []
        for bi in range(nb):
            ri_ = int(rung_np[bi]); t = Lt(ri_); pos = pos_at(ri_)
            nc = t['nc']; C = int(nc.numel()); cdf = t['cdf']; coff = t['coff']
            s0 = bi * P; n = min((bi + 1) * P, T) - s0
            jj = nc.repeat(n); tt = (torch.arange(n, device=dev) + s0).repeat_interleave(C)
            psym = pos[tt, jj]; basej = coff[nc.repeat(n)]
            st = cdf[basej + psym]; fr = cdf[basej + psym + 1] - st
            jobs.append(dict(freq=fr.to(torch.int64), start=st.to(torch.int64), C=C, n=n, ri=ri_))
        self._enc_clear()
        return rung_np, T, nb, jobs


# --------------------------------------------------------------------------- #
# Factory helpers

def load_ext(source: str = "rans_decode.cu", name: str = "rans_decode",
             verbose: bool = False):
    """Compile rans_decode.cu and return the extension module.  Call once."""
    return _cu_load(name=name, sources=[source], verbose=verbose)


def load_enc_ext(source: str = "rans_encode.cu", name: str = "rans_encode",
                 verbose: bool = False):
    """Compile rans_encode.cu and return the extension module. Call once."""
    return _cu_load(name=name, sources=[source], verbose=verbose)


def precompute_all_ladder_data(ladder, n_layers, n_kv, d):
    """Scan the entire ladder and precompute integer frequencies, padding layouts,
    and masks for ALL layers, heads, and rungs at once (NumPy-vectorized)."""
    cache = {}
    for ri, (m_scale, delta, model_dict) in enumerate(ladder):
        cache[ri] = {}

        A_max = 1
        for (l, h), models in model_dict.items():
            for m in models:
                if not m.constant:
                    A_max = max(A_max, len(m.vals))

        items = []  # ((l, h), j, m.p)
        for (l, h), models in model_dict.items():
            cache[ri][(l, h)] = {
                'nc_mask': np.zeros(d, dtype=bool),
                'cv': np.zeros(d, dtype=np.int64),
                'vals_padded': np.zeros((d, A_max), dtype=np.int64),
                'freqs': {}
            }
            for j, m in enumerate(models):
                meta = cache[ri][(l, h)]
                if m.constant:
                    meta['cv'][j] = m.const_val
                    meta['freqs'][j] = np.array([16384], dtype=np.int32)
                else:
                    meta['nc_mask'][j] = True
                    v_len = len(m.vals)
                    meta['vals_padded'][j, :v_len] = m.vals
                    items.append(((l, h), j, m.p))

        if not items:
            continue

        num_models = len(items)
        P_batch = np.zeros((num_models, A_max), dtype=np.float64)
        mask_batch = np.zeros((num_models, A_max), dtype=bool)

        for idx, (_, _, p) in enumerate(items):
            p_len = len(p)
            P_batch[idx, :p_len] = p
            mask_batch[idx, :p_len] = True

        p_sums = P_batch.sum(axis=-1, keepdims=True)
        p_sums[p_sums == 0] = 1.0
        P_batch /= p_sums

        TOTAL_ = 16384
        raw = P_batch * TOTAL_
        freqs = np.floor(raw).astype(np.int64)
        deficit = TOTAL_ - freqs.sum(axis=-1, keepdims=True)

        fracs = raw - freqs
        fracs[~mask_batch] = -1e9  # hide padded elements from the deficit

        idx_sort = np.argsort(-fracs, axis=-1)
        rank = np.argsort(idx_sort, axis=-1)

        mask_deficit = rank < deficit
        freqs_final = freqs + mask_deficit.astype(np.int64)

        for idx, ((l, h), j, p) in enumerate(items):
            p_len = len(p)
            cache[ri][(l, h)]['freqs'][j] = freqs_final[idx, :p_len].astype(np.int32)

    return cache


def build_codecs_from_ladder_rans_cuda(
    F, inv, k_mean, ladder, n_layers: int, n_kv: int,
    page_bits: int, P_tok: int, dz: float,
    lanes: int = 1, ext=None, enc_ext=None, device: str = "cuda",
) -> dict:
    if ext is None:
        ext = load_ext()
    if enc_ext is None:
        enc_ext = load_enc_ext()
    print(f"  [Calibration] Initializing vectorized CUDA tables across {n_layers}x{n_kv} grid...")
    codecs = {}
    for l in range(n_layers):
        for h in range(n_kv):
            rungs_lh = [(delta[l, h], model[(l, h)]) for (_, delta, model) in ladder]
            codecs[(l, h)] = PageCodecRANSCUDA(
                F[l, h], inv[l, h], k_mean[l, h],
                rungs_lh, page_bits, P_tok, dz,
                lanes=lanes, ext=ext, enc_ext=enc_ext, device=device
            )
    return codecs


class BatchRANSDecoder:
    def __init__(self, codecs_dict):
        self.codecs = codecs_dict  # Mapping of (l, h) -> PageCodecRANSCUDA

    def decode_grid(self, bufs_grid):
        for c in self.codecs.values():
            c._kernel_events = []

        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end   = torch.cuda.Event(enable_timing=True)
        decoded_grid = {}

        ev_start.record()
        for (l, h), buf in bufs_grid.items():
            decoded_grid[(l, h)] = self.codecs[(l, h)].decode_to_gpu(buf)
        ev_end.record()

        torch.cuda.synchronize()          # the ONLY sync -- after everything is queued

        total_ms = ev_start.elapsed_time(ev_end)
        kernel_ms = sum(
            s.elapsed_time(e)
            for c in self.codecs.values()
            for (s, e) in c._kernel_events
        )
        self.last_total_ms = total_ms
        self.last_kernel_ms = kernel_ms
        print(f"[decode_grid] GPU total {total_ms:.2f} ms | "
              f"entropy kernel {kernel_ms:.2f} ms ({100*kernel_ms/max(total_ms,1e-9):.1f}%) "
              f"| {len(bufs_grid)} heads")
        return decoded_grid


class BatchRANSEncoder:
    """Batched GPU encode: one encode-kernel launch + one host sync across all
    heads' pages. Heads whose pages overflow fall back to per-head encode_gpu.
    Produces buffers byte-identical to per-head encode_gpu."""
    def __init__(self, codecs_dict):
        self.codecs = codecs_dict
        any_c = next(iter(codecs_dict.values()))
        self.N = any_c.N; self.P = any_c.P; self.MAGIC = any_c.MAGIC
        self.id_bits = any_c.id_bits
        self.enc_ext = any_c.enc_ext
        if self.enc_ext is None:
            raise ValueError("codecs have no enc_ext; build with enc_ext set")

    def encode_grid(self, k_grid):
        N = self.N; P = self.P; dev = next(iter(self.codecs.values())).device
        heads = list(k_grid.keys())

        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_kstart = torch.cuda.Event(enable_timing=True)
        ev_kend = torch.cuda.Event(enable_timing=True)
        ev_start.record()

        # PHASE 1: per-head selection + gather (GPU)
        ctx = {}            # head -> (rung_np, T, nb, jobs)
        for h in heads:
            ctx[h] = self.codecs[h]._prepare_head(k_grid[h])

        # PHASE 2: batch all page-jobs into one launch
        job_index = []      # (head, bi)
        freq_parts = []; start_parts = []; k0a = []; k1a = []; psoff = []
        cur = 0; max_lane = 0
        for h in heads:
            _, T, nb, jobs = ctx[h]
            for bi, job in enumerate(jobs):
                C = job['C']; n = job['n']; S = n * C
                freq_parts.append(job['freq']); start_parts.append(job['start'])
                bnd = [round(x * S / N) for x in range(N + 1)]
                k0a.append(bnd[:-1]); k1a.append(bnd[1:]); psoff.append(cur)
                cur += S; job_index.append((h, bi))
                max_lane = max(max_lane, 2 * (S // N) + 64)

        n_jobs = len(job_index)
        freq_cat = torch.cat(freq_parts); start_cat = torch.cat(start_parts)
        def _T(a, dt): return torch.as_tensor(a, dtype=dt, device=dev)
        out_bytes = torch.zeros(n_jobs * N * max_lane, dtype=torch.uint8, device=dev)
        out_len = torch.zeros(n_jobs * N, dtype=torch.int32, device=dev)
        ev_kstart.record()
        self.enc_ext.encode_pages(
            freq_cat, start_cat,
            _T(np.array(k0a, np.int32).reshape(-1), torch.int32),
            _T(np.array(k1a, np.int32).reshape(-1), torch.int32),
            _T(np.array(psoff, np.int64), torch.int64),
            N, n_jobs, max_lane, out_bytes, out_len)
        ev_kend.record()
        out_len = out_len.cpu().numpy().reshape(n_jobs, N)
        ob = out_bytes.cpu().numpy()                       # ONE transfer

        t_asm = time.perf_counter()
        # PHASE 3: assemble per head; mark overflow heads for fallback
        budget = next(iter(self.codecs.values())).payload_budget_bits
        page_blobs = {h: [None] * ctx[h][2] for h in heads}
        redo_heads = set()
        for ji, (h, bi) in enumerate(job_index):
            _, T, nb, jobs = ctx[h]
            C = jobs[bi]['C']; n = jobs[bi]['n']; S = n * C
            lane_byte = [ob[(ji * N + l) * max_lane:(ji * N + l) * max_lane + int(out_len[ji, l])]
                         for l in range(N)]
            if 8 * sum(len(x) for x in lane_byte) > budget and jobs[bi]['ri'] < self.codecs[h].R - 1:
                redo_heads.add(h); continue
            hdr = struct.pack("<II", N, S) + b"".join(struct.pack("<I", len(x)) for x in lane_byte)
            page_blobs[h][bi] = hdr + b"".join(bytes(x) for x in lane_byte)

        asm_ms = (time.perf_counter() - t_asm) * 1e3
        # PHASE 4: emit buffers
        results = {}
        for h in heads:
            if h in redo_heads:
                results[h] = self.codecs[h].encode_gpu(k_grid[h])   # verified per-head path
                continue
            rung_np, T, nb, _ = ctx[h]
            out = bytearray()
            out += struct.pack("<IIII", self.MAGIC, T, P, nb)
            out += _pack(rung_np.astype(np.int64), self.id_bits)
            for bi in range(nb):
                blob = page_blobs[h][bi]
                out += struct.pack("<I", len(blob)) + blob
            results[h] = bytes(out)
        ev_end.record()
        torch.cuda.synchronize()
        total_ms = ev_start.elapsed_time(ev_end)
        kernel_ms = ev_kstart.elapsed_time(ev_kend)
        self.last_total_ms = total_ms
        self.last_kernel_ms = kernel_ms
        print(f"[encode_grid] GPU total {total_ms:.2f} ms | "
              f"encode kernel {kernel_ms:.2f} ms ({100*kernel_ms/max(total_ms,1e-9):.1f}%) "
              f"| {len(heads)} heads"
              + (f" | {asm_ms:.2f} ms assembly" if asm_ms > 1.0 else "")
              + (f" | {len(redo_heads)} overflow fallback" if redo_heads else ""))
        return results
