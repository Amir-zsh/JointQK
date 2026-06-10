#!/usr/bin/env python3
"""PageCodecRDO: per-token Lagrangian RDO within a fixed page budget.

Unlike PageCodecRANS (one rung per page), each TOKEN gets its own rung, chosen
to minimize D(t) + lambda*R(t); lambda is bisected per page so total rate fits
page_bits. Symbol (t,j) is coded with rtab[rung[t]][j]. Per-token rung ids are
stored in the page. Self-consistent CPU encode/decode (reference; the CUDA kernel
needs per-token table indirection to match this).

Rate term for allocation = true cross-entropy -log2 p(snapped sym) under the
frozen calib model (matches run_paged_split.py's _tok_bits_np), NOT the
normalized-freq proxy SCALE_BITS-log2(freq).

Page blob layout:
  <u32 N><u32 S><u32 n><n * u16 rung_id><u32 lane_len[N]><lane bytes...>
"""
import struct
import numpy as np

from kvq_codec import PageCodec, _timed, _dz_dequant, _dz_round, _pack, _unpack, _np
import rans_interleaved as ri


class PageCodecRDO(PageCodec):
    MAGIC = 0x4B564335  # 'KVC5'

    def __init__(self, fwd, inv, mu, rungs, page_bits, P_tok, dz, lanes=1):
        super().__init__(fwd, inv, mu, rungs, page_bits, P_tok, dz)
        self.N = int(lanes)

        with _timed("build.rans_tables"):
            self.rtab = []
            for r in range(self.R):
                row = []
                for m in self.models[r]:
                    if m.constant:
                        row.append(ri.FreqTable(m.vals, np.array([1.0])))
                    else:
                        row.append(ri.FreqTable(m.vals, m.p))
                self.rtab.append(row)

        # flat per-rung tables (no padding): concatenated across coords + offsets.
        # _bc_f = normalized-freq proxy (kept for reference); _nlp_f = true -log2 p
        # (used for allocation).
        d = self.d
        self._coff = []
        self._vals_f, self._freq_f, self._cdf_f, self._bc_f, self._nlp_f = [], [], [], [], []
        for r in range(self.R):
            off = np.zeros(d + 1, np.int64)
            vparts, fparts, cparts, bparts, nparts = [], [], [], [], []
            for j in range(d):
                t = self.rtab[r][j]
                if t.constant:
                    vparts.append(np.array([t.const_val], np.int64))
                    fparts.append(np.array([ri.TOTAL], np.int64))
                    cparts.append(np.array([0], np.int64))
                    bparts.append(np.array([0.0]))
                    nparts.append(np.array([0.0]))
                    off[j + 1] = off[j] + 1
                else:
                    a = len(t.vals)
                    vparts.append(t.vals.astype(np.int64))
                    fparts.append(t.freq.astype(np.int64))
                    cparts.append(t.cdf[:a].astype(np.int64))
                    bparts.append(ri.SCALE_BITS - np.log2(np.maximum(t.freq, 1)))
                    # true -log2 p, aligned to t.vals ordering (same as _bc_f indexing)
                    p = np.asarray(self.models[r][j].p, dtype=np.float64)
                    nparts.append(-np.log2(np.maximum(p, 1e-12)))
                    off[j + 1] = off[j] + a
            self._coff.append(off)
            self._vals_f.append(np.concatenate(vparts))
            self._freq_f.append(np.concatenate(fparts))
            self._cdf_f.append(np.concatenate(cparts))
            self._bc_f.append(np.concatenate(bparts))
            self._nlp_f.append(np.concatenate(nparts))

        # page coded-coord set = non-constant coords at the FINEST rung
        self._nc0 = [j for j in range(d) if not self.models[0][j].constant]
        assert self.R <= 65535
        self.rdo_overflow = 0

        # ---- GPU-resident stacked tables for the kernel (model (a): per head) ----
        import torch
        dev = torch.device("cuda")
        self.device = dev
        cdf_parts, vals_parts = [], []
        off2d = np.zeros(self.R * (d + 1), np.int64)
        voff2d = np.zeros(self.R * (d + 1), np.int64)
        ccur = vcur = 0
        for r in range(self.R):
            co = self._coff[r]
            for j in range(d):
                off2d[r * (d + 1) + j] = ccur
                voff2d[r * (d + 1) + j] = vcur
                t = self.rtab[r][j]
                if t.constant:
                    cdf_parts.append(np.array([0, ri.TOTAL], np.int64))
                    vals_parts.append(np.array([t.const_val], np.int64))
                    ccur += 2; vcur += 1
                else:
                    a = int(co[j + 1] - co[j])
                    cdf_parts.append(np.concatenate([self._cdf_f[r][co[j]:co[j] + a], [ri.TOTAL]]))
                    vals_parts.append(self._vals_f[r][co[j]:co[j] + a])
                    ccur += a + 1; vcur += a
            off2d[r * (d + 1) + d] = ccur
            voff2d[r * (d + 1) + d] = vcur
        self._cdf_gpu   = torch.as_tensor(np.concatenate(cdf_parts).astype(np.int64),
                                          dtype=torch.int32, device=dev)
        self._off2d_gpu = torch.as_tensor(off2d, dtype=torch.int64, device=dev)
        self._vals_stk  = torch.as_tensor(np.concatenate(vals_parts), dtype=torch.int64, device=dev)
        self._voff2d    = torch.as_tensor(voff2d, dtype=torch.int64, device=dev)
        self._nc0_gpu   = torch.as_tensor(np.array(self._nc0, np.int32), dtype=torch.int32, device=dev)
        self._delta_stk = torch.as_tensor(np.stack([self.deltas[r] for r in range(self.R)]),
                                          dtype=torch.float32, device=dev)
        self._inv_gpu   = torch.as_tensor(self.inv, dtype=torch.float32, device=dev)
        self._mu_gpu    = torch.as_tensor(self.mu,  dtype=torch.float32, device=dev)
        self.ext = None

    def _score_all_rungs(self, r):
        """r:(T,d). D = raw-rounded-index SE (no codebook snap); R = true -log2 p
        of the snapped symbol summed over coords. Both (T, Rungs)."""
        T = r.shape[0]; d = self.d
        D = np.empty((T, self.R)); R = np.empty((T, self.R))
        for rr in range(self.R):
            pos = self._snap_rung(r, rr)                 # (T,d) snapped positions
            base = self._coff[rr][:d]
            flat_idx = base[None, :] + pos
            idx_raw = _dz_round(r, self.deltas[rr], self.dz)
            dq_raw = _dz_dequant(idx_raw, self.deltas[rr], self.dz)
            D[:, rr] = ((r - dq_raw) ** 2).sum(1)
            R[:, rr] = self._nlp_f[rr][flat_idx].sum(1)  # true cross-entropy
        return D, R

    def _encode_page_rdo(self, r_page, rung_tok, pos_by_rung, t0):
        n = r_page.shape[0]; nc = self._nc0
        order = [(t, j) for t in range(n) for j in nc]
        S = len(order)
        b = ri._lane_bounds(S, self.N)
        ts = np.fromiter((t for (t, _) in order), np.int64, S)
        js = np.fromiter((j for (_, j) in order), np.int64, S)
        rsym = rung_tok[ts]
        freq = np.empty(S, np.int64); start = np.empty(S, np.int64)
        for rr in np.unique(rsym):
            msk = rsym == rr
            tt = ts[msk]; jj = js[msk]
            posn = pos_by_rung[rr][t0 + tt, jj]
            fi = self._coff[rr][jj] + posn
            freq[msk] = self._freq_f[rr][fi]
            start[msk] = self._cdf_f[rr][fi]
        blobs = [ri._encode_lane_nb(freq[b[l]:b[l+1]], start[b[l]:b[l+1]]) for l in range(self.N)]
        head = struct.pack("<III", self.N, S, n)
        head += rung_tok.astype("<u2").tobytes()
        head += b"".join(struct.pack("<I", len(x)) for x in blobs)
        return head + b"".join(bytes(x) for x in blobs)

    def _decode_page_rdo(self, blob, n):
        N, S, nn = struct.unpack_from("<III", blob, 0)
        off = 12
        rung_tok = np.frombuffer(blob, dtype="<u2", count=nn, offset=off).astype(np.int64)
        off += 2 * nn
        lane_len = [struct.unpack_from("<I", blob, off + 4 * l)[0] for l in range(N)]
        off += 4 * N
        nc = self._nc0
        order = [(t, j) for t in range(n) for j in nc]
        b = ri._lane_bounds(S, N)
        starts = np.cumsum([0] + lane_len)
        idx = np.zeros((n, self.d), np.int64)
        for l in range(N):
            seg = order[b[l]:b[l + 1]]
            tabs = [self.rtab[int(rung_tok[t])][j] for (t, j) in seg]
            lane_buf = blob[off + starts[l]: off + starts[l + 1]]
            syms = ri.decode_lane(lane_buf, tabs)
            for (t, j), s in zip(seg, syms):
                idx[t, j] = self.rtab[int(rung_tok[t])][j].vals[s]
        const0 = [j for j in range(self.d) if self.models[0][j].constant]
        for j in const0:
            for t in range(n):
                idx[t, j] = self.rtab[int(rung_tok[t])][j].vals[0]
        return idx, rung_tok

    def _alloc_page(self, D, R, budget_bits):
        n = D.shape[0]
        ar = np.arange(n)
        def alloc(lam):
            return (D + lam * R).argmin(1)
        rung_min = R.argmin(1)
        if R[ar, rung_min].sum() > budget_bits:
            return rung_min                       # even coarsest-rate overflows
        lam_hi = max(1.0, float(D.max())) * 1e3   # match old: large enough to force coarsest
        lo, hi = 0.0, lam_hi
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            a = alloc(mid)
            if R[ar, a].sum() > budget_bits:      # over budget -> need larger lam
                lo = mid
            else:                                  # fits -> try smaller lam (finer)
                hi = mid
        return alloc(hi)                           # hi side always fits
    
    # ----- full encode/decode -----
    def encode(self, k):
        k = _np(k)
        with _timed("encode.transform"):
            r = (k - self.mu) @ self.fwd
        T = r.shape[0]; P = self.P; nb = (T + P - 1) // P

        with _timed("encode.estimate"):
            D, R = self._score_all_rungs(r)
            pos_by_rung = {rr: self._snap_rung(r, rr) for rr in range(self.R)}

        rung_all = np.empty(T, np.int64); blobs = []
        for bi in range(nb):
            sl = slice(bi * P, min((bi + 1) * P, T)); n = sl.stop - sl.start
            side_bits = 8 * (12 + 2 * n + 4 * self.N)
            budget = self.page_bits - side_bits
            rung_tok = self._alloc_page(D[sl], R[sl], budget)
            with _timed("encode.rangecode"):
                blob = self._encode_page_rdo(r[sl], rung_tok, pos_by_rung, sl.start)
            guard = 0
            while len(blob) * 8 > self.page_bits and rung_tok.max() < self.R - 1 and guard < self.R:
                rung_tok = np.minimum(rung_tok + 1, self.R - 1)
                with _timed("encode.rangecode"):
                    blob = self._encode_page_rdo(r[sl], rung_tok, pos_by_rung, sl.start)
                guard += 1
            if len(blob) * 8 > self.page_bits:
                self.rdo_overflow += 1
            rung_all[sl] = rung_tok; blobs.append(blob)
            for rr in rung_tok:
                self.rung_hist[int(rr)] += 1
        self.nblocks += nb

        with _timed("encode.assemble"):
            out = bytearray()
            out += struct.pack("<IIII", self.MAGIC, T, P, nb)
            for blob in blobs:
                out += struct.pack("<I", len(blob)) + blob
        return bytes(out)

    def decode(self, buf):
        buf = bytes(buf)
        with _timed("decode.parse"):
            magic, T, P, nb = struct.unpack_from("<IIII", buf, 0)
            assert magic == self.MAGIC, "bad RDO stream"
            off = 16
        idx = np.zeros((T, self.d), np.int64)
        rung_all = np.zeros(T, np.int64)
        with _timed("decode.rangedecode"):
            for bi in range(nb):
                (blen,) = struct.unpack_from("<I", buf, off); off += 4
                blob = buf[off:off + blen]; off += blen
                sl = slice(bi * P, min((bi + 1) * P, T)); n = sl.stop - sl.start
                idx_p, rung_tok = self._decode_page_rdo(blob, n)
                idx[sl] = idx_p; rung_all[sl] = rung_tok
        with _timed("decode.dequant"):
            r_hat = np.empty((T, self.d))
            for t in range(T):
                r_hat[t] = _dz_dequant(idx[t], self.deltas[int(rung_all[t])], self.dz)
        with _timed("decode.inverse"):
            k_hat = ((r_hat @ self.inv) + self.mu).astype(np.float32)
        return k_hat

    def decode_to_gpu_rdo(self, buf):
        import torch
        buf = bytes(buf); dev = self.device
        magic, T, P, nb = struct.unpack_from("<IIII", buf, 0); o = 16
        C = int(self._nc0_gpu.numel()); d = self.d; N = self.N

        all_bytes = bytearray()
        page_byte_off = np.zeros(nb, np.int64)
        lane_off = np.zeros((nb, N), np.int64)
        k0a = np.zeros((nb, N), np.int32); k1a = np.zeros((nb, N), np.int32)
        rung_tok = np.zeros((nb, P), np.int32)
        ns = np.zeros(nb, np.int32)
        cum = 0
        for bi in range(nb):
            (blen,) = struct.unpack_from("<I", buf, o); o += 4
            blob = buf[o:o+blen]; o += blen
            n = min((bi+1)*P, T) - bi*P; ns[bi] = n
            Nh, S, nn = struct.unpack_from("<III", blob, 0); p = 12
            rt = np.frombuffer(blob, "<u2", nn, p).astype(np.int32); p += 2*nn
            rung_tok[bi, :nn] = rt
            lane_len = [struct.unpack_from("<I", blob, p+4*l)[0] for l in range(N)]; p += 4*N
            starts = np.cumsum([0]+lane_len); bounds = [round(k*S/N) for k in range(N+1)]
            page_byte_off[bi] = cum
            for l in range(N):
                lane_off[bi, l] = p + int(starts[l]); k0a[bi, l] = bounds[l]; k1a[bi, l] = bounds[l+1]
            all_bytes += blob; cum += blen
        blob_arr = np.frombuffer(bytes(all_bytes), np.uint8).copy()

        def _T(a, dt): return torch.as_tensor(a, dtype=dt, device=dev)
        _ks = torch.cuda.Event(enable_timing=True); _ke = torch.cuda.Event(enable_timing=True)
        _ks.record()
        pos = self.ext.decode_pages_rdo(
            _T(blob_arr, torch.uint8), _T(page_byte_off, torch.int64),
            _T(lane_off.reshape(-1), torch.int64), _T(k0a.reshape(-1), torch.int32),
            _T(k1a.reshape(-1), torch.int32), _T(rung_tok.reshape(-1), torch.int32),
            self._nc0_gpu, self._cdf_gpu, self._off2d_gpu, N, C, P, d, self.R)
        _ke.record()
        if not hasattr(self, "_kernel_events"): self._kernel_events = []
        self._kernel_events.append((_ks, _ke))

        rt_gpu = _T(rung_tok, torch.int64)
        ar = torch.arange(d, device=dev)
        idx = torch.zeros((T, d), dtype=torch.int64, device=dev)
        for bi in range(nb):
            n = int(ns[bi]); r_row = rt_gpu[bi, :n]
            base = self._voff2d[r_row[:, None]*(d+1) + ar[None, :]]
            pp = pos[bi, :n].long()
            idx[bi*P: bi*P+n] = self._vals_stk[base + pp]

        rt_flat = torch.zeros(T, dtype=torch.int64, device=dev)
        for bi in range(nb):
            n = int(ns[bi]); rt_flat[bi*P: bi*P+n] = rt_gpu[bi, :n]
        delta_tok = self._delta_stk[rt_flat]
        idx_f = idx.float()
        r_hat = idx_f.sign() * (idx_f.abs() + (0.5 - self.dz)) * delta_tok
        return r_hat @ self._inv_gpu + self._mu_gpu


def build_codecs_from_ladder_rdo(F, inv, k_mean, ladder, n_layers, n_kv,
                                 page_bits, P_tok, dz, lanes=1):
    codecs = {}
    for l in range(n_layers):
        for h in range(n_kv):
            rungs_lh = [(delta[l, h], model[(l, h)]) for (_, delta, model) in ladder]
            codecs[(l, h)] = PageCodecRDO(F[l, h], inv[l, h], k_mean[l, h],
                                          rungs_lh, page_bits, P_tok, dz, lanes=lanes)
    return codecs


def build_codecs_from_ladder_rdo_cuda(F, inv, k_mean, ladder, n_layers, n_kv,
                                      page_bits, P_tok, dz, lanes=1, ext=None):
    from kvq_codec_cuda import load_ext
    if ext is None:
        ext = load_ext(source="rans_decode_rdo.cu", name="rans_decode_rdo")
    codecs = build_codecs_from_ladder_rdo(F, inv, k_mean, ladder, n_layers, n_kv,
                                          page_bits, P_tok, dz, lanes=lanes)
    for c in codecs.values():
        c.ext = ext
    return codecs