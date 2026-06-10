#!/usr/bin/env python3
"""PageCodecRANS: the PageCodec pipeline with the per-page coder swapped from
constriction to the interleaved-rANS reference (rans_interleaved.py).

Same external interface as PageCodec (encode/decode/attrs), so it drops into
test_codec_on_data.py and reconstructs the SAME snapped indices -> identical
K_hat. Use it to test the exact GPU bitstream format on your real data before any
CUDA work: if PageCodecRANS matches PageCodec bit-for-bit on K_hat, the rANS format
is correct and the kernel just has to reproduce decode_page.

N = intra-page lanes. N=1 (default) = one rANS stream per page, lowest overhead,
still fully parallel ACROSS pages (block per page). N>1 shortens the per-page
serial chain at the cost of per-lane header/state (eats the fixed-page budget --
see the overhead table). Selection/fit/step-coarser are inherited unchanged; the
real-byte fit check now measures rANS bytes.
"""
import struct
import numpy as np

from kvq_codec import PageCodec, _timed
import rans_interleaved as ri


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
                        row.append(ri.FreqTable(m.vals, np.array([1.0])))
                    else:
                        row.append(ri.FreqTable(m.vals, m.p))
                self.rtab.append(row)

    # --- per-page coder overrides (bytes instead of uint32 words) -----------
    def _encode_page(self, pos_ri, sl, ridx):
        sub = pos_ri[sl]                                  # (n,d) positions
        n = sub.shape[0]
        return ri.encode_page(sub, self.rtab[ridx], n, self.d, self.N)

    def _blob_nbits(self, blob):
        return len(blob) * 8                              # raw bytes

    def _serialize_blob(self, blob):
        return struct.pack("<I", len(blob)) + blob

    def _read_blob(self, buf, off):
        (nb_,) = struct.unpack_from("<I", buf, off); off += 4
        return bytes(buf[off:off + nb_]), off + nb_

    def _decode_page(self, blob, ridx, n):
        positions = ri.decode_page(blob, self.rtab[ridx], n, self.d)
        return ri.positions_to_values(positions, self.rtab[ridx], n, self.d)


def build_codecs_from_ladder_rans(F, inv, k_mean, ladder, n_layers, n_kv,
                                  page_bits, P_tok, dz, lanes=1):
    codecs = {}
    for l in range(n_layers):
        for h in range(n_kv):
            rungs_lh = [(delta[l, h], model[(l, h)]) for (_, delta, model) in ladder]
            codecs[(l, h)] = PageCodecRANS(F[l, h], inv[l, h], k_mean[l, h],
                                           rungs_lh, page_bits, P_tok, dz, lanes=lanes)
    return codecs