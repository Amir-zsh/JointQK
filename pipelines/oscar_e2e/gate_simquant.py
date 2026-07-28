"""Gate: is the SERVED simulated quantizer the same quantizer the kvpress harness uses?

If these two disagree, the served TurboQuant row is measuring a quantizer nobody else ran
and the number is worthless. So this compares, on identical input:

  A) sglang's ``sim_quant.apply_sim_quant`` (bundle-driven, what the server does)
  B) ``turboquant_pytorch`` MSECompressor.compress -> decompress (what TurboQuantPress does)

per layer, for K and V. They must agree to bf16 round-off; anything larger means the
bundle's rotations/centroids or the normalisation convention drifted.

Ported from Samuel's ``logs/gate_simquant.py`` with two additions this tree needs:

  * both geometries -- head_dim 128 (Qwen/Llama) and head_dim 64 (gpt-oss). His gate is
    128-only, so a gpt-oss bundle would go ungated.
  * the hybrid-SWA layer-id check. ``SWAKVPool.set_kv_buffer`` hands each inner pool a
    POOL-LOCAL layer id, so the SWA and full-attention pools both count 0..N-1 and would
    index the same bundle rows. That collision cannot occur on a flat model, so no
    upstream gate covers it.

  python pipelines/oscar_e2e/gate_simquant.py            # both geometries
  python pipelines/oscar_e2e/gate_simquant.py --d 64     # gpt-oss only
"""
import argparse
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "vendor" / "OSCAR-vq" / "sglang-research" / "python"))
sys.path.insert(0, str(REPO / "vendor"))

from sglang.srt.mem_cache.sim_quant import apply_sim_quant, describe  # noqa: E402
from turboquant_pytorch.compressors_v3 import MSECompressor  # noqa: E402

# (head_dim, n_layers, bundle, layers to probe). Probe layers must be < n_layers: the
# rotation is per layer, so testing only layer 0 would pass on a bundle whose later
# rotations are garbage.
GEOMS = {
    128: (36, "artifacts/simquant/simquant_turboquant_d128_L36_k3v3.pt", (0, 7, 35)),
    64: (12, "artifacts/simquant/simquant_turboquant_d64_L12_k3v3.pt", (0, 5, 11)),
}
SEED = 42  # matches TurboQuantPress default
K_BITS = V_BITS = 3
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def check_geometry(D, n_layers, bundle, probe_layers, fail):
    path = str(REPO / bundle)
    if not os.path.exists(path):
        fail.append(f"missing bundle {bundle}")
        print(f"  [FAIL] bundle not found: {bundle}")
        return
    print(f"\n=== head_dim {D}, {n_layers} layers ===")
    print("  " + describe(path))
    H, N = 8, 777  # odd token count exercises the chunked-argmin tail
    torch.manual_seed(0)

    for layer in probe_layers:
        k = torch.randn(N, H, D, device=DEV, dtype=torch.bfloat16)
        v = torch.randn(N, H, D, device=DEV, dtype=torch.bfloat16)
        k_srv, v_srv = apply_sim_quant(path, layer, k, v)

        # Reference: the press builds MSECompressor(head_dim, bits, seed=SEED+layer*1000)
        # for K and +500 for V, over a (B, H, S, D) tensor.
        sb = SEED + layer * 1000
        kc = MSECompressor(D, K_BITS, seed=sb, device=DEV)
        vc = MSECompressor(D, V_BITS, seed=sb + 500, device=DEV)
        k_ref = kc.decompress(kc.compress(k.permute(1, 0, 2).unsqueeze(0).float()))
        v_ref = vc.decompress(vc.compress(v.permute(1, 0, 2).unsqueeze(0).float()))
        k_ref = k_ref.squeeze(0).permute(1, 0, 2).to(torch.bfloat16)
        v_ref = v_ref.squeeze(0).permute(1, 0, 2).to(torch.bfloat16)

        dk = (k_srv.float() - k_ref.float()).abs().max().item()
        dv = (v_srv.float() - v_ref.float()).abs().max().item()
        # bf16 carries ~3 decimal digits; 1e-2 or above is real divergence, not round-off.
        ok = dk < 1e-2 and dv < 1e-2
        print(f"  [{'PASS' if ok else 'FAIL'}] layer {layer:2d}: "
              f"max|served - kvpress|  K {dk:.2e}  V {dv:.2e}")
        if not ok:
            fail.append(f"d{D} layer {layer}")

        if layer == probe_layers[0]:
            # A silently inert quantizer (flag ignored, zeroed bundle) shows NMSE 0.
            nk = ((k.float() - k_srv.float()).pow(2).sum() / k.float().pow(2).sum()).item()
            nv = ((v.float() - v_srv.float()).pow(2).sum() / v.float().pow(2).sum()).item()
            print(f"         served NMSE: K({K_BITS}b) {nk:.5f}  V({V_BITS}b) {nv:.5f}"
                  f"   (nonzero => not a no-op)")
            if nk < 1e-6 or nv < 1e-6:
                fail.append(f"d{D} quantizer is a no-op")

    # Per-layer rotations must actually differ, or the bundle collapses to one layer and
    # every layer silently shares constants.
    b = torch.load(path, map_location="cpu", weights_only=False)
    if n_layers > 1:
        same = torch.allclose(b["k_rotation"][0], b["k_rotation"][min(1, n_layers - 1)])
        print(f"  [{'FAIL' if same else 'PASS'}] per-layer rotations differ")
        if same:
            fail.append(f"d{D} rotations identical across layers")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, choices=sorted(GEOMS), default=None,
                    help="only gate this head_dim (default: both)")
    args = ap.parse_args()
    fail = []

    for D in ([args.d] if args.d else sorted(GEOMS)):
        check_geometry(D, *GEOMS[D], fail)

    # Default-off: every existing BF16 number depends on the write path being untouched
    # when the env var is unset.
    from sglang.srt import environ as _e
    print()
    if os.environ.get("SGLANG_SIMQUANT_PATH"):
        print("  [INFO] SGLANG_SIMQUANT_PATH is set in this shell; skipping default-off check")
    else:
        d = _e.envs.SGLANG_SIMQUANT_PATH.get()
        print(f"  [{'PASS' if d == '' else 'FAIL'}] SGLANG_SIMQUANT_PATH default = {d!r} "
              f"(empty -> BF16 write path untouched)")
        if d != "":
            fail.append("simquant not default-off")
        swa = _e.envs.SGLANG_SIMQUANT_SKIP_SWA.get()
        print(f"  [{'PASS' if swa else 'FAIL'}] SGLANG_SIMQUANT_SKIP_SWA default = {swa} "
              f"(True -> hybrid SWA pool opts out of the pool-local layer-id collision)")
        if not swa:
            fail.append("SKIP_SWA not default-on")

    print("\n" + ("GATE PASS" if not fail else "GATE FAIL: " + ", ".join(fail)))
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
