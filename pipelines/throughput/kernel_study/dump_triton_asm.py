"""Dump and diff the compiled code Triton actually emits for the vq2 gather.

Two things this is for.

1. **Verifying that source-level hints survive to PTX.** An earlier round swept
   `eviction_policy` and `cache_modifier` on the codebook and stream loads and
   measured 0.99-1.03x -- "no effect". That conclusion is only meaningful if the
   hints reached the instruction stream. If Triton dropped them, the sweep
   measured nothing at all. `--hints` compiles with and without and diffs the
   `ld.global` qualifiers.

2. **Comparing instruction mix against the CUDA kernel.** The CUDA version wins
   by moving the codebook gather from `LDG` (global, sector-limited) to `LDS`
   (shared, bank-limited). This prints the load-instruction histogram at SASS
   level so the two are directly comparable.

  docker exec -e CUDA_VISIBLE_DEVICES=0 oscar-ab bash -lc 'cd <repo> && \
    PYTHONPATH=vendor/OSCAR-vq/sglang-research/python \
    /opt/venv-oscar/bin/python pipelines/throughput/kernel_study/dump_triton_asm.py --hints'
"""
import argparse
import collections
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parent))

BATCH, SEQ, SPLITS = 8, 4096, 8
H_Q, H_KV, L, NG, KC = 32, 8, 128, 32, 256
SM_SCALE = 1.0 / (L ** 0.5)


def build(dev="cuda", seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    total, cache = BATCH * SEQ, BATCH * SEQ + 64
    q = torch.randn(BATCH, H_Q, L, generator=g, device=dev, dtype=torch.float16)
    k_idx = torch.randint(0, KC, (cache, H_KV, NG), generator=g, device=dev, dtype=torch.uint8)
    cent = torch.randn(H_KV, NG, KC, 4, generator=g, device=dev, dtype=torch.float32) * (L ** -0.5)
    cb = cent.to(torch.float8_e5m2).view(torch.int32).squeeze(-1).contiguous()
    v_buf = torch.randint(0, 256, (cache, H_KV, L // 4), generator=g, device=dev, dtype=torch.uint8)
    k_sz = torch.rand(cache, H_KV, 2, generator=g, device=dev) * 0.5 + 0.75
    v_sz = torch.rand(cache, H_KV, 2, generator=g, device=dev) * 0.1 + 0.05
    kv_indptr = torch.arange(0, BATCH + 1, device=dev, dtype=torch.int32) * SEQ
    kv_indices = torch.randperm(cache, generator=g, device=dev)[:total].to(torch.int32)
    splits = torch.full((BATCH,), SPLITS, device=dev, dtype=torch.int32)
    out = torch.zeros(BATCH, H_Q, SPLITS, L, device=dev, dtype=torch.float32)
    lse = torch.zeros(BATCH, H_Q, SPLITS, device=dev, dtype=torch.float32)
    return locals()


def compiled(fn) -> dict:
    """Pull the newest compiled binary out of a JITFunction's cache."""
    dc = getattr(fn, "device_caches", None) or {}
    for _, tup in dc.items():
        kc = tup[0] if isinstance(tup, (tuple, list)) else tup
        for _, ck in kc.items():
            return ck.asm
    return {}


def ptx_loads(ptx: str) -> collections.Counter:
    """Histogram of ld.global variants, including cache/eviction qualifiers."""
    c = collections.Counter()
    for m in re.finditer(r"\bld\.global[.\w:]*", ptx):
        c[m.group(0).split()[0]] += 1
    return c


def sass_hist(cubin: bytes) -> collections.Counter:
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(cubin)
        path = f.name
    try:
        out = subprocess.run(["nvdisasm", "-c", path], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return collections.Counter()
    finally:
        os.unlink(path)
    c = collections.Counter()
    for line in out.splitlines():
        m = re.search(r"^\s*/\*[0-9a-f]+\*/\s+@?!?\S*\s*([A-Z][A-Z0-9_.]*)", line)
        if m:
            op = m.group(1).split(".")[0]
            if op in ("LDG", "LDS", "LDSM", "STS", "STG", "LDL", "STL", "FFMA",
                      "HFMA2", "IMAD", "PRMT", "LOP3", "SHF", "MUFU", "BAR"):
                c[op] += 1
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hints", action="store_true",
                    help="diff PTX with and without cache/eviction hints")
    a = ap.parse_args()
    d = build()
    from sglang.srt.layers.attention.triton_ops import decode_attention as D

    os.environ.update(SGL_VQ2_BLOCK_N="64", SGL_VQ2_BLOCK_H="8",
                      SGL_VQ2_NUM_WARPS="2", SGL_VQ2_NUM_STAGES="3")
    D._decode_grouped_att_m_fwd_quant_vq2(
        d["q"], d["k_idx"], d["cb"], d["v_buf"], d["k_sz"], d["v_sz"],
        d["out"], d["lse"], d["kv_indptr"], d["kv_indices"], d["splits"],
        SPLITS, SM_SCALE, 0.0)
    torch.cuda.synchronize()
    asm = compiled(D._fwd_grouped_kernel_stage1_quant_vq2)
    print(f"shipped vq2 kernel: asm keys = {sorted(asm)}")
    if "ptx" in asm:
        print("\n=== ld.global variants in PTX (shipped kernel) ===")
        for k, v in sorted(ptx_loads(asm["ptx"]).items(), key=lambda x: -x[1]):
            print(f"  {v:>5}  {k}")
    if "cubin" in asm:
        h = sass_hist(asm["cubin"])
        if h:
            print("\n=== SASS instruction histogram (shipped kernel) ===")
            for k, v in sorted(h.items(), key=lambda x: -x[1]):
                print(f"  {v:>6}  {k}")
        else:
            print("\n(nvdisasm unavailable; skipped SASS)")

    if a.hints:
        import vq2_variants as V
        print("\n=== do source-level hints reach PTX? ===")
        seen = {}
        for tag, evs, evc, cms, cmc in (
                ("no hints", "", "", "", ""),
                ("evict_first/evict_last", "evict_first", "evict_last", "", ""),
                ("cache .cg on streams", "", "", ".cg", ""),
                ("cache .cs on streams", "", "", ".cs", "")):
            try:
                V.launch_ev(d, 64, 8, 2, 3, evs, evc, cms, cmc)
                torch.cuda.synchronize()
            except Exception as e:
                print(f"  {tag:<26} FAILED {type(e).__name__}: "
                      f"{str(e).splitlines()[0][:70]}")
                continue
            ptx = compiled(V._stage1_ev).get("ptx", "")
            hist = ptx_loads(ptx)
            seen[tag] = hist
            qual = {k: v for k, v in hist.items() if k != "ld.global"}
            print(f"  {tag:<26} {dict(qual) if qual else 'NO qualified loads -- '
                  'every ld.global is plain'}")
        base = seen.get("no hints")
        for tag, h in seen.items():
            if tag != "no hints" and base is not None:
                print(f"  {tag:<26} {'DIFFERS from no-hints' if h != base else 'IDENTICAL to no-hints'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
