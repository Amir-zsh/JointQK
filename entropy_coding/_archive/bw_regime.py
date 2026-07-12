#!/usr/bin/env python3
"""Bandwidth-regime three-way (no codec needed; rANS uses measured 12.394us/page marginal).
Full-model decode step, batched, long context. Reports GB/s + %peak to verify regime."""
import torch, triton, triton.language as tl

dev = torch.device("cuda")
L, Hkv, d = 36, 8, 128
P = 64
A100_PEAK_GBs = 1555.0
RANS_US_PER_PAGE = 12.394   # canonical, full-280-head-grid kernel-only rate (see eg_vs_rans_matched.py; was 3.88 from an 8-head sample)

@triton.jit
def int2_unpack_kernel(packed_ptr, scale_ptr, zero_ptr, out_ptr, n_elem, G: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid*BLOCK + tl.arange(0,BLOCK).to(tl.int64)
    mask = offs < n_elem
    b = tl.load(packed_ptr + offs//4, mask=mask, other=0).to(tl.uint32)
    q = (b >> ((offs%4)*2).to(tl.uint32)) & 0x3
    grp = offs//G
    s = tl.load(scale_ptr+grp, mask=mask, other=1.0); z = tl.load(zero_ptr+grp, mask=mask, other=0.0)
    tl.store(out_ptr+offs, ((q.to(tl.float32)-z)*s).to(tl.float16), mask=mask)

@triton.jit
def read_kernel(ptr, n, BLOCK: tl.constexpr, out_ptr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid*BLOCK + tl.arange(0,BLOCK).to(tl.int64)
    mask = offs < n
    x = tl.load(ptr+offs, mask=mask, other=0)
    tl.atomic_add(out_ptr, tl.sum(x.to(tl.float32)))

def timeit_ev(f, reps=20, warmup=5):
    for _ in range(warmup): f()
    torch.cuda.synchronize()
    s,e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(reps): f()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/reps

def free_gb():
    free, total = torch.cuda.mem_get_info(); return free/1e9

print(f"A100 peak ~{A100_PEAK_GBs/1000:.2f} TB/s. %peak high => bandwidth-bound => conclusive.")
print(f"rANS = measured {RANS_US_PER_PAGE}us/page (kernel-only, saturated) x full-model pages.\n")
print(f"{'T':>7} {'B':>3} {'cacheGB':>8} {'BF16_ms':>9} {'GB/s':>7} {'%pk':>5} "
      f"{'INT2_ms':>9} {'GB/s':>7} {'INT2/BF16':>10} {'rANS_ms':>10} {'rANS/BF16':>10}")

configs = [(32768,1),(32768,4),(32768,8),(65536,1),(65536,4),(100000,1),(100000,4)]

for (T, B) in configs:
    nb = (T+P-1)//P
    n_elem = L*Hkv*T*d*2*B
    bf16_bytes = n_elem*2
    int2_read_bytes = n_elem/4
    cache_gb = bf16_bytes/1e9

    bf16_ok = (cache_gb < free_gb()*0.80)
    kv16 = None
    if bf16_ok:
        try: kv16 = torch.randn(n_elem, device=dev, dtype=torch.float16)
        except RuntimeError: bf16_ok = False; torch.cuda.empty_cache()

    sink = torch.zeros(1, device=dev, dtype=torch.float32)
    if bf16_ok:
        def read_bf16():
            read_kernel[(triton.cdiv(kv16.numel(),8192),)](kv16, kv16.numel(), 8192, sink)
        tb = timeit_ev(read_bf16)
        bf16_gbs = bf16_bytes/1e9/(tb/1e3); pct = bf16_gbs/A100_PEAK_GBs*100
        bf16_str = f"{tb:>7.3f}ms {bf16_gbs:>6.0f} {pct:>4.0f}%"
    else:
        tb = None; bf16_str = f"{'OOM':>7}  {'--':>6} {'--':>5}"

    if kv16 is not None: del kv16
    torch.cuda.empty_cache()

    try:
        packed = torch.randint(0,256,((n_elem+3)//4,), device=dev, dtype=torch.uint8)
        n_groups = (n_elem+63)//64
        scale_g = torch.rand(n_groups, device=dev, dtype=torch.float32)+0.1
        zero_g = torch.zeros(n_groups, device=dev, dtype=torch.float32)
        out16 = torch.empty(n_elem, device=dev, dtype=torch.float16)
        def do_int2():
            int2_unpack_kernel[(triton.cdiv(n_elem,1024),)](packed, scale_g, zero_g, out16, n_elem, 64, 1024)
        ti = timeit_ev(do_int2)
        int2_gbs = int2_read_bytes/1e9/(ti/1e3)
        int2_speedup = (tb/ti) if bf16_ok else float('nan')
        int2_sp = f"{int2_speedup:>8.2f}x" if bf16_ok else f"{'BF16 OOM':>8}"
        int2_str = f"{ti:>7.3f}ms {int2_gbs:>6.0f}"
        del packed, scale_g, zero_g, out16
    except RuntimeError:
        int2_str = f"{'OOM':>7}  {'--':>6}"; int2_sp = f"{'--':>8}"
    torch.cuda.empty_cache()

    rans_ms = RANS_US_PER_PAGE * (L*Hkv*nb*B) / 1e3
    rans_sp = f"{rans_ms/tb:>8.0f}x" if bf16_ok else f"{'--':>9}"
    print(f"{T:>7} {B:>3} {cache_gb:>7.1f}G {bf16_str} {int2_str} {int2_sp} {rans_ms:>8.1f}ms {rans_sp}")

print("\n--- how to read ---")
print("%pk (BF16): >50% => BANDWIDTH-BOUND => INT2/BF16 ratio CONCLUSIVE. low => inconclusive.")
print("INT2/BF16 < 1, -> byte ratio (~8x) as bandwidth-bound: INT2 wins (OSCAR mechanism).")
print("rANS_ms: full-model entropy decode/token; same 2-bit read as INT2 but 3.88us/page serial -> decode-bound.")
print("BF16 OOM where INT2 runs = capacity win (8x smaller fits).")