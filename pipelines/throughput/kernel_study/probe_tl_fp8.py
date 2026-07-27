"""Nail the fp8-e5m2 -> f16 bit trick in TileLang before building the big kernel.

e5m2 is [sign|5 exp|2 mant] and float16 is [sign|5 exp|10 mant] with the SAME
exponent bias, so decoding is exactly `byte << 8` reinterpreted as f16 -- no
arithmetic, no lookup table. Triton spells this `.to(tl.float8e5, bitcast=True)`.
Verify against torch's own float8_e5m2 cast over all 256 byte values.
"""
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def decode256():
    @T.prim_func
    def main(b: T.Tensor((256,), "uint8"), out: T.Tensor((256,), "float16")):
        with T.Kernel(1, threads=256) as _:
            for i in T.Parallel(256):
                out[i] = T.reinterpret("float16",
                                       T.Cast("uint16", b[i]) << T.Cast("uint16", 8))

    return main


def main() -> int:
    dev = "cuda"
    b = torch.arange(256, device=dev, dtype=torch.uint8)
    got = decode256()(b)
    ref = b.view(torch.float8_e5m2).to(torch.float16)

    finite = torch.isfinite(ref)
    exact = bool((got[finite] == ref[finite]).all().item())
    nan_ok = bool((torch.isnan(got[~finite]) == torch.isnan(ref[~finite])).all().item())
    print(f"finite values exact : {exact}  ({int(finite.sum())}/256)")
    print(f"nan/inf pattern ok  : {nan_ok}")
    if not exact:
        bad = (got != ref) & finite
        i = int(bad.nonzero()[0].item())
        print(f"  first mismatch byte {i}: got {got[i]} ref {ref[i]}")
    print("VERDICT:", "PASS" if (exact and nan_ok) else "FAIL")
    return 0 if (exact and nan_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
