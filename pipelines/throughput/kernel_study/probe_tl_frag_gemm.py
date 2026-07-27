"""Can T.gemm take its A operand from a FRAGMENT (registers) instead of shared?

This is the hinge for the vq2 decode kernel. Today K and V are computed
element-wise into shared and read back by the gemm -- 64 KiB of shared traffic
per KV block that the Triton kernel never pays, because it keeps K/V in
registers and feeds tl.dot directly.

On Hopper WGMMA the A operand may live in registers while B must come from
shared. If TileLang exposes that, the vq2 kernel can be restructured as
  qk^T = K @ q^T      (A = K in registers, B = q^T in shared)
and the shared round-trip disappears. If it does not, this direction is dead
and the Triton kernel stays.
"""
import torch
import tilelang
import tilelang.language as T

M, N, K = 128, 16, 128


@tilelang.jit(out_idx=[-1])
def gemm_a_shared():
    @T.prim_func
    def main(A: T.Tensor((M, K), "float16"), B: T.Tensor((N, K), "float16"),
             C: T.Tensor((M, N), "float32")):
        with T.Kernel(1, threads=128) as _:
            a_s = T.alloc_shared((M, K), "float16")
            b_s = T.alloc_shared((N, K), "float16")
            c_f = T.alloc_fragment((M, N), "float32")
            T.copy(A, a_s)
            T.copy(B, b_s)
            T.clear(c_f)
            T.gemm(a_s, b_s, c_f, transpose_B=True,
                   policy=T.GemmWarpPolicy.FullRow)
            T.copy(c_f, C)

    return main


@tilelang.jit(out_idx=[-1])
def gemm_a_fragment():
    @T.prim_func
    def main(A: T.Tensor((M, K), "float16"), B: T.Tensor((N, K), "float16"),
             C: T.Tensor((M, N), "float32")):
        with T.Kernel(1, threads=128) as _:
            a_f = T.alloc_fragment((M, K), "float16")   # <-- registers
            b_s = T.alloc_shared((N, K), "float16")
            c_f = T.alloc_fragment((M, N), "float32")
            for i, j in T.Parallel(M, K):
                a_f[i, j] = A[i, j]
            T.copy(B, b_s)
            T.clear(c_f)
            T.gemm(a_f, b_s, c_f, transpose_B=True,
                   policy=T.GemmWarpPolicy.FullRow)
            T.copy(c_f, C)

    return main


def main() -> int:
    dev = "cuda"
    a = torch.randn(M, K, device=dev, dtype=torch.float16)
    b = torch.randn(N, K, device=dev, dtype=torch.float16)
    ref = (a.float() @ b.float().T)

    for name, maker in (("A in shared  ", gemm_a_shared),
                        ("A in fragment", gemm_a_fragment)):
        try:
            c = maker()(a, b)
            err = (c - ref).abs().max().item() / ref.abs().max().item()
            print(f"  {name}: OK   rel err {err:.2e}")
        except Exception as e:
            first = str(e).splitlines()[0][:120] if str(e) else type(e).__name__
            print(f"  {name}: FAILED {type(e).__name__}: {first}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
