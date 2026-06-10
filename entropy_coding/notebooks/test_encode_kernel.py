import numpy as np, torch
from torch.utils.cpp_extension import load

f = np.load("encode_fixtures.npz")
N=int(f["N"]); C=int(f["C"]); n0=int(f["n0"]); SB=int(f["SB"])
assert SB==14
freq=f["freq_p0"].astype(np.int64); start=f["start_p0"].astype(np.int64)
lane_lens=f["lane_lens_p0"].astype(np.int64)
lane_bytes=f["lane_bytes_p0"].astype(np.uint8)
S=freq.shape[0]

# lane bounds, same rule as ri._lane_bounds
bounds=[round(k*S/N) for k in range(N+1)]
k0a=np.array(bounds[:-1],np.int32); k1a=np.array(bounds[1:],np.int32)
page_sym_off=np.zeros(1,np.int64)               # single page, offset 0
max_lane_bytes=int(2*(S//N)+64)                 # generous

dev="cuda"
ext=load(name="rans_encode",sources=["rans_encode.cu"],verbose=True)
def T(a,dt): return torch.as_tensor(a,dtype=dt,device=dev)
out_bytes=torch.zeros(N*max_lane_bytes,dtype=torch.uint8,device=dev)
out_len=torch.zeros(N,dtype=torch.int32,device=dev)
ext.encode_pages(
    T(freq,torch.int64),T(start,torch.int64),
    T(k0a,torch.int32),T(k1a,torch.int32),T(page_sym_off,torch.int64),
    N,1,max_lane_bytes,out_bytes,out_len)
out_len=out_len.cpu().numpy(); out_bytes=out_bytes.cpu().numpy()

# reconstruct CPU lane byte slices from the fixture
cpu_off=np.concatenate([[0],np.cumsum(lane_lens)])
ok=True
for l in range(N):
    gpu_l=out_bytes[l*max_lane_bytes : l*max_lane_bytes+int(out_len[l])]
    cpu_l=lane_bytes[cpu_off[l]:cpu_off[l+1]]
    same = (len(gpu_l)==len(cpu_l)) and np.array_equal(gpu_l,cpu_l)
    if not same:
        ok=False
        print(f"lane {l}: GPU len {len(gpu_l)} vs CPU len {len(cpu_l)}",
              "| first diff at", int(np.argmax(gpu_l[:min(len(gpu_l),len(cpu_l))]!=cpu_l[:min(len(gpu_l),len(cpu_l))])) if len(gpu_l)==len(cpu_l) else "len mismatch")
print("lane lengths GPU:", out_len.tolist())
print("lane lengths CPU:", lane_lens.tolist())
print("ENCODE KERNEL byte-exact vs CPU:", ok)