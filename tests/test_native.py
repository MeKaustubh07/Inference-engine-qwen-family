"""Week 10: the native Objective-C++ Metal runtime runs the same MSL matvec as the torch path, correctly."""
import sys
import time
import torch

sys.path.insert(0, "src")
from native.metal_native import NativeMetal

results = []
def check(name, ok):
    results.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}")

nm = NativeMetal()
print(f"      device: {nm.device_name}")
torch.manual_seed(0)
for N, K in ((896, 896), (4864, 896), (151936, 896)):
    W = (torch.randn(N, K) * 0.05).to(torch.bfloat16)
    x = torch.randn(K)
    w_h, x_h, y_h = nm.upload(W), nm.upload(x), nm.alloc(N * 4)
    nm.matvec(w_h, x_h, y_h, N, K, 1)
    y = nm.view(y_h, N).clone()
    ref = W.float() @ x
    rel = (y - ref).abs().max().item() / ref.abs().max().item()
    gpu_s = nm.matvec(w_h, x_h, y_h, N, K, 50)                    # 50 dispatches, one command buffer
    check(f"native matvec {N}x{K}: rel diff {rel:.1e} | {gpu_s * 1e6:.0f} us GPU/iter ({N * K * 2 / gpu_s / 1e9:.0f} GB/s)", rel < 1e-3)

# the shared buffer really is shared: writing through the torch view is visible to the GPU
x = torch.zeros(896); x_h = nm.upload(x); nm.view(x_h, 896)[:] = 1.0
W = torch.ones(8, 896, dtype=torch.bfloat16); w_h, y_h = nm.upload(W), nm.alloc(8 * 4)
nm.matvec(w_h, x_h, y_h, 8, 896, 1)
check("unified memory: a CPU write through the torch view is seen by the GPU kernel (8 x 896 ones -> 896)",
      torch.allclose(nm.view(y_h, 8), torch.full((8,), 896.0)))

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
