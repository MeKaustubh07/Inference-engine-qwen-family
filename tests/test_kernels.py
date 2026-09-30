"""Weeks 9-10: every Metal kernel vs the torch reference op, micro-benchmarks, then the model end to end."""
import sys
import time
import torch

sys.path.insert(0, "src")
import ops
from backend.metal import MetalBackend

if not torch.backends.mps.is_available():
    print("SKIP: no MPS device"); sys.exit(0)

mb = MetalBackend()
dev = torch.device("mps")
torch.manual_seed(0)
results = []
def check(name, ok):
    results.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}")
def maxdiff(a, b):
    return (a.float().cpu() - b.float().cpu()).abs().max().item()
def bench(fn, iters=200):
    for _ in range(10): fn()
    torch.mps.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.mps.synchronize(); return (time.perf_counter() - t0) / iters * 1e6   # microseconds

# silu_mul
g, u = torch.randn(1, 4864, device=dev), torch.randn(1, 4864, device=dev)
check(f"silu_mul max|diff|={maxdiff(mb.silu_mul(g, u), ops.silu_mul(g, u)):.1e}", maxdiff(mb.silu_mul(g, u), ops.silu_mul(g, u)) < 1e-5)

# rms_norm, several row counts (decode row and a prefill block)
w = (torch.randn(896) * 0.1).to(torch.bfloat16).to(dev)
for T in (1, 7, 64):
    x = torch.randn(T, 896, device=dev) * 3
    d = maxdiff(mb.rms_norm(x, w, 1e-6), ops.rms_norm(x, w, 1e-6))
    check(f"rms_norm T={T} max|diff|={d:.1e}", d < 1e-4)

# rope: queries (14 heads) and keys (2 heads), large positions too
for H, pos in ((14, torch.arange(5)), (2, torch.tensor([0, 1, 700, 4095, 31999]))):
    x = torch.randn(len(pos), H, 64, device=dev) * 50
    ours, ref = mb.rope(x, pos.to(dev), 1e6), ops.rope(x, pos.to(dev), 1e6)
    rel = maxdiff(ours, ref) / ref.abs().max().item()
    check(f"rope H={H} positions up to {int(pos.max())}: relative max|diff|={rel:.1e}", rel < 1e-4)

# matvec: every projection shape in the model, with and without bias, plus the 151,936-row output head
for N, K, bias in ((896, 896, True), (128, 896, True), (4864, 896, False), (896, 4864, False), (151936, 896, False)):
    W = (torch.randn(N, K) * 0.05).to(torch.bfloat16).to(dev)
    b = (torch.randn(N) * 0.1).to(torch.bfloat16).to(dev) if bias else None
    x = torch.randn(1, K, device=dev)
    ref = (x @ W.float().T) + (b.float() if bias else 0)
    rel = maxdiff(mb.linear(x, W, b), ref) / ref.abs().max().item()
    # Timing with COLD weights: cycle through distinct copies totalling >= 64 MB so reads come from DRAM, as in
    # decode (re-reading one small matrix would be served from the on-chip cache and overstate bandwidth).
    Ws = [W] + [W.clone() for _ in range(max(1, (64 << 20) // (N * K * 2)) - 1)]
    it = iter(range(10 ** 9))
    t_ours = bench(lambda: mb.linear(x, Ws[next(it) % len(Ws)], b), 2 * len(Ws))
    it = iter(range(10 ** 9))
    torch_lin = lambda w: (x.to(torch.bfloat16) @ w.T).float() + b.float() if bias else (x.to(torch.bfloat16) @ w.T).float()
    t_torch = bench(lambda: torch_lin(Ws[next(it) % len(Ws)]), 2 * len(Ws))                 # same work as TorchBackend.linear
    gbps = N * K * 2 / (t_ours * 1e-6) / 1e9
    check(f"matvec {N}x{K}{' +bias' if bias else ''}: rel diff {rel:.1e} | cold weights: ours {t_ours:.0f} us ({gbps:.0f} GB/s) vs torch {t_torch:.0f} us", rel < 1e-3)

# tiled GEMM (prefill) vs PyTorch's tuned GEMM, including ragged sizes not divisible by the tile
for T, N, K in ((7, 896, 896), (64, 4864, 896), (256, 896, 4864), (1024, 1152, 896)):
    W = (torch.randn(N, K) * 0.05).to(torch.bfloat16).to(dev); x = torch.randn(T, K, device=dev)
    ref = x @ W.float().T
    rel = maxdiff(mb.gemm(x, W), ref) / ref.abs().max().item()
    t_ours = bench(lambda: mb.gemm(x, W), 50); t_torch = bench(lambda: (x.to(torch.bfloat16) @ W.T).float(), 50)
    tflops = 2 * T * N * K / (t_ours * 1e-6) / 1e12
    check(f"gemm T={T} {N}x{K}: rel diff {rel:.1e} | tiled {t_ours:.0f} us ({tflops:.2f} TFLOP/s) vs torch {t_torch:.0f} us", rel < 1e-3)

# fused kernels: matvec + residual, and SwiGLU (gate and up rows read in one pass)
W = (torch.randn(896, 4864) * 0.05).to(torch.bfloat16).to(dev); x = torch.randn(1, 4864, device=dev); r = torch.randn(1, 896, device=dev)
ref = x @ W.float().T + r
check(f"matvec + fused residual: rel diff {maxdiff(mb.linear(x, W, residual=r), ref) / ref.abs().max().item():.1e}",
      maxdiff(mb.linear(x, W, residual=r), ref) / ref.abs().max().item() < 1e-3)
Wgu = (torch.randn(2 * 4864, 896) * 0.05).to(torch.bfloat16).to(dev); x = torch.randn(1, 896, device=dev)
yg = x @ Wgu.float().T; ref = ops.silu_mul(yg[:, :4864], yg[:, 4864:])
rel = maxdiff(mb.swiglu(x, Wgu), ref) / ref.abs().max().item()
t_fused = bench(lambda: mb.swiglu(x, Wgu)); t_split = bench(lambda: mb.silu_mul(mb.linear(x, Wgu[:4864]), mb.linear(x, Wgu[4864:])))
check(f"fused swiglu: rel diff {rel:.1e} | fused {t_fused:.0f} us vs 3 separate kernels {t_split:.0f} us", rel < 1e-3)

# decode attention: 1 query, S cached positions, 14 query heads sharing 2 KV heads
for S, scale in ((1, 1), (5, 1), (300, 1), (2048, 1), (300, 30), (2048, 30)):
    # scale 30 pushes scores past exp()'s fp32 range: only a correct max-subtracting softmax stays finite
    q = torch.randn(1, 14, 64, device=dev) * scale; k = torch.randn(S, 2, 64, device=dev); v = torch.randn(S, 2, 64, device=dev)
    d = maxdiff(mb.attention(q, k, v), ops.attention(q, k, v, causal=True))
    t_ours = bench(lambda: mb.attention(q, k, v), 100); t_torch = bench(lambda: ops.attention(q, k, v, causal=True), 100)
    check(f"attention_decode S={S} q-scale {scale}: max|diff|={d:.1e} | ours {t_ours:.0f} us vs torch ops {t_torch:.0f} us", d < 1e-4)

# end to end: the whole model on the Metal backend
from config import ModelConfig
from generate import generate_greedy
from models.qwen2 import Qwen2Model
from tokenizer import Tokenizer
from weight_loader import SafetensorsFile
D = "models/qwen2.5-0.5b"
model = Qwen2Model(ModelConfig.from_json(f"{D}/config.json"), SafetensorsFile(f"{D}/model.safetensors"), mb)
tok = Tokenizer(f"{D}/tokenizer.json")
agree_all, flips, match = [], [], 0
for i in range(5):
    gd = torch.load(f"tests/golden/{i}.pt")
    lg = model.forward(gd["ids"]).cpu()
    same = lg.argmax(-1) == gd["logits"].argmax(-1)
    agree_all.append(same.float().mean().item()); flips.append(int((~same).sum()))
    ours = generate_greedy(model, tok, gd["text"], 10, {151643, 151645})
    hf = gd["greedy"].tolist()
    n = next((j for j, (a, b) in enumerate(zip(ours, hf)) if a != b), min(len(ours), len(hf)))
    match += n
    print(f"      prompt {i}: prefill top-1 agreement {agree_all[-1]:.0%}; greedy matches HF for {n}/10 tokens -> {tok.decode(ours)!r}")
check(f"Metal model: at most 1 near-tie top-1 flip per prompt vs HF fp32 (flips per prompt: {flips})", max(flips) <= 1)
check(f"Metal model: greedy decode matches HF fp32 for {match}/50 tokens (bf16 may diverge late)", match >= 35)

# weights locked in RAM (backend/pinning.py): mlock of the MTLBuffer pages behind MPS tensors. The kernel's user wire
# count on the buffers' memory must go to 1 and back to 0 after unlock, and the GPU must still read the same data.
# (The system-wide wired-page count is no test: the GPU driver itself wires buffers it has just used, and lets go of
# idle ones, which is when the user wire matters.)
import ctypes
from backend.pinning import _ranges, lock_in_memory, unlock
class SubmapInfo64(ctypes.Structure):                    # <mach/vm_region.h> vm_region_submap_info_64, pack(4)
    _pack_ = 4
    _fields_ = [(n, t) for n, t in [
        ("protection", ctypes.c_int), ("max_protection", ctypes.c_int), ("inheritance", ctypes.c_uint),
        ("offset", ctypes.c_uint64), ("user_tag", ctypes.c_uint), ("pages_resident", ctypes.c_uint),
        ("pages_shared_now_private", ctypes.c_uint), ("pages_swapped_out", ctypes.c_uint), ("pages_dirtied", ctypes.c_uint),
        ("ref_count", ctypes.c_uint), ("shadow_depth", ctypes.c_ushort), ("external_pager", ctypes.c_ubyte),
        ("share_mode", ctypes.c_ubyte), ("is_submap", ctypes.c_int), ("behavior", ctypes.c_int),
        ("object_id", ctypes.c_uint32), ("user_wired_count", ctypes.c_ushort), ("pages_reusable", ctypes.c_uint)]]
libc = ctypes.CDLL(None)
def user_wired(addr: int) -> int:
    """user_wired_count of the VM map entry holding addr (descending into submaps)."""
    a, size, depth, info = ctypes.c_uint64(addr), ctypes.c_uint64(0), ctypes.c_uint(0), SubmapInfo64()
    while True:
        cnt = ctypes.c_uint(ctypes.sizeof(SubmapInfo64) // 4)
        kr = libc.mach_vm_region_recurse(ctypes.c_uint.in_dll(libc, "mach_task_self_"), ctypes.byref(a),
                                         ctypes.byref(size), ctypes.byref(depth), ctypes.byref(info), ctypes.byref(cnt))
        if kr != 0 or not info.is_submap:
            return -1 if kr != 0 else info.user_wired_count
        depth.value += 1
ws = [torch.randn(4096, 4096, device=dev) for _ in range(4)]            # 4 x 64 MiB
xv = torch.randn(4096, device=dev)
ref = [w @ xv for w in ws]
torch.mps.synchronize()
need = sum(w.untyped_storage().nbytes() for w in ws)
addrs = [p for p, _ in _ranges(ws)]
before = [user_wired(p) for p in addrs]
locked = lock_in_memory(ws + [ws[0][:10]])                               # a view: its buffer counted once
during = [user_wired(p) for p in addrs]
same = all(maxdiff(w @ xv, r) == 0 for w, r in zip(ws, ref))
unlock(ws)
after = [user_wired(p) for p in addrs]
check(f"lock_in_memory: {locked / 2**20:.0f} MiB locked for 4 x 64 MiB MPS tensors (+ a view of one); user wire "
      f"count of their memory {before} -> {during} -> {after} after unlock; GPU results unchanged",
      need <= locked < need + 4 * 2**20 and before == [0] * 4 and during == [1] * 4 and after == [0] * 4 and same)
del ws, ref

print(f"\n{sum(results)}/{len(results)} kernel checks passed")
sys.exit(0 if all(results) else 1)
