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
    t_ours = bench(lambda: mb.linear(x, W, b)); t_torch = bench(lambda: (x.to(torch.bfloat16) @ W.T).float() + (b.float() if bias else 0))
    gbps = N * K * 2 / (t_ours * 1e-6) / 1e9
    check(f"matvec {N}x{K}{' +bias' if bias else ''}: rel diff {rel:.1e} | ours {t_ours:.0f} us ({gbps:.0f} GB/s) vs torch {t_torch:.0f} us", rel < 1e-3)

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
for S in (1, 5, 300, 2048):
    q = torch.randn(1, 14, 64, device=dev); k = torch.randn(S, 2, 64, device=dev); v = torch.randn(S, 2, 64, device=dev)
    d = maxdiff(mb.attention(q, k, v), ops.attention(q, k, v, causal=True))
    t_ours = bench(lambda: mb.attention(q, k, v), 100); t_torch = bench(lambda: ops.attention(q, k, v, causal=True), 100)
    check(f"attention_decode S={S}: max|diff|={d:.1e} | ours {t_ours:.0f} us vs torch ops {t_torch:.0f} us", d < 1e-4)

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

print(f"\n{sum(results)}/{len(results)} kernel checks passed")
sys.exit(0 if all(results) else 1)
