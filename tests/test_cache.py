"""Week 7: KV cache correctness (cached == uncached), cached greedy vs HF, speed, and the GPU backend."""
import sys
import time
import torch

sys.path.insert(0, "src")
from backend.torch_ref import TorchBackend
from config import ModelConfig
from generate import generate_greedy
from models.qwen2 import Qwen2Model
from tokenizer import Tokenizer
from weight_loader import SafetensorsFile

D = "models/qwen2.5-0.5b"
EOS = {151643, 151645}
cfg = ModelConfig.from_json(f"{D}/config.json")
weights = SafetensorsFile(f"{D}/model.safetensors")
tok = Tokenizer(f"{D}/tokenizer.json")
model = Qwen2Model(cfg, weights)                                   # fp32 CPU reference backend

results = []
def check(name, ok):
    results.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}")

# 1. Cached logits == uncached logits at every position (prefill part of the prompt, then one token at a time)
for i in range(5):
    g = torch.load(f"tests/golden/{i}.pt")
    ids = g["ids"]
    full = model.forward(ids)                                      # [T, vocab], no cache
    state = model.new_state(len(ids))
    k = max(1, len(ids) // 2)
    parts = [model.forward(ids[:k], state=state)]                  # prefill k tokens
    for t in range(k, len(ids)):
        parts.append(model.forward(ids[t:t + 1], state=state))     # decode the rest one by one
    inc = torch.cat(parts)
    d = (inc - full).abs().max().item()
    check(f"prompt {i}: cached vs uncached logits max|diff|={d:.1e}, same argmax everywhere",
          d < 1e-3 and torch.equal(inc.argmax(-1), full.argmax(-1)) and state.length == len(ids))

# 1b. A multi-token chunk on a non-empty cache (S > T > 1): prefill 3, then a chunk of 4, then single tokens
ids = torch.load("tests/golden/4.pt")["ids"]
full = model.forward(ids)
state = model.new_state(len(ids))
inc = torch.cat([model.forward(ids[:3], state=state), model.forward(ids[3:7], state=state)] +
                [model.forward(ids[t:t + 1], state=state) for t in range(7, len(ids))])
d = (inc - full).abs().max().item()
check(f"chunked prefill on a non-empty cache (3 + 4 + 1...) matches uncached: max|diff|={d:.1e}", d < 1e-3)

# 2. Cached greedy decoding still equals HF generate() token for token
for i in range(5):
    g = torch.load(f"tests/golden/{i}.pt")
    ours = generate_greedy(model, tok, g["text"], 10, EOS)
    check(f"prompt {i}: cached greedy == HF greedy (all 10 tokens)", ours == g["greedy"].tolist() and len(ours) == 10)

# 3. Speed: 32 new tokens with and without the cache
prompt = "The capital of France is"
t0 = time.perf_counter(); a = generate_greedy(model, tok, prompt, 32, set(), use_cache=False); t_nc = time.perf_counter() - t0
t0 = time.perf_counter(); b = generate_greedy(model, tok, prompt, 32, set(), use_cache=True); t_c = time.perf_counter() - t0
print(f"      CPU fp32, 32 tokens: no cache {t_nc:.2f}s ({32/t_nc:.1f} tok/s)  |  cache {t_c:.2f}s ({32/t_c:.1f} tok/s)  -> {t_nc/t_c:.1f}x")
check("cached and uncached produce the same 32 tokens", a == b)
check("cache is faster", t_c < t_nc)

# 4. Cache memory matches the formula: layers x 2 (K,V) x kv_heads x head_dim x bytes x tokens
state = model.new_state(100); model.forward(torch.tensor(tok.encode(prompt)), state=state)
per_tok = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * 4
check(f"KV cache bytes = {state.bytes_used()} for {state.length} tokens ({per_tok} B/token in fp32, {per_tok // 2} in bf16)",
      state.bytes_used() == per_tok * state.length)

# 5. GPU backends.
#    (a) Cache CORRECTNESS on the GPU, isolated from rounding: with fp32 weights, cached must equal uncached tightly.
#    (b) bf16 backends: prefill (matrix-matrix) and decode (matrix-vector) round differently, so cached vs uncached may
#        differ by as much as each path differs from the fp32 reference. Hold them to that budget, and require the same
#        argmax wherever the top-2 logits are not a near-tie (bf16 can produce exact ties).
if torch.backends.mps.is_available():
    from backend.metal import MetalBackend
    g0 = torch.load("tests/golden/3.pt"); ids = g0["ids"]; ref = g0["logits"]

    def cached_vs_uncached(gpu, st):
        full = gpu.forward(ids).cpu()
        inc = torch.cat([gpu.forward(ids[:2], state=st).cpu()] + [gpu.forward(ids[t:t + 1], state=st).cpu() for t in range(2, len(ids))])
        return full, inc

    gpu = Qwen2Model(cfg, weights, TorchBackend("mps", torch.float32))
    for kind, st in (("contiguous", gpu.new_state(len(ids))), ("paged", gpu.new_paged_pool(16, 4).new_sequence())):
        full, inc = cached_vs_uncached(gpu, st)
        d = (inc - full).abs().max().item()
        check(f"MPS fp32 weights: cached ({kind}) == uncached at every position, max|diff|={d:.1e}", d < 1e-3)
    del gpu

    for name, be in (("MPS bf16", TorchBackend("mps", torch.bfloat16)), ("Metal kernels", MetalBackend())):
        gpu = Qwen2Model(cfg, weights, be)
        for kind, st in (("contiguous", gpu.new_state(len(ids))), ("paged", gpu.new_paged_pool(16, 4).new_sequence())):
            full, inc = cached_vs_uncached(gpu, st)
            gap, budget = (inc - full).abs().max().item(), 1.5 * (full - ref).abs().max().item()
            top2 = full.topk(2, -1).values
            clear = (top2[:, 0] - top2[:, 1]) > 0.5
            same = torch.equal(inc.argmax(-1)[clear], full.argmax(-1)[clear])
            check(f"{name} ({kind}): cached vs uncached {gap:.2f} within bf16 budget {budget:.2f}; argmax equal at "
                  f"{int(clear.sum())}/{len(clear)} non-tied positions", gap <= budget and same)
        out = generate_greedy(gpu, tok, g0["text"], 10, EOS)
        n = next((j for j, (a, b) in enumerate(zip(out, g0["greedy"].tolist())) if a != b), len(out))
        check(f"{name}: 10-token cached greedy matches HF fp32 for {n}/10 tokens", n >= 8)
        del gpu

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
