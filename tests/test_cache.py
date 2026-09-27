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

# 2. Cached greedy decoding still equals HF generate() token for token
for i in range(5):
    g = torch.load(f"tests/golden/{i}.pt")
    ours = generate_greedy(model, tok, g["text"], 10, EOS)
    check(f"prompt {i}: cached greedy == HF greedy", ours == g["greedy"].tolist()[: len(ours)])

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

# 5. GPU backend: bf16 weights on MPS
if torch.backends.mps.is_available():
    gpu = Qwen2Model(cfg, weights, TorchBackend("mps", torch.bfloat16))
    g0 = torch.load("tests/golden/0.pt")
    lg = gpu.forward(g0["ids"]).cpu()
    d = (lg - g0["logits"]).abs().max().item()
    agree = (lg.argmax(-1) == g0["logits"].argmax(-1)).float().mean().item()
    check(f"MPS bf16 logits vs HF fp32: max|diff|={d:.2f}, top-1 agreement {agree:.0%}", agree >= 0.8)
    generate_greedy(gpu, tok, prompt, 4, set())                    # warm-up (kernel compilation, weight upload)
    t0 = time.perf_counter(); out = generate_greedy(gpu, tok, prompt, 32, set()); gpu.b.sync(); t_g = time.perf_counter() - t0
    print(f"      MPS bf16, 32 tokens with cache: {t_g:.2f}s ({32/t_g:.1f} tok/s) -> {tok.decode(out)!r}")
    check("MPS output starts with ' Paris'", tok.decode(out).startswith(" Paris"))

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
