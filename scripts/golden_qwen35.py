"""Answer key for Qwen3.5: HF model in fp32 with eager attention, ids from the model's own tokenizer.json.

Saved per prompt (tests/golden_qwen35/<i>.pt): text, ids, embed, l{i}_out for all 24 layers, final_norm, logits,
greedy (10 tokens, do_sample=False, no repetition penalty).
Note: AutoTokenizer resolves this checkpoint to Qwen2Tokenizer, whose regex differs from tokenizer.json on
combining marks, so ids come from tokenizer.json directly (the engine's reference).
"""
import sys

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

D = sys.argv[1] if len(sys.argv) > 1 else "models/qwen3.5-0.8b"
OUT = sys.argv[2] if len(sys.argv) > 2 else "tests/golden_qwen35"
PROMPTS = [
    "The capital of France is",
    "Hello world",
    "def fibonacci(n):",
    "The quick brown fox jumps over the lazy dog.",
    "नमस्ते दुनिया, यह एक परीक्षण है।",
]
tok = Tokenizer.from_file(f"{D}/tokenizer.json")
model = AutoModelForCausalLM.from_pretrained(D, dtype=torch.float32, attn_implementation="eager").eval()
assert type(model).__name__ == "Qwen3_5ForCausalLM", type(model).__name__
m = model.model
cap = {}
def hook(name):
    def h(mod, args, out):
        cap[name] = (out[0] if isinstance(out, tuple) else out).detach().clone()
    return h
m.embed_tokens.register_forward_hook(hook("embed"))
for i, layer in enumerate(m.layers):
    layer.register_forward_hook(hook(f"l{i}_out"))
m.norm.register_forward_hook(hook("final_norm"))

for n, text in enumerate(PROMPTS):
    ids = torch.tensor([tok.encode(text).ids])
    with torch.no_grad():
        logits = model(ids, use_cache=False).logits
    snap = {k: v[0] for k, v in cap.items()}                      # snapshot BEFORE generate() re-fires the hooks
    with torch.no_grad():
        greedy = model.generate(ids, max_new_tokens=10, do_sample=False, repetition_penalty=1.0, temperature=None,
                                top_p=None, top_k=None, eos_token_id=[248046, 248044])[0, ids.shape[1]:]
    torch.save({"text": text, "ids": ids[0], "logits": logits[0], "greedy": greedy, **snap}, f"{OUT}/{n}.pt")
    print(f"{n}: {text!r} -> {ids.shape[1]} tokens | greedy {tok.decode(greedy.tolist())!r}")
