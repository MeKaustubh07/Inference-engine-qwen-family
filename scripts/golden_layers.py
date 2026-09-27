"""Answer key: run the official HF model in fp32 on CPU and save intermediate tensors.

Saved per prompt (tests/golden/<i>.pt): ids, embed, l0_norm_in, l0_attn, l0_out, final_norm, logits,
greedy (HF generate, do_sample=False, no repetition penalty, 10 new tokens).
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "models/qwen2.5-0.5b"
PROMPTS = [
    "The capital of France is",
    "Hello world",
    "def fibonacci(n):",
    "The quick brown fox jumps over the lazy dog.",
    "Kaustubh builds an inference engine in Python.",
]

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()

captured = {}
def grab(name):
    def hook(module, inputs, output):
        captured[name] = (output[0] if isinstance(output, tuple) else output).detach()
    return hook

m = model.model
m.embed_tokens.register_forward_hook(grab("embed"))
m.layers[0].input_layernorm.register_forward_hook(grab("l0_norm_in"))
m.layers[0].self_attn.register_forward_hook(grab("l0_attn"))
m.layers[0].register_forward_hook(grab("l0_out"))
m.norm.register_forward_hook(grab("final_norm"))

for i, text in enumerate(PROMPTS):
    ids = tok(text, return_tensors="pt").input_ids            # [1, T]
    with torch.no_grad():
        logits = model(ids).logits                            # [1, T, vocab]
    captured_now = dict(captured)                             # snapshot BEFORE generate() re-fires the hooks
    with torch.no_grad():
        greedy = model.generate(ids, max_new_tokens=10, do_sample=False, repetition_penalty=1.0,
                                temperature=None, top_p=None, top_k=None)[0, ids.shape[1]:]
    torch.save({"text": text, "ids": ids[0], "logits": logits[0], "greedy": greedy,
                **{k: v[0] for k, v in captured_now.items()}}, f"tests/golden/{i}.pt")
    print(f"{i}: {text!r} -> {ids.shape[1]} tokens, logits {tuple(logits[0].shape)}")
