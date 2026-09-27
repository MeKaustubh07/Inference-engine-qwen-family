"""Week 5: full forward pass vs the HF answer key, then greedy generation."""
import sys
import time
import torch

sys.path.insert(0, "src")
from config import ModelConfig
from generate import generate_greedy
from models.qwen2 import Qwen2Model
from tokenizer import Tokenizer
from weight_loader import SafetensorsFile

MODEL_DIR = "models/qwen2.5-0.5b"
cfg = ModelConfig.from_json(f"{MODEL_DIR}/config.json")
model = Qwen2Model(cfg, SafetensorsFile(f"{MODEL_DIR}/model.safetensors"))
tokenizer = Tokenizer(f"{MODEL_DIR}/tokenizer.json")

all_ok = True
for i in range(5):
    g = torch.load(f"tests/golden/{i}.pt")
    logits = model.forward(g["ids"])
    d = (logits - g["logits"]).abs().max().item()
    agree = (logits.argmax(-1) == g["logits"].argmax(-1)).float().mean().item()
    ok = agree == 1.0 and d < 1e-2
    all_ok &= ok
    print(f"{i}: tokens={len(g['ids']):2d}  logits max|diff|={d:.2e} (values up to {g['logits'].abs().max().item():.1f})  top-1 agreement={agree:.0%}  {'PASS' if ok else 'FAIL'}")

t0 = time.perf_counter()
new = generate_greedy(model, tokenizer, "The capital of France is", 8, eos_ids={151643, 151645})
text = tokenizer.decode(new)
print(f"greedy: 'The capital of France is' -> {text!r}  ({len(new)} tokens in {time.perf_counter() - t0:.1f}s)")
print("first word is ' Paris':", "PASS" if text.startswith(" Paris") else "FAIL")
