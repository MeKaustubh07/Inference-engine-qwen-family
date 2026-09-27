"""Week 4: one transformer station vs the HF answer key."""
import sys
import torch

sys.path.insert(0, "src")
from config import ModelConfig
from models.qwen2 import Qwen2Model
from ops import rms_norm
from tokenizer import Tokenizer
from weight_loader import SafetensorsFile

MODEL_DIR = "models/qwen2.5-0.5b"
cfg = ModelConfig.from_json(f"{MODEL_DIR}/config.json")
weights = SafetensorsFile(f"{MODEL_DIR}/model.safetensors")
model = Qwen2Model(cfg, weights)
tokenizer = Tokenizer(f"{MODEL_DIR}/tokenizer.json")

for i in range(5):
    g = torch.load(f"tests/golden/{i}.pt")
    ids = torch.tensor(tokenizer.encode(g["text"]))
    x = rms_norm(model.embed(ids), weights.get("model.layers.0.input_layernorm.weight"), cfg.rms_norm_eps)

    attn = model.self_attn(0, x)
    d_attn = (attn - g["l0_attn"]).abs().max().item()

    block = model.block(0, model.embed(ids))
    d_block = (block - g["l0_out"]).abs().max().item()
    scale = g["l0_out"].abs().max().item()

    ok = d_attn < 1e-3 and d_block < 1e-3 * max(1.0, scale)
    print(f"{i}: tokens={len(ids):2d}  attention max|diff|={d_attn:.2e}  block max|diff|={d_block:.2e} (values up to {scale:.1f})  {'PASS' if ok else 'FAIL'}")
