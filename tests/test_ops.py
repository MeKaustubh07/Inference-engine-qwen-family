"""Week 3: embedding lookup + RMSNorm vs the HF answer key."""
import sys
import torch

sys.path.insert(0, "src")
from config import ModelConfig
from models.qwen2 import Qwen2Model
from ops import rms_norm
from weight_loader import SafetensorsFile
from tokenizer import Tokenizer

MODEL_DIR = "models/qwen2.5-0.5b"
cfg = ModelConfig.from_json(f"{MODEL_DIR}/config.json")
weights = SafetensorsFile(f"{MODEL_DIR}/model.safetensors")
model = Qwen2Model(cfg, weights)
tokenizer = Tokenizer(f"{MODEL_DIR}/tokenizer.json")
norm_w = weights.get("model.layers.0.input_layernorm.weight")

all_ok = True
for i in range(5):
    g = torch.load(f"tests/golden/{i}.pt")
    ids = torch.tensor(tokenizer.encode(g["text"]))
    assert torch.equal(ids, g["ids"]), f"tokenizer mismatch on {g['text']!r}"

    ours_embed = model.embed(ids)
    d_embed = (ours_embed - g["embed"]).abs().max().item()

    ours_norm = rms_norm(ours_embed, norm_w, cfg.rms_norm_eps)
    d_norm = (ours_norm - g["l0_norm_in"]).abs().max().item()

    ok = d_embed < 1e-6 and d_norm < 1e-4
    all_ok &= ok
    print(f"{i}: tokens={len(ids):2d}  embed max|diff|={d_embed:.2e}  rmsnorm max|diff|={d_norm:.2e}  {'PASS' if ok else 'FAIL'}")

sys.exit(0 if all_ok else 1)
