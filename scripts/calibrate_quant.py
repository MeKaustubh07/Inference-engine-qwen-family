"""Measure INT4 sensitivity per weight tensor and write a mixed-precision policy.

For each 2-D weight, quantize ONLY that tensor to INT4 (everything else bf16) and measure the KL divergence of the
next-token distribution against the HF fp32 goldens, separately at position 0 (the attention-sink first token) and
at later positions. Tensors whose damage exceeds the threshold stay INT8 in INT4 mode.

usage: calibrate_quant.py [--threshold 0.005] [--out configs/quant/qwen2.5-0.5b.json]
"""
import argparse
import json
import sys

import torch

sys.path.insert(0, "src")
from backend.torch_ref import TorchBackend
from config import ModelConfig
from models.qwen2 import Qwen2Model
from ops import softmax
from quant import quantize
from weight_loader import SafetensorsFile

D = "models/qwen2.5-0.5b"


def kl_scores(model, goldens, vocab):
    pos0, later = 0.0, 0.0
    for g in goldens:
        p = softmax(g["logits"][:, :vocab]); q = softmax(model.forward(g["ids"]).cpu()[:, :vocab])
        kl = (p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum(-1)
        pos0 += kl[0].item() / len(goldens); later += kl[1:].mean().item() / len(goldens)
    return pos0, later


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.005, help="KL damage (nats) above which a tensor stays INT8")
    ap.add_argument("--out", default="configs/quant/qwen2.5-0.5b.json")
    a = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    m = Qwen2Model(ModelConfig.from_json(f"{D}/config.json"), SafetensorsFile(f"{D}/model.safetensors"),
                   TorchBackend(device, torch.bfloat16))
    goldens = [torch.load(f"tests/golden/{i}.pt") for i in range(5)]
    base0, base_later = kl_scores(m, goldens, 151665)                     # also fills the resident cache
    damage = {}
    for key in [k for k in list(m._cache) if m._cache[k].ndim == 2 and "embed" not in k]:
        orig = m._cache[key]
        m._cache[key] = quantize(orig.float().cpu(), "int4").dequantize().to(device, torch.bfloat16)
        k0, kl = kl_scores(m, goldens, 151665)
        damage[key] = {"pos0": round(k0 - base0, 4), "later": round(kl - base_later, 4)}
        m._cache[key] = orig
    keep = sorted(k for k, v in damage.items() if max(v["pos0"], v["later"]) > a.threshold)
    policy = {"model": "Qwen2.5-0.5B-Instruct", "scheme": "int4", "threshold_nats": a.threshold,
              "keep_int8": keep, "always_int8": ["model.embed_tokens.weight"],
              "damage": dict(sorted(damage.items(), key=lambda kv: -kv[1]["pos0"]))}
    json.dump(policy, open(a.out, "w"), indent=1)
    print(f"{len(keep)} of {len(damage)} tensors stay INT8: {keep}\nwrote {a.out}")


if __name__ == "__main__":
    main()
