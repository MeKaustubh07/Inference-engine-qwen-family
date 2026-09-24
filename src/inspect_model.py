"""Visitor script: ask the loader questions and print the answers."""
import json
import re
import resource
import time

import torch

from weight_loader import SafetensorsFile

MODEL_DIR = "models/qwen2.5-0.5b"
config = json.load(open(f"{MODEL_DIR}/config.json"))
t0 = time.perf_counter()
f = SafetensorsFile(f"{MODEL_DIR}/model.safetensors")
print(f"opened in {(time.perf_counter() - t0) * 1000:.1f} ms; tensor count: {len(f.tensor_names())}")

layers = {int(m.group(1)) for n in f.tensor_names() if (m := re.match(r"^model\.layers\.(\d+)\.", n))}
print("check 1 (layers):     ", "PASS" if len(layers) == config["num_hidden_layers"] else "FAIL")
emb = f.info("model.embed_tokens.weight")
print("check 2 (embed shape):", "PASS" if emb["shape"] == [config["vocab_size"], config["hidden_size"]] else "FAIL")
print("check 3 (all BF16):   ", "PASS" if all(f.info(n)["dtype"] == "BF16" for n in f.tensor_names()) else "FAIL")
print("check 4 (integrity):  ", "PASS" if f.integrity_check() else "FAIL")

norm = f.get("model.layers.0.input_layernorm.weight")
print("norm:", tuple(norm.shape), norm.dtype, "first 4:", norm[:4].float().tolist())

t0 = time.perf_counter()
e = f.get("model.embed_tokens.weight")
print(f"embed: {tuple(e.shape)} {e.dtype} {e.numel():,} elements, 'loaded' in {(time.perf_counter() - t0) * 1000:.3f} ms (it's a view)")

# Touch one row: this is the moment the OS actually pulls those bytes from disk
row = e[9707].float()
print("row 9707 (token 'Hello') first 4:", row[:4].tolist())

# Move to the GPU: unified memory means this is cheap, and stays bf16
g = e.to("mps")
print("on GPU:", g.device, g.dtype)

rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
print(f"peak resident memory of this process: {rss:.0f} MB")
