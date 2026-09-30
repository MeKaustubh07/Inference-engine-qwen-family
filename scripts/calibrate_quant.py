"""Measure INT4 sensitivity per weight tensor and write a mixed-precision policy (any supported model).

For each 2-D weight, quantize ONLY that tensor to INT4 (everything else bf16) and measure the KL divergence of the
next-token distribution against reference logits (HF goldens), separately at position 0 (the attention-sink first
token) and at later positions. Tensors whose damage exceeds the threshold stay INT8 in INT4 mode.

usage: calibrate_quant.py --model qwen3.5-0.8b|qwen3.5-2b [--golden-dir DIR] [--threshold 0.005] [--out F]
"""
import argparse
import glob
import json
import sys

import torch

sys.path.insert(0, "src")
from engine import load_engine
from ops import softmax
from quant import quantize


def kl_scores(model, goldens, vocab):
    pos0, later = 0.0, 0.0
    for g in goldens:
        p = softmax(g["logits"][:, :vocab].float()); q = softmax(model.forward(g["ids"]).cpu()[:, :vocab])
        kl = (p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum(-1)
        pos0 += kl[0].item() / len(goldens); later += kl[1:].mean().item() / len(goldens)
    return pos0, later


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--golden-dir", default=None, help="HF reference logits (default: the model's golden folder)")
    ap.add_argument("--threshold", type=float, default=0.005, help="KL damage (nats) above which a tensor stays INT8")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"configs/quant/{a.model}.json"
    golden_dir = a.golden_dir or {"qwen3.5-0.8b": "tests/golden_qwen35", "qwen3.5-2b": "tests/golden_qwen35_2b"}[a.model]
    files = sorted(glob.glob(f"{golden_dir}/*.pt"))
    if not files:                                                        # else every KL is 0 and nothing is kept INT8
        sys.exit(f"no goldens in {golden_dir}; run scripts/golden_qwen35.py")
    eng = load_engine(a.model, "mps")                                    # bf16 weights on the GPU
    m, vocab = eng.model, eng.tokenizer.vocab_size()
    goldens = [torch.load(f) for f in files]
    base0, base_later = kl_scores(m, goldens, vocab)                      # also fills the resident cache
    keys = [k for k, w in m._cache.items() if isinstance(w, torch.Tensor) and w.ndim == 2
            and w.dtype == torch.bfloat16 and "embed" not in k]
    damage = {}
    for n, key in enumerate(keys):
        orig = m._cache[key]
        m._cache[key] = quantize(orig.float().cpu(), "int4").dequantize().to(orig.device, torch.bfloat16)
        k0, kl = kl_scores(m, goldens, vocab)
        damage[key] = {"pos0": round(k0 - base0, 4), "later": round(kl - base_later, 4)}
        m._cache[key] = orig
        print(f"\r{n + 1}/{len(keys)} tensors measured", end="", flush=True)
    keep = sorted(k for k, v in damage.items() if max(v["pos0"], v["later"]) > a.threshold)
    policy = {"model": a.model, "scheme": "int4", "threshold_nats": a.threshold, "reference": golden_dir,
              "baseline_kl_bf16": {"pos0": round(base0, 4), "later": round(base_later, 4)},
              "keep_int8": keep, "always_int8": ["embed_tokens.weight"],
              "damage": dict(sorted(damage.items(), key=lambda kv: -max(kv[1].values())))}
    json.dump(policy, open(out, "w"), indent=1)
    print(f"\n{len(keep)} of {len(damage)} tensors stay INT8\nwrote {out}")


if __name__ == "__main__":
    main()
