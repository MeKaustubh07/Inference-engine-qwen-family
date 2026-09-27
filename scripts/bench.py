"""Benchmark harness: TTFT, TPOT, prefill/decode throughput, peak memory, and the bandwidth ceiling.

usage: bench.py --backend cpu|mps [--prompt-lens 16,256,1024] [--new 32] [--no-cache] [--out docs/bench/x.md]
"""
import argparse
import resource
import sys
import time

import torch

sys.path.insert(0, "src")
from backend.torch_ref import TorchBackend
from config import ModelConfig
from models.qwen2 import Qwen2Model
from tokenizer import Tokenizer
from weight_loader import SafetensorsFile

D = "models/qwen2.5-0.5b"
TEXT = ("The history of computing is a story of abstraction. Each generation of engineers built tools that "
        "hid the details of the layer below, so the next generation could think in bigger pieces. ")
BANDWIDTH = 100e9   # M2 unified memory, bytes/s


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20      # bytes on macOS


def prompt_ids(tok: Tokenizer, n: int) -> list[int]:
    ids: list[int] = []
    while len(ids) < n:
        ids += tok.encode(TEXT)
    return ids[:n]


def timed(fn, sync):
    sync(); t0 = time.perf_counter(); out = fn(); sync()
    return out, time.perf_counter() - t0


def bench_one(model, ids: list[int], new: int, use_cache: bool, sync) -> dict:
    x = torch.tensor(ids)
    if use_cache:
        state = model.new_state(len(ids) + new)
        logits, ttft = timed(lambda: model.forward(x, state=state, last_only=True), sync)
        nxt = int(logits[0].argmax())
        steps = []
        for _ in range(new - 1):
            logits, dt = timed(lambda: model.forward(torch.tensor([nxt]), state=state, last_only=True), sync)
            nxt = int(logits[0].argmax()); steps.append(dt)
    else:
        seq = list(ids)
        logits, ttft = timed(lambda: model.forward(torch.tensor(seq), last_only=True), sync)
        seq.append(int(logits[0].argmax()))
        steps = []
        for _ in range(new - 1):
            logits, dt = timed(lambda: model.forward(torch.tensor(seq), last_only=True), sync)
            seq.append(int(logits[0].argmax())); steps.append(dt)
    tpot = sum(steps) / len(steps)
    return {"prompt": len(ids), "cache": use_cache, "ttft_ms": ttft * 1e3, "tpot_ms": tpot * 1e3,
            "prefill_tps": len(ids) / ttft, "decode_tps": 1 / tpot}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["cpu", "mps"], default="cpu")
    ap.add_argument("--prompt-lens", default="16,256,1024")
    ap.add_argument("--new", type=int, default=32)
    ap.add_argument("--no-cache", action="store_true", help="also measure recompute-everything decoding")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    backend = TorchBackend("cpu", torch.float32) if a.backend == "cpu" else TorchBackend("mps", torch.bfloat16)
    cfg = ModelConfig.from_json(f"{D}/config.json")
    model = Qwen2Model(cfg, SafetensorsFile(f"{D}/model.safetensors"), backend)
    tok = Tokenizer(f"{D}/tokenizer.json")
    sync = backend.sync if hasattr(backend, "sync") else (lambda: None)

    bench_one(model, prompt_ids(tok, 8), 4, True, sync)                       # warm-up: weights resident, kernels compiled
    weight_bytes = sum(model._w(n).numel() * model._w(n).element_size() for n in model.weights.tensor_names())
    ceiling = BANDWIDTH / weight_bytes

    rows = []
    for n in [int(v) for v in a.prompt_lens.split(",")]:
        rows.append(bench_one(model, prompt_ids(tok, n), a.new, True, sync))
        if a.no_cache:
            rows.append(bench_one(model, prompt_ids(tok, n), min(a.new, 8), False, sync))

    lines = [f"### {backend.name} — Qwen2.5-0.5B, {a.new} new tokens, M2 8 GB",
             "", f"Weights resident: {weight_bytes / 1e9:.2f} GB → bandwidth ceiling ≈ {ceiling:.0f} tok/s "
             f"(every decode step reads every weight once at ~100 GB/s). Peak RSS: {peak_rss_mb():.0f} MB.", "",
             "| prompt tokens | KV cache | TTFT (ms) | TPOT (ms) | prefill tok/s | decode tok/s | % of ceiling |",
             "|---:|:---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['prompt']} | {'yes' if r['cache'] else 'no'} | {r['ttft_ms']:.0f} | {r['tpot_ms']:.1f} | "
                     f"{r['prefill_tps']:.0f} | {r['decode_tps']:.1f} | {100 * r['decode_tps'] / ceiling:.0f}% |")
    report = "\n".join(lines) + "\n"
    print(report)
    if a.out:
        with open(a.out, "a") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
