"""Benchmark harness: TTFT, TPOT, prefill/decode throughput, peak memory, and the bandwidth ceiling.

usage: bench.py --backend cpu|mps|metal [--prompt-lens 16,256,1024] [--new 32] [--no-cache] [--out docs/bench/x.md]
"""
import argparse
import resource
import sys
import time

import torch

sys.path.insert(0, "src")
from backend.metal import MetalBackend
from backend.torch_ref import TorchBackend
from config import ModelConfig
from models.qwen2 import Qwen2Model
from tokenizer import Tokenizer
from weight_loader import SafetensorsFile

D = "models/qwen2.5-0.5b"
TEXT = ("The history of computing is a story of abstraction. Each generation of engineers built tools that "
        "hid the details of the layer below, so the next generation could think in bigger pieces. ")
BANDWIDTH = 100e9   # M2 unified memory, bytes/s


def peak_rss_mib() -> float:
    """Peak resident set of the CPU process (MiB). Does NOT include memory the Metal driver holds for the GPU."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20      # ru_maxrss is bytes on macOS


def gpu_mib() -> float:
    return torch.mps.driver_allocated_memory() / 2**20 if torch.backends.mps.is_available() else 0.0


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
    tpot = sorted(steps)[len(steps) // 2]                       # median decode step
    return {"prompt": len(ids), "cache": use_cache, "ttft_ms": ttft * 1e3, "tpot_ms": tpot * 1e3,
            "prefill_tps": len(ids) / ttft, "decode_tps": 1 / tpot}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["cpu", "mps", "metal", "metal-int8", "metal-int4"], default="cpu")
    ap.add_argument("--prompt-lens", default="16,256,1024")
    ap.add_argument("--new", type=int, default=32)
    ap.add_argument("--no-cache", action="store_true", help="also measure recompute-everything decoding")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.new < 2:
        ap.error("--new must be >= 2 (TPOT needs at least one decode step after the first token)")

    backend = {"cpu": lambda: TorchBackend("cpu", torch.float32), "mps": lambda: TorchBackend("mps", torch.bfloat16),
               "metal": MetalBackend, "metal-int8": lambda: MetalBackend("int8"),
               "metal-int4": lambda: MetalBackend("int4", "configs/quant/qwen2.5-0.5b.json")}[a.backend]()
    cfg = ModelConfig.from_json(f"{D}/config.json")
    model = Qwen2Model(cfg, SafetensorsFile(f"{D}/model.safetensors"), backend)
    tok = Tokenizer(f"{D}/tokenizer.json")
    sync = backend.sync if hasattr(backend, "sync") else (lambda: None)

    bench_one(model, prompt_ids(tok, 8), 4, True, sync)                       # warm-up: weights resident, kernels compiled
    weight_bytes = sum(w.nbytes if hasattr(w, "scheme") else w.numel() * w.element_size() for w in model._cache.values())
    ceiling = BANDWIDTH / weight_bytes

    def median_run(n: int, new: int, use_cache: bool) -> dict:
        bench_one(model, prompt_ids(tok, n), 2, use_cache, sync)            # warm-up at THIS shape (graphs, allocator)
        runs = [bench_one(model, prompt_ids(tok, n), new, use_cache, sync) for _ in range(3)]
        return sorted(runs, key=lambda r: r["ttft_ms"])[1]                 # median of 3 by TTFT

    rows = []
    for n in [int(v) for v in a.prompt_lens.split(",")]:
        rows.append(median_run(n, a.new, True))
        if a.no_cache:
            rows.append(median_run(n, min(a.new, 6), False))

    lines = [f"### {backend.name} — Qwen2.5-0.5B, {a.new} new tokens, M2 8 GB",
             "", f"Weights resident: {weight_bytes / 1e9:.2f} GB → bandwidth ceiling ≈ {ceiling:.0f} tok/s "
             f"(every decode step reads every weight once at ~100 GB/s).", f"Memory: peak CPU RSS {peak_rss_mib():.0f} MiB, "
             f"Metal driver {gpu_mib():.0f} MiB. Each row: median of 3 runs after a warm-up at that prompt length.", "",
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
