# Weight quantization (INT8 / INT4) on Qwen2.5-0.5B

### metal-kernels-bf16 — Qwen2.5-0.5B, 32 new tokens, M2 8 GB

Weights resident: 0.99 GB → bandwidth ceiling ≈ 101 tok/s (every decode step reads every weight once at ~100 GB/s).
Memory: peak CPU RSS 762 MiB, Metal driver 2065 MiB. Each row: median of 3 runs after a warm-up at that prompt length.

| prompt tokens | KV cache | TTFT (ms) | TPOT (ms) | prefill tok/s | decode tok/s | % of ceiling |
|---:|:---:|---:|---:|---:|---:|---:|
| 16 | yes | 35 | 15.4 | 456 | 64.9 | 64% |
| 256 | yes | 209 | 16.4 | 1223 | 61.1 | 60% |
| 1024 | yes | 1130 | 16.6 | 906 | 60.3 | 60% |

### metal-kernels-int8 — Qwen2.5-0.5B, 32 new tokens, M2 8 GB

Weights resident: 0.52 GB → bandwidth ceiling ≈ 190 tok/s (every decode step reads every weight once at ~100 GB/s).
Memory: peak CPU RSS 2405 MiB, Metal driver 1105 MiB. Each row: median of 3 runs after a warm-up at that prompt length.

| prompt tokens | KV cache | TTFT (ms) | TPOT (ms) | prefill tok/s | decode tok/s | % of ceiling |
|---:|:---:|---:|---:|---:|---:|---:|
| 16 | yes | 164 | 11.8 | 98 | 85.0 | 45% |
| 256 | yes | 252 | 10.8 | 1015 | 92.9 | 49% |
| 1024 | yes | 907 | 11.3 | 1129 | 88.8 | 47% |

### metal-kernels-int4 — Qwen2.5-0.5B, 32 new tokens, M2 8 GB

Weights resident: 0.41 GB → bandwidth ceiling ≈ 245 tok/s (every decode step reads every weight once at ~100 GB/s).
Memory: peak CPU RSS 2405 MiB, Metal driver 1105 MiB. Each row: median of 3 runs after a warm-up at that prompt length.

| prompt tokens | KV cache | TTFT (ms) | TPOT (ms) | prefill tok/s | decode tok/s | % of ceiling |
|---:|:---:|---:|---:|---:|---:|---:|
| 16 | yes | 333 | 10.1 | 48 | 99.3 | 40% |
| 256 | yes | 445 | 10.9 | 576 | 92.0 | 37% |
| 1024 | yes | 1083 | 10.0 | 946 | 99.8 | 41% |

## Accuracy (tests/test_quant.py)

Reference: HF fp32 goldens. Perplexity on a fixed 70-token paragraph; KL and top-1 agreement over 5 golden prompts.

| weights | resident | perplexity | KL vs fp32 | top-1 agreement | greedy after "The capital of France is" |
|---|---:|---:|---:|---:|---|
| bf16 | 0.99 GB | 11.34 | 0.0009 | 98% | Paris. It is the largest city in Europe and |
| INT8 (symmetric, block 32) | 0.52 GB | 11.30 | 0.019 | 98% | *identical to bf16* |
| INT4 symmetric (first attempt) | 0.37 GB | 14.54 | 2.26 | 62% | Paris. It is the largest city in Europe, |
| INT4 asymmetric + INT8 embedding | 0.37 GB | 11.50 | 1.85 | 64% | Paris. Paris is the capital of which country? |
| **INT4 asymmetric + calibrated policy** | **0.41 GB** | **11.20** | **0.114** | **94%** | Paris. Paris is the capital of France. Paris |

## How INT4 got there

1. **Symmetric INT4 costs +28% perplexity.** Scale = absmax / 7 spends half the 16 levels on the side of zero a block barely uses. Switching to **asymmetric** (scale = (max − min) / 15, plus a per-block fp16 minimum) brought perplexity to within 1.4% of bf16, and keeping the tied embedding / output head at INT8 (it scores all 151,936 tokens) helped further. Inside the kernel the minimum folds in as `scale·dot(q, x) + min·sum(x)`.
2. **Perplexity hid a first-token problem.** Per-position KL showed 6.8–8.2 nats at position 0 and 0.01–0.9 elsewhere. Small LLMs route huge activations through the first token (an "attention sink"), which makes a handful of weights extremely sensitive.
3. **Measured sensitivity instead of a guess.** `scripts/calibrate_quant.py` quantizes one tensor at a time to INT4 (the rest bf16) and records the KL damage. Four tensors (layer 21's MLP, `down_proj` of layers 2–3) each added 3–5 nats; the median tensor added 0.0004. This matches the massive-activation pattern described for small transformers.
4. **Threshold chosen at the knee of a sweep:**

| keep INT8 if damage > | tensors INT8 | resident | KL vs fp32 | top-1 agreement |
|---:|---:|---:|---:|---:|
| 0.1 | 4 | 0.378 GB | 0.169 | 86% |
| 0.01 | 9 | 0.395 GB | 0.124 | 87% |
| **0.005 (default)** | **19** | **0.408 GB** | **0.114** | **94%** |
| 0.002 | 37 | 0.437 GB | 0.084 | 94% |
| 0.001 | 62 | 0.497 GB | 0.035 | 98% |

The policy lives in `configs/quant/qwen2.5-0.5b.json` (names are fused tensors: `qkv`, `gate_up`; every component of a fused tensor gets the same scheme).

## Speed

- **Decode gets faster as bytes shrink:** ~61–65 tok/s (bf16) → ~85–93 (INT8) → ~92–100 (INT4). Fewer bytes per token is the whole point: decode reads every weight once per token.
- **The quantized kernels sit further below their (higher) ceilings** (40–49% vs 60–64%): unpacking nibbles and applying scales adds ALU work per byte, and the fixed per-token overhead is a larger share of a shorter step.
- **INT4 prefill is slower** (TTFT 333 ms at 16 tokens vs 35 ms for bf16) because prefill dequantizes each weight matrix on the fly before PyTorch's GEMM. A quantized GEMM kernel would remove this; it is the main open item.
- `.qt` files (`scripts/quantize.py`, streamed one tensor at a time): 0.41 GB INT4 / 0.52 GB INT8 vs 0.99 GB safetensors. Loading the INT4 file takes 1.4 s vs 3.4 s for quantizing at startup, and produces bit-identical logits.
