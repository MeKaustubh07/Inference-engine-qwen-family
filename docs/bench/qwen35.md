# Qwen3.5-2B on the Metal backend: accuracy and speed (bf16 / INT8 / INT4)

The final model: 24 layers (18 Gated DeltaNet + 6 gated attention), hidden 2048, vocabulary 248,320, tied
embedding. MacBook Air M2 (8-core GPU), 8 GB. Code at commit `f9280fa`. Raw output: `docs/bench/raw/ours_2b_*.md`,
`docs/bench/raw/accuracy_2b_vs_hf_bf16.txt`, `docs/bench/raw/tests/test_qwen35_2b.txt`.

## Accuracy vs Hugging Face (tests/test_qwen35_2b.py)

An fp32 reference would need ~7.5 GB for the weights alone, so the answer key is HF `transformers` in **bf16**
(eager attention), on 7 prompts of 2–126 tokens (including Hindi; the 126-token prompt crosses the 64-token
DeltaNet chunk boundary, the 57-token one comes close). Our engine computes decode and short prompts in fp32
activations; bf16 weights are multiplied in bf16 by MPS for prompts longer than 32 tokens. "Flips" count top-1
changes at positions where the reference's top-2 gap is above 0.5 logits (outside bf16 noise). "Greedy tokens
equal" sums, over the 7 prompts, the generated tokens that match HF before the first divergence (10 per prompt).

| weights | resident | KL vs HF bf16 | top-1 flips | greedy tokens equal to HF | continuation of "The capital of France is" |
|---|---:|---:|---:|---:|---|
| bf16 | 3.76 GB | 0.0004 | 0 / 190 | 64 / 70 (one divergence, where our top-2 gap is 0.015 logits) | Paris.\nA. True |
| INT8 (block 32) | 2.00 GB | 0.0007 | 0 / 190 | 70 / 70 | Paris.\nA. True |
| INT4 (asymmetric, block 32, 5 tensors + head INT8) | 1.41 GB | 0.045 | 3 / 190 | 47 / 70 (three divergences, where our top-2 gaps are 0.04–0.11 logits) | Paris. The capital of the |

The INT4 policy (`configs/quant/qwen3.5-2b.json`) comes from `scripts/calibrate_quant.py`: each tensor is
quantized alone and its KL damage measured; the 5 tensors above 0.005 nats stay INT8 (2 DeltaNet `out_proj`,
2 attention `o_proj`, 1 MLP `down_proj`). The tied embedding / output head is always INT8 (a fixed rule from the
Qwen2.5 work; the calibration does not test it).

## Speed (scripts/bench.py)

Prompt rows: after a warm-up at that prompt length, 3 runs are timed and the run with the median time to first
token is reported: prefill = prompt tokens / TTFT, decode = 1 / that run's median time per generated token. Batch rows: the median `decode_batch` step over 32 steps of
one run (after a warm-up), B sequences with 16-token prompts. `decode_batch` is the step the server runs for every
decode, including B = 1; timed alone it leaves out per-token sampling (~1.5 ms per request) and HTTP.

### INT4 (deployment configuration): two runs on the same code

| | run 1 | run 2 |
|---|---:|---:|
| decode, `decode_batch` B = 1 (tok/s) | 50.1 | 49.7 |
| decode after a 16-token prompt, standalone forward (tok/s) | 50.0 | 42.7 |
| decode after a 1024-token prompt, standalone forward (tok/s) | 45.6 | 36.9 |
| prefill 16 / 256 / 1024 tokens (tok/s) | 101 / 398 / 450 | 100 / 395 / 444 |
| TTFT for a 16-token prompt | 158 ms | 160 ms |
| batch 2 / 4 / 8 aggregate decode (tok/s) | 53.6 / 85.4 / 115.8 | 53.5 / 85.4 / 112.7 |

The batched path repeats within ~3%; the standalone single-sequence path varied by up to ~20% on this fanless
machine (45.6 vs 36.9 tok/s after a 1024-token prompt). Bandwidth ceiling for 1.41 GB of weights: ~71 tok/s; `decode_batch` at B = 1 reaches ~71% of it.

### All three weight formats

| | bf16 | INT8 | INT4 |
|---|---:|---:|---:|
| weights resident | 3.76 GB | 2.00 GB | 1.41 GB |
| Metal driver memory (weights + allocator cache; read after the prompt runs, before any paged KV pool exists) | 5,177 MiB | 3,161 MiB | 2,161 MiB |
| decode, `decode_batch` B = 1 (tok/s) | 20.6 | 39.2 | 50.1 |
| % of bandwidth ceiling (100 GB/s ÷ weight bytes) | 77% | 78% | 71% |
| prefill 256 / 1024 tokens (tok/s) | 281 / 105* | 389 / 443 | 398 / 450 |
| batch 8 aggregate decode (tok/s) | 70.5 | **126.4** | 115.8 |
| batch 8 vs batch 1 | 3.4× | 3.2× | 2.3× |

\* 1024-token bf16 prefill took 9.8 s (TTFT 9,757 ms), superlinear against 256 tokens (0.9 s). bf16 holds
5.1 GiB of GPU memory on an 8 GB machine, so paging is the likely cause (not measured). bf16 prefill also uses
MPS's bf16 GEMM, which runs at 1.4 TFLOPS on the M2 vs 2.5 for fp32.

**INT4 vs INT8 is a real trade-off.** Single stream, INT4 is faster (fewer bytes per token). From batch 4, INT8
is clearly faster (126 vs 116 tok/s at batch 8); a likely reason (not profiled) is that sharing each weight read
across rows makes the step compute-bound, where INT4's nibble unpacking costs more than INT8's extra bytes. INT4
remains the deployment default for memory (1.41 vs 2.00 GB leaves room for KV cache and the OS on 8 GB) and
single-user latency.

## How the numbers got here

| change | commit | effect |
|---|---|---|
| fused DeltaNet decode kernels | `c8ac5c8` | 0.8B bf16 decode 17 → 46 tok/s |
| row-chunked quantized prefill (OOM fix) | `32de01f` | 2B INT4 full-sequence logits correct: KL 6.0 → 0.045 (`raw/early_measurements.md`) |
| pooled DeltaNet state + paged attention + 2-rows kernels | `ae478b6` | 2B INT4 B=1 step 26.5 → 20 ms; B=8 step 186 → 69 ms |
| one-pass dequant kernel (bf16) | `34b1b59` | 2B INT4 prefill 256: 128 → 305 tok/s |
| fp32 instead of bf16 expansion | `f9280fa` | 2B INT4 prefill 256: 305 → 395; 1024: 323 → 446 tok/s |

Comparison with llama.cpp and MLX-LM: `docs/bench/compare.md`. Serving under load: `docs/bench/serving.md`.
