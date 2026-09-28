# Profiling and kernel-experiment outputs (Qwen3.5-2B unless stated), transcribed from the runs

All on the M2 Air, Metal backend, 2026-09-28. Scripts were ad-hoc (not kept in the repo); each block shows what was
measured and the output.

## 1. First load test vs raw engine: where a batched step spends its time (before the speed-up)

`decode_batch` on INT4 with B sequences (3-token prompts), linears = the 97 quantized matrices run alone at M = B.

```
B=1: decode_batch 23.9 ms = linears 16.8 ms + rest 7.1 ms (7.1 ms/seq)
B=2: decode_batch 49.5 ms = linears 35.9 ms + rest 13.6 ms (6.8 ms/seq)
B=4: decode_batch 82.7 ms = linears 59.9 ms + rest 22.8 ms (5.7 ms/seq)
B=8: decode_batch 188.6 ms = linears 110.6 ms + rest 78.0 ms (9.8 ms/seq)
```

Single quantized linears, batched kernel (one output row per SIMD group) vs M separate single-row calls:

```
gate_up (12288, 2048) int4
  M=1: batched kernel 0.280 ms   1 x single 0.208 ms
  M=2: batched kernel 0.465 ms   2 x single 0.448 ms
  M=4: batched kernel 0.754 ms   4 x single 0.856 ms
  M=8: batched kernel 1.388 ms   8 x single 1.846 ms
  head int8 M=1: 8.433 ms
  head int8 M=8: 44.139 ms
decode_batch B=1: 26.5 ms   B=2: 53.9 ms   B=4: 97.0 ms   B=8: 185.9 ms
```

## 2. Kernel experiments for batched matvec (time per call; "current" = one-row batched kernel)

x tile staged in threadgroup memory (8 rows per threadgroup):
```
int4 12288x2048 M=1: tiled 0.533 ms  current 0.195 ms | M=8: tiled 1.264 ms  current 1.364 ms
int4 2048x6144  M=1: tiled 0.267 ms  current 0.115 ms | M=8: tiled 0.641 ms  current 0.715 ms
int8 248320x2048 M=1: tiled 16.137 ms current 6.613 ms | M=8: tiled 24.767 ms current 35.143 ms
bf16 12288x2048 M=1: tiled 1.206 ms  current 0.576 ms | M=8: tiled 1.256 ms  current 1.718 ms
```
4 output rows per SIMD group:
```
int4 12288x2048 M=1: 0.443 ms (current 0.196) | M=2: 0.627 (0.580) | M=4: 0.715 (0.752) | M=8: 0.975 (1.362)
int4 2048x6144  M=1: 0.207 ms (0.114) | M=2: 0.242 (0.231) | M=4: 0.315 (0.380) | M=8: 0.471 (0.708)
int8 248320x2048 M=1: 9.812 ms (5.801) | M=2: 11.577 (10.916) | M=4: 15.188 (18.935) | M=8: 23.828 (35.206)
bf16 12288x2048 M=1: 0.835 ms (0.599) | M=2: 0.753 (0.578) | M=4: 0.868 (0.917) | M=8: 1.164 (1.720)
```
2 output rows per SIMD group (adopted):
```
int4 12288x2048 M=1: 0.420 ms (current 0.205) | M=2: 0.821 (0.506) | M=4: 0.629 (0.764) | M=8: 0.902 (1.417)
int4 2048x6144  M=1: 0.194 ms (0.162) | M=2: 0.228 (0.228) | M=4: 0.344 (0.383) | M=8: 0.486 (0.717)
int8 248320x2048 M=1: 9.402 ms (7.432) | M=2: 9.014 (10.896) | M=4: 9.342 (18.930) | M=8: 13.966 (35.215)
bf16 12288x2048 M=1: 0.894 ms (0.588) | M=2: 0.718 (0.696) | M=4: 0.863 (0.918) | M=8: 1.084 (1.718)
```
(M = 1 keeps the single-row kernel.)

## 3. Step split after each fix

After pooled DeltaNet state + 2-rows kernels (attention still per sequence):
```
B=1: decode_batch 25.3 ms = linears 17.4 ms + rest 7.8 ms
B=2: decode_batch 44.5 ms = linears 34.4 ms + rest 10.1 ms
B=4: decode_batch 59.1 ms = linears 41.7 ms + rest 17.4 ms
B=8: decode_batch 96.5 ms = linears 62.5 ms + rest 34.0 ms (4.3 ms/seq)
```
After the paged-attention kernel:
```
B=1: decode_batch 20.3 ms = linears 17.0 ms + rest 3.3 ms
B=2: decode_batch 36.6 ms = linears 33.2 ms + rest 3.4 ms
B=4: decode_batch 45.7 ms = linears 41.9 ms + rest 3.8 ms
B=8: decode_batch 68.8 ms = linears 62.3 ms + rest 6.5 ms (0.8 ms/seq)
```
Batched decode equals single-sequence decode (0.8B INT4, 4 prompts, 20 tokens): greedy equal, max |Δlogit| 1.2e-05.

## 4. Server overhead per token (scheduler without HTTP, INT4, 23-token chat prompt)

```
logits -> cpu: 0.33 ms
sample (CPU, top-k 20 / top-p 0.8 over 248k): 1.53 ms   greedy: 0.53 ms
detokenize window: 0.007 ms
```

## 5. Prefill profile (256-token prompt, INT4)

```
forward T=256 (last_only): 1924 ms
all 96 quantized linears at M=256 (dequant + GEMM): 1671 ms
  dequantize only (torch ops): 546 ms
```

MPS GEMM throughput on the M2 (x [M, K] @ w.T):
```
M=64   12288x2048: float32 1.79 TFLOPS | bfloat16 1.33 | float16 1.65
M=64   2048x6144:  float32 1.51 TFLOPS | bfloat16 1.06 | float16 2.14
M=256  12288x2048: float32 2.52 TFLOPS | bfloat16 1.41 | float16 2.56
M=256  2048x6144:  float32 2.38 TFLOPS | bfloat16 1.36 | float16 2.45
M=1024 12288x2048: float32 2.52 TFLOPS | bfloat16 1.43 | float16 2.61
M=1024 2048x6144:  float32 2.47 TFLOPS | bfloat16 1.40 | float16 2.58
```

Prefill after each fix (INT4 / INT8, median of 3):
```
bf16 expansion kernel + bf16 GEMM:  int4 T=256 838 ms (305 tok/s), T=1024 3168 ms (323 tok/s); int8 same ±1%
fp32 expansion kernel + fp32 GEMM:  int4 T=256 648 ms (395 tok/s), T=1024 2296 ms (446 tok/s); int8 654 / 2287 ms
```
Expansion kernels are bit-identical to `QuantTensor.dequantize()` (fp32) and to its bf16 rounding.

## 6. Sanity runs of the comparison engines (before the timed matrix)

```
mlx_lm.generate "The capital of France is" -m 12 --temp 0: Generation 73.6 tok/s, peak memory 1.101 GB
llama-bench Q4_1 -p 16 -n 8 (machine still busy): pp16 292 tok/s, tg8 53.1 tok/s
```
