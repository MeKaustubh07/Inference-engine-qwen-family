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

## 7. Where a server decode step goes (2026-09-29, commit bbeb6c6, Qwen3.5-2B INT4, batch 8)

Scheduler methods wrapped with wall-clock timers; `decode_batch` followed by `torch.mps.synchronize()` to split
issue time from GPU time. In-process = requests submitted straight to the Scheduler; HTTP = the real app under
uvicorn driven by scripts/loadgen.py at concurrency 8.
```
                                    in-process   HTTP
batch-8 decode step, mean             79.4 ms   84.1 ms
  decode_batch: issue (CPU)            3.8       4.1
  decode_batch: GPU wait               67.3      69.2
  sample (CPU, per request)             6.9       8.8
  emit (detokenize + hand-off)          0.2       0.2
  rest (logits .cpu(), bookkeeping)     1.2       1.7
admit/prefill over the run            5.7 s     5.6 s   (16 prompts of 19-24 tokens, one at a time)
```
There is no unexplained per-step overhead: the gap to the engine-only numbers was prefill, done one prompt at a
time inside the loop (~0.35 s each under load) while every running request waited.

## 8. Prefill time vs prompt length, one forward (same commit)

```
prefill T=1: 25.9 ms   T=8: 122.3   T=16: 175.3   T=23: 222.0   T=32: 292.1   T=33: 255.4
        T=64: 267.4    T=128: 394.2   T=184: 542.0   T=256: 676.3
all 96 quantized linears (no head) at M=8: 49.4 ms  M=23: 146.5  M=32: 203.3  M=33: 183.6  M=64: 193.5  M=184: 336.4
```
8 chat prompts of ~23 tokens: one at a time 8 x 222 = ~1.8 s; packed into one 184-token forward 0.54 s.

## 9. After packed + chunked prefill, the batching window and batched sampling (2026-09-29, same model)

Batched sampling, first placed on the GPU (`bench_sample`: random logits [8, 248320] on MPS, API-default params
temperature 0.7 / top-k 20 / top-p 0.8, medians of 30-40 synced repeats):
```
torch.topk on MPS, 8 x 248320:   k=1 4.53 ms  k=8 4.55  k=16 4.56  k=17 4.56  k=28 4.56  k=64 4.55
torch.topk on CPU, 8 x 248320:   k=1 1.04 ms  k=8 1.04  k=16 1.03  k=17 1.03  k=28 1.03  k=64 1.05   (4 threads)
copy the batch's logits to the CPU: 0.48 ms (whole tensor), 0.99 ms (the [:, :248070] strided slice)
old per-row sampler (copy + sample() per row):  6.65 ms   (6.87 with repetition_penalty 1.1)
sample_batch on MPS logits:                     5.31-7.26 ms (two runs)   (7.87 with repetition_penalty 1.1)
sample_batch(logits.cpu()), copy included:     2.22 ms   (3.02 with repetition_penalty 1.1)
```
MPS top-k costs ~4.5 ms whatever k is, 4x the CPU's; with unified memory the copy is nearly free. So the batched
selection runs on the CPU; the logits are copied whole and sliced after.

Where a batch-8 decode step goes (harness as in section 7, 2 waves of 8 x 64 tokens in-process; the HTTP line is
the real app under loadgen at concurrency 8):
```
                                   before        after, sampling on MPS      after, CPU sampling
                                   in-process    in-process   HTTP           in-process
batch-8 decode step, mean            79.4 ms       78.7 ms    81.0 ms          73.3 ms
  decode_batch: issue (CPU)           3.8           7.8        9.7              4.2
  decode_batch: GPU wait              67.3          63.4       62.1             65.8
  sample                               6.9           7.2        8.4              3.2
  emit                                 0.2           0.3        0.6              0.1
  rest                                 1.2           0.1        0.2              0.0
throughput per wave                                81.7 / 88.1                  86.1 / 93.3 tok/s
```
An intermediate CPU version that copied the strided [:, :vocab] slice measured 75.4 ms per step, 4.7 ms of it
sampling. The issue/GPU split moves with CPU load (the GPU starts while Python is still issuing); their sum is
~70-73 ms in every column. The "before" column is section 7's. The review fixes that came after the last column
(window arrival counting, batch_wait bounds, the sampling fallback's structure) do not touch the decode step or
the prefill pass.

Packed prefill passes (CPU-sampling version, in-process, each wave = 8 chat prompts, 175 tokens, synced):
```
wave 0: 1,332 ms (first pass of this size)   wave 1: 820 ms
```
vs 8 x ~222 ms = ~1.8 s one prompt at a time, and 542 ms for a single 184-token prompt (section 8): attention and
the DeltaNet recurrence run per packed sequence, ~40 ms per extra sequence at this size ((820 - 542) / 7).

One request at a time (in-process, 8 prompts x 2 rounds after a warm-up round, 2 tokens each):
```
                              before     after (MPS-sampling version, window before the review fix;
                                         a lone request waits one 5 ms window in both)
TTFT mean                     220.4 ms   222.9 ms
  batching window (gather)       -         6.2 ms
  prefill forward (synced)    218.1       214.5
  sampling, per call            1.7         1.7
```

## 10. The 2026-09-28 run's "108-110 ms per token" did not reproduce

Per-token time of the last-admitted request in each of the two 8-client waves ((latency - TTFT) / 63, from each
server's `request_finished` log lines; every wave arrived and finished together, so those 63 tokens had no prefill
in between):
```
2026-09-28 published run (raw/server_request_log_final.jsonl)   107.8 / 109.9 ms
same code, re-measured 2026-09-29/30 (kept runs):
  2026-09-29 baseline runs 1 and 2 (loadgen_2b_int4_2026-09-29_before_run*)  77.7 / 78.1,  82.7 / 81.9
  A/B 1 block 1 runs 1 and 4                                      79.6 / 82.4,  89.2 / 90.4
  A/B 1 block 2 run 2                                              80.5 / 81.1
  final A/B runs 1, 4, 6, 7                                        78.1 / 78.4,  78.7 / 78.0,  80.2 / 77.8,  77.0 / 77.7
  (excluded runs: 119.2 / 144.5 and 238.4 / 175.5)
final code, final A/B runs 2, 3, 5, 8                            72.5 / 72.7,  72.9 / 72.6,  72.2 / 72.2,  71.9 / 72.5
```
So the ~28 ms per step blamed on "unprofiled engine-loop work" belonged to that run, which was slower in every
step (cause not established; its throughput, 55.9 tok/s at 8 clients, is the same outlier). In the re-measured
runs the gap to the engine's 116 tok/s (104 vs 69 ms per 8 tokens) is ~20 ms of one-at-a-time prefill and ~15 ms
of per-step work around decode_batch (section 7).
