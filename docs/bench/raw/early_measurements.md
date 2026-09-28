# Measurements recorded outside the repo until now (transcribed, with provenance)

## Phase 0: TypeScript on Node 25, 2026-09-05 (from the project's working notes, measured on this machine)

Decode-shaped matvec 4864×896 and prefill-shaped GEMM 64×896×4864, fp32:
```
naive typed-array loop           3.8 ms   (2.3 GFLOP/s)
@stdlib/blas-base-sgemv v0.1.1   5.8 ms
bf16-resident loop (<<16 fused)  4.4 ms   (+15%, halves RAM)
prefill: naive 301 ms, best @stdlib sgemm layout 580 ms
sdot 0.77 GFLOP/s vs naive dot 2.28 GFLOP/s
PyTorch on MPS, same shapes: matvec 0.15 ms, GEMM 0.66 ms
```
Weights (Qwen2.5-0.5B, 494,032,768 parameters): max |w| 214; 0 values above fp16 max; 1,472,401 (0.298%)
fp16-subnormal. JS has no bf16 type: loading upcast the 1 GB checkpoint to ~3 GB resident.

## Week 3–4 checks against HF fp32 (from the working notes of those sessions)

```
W3 embedding: max |diff| 0 (exact); RMSNorm: max |diff| <= 2.4e-7
W4 layer-0 q/k/v projections: q 1.3e-4, k 1.8e-4 (k bias entries reach ~130, so ~1e-6 relative), v 7e-8
W4 RoPE-rotated q/k: 1.3e-4 / 1.8e-4 (unchanged by the rotation)
W4 layer-0 attention output: <= 4e-6 on values <= 0.4 (5/5 prompts)
```

## The Qwen3.5-2B suite before the prefill OOM fix (2026-09-28, first run of tests/test_qwen35_2b.py)

The INT8/INT4 full-sequence forward dequantized the 248,320 × 2048 tied head in one piece (2.0 GB in fp32 for
the result alone, plus the intermediate tensors of `QuantTensor.dequantize`); a Metal command-buffer error was
printed and the logits were garbage, while greedy decoding (which only computes the last position's head row
through the matvec kernel) was fine:
```
bf16 : weights 3.76 GB | greedy = HF for 64/70 tokens | top-1 flips 0/190   | KL vs HF bf16 0.0004 | decode 19.4 tok/s
int8 : weights 2.00 GB | greedy = HF for 70/70 tokens | top-1 flips 180/190 | KL vs HF bf16 5.9841 | decode 33.5 tok/s
int4 : weights 1.41 GB | greedy = HF for 47/70 tokens | top-1 flips 180/190 | KL vs HF bf16 5.9993 | decode 12.2 tok/s
```
After the row-chunked fix: INT8 KL 0.0007 (0 flips), INT4 KL 0.0454 (3 flips); see `accuracy_2b_vs_hf_bf16.txt`.

## Load-test runs discarded because the machine was busy (2026-09-28)

Taken on the batching speed-up code while 6.58 GB of comparison models were downloading and being hashed
(Q4_1 1.38 + Q4_K_M 1.40 + Q8_0 2.08 GB GGUF + 1.72 GB MLX):
```
run A: 23.5 / 22.2 / 34.0 / 41.4 tok/s at 1/2/4/8 clients (TPOT p50 38.5 / 78.3 / 106.8 / 172.5 ms)
run B: 22.3 / 21.2 / 30.1 / 37.1 tok/s                     (TPOT p50 40.0 / 87.1 / 120.6 / 190.5 ms)
```
The final load test (40.3 / 41.5 / 52.9 / 55.9 tok/s, `loadgen_2b_int4_final.md`) ran later on a quiet machine on
commit f9280fa, which adds only the prefill-expansion kernels; those do not touch the 19–24-token load-test prompts.
A VS Code C/C++ indexer at ~100% CPU caused a similar slowdown in engine micro-benchmarks (B=8 step 69 → 177 ms).

## Review workflows (multi-agent review with two-vote adversarial verification)

| review | agents | findings | confirmed by both skeptics |
|---|---:|---:|---:|
| W7–W8 (commit 63d80d2) | 56 | — | 18 |
| W9–W10 (commit 595b5ce) | 48 | — | 5 (measurement) |
| W15 server (run wf_47c6d5c8) | 106 | 50 | 9 |
| batching speed-up (run wf_de09275c) | 20 | 8 | 1 |
| documentation numbers, round 1 (run wf_4cdb7209) | 140 | 67 | 38 (all fixed in the W17 docs commit) |
| documentation numbers, round 2 (run wf_35f69d1a) | ~90 | 45 | fixed in the same commit |

## Quantizing Qwen3.5-2B (`/usr/bin/time -l scripts/quantize.py ...`, 2026-09-28)

```
wrote models/qwen3.5-2b/model.int4.qt: 1.41 GB of tensor data   161.47 s real   maximum resident set size 3,455,434,752 bytes
wrote models/qwen3.5-2b/model.int8.qt: 2.00 GB of tensor data    55.46 s real   maximum resident set size 3,504,046,080 bytes
```

## Profiling-run chronology

The batch-8 `decode_batch` step before the speed-up was timed twice: first 185.9 ms (single-linear experiment run),
then 188.6 ms (the linears/rest split). Docs quote "186–189 ms".

## An earlier run of tests/test_server.py (same code as the archived one)

```
PASS  real uvicorn: a non-streaming request whose client gave up is cancelled (29 of 400 tokens made), blocks freed
```
The archived run (`tests/test_server.txt`) printed 43 of 400.

## Container smoke tests (2026-09-28, transcribed from the session)

```
docker info: Server Version 29.5.3 (Docker Desktop), VM memory 4,108,632,064 bytes (3.83 GiB), 8 CPUs, aarch64
docker build -t inference-engine .   1:13.16 total      image inference-engine:latest 1.37GB
run --model qwen2.5-0.5b --backend cpu: ready after ~12 s; health: healthy; docker stats: mem 2.401GiB / 3.826GiB
  chat "What is the capital of France? One word." max_tokens 8, temperature 0 -> 'Paris.' (2 completion tokens)
  completion "The history of computing is" max_tokens 64 -> 64 tokens, latency_s 15.7144 (e2e 15.80 s) => ~4 tok/s
  streaming 120 tokens, docker stop 2 s into it: `docker stop` returned after 26.770 s; the stream received all
  121 data chunks and [DONE]; log order: request_finished (req-1), "Shutting down", request_finished (req-2,
  120 tokens, latency 26.25 s), "Waiting for application shutdown", "Application shutdown complete", exit
run (default then: --model qwen3.5-0.8b --backend cpu): ready after ~48 s; docker stats during one request:
  3.019 / 3.537 / 3.489 / 2.727 GiB of 3.826 GiB at t+20/40/60/80 s; completion "The capital of France is"
  max_tokens 6 -> " Paris.\nThe capital of" (same as the HF golden), 114 s end to end (ttft_s 15.90); OOMKilled=false
after changing the default CMD to qwen2.5-0.5b and rebuilding: plain `docker run` -> /ready 200, clean shutdown
```

