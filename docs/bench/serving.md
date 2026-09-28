# Serving Qwen3.5-2B INT4 under load

`scripts/serve.py` defaults (Metal INT4, `max_batch` 8, 1024 KV blocks of 16 tokens, queue 64) driven by
`scripts/loadgen.py`: closed-loop clients at 1/2/4/8 concurrency, 16 streaming chat requests per level, 64 new
tokens each, API-default sampling (temperature 0.7, top-k 20, top-p 0.8, fixed seeds). MacBook Air M2 (8-core
GPU), 8 GB.
Raw: `docs/bench/raw/loadgen_2b_int4_*.md`, `metrics_after_final_loadtest.txt`, `server_request_log_final.jsonl`.

## Result: first server vs final

| clients | throughput: first → final | TTFT p50 / p95: first → final | TPOT p50: first → final | e2e p50: first → final |
|---:|---:|---:|---:|---:|
| 1 | 30.0 → **40.3** tok/s | 291 / 331 → 221 / 226 ms | 27.1 → 21.8 ms | 2.00 → 1.59 s |
| 2 | 30.8 → **41.5** | 550 / 625 → 418 / 451 ms | 57.5 → 45.4 ms | 3.95 → 3.09 s |
| 4 | 35.5 → **52.9** | 845 / 1,285 → 674 / 995 ms | 101.6 → 67.5 ms | 7.34 → 4.95 s |
| 8 | 31.4 → **55.9** | 1,786 / 2,978 → 1,397 / 2,292 ms | 233.0 → 126.4 ms | 16.88 → 9.25 s |

No request was rejected or preempted in either run (the pool holds 16k tokens; this load needs < 1k).
Final-run server counters (including loadgen's warm-up of 8 short requests): 72 requests, 4,160 generated tokens.
Two runs taken while 6.58 GB of comparison models were downloading and being hashed (21–41 tok/s) were
discarded (`raw/early_measurements.md`): the engine thread issues GPU work from Python, so CPU contention shows up
directly in step time.

## What the first load test showed, and what was changed

The first version batched correctly (outputs identical to decoding each request alone) but gained almost nothing
from it: 30 tok/s with 1 client, 31 with 8. Profiling one batch-8 step of `decode_batch` (`raw/profiling.md`):

| part of a batch-8 step | first version | final |
|---|---:|---:|
| linears (97 quantized matrices) | 111 ms | 62 ms |
| everything else in the step (attention and DeltaNet, per sequence in the first version; plus embedding, norms, RoPE, gating) | 78 ms | 7 ms |
| **total step** | **186–189 ms** (two profiling runs) | **69 ms** |

1. **Everything but the linears (78 → 7 ms; 7.1 ms of it was already there at batch 1).** Attention and the
   DeltaNet update ran as a Python loop over sequences
   (for attention: index tensors built from the block table, a gather of the sequence's K/V, the kernel; for
   DeltaNet: slicing, contiguous copies and 2 kernels, per sequence and layer). Fixes: a paged-attention kernel
   that reads each sequence's K/V in place through its block table (one dispatch per layer for the whole batch),
   and a `HybridPool` that keeps every sequence's DeltaNet state in one tensor (2 dispatches per layer for the
   whole batch: conv step and delta-rule update).
2. **Linears (111 → 62 ms).** The first batched matvec kernel gave every output row its own SIMD group. At 8
   activation rows it was only 1.3× cheaper than 8 separate single-row calls (gate_up: 1.39 vs 1.85 ms), far from
   the ~8× that one shared weight read should allow; with 8 rows each weight costs 8 activation loads. Staging
   activations in threadgroup memory was faster at 8 rows (1.1–1.4×) but 2–3× slower at 1 row; computing 2 output
   rows per SIMD group, which reuses each activation load, was faster still at 8 rows and was adopted for every
   batch of 2 or more: the INT8 output head at batch 8 went from 35 ms to 14 ms.
3. **Prefill of long prompts (does not affect the load-test numbers).** A 256-token INT4 prompt took 2.0 s (1.9 s
   in the profiling run): 0.55 s of dequantization through several full-size fp32 tensor ops and 1.1 s of GEMM. A
   one-pass dequantization kernel into fp32 (bf16 was tried first and turned out slower: the M2's bf16 GEMM runs
   at 1.4 TFLOPS vs 2.5 for fp32) brought it to 0.65 s. The load test's chat prompts are 19–24 tokens, which go
   through the batched matvec kernels (up to 32 rows), not this path.

## Why the server scales less than the raw engine

The engine alone produces 116 tok/s at batch 8; the server delivers 56 tok/s with 8 clients. From the request
log (`raw/server_request_log_final.jsonl`), three things add up:
- **Prefill bursts.** Every request produced exactly 64 tokens, so the 8 clients move in lockstep: each wave
  starts with 8 back-to-back prefills inside the engine loop (~0.29 s each: the last four requests of the final
  wave got their first token at 1.44, 1.73, 2.03 and 2.32 s), and no decode step runs meanwhile. Chunked prefill
  (prompt slices interleaved with decode steps) and prefill/decode disaggregation solve this in vLLM-class servers.
- **Sampling on the CPU.** One device→host copy of the batch's logits, then top-k/top-p over a 248k vocabulary for
  each request: ~1.5 ms each, ~12 ms per step at batch 8. Batched sampling on the GPU would remove most of it.
- **Other per-step server work (~28 ms, not profiled).** Once a wave's prefills are done, the last-admitted
  request still averaged 108–110 ms per token, against a 69 ms `decode_batch` step plus ~12 ms of sampling.
  Candidates: per-request token emission and detokenization on the engine thread, and the event loop's
  per-token work competing for the GIL. Profiling the engine loop is the next step before optimizing it.

The TPOT median at 8 clients (126 ms) mixes the prefill bursts with this steady state.

## Operational behaviour verified (tests/test_server.py, 45 checks; output in `raw/tests/test_server.txt`)

Scope: the HTTP and scheduler checks run on Qwen2.5-0.5B on the CPU (fp32, fast and exact to compare against);
Qwen3.5 on Metal is covered by batched == single-sequence decode (bf16, INT4) and by a scheduler run over its
pooled state with forced preemption (INT4). The served 2B model itself was exercised by the load test.

| behaviour | how it is tested |
|---|---|
| continuous batching changes nothing | 4 concurrent requests == sequential greedy; mean decode batch > 1.5 asserted |
| preemption changes nothing | KV pool too small for the load → preempted, recomputed, identical output; every block and state slot returned |
| backpressure | queue of 1, 6 simultaneous requests → 429 + Retry-After, counted in `/metrics` |
| streaming | SSE chunks concatenate to the non-stream text; UTF-8 held back until complete (Hindi); first chat delta carries the role |
| client disconnect | real uvicorn: a non-streaming client that gives up after 1.5 s is cancelled server-side (43 of 400 tokens generated in the archived run, 29 in an earlier one recorded in `raw/early_measurements.md`; the server polls every second) |
| graceful shutdown | scheduler drain: in-flight request finishes, new ones refused, `ready()` false (the flag `/ready` turns into a 503); real uvicorn: the lifespan stops the engine thread; manual `docker stop` mid-stream: the stream completed before exit (`raw/early_measurements.md`) |
| bad input | 422 for invalid parameters (incl. lone surrogates, which crashed FastAPI's default handler), 400 for prompts that can never fit |
