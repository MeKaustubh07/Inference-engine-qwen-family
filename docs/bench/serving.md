# Serving Qwen3.5-2B INT4 under load

`scripts/serve.py` defaults (Metal INT4, `max_batch` 8, 1024 KV blocks of 16 tokens, queue 64) driven by
`scripts/loadgen.py`: closed-loop clients at 1/2/4/8 concurrency, 16 streaming chat requests per level, 64 new
tokens each, API-default sampling (temperature 0.7, top-k 20, top-p 0.8, fixed seeds). MacBook Air M2 (8-core
GPU), 8 GB.
Raw: `docs/bench/raw/loadgen_2b_int4_*.md`, `metrics_after_final_loadtest.txt`, `server_request_log_final.jsonl`.

**Latest (2026-09-30): 94.3 tok/s at 8 clients, TTFT p50 0.84 s**, after profiling the server and adding packed +
chunked prefill, a batching window and batched sampling; measured side by side with the previous server at
77.0 tok/s ([Closing the gap](#closing-the-gap-profile-packed--chunked-prefill-batched-sampling)).

## Result: first server vs the batching speed-up (commit f9280fa)

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

## Why the server scaled less than the raw engine (first analysis; corrected by the profile below)

The engine alone produces 116 tok/s at batch 8; the server delivered 56 tok/s with 8 clients. From the request
log (`raw/server_request_log_final.jsonl`), three things seemed to add up:
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

## Closing the gap: profile, packed + chunked prefill, batched sampling

**The profile accounted for every millisecond of a step** (`raw/profiling.md` §7: the scheduler's methods wrapped
with timers, `decode_batch` followed by a GPU synchronize to split issue time from GPU time). Under the HTTP load at
8 clients a batch-8 decode step took 84.1 ms: 69.2 ms GPU, 4.1 ms issuing it from Python, 8.8 ms sampling, 0.2 ms
emitting tokens, 1.7 ms the rest. **The "~28 ms" did not reproduce.** In the 2026-09-28 run each 8-client wave
arrived and finished together, so the last-admitted request decoded its 63 tokens with no prefill in between, at
108–110 ms per token: that was the run's steady state. Re-measuring the same code, the same request decoded at
77–90 ms per token in 9 runs (`raw/profiling.md` §10). That run was slower at 4 and 8 clients (at 1–2 clients it
matches the re-measured runs), the same unexplained slowness behind its 55.9 tok/s (last paragraph of this
section). In the final A/B's before-runs (77.0 tok/s, i.e. 104 ms per 8 tokens against the engine's 69) the gap
is **about three quarters prefill**: each wave's 8 prompts were prefilled one at a time in ~1.7 s (~0.21 s each,
~27 ms per 8 tokens) while the admitted requests waited. The other quarter is per-step work around `decode_batch`:
the last-admitted request decoded at ~78 ms per token, ~9 ms above the engine step, most of it sampling (8.8 ms
per step in the profiled run, not the ~12 estimated). The instrumented profiling run, with its slower 84 ms
steps, puts about two thirds of its gap in prefill.

**Prefill cost grows slowly with prompt length once the prompt is past 32 tokens** (`raw/profiling.md` §8): one
forward of 23 tokens takes 222 ms, of 184 tokens 542 ms. Up to 32 rows the quantized linears run as batched
matvec kernels that re-read every weight once per 8 rows (49.4 / 146.5 / 203.3 ms at 8 / 23 / 32 rows); above 32
the weights are expanded to fp32 once per pass and multiplied by GEMM, which starts high and grows slowly (183.6 ms
at 33 rows, 193.5 at 64, 336.4 at 184). Eight ~23-token chat prompts cost ~1.8 s one at a time; packed into one pass (item 1 below) they
take 0.82 s (1.33 s the first time that size is seen; `raw/profiling.md` §9), more than a single 184-token prompt
(0.54 s) because attention and the DeltaNet recurrence still run per sequence (~40 ms per extra sequence).

What changed (`src/server/scheduler.py`, `src/models/packing.py`, `forward_packed` in `src/models/qwen3_5.py` and
`src/models/qwen2.py` (since removed), `src/sampler.py`, `src/backend/metal.py`, flags in `scripts/serve.py`):
1. **Packed prefill.** `forward_packed` runs the prompt chunks of several sequences as one forward: embedding,
   norms, projections and MLP once over all their tokens; attention and the DeltaNet recurrence per sequence on
   its slice, each against its own KV blocks and recurrent state. Result for each prompt vs prefilling it alone
   (`tests/test_prefill.py`): max |Δlogit| 2.9e-6 (INT4 Metal), 0 (bf16 Metal), ≤ 2.7e-5 (fp32 CPU) for 4 short
   prompts; ≤ 2.9e-5 for a 43-token pack, which takes the GEMM path while each prompt alone takes the batched
   kernels (both keep fp32 activations).
2. **Chunked prefill.** Requests go waiting → prefilling → running. Each loop iteration runs one packed pass over
   at most `--prefill-chunk` prompt tokens (512), first come first served, then one decode step for every running
   request, so a long prompt no longer freezes the batch. Admission reserves the prompt's KV blocks up front.
3. **Batching window.** In the first load test of items 1–2, each 8-client wave's first request was prefilled
   alone (~20 tokens) and the other 7, arriving a few ms later, together (~155 tokens): the idle engine started on
   the first arrival (`raw/loadgen_2b_int4_2026-09-29_ab.md`, last section). An idle engine now waits while arrivals
   keep coming within `--batch-wait-ms` (default 5 ms) of each other, twice that at most; a lone request waits
   one window (6.2 ms measured, with the wake-up).
4. **Batched sampling, which ended up on the CPU.** `sample_batch` applies the repetition penalty and selects
   each row's top-(k + 8) candidates once for the whole batch; each request's temperature / top-k / top-p / seeded
   draw then runs on its candidates only, exactly as the single-request sampler (300 random batches with ties and
   penalties: identical tokens; `tests/test_sampling.py`). top_k 0 or > 256 and ties straddling the cut fall back
   to the full-row sampler for that request alone. The plan was to run the selection on the GPU, where the logits
   are. Measured, that saved nothing (8.8 → 8.4 ms per batch-8 step under HTTP load): `torch.topk` over 8 × 248k
   logits takes 4.5 ms on MPS whatever k is, against 1.0 ms on the CPU, and with unified memory copying the
   batch's logits to the CPU takes 0.5 ms. With the selection on the CPU, sampling takes 3.2 ms per step (was 6.9
   in-process), and the batch-8 step 73.3 ms instead of 79.4 (`raw/profiling.md` §9).

**Before/after** (raw: `raw/loadgen_2b_int4_2026-09-30_ab_final.md`). Same load test, the old code (bbeb6c6) and
the final code measured side by side: 8 runs alternating in two opposite-order blocks (before-after-after-before,
then after-before-before-after), each on a fresh server after a 90 s idle cool-down. Means over 4 runs each:

| clients | throughput (tok/s) | TTFT p50 | TTFT p95 | TPOT p50 | e2e p50 |
|---:|---:|---:|---:|---:|---:|
| 1 | 40.1 → 40.1 | 221 → 229 ms | 229 → 242 ms | 21.9 → 21.6 ms | 1.60 → 1.59 s |
| 2 | 42.0 → **45.3** | 417 → **334** ms | 447 → **402** ms | 44.0 → **39.6** ms | 3.06 → **2.83** s |
| 4 | 62.1 → **69.5** | 634 → **518** ms | 880 → **523** ms | 57.6 → **50.6** ms | 4.16 → **3.71** s |
| 8 | 77.0 → **94.3** (+22%) | 1,056 → **838** ms | 1,710 → **892** ms | 91.5 → **72.6** ms | 6.67 → **5.44** s |

Ranges at 8 clients: before 76.8–77.6 tok/s, after 92.5–95.0; at 2, 4 and 8 clients the slowest after-run beat
the fastest before-run. At 1 client TTFT is 8 ms worse: in-process, the batching window costs a lone request
6.2 ms (its 5 ms wait plus wake-up; `raw/profiling.md` §9) while the packed forward was 3.6 ms faster than the old
one; the rest was not isolated. `--batch-wait-ms 0` turns the window off.

**What is left of the gap to the engine's 116 tok/s.** Every request makes exactly 64 tokens, so 8 clients move in
lockstep waves: one packed prefill pass for the wave's 8 prompts (0.82 s, which also yields their first tokens),
then 63 decode steps of ~73 ms (the 70 ms `decode_batch` step, 3 ms sampling): ~5.4 s per 512 tokens, 94 tok/s,
as measured (e2e p50 5.44 s). The prefill pass is ~15% of the time and sampling ~4%. Next: run the packed
sequences' attention and DeltaNet prefill as one variable-length dispatch instead of one per sequence (~0.28 s of
the pass; it would approach a single 184-token prompt's 0.54 s), and issue the decode step without Python in the
loop.

**An intermediate version** (the batched selection on MPS; raw: `raw/loadgen_2b_int4_2026-09-29_ab.md`) measured
72.4 → 88.1 tok/s at 8 clients (TPOT 97.7 → 78.3 ms) in an earlier 10-run A/B. That evening 3 runs, 2 before and
1 after, were excluded by a rule applied to both (8-client TPOT p50 above 120 ms; the others 76–106 ms): in each,
the 1-client level was normal and the slowdown began partway through the run (at 4 clients in one, during the
2-client level in the other two). In the one with CPU samples, macOS background services (ModelCatalogRuntime,
AssistantServices, asset downloads, Shortcuts) took the top CPU slots at that moment and the server's step time
tripled, while the python process shown in the samples fell from 28–42% to 4–14% CPU. That suggests the server was
waiting (on a shared GPU, or on memory: swap was 5.5 of 6 GiB in use right after the first block) rather than
computing; the samples cannot prove it. The final A/B applied the same rule and excluded none.

**The same code measured 77 tok/s at 8 clients here, not the 55.9 published above** (67–78 over the 7
before-runs kept in both A/Bs). At 1–2 clients the 2026-09-28 run and these agree within ~1% in throughput and
TTFT p50 (TPOT p50 at 2 clients: 45.4 vs 44.0 ms); at 4–8 clients it was slower (TPOT p50 126 vs 91–106 ms).
Its cause was not established, but it has the shape of the excluded runs (normal early in the run, slow later);
the comparison above uses only runs taken side by side.

## Running it by hand: what testing the deployed server found (2026-09-30)

The server was started natively and exercised the way an operator would (raw: `raw/deployment_tour_2026-09-30.md`):
`/health`, `/ready`, `/metrics`, chat and text completions, streaming, seeded and greedy determinism, input
validation (422 / 400), a load test at 1 and 8 clients (94.8 tok/s at 8, as in the A/B above), backpressure (80
simultaneous requests: 72 accepted, i.e. 8 in flight plus a full 64-slot queue, and 8 rejected with 429 and
`Retry-After: 1`), client disconnects (all 80 clients dropped: everything cleaned up within ~1 s; one stream closed
after 20 tokens: stopped 0.12 s later) and SIGTERM mid-stream (the 150-token stream finished with `[DONE]`, new
connections were refused, the process exited 2.9 s after the signal). Three problems showed up that no benchmark or
test had caught:

1. **An idle server loses its weights to the OS.** The same 19-token request got its first token in 0.24 s back to
   back, but 0.69–0.81 s after 1–4 minutes of idle. On unified memory the 2.1 GB of Metal buffers holding the
   weights are ordinary RAM, and macOS compressed or swapped them while the server idled: 1.4–2.2 GB came back in
   during the slow requests, 28 MB during the fast one. The load tests never saw it because their requests come
   back to back. Fix: after its warm-up the scheduler `mlock`s the pages of every weight buffer
   (`src/backend/pinning.py`; `--no-lock-weights` turns it off; `engine_weights_locked_bytes` in `/metrics`). MPS
   tensors live in shared-storage `MTLBuffer`s, so each has a CPU address, and wired pages can be neither compressed
   nor swapped; the test reads the kernel's wire count on that memory (0 → 1 → 0 after unlock). A Metal residency set
   with `requestResidency`, MLX's approach, was tried first and did not stop the reclaim here.

   | after idle | weights not locked | `mlock` | residency set |
   |---|---:|---:|---:|
   | in-process prefill, 60 s | 438 / 396 ms | 229 / 235 ms | 494 ms |
   | in-process prefill, 3 min | 530 / 433 ms | 264 / 278 ms | 617 ms |
   | server first token, 60 s | 0.330 / 0.450 s | 0.343 / 0.310 s | — |
   | server first token, 4 min | 0.512 / 0.668 s | 0.323 / 0.447 s | — |

   Warm: prefill 186–190 ms in-process, first token 0.25–0.26 s through the server. The in-process runs saw more
   memory pressure (21–32% free) than the server runs (31–40%). The lock covers the 1.32 GiB of weights, not the KV
   cache and DeltaNet state pools (~0.5 GB) or the rest of the process: in the slowest locked run 808 MB still came
   back. Locking the pools too is the next step; the operational answer is to give a model server memory of its own
   (a dedicated host, or memory requests and limits under Kubernetes).
2. **The queue gauges missed short waits.** `waiting` and `prefilling` were refreshed once per engine iteration,
   after admission, so a request that waited up to ~0.8 s behind a prefill pass never showed (peak `waiting` 0
   during the load test). `/metrics` now reads them from the scheduler at scrape time; the test holds the engine
   mid-step and requires `waiting 1, running 1`.
3. **Cancelled requests were logged with `finish_reason: null`.** The handler writes the `request_finished` line
   before the engine thread processes the cancel (17 of 73 lines in the burst). The line now carries the reason the
   client was told, then the engine's, then `cancelled` for a client that left. The code review caught that the
   first version of this fix logged stop-string completions, which end through the same cancel, as `cancelled`;
   fixed, and tested both ways.

Also seen, not fixed: the first request of a given prompt length after start-up costs ~0.1–0.2 s extra (first token
0.34–0.45 s vs 0.25 s, with little memory brought back), most likely MPS compiling kernels for new tensor shapes;
warming the common prompt lengths at start-up would remove it.

## Operational behaviour verified (tests/test_server.py, 45 checks; output in `raw/tests/test_server.txt`)

Scope: the HTTP and scheduler checks run on Qwen3.5-0.8B on the CPU (fp32, exact to compare against); until
2026-09-30 they ran on Qwen2.5-0.5B (48 checks, `raw/tests/2026-09-30-with-qwen2.5/test_server.txt`). Qwen3.5 on
Metal is covered by batched == single-sequence decode (bf16, INT4) and by a scheduler run over its pooled state with
forced preemption (INT4). The served 2B model itself was exercised by the load test.

| behaviour | how it is tested |
|---|---|
| continuous batching changes nothing | 4 concurrent requests == sequential greedy; mean decode batch > 1.5 asserted |
| preemption changes nothing | KV pool too small for the load → preempted, recomputed, identical output; every block and state slot returned |
| backpressure | queue of 1, 6 simultaneous requests → 429 + Retry-After, counted in `/metrics` |
| streaming | SSE chunks concatenate to the non-stream text; UTF-8 held back until complete (the scheduler's detokenizer fed an emoji split across tokens; Hindi stream == non-stream); first chat delta carries the role |
| client disconnect | real uvicorn: a non-streaming client that gives up after 1.5 s is cancelled server-side (4 of 400 tokens generated in the archived Qwen3.5-0.8B run; the Qwen2.5-0.5B runs made 44 (`raw/tests/2026-09-30-with-qwen2.5/test_server.txt`), 45, 43 and 29 (`raw/early_measurements.md`); the server polls every second) |
| graceful shutdown | scheduler drain: in-flight request finishes, new ones refused, `ready()` false (the flag `/ready` turns into a 503); real uvicorn: the lifespan stops the engine thread; manual `docker stop` mid-stream: the stream completed before exit (`raw/early_measurements.md`) |
| bad input | 422 for invalid parameters (incl. lone surrogates, which crashed FastAPI's default handler), 400 for prompts that can never fit |
| queue gauges | the engine held mid-step, a second request submitted: `/metrics` shows `waiting 1, running 1` (read at scrape time) |
| request log reasons | a client that gives up mid-request is logged `cancelled`; a stop-string completion is logged `stop`, as the client was told |
| weights locked in RAM | `tests/test_kernels.py`: the kernel's wire count on the MPS buffers' memory goes 0 → 1 → 0 after unlock, GPU results unchanged; `tests/test_prefill.py`: a scheduler with the lock on (0.8B INT4) matches decoding alone |
