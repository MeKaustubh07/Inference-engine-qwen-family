# Results log: every measurement, in the order it happened

The raw material for the project write-up: what was measured at each step, what it showed, and what was decided.
Hardware throughout: MacBook Air M2 (Mac14,2: 4P+4E CPU, 8-core GPU), 8 GB unified memory, ~100 GB/s. Commits
are in `git log`; detailed tables live in `docs/bench/` (raw tool output in `docs/bench/raw/`, test output in
`docs/bench/raw/tests/`, numbers first recorded in working notes in `docs/bench/raw/early_measurements.md`).

## Phase 0 — TypeScript on Node (Aug 30 – Sep 5, commits `bd2b183`, `a3d84d0`)

Hypothesis: build the engine on `@stdlib/blas`, the BLAS routines the author contributes to. (Numbers from the
working notes of 2026-09-05; `raw/early_measurements.md`.)

| measurement (decode matvec 4864×896, prefill GEMM 64×896×4864) | result |
|---|---|
| naive typed-array loop, fp32 | 3.8 ms (2.3 GFLOP/s) |
| `@stdlib/blas-base-sgemv` v0.1.1 | 5.8 ms (slower than the naive loop) |
| prefill: naive vs best `sgemm` layout | 301 ms vs 580 ms |
| `sdot` vs naive dot | 0.77 vs 2.28 GFLOP/s |
| same ops in PyTorch on MPS | 0.15 ms and 0.66 ms (25× and 450× faster than the naive JS loop) |
| weights | JS has no bf16: a 1 GB checkpoint becomes ~3 GB resident; 0.3% of weights are fp16-subnormal |

Loader and BPE tokenizer were correct (51/51 vs HF on the first run). **Decision (Sep 5):** move the numeric
core to Python + PyTorch (bf16 native, a route to hand-written Metal kernels, same language as the answer key).

## Correctness on Qwen2.5-0.5B (W1–W6)

| week | commit | result |
|---|---|---|
| W1 loader | `2f11d14` | mmap + zero-copy bf16 views: open 7 ms, 4/4 integrity checks, peak RSS 215 MB (vs ~3 GB in TS) |
| W2 tokenizer | `e12611d` | byte-level BPE, 51/51 identical to HF `tokenizers` |
| W3 embedding, RMSNorm | `40f4c66` | embedding exact (diff 0), RMSNorm ≤ 2.4e-7 vs HF fp32 (`raw/early_measurements.md`, `raw/tests/test_ops.txt`) |
| W4 block | `8e74eb4` | RoPE q/k 1.3e-4/1.8e-4 abs (keys reach 130: ~1e-6 relative); layer-0 attention ≤ 4e-6; full block ≤ 3.3e-5 (`raw/early_measurements.md`, `raw/tests/test_block.txt`) |
| W5 model | `390685e` | 24 layers, logits ≤ 3e-4 vs HF; greedy continuation "Paris" (`raw/tests/test_model.txt`) |
| W6 sampling, chat, REPL | `c907515` | ChatML template identical to HF; top-k/top-p/temperature/repetition as HF's warpers |
| W6 review | `88555fe` | review workflow: 9 confirmed defects (top-p boundary, padding ids sampled, stream tail) fixed; greedy == HF `generate()` token for token on 5 prompts |

## Speed on Qwen2.5-0.5B (W7–W12)

| step | commit | decode tok/s | note |
|---|---|---:|---|
| CPU fp32 reference | `028caba` | 10.2 | backend protocol, KV cache (W7 commit; 23.8 at a 16-token prompt in the W8 harness, `decode.md`) |
| MPS bf16 (PyTorch ops) | `028caba` | 20.8 | |
| KV cache effect (CPU, 1024 tokens) | `f79f354` | 53 ms vs 1,512 ms per token | ~28× |
| hand-written Metal kernels | `0a36780` | ~41 | matvec, RMSNorm, RoPE, decode attention |
| + fusion alone | | ~37–39 | dispatch count was not the bottleneck |
| + remove `int(positions[0])` sync per layer | | ~68 | 24 forced CPU↔GPU waits per token gone |
| + embedding lookup on GPU | `0a36780` | **~73** | CPU issue 23.1 ms → 1.6 ms per token (72.7 at a 16-token prompt in `decode.md`; the W9 commit's end-to-end run said 66) |
| native Obj-C++ runtime (same MSL) | `7dd7f61` | 63 / 75 / 89 GB/s matvec with cold weights (`decode.md`; the commit's 79–88 GB/s was cache-served) | tiled GEMM correct but 4× slower than MPS → prefill keeps MPS |
| review of W7–8 (56 agents, 18 confirmed) | `63d80d2` | | atomic block allocation, sound benchmark method |
| review of W9–10 (48 agents) | `595b5ce` | | matvec numbers were cache-inflated: 49–83 GB/s with cold weights (commit; `decode.md` quotes 58–83 for the shapes it lists) |
| INT8 (block 32, symmetric) | `fb6cff0` | ~89 (85.0 at 16 tokens; mean of 16/256/1024-token prompts) | 0.52 GB, perplexity 11.30 vs bf16 11.34, greedy identical |
| INT4 symmetric (first attempt) | | | perplexity +28% (14.54), top-1 62% |
| INT4 asymmetric + INT8 embedding | | | perplexity 11.50, but KL 1.85: position-0 "attention sink" damage |
| **INT4 asymmetric + calibrated policy** | `fb6cff0` | **~97** (99.3 at 16 tokens; mean) | 19 tensors kept INT8, 0.41 GB, KL 0.114, top-1 94% |

Details: `docs/bench/decode.md`, `docs/bench/quant.md`.

## The port to Qwen3.5 (W13–W14)

| step | commit | result |
|---|---|---|
| Qwen3.5-0.8B text path | `c8ac5c8` | 24 layers ≤ 4e-6 relative vs HF fp32, logits 6e-5, greedy identical incl. Hindi |
| fused DeltaNet decode kernels | `c8ac5c8` | 17 → 46 tok/s (0.8B, bf16) |
| tokenizer finding | `c8ac5c8` | HF `AutoTokenizer` resolves this checkpoint to `Qwen2Tokenizer` (older regex, splits combining marks); the engine follows `tokenizer.json` |
| review fixes | `4b31841` | exact chat-template rules, 7 missing special tokens, atomic state reservation; goldens with 57- and 126-token prompts (only the 126-token one crosses the 64-token chunk; the commit message says both) |
| 0.8B suite (7 prompts) | | worst layer 3.8e-6 relative, logits ≤ 6.4e-5, top-1 100%, greedy 70/70 = HF; paged hybrid state == uncached (4.3e-5) (`raw/tests/test_qwen35.txt`) |
| Qwen3.5-2B bf16 vs HF bf16 | `32de01f` | KL 0.0004, 0/190 top-1 flips, greedy 64/70 (one divergence at a 0.015-logit near-tie) |
| 2B INT8 | `32de01f` | 2.00 GB, KL 0.0007, 0 flips, greedy 70/70 |
| 2B INT4 (5 tensors kept INT8) | `32de01f` | 1.41 GB, KL 0.045, 3/190 flips, answers "Paris" |
| bug found by the 2B suite | `32de01f` | quantized prefill dequantized the 248k-row tied head at once (2 GB for the fp32 result alone plus dequantization intermediates → Metal command-buffer error, KL 5.98 / 6.00, 180/190 flips); fixed with row chunks (`raw/early_measurements.md`) |

## Serving (W15–W16)

| step | commit | result |
|---|---|---|
| server + continuous batching | `2278c0e` | batched == sequential greedy for both families (fp32 CPU: tokens identical, logits within 6e-5); preemption == uninterrupted; SSE == non-stream |
| review of W15 (106 agents, 50 findings) | `2278c0e` / `ae478b6` | 9 confirmed (all fixed) + more fixed from the rejected pile: non-stream disconnect never cancelled, tiny temperature crashed a whole batch, queued cancels held queue slots, missing `regex` in requirements (landed with the requirements files in `378d540`) (`raw/early_measurements.md`) |
| bugs found by tests | `2278c0e` | lone-surrogate JSON crashed FastAPI's 422 handler (500); drain lost a request mid-prefill; disconnect only checked when idle |
| **first load test** (2B INT4, 16 req/level, 64 tokens) | | 30.0 / 30.8 / 35.5 / 31.4 tok/s at 1/2/4/8 clients: **batching gave almost nothing** |
| profile of one batch-8 step | | 111 ms linears + 78 ms per-sequence attention/DeltaNet (5.7–9.8 ms per sequence at B = 1–8) (`raw/profiling.md`) |
| kernel experiment: x tile in threadgroup memory | | 2–3× slower at M = 1, 1.1–1.4× faster at M = 8 (M = 2–4 not measured); beaten by 2 rows per SIMD group → not adopted |
| kernel experiment: 4 rows per SIMD group | | 1.4–1.5× at M = 8, slower at M ≤ 2 |
| kernel experiment: 2 rows per SIMD group | | INT8 head at M = 8: 35 → 14 ms; adopted for all M ≥ 2 |
| pooled DeltaNet state (2 dispatches per layer for the batch) + 2-rows kernels | `ae478b6` | batch-8 step 189 → 97 ms (188.6 → 96.5) |
| paged-attention kernel (one dispatch per layer) | `ae478b6` | batch-8 step 97 → 69 ms; per-sequence overhead 0.8 ms |
| decode step, 2B INT4 | `ae478b6` | B=1 26.5 → 20 ms; B=8 186 → 69 ms (43 → 116 tok/s aggregate) |
| batching review (20 agents) | `cc40106` | 1 confirmed (no scheduler test on the pooled state) → added, with defensive fixes: a freed pooled state could alias a reused slot, admission waits for a free state slot, K % 4 guard, max_batch validated |
| prefill profile, 256 tokens INT4 | | 1.67 s of 1.92 s in quantized linears: 0.55 s torch-op dequant + 1.1 s fp32 GEMM |
| one-pass dequant → bf16 + bf16 GEMM | `34b1b59` | 128 → 305 tok/s (256 tokens) |
| MPS GEMM on M2 (M = 256–1024) | | **fp32 2.5 TFLOPS, fp16 2.6, bf16 1.4** (no native bf16 math before M3) |
| one-pass dequant → fp32 + fp32 GEMM | `f9280fa` | **395 tok/s (256), 446 tok/s (1024)**, exact vs the reference path |
| final engine benchmark, INT4 (2 runs) | `f9280fa` | decode B=1 50.1 / 49.7 tok/s; batch 8 115.8 / 112.7; prefill 256/1024: 398/450 and 395/444 tok/s |
| final engine benchmark, INT8 | `f9280fa` | decode B=1 39.2; batch 8 **126.4** (INT8 ahead of INT4 from batch 2 (54.3 vs 53.6, within noise) and clearly at 4–8; likely a compute-bound shared-weight step where nibble unpacking costs more than bytes, not profiled) |
| final engine benchmark, bf16 | | decode 20.6; batch 8 70.5 (3.4×); 1024-token prefill 9.8 s, likely paging (5,177 MiB of GPU memory) |
| **final load test** (same setup as the first) | `f9280fa` (the prefill kernels it adds do not touch these 19–24-token prompts) | 40.3 / 41.5 / 52.9 / 55.9 tok/s at 1/2/4/8 clients; TTFT p50 221 ms → 1.4 s; 0 rejected |
| why the server trails the engine (56 vs 116 at 8) | | lockstep waves (every request 64 tokens) start with 8 back-to-back prefills (~0.29 s each); then 108–110 ms per token vs a 69 ms step + ~12 ms CPU sampling, i.e. ~28 ms per step of unprofiled engine-loop work (`serving.md`) |

## Container (Docker Engine 29.5.3 in Docker Desktop, 3.83 GiB VM, 8 vCPU, arm64; transcript in `raw/early_measurements.md`)

| check | result |
|---|---|
| `docker build` | 73 s, image 1.37 GB (python:3.14-slim + CPU torch wheel) |
| Qwen2.5-0.5B, CPU fp32 | ready in ~12 s, 2.4 GB, answers "Paris.", healthcheck healthy, ~4 tok/s decode (64 tokens in 15.7 s) |
| `docker stop` mid-stream (120 tokens) | stream completed with [DONE], lifespan shutdown, exit at 26.8 s (< 30 s stop timeout) |
| Qwen3.5-0.8B, CPU fp32 | output " Paris.\nThe capital of" (same as the HF golden), but peaks at 3.54 of 3.83 GiB and pages: 6 tokens in 114 s → give the VM more memory (≥ 6 GB suggested, untested) |

## Comparison with llama.cpp and MLX-LM (Qwen3.5-2B, same machine, one engine at a time)

| metric | ours INT4 | ours INT8 | llama.cpp Q4_1 | llama.cpp Q4_K_M | llama.cpp Q8_0 | MLX-LM 4-bit |
|---|---:|---:|---:|---:|---:|---:|
| decode, 1 stream (ours: `decode_batch` step; llama.cpp: llama-bench / batched-bench) | 50.1 | 39.2 | 42.3 / 53.3 | 32.5 | 28.6 / 38.6 | 69.9 |
| prefill 256 / 1024 | 398 / 450 | 389 / 443 | 601 / 452 | 445 / 375 | 513 / 438 | 499 / 448 |
| batch 8 aggregate | 115.8 | 126.4 | 61.9 | — | 66.1 | 73.3 |

llama.cpp build 11146 (Homebrew), bartowski GGUFs; MLX-LM 0.31.3, mlx-community 4-bit. Analysis: `docs/bench/compare.md`.

Load-test and benchmark tables: `docs/bench/serving.md`, `docs/bench/qwen35.md`, `docs/bench/compare.md`.

## Review workflows run

| week | agents | findings → confirmed | notable |
|---|---:|---|---|
| W6 | — | 9 confirmed | top-p boundary, padding ids, stream tail |
| W7–8 | 56 | 18 confirmed | non-atomic OutOfBlocks, benchmark method |
| W9–10 | 48 | 5 (measurement) | cache-hot bandwidth overstated |
| W13 | — | — | chat reasoning rules, missing special tokens, atomic state reservation, long-prompt goldens (commit `4b31841`) |
| W15 | 106 | 50 → 9 confirmed | disconnect, regex, serve defaults, /ready before load |
| batching speed-up | 20 | 1 confirmed | pooled-state scheduler test |
| documentation numbers, round 1 | 140 | 67 → 38 confirmed, all fixed | GPU is 8-core, DeltaNet batch update is 2 dispatches, 2B answer key is bf16 |
| documentation numbers, round 2 | ~90 | 45, fixed | server gap is lockstep prefill + CPU sampling + ~28 ms unprofiled work; variance up to ~20% |
