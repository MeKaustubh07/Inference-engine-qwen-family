Model `qwen3.5-2b`, backend `metal-kernels-int4`, 16 requests per level, max_tokens 64, temperature 0.7 (top-k 20, top-p 0.8), streaming chat completions.

| concurrency | throughput (tok/s) | TTFT p50 | TTFT p95 | TPOT p50 | e2e p50 | e2e p95 | rejected |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40.3 | 221 ms | 226 ms | 21.8 ms | 1.59 s | 1.60 s | 0 |
| 2 | 41.5 | 418 ms | 451 ms | 45.4 ms | 3.09 s | 3.12 s | 0 |
| 4 | 52.9 | 674 ms | 995 ms | 67.5 ms | 4.95 s | 5.27 s | 0 |
| 8 | 55.9 | 1397 ms | 2292 ms | 126.4 ms | 9.25 s | 9.25 s | 0 |
