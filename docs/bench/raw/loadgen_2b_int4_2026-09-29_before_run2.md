Model `qwen3.5-2b`, backend `metal-kernels-int4`, 16 requests per level, max_tokens 64, temperature 0.7 (top-k 20, top-p 0.8), streaming chat completions.

| concurrency | throughput (tok/s) | TTFT p50 | TTFT p95 | TPOT p50 | e2e p50 | e2e p95 | rejected |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 36.5 | 229 ms | 280 ms | 23.9 ms | 1.75 s | 1.88 s | 0 |
| 2 | 40.5 | 425 ms | 458 ms | 46.3 ms | 3.16 s | 3.20 s | 0 |
| 4 | 61.6 | 645 ms | 888 ms | 58.5 ms | 4.15 s | 4.24 s | 0 |
| 8 | 73.6 | 1085 ms | 1754 ms | 96.3 ms | 6.98 s | 6.99 s | 0 |
