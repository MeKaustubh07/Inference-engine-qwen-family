Model `qwen3.5-2b`, backend `metal-kernels-int4`, 16 requests per level, max_tokens 64, temperature 0.7 (top-k 20, top-p 0.8), streaming chat completions.

| concurrency | throughput (tok/s) | TTFT p50 | TTFT p95 | TPOT p50 | e2e p50 | e2e p95 | rejected |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 39.3 | 222 ms | 234 ms | 22.5 ms | 1.64 s | 1.66 s | 0 |
| 2 | 41.3 | 420 ms | 479 ms | 44.6 ms | 3.10 s | 3.18 s | 0 |
| 4 | 61.2 | 642 ms | 883 ms | 58.5 ms | 4.18 s | 4.22 s | 0 |
| 8 | 77.0 | 1097 ms | 1731 ms | 91.0 ms | 6.67 s | 6.68 s | 0 |
