Model `qwen3.5-2b`, backend `metal-kernels-int4`, 16 requests per level, max_tokens 64, temperature 0.7 (top-k 20, top-p 0.8), streaming chat completions.

| concurrency | throughput (tok/s) | TTFT p50 | TTFT p95 | TPOT p50 | e2e p50 | e2e p95 | rejected |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 30.0 | 291 ms | 331 ms | 27.1 ms | 2.00 s | 2.90 s | 0 |
| 2 | 30.8 | 550 ms | 625 ms | 57.5 ms | 3.95 s | 4.90 s | 0 |
| 4 | 35.5 | 845 ms | 1285 ms | 101.6 ms | 7.34 s | 7.75 s | 0 |
| 8 | 31.4 | 1786 ms | 2978 ms | 233.0 ms | 16.88 s | 16.89 s | 0 |
