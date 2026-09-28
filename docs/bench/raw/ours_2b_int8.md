### metal-kernels-int8 — qwen3.5-2b, 32 new tokens, M2 8 GB

Weights resident: 2.00 GB → bandwidth ceiling ≈ 50 tok/s (every decode step reads every weight once at ~100 GB/s).
Memory: peak CPU RSS 816 MiB, Metal driver 3161 MiB. Each row: median of 3 runs after a warm-up at that prompt length.

| prompt tokens | KV cache | TTFT (ms) | TPOT (ms) | prefill tok/s | decode tok/s | % of ceiling |
|---:|:---:|---:|---:|---:|---:|---:|
| 16 | yes | 147 | 26.6 | 109 | 37.6 | 75% |
| 256 | yes | 658 | 32.2 | 389 | 31.1 | 62% |
| 1024 | yes | 2311 | 28.0 | 443 | 35.8 | 72% |

| batch | step (ms) | per-sequence tok/s | aggregate tok/s | aggregate vs batch 1 |
|---:|---:|---:|---:|---:|
| 1 | 25.5 | 39.2 | 39.2 | 1.00x |
| 2 | 36.9 | 27.1 | 54.3 | 1.39x |
| 4 | 42.7 | 23.4 | 93.7 | 2.39x |
| 8 | 63.3 | 15.8 | 126.4 | 3.23x |

