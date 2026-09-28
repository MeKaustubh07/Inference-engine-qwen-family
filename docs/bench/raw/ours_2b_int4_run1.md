### metal-kernels-int4 — qwen3.5-2b, 32 new tokens, M2 8 GB

Weights resident: 1.41 GB → bandwidth ceiling ≈ 71 tok/s (every decode step reads every weight once at ~100 GB/s).
Memory: peak CPU RSS 778 MiB, Metal driver 2161 MiB. Each row: median of 3 runs after a warm-up at that prompt length.

| prompt tokens | KV cache | TTFT (ms) | TPOT (ms) | prefill tok/s | decode tok/s | % of ceiling |
|---:|:---:|---:|---:|---:|---:|---:|
| 16 | yes | 158 | 20.0 | 101 | 50.0 | 71% |
| 256 | yes | 643 | 21.9 | 398 | 45.6 | 65% |
| 1024 | yes | 2277 | 21.9 | 450 | 45.6 | 64% |

| batch | step (ms) | per-sequence tok/s | aggregate tok/s | aggregate vs batch 1 |
|---:|---:|---:|---:|---:|
| 1 | 20.0 | 50.1 | 50.1 | 1.00x |
| 2 | 37.3 | 26.8 | 53.6 | 1.07x |
| 4 | 46.8 | 21.3 | 85.4 | 1.71x |
| 8 | 69.1 | 14.5 | 115.8 | 2.31x |

