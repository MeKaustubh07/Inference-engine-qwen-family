### metal-kernels-int4 — qwen3.5-2b, 32 new tokens, M2 8 GB

Weights resident: 1.41 GB → bandwidth ceiling ≈ 71 tok/s (every decode step reads every weight once at ~100 GB/s).
Memory: peak CPU RSS 642 MiB, Metal driver 2161 MiB. Each row: median of 3 runs after a warm-up at that prompt length.

| prompt tokens | KV cache | TTFT (ms) | TPOT (ms) | prefill tok/s | decode tok/s | % of ceiling |
|---:|:---:|---:|---:|---:|---:|---:|
| 16 | yes | 160 | 23.4 | 100 | 42.7 | 60% |
| 256 | yes | 649 | 23.9 | 395 | 41.9 | 59% |
| 1024 | yes | 2308 | 27.1 | 444 | 36.9 | 52% |

| batch | step (ms) | per-sequence tok/s | aggregate tok/s | aggregate vs batch 1 |
|---:|---:|---:|---:|---:|
| 1 | 20.1 | 49.7 | 49.7 | 1.00x |
| 2 | 37.4 | 26.7 | 53.5 | 1.08x |
| 4 | 46.9 | 21.3 | 85.4 | 1.72x |
| 8 | 71.0 | 14.1 | 112.7 | 2.27x |

