### metal-kernels-bf16 — qwen3.5-2b, 32 new tokens, M2 8 GB

Weights resident: 3.76 GB → bandwidth ceiling ≈ 27 tok/s (every decode step reads every weight once at ~100 GB/s).
Memory: peak CPU RSS 630 MiB, Metal driver 5177 MiB. Each row: median of 3 runs after a warm-up at that prompt length.

| prompt tokens | KV cache | TTFT (ms) | TPOT (ms) | prefill tok/s | decode tok/s | % of ceiling |
|---:|:---:|---:|---:|---:|---:|---:|
| 16 | yes | 191 | 47.3 | 84 | 21.1 | 80% |
| 256 | yes | 912 | 49.8 | 281 | 20.1 | 76% |
| 1024 | yes | 9757 | 56.7 | 105 | 17.6 | 66% |

| batch | step (ms) | per-sequence tok/s | aggregate tok/s | aggregate vs batch 1 |
|---:|---:|---:|---:|---:|
| 1 | 48.6 | 20.6 | 20.6 | 1.00x |
| 2 | 65.3 | 15.3 | 30.6 | 1.49x |
| 4 | 95.9 | 10.4 | 41.7 | 2.03x |
| 8 | 113.5 | 8.8 | 70.5 | 3.42x |

