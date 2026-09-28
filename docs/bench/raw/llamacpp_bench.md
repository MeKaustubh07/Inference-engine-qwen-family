| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen35 2B Q4_1                 |   1.27 GiB |     1.94 B | BLAS,MTL   |       4 |           pp256 |       600.96 ± 36.03 |
| qwen35 2B Q4_1                 |   1.27 GiB |     1.94 B | BLAS,MTL   |       4 |          pp1024 |        451.90 ± 6.92 |
| qwen35 2B Q4_1                 |   1.27 GiB |     1.94 B | BLAS,MTL   |       4 |            tg32 |         42.28 ± 1.69 |
| qwen35 2B Q4_K - Medium        |   1.29 GiB |     1.94 B | BLAS,MTL   |       4 |           pp256 |       445.44 ± 49.91 |
| qwen35 2B Q4_K - Medium        |   1.29 GiB |     1.94 B | BLAS,MTL   |       4 |          pp1024 |        374.88 ± 6.87 |
| qwen35 2B Q4_K - Medium        |   1.29 GiB |     1.94 B | BLAS,MTL   |       4 |            tg32 |         32.54 ± 2.74 |
| qwen35 2B Q8_0                 |   1.93 GiB |     1.94 B | BLAS,MTL   |       4 |           pp256 |       513.06 ± 52.33 |
| qwen35 2B Q8_0                 |   1.93 GiB |     1.94 B | BLAS,MTL   |       4 |          pp1024 |       438.14 ± 16.82 |
| qwen35 2B Q8_0                 |   1.93 GiB |     1.94 B | BLAS,MTL   |       4 |            tg32 |         28.58 ± 1.65 |

build: 7fe450e19 (11146)
