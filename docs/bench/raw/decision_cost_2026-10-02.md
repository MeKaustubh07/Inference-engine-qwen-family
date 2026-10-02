# Cost of a decision: forked scoring vs re-reading the context per option (2026-10-02)

Qwen3.5-2B INT4 (Metal), MacBook Air M2 8 GB. Context: a repeated English paragraph + "What is this text about?"
(chat-formatted); options: 4 or 16 short labels (2-6 tokens). Forked = `decision.score_options` (context prefilled
once in 512-token chunks, each option from its own fork, all options in one packed pass). Naive = for each option, a
fresh state, the context prefilled in 512-token chunks, then the option. Best of 2 runs (forked) / 1 run (naive),
after a warm-up; the max |dlogprob| column compares the two paths' scores.
```
context tokens | options | forked s | naive s | speed-up | max |dlogprob|
            89 |       4 |     0.67 |    1.81 |     2.7x | 9.5e-06
            89 |      16 |     1.05 |    6.71 |     6.4x | 1.3e-05
           439 |       4 |     1.39 |    4.65 |     3.4x | 1.5e-05
           439 |      16 |     1.89 |   17.75 |     9.4x | 1.5e-05
          1279 |       4 |     3.36 |   12.53 |     3.7x | 5.7e-06
          1279 |      16 |     3.81 |   53.25 |    14.0x | 1.1e-05
```
A first run used an unchunked naive baseline and measured it at 182 s (4 options) and 759 s (16 options) at 1,279
tokens, i.e. ~45 s per 1,283-token forward. It did not reproduce: re-measured right after, an unchunked forward of
1,024 / 1,152 / 1,280 / 1,283 / 1,344 tokens took 2.29 / 2.59 / 2.99 / 3.02 / 3.03 s, the same as 512-token chunks
(2.37 / 2.75 / 3.02 / 3.09 / 3.15 s). That first run overlapped a network outage and other load on the machine; its
numbers are discarded.
