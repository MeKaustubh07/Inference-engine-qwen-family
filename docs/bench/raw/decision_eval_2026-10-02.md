# Zero-shot decisions on avbiswas/bev-decision (test split): qwen3.5-2b metal-kernels-int4

Sample: 30 random pages of 100 rows (seed 0), then up to 100 questions per type with a prompt of at most 1536 tokens, at most 16 options and options of at most 64 tokens (questions skipped: {'choice: option too long': 4, 'score: prompt too long': 1}). Scoring: each option's log-probability per token from its own fork of the context's state (src/decision.py); chat-formatted prompt, each option scored as a whole word at the start of the assistant's answer (its tokens, then a token that cannot continue its last word or number).

| type | questions | accuracy | baselines | notes |
|---|---:|---:|---|---|
| choice | 100 | **44.0%** | random 24.1%; always the first option 29.0% | 2-10 options |
| boolean | 100 | **73.0%** | majority class 70.0% | 30% of labels are yes |
| score | 100 | **33.0%** exact, 68.0% within one | always the middle label 25.0% exact, MAE 1.16 | MAE of the expected label 1.07 |

Order invariance: 40/40 shuffled choice/score questions gave the same decision with exactly the same probabilities.
Time: 258 s for 300 questions (0.86 s each; mean prompt 183 tokens).

Accuracy by domain (all types):

| domain | questions | accuracy |
|---|---:|---:|
| Sentiment, emotion, and moderation | 63 | 59% |
| Retail, product, and shopping | 62 | 47% |
| Spatial and logical reasoning | 42 | 29% |
| Game-state decisions | 22 | 77% |
| Financial reporting and banking | 22 | 45% |
| Response preference and quality | 22 | 32% |
| Software security | 21 | 33% |
| Support and intent routing | 19 | 68% |
| Software engineering and code | 11 | 55% |
| Tool and workflow decisions | 6 | 67% |
| Scientific and paper understanding | 3 | 100% |
| Reading comprehension | 3 | 67% |
| Browser interaction | 2 | 50% |
| Contract evidence | 1 | 100% |
| Spam detection | 1 | 100% |
