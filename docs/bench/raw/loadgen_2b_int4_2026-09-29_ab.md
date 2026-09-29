# Server gap A/B load test (2026-09-29): before (commit bbeb6c6) vs an intermediate after

After = packed + chunked prefill, batching window, and batched sampling with its selection on MPS (later moved to
the CPU, which is 4x faster for it; the final code is measured in `loadgen_2b_int4_2026-09-30_ab_final.md`).

Qwen3.5-2B INT4, `scripts/serve.py` defaults (max_batch 8, 1024 KV blocks, `--prefill-chunk 512`, `--batch-wait-ms 5` for
after), `scripts/loadgen.py --levels 1,2,4,8 --requests 16 --max-tokens 64` (streaming chat, temperature 0.7, top-k 20,
top-p 0.8, fixed seeds). Before = a git worktree at bbeb6c6 (same scheduler code as the published final load test at
f9280fa); after = the working tree. Each run: 90 s idle cool-down, fresh server process, warm-up, 4 levels.
Block 1 order: before, after, after, before, before, after. Block 2 order: after, before, before, after (reversed).
Block 2 also sampled the top 3 processes by CPU every 10 s during each run.

**Exclusion rule, applied to both variants:** a run is excluded when its 8-client TPOT p50 exceeds 120 ms (the
other 7 runs: 75.8–105.7 ms). Excluded: block 1 run 5 (before, 150.3 ms), block 1 run 6 (after, 242.6 ms), block 2
run 3 (before, 240.8 ms). In each, the 1-client level was normal and the slowdown started partway through the run:
at the 4-client level in block 1 run 5, during the 2-client level's last waves in block 1 run 6 and block 2 run 3.
Block 2 run 3's CPU samples: from 23:41:03, while the server's step time tripled, macOS background services took
the top slots (ModelCatalogRuntime, AssistantServices, mobileassetd, coreduetd, spotlightknowledged,
WorkflowKit/VoiceShortcuts at 28–66% each). The python process listed fell from ~33–42% to 4–14% in the four of
those ten samples that list one (it may be the loadgen client there); in the other six no python process made the
top three (third slot 8–45%). If the server did use less CPU while running slower, it was waiting: on a GPU shared
with another workload, or on memory (swap in use right after block 1, `sysctl vm.swapusage` at 23:31:48:
`total = 6144.00M  used = 5615.50M  free = 528.50M  (encrypted)`). These samples cannot settle either point.

## All runs

### Block 1, run 1: before

| concurrency | throughput (tok/s) | TTFT p50 | TTFT p95 | TPOT p50 | e2e p50 | e2e p95 | rejected |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40.0 | 217 ms | 231 ms | 22.0 ms | 1.60 s | 1.61 s | 0 |
| 2 | 41.5 | 420 ms | 450 ms | 45.1 ms | 3.09 s | 3.12 s | 0 |
| 4 | 62.3 | 635 ms | 867 ms | 57.6 ms | 4.13 s | 4.15 s | 0 |
| 8 | 75.0 | 1059 ms | 1724 ms | 93.4 ms | 6.91 s | 6.92 s | 0 |

### Block 1, run 2: after

| concurrency | throughput (tok/s) | TTFT p50 | TTFT p95 | TPOT p50 | e2e p50 | e2e p95 | rejected |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 39.6 | 232 ms | 240 ms | 22.1 ms | 1.61 s | 1.65 s | 0 |
| 2 | 43.8 | 340 ms | 592 ms | 40.4 ms | 2.91 s | 3.12 s | 0 |
| 4 | 66.6 | 540 ms | 638 ms | 52.1 ms | 3.85 s | 3.92 s | 0 |
| 8 | 87.0 | 835 ms | 1068 ms | 80.9 ms | 5.93 s | 5.94 s | 0 |

### Block 1, run 3: after

| concurrency | throughput (tok/s) | TTFT p50 | TTFT p95 | TPOT p50 | e2e p50 | e2e p95 | rejected |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 39.9 | 228 ms | 238 ms | 21.8 ms | 1.60 s | 1.63 s | 0 |
| 2 | 43.3 | 364 ms | 612 ms | 40.9 ms | 2.96 s | 3.14 s | 0 |
| 4 | 67.0 | 602 ms | 670 ms | 51.7 ms | 3.86 s | 3.89 s | 0 |
| 8 | 84.6 | 836 ms | 1392 ms | 78.6 ms | 6.26 s | 6.31 s | 0 |

### Block 1, run 4: before

| concurrency | throughput (tok/s) | TTFT p50 | TTFT p95 | TPOT p50 | e2e p50 | e2e p95 | rejected |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 39.9 | 216 ms | 230 ms | 22.0 ms | 1.60 s | 1.63 s | 0 |
| 2 | 42.1 | 416 ms | 449 ms | 42.4 ms | 3.07 s | 3.12 s | 0 |
| 4 | 60.1 | 652 ms | 888 ms | 59.2 ms | 4.28 s | 4.51 s | 0 |
| 8 | 67.3 | 1197 ms | 1923 ms | 105.7 ms | 7.62 s | 7.63 s | 0 |

### Block 1, run 5: before — **excluded**

| concurrency | throughput (tok/s) | TTFT p50 | TTFT p95 | TPOT p50 | e2e p50 | e2e p95 | rejected |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 39.9 | 220 ms | 230 ms | 21.9 ms | 1.60 s | 1.62 s | 0 |
| 2 | 41.7 | 416 ms | 441 ms | 44.6 ms | 3.10 s | 3.11 s | 0 |
| 4 | 53.6 | 652 ms | 995 ms | 67.5 ms | 4.82 s | 5.38 s | 0 |
| 8 | 46.7 | 1497 ms | 2540 ms | 150.3 ms | 12.01 s | 12.02 s | 0 |

### Block 1, run 6: after — **excluded**

| concurrency | throughput (tok/s) | TTFT p50 | TTFT p95 | TPOT p50 | e2e p50 | e2e p95 | rejected |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40.0 | 230 ms | 235 ms | 21.7 ms | 1.60 s | 1.65 s | 0 |
| 2 | 41.5 | 347 ms | 436 ms | 42.8 ms | 3.03 s | 3.50 s | 0 |
| 4 | 35.6 | 813 ms | 946 ms | 117.0 ms | 8.19 s | 9.27 s | 0 |
| 8 | 30.1 | 1960 ms | 1962 ms | 242.6 ms | 17.25 s | 17.25 s | 0 |

### Block 2, run 1: after

| concurrency | throughput (tok/s) | TTFT p50 | TTFT p95 | TPOT p50 | e2e p50 | e2e p95 | rejected |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40.2 | 228 ms | 237 ms | 21.7 ms | 1.59 s | 1.61 s | 0 |
| 2 | 44.4 | 333 ms | 392 ms | 40.3 ms | 2.88 s | 2.92 s | 0 |
| 4 | 66.1 | 528 ms | 682 ms | 52.4 ms | 3.86 s | 3.95 s | 0 |
| 8 | 89.8 | 842 ms | 842 ms | 78.0 ms | 5.75 s | 5.75 s | 0 |

### Block 2, run 2: before

| concurrency | throughput (tok/s) | TTFT p50 | TTFT p95 | TPOT p50 | e2e p50 | e2e p95 | rejected |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40.0 | 223 ms | 230 ms | 21.9 ms | 1.61 s | 1.62 s | 0 |
| 2 | 41.8 | 417 ms | 450 ms | 43.8 ms | 3.09 s | 3.10 s | 0 |
| 4 | 62.6 | 632 ms | 871 ms | 57.6 ms | 4.09 s | 4.12 s | 0 |
| 8 | 74.9 | 1082 ms | 1738 ms | 94.1 ms | 6.86 s | 6.88 s | 0 |

### Block 2, run 3: before — **excluded**

| concurrency | throughput (tok/s) | TTFT p50 | TTFT p95 | TPOT p50 | e2e p50 | e2e p95 | rejected |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40.0 | 220 ms | 230 ms | 22.0 ms | 1.60 s | 1.62 s | 0 |
| 2 | 37.2 | 418 ms | 574 ms | 46.1 ms | 3.18 s | 4.42 s | 0 |
| 4 | 16.9 | 1528 ms | 3558 ms | 239.3 ms | 18.63 s | 21.89 s | 0 |
| 8 | 27.5 | 3097 ms | 6173 ms | 240.8 ms | 22.02 s | 22.03 s | 0 |

### Block 2, run 4: after

| concurrency | throughput (tok/s) | TTFT p50 | TTFT p95 | TPOT p50 | e2e p50 | e2e p95 | rejected |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40.2 | 229 ms | 238 ms | 21.7 ms | 1.59 s | 1.61 s | 0 |
| 2 | 44.7 | 334 ms | 345 ms | 40.3 ms | 2.87 s | 2.90 s | 0 |
| 4 | 66.3 | 564 ms | 687 ms | 52.1 ms | 3.86 s | 3.98 s | 0 |
| 8 | 91.1 | 844 ms | 847 ms | 75.8 ms | 5.62 s | 5.62 s | 0 |

Engine counters per after run: 32 packed prefill passes (31 in block 1 run 6) for 1,575 prompt tokens over 72
requests (8 warm-up + 4 levels x 16); 1,897–1,898 decode steps, 4,088 decode rows in both variants.

## CPU samples, block 2 (top 3 processes by CPU every 10 s)

python = the server, or the loadgen client (same interpreter): when two appear, the larger is the server.

### run 1: after
```
23:33:59  VoiceShortcuts 18.0 | WorkflowKit 13.5 | SkyLight 9.2
23:34:09  WorkflowKit 55.2 | AXAssetLoader 37.1 | python 35.7
23:34:19  python 33.9 | wifip2pd 17.7 | airportd 8.5
23:34:29  python 28.7 | SkyLight 5.9 | coreaudiod 3.3
23:34:40  replayd 77.3 | python 27.9 | aned 12.6
23:34:50  python 29.6 | SkyLight 7.7 | WebKit 4.1
23:35:00  python 31.1 | NotificationCenter 11.9 | SkyLight 8.1
23:35:10  DataAccess 11.9 | trustd 7.9 | python 7.7
```

### run 2: before
```
23:36:58  UniversalControl 16.5 | launchd 13.8 | SkyLight 10.5
23:37:08  python 40.8 | SkyLight 7.4 | online-auth-agent 5.4
23:37:18  python 40.4 | replayd 33.3 | SkyLight 7.3
23:37:28  python 38.3 | CommerceKit 21.1 | launchd 6.7
23:37:38  python 29.4 | SkyLight 5.8 | WebKit 3.1
23:37:48  python 33.9 | SkyLight 8.3 | PerfPowerServices 4.3
23:37:58  python 23.2 | SkyLight 5.5 | launchd 3.8
23:38:08  AppleMediaServices 35.5 | pkd 33.0 | python 32.2
23:38:18  mobileassetd 93.5 | spotlightknowledged.updater 39.8 | python 23.6
```

### run 3: before
```
23:40:02  SkyLight 7.0 | Finder 6.1 | WebKit 4.4
23:40:12  CoreTelephony 58.5 | networkserviceproxy 29.5 | python 28.2
23:40:22  python 42.2 | SkyLight 6.1 | coreaudiod 3.1
23:40:32  python 39.5 | SkyLight 6.3 | coreaudiod 3.1
23:40:42  python 34.9 | SkyLight 8.1 | WebKit 4.7
23:40:53  python 32.5 | SkyLight 6.4 | ContinuityCaptureAgent 6.0
23:41:03  python 13.8 | SkyLight 5.5 | coreaudiod 3.0
23:41:13  ModelCatalogRuntime 46.0 | opendirectoryd 33.0 | AssistantServices 27.9
23:41:23  WebKit 18.6 | WebKit 17.4 | SkyLight 16.7
23:41:33  SkyLight 5.9 | python 4.1 | WebKit 3.5
23:41:43  SkyLight 7.3 | python 5.0 | coreaudiod 2.9
23:41:53  mobileassetd 31.0 | assetsubscriptiond 25.0 | pkd 20.0
23:42:03  coreduetd 52.4 | spotlightknowledged.updater 47.5 | VoiceShortcuts 44.5
23:42:13  WorkflowKit 63.5 | VoiceShortcuts 19.6 | SkyLight 8.2
23:42:23  WorkflowKit 65.5 | VoiceShortcuts 23.6 | Intents 7.9
23:42:33  python 12.5 | SkyLight 6.1 | WebKit 4.0
```

### run 4: after
```
23:44:18  searchpartyd 15.0 | SkyLight 6.1 | locationd 3.4
23:44:28  python 32.9 | SkyLight 5.7 | WebKit 3.4
23:44:38  python 32.3 | Visual Studio Code 7.3 | SkyLight 6.4
23:44:48  python 32.2 | SkyLight 5.8 | coreaudiod 3.3
23:44:58  python 29.4 | launchd 10.0 | SkyLight 7.7
23:45:08  python 25.6 | SkyLight 8.1 | audioanalyticsd 5.1
23:45:18  python 29.5 | SkyLight 5.5 | coreaudiod 2.7
23:45:28  SkyLight 12.7 | launchd 9.3 | python 7.5
```

## Earlier today, not part of the A/B

A first run of the new code before the batching window was added, same settings, right after the test suites:
```
after_run1:
| 1 | 40.2 | 220 ms | 231 ms | 21.7 ms | 1.59 s | 1.62 s | 0 |
| 2 | 43.0 | 432 ms | 471 ms | 43.4 ms | 2.98 s | 3.01 s | 0 |
| 4 | 60.3 | 682 ms | 1096 ms | 55.6 ms | 4.30 s | 4.40 s | 0 |
| 8 | 72.3 | 1048 ms | 1106 ms | 98.1 ms | 7.28 s | 7.29 s | 0 |
after_run2:
| 1 | 24.2 | 421 ms | 493 ms | 36.7 ms | 2.74 s | 3.03 s | 0 |
| 2 | 22.9 | 738 ms | 1514 ms | 73.8 ms | 5.30 s | 7.29 s | 0 |
| 4 | 47.3 | 844 ms | 879 ms | 74.1 ms | 5.48 s | 5.55 s | 0 |
| 8 | 62.4 | 1192 ms | 1232 ms | 113.7 ms | 8.39 s | 8.40 s | 0 |
```
Run 1 was no better than that session's baseline (`loadgen_2b_int4_2026-09-29_before_run1.md`, 77.0 tok/s at 8
clients); run 2 was slow even at 1 client (24.2 tok/s): something outside the server slowed the machine (not identified). The server log of run 1
(per-request TTFT and latency; no arrival timestamps) shows each 8-client wave's first request prefilled alone
(~20 tokens) and the other 7, arriving within ~3–11 ms (gaps up to ~6 ms, inferred from their TTFTs), packed
into a ~155-token pass: the idle engine started on the first arrival. The request prefilled alone then finished a
decode step ahead of the rest, so its client's next request again arrived first. This is what the batching
window fixes.
