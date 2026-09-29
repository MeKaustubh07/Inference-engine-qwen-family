# Server gap A/B load test, final code (2026-09-30 00:12–00:36): before (commit bbeb6c6) vs after

After = packed + chunked prefill, batching window (5 ms), batched sampling on CPU logits, plus the review fixes
(the commit that adds this file). Same protocol as `loadgen_2b_int4_2026-09-29_ab.md` (which measured an
intermediate version with the sampling selection on MPS): Qwen3.5-2B INT4, `scripts/serve.py` defaults,
`scripts/loadgen.py --levels 1,2,4,8 --requests 16 --max-tokens 64`, a 90 s idle cool-down and a fresh server
per run, the top 3 processes by CPU sampled every 10 s. Order: before, after, after, before (block 1), after,
before, before, after (block 2). Exclusion rule (8-client TPOT p50 above 120 ms): no run excluded.

## All runs

### Run 1: before

| concurrency | throughput (tok/s) | TTFT p50 | TTFT p95 | TPOT p50 | e2e p50 | e2e p95 | rejected |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 39.8 | 222 ms | 232 ms | 22.0 ms | 1.61 s | 1.63 s | 0 |
| 2 | 43.0 | 418 ms | 444 ms | 42.7 ms | 3.00 s | 3.07 s | 0 |
| 4 | 62.6 | 634 ms | 870 ms | 57.0 ms | 4.10 s | 4.15 s | 0 |
| 8 | 76.9 | 1059 ms | 1723 ms | 92.0 ms | 6.66 s | 6.67 s | 0 |

### Run 2: after

| concurrency | throughput (tok/s) | TTFT p50 | TTFT p95 | TPOT p50 | e2e p50 | e2e p95 | rejected |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40.3 | 230 ms | 237 ms | 21.5 ms | 1.59 s | 1.61 s | 0 |
| 2 | 45.1 | 330 ms | 410 ms | 39.7 ms | 2.83 s | 2.91 s | 0 |
| 4 | 69.5 | 526 ms | 531 ms | 51.0 ms | 3.74 s | 3.76 s | 0 |
| 8 | 94.6 | 838 ms | 841 ms | 72.7 ms | 5.42 s | 5.42 s | 0 |

### Run 3: after

| concurrency | throughput (tok/s) | TTFT p50 | TTFT p95 | TPOT p50 | e2e p50 | e2e p95 | rejected |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40.2 | 226 ms | 236 ms | 21.7 ms | 1.59 s | 1.63 s | 0 |
| 2 | 45.0 | 343 ms | 462 ms | 39.7 ms | 2.84 s | 2.90 s | 0 |
| 4 | 68.8 | 519 ms | 524 ms | 50.9 ms | 3.73 s | 3.74 s | 0 |
| 8 | 92.5 | 842 ms | 1053 ms | 73.0 ms | 5.56 s | 5.63 s | 0 |

### Run 4: before

| concurrency | throughput (tok/s) | TTFT p50 | TTFT p95 | TPOT p50 | e2e p50 | e2e p95 | rejected |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40.1 | 223 ms | 230 ms | 21.9 ms | 1.60 s | 1.61 s | 0 |
| 2 | 41.8 | 417 ms | 446 ms | 43.7 ms | 3.08 s | 3.13 s | 0 |
| 4 | 63.0 | 629 ms | 872 ms | 57.5 ms | 4.07 s | 4.08 s | 0 |
| 8 | 76.9 | 1067 ms | 1718 ms | 92.0 ms | 6.67 s | 6.68 s | 0 |

### Run 5: after

| concurrency | throughput (tok/s) | TTFT p50 | TTFT p95 | TPOT p50 | e2e p50 | e2e p95 | rejected |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 39.3 | 230 ms | 256 ms | 21.9 ms | 1.60 s | 1.75 s | 0 |
| 2 | 45.4 | 329 ms | 401 ms | 39.5 ms | 2.82 s | 2.88 s | 0 |
| 4 | 69.3 | 512 ms | 515 ms | 50.4 ms | 3.69 s | 3.71 s | 0 |
| 8 | 95.0 | 837 ms | 838 ms | 72.2 ms | 5.39 s | 5.39 s | 0 |

### Run 6: before

| concurrency | throughput (tok/s) | TTFT p50 | TTFT p95 | TPOT p50 | e2e p50 | e2e p95 | rejected |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40.3 | 221 ms | 226 ms | 21.8 ms | 1.59 s | 1.60 s | 0 |
| 2 | 41.8 | 416 ms | 449 ms | 44.3 ms | 3.08 s | 3.09 s | 0 |
| 4 | 59.7 | 643 ms | 909 ms | 59.0 ms | 4.42 s | 4.45 s | 0 |
| 8 | 76.8 | 1031 ms | 1677 ms | 91.4 ms | 6.72 s | 6.73 s | 0 |

### Run 7: before

| concurrency | throughput (tok/s) | TTFT p50 | TTFT p95 | TPOT p50 | e2e p50 | e2e p95 | rejected |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40.3 | 219 ms | 229 ms | 21.8 ms | 1.60 s | 1.61 s | 0 |
| 2 | 41.6 | 418 ms | 449 ms | 45.3 ms | 3.08 s | 3.10 s | 0 |
| 4 | 63.1 | 631 ms | 869 ms | 57.1 ms | 4.06 s | 4.10 s | 0 |
| 8 | 77.6 | 1066 ms | 1723 ms | 90.6 ms | 6.62 s | 6.62 s | 0 |

### Run 8: after

| concurrency | throughput (tok/s) | TTFT p50 | TTFT p95 | TPOT p50 | e2e p50 | e2e p95 | rejected |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40.7 | 231 ms | 238 ms | 21.3 ms | 1.57 s | 1.59 s | 0 |
| 2 | 45.7 | 333 ms | 337 ms | 39.4 ms | 2.81 s | 2.85 s | 0 |
| 4 | 70.2 | 517 ms | 521 ms | 50.3 ms | 3.69 s | 3.71 s | 0 |
| 8 | 95.0 | 835 ms | 835 ms | 72.6 ms | 5.40 s | 5.41 s | 0 |

Engine counters per after run: 31–32 packed prefill passes for 1,575 prompt tokens over 72 requests (8 warm-up +
4 levels x 16); 1,897–1,898 decode steps and 4,088 decode rows in both versions.

## CPU samples (top 3 processes by CPU every 10 s)

python = the server, or the loadgen client (same interpreter): when two appear, the larger is the server.

### run 1: before
```
00:12:35  AssistantServices 65.8 | Keychain Circle Notification 28.5 | WebKit 12.1
00:12:45  AssistantServices 91.4 | python 36.5 | Contacts 8.0
00:12:55  AssistantServices 89.3 | python 41.4 | Contacts 6.9
00:13:05  AssistantServices 92.2 | python 38.0 | Contacts 7.3
00:13:15  AssistantServices 89.5 | python 37.6 | Contacts 10.0
00:13:25  AssistantServices 93.0 | python 30.2 | Contacts 7.3
00:13:36  AssistantServices 90.0 | python 21.2 | Contacts 8.1
00:13:46  python 30.7 | CoreServices 27.3 | WebKit 10.4
00:13:56  python 20.9 | WebKit 7.5 | SkyLight 6.2
```

### run 2: after
```
00:15:39  SkyLight 7.8 | WebKit 5.3 | openai.chatgpt-26.917.62051-d 4.2
00:15:49  python 31.6 | SkyLight 6.1 | searchpartyd 4.7
00:15:59  python 29.3 | BiomeStreams 24.7 | SiriTTSService 20.1
00:16:09  replayd 40.0 | python 31.5 | CoreServices 27.5
00:16:19  python 30.9 | SkyLight 7.4 | iCloudDriveCore 5.0
00:16:29  python 40.7 | WebKit 7.2 | SkyLight 6.7
00:16:39  wifip2pd 47.4 | python 19.5 | networkserviceproxy 16.2
00:16:49  python 30.5 | SkyLight 5.9 | python 4.6
```

### run 3: after
```
00:18:36  SkyLight 6.2 | CloudKitDaemon 3.7 | python 3.7
00:18:46  python 36.7 | SkyLight 6.4 | coreaudiod 2.9
00:18:56  python 36.5 | SkyLight 6.2 | coreaudiod 3.3
00:19:06  replayd 37.3 | aned 26.7 | Ecosystem 26.5
00:19:16  python 29.4 | ollama 7.5 | SkyLight 6.1
00:19:26  python 40.3 | SkyLight 6.8 | python 5.4
00:19:36  mobileassetd 94.5 | python 39.6 | SkyLight 6.8
00:19:46  mobileassetd 105.1 | ModelCatalogRuntime 37.4 | assetsubscriptiond 35.8
```

### run 4: before
```
00:21:33  SkyLight 7.6 | WebKit 4.1 | WebKit 3.0
00:21:43  python 40.7 | SkyLight 5.7 | coreaudiod 3.1
00:21:53  replayd 85.7 | python 40.2 | SkyLight 9.4
00:22:03  python 37.3 | SkyLight 6.6 | WebKit 3.7
00:22:13  TelephonyUtilities 72.7 | networkserviceproxy 51.5 | python 24.6
00:22:23  python 28.2 | SkyLight 7.1 | WebKit 5.8
00:22:33  python 22.6 | SkyLight 6.6 | WebKit 3.1
00:22:43  python 31.8 | SkyLight 8.0 | WebKit 3.6
00:22:53  python 22.0 | SkyLight 6.9 | WebKit 4.3
```

### run 5: after
```
00:24:39  ReplicatorCore 33.3 | SkyLight 25.8 | Visual Studio Code 25.7
00:24:49  WorkflowKit 48.8 | python 33.7 | VoiceShortcuts 24.2
00:24:59  WorkflowKit 63.9 | python 35.5 | SkyLight 21.4
00:25:09  replayd 79.2 | WorkflowKit 47.8 | AXAssetLoader 43.3
00:25:20  WorkflowKit 61.7 | python 31.1 | VoiceShortcuts 26.2
00:25:30  python 30.0 | desktop app 1.3 | desktop app 1.1
00:25:40  python 35.9 | desktop app 1.6 | python 1.5
00:25:50  ControlCenter 19.2 | python 15.6 | CoreDuetContext 3.8
```

### run 6: before
```
00:27:37  desktop app 6.3 | Visual Studio Code 2.1 | Visual Studio Code 1.5
00:27:48  python 39.9 | Visual Studio Code 1.3 | python 0.6
00:27:58  python 45.0 | launchd 12.8 | CommerceKit 4.8
00:28:08  python 38.6 | mongod 1.1 | python 0.9
00:28:18  python 38.4 | desktop app 1.0 | python 1.0
00:28:28  python 32.4 | launchd 18.4 | Keychain Circle Notification 1.0
00:28:38  WallpaperAerialsExtensio 67.9 | AuthenticationServicesAgent 36.0 | biomesyncd 34.6
00:28:48  AppleNeuralEngine 98.8 | AssetMetricsExtension.ap 85.2 | fairplaydeviceidentityd 83.3
00:28:58  python 20.4 | launchd 12.1 | ContinuityCaptureAgent 8.2
```

### run 7: before
```
00:30:42  desktop app 5.5 | desktop app 4.9 | python 3.0
00:30:52  python 41.2 | python 0.6 | desktop app 0.6
00:31:02  python 42.0 | Keychain Circle Notification 2.2 | loginwindow 2.0
00:31:12  python 37.7 | Google Chrome 3.0 | feedbackd 1.7
00:31:22  python 39.6 | sysmond 3.0 | python 1.1
00:31:32  python 33.1 | ScreenTimeCore 9.1 | HearingCore 3.0
00:31:42  python 20.4 | Google Chrome 2.3 | Google Chrome 0.6
00:31:52  python 32.7 | Visual Studio Code 0.9 | Google Chrome 0.8
00:32:03  python 19.6 | Visual Studio Code 15.9 | sysmond 4.2
```

### run 8: after
```
00:33:46  searchpartyd 16.1 | dasd 6.9 | docker 6.3
00:33:56  python 35.1 | ospredictiond 31.2 | BiomeStreams 1.0
00:34:06  python 36.6 | airportd 0.8 | mongod 0.7
00:34:16  python 40.8 | CoreDuetContext 1.7 | mongod 1.2
00:34:26  python 30.6 | sysmond 3.7 | python 0.9
00:34:36  python 32.9 | desktop app 3.3 | cfprefsd 1.5
00:34:46  Keychain Circle Notification 26.9 | coreduetd 23.8 | CalendarDaemon 23.0
00:34:56  python 33.0 | desktop app 6.6 | HearingCore 2.2
```
