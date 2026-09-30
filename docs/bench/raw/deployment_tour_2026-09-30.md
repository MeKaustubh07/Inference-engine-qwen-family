# Deployment tour: the server run and tested by hand (2026-09-30)

Qwen3.5-2B INT4 on the MacBook Air M2 (8 GB), `scripts/serve.py` defaults. Commands and outputs in the order they ran.
"brought back in" = system-wide `vm_stat` Decompressions + Swapins during the request, x 16 KiB pages (other
processes contribute noise of tens of MB). TTFT here is client-side (time to the first content chunk of a
streaming chat request, `probe.py`); the server log's `ttft_s` measures from arrival and runs ~30 ms lower.

## 1. What was holding memory when the slow first tokens were seen (Docker Desktop and VS Code still open)
```
swap 5.8 of 7.2 GB used; compressor ~3.3 GB; ~58 MB free (earlier) / top by memory later:
Python (the server) 2450M | com.apple.Virtualization (Docker Desktop VM) 1877M | desktop app 555M | Code Helper 474M
mysqld 449M | WindowServer 388M | Code Helper 373M | WebKit 342M
server footprint 2,451 MB, of which IOAccelerator (graphics) 2,137 MB: the weights live in Metal buffers
```

## 2. Idle test through the server, before any fix (Docker Desktop and VS Code closed)
```
server ready after ~0 s of waiting
   [13:27:19] swap 2543.75M used, 51% free, compressor 2410 MB
A first request     TTFT 0.688 s   total 1.53 s   brought back in: 1402 MB
   [13:27:20] swap 2543.75M used, 26% free, compressor 1982 MB
B immediately       TTFT 0.236 s   total 1.07 s   brought back in: 28 MB
   [13:27:22] swap 2543.75M used, 27% free, compressor 1974 MB
C after 60 s idle   TTFT 0.717 s   total 1.50 s   brought back in: 1878 MB
   [13:28:23] swap 2535.75M used, 26% free, compressor 2021 MB
D after 180 s more  TTFT 0.809 s   total 1.64 s   brought back in: 2237 MB
   [13:31:25] swap 2717.19M used, 27% free, compressor 1820 MB
```

## 3. In-process experiment: weights not locked vs mlock vs a Metal residency set

One process per variant (`idle_exp.py`): load, warm 3 prefills, lock or not, idle, then time a synced 19-token
prefill. Smoke test with 3 s idle first, then the variants in the order none, mlock, residency, mlock, none.
```
[mlock] 418 weight tensors in 418 buffers, 1.32 GiB; pinned result 1.32; warm prefill 189 ms
[mlock] after   3 s idle: prefill  234 ms, brought back in   274 MB   (21% free)
[residency] 418 weight tensors in 418 buffers, 1.32 GiB; pinned result 2.09; warm prefill 201 ms
[residency] after   3 s idle: prefill  489 ms, brought back in  1560 MB   (18% free)

[none] 418 weight tensors in 418 buffers, 1.32 GiB; pinned result 0.00; warm prefill 186 ms
[none] after  60 s idle: prefill  438 ms, brought back in  1412 MB   (30% free)
[none] after 120 s idle: prefill  530 ms, brought back in  2157 MB   (31% free)
[mlock] 418 weight tensors in 418 buffers, 1.32 GiB; pinned result 1.32; warm prefill 188 ms
[mlock] after  60 s idle: prefill  229 ms, brought back in   416 MB   (32% free)
[mlock] after 120 s idle: prefill  264 ms, brought back in   827 MB   (21% free)
[residency] 418 weight tensors in 418 buffers, 1.32 GiB; pinned result 2.08; warm prefill 198 ms
[residency] after  60 s idle: prefill  494 ms, brought back in  1894 MB   (28% free)
[residency] after 120 s idle: prefill  617 ms, brought back in  2083 MB   (23% free)
[mlock] 418 weight tensors in 418 buffers, 1.32 GiB; pinned result 1.32; warm prefill 187 ms
[mlock] after  60 s idle: prefill  235 ms, brought back in   445 MB   (26% free)
[mlock] after 120 s idle: prefill  278 ms, brought back in   819 MB   (23% free)
[none] 418 weight tensors in 418 buffers, 1.32 GiB; pinned result 0.00; warm prefill 190 ms
[none] after  60 s idle: prefill  396 ms, brought back in  1197 MB   (27% free)
[none] after 120 s idle: prefill  433 ms, brought back in   928 MB   (29% free)
```
("after 120 s idle" is 120 s after the 60 s probe, i.e. 3 minutes of idle in total. The residency set covered
whole MTLHeaps, 2.08 GiB, and did not stop the reclaim; mlock covered exactly the 418 weight buffers.)

mlock's effect, read from the kernel (`mach_vm_region_recurse`, user_wired_count of the VM map entry holding each
buffer; two 64 MiB MPS tensors, 1 GiB heap mapping):
```
before:   (region 1073741824 bytes, user_wired_count 0, pages_resident 65536)
locked:   (region  134217728 bytes, user_wired_count 1, pages_resident  8192)   <- the entry is split to the locked range
unlocked: (region 1073741824 bytes, user_wired_count 0, pages_resident 65536)
```
The system-wide `Pages wired down` count did not move (+0): the GPU driver already wires buffers it has just used, and
releases idle ones, which is when the user wire keeps them in RAM.

## 4. End to end with the fix: weights locked vs not (`--no-lock-weights`), order on, off, off, on
```
=== lock on (server pid 2341)
server ready after ~5 s of waiting
   [21:57:03] swap 2881.31M used, 40% free, compressor 1210 MB
A first request     TTFT 0.383 s   total 1.21 s   brought back in: 91 MB
   [21:57:04] swap 2881.31M used, 37% free, compressor 1411 MB
B immediately       TTFT 0.259 s   total 1.09 s   brought back in: 25 MB
   [21:57:05] swap 2881.31M used, 37% free, compressor 1403 MB
C after 60 s idle   TTFT 0.343 s   total 1.17 s   brought back in: 383 MB
   [21:58:07] swap 2729.31M used, 38% free, compressor 1294 MB
D after 180 s more  TTFT 0.323 s   total 1.15 s   brought back in: 313 MB
   [22:01:08] swap 2633.31M used, 37% free, compressor 1393 MB
weights locked in memory: 1.32 GiB
engine_weights_locked_bytes 1422694400
=== lock off (server pid 2809)
server ready after ~5 s of waiting
   [22:01:33] swap 2617.31M used, 35% free, compressor 1668 MB
A first request     TTFT 0.445 s   total 1.27 s   brought back in: 247 MB
   [22:01:34] swap 2617.31M used, 33% free, compressor 1733 MB
B immediately       TTFT 0.253 s   total 1.06 s   brought back in: 29 MB
   [22:01:35] swap 2617.31M used, 33% free, compressor 1721 MB
C after 60 s idle   TTFT 0.330 s   total 1.16 s   brought back in: 152 MB
   [22:02:37] swap 2609.31M used, 34% free, compressor 1588 MB
D after 180 s more  TTFT 0.512 s   total 1.33 s   brought back in: 881 MB
   [22:05:38] swap 2577.31M used, 35% free, compressor 1557 MB
engine_weights_locked_bytes 0
=== lock off (server pid 3600)
server ready after ~5 s of waiting
   [22:06:03] swap 2577.31M used, 34% free, compressor 1740 MB
A first request     TTFT 0.343 s   total 1.17 s   brought back in: 95 MB
   [22:06:04] swap 2577.31M used, 32% free, compressor 1790 MB
B immediately       TTFT 0.258 s   total 1.08 s   brought back in: 33 MB
   [22:06:05] swap 2577.31M used, 32% free, compressor 1776 MB
C after 60 s idle   TTFT 0.450 s   total 1.28 s   brought back in: 601 MB
   [22:07:07] swap 2569.31M used, 33% free, compressor 1667 MB
D after 180 s more  TTFT 0.668 s   total 1.49 s   brought back in: 1788 MB
   [22:10:08] swap 2465.25M used, 31% free, compressor 1773 MB
engine_weights_locked_bytes 0
=== lock on (server pid 4950)
server ready after ~5 s of waiting
   [22:10:34] swap 2465.25M used, 34% free, compressor 1688 MB
A first request     TTFT 0.431 s   total 1.25 s   brought back in: 135 MB
   [22:10:35] swap 2441.25M used, 31% free, compressor 1751 MB
B immediately       TTFT 0.255 s   total 1.08 s   brought back in: 12 MB
   [22:10:37] swap 2441.25M used, 31% free, compressor 1758 MB
C after 60 s idle   TTFT 0.310 s   total 1.14 s   brought back in: 28 MB
   [22:11:38] swap 2441.25M used, 32% free, compressor 1700 MB
D after 180 s more  TTFT 0.447 s   total 1.30 s   brought back in: 808 MB
   [22:14:39] swap 2281.50M used, 31% free, compressor 1735 MB
weights locked in memory: 1.32 GiB
engine_weights_locked_bytes 1422694400
```

## 5. Stage 5: load test with /metrics sampled every 0.2 s (before the gauge fix)

`scripts/loadgen.py --levels 1,8 --requests 16 --max-tokens 64` against the running server:
```
conc 1: 41.7 tok/s  TTFT p50 223 ms  TPOT p50 20.9 ms  e2e p50 1.54s  (16 ok, 0 rejected)
conc 8: 94.8 tok/s  TTFT p50 817 ms  TPOT p50 73.0 ms  e2e p50 5.41s  (16 ok, 0 rejected)

engine over the whole run (incl. loadgen's 8-prompt warm-up):
  requests 40, generated tokens 2112
  decode steps 1141, rows decoded 2072 -> mean decode batch 1.82
  prefill passes 19, prompt tokens 875 -> 46.1 tokens per packed pass
  peak waiting 0, peak prefilling 0, peak running 8, min free KV blocks 976 of 1024
```
19 passes = 1 (the 8 warm-up prompts) + 16 (one per request at 1 client) + 2 (one per 8-request wave at 8 clients).
Peak waiting and prefilling read 0 because the engine refreshed those gauges once per loop iteration, after admitting:
fixed, /metrics now reads them at scrape time.

## 6. Stage 6: failure behaviour against the live server
```
(1) burst of 80 streaming requests: {200: 72, 429: 8}  Retry-After on 429: {'1'}
    requests_rejected_total +8
(2) all 80 clients gone at t=1.6s; server state while it notices:
    + 0.0s  waiting 20  prefilling 0  running 0  KV free 1024
    + 0.5s  waiting 20  prefilling 0  running 0  KV free 1024
    + 1.0s  waiting 0  prefilling 0  running 0  KV free 1024
(3) one stream closed by the client after 20 of up to 400 tokens; waiting for the server to cancel it:
    running back to 0 after 0.12s; KV free 1024
--- server log since the test started: finish reasons
    {None: 17, 'cancelled': 56} over 73 requests          <- the None lines: fixed, now 'cancelled'
(4) SIGTERM sent to server pid 98577 while the stream had 16 tokens
    new request during the drain: refused (ConnectError)
    in-flight stream: 150 tokens, finish_reason=length, [DONE] received=True, finished 2.77s after SIGTERM
    server process exited 2.87s after SIGTERM
    log: Shutting down / Waiting for connections to close / request_finished (150 tokens, length) /
         Waiting for application shutdown / Application shutdown complete / Finished server process
```
72 accepted = 8 in flight + a full 64-slot queue.
