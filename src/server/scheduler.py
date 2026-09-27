"""Continuous-batching scheduler.

One engine thread owns the GPU. Requests wait in a bounded queue (full -> QueueFull -> HTTP 429). Each loop:
  1. admit waiting requests while there is a batch slot and enough free KV blocks, prefilling each one;
  2. run ONE batched decode step for every running request (each weight read once for the whole batch);
  3. stream each new token to its request; finished requests free their KV blocks immediately, so a waiting
     request can take the slot on the very next step (no waiting for the whole batch to finish).
If the KV pool runs dry mid-generation, the newest running request is preempted: its blocks are freed and it is
re-queued to recompute its prefix later (vLLM-style recompute preemption); its client sees no gap in the text.
"""
import collections
import itertools
import queue
import threading
import time
from dataclasses import dataclass, field

import torch

from sampler import SamplingParams, sample
from state import OutOfBlocks


class QueueFull(Exception):
    pass


@dataclass(eq=False)                                  # identity semantics: hashable, compared by object
class Request:
    id: str
    prompt_ids: list[int]
    params: SamplingParams
    max_new_tokens: int
    out: object = field(default_factory=queue.Queue)       # anything with .put: ("token", text)... then ("done", reason) or ("error", msg)
    arrived: float = field(default_factory=time.perf_counter)
    generated: list[int] = field(default_factory=list)
    emitted: str = ""
    prefix_offset: int = 0                                # incremental detokenization window (see _emit)
    read_offset: int = 0
    state: object = None
    first_token_at: float | None = None
    last_token_at: float | None = None
    generator: torch.Generator | None = None
    cancelled: bool = False
    finish_reason: str | None = None


class Scheduler:
    def __init__(self, engine, metrics, max_batch: int = 8, max_waiting: int = 64, kv_blocks: int = 1024,
                 block_size: int = 16, max_model_len: int = 4096):
        self.eng, self.metrics = engine, metrics
        self.model, self.tok = engine.model, engine.tokenizer
        self.max_batch, self.max_waiting, self.max_model_len = max_batch, max_waiting, max_model_len
        self.pool = self.model.new_paged_pool(kv_blocks, block_size)
        self.block_size = block_size
        self.waiting: collections.deque[Request] = collections.deque()
        self.running: list[Request] = []
        self.active: set[Request] = set()                     # submitted and not finished (waiting, prefilling, running)
        self.cond = threading.Condition()
        self.accepting = True
        self._stop = False
        self._ids = itertools.count()
        self._warmup()
        metrics.set("kv_blocks_total", kv_blocks)
        metrics.set("kv_blocks_free", self.pool.allocator.num_free)
        self.thread = threading.Thread(target=self._loop, name="engine", daemon=True)
        self.thread.start()

    def _warmup(self) -> None:
        """Page in every weight and build every fused/quantized tensor before /ready says yes (weights load lazily),
        through both the single-sequence and the batched decode paths. Fails fast on a bad model/backend combo."""
        states = [self.model.new_paged_state(self.pool) for _ in range(2)]
        for st in states:
            self.model.forward(torch.tensor([0]), state=st, last_only=True)
        self.model.decode_batch([0, 0], states)
        for st in states:
            st.free()

    # ------------------------------------------------------------------ public API (any thread)
    def submit(self, prompt_ids: list[int], params: SamplingParams, max_new_tokens: int, out=None) -> Request:
        if not prompt_ids:
            raise ValueError("empty prompt")
        if len(prompt_ids) + max_new_tokens > self.max_model_len:
            raise ValueError(f"prompt ({len(prompt_ids)}) + max_tokens ({max_new_tokens}) exceeds {self.max_model_len}")
        if self._blocks_for(len(prompt_ids) + max_new_tokens) > self.pool.allocator.num_blocks:
            raise ValueError("request can never fit in the KV pool")    # would be preempted forever otherwise
        req = Request(f"req-{next(self._ids)}", list(prompt_ids), params, max_new_tokens)
        if out is not None:
            req.out = out
        if params.seed is not None:
            req.generator = torch.Generator().manual_seed(params.seed)
        with self.cond:
            if not self.accepting or len(self.waiting) >= self.max_waiting:
                self.metrics.inc("requests_rejected_total")
                raise QueueFull("server busy")
            self.waiting.append(req)
            self.active.add(req)
            self.metrics.inc("requests_total")
            self.metrics.inc("prompt_tokens_total", len(prompt_ids))
            self.cond.notify()
        return req

    def cancel(self, req: Request) -> None:
        """Any thread. A queued request is dropped now (its queue slot frees at once); a running one stops at the
        engine's next step."""
        req.cancelled = True
        with self.cond:
            queued = req in self.waiting
            if queued:
                self.waiting.remove(req)
        if queued:
            self._finish(req, "cancelled")

    def ready(self) -> bool:
        return self.accepting and self.thread.is_alive() and len(self.waiting) < self.max_waiting

    def shutdown(self, drain_timeout: float = 30.0) -> None:
        """Graceful: stop accepting, let running/waiting requests finish (up to the timeout), then stop the loop."""
        with self.cond:
            self.accepting = False
            self.cond.notify()
        deadline = time.perf_counter() + drain_timeout
        while self.active and time.perf_counter() < deadline:  # includes a request mid-prefill (in neither list)
            time.sleep(0.05)
        with self.cond:
            self._stop = True
            self.cond.notify()
        self.thread.join(timeout=30)                          # the loop exits after its current step
        for r in list(self.active):                           # drain timed out: tell the stragglers
            self._finish(r, "error", "server shutting down")

    # ------------------------------------------------------------------ engine thread
    def _loop(self) -> None:
        while True:
            with self.cond:
                while not self._stop and not self.waiting and not self.running:
                    self.cond.wait()
                if self._stop:
                    return
            try:
                self._admit()
                if self.running:
                    self._decode_step()
            except Exception as e:                            # never let one bad step kill the server
                for r in list(self.running):
                    self._finish(r, "error", str(e))
            self._update_gauges()

    def _blocks_for(self, n_tokens: int) -> int:
        return -(-n_tokens // self.block_size)

    def _admit(self) -> None:
        while True:
            with self.cond:
                if not self.waiting or len(self.running) >= self.max_batch:
                    return
                req = self.waiting[0]
                if req.cancelled:
                    self.waiting.popleft(); self._finish(req, "cancelled"); continue
                ids = req.prompt_ids + req.generated          # a preempted request recomputes what it produced
                # headroom: keep one free block per running sequence, since each may cross a block boundary on the
                # next step; otherwise the newcomer would be preempted right away and its prefill wasted
                if self._blocks_for(len(ids) + 1) + len(self.running) > self.pool.allocator.num_free:
                    return                                    # wait for running requests to free blocks
                self.waiting.popleft()
            try:
                req.state = self.model.new_paged_state(self.pool)
                logits = self.model.forward(torch.tensor(ids), state=req.state, last_only=True)[0]
                token = self._sample(req, logits[: self.tok.vocab_size()].float().cpu())
            except Exception as e:                            # fail this request, not the server
                self._finish(req, "error", f"prefill failed: {e}")
                continue
            self.running.append(req)
            self._emit(req, token)                            # first token (or, after preemption, the next one)

    def _decode_step(self) -> None:
        active = [r for r in self.running if r.finish_reason is None]
        for r in active:
            if r.cancelled:
                self._finish(r, "cancelled")
        active = [r for r in self.running if r.finish_reason is None]
        # every sequence needs room for one more position; preempt the newest until they fit
        while active and sum(self._blocks_for(r.state.length + 1) - self._blocks_for(r.state.length) for r in active) \
                > self.pool.allocator.num_free:
            victim = active.pop()
            self._preempt(victim)
        if not active:
            return
        logits = self.model.decode_batch([r.generated[-1] for r in active], [r.state for r in active])
        self.metrics.inc("decode_steps_total")
        self.metrics.inc("decode_sequences_total", len(active))
        logits = logits[:, : self.tok.vocab_size()].float().cpu()      # one device->host copy for the batch
        for r, lg in zip(active, logits):
            try:
                token = self._sample(r, lg)
            except Exception as e:                            # a bad request fails alone, not its batch-mates
                self._finish(r, "error", f"sampling failed: {e}")
                continue
            self._emit(r, token)

    def _sample(self, req: Request, logits: torch.Tensor) -> int:
        """logits: [vocab] on the CPU, padding ids already sliced off."""
        return sample(logits, req.prompt_ids + req.generated, req.params, req.generator)

    def _emit(self, req: Request, token: int) -> None:
        now = time.perf_counter()
        if token in self.eng.eos_ids:
            self._finish(req, "stop"); return
        if req.first_token_at is None:
            req.first_token_at = now
            self.metrics.ttft.observe(now - req.arrived)
        else:
            self.metrics.tpot.observe(now - req.last_token_at)
        req.last_token_at = now
        req.generated.append(token)
        self.metrics.inc("generation_tokens_total")
        # Incremental detokenization: decode only a short window instead of the whole output every token.
        # prefix = text of the last emitted tokens, full = that plus the new ones; the difference is the new text.
        # Held back while it ends in U+FFFD (a multi-byte character split across tokens).
        prefix = self.tok.decode(req.generated[req.prefix_offset:req.read_offset])
        full = self.tok.decode(req.generated[req.prefix_offset:])
        if len(full) > len(prefix) and not full.endswith("\ufffd"):
            piece = full[len(prefix):]
            req.out.put(("token", piece))
            req.emitted += piece
            req.prefix_offset, req.read_offset = req.read_offset, len(req.generated)
        if len(req.generated) >= req.max_new_tokens:
            self._finish(req, "length")

    def _preempt(self, req: Request) -> None:
        req.state.free()
        req.state = None
        self.running.remove(req)
        with self.cond:
            self.waiting.appendleft(req)                      # resume first, recomputing prompt + generated
        self.metrics.inc("requests_preempted_total")

    def _finish(self, req: Request, reason: str, error: str | None = None) -> None:
        if req.finish_reason is not None:
            return
        req.finish_reason = reason
        text = self.tok.decode(req.generated) if req.generated else ""
        if len(text) > len(req.emitted):
            req.out.put(("token", text[len(req.emitted):]))
            req.emitted = text
        if req.state is not None:                             # free first: whoever sees "done" sees the blocks back
            req.state.free()
            req.state = None
        if req in self.running:
            self.running.remove(req)
        self.active.discard(req)
        req.out.put(("error", error) if error else ("done", reason))
        self.metrics.inc("requests_finished_total")
        if reason in ("stop", "length"):                      # completed requests only; errors/cancels are counted
            self.metrics.e2e.observe(time.perf_counter() - req.arrived)

    def _update_gauges(self) -> None:
        self.metrics.set("running_requests", len(self.running))
        self.metrics.set("waiting_requests", len(self.waiting))
        self.metrics.set("kv_blocks_free", self.pool.allocator.num_free)
