"""Order-invariant decisions (src/decision.py, POST /v1/decide): a forked state is independent of its original,
scoring options from forks == scoring each option alone, shuffling the options only shuffles their scores and
embeddings (exactly), options are complete answers, the context is prefilled in chunks, the server runs decisions
stepwise between decode steps and drops cancelled ones, and the choice / boolean / score modes and the endpoint
behave."""
import asyncio
import gc
import math
import sys
import threading
import time

import httpx
import torch

sys.path.insert(0, "src")
from decision import boundary_mask, prepare, prompt_ids, score, score_options, score_steps
from engine import load_engine
from sampler import SamplingParams
from server.app import create_app
from server.metrics import Metrics
from server.scheduler import QueueFull, Scheduler

results = []
def check(name, ok):
    results.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}")

QUESTION = "What is the capital of France? Answer with the city name."
OPTIONS = ["Paris", "Lyon", "Marseille is the capital", "Nice"]        # single- and multi-token options
PERM = [2, 0, 3, 1]


def alone(m, ctx, o, V):
    """log P(o | ctx) from one forward over ctx + o with a fresh state: the reference."""
    lg = m.forward(torch.tensor(ctx + o), state=m.new_state(len(ctx) + len(o)))[:, :V].float().cpu()
    lp = lg - torch.logsumexp(lg, 1, keepdim=True)
    return sum(lp[len(ctx) - 1 + j, o[j]].item() for j in range(len(o)))


def core(eng, tol):
    m, tok, V = eng.model, eng.tokenizer, eng.tokenizer.vocab_size()
    name = f"{eng.name} {m.b.name}"
    ctx = prompt_ids(tok, "", QUESTION, True, eng.chat_style)
    ids = [tok.encode(o) for o in OPTIONS]

    # a fork continues exactly like the original would, and feeding the fork leaves the original untouched
    st = m.new_state(len(ctx) + 8); m.forward(torch.tensor(ctx), state=st, last_only=True)
    fresh = m.new_state(len(ctx) + 8); m.forward(torch.tensor(ctx), state=fresh, last_only=True)
    f = st.fork()
    via_fork = m.forward(torch.tensor(ids[0]), state=f, last_only=True)
    m.forward(torch.tensor(ids[2]), state=st.fork())                    # another fork, fed something else
    orig = m.forward(torch.tensor(ids[0]), state=st, last_only=True)
    ref = m.forward(torch.tensor(ids[0]), state=fresh, last_only=True)
    check(f"{name}: a fork continues like the original ({(via_fork - ref).abs().max().item():.1e}), and feeding a "
          f"fork leaves the original exactly untouched ({(orig - ref).abs().max().item():.1e})",
          torch.equal(orig.cpu(), ref.cpu()) and (via_fork - ref).abs().max().item() < tol and f.length == st.length)

    a = score_options(m, ctx, ids, V, embeddings=True)
    want = [alone(m, ctx, o, V) for o in ids]
    d = max(abs(s.logprob - w) for s, w in zip(a, want))
    check(f"{name}: forked scores == each option scored alone as context + option, max |dlogprob| {d:.1e} < {tol} "
          f"({[round(s.logprob, 3) for s in a]})", d < tol)

    b = score_options(m, ctx, [ids[i] for i in PERM], V, embeddings=True)
    same = all(b[k].logprob == a[i].logprob and torch.equal(b[k].embedding, a[i].embedding) for k, i in enumerate(PERM))
    check(f"{name}: shuffling the options only shuffles their scores and embeddings (bit-exact)", same)

    c = score_options(m, ctx, ids, V, budget=1)                          # one option per packed pass
    d = max(abs(x.logprob - y.logprob) for x, y in zip(a, c))
    check(f"{name}: options in groups of one (memory budget) == all in one pass, max |d| {d:.1e} < {tol}", d < tol)
    return a


# ---------------------------------------------------------------- CPU fp32 (exact)
eng = load_engine("qwen3.5-0.8b", "cpu")
core(eng, 2e-4)
from decision import decide
yes = decide(eng, "boolean", "", "Is Paris the capital of France?")
no = decide(eng, "boolean", "", "Is Lyon the capital of France?")
check(f"boolean: Paris is the capital -> {yes['decision']} (P(yes) {yes['probability']:.2f}); Lyon -> {no['decision']} "
      f"(P(yes) {no['probability']:.2f})", yes["decision"] is True and no["decision"] is False)
labels = ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]
pos = decide(eng, "score", "", "How positive is this review: 'an absolute delight, I loved every minute'?", labels)
neg = decide(eng, "score", "", "How positive is this review: 'a dull, painful waste of two hours'?", labels)
check(f"score: a glowing review rates {pos['expected']:.2f}, a scathing one {neg['expected']:.2f} (0-4 scale)",
      pos["expected"] > 2.5 > neg["expected"])
pick = decide(eng, "choice", "", QUESTION, OPTIONS)
check(f"choice: picks {OPTIONS[pick['decision']]!r}; probabilities sum to 1 "
      f"({sum(o['probability'] for o in pick['options']):.6f})",
      pick["decision"] == 0 and abs(sum(o["probability"] for o in pick["options"]) - 1) < 1e-6)
bad = 0
for args in [("pick", "", "q", ["a", "b"]), ("boolean", "", "q", ["a", "b", "c"]), ("choice", "", "q", ["a"]),
             ("score", "", "q", None), ("choice", "", "q", ["a", ""])]:
    try:
        prepare(eng, *args)
    except ValueError:
        bad += 1
check(f"prepare rejects an unknown type, 3 boolean options, 1 choice, no score labels, an empty option ({bad}/5)",
      bad == 5)

# options are whole words: their tokens, then a token that cannot continue the last word or number
tok, V = eng.tokenizer, eng.tokenizer.vocab_size()
mask = boundary_mask(tok, V)
kinds = {t: bool(mask[tok.encode(t)[0]]) for t in ["0", "es", " Paris", ",", "\n", "<|im_end|>"]}
check(f"boundary tokens: {kinds}", kinds == {"0": False, "es": False, " Paris": True, ",": True, "\n": True,
                                            "<|im_end|>": True})
_, _, plain_ids = prepare(eng, "choice", "", "q", ["Paris", " Lyon"], chat=False)
check("plain prompts put one space before each option", plain_ids == [tok.encode(" Paris"), tok.encode(" Lyon")])
ctx = prompt_ids(tok, "", QUESTION, True, eng.chat_style)
ids = [tok.encode(o) for o in OPTIONS]
with_b = score_options(eng.model, ctx, ids, V, boundary=mask)
want = []
for o in ids:                                                    # reference: one sequence, boundary at its last row
    lg = eng.model.forward(torch.tensor(ctx + o), state=eng.model.new_state(len(ctx) + len(o)))[:, :V].float()
    lp = lg - torch.logsumexp(lg, 1, keepdim=True)
    want.append(sum(lp[len(ctx) - 1 + j, o[j]].item() for j in range(len(o)))
                + torch.logsumexp(lp[-1].masked_fill(~mask, float("-inf")), 0).item())
d = max(abs(x.logprob - w) for x, w in zip(with_b, want))
check(f"forked scores with the boundary term == one-sequence reference, max |d| {d:.1e} < 2e-4", d < 2e-4)
r = decide(eng, "choice", "", "What is 7 + 3? Answer with the number only.", ["1", "10", "7"])
p1, p10 = math.exp(r["options"][0]["logprob"]), math.exp(r["options"][1]["logprob"])
check(f"'10' is not swallowed by its prefix '1': 7 + 3 -> {['1', '10', '7'][r['decision']]!r} "
      f"(P('1' as a word) {p1:.3f}, P('10') {p10:.3f}, disjoint: sum <= 1)", r["decision"] == 1 and p1 + p10 <= 1)

# the context is prefilled in chunks (as the scheduler does): 4-token chunks give the same scores as one pass
ctx = prompt_ids(eng.tokenizer, "Inference engines turn trained weights into answers. " * 6, QUESTION, True,
                 eng.chat_style)
ids = [eng.tokenizer.encode(o) for o in OPTIONS]
big, small = (score_options(eng.model, ctx, ids, eng.tokenizer.vocab_size(), chunk=c) for c in (512, 4))
d = max(abs(x.logprob - y.logprob) for x, y in zip(big, small))
check(f"a {len(ctx)}-token context prefilled in 4-token chunks scores like one pass (max |d| {d:.1e} < 2e-4)", d < 2e-4)

# on the server: a decision runs stepwise (one context chunk per engine step) while a stream keeps generating; a
# cancelled queued decision never runs; queued decisions count against readiness
GREEDY = SamplingParams(temperature=0, repetition_penalty=1.0)
s = Scheduler(eng, Metrics(), max_batch=2, max_waiting=2, kv_blocks=64, prefill_chunk=16)
stream = s.submit(eng.tokenizer.encode("The history of computing is"), GREEDY, 400)
stream.out.get(timeout=120)                                      # it is decoding now
opts, ctx2, opt_ids = prepare(eng, "choice", "Inference engines turn trained weights into answers. " * 8,
                              "What is this text about?", ["Computing", "Cooking"])
n0 = len(stream.generated)
res = s.run_job(lambda: score_steps(eng, "choice", opts, ctx2, opt_ids, chunk=16)).result(timeout=600)
during = len(stream.generated) - n0
chunks = -(-len(ctx2) // 16)
check(f"a decision over a {len(ctx2)}-token context ran in {chunks} chunks while a stream generated {during} tokens, "
      f"and gives the same result as running it whole",
      during >= chunks - 1 and res == score(eng, "choice", opts, ctx2, opt_ids, chunk=16))
s.cancel(stream)
gate, ran = threading.Event(), []
first = s.run_job(lambda: gate.wait(30))                         # holds the engine thread
while s.job is None:
    time.sleep(0.01)
dropped = s.run_job(lambda: ran.append("dropped"))
later = s.run_job(lambda: ran.append("later"))
busy = not s.ready()
try:
    s.run_job(lambda: None); full = False
except QueueFull:
    full = True
dropped.cancel(); gate.set()
later.result(timeout=30)
check(f"a cancelled queued decision never runs ({ran}); with 2 queued, /ready is false and a third is turned away",
      ran == ["later"] and busy and full)
s.shutdown(5)


async def http_suite():
    app = create_app(eng, max_batch=2, max_waiting=4, kv_blocks=64, max_model_len=512)
    sched = app.state.scheduler
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", timeout=300) as c:
        body = {"type": "choice", "question": QUESTION, "options": OPTIONS}
        r = await c.post("/v1/decide", json=body)
        rp = await c.post("/v1/decide", json={**body, "options": [OPTIONS[i] for i in PERM]})
        j, jp = r.json(), rp.json()
        probs = [o["probability"] for o in j["options"]]
        check(f"/v1/decide choice: decision {j['decision']} ({OPTIONS[j['decision']]!r}); the shuffled request picks "
              f"the same option with the same probabilities",
              r.status_code == 200 and j["decision"] == 0 and OPTIONS[PERM[jp["decision"]]] == OPTIONS[j["decision"]]
              and all(jp["options"][k]["probability"] == probs[i] for k, i in enumerate(PERM)))
        rb = await c.post("/v1/decide", json={"type": "boolean", "question": "Is Paris the capital of France?",
                                              "embeddings": True})
        jb = rb.json()
        check(f"/v1/decide boolean with embeddings: {jb['decision']}, one {len(jb['options'][0]['embedding'])}-dim "
              f"embedding per option", rb.status_code == 200 and jb["decision"] is True
              and len(jb["options"][0]["embedding"]) == eng.model.config.hidden_size)
        codes = [(await c.post("/v1/decide", json=b)).status_code for b in (
            {"type": "pick", "question": "q", "options": ["a", "b"]},              # unknown type
            {"type": "choice", "question": "q", "options": ["a"]},                 # one option
            {"type": "choice", "question": "q", "options": ["a", "x" * 2000]},     # option too long
            {"type": "choice", "question": "q " * 600, "options": ["a", "b"]})]    # prompt over max_model_len
        check(f"/v1/decide rejects bad input: {codes} == [422, 422, 400, 400]", codes == [422, 422, 400, 400])
        m = (await c.get("/metrics")).text
        n = [l for l in m.splitlines() if l.startswith("engine_decisions_total ")]
        check(f"decisions counted in /metrics ({n[0] if n else 'missing'})", n and float(n[0].split()[1]) == 3)
    sched.shutdown(5)
    check("no KV blocks used by decisions (they run on their own forked states)", sched.pool.allocator.num_free == 64)


asyncio.run(http_suite())
del eng
gc.collect()

# ---------------------------------------------------------------- Metal INT4: the same properties within INT4/bf16 noise
if torch.backends.mps.is_available():
    eng = load_engine("qwen3.5-0.8b", "metal-int4")
    core(eng, 2e-2)
    del eng
    gc.collect(); torch.mps.empty_cache()

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
