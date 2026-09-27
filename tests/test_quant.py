"""Weeks 11-12: INT8/INT4 quantization math, kernels, .qt files, and model accuracy vs bf16."""
import gc
import math
import os
import sys
import tempfile
import torch

sys.path.insert(0, "src")
from backend.metal import MetalBackend
from config import ModelConfig
from generate import generate_greedy
from models.qwen2 import Qwen2Model
from ops import softmax
from quant import QtFile, QuantTensor, concat_rows, quantize, save_qt
from tokenizer import Tokenizer
from weight_loader import SafetensorsFile

results = []
def check(name, ok):
    results.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}")
torch.manual_seed(0)

# 1. quantize -> dequantize error stays within half a quantization step, per block (+ fp16 rounding of min/scale)
w = torch.randn(64, 256) * 0.05 + 0.02          # off-center blocks: where asymmetric INT4 pays off
for scheme in ("int8", "int4"):
    q = quantize(w, scheme)
    err = (q.dequantize() - w).abs().view(64, -1, 32)
    step = q.scales.float()[..., None]
    check(f"{scheme}: |dequant - w| <= step/2 everywhere (worst ratio {(err / step).max().item():.3f})", (err <= step / 2 * 1.01 + 1e-6).all())
    check(f"{scheme}: {q.nbytes} bytes vs {w.numel() * 2} in bf16 ({w.numel() * 2 / q.nbytes:.1f}x smaller)", q.nbytes < w.numel() * 2)

# 2. INT4 packing order: element 2j in the low nibble, 2j+1 in the high nibble
w = torch.tensor([[1.0, -1.0] * 16])                    # min -1, max 1 -> scale 2/15 -> q = 15 (for 1), 0 (for -1)
q = quantize(w, "int4")
check(f"int4 packing: first byte {q.data[0, 0].item():#04x} == 0x0f (low nibble q=15 for +1, high q=0 for -1)", q.data[0, 0].item() == 0x0F)

# 3. stacking quantized rows == quantizing the stacked matrix (why fused QKV needs no re-quantization)
a, b = torch.randn(96, 128), torch.randn(32, 128)
qa, qb, qab = quantize(a, "int4"), quantize(b, "int4"), quantize(torch.cat([a, b]), "int4")
st = concat_rows([qa, qb])
check("concat_rows(q(a), q(b)) == q(cat(a, b))", torch.equal(st.data, qab.data) and torch.equal(st.scales, qab.scales))

# 4. quantized matvec kernels == dequantized reference (the kernels dequantize in registers)
mb8, mb4 = MetalBackend("int8"), MetalBackend("int4")
for N, K in ((896, 896), (4864, 896), (896, 4864)):
    W = (torch.randn(N, K) * 0.05).to(torch.bfloat16)
    x = torch.randn(1, K, device="mps"); r = torch.randn(1, N, device="mps")
    for mb in (mb8, mb4):
        qw = mb.prepare(W)
        ref = x @ qw.dequantize().T + r
        rel = ((mb.linear(x, qw, residual=r) - ref).abs().max() / ref.abs().max()).item()
        check(f"{mb.scheme} matvec {N}x{K} (+residual) vs dequantized reference: rel diff {rel:.1e}", rel < 1e-4)

# 4b. policy names fused tensors; checkpoint component names must map onto them
from quant import fused_name, scheme_for
from quant import policy_group
check("policy groups: Qwen2 and Qwen3.5 component and fused names map to one group id",
      policy_group("model.layers.3.self_attn.k_proj.weight") == policy_group("model.layers.3.self_attn.qkv.weight")
      == policy_group("layers.3.self_attn.qkvg.weight") == policy_group("model.language_model.layers.3.self_attn.q_proj.weight")
      == "layers.3.attn_in"
      and policy_group("model.language_model.layers.5.linear_attn.in_proj_z.weight") == policy_group("layers.5.linear_attn.in_proj.weight")
      and policy_group("model.layers.3.mlp.down_proj.weight") == "layers.3.mlp.down_proj.weight")
keep = frozenset({"model.layers.3.self_attn.qkv.weight"})
check("a kept fused tensor keeps ALL its components int8 (q, k, v)",
      {scheme_for(f"model.layers.3.self_attn.{c}_proj.weight", "int4", keep) for c in "qkv"} == {"int8"})

# 5. .qt round trip through a file
class Src:
    def __init__(self): self.t = {"a.weight": torch.randn(64, 128).to(torch.bfloat16), "n.weight": torch.randn(128).to(torch.bfloat16)}
    def tensor_names(self): return list(self.t)
    def get(self, n): return self.t[n]
src = Src()
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "x.qt"); save_qt(path, src, "int4", log=lambda *_: None)
    f = QtFile(path)
    qa, direct = f.get("a.weight"), quantize(src.t["a.weight"], "int4")
    check(".qt round trip: quantized weight bit-identical (data, scales, mins), norm kept as bf16",
          isinstance(qa, QuantTensor) and torch.equal(qa.data, direct.data) and torch.equal(qa.scales, direct.scales)
          and torch.equal(qa.mins, direct.mins) and torch.equal(f.get("n.weight"), src.t["n.weight"]))
    del f, qa

# 6. model accuracy: bf16 vs int8 vs int4 on the Metal backend
D = "models/qwen2.5-0.5b"
cfg = ModelConfig.from_json(f"{D}/config.json"); weights = SafetensorsFile(f"{D}/model.safetensors"); tok = Tokenizer(f"{D}/tokenizer.json")
TEXT = ("The Industrial Revolution began in Britain in the late eighteenth century. New machines powered by steam "
        "transformed the production of textiles, and factories drew workers from the countryside into rapidly "
        "growing towns. Railways followed, shrinking the time it took to move goods and people across the country, "
        "and within a few decades the same changes had spread to Europe and North America.")
text_ids = torch.tensor(tok.encode(TEXT))
goldens = [torch.load(f"tests/golden/{i}.pt") for i in range(5)]
summary = {}
POLICY = "configs/quant/qwen2.5-0.5b.json"
for scheme in (None, "int8", "int4", "int4+policy"):
    be = MetalBackend("int4", POLICY) if scheme == "int4+policy" else MetalBackend(scheme)
    m = Qwen2Model(cfg, weights, be)
    lg = m.forward(text_ids).cpu()[:-1, : tok.vocab_size()]
    nll = -torch.log(softmax(lg)[torch.arange(len(text_ids) - 1), text_ids[1:]]).mean().item()
    kls, agree = [], []
    for g in goldens:
        p = softmax(g["logits"][:, : tok.vocab_size()]); q_ = softmax(m.forward(g["ids"]).cpu()[:, : tok.vocab_size()])
        kls.append((p * (torch.log(p + 1e-12) - torch.log(q_ + 1e-12))).sum(-1).mean().item())
        agree.append((p.argmax(-1) == q_.argmax(-1)).float().mean().item())
    greedy = tok.decode(generate_greedy(m, tok, "The capital of France is", 10, {151643, 151645}))
    nbytes = sum(w.nbytes if isinstance(w, QuantTensor) else w.numel() * w.element_size() for w in m._cache.values())
    summary[scheme or "bf16"] = dict(ppl=math.exp(nll), kl=sum(kls) / len(kls), agree=sum(agree) / len(agree), greedy=greedy, gb=nbytes / 1e9)
    print(f"      {scheme or 'bf16':5s}: weights {nbytes / 1e9:.2f} GB | perplexity {math.exp(nll):6.3f} | KL vs fp32 {summary[scheme or 'bf16']['kl']:.4f} | "
          f"top-1 agreement {summary[scheme or 'bf16']['agree']:.0%} | greedy {greedy!r}")
    del m; gc.collect(); torch.mps.empty_cache()
# 7. a .qt file written by scripts/quantize.py loads to exactly the same model as quantizing on the fly
import subprocess
with tempfile.TemporaryDirectory() as d:
    qt = os.path.join(d, "m.qt")
    subprocess.run([sys.executable, "scripts/quantize.py", f"{D}/model.safetensors", qt, "--scheme", "int4", "--policy", POLICY],
                   check=True, capture_output=True)
    a = Qwen2Model(cfg, QtFile(qt), MetalBackend("int4"))
    b = Qwen2Model(cfg, weights, MetalBackend("int4", POLICY))
    ids = goldens[3]["ids"]
    check(".qt-loaded int4 model == on-the-fly int4 model (identical logits)", torch.equal(a.forward(ids).cpu(), b.forward(ids).cpu()))
    del a, b; gc.collect(); torch.mps.empty_cache()

s = summary
check(f"int8 perplexity within 1% of bf16 ({s['int8']['ppl']:.3f} vs {s['bf16']['ppl']:.3f})", s["int8"]["ppl"] < s["bf16"]["ppl"] * 1.01)
check(f"int4 (asymmetric, int8 embedding) perplexity within 5% of bf16 ({s['int4']['ppl']:.3f} vs {s['bf16']['ppl']:.3f})", s["int4"]["ppl"] < s["bf16"]["ppl"] * 1.05)
check("int8 greedy output identical to bf16", s["int8"]["greedy"] == s["bf16"]["greedy"])
check("int4 still answers Paris", s["int4"]["greedy"].startswith(" Paris"))
p4 = s["int4+policy"]
check(f"calibrated int4 (sensitive tensors kept int8): KL vs fp32 {p4['kl']:.3f} < 0.15 and top-1 agreement {p4['agree']:.0%} >= 90%",
      p4["kl"] < 0.15 and p4["agree"] >= 0.9)
check(f"calibrated int4 stays well below int8 memory ({p4['gb']:.3f} vs int8 {s['int8']['gb']:.3f} GB)", p4["gb"] < s["int8"]["gb"] * 0.85)
check(f"sizes shrink: bf16 {s['bf16']['gb']:.2f} > int8 {s['int8']['gb']:.2f} > int4 {s['int4']['gb']:.2f} GB",
      s["bf16"]["gb"] > s["int8"]["gb"] > s["int4"]["gb"])

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
